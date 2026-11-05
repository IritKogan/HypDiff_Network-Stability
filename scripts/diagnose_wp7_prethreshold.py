from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from stability.edge_perturbation import remove_edges
from stability.pilot_artifacts import load_pilot_artifact, save_pilot_artifact
from stability.pilot_config import PilotConfig
from stability.partial_recovery import reconstruct_k
from stability.synthetic_instances import (
    DIAGNOSTIC_SEEDS,
    generate_synthetic_graph,
)
from stability.wp7_training import (
    REMOVED_EDGE_COUNT,
    RECONSTRUCTIONS,
    T_STAR,
    make_adjacency,
    train_graph_models,
)


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ARTIFACT = (
    ROOT / "artifacts" / "wp7_synthetic_calibration" / "calibration.json"
)
ARTIFACT_DIR = ROOT / "artifacts" / "wp7_prethreshold_diagnostic"
MODEL_SEED_OFFSET = 93000
PERTURBATION_SEED_OFFSET = 94000
RECONSTRUCTION_SEED_OFFSET = 95000


def _ranking_metrics(candidate_edges, removed_edges, probabilities):
    removed_keys = {frozenset(edge) for edge in removed_edges}
    labels = np.asarray(
        [int(frozenset(edge) in removed_keys) for edge in candidate_edges],
        dtype=np.int64,
    )
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != labels.shape:
        raise ValueError("probabilities must align with candidate_edges")
    if len(set(labels.tolist())) != 2:
        raise ValueError("candidate edges must include removed edges and true nonedges")
    if not np.isfinite(scores).all():
        raise ValueError("ranking probabilities must be finite")
    order = sorted(
        range(len(candidate_edges)),
        key=lambda index: (-scores[index], candidate_edges[index]),
    )
    top_three = {frozenset(candidate_edges[index]) for index in order[:3]}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "removed_edges_in_top_three": len(top_three & removed_keys),
        "top_three_candidates": [
            list(candidate_edges[index]) for index in order[:3]
        ],
    }


def _model_probabilities(model, embeddings, node_ids, candidate_edges):
    node_order = {node: index for index, node in enumerate(node_ids)}
    pairs = torch.tensor(
        [(node_order[source], node_order[target]) for source, target in candidate_edges],
        dtype=torch.long,
        device=embeddings.device,
    ).reshape(-1, 2)
    with torch.no_grad():
        probabilities = model.decode(embeddings, pairs)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("decoder produced non-finite probabilities")
    return probabilities.detach().cpu().numpy()


def _graph_diagnostic(graph, frozen_tau):
    config = PilotConfig(curvature=1.0, tau=frozen_tau)
    model_seed = graph.seed + MODEL_SEED_OFFSET
    perturbation_seed = graph.seed + PERTURBATION_SEED_OFFSET
    reconstruction_seed = graph.seed + RECONSTRUCTION_SEED_OFFSET
    encoder, denoiser, context, original_adjacency, training = train_graph_models(
        graph=graph,
        config=config,
        model_seed=model_seed,
        checkpoint_dir=ARTIFACT_DIR / "models",
    )

    perturbed = remove_edges(
        graph.node_ids,
        graph.features,
        graph.edges,
        REMOVED_EDGE_COUNT,
        perturbation_seed,
    )
    node_order = {node: index for index, node in enumerate(graph.node_ids)}
    indexed_damaged = tuple(
        (node_order[source], node_order[target]) for source, target in perturbed.edges
    )
    damaged_adjacency = make_adjacency(len(graph.node_ids), indexed_damaged)

    with torch.no_grad():
        intact_embeddings = encoder.encode(graph.features, original_adjacency)[2]
        damaged_embeddings = encoder.encode(graph.features, damaged_adjacency)[2]
    if not torch.isfinite(intact_embeddings).all():
        raise RuntimeError(f"{graph.graph_id}: intact embeddings are non-finite")
    if not torch.isfinite(damaged_embeddings).all():
        raise RuntimeError(f"{graph.graph_id}: damaged embeddings are non-finite")

    original_set = {frozenset(edge) for edge in graph.edges}
    candidate_edges = tuple(
        (graph.node_ids[source], graph.node_ids[target])
        for source, target in itertools.combinations(range(len(graph.node_ids)), 2)
        if frozenset((graph.node_ids[source], graph.node_ids[target]))
        not in original_set
        or frozenset((graph.node_ids[source], graph.node_ids[target]))
        in {frozenset(edge) for edge in perturbed.removed_edges}
    )
    removed_set = {frozenset(edge) for edge in perturbed.removed_edges}
    candidate_set = {frozenset(edge) for edge in candidate_edges}
    if not removed_set.issubset(candidate_set):
        raise RuntimeError("candidate additions are missing removed edges")
    all_pairs = {
        frozenset(edge) for edge in itertools.combinations(graph.node_ids, 2)
    }
    expected_additions = len(all_pairs - original_set) + len(
        perturbed.removed_edges
    )
    if len(candidate_edges) != expected_additions:
        raise RuntimeError("candidate additions do not match removed edges plus nonedges")

    intact_probabilities = _model_probabilities(
        encoder, intact_embeddings, graph.node_ids, candidate_edges
    )
    damaged_probabilities = _model_probabilities(
        encoder, damaged_embeddings, graph.node_ids, candidate_edges
    )
    batch = reconstruct_k(
        encoder=encoder,
        denoiser=denoiser,
        damaged_embeddings=damaged_embeddings,
        damaged_adjacency=damaged_adjacency,
        context=context,
        node_ids=graph.node_ids,
        candidate_edges=candidate_edges,
        t_star=T_STAR,
        num_reconstructions=RECONSTRUCTIONS,
        seed=reconstruction_seed,
        config=config,
    )
    partial_probabilities = np.stack(
        [probabilities.detach().cpu().numpy() for probabilities in batch.probabilities]
    )
    if partial_probabilities.shape != (RECONSTRUCTIONS, len(candidate_edges)):
        raise RuntimeError("partial-recovery probabilities have unexpected dimensions")

    intact_metrics = _ranking_metrics(
        candidate_edges, perturbed.removed_edges, intact_probabilities
    )
    damaged_metrics = _ranking_metrics(
        candidate_edges, perturbed.removed_edges, damaged_probabilities
    )
    sample_metrics = [
        _ranking_metrics(candidate_edges, perturbed.removed_edges, probabilities)
        for probabilities in partial_probabilities
    ]
    mean_partial_probabilities = partial_probabilities.mean(axis=0)
    partial_metrics = _ranking_metrics(
        candidate_edges, perturbed.removed_edges, mean_partial_probabilities
    )
    return {
        "graph_id": graph.graph_id,
        "graph_seed": graph.seed,
        "model_seed": model_seed,
        "perturbation_seed": perturbation_seed,
        "reconstruction_seed_base": reconstruction_seed,
        "node_ids": list(graph.node_ids),
        "original_edges": [list(edge) for edge in graph.edges],
        "damaged_edges": [list(edge) for edge in perturbed.edges],
        "removed_edges": [list(edge) for edge in perturbed.removed_edges],
        "candidate_edges": [list(edge) for edge in candidate_edges],
        "candidate_edge_count": len(candidate_edges),
        "true_nonedges_count": len(candidate_edges) - len(perturbed.removed_edges),
        "curvature": config.curvature,
        "frozen_tau_recorded_but_not_applied": frozen_tau,
        "training": training,
        "paths": {
            "intact_graph_encode_decode": {
                **intact_metrics,
                "probabilities": intact_probabilities.tolist(),
            },
            "damaged_graph_encode_decode_no_diffusion": {
                **damaged_metrics,
                "probabilities": damaged_probabilities.tolist(),
            },
            "partial_recovery": {
                **partial_metrics,
                "aggregation": (
                    "mean decoder probability for each candidate pair over 8 "
                    "independently seeded reconstructions; graph ROC-AUC, average "
                    "precision, and top-three ranking use this mean vector"
                ),
                "sample_roc_auc_mean": float(
                    np.mean([metrics["roc_auc"] for metrics in sample_metrics])
                ),
                "sample_roc_auc_min": float(
                    np.min([metrics["roc_auc"] for metrics in sample_metrics])
                ),
                "sample_roc_auc_max": float(
                    np.max([metrics["roc_auc"] for metrics in sample_metrics])
                ),
                "sample_average_precision_mean": float(
                    np.mean(
                        [metrics["average_precision"] for metrics in sample_metrics]
                    )
                ),
                "sample_average_precision_min": float(
                    np.min(
                        [metrics["average_precision"] for metrics in sample_metrics]
                    )
                ),
                "sample_average_precision_max": float(
                    np.max(
                        [metrics["average_precision"] for metrics in sample_metrics]
                    )
                ),
                "sample_removed_edges_in_top_three": [
                    metrics["removed_edges_in_top_three"]
                    for metrics in sample_metrics
                ],
                "sample_probabilities": partial_probabilities.tolist(),
                "mean_probabilities": mean_partial_probabilities.tolist(),
            },
        },
    }


def main():
    calibration = load_pilot_artifact(CALIBRATION_ARTIFACT)
    if calibration.get("status") != "calibration_complete":
        raise ValueError("diagnostic requires the existing frozen WP7 calibration")
    frozen_tau = float(calibration["frozen_threshold_selection"]["tau"])
    graphs = []
    for index, seed in enumerate(DIAGNOSTIC_SEEDS):
        graph = generate_synthetic_graph(seed, "diagnostic")
        print(f"Training diagnostic graph {index + 1}/{len(DIAGNOSTIC_SEEDS)}: {graph.graph_id}")
        graphs.append(_graph_diagnostic(graph, frozen_tau))

    payload = {
        "work_package": "WP7 follow-up pre-threshold diagnostic",
        "status": "descriptive_only",
        "frozen_wp7_threshold": frozen_tau,
        "threshold_used_for_ranking": False,
        "wp7_artifacts_modified": False,
        "method": {
            "fresh_diagnostic_seeds": list(DIAGNOSTIC_SEEDS),
            "graph_count": len(graphs),
            "node_count_per_graph": 12,
            "autoencoder_epochs_per_graph": 30,
            "denoiser_epochs_per_graph": 50,
            "curvature": 1.0,
            "removed_edges_per_graph": REMOVED_EDGE_COUNT,
            "partial_reconstructions_per_graph": RECONSTRUCTIONS,
            "t_star": T_STAR,
            "ranking_candidates": (
                "all absent pairs after perturbation: the three removed edges "
                "and true nonedges; identical candidate order across all three paths"
            ),
            "top_three_tie_break": (
                "descending probability, then lexicographic node-ID pair"
            ),
            "loss": (
                "WP7 stable logit-form binary cross-entropy, equivalent to "
                "probability-form BCE"
            ),
        },
        "graphs": graphs,
    }
    destination = ARTIFACT_DIR / "prethreshold-diagnostic.json"
    save_pilot_artifact(destination, payload)
    summary = []
    for graph in graphs:
        summary.append(
            {
                "graph_id": graph["graph_id"],
                **{
                    path: {
                        key: graph["paths"][path][key]
                        for key in (
                            "roc_auc",
                            "average_precision",
                            "removed_edges_in_top_three",
                        )
                    }
                    for path in graph["paths"]
                },
                "partial_sample_auc_range": [
                    graph["paths"]["partial_recovery"]["sample_roc_auc_min"],
                    graph["paths"]["partial_recovery"]["sample_roc_auc_max"],
                ],
                "partial_sample_ap_range": [
                    graph["paths"]["partial_recovery"][
                        "sample_average_precision_min"
                    ],
                    graph["paths"]["partial_recovery"][
                        "sample_average_precision_max"
                    ],
                ],
            }
        )
    print(json.dumps({"artifact": str(destination), "graphs": summary}, indent=2))


if __name__ == "__main__":
    main()
