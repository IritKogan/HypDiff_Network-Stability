from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stability.edge_perturbation import remove_edges
from stability.partial_recovery import (
    decode_without_diffusion,
    reconstruct_k,
)
from stability.pilot_config import PilotConfig
from stability.scoring import score_recovery
from stability.wp7_training import (
    REMOVED_EDGE_COUNT,
    RECONSTRUCTIONS,
    T_STAR,
    make_adjacency,
    train_graph_models,
)


def _indexed_edges(edges, node_ids):
    indices = {node: index for index, node in enumerate(node_ids)}
    return tuple((indices[source], indices[target]) for source, target in edges)


def _score_metrics(graph, removed_edges, edge_sets):
    score = score_recovery(removed_edges, edge_sets)
    original = {frozenset(edge) for edge in graph.edges}
    removed = {frozenset(edge) for edge in removed_edges}
    all_pairs = {
        frozenset((source, target))
        for index, source in enumerate(graph.node_ids)
        for target in graph.node_ids[index + 1 :]
    }
    true_nonedges = all_pairs - original
    additions = removed | true_nonedges
    per_sample = []
    for edges in edge_sets:
        predicted = {frozenset(edge) for edge in edges}
        true_positive = len(predicted & removed)
        false_positive = len(predicted & true_nonedges)
        predicted_additions = true_positive + false_positive
        per_sample.append(
            {
                "removed_edges_recovered": true_positive,
                "false_positive_edges": false_positive,
                "predicted_addition_count": predicted_additions,
                "added_edge_precision": (
                    true_positive / predicted_additions
                    if predicted_additions
                    else None
                ),
                "false_positive_rate": (
                    false_positive / len(true_nonedges)
                    if true_nonedges
                    else 0.0
                ),
                "predicted_edge_count": len(predicted),
                "surviving_edge_retention": (
                    len(predicted & (original - removed)) / len(original - removed)
                    if original - removed
                    else 1.0
                ),
            }
        )
    defined_precisions = [
        row["added_edge_precision"]
        for row in per_sample
        if row["added_edge_precision"] is not None
    ]
    return {
        "book_stability_score": score.scenario_score,
        "edge_recovery_frequencies": [
            {"edge": list(edge), "frequency": frequency}
            for edge, frequency in score.edge_recovery_frequencies.items()
        ],
        "mean_added_edge_precision": (
            float(np.mean(defined_precisions)) if defined_precisions else None
        ),
        "mean_false_positive_rate": float(
            np.mean([row["false_positive_rate"] for row in per_sample])
        ),
        "mean_false_positive_edges": float(
            np.mean([row["false_positive_edges"] for row in per_sample])
        ),
        "mean_predicted_edge_count": float(
            np.mean([row["predicted_edge_count"] for row in per_sample])
        ),
        "mean_surviving_edge_retention": float(
            np.mean([row["surviving_edge_retention"] for row in per_sample])
        ),
        "distinct_thresholded_reconstructions": len(
            {frozenset(frozenset(edge) for edge in edges) for edges in edge_sets}
        ),
        "per_sample": per_sample,
        "true_nonedge_count": len(true_nonedges),
        "candidate_addition_count": len(additions),
    }


def run_proteins_graph(
    graph,
    config: PilotConfig,
    model_seed: int,
    perturbation_seed: int,
    reconstruction_seed: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    node_count = len(graph.node_ids)
    if node_count < 12 or node_count > 50:
        raise ValueError(f"{graph.graph_id}: graph is outside the approved node bounds")
    if graph.features.shape[0] != node_count or not torch.isfinite(graph.features).all():
        raise ValueError(f"{graph.graph_id}: features are invalid")

    encoder, denoiser, context, _original_adjacency, training = train_graph_models(
        graph, config, model_seed, checkpoint_dir
    )
    perturbed = remove_edges(
        graph.node_ids,
        graph.features,
        graph.edges,
        REMOVED_EDGE_COUNT,
        perturbation_seed,
    )
    if perturbed.node_ids != graph.node_ids or not torch.equal(
        perturbed.features, graph.features
    ):
        raise RuntimeError(f"{graph.graph_id}: perturbation changed node IDs/features")
    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    damaged_adjacency = make_adjacency(
        node_count, _indexed_edges(perturbed.edges, graph.node_ids)
    )
    candidate_edges = tuple(
        (graph.node_ids[source], graph.node_ids[target])
        for source, target in itertools.combinations(range(node_count), 2)
    )
    with torch.no_grad():
        damaged_embeddings = encoder.encode(graph.features, damaged_adjacency)[2]
    if not torch.isfinite(damaged_embeddings).all():
        raise RuntimeError(f"{graph.graph_id}: damaged embeddings are non-finite")

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
    baseline_edges, baseline_probabilities = decode_without_diffusion(
        encoder,
        damaged_embeddings,
        graph.node_ids,
        candidate_edges,
        config.tau,
    )
    partial_records = [
        {
            "seed": reconstruction_seed + index,
            "probabilities": probabilities.detach().cpu().tolist(),
            "edges": [list(edge) for edge in edges],
        }
        for index, (probabilities, edges) in enumerate(
            zip(batch.probabilities, batch.edge_sets)
        )
    ]
    partial_metrics = _score_metrics(graph, perturbed.removed_edges, batch.edge_sets)
    baseline_metrics = _score_metrics(
        graph, perturbed.removed_edges, (baseline_edges,)
    )
    return {
        "graph_id": graph.graph_id,
        "dataset_index": graph.dataset_index,
        "content_sha256": graph.content_sha256,
        "node_ids": list(graph.node_ids),
        "features": graph.features.tolist(),
        "original_edges": [list(edge) for edge in graph.edges],
        "damaged_edges": [list(edge) for edge in perturbed.edges],
        "removed_edges": [list(edge) for edge in perturbed.removed_edges],
        "perturbed_graph_connected": perturbed.is_connected,
        "perturbation_seed": perturbation_seed,
        "reconstruction_seed_base": reconstruction_seed,
        "candidate_edges": [list(edge) for edge in candidate_edges],
        "community_context": {
            "labels": context.labels.cpu().tolist(),
            "centers": context.centers.cpu().tolist(),
            "center_norms": torch.linalg.vector_norm(
                context.centers, dim=1
            ).cpu().tolist(),
            "curvature": context.curvature,
            "computed_once_before_perturbation": True,
        },
        "training": training,
        "partial_recovery": {
            **partial_metrics,
            "t_star": T_STAR,
            "reconstruction_count": RECONSTRUCTIONS,
            "reconstructions": partial_records,
        },
        "no_diffusion_baseline": {
            **baseline_metrics,
            "edges": [list(edge) for edge in baseline_edges],
            "probabilities": baseline_probabilities.detach().cpu().tolist(),
        },
    }


def macro_metrics(graph_results, method):
    metric_names = (
        "book_stability_score",
        "mean_added_edge_precision",
        "mean_false_positive_rate",
        "mean_false_positive_edges",
        "mean_predicted_edge_count",
        "mean_surviving_edge_retention",
        "distinct_thresholded_reconstructions",
    )
    out = {}
    for metric in metric_names:
        values = [
            graph[method][metric]
            for graph in graph_results
            if graph[method][metric] is not None
        ]
        out[metric] = float(np.mean(values)) if values else None
    return out
