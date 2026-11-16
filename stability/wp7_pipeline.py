from __future__ import annotations

import itertools

import numpy as np
import torch

from stability.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.edge_perturbation import remove_edges
from stability.partial_recovery import (
    decode_without_diffusion,
    reconstruct_k,
)
from stability.pilot_config import PilotConfig
from stability.scoring import score_recovery
from stability.synthetic_instances import NODE_COUNT
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


def _mean_recovery_metrics(original_edges, removed_edges, edge_sets, node_ids):
    score = score_recovery(removed_edges, edge_sets)
    original = {frozenset(edge) for edge in original_edges}
    removed = {frozenset(edge) for edge in removed_edges}
    surviving = original - removed
    true_nonedges = {
        frozenset((source, target))
        for index, source in enumerate(node_ids)
        for target in node_ids[index + 1 :]
    } - original
    false_positives = []
    predicted_counts = []
    retentions = []
    recovered_counts = []
    for edges in edge_sets:
        predicted = {frozenset(edge) for edge in edges}
        false_positives.append(len(predicted & true_nonedges))
        predicted_counts.append(len(predicted))
        retentions.append(
            len(predicted & surviving) / len(surviving) if surviving else 1.0
        )
        recovered_counts.append(len(predicted & removed))
    return {
        "book_stability_score": score.scenario_score,
        "removed_edge_recovery_frequencies": [
            {"edge": list(edge), "frequency": frequency}
            for edge, frequency in score.edge_recovery_frequencies.items()
        ],
        "mean_removed_edges_recovered": float(np.mean(recovered_counts)),
        "mean_false_positive_edges": float(np.mean(false_positives)),
        "mean_predicted_edge_count": float(np.mean(predicted_counts)),
        "mean_surviving_edge_retention": float(np.mean(retentions)),
        "distinct_thresholded_reconstructions": len(
            {frozenset(frozenset(edge) for edge in edges) for edges in edge_sets}
        ),
        "per_reconstruction_false_positive_edges": false_positives,
        "per_reconstruction_predicted_edge_count": predicted_counts,
        "per_reconstruction_surviving_edge_retention": retentions,
        "true_nonedge_count": len(true_nonedges),
    }


def run_graph(
    graph,
    config: PilotConfig,
    model_seed: int,
    perturbation_seed: int,
    reconstruction_seed: int,
    checkpoint_dir,
):
    if len(graph.node_ids) != NODE_COUNT:
        raise ValueError(f"{graph.graph_id}: expected exactly {NODE_COUNT} nodes")
    (
        encoder,
        denoiser,
        context,
        _original_adjacency,
        training,
    ) = train_graph_models(graph, config, model_seed, checkpoint_dir)
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
        raise RuntimeError(f"{graph.graph_id}: perturbation changed node IDs or features")
    candidate_edges = tuple(
        (graph.node_ids[source], graph.node_ids[target])
        for source, target in itertools.combinations(range(len(graph.node_ids)), 2)
    )
    damaged_adjacency = make_adjacency(
        len(graph.node_ids), _indexed_edges(perturbed.edges, graph.node_ids)
    )
    with torch.no_grad():
        damaged_embeddings = encoder.encode(graph.features, damaged_adjacency)[2]
    if not torch.isfinite(damaged_embeddings).all():
        raise RuntimeError(f"{graph.graph_id}: damaged embeddings are non-finite")

    sampler_config = PilotConfig(curvature=config.curvature, tau=config.tau)
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
        config=sampler_config,
    )
    reconstructions = [
        {
            "seed": reconstruction_seed + index,
            "probabilities": probabilities.detach().cpu().tolist(),
            "edges": [list(edge) for edge in edge_set],
        }
        for index, (probabilities, edge_set) in enumerate(
            zip(batch.probabilities, batch.edge_sets)
        )
    ]

    baseline_edges, baseline_probabilities = decode_without_diffusion(
        encoder,
        damaged_embeddings,
        graph.node_ids,
        candidate_edges,
        config.tau,
    )
    baseline = _mean_recovery_metrics(
        graph.edges, perturbed.removed_edges, (baseline_edges,), graph.node_ids
    )
    baseline["probabilities"] = baseline_probabilities.detach().cpu().tolist()
    baseline["edges"] = [list(edge) for edge in baseline_edges]
    result = {
        "graph_id": graph.graph_id,
        "split": graph.graph_id.split("-", 1)[0],
        "graph_seed": graph.seed,
        "model_seed": model_seed,
        "perturbation_seed": perturbation_seed,
        "reconstruction_seed_base": reconstruction_seed,
        "node_ids": list(graph.node_ids),
        "features": graph.features.tolist(),
        "original_edges": [list(edge) for edge in graph.edges],
        "damaged_edges": [list(edge) for edge in perturbed.edges],
        "removed_edges": [list(edge) for edge in perturbed.removed_edges],
        "perturbed_graph_connected": perturbed.is_connected,
        "candidate_edges": [list(edge) for edge in candidate_edges],
        "community_context": {
            "labels": context.labels.cpu().tolist(),
            "centers": context.centers.cpu().tolist(),
            "curvature": context.curvature,
            "computed_before_perturbation": True,
            "computed_once": True,
        },
        "training": training,
        "partial_recovery": {
            **_mean_recovery_metrics(
                graph.edges,
                perturbed.removed_edges,
                batch.edge_sets,
                graph.node_ids,
            ),
            "reconstructions": reconstructions,
            "t_star": T_STAR,
            "reconstruction_count": RECONSTRUCTIONS,
        },
        "no_diffusion_baseline": baseline,
    }
    return result
