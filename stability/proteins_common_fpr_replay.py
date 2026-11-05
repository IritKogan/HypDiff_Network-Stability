from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from hyperbolic_learning.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from models.base_models import LPModel
from models.direction_diffusion import diffusion
from stability.partial_recovery import (
    compute_community_context,
    decode_without_diffusion,
    reconstruct_k,
)
from stability.pilot_config import PilotConfig
from stability.proteins_dataset import ProteinsGraph
from stability.wp7_training import make_adjacency, model_args


FALSE_POSITIVE_RATE_CAP = 0.05
RECONSTRUCTIONS_PER_MASK = 8
T_STAR = 7


def load_saved_graph_models(
    graph: ProteinsGraph,
    checkpoint_dir: Path,
    config: PilotConfig,
    model_seed: int,
):
    args = model_args(graph, config)
    encoder = LPModel(args).cpu()
    denoiser = diffusion(args, dim=args.dim, curvature=config.curvature).cpu()
    encoder_path = checkpoint_dir / graph.graph_id / "autoencoder.pt"
    denoiser_path = checkpoint_dir / graph.graph_id / "denoiser.pt"
    if not encoder_path.is_file() or not denoiser_path.is_file():
        raise FileNotFoundError(f"saved checkpoints missing for {graph.graph_id}")
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True)
    )
    denoiser.load_state_dict(
        torch.load(denoiser_path, map_location="cpu", weights_only=True)
    )
    encoder.eval()
    denoiser.eval()

    original_adjacency = make_adjacency(
        len(graph.node_ids),
        _indexed_edges(graph.edges, graph.node_ids),
    )
    np.random.seed(model_seed)
    clusterer = HyperbolicKMeans(n_clusters=2, curvature=config.curvature)
    context = compute_community_context(
        encoder,
        graph.features,
        original_adjacency,
        clusterer,
        graph.node_ids,
    )
    return encoder, denoiser, context


def _indexed_edges(edges, node_ids):
    indices = {node: index for index, node in enumerate(node_ids)}
    return tuple((indices[source], indices[target]) for source, target in edges)


def regenerate_graph_scores(
    graph: ProteinsGraph,
    planned_cases: list[dict],
    checkpoint_dir: Path,
    model_seed: int,
):
    config = PilotConfig(curvature=1.0, tau=0.0)
    encoder, denoiser, context = load_saved_graph_models(
        graph, checkpoint_dir, config, model_seed
    )
    candidates = tuple(
        (graph.node_ids[source], graph.node_ids[target])
        for source, target in itertools.combinations(range(len(graph.node_ids)), 2)
    )
    original_adjacency_edges = {frozenset(edge) for edge in graph.edges}
    nonedge_indices = np.asarray(
        [
            index
            for index, edge in enumerate(candidates)
            if frozenset(edge) not in original_adjacency_edges
        ],
        dtype=np.int64,
    )
    candidate_index = {
        frozenset(edge): index for index, edge in enumerate(candidates)
    }
    results = []
    for planned in planned_cases:
        damaged_edges = tuple(tuple(edge) for edge in planned["damaged_edges"])
        removed_edges = tuple(tuple(edge) for edge in planned["removed_edges"])
        adjacency = make_adjacency(
            len(graph.node_ids),
            _indexed_edges(damaged_edges, graph.node_ids),
        )
        with torch.no_grad():
            damaged_embeddings = encoder.encode(graph.features, adjacency)[2]
            _, baseline_probabilities = decode_without_diffusion(
                encoder,
                damaged_embeddings,
                graph.node_ids,
                candidates,
                tau=0.0,
            )
        batch = reconstruct_k(
            encoder=encoder,
            denoiser=denoiser,
            damaged_embeddings=damaged_embeddings,
            damaged_adjacency=adjacency,
            context=context,
            node_ids=graph.node_ids,
            candidate_edges=candidates,
            t_star=T_STAR,
            num_reconstructions=RECONSTRUCTIONS_PER_MASK,
            seed=int(planned["reconstruction_seed_base"]),
            config=config,
        )
        if not torch.isfinite(baseline_probabilities).all() or any(
            not torch.isfinite(row).all() for row in batch.probabilities
        ):
            raise RuntimeError(f"non-finite probability for {graph.graph_id}")
        removed_indices = np.asarray(
            [candidate_index[frozenset(edge)] for edge in removed_edges],
            dtype=np.int64,
        )
        results.append(
            {
                **{
                    key: planned[key]
                    for key in (
                        "attack_type",
                        "damage_fraction_requested",
                        "damage_fraction_realized",
                        "edges_removed",
                        "mask_index",
                        "mask_seed",
                        "reconstruction_seed_base",
                        "removed_edges",
                    )
                },
                "candidate_edges": candidates,
                "removed_indices": removed_indices,
                "nonedge_indices": nonedge_indices,
                "baseline_probabilities": baseline_probabilities.cpu().numpy(),
                "partial_probabilities": np.stack(
                    [row.cpu().numpy() for row in batch.probabilities]
                ),
            }
        )
    return {
        "graph_id": graph.graph_id,
        "node_ids": graph.node_ids,
        "original_edges": graph.edges,
        "cases": results,
    }


def _event_curves(graph_results, method: str):
    score_values = []
    score_weights = []
    fpr_values = []
    fpr_weights = []
    fp_values = []
    fp_weights = []
    graph_count = len(graph_results)
    if not graph_count:
        raise ValueError("threshold calibration requires graphs")
    levels = sorted(
        {
            case["damage_fraction_requested"]
            for graph in graph_results
            for case in graph["cases"]
        }
    )
    attacks = sorted(
        {case["attack_type"] for graph in graph_results for case in graph["cases"]}
    )
    if len(attacks) != 2:
        raise ValueError("calibration requires the two approved attack types")

    for graph in graph_results:
        for attack in attacks:
            for fraction in levels:
                cell = [
                    case
                    for case in graph["cases"]
                    if case["attack_type"] == attack
                    and case["damage_fraction_requested"] == fraction
                ]
                if not cell:
                    raise ValueError(
                        f"missing calibration cell: {graph['graph_id']} {attack} {fraction}"
                    )
                cell_mask_weight = 1.0 / (
                    graph_count * len(levels) * len(attacks) * len(cell)
                )
                for case in cell:
                    probabilities = (
                        case["partial_probabilities"]
                        if method == "partial_recovery"
                        else case["baseline_probabilities"][None, :]
                    )
                    recon_count = len(probabilities)
                    positive_indices = case["removed_indices"]
                    negative_indices = case["nonedge_indices"]
                    if not len(positive_indices) or not len(negative_indices):
                        raise ValueError("each mask requires positives and true nonedges")

                    positive_probabilities = probabilities[:, positive_indices].reshape(-1)
                    positive_weights = np.full(
                        len(positive_probabilities),
                        cell_mask_weight
                        / recon_count
                        / len(positive_indices),
                        dtype=np.float64,
                    )
                    score_values.append(positive_probabilities)
                    score_weights.append(positive_weights)

                    negative_probabilities = probabilities[:, negative_indices].reshape(-1)
                    negative_fpr_weights = np.full(
                        len(negative_probabilities),
                        cell_mask_weight / recon_count / len(negative_indices),
                        dtype=np.float64,
                    )
                    fpr_values.append(negative_probabilities)
                    fpr_weights.append(negative_fpr_weights)
                    negative_fp_weights = np.full(
                        len(negative_probabilities),
                        cell_mask_weight / recon_count,
                        dtype=np.float64,
                    )
                    fp_values.append(negative_probabilities)
                    fp_weights.append(negative_fp_weights)

    values = np.concatenate(score_values + fpr_values + fp_values)
    unique = np.unique(np.concatenate((values, np.asarray([1.0]))))

    def cumulative(parts, weights):
        indexes = np.searchsorted(unique, np.concatenate(parts))
        increments = np.bincount(
            indexes,
            weights=np.concatenate(weights),
            minlength=len(unique),
        )
        return np.cumsum(increments[::-1])[::-1]

    score_curve = cumulative(score_values, score_weights)
    fpr_curve = cumulative(fpr_values, fpr_weights)
    fp_curve = cumulative(fp_values, fp_weights)
    return unique, score_curve, fpr_curve, fp_curve


def select_method_threshold(graph_results, method: str):
    if method not in {"partial_recovery", "no_diffusion_baseline"}:
        raise ValueError(f"unsupported method {method!r}")
    thresholds, scores, fprs, fp_counts = _event_curves(graph_results, method)
    feasible = np.flatnonzero(fprs <= FALSE_POSITIVE_RATE_CAP)
    if not len(feasible):
        return {
            "status": "no_threshold_meets_fpr_cap",
            "tau": None,
            "false_positive_rate_cap": FALSE_POSITIVE_RATE_CAP,
            "nonzero_recovery_feasible": False,
        }
    best_score = float(np.max(scores[feasible]))
    best_indices = feasible[scores[feasible] == best_score]
    best_fpr = float(np.min(fprs[best_indices]))
    best_indices = best_indices[fprs[best_indices] == best_fpr]
    selected_index = int(best_indices[-1])
    selected_tau = float(thresholds[selected_index])
    return {
        "status": (
            "threshold_selected"
            if best_score > 0
            else "no_nonzero_recovery_under_fpr_cap"
        ),
        "tau": selected_tau,
        "false_positive_rate_cap": FALSE_POSITIVE_RATE_CAP,
        "calibration_book_stability_score": best_score,
        "calibration_realized_fpr": float(fprs[selected_index]),
        "calibration_absolute_false_positive_edges": float(fp_counts[selected_index]),
        "nonzero_recovery_feasible": best_score > 0,
        "candidate_threshold_count": len(thresholds),
        "tie_policy": (
            "all candidates with probability equal to tau are included together; "
            "among equal book scores, prefer lower FPR then higher tau"
        ),
    }


def summarize_at_threshold(graph_results, thresholds: dict[str, float]):
    rows = []
    for graph in graph_results:
        original = {frozenset(edge) for edge in graph["original_edges"]}
        all_pairs = {
            frozenset(edge) for edge in graph["cases"][0]["candidate_edges"]
        }
        true_nonedges = all_pairs - original
        for attack in sorted({c["attack_type"] for c in graph["cases"]}):
            for fraction in sorted(
                {c["damage_fraction_requested"] for c in graph["cases"]}
            ):
                cell_cases = [
                    case
                    for case in graph["cases"]
                    if case["attack_type"] == attack
                    and case["damage_fraction_requested"] == fraction
                ]
                method_results = {}
                for method, tau in thresholds.items():
                    mask_rows = []
                    for case in cell_cases:
                        probabilities = (
                            case["partial_probabilities"]
                            if method == "partial_recovery"
                            else case["baseline_probabilities"][None, :]
                        )
                        removed = {frozenset(edge) for edge in case["removed_edges"]}
                        per_reconstruction = []
                        for row in probabilities:
                            predicted = {
                                frozenset(edge)
                                for edge, probability in zip(
                                    case["candidate_edges"], row
                                )
                                if probability >= tau
                            }
                            tp = len(predicted & removed)
                            fp = len(predicted & true_nonedges)
                            per_reconstruction.append(
                                {
                                    "recovered_removed_edges": tp,
                                    "book_stability_score": tp / len(removed),
                                    "false_positive_edges": fp,
                                    "false_positive_rate": fp / len(true_nonedges),
                                }
                            )
                        mask_rows.append(
                            {
                                "mask_index": case["mask_index"],
                                "mask_seed": case["mask_seed"],
                                "reconstruction_count": len(per_reconstruction),
                                "book_stability_score": float(
                                    np.mean(
                                        [
                                            row["book_stability_score"]
                                            for row in per_reconstruction
                                        ]
                                    )
                                ),
                                "absolute_false_positive_edges": float(
                                    np.mean(
                                        [
                                            row["false_positive_edges"]
                                            for row in per_reconstruction
                                        ]
                                    )
                                ),
                                "realized_fpr": float(
                                    np.mean(
                                        [
                                            row["false_positive_rate"]
                                            for row in per_reconstruction
                                        ]
                                    )
                                ),
                                "per_reconstruction": per_reconstruction,
                            }
                        )
                    method_results[method] = {
                        "book_stability_score": float(
                            np.mean(
                                [row["book_stability_score"] for row in mask_rows]
                            )
                        ),
                        "absolute_false_positive_edges": float(
                            np.mean(
                                [
                                    row["absolute_false_positive_edges"]
                                    for row in mask_rows
                                ]
                            )
                        ),
                        "realized_fpr": float(
                            np.mean([row["realized_fpr"] for row in mask_rows])
                        ),
                        "mask_count": len(mask_rows),
                        "masks": mask_rows,
                    }
                rows.append(
                    {
                        "graph_id": graph["graph_id"],
                        "attack_type": attack,
                        "damage_fraction_requested": fraction,
                        "method_metrics": method_results,
                    }
                )
    return rows


def summarize_attack_balanced(rows):
    result = {}
    methods = ("partial_recovery", "no_diffusion_baseline")
    attacks = sorted({row["attack_type"] for row in rows})
    for method in methods:
        result[method] = {}
        for metric in (
            "book_stability_score",
            "realized_fpr",
            "absolute_false_positive_edges",
        ):
            attack_means = []
            for attack in attacks:
                values = [
                    row["method_metrics"][method][metric]
                    for row in rows
                    if row["attack_type"] == attack
                ]
                attack_means.append(float(np.mean(values)))
            result[method][metric] = float(np.mean(attack_means))
    return result
