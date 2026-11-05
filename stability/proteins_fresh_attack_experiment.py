from __future__ import annotations

import itertools
import math
import random

import numpy as np
import torch

from stability.partial_recovery import decode_without_diffusion, reconstruct_k
from stability.pilot_config import PilotConfig
from stability.proteins_dataset import ProteinsGraph
from stability.proteins_targeted_classical import (
    rounded_removal_count,
    targeted_edge_order,
)
from stability.scoring import score_recovery
from stability.wp7_training import make_adjacency, train_graph_models


DAMAGE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
RANDOM_MASKS_PER_LEVEL = 8
RECONSTRUCTIONS_PER_MASK = 8
T_STAR = 7
FALSE_POSITIVE_RATE_CAP = 0.05
SELECTION_SEED = 20261112
USED_GRAPH_IDS = frozenset(
    {
        "PROTEINS-0151",
        "PROTEINS-0215",
        "PROTEINS-0282",
        "PROTEINS-0330",
        "PROTEINS-0395",
        "PROTEINS-0555",
        "PROTEINS-0619",
        "PROTEINS-0772",
        "PROTEINS-0925",
        "PROTEINS-1034",
        "PROTEINS-1086",
    }
)


def select_fresh_splits(graphs, seed: int = SELECTION_SEED):
    eligible = sorted(
        (graph for graph in graphs if graph.graph_id not in USED_GRAPH_IDS),
        key=lambda graph: (graph.content_sha256, graph.dataset_index),
    )
    unique = []
    seen_hashes = set()
    for graph in eligible:
        if graph.content_sha256 not in seen_hashes:
            unique.append(graph)
            seen_hashes.add(graph.content_sha256)
    if len(unique) < 10:
        raise ValueError(f"only {len(unique)} fresh distinct graphs; ten are required")
    permutation = np.random.default_rng(seed).permutation(len(unique))
    selected = [unique[index] for index in permutation[:10]]
    calibration = tuple(selected[:5])
    evaluation = tuple(selected[5:])
    if {g.content_sha256 for g in calibration} & {
        g.content_sha256 for g in evaluation
    }:
        raise RuntimeError("fresh calibration and evaluation graphs overlap")
    return calibration, evaluation, len(eligible), len(unique)


def _canonical_edges(graph: ProteinsGraph):
    node_order = {node: index for index, node in enumerate(graph.node_ids)}
    return tuple(
        sorted(
            (
                edge
                if node_order[edge[0]] < node_order[edge[1]]
                else (edge[1], edge[0])
                for edge in graph.edges
            ),
            key=lambda edge: (node_order[edge[0]], node_order[edge[1]]),
        )
    )


def random_edge_masks(node_ids, edges, count: int, seed: int, mask_count: int = 8):
    canonical = tuple(edges)
    if count < 0 or count > len(canonical):
        raise ValueError("removal count is outside the edge set")
    seen = set()
    results = []
    for mask_index in range(mask_count):
        attempt = 0
        while True:
            mask_seed = seed + mask_index * 100 + attempt
            sampled = tuple(
                sorted(
                    random.Random(mask_seed).sample(canonical, count),
                    key=lambda edge: (
                        node_ids.index(edge[0]),
                        node_ids.index(edge[1]),
                    ),
                )
            )
            key = frozenset(frozenset(edge) for edge in sampled)
            if key not in seen:
                seen.add(key)
                results.append((mask_seed, sampled))
                break
            attempt += 1
            if attempt > 10000:
                raise RuntimeError("could not generate distinct random masks")
    return tuple(results)


def _indexed_edges(edges, node_ids):
    index = {node: position for position, node in enumerate(node_ids)}
    return tuple((index[source], index[target]) for source, target in edges)


def _mask_seed(split_code, graph_index, level_index, mask_index):
    return (
        SELECTION_SEED
        + split_code * 2_000_000
        + graph_index * 10_000
        + level_index * 1_000
        + mask_index * 10
    )


def _reconstruction_seed(split_code, graph_index, level_index, attack_code, mask_index):
    return (
        5_000_000
        + split_code * 2_000_000
        + graph_index * 100_000
        + level_index * 1_000
        + attack_code * 100
        + mask_index * 10
    )


def _make_attack_cases(graph: ProteinsGraph, graph_index: int, split_code: int):
    edges = _canonical_edges(graph)
    targeted_order, _ = targeted_edge_order(graph.node_ids, edges)
    cases = []
    for level_index, fraction in enumerate(DAMAGE_FRACTIONS, start=1):
        removal_count = rounded_removal_count(len(edges), fraction)
        targeted_removed = tuple(targeted_order[:removal_count])
        masks = [
            {
                "attack_type": "targeted_degree_product",
                "attack_code": 0,
                "mask_index": 0,
                "mask_seed": None,
                "removed_edges": targeted_removed,
                "reconstruction_seed_base": _reconstruction_seed(
                    split_code, graph_index, level_index, 0, 0
                ),
            }
        ]
        random_seed = _mask_seed(split_code, graph_index, level_index, 1)
        masks.extend(
            {
                "attack_type": "uniform_random",
                "attack_code": 1,
                "mask_index": mask_index,
                "mask_seed": mask_seed,
                "removed_edges": removed,
                "reconstruction_seed_base": _reconstruction_seed(
                    split_code, graph_index, level_index, 1, mask_index
                ),
            }
            for mask_index, (mask_seed, removed) in enumerate(
                random_edge_masks(
                    graph.node_ids,
                    edges,
                    removal_count,
                    random_seed,
                    RANDOM_MASKS_PER_LEVEL,
                )
            )
        )
        for mask in masks:
            removed_set = {frozenset(edge) for edge in mask["removed_edges"]}
            damaged_edges = tuple(
                edge for edge in edges if frozenset(edge) not in removed_set
            )
            cases.append(
                {
                    "damage_fraction_requested": fraction,
                    "damage_fraction_realized": removal_count / len(edges),
                    "edges_removed": removal_count,
                    "attack_type": mask["attack_type"],
                    "mask_index": mask["mask_index"],
                    "mask_seed": mask["mask_seed"],
                    "removed_edges": [list(edge) for edge in mask["removed_edges"]],
                    "damaged_edges": [list(edge) for edge in damaged_edges],
                    "reconstruction_seed_base": mask["reconstruction_seed_base"],
                }
            )
    return cases


def run_graph_cases(
    graph: ProteinsGraph,
    graph_index: int,
    split_code: int,
    checkpoint_dir,
    planned_cases=None,
):
    if len(graph.node_ids) < 12 or len(graph.node_ids) > 50:
        raise ValueError(f"{graph.graph_id}: graph is outside the approved node bounds")
    model_seed = 3_000_000 + split_code * 100_000 + graph.dataset_index
    config = PilotConfig(curvature=1.0, tau=0.0)
    encoder, denoiser, context, _original_adjacency, training = train_graph_models(
        graph, config, model_seed, checkpoint_dir
    )
    candidates = tuple(
        (graph.node_ids[i], graph.node_ids[j])
        for i, j in itertools.combinations(range(len(graph.node_ids)), 2)
    )
    original = {frozenset(edge) for edge in graph.edges}
    nonedge_indices = np.asarray(
        [
            index
            for index, edge in enumerate(candidates)
            if frozenset(edge) not in original
        ],
        dtype=np.int64,
    )
    original_adjacency = make_adjacency(
        len(graph.node_ids), _indexed_edges(graph.edges, graph.node_ids)
    )
    with torch.no_grad():
        intact_embeddings = encoder.encode(graph.features, original_adjacency)[2]
    if not torch.isfinite(intact_embeddings).all():
        raise RuntimeError(f"{graph.graph_id}: intact embeddings are non-finite")
    cases = planned_cases or _make_attack_cases(graph, graph_index, split_code)
    for case in cases:
        adjacency = make_adjacency(
            len(graph.node_ids),
            _indexed_edges(
                tuple(tuple(edge) for edge in case["damaged_edges"]),
                graph.node_ids,
            ),
        )
        with torch.no_grad():
            damaged_embeddings = encoder.encode(graph.features, adjacency)[2]
            _, baseline_probabilities = decode_without_diffusion(
                encoder,
                damaged_embeddings,
                graph.node_ids,
                candidates,
                0.0,
            )
        if not torch.isfinite(damaged_embeddings).all():
            raise RuntimeError(f"{graph.graph_id}: damaged embeddings are non-finite")
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
            seed=case["reconstruction_seed_base"],
            config=config,
        )
        case["baseline_probabilities"] = baseline_probabilities.cpu().tolist()
        case["partial_probabilities"] = [
            probabilities.cpu().tolist() for probabilities in batch.probabilities
        ]
        case["candidate_edges"] = [list(edge) for edge in candidates]
    return {
        "graph_id": graph.graph_id,
        "dataset_index": graph.dataset_index,
        "content_sha256": graph.content_sha256,
        "node_ids": list(graph.node_ids),
        "features": graph.features.tolist(),
        "original_edges": [list(edge) for edge in graph.edges],
        "training": training,
        "cases": cases,
    }


def _threshold_sweep(graph_results):
    score_parts = []
    partial_book_parts = []
    partial_fpr_parts = []
    baseline_fpr_parts = []
    graph_count = len(graph_results)
    level_count = len(DAMAGE_FRACTIONS)
    for graph in graph_results:
        original = {frozenset(edge) for edge in graph["original_edges"]}
        all_pairs = {
            frozenset(edge) for edge in graph["cases"][0]["candidate_edges"]
        }
        true_nonedges = all_pairs - original
        for fraction in DAMAGE_FRACTIONS:
            for attack_type in ("targeted_degree_product", "uniform_random"):
                cell = [
                    case
                    for case in graph["cases"]
                    if case["damage_fraction_requested"] == fraction
                    and case["attack_type"] == attack_type
                ]
                cell_weight = 0.5 / (graph_count * level_count * len(cell))
                for case in cell:
                    removed = {frozenset(edge) for edge in case["removed_edges"]}
                    case_scores = np.asarray(
                        case["baseline_probabilities"], dtype=np.float64
                    )
                    case_nonedge_indices = np.asarray(
                        [
                            index
                            for index, edge in enumerate(case["candidate_edges"])
                            if frozenset(edge) in true_nonedges
                        ],
                        dtype=np.int64,
                    )
                    case_removed_indices = np.asarray(
                        [
                            index
                            for index, edge in enumerate(case["candidate_edges"])
                            if frozenset(edge) in removed
                        ],
                        dtype=np.int64,
                    )
                    for probabilities in case["partial_probabilities"]:
                        probs = np.asarray(probabilities, dtype=np.float64)
                        score_parts.append(probs)
                        weights = np.zeros(len(probs), dtype=np.float64)
                        weights[case_removed_indices] = cell_weight / (
                            len(case_removed_indices)
                            * RECONSTRUCTIONS_PER_MASK
                        )
                        partial_book_parts.append(weights)
                        fpr_weights = np.zeros(len(probs), dtype=np.float64)
                        fpr_weights[case_nonedge_indices] = cell_weight / (
                            len(true_nonedges) * RECONSTRUCTIONS_PER_MASK
                        )
                        partial_fpr_parts.append(fpr_weights)
                    baseline_fpr_weights = np.zeros(len(case_scores), dtype=np.float64)
                    baseline_fpr_weights[case_nonedge_indices] = (
                        cell_weight / len(true_nonedges)
                    )
                    baseline_fpr_parts.append(baseline_fpr_weights)
    scores = np.concatenate(score_parts)
    partial_book_weights = np.concatenate(partial_book_parts)
    partial_fpr_weights = np.concatenate(partial_fpr_parts)

    baseline_scores = np.concatenate(
        [
            np.asarray(case["baseline_probabilities"], dtype=np.float64)
            for graph in graph_results
            for case in graph["cases"]
        ]
    )
    baseline_weights = np.concatenate(baseline_fpr_parts)
    if len(scores) != len(partial_book_weights) or len(scores) != len(
        partial_fpr_weights
    ):
        raise RuntimeError("partial-recovery calibration sweep arrays do not align")
    if len(baseline_scores) != len(baseline_weights):
        raise RuntimeError("no-diffusion calibration sweep arrays do not align")

    all_scores = np.concatenate((scores, baseline_scores))
    unique_scores, inverse = np.unique(all_scores, return_inverse=True)
    partial_inverse = inverse[: len(scores)]
    baseline_inverse = inverse[len(scores) :]
    partial_book = np.bincount(
        partial_inverse, weights=partial_book_weights, minlength=len(unique_scores)
    )
    partial_fpr = np.bincount(
        partial_inverse, weights=partial_fpr_weights, minlength=len(unique_scores)
    )
    baseline_fpr = np.bincount(
        baseline_inverse, weights=baseline_weights, minlength=len(unique_scores)
    )
    order = np.arange(len(unique_scores) - 1, -1, -1)
    taus = unique_scores[order]
    partial_recovery = np.cumsum(partial_book[order])
    partial_false_positive_rate = np.cumsum(partial_fpr[order])
    baseline_false_positive_rate = np.cumsum(baseline_fpr[order])
    if taus[-1] > 0.0:
        taus = np.append(taus, 0.0)
        partial_recovery = np.append(partial_recovery, partial_recovery[-1])
        partial_false_positive_rate = np.append(
            partial_false_positive_rate, partial_false_positive_rate[-1]
        )
        baseline_false_positive_rate = np.append(
            baseline_false_positive_rate, baseline_false_positive_rate[-1]
        )
    return taus, partial_recovery, partial_false_positive_rate, baseline_false_positive_rate


def select_calibration_threshold(graph_results):
    taus, recovery, partial_fpr, baseline_fpr = _threshold_sweep(graph_results)
    feasible = [
        index
        for index in range(len(taus))
        if partial_fpr[index] <= FALSE_POSITIVE_RATE_CAP
        and baseline_fpr[index] <= FALSE_POSITIVE_RATE_CAP
        and recovery[index] > 0.0
    ]
    rows = [
        {
            "tau": float(taus[index]),
            "partial_recovery_book_score": float(recovery[index]),
            "partial_recovery_mean_fpr": float(partial_fpr[index]),
            "no_diffusion_mean_fpr": float(baseline_fpr[index]),
        }
        for index in range(len(taus))
    ]
    first_nonzero = next((row for row in rows if row["partial_recovery_book_score"] > 0), None)
    feasible_rows = [rows[index] for index in feasible]
    best_under_cap = max(
        (
            row
            for row in rows
            if row["partial_recovery_mean_fpr"] <= FALSE_POSITIVE_RATE_CAP
            and row["no_diffusion_mean_fpr"] <= FALSE_POSITIVE_RATE_CAP
        ),
        key=lambda row: (
            row["partial_recovery_book_score"],
            -max(row["partial_recovery_mean_fpr"], row["no_diffusion_mean_fpr"]),
            row["tau"],
        ),
        default=None,
    )
    tradeoff_indices = set()
    for cap in (0.0, 0.01, 0.025, FALSE_POSITIVE_RATE_CAP, 0.10, 0.25, 0.50, 1.0):
        matching = [
            index
            for index, row in enumerate(rows)
            if row["partial_recovery_mean_fpr"] <= cap
            and row["no_diffusion_mean_fpr"] <= cap
        ]
        if matching:
            tradeoff_indices.add(
                max(
                    matching,
                    key=lambda index: (
                        rows[index]["partial_recovery_book_score"],
                        -max(
                            rows[index]["partial_recovery_mean_fpr"],
                            rows[index]["no_diffusion_mean_fpr"],
                        ),
                        rows[index]["tau"],
                    ),
                )
            )
    if first_nonzero is not None:
        tradeoff_indices.add(rows.index(first_nonzero))
    if best_under_cap is not None:
        tradeoff_indices.add(rows.index(best_under_cap))
    max_recovery_row = max(
        rows,
        key=lambda row: (
            row["partial_recovery_book_score"],
            -max(
                row["partial_recovery_mean_fpr"],
                row["no_diffusion_mean_fpr"],
            ),
            row["tau"],
        ),
    )
    tradeoff_indices.add(rows.index(max_recovery_row))
    tradeoff = [rows[index] for index in sorted(tradeoff_indices, key=lambda i: rows[i]["tau"], reverse=True)]
    if not feasible_rows:
        return {
            "status": "no_threshold_meets_nonzero_recovery_and_fpr_gate",
            "tau": None,
            "false_positive_rate_cap": FALSE_POSITIVE_RATE_CAP,
            "selection_objective": (
                "maximize macro partial-recovery book stability score subject "
                "to both methods' macro mean FPR <= 0.05 and nonzero partial recovery; "
                "ties: lower mean of the two FPRs, then higher tau"
            ),
            "best_threshold_under_fpr_cap": best_under_cap,
            "first_nonzero_recovery_threshold": first_nonzero,
            "calibration_tradeoff": tradeoff,
            "maximum_recovery_at_any_threshold": max_recovery_row,
        }
    selected = max(
        feasible_rows,
        key=lambda row: (
            row["partial_recovery_book_score"],
            -((row["partial_recovery_mean_fpr"] + row["no_diffusion_mean_fpr"]) / 2),
            row["tau"],
        ),
    )
    return {
        "status": "threshold_selected",
        "tau": selected["tau"],
        "false_positive_rate_cap": FALSE_POSITIVE_RATE_CAP,
        "selection_objective": (
            "maximize macro partial-recovery book stability score subject "
            "to both methods' macro mean FPR <= 0.05 and nonzero partial recovery; "
            "ties: lower mean FPR, then higher tau"
        ),
        "calibration_partial_recovery_book_score": selected[
            "partial_recovery_book_score"
        ],
        "calibration_partial_recovery_mean_fpr": selected[
            "partial_recovery_mean_fpr"
        ],
        "calibration_no_diffusion_mean_fpr": selected[
            "no_diffusion_mean_fpr"
        ],
        "first_nonzero_recovery_threshold": first_nonzero,
        "best_threshold_under_fpr_cap": best_under_cap,
        "candidate_threshold_count": len(rows),
        "calibration_tradeoff": tradeoff,
    }


def score_case(case, original_edges, node_ids, tau):
    candidates = tuple(tuple(edge) for edge in case["candidate_edges"])
    original = {frozenset(edge) for edge in original_edges}
    removed = tuple(tuple(edge) for edge in case["removed_edges"])
    removed_keys = {frozenset(edge) for edge in removed}
    surviving = original - removed_keys
    nonedges = {
        frozenset((source, target))
        for index, source in enumerate(node_ids)
        for target in node_ids[index + 1 :]
    } - original

    def metrics(probability_rows):
        edge_sets = []
        rows = []
        for probability_row in probability_rows:
            probabilities = np.asarray(probability_row)
            predicted = {
                frozenset(edge)
                for edge, probability in zip(candidates, probabilities)
                if probability >= tau
            }
            edge_sets.append(tuple(edge for edge in candidates if frozenset(edge) in predicted))
            tp = len(predicted & removed_keys)
            fp = len(predicted & nonedges)
            additions = tp + fp
            rows.append(
                {
                    "removed_edges_recovered": tp,
                    "false_positive_edges": fp,
                    "false_positive_rate": fp / len(nonedges) if nonedges else 0.0,
                    "added_edge_precision": tp / additions if additions else None,
                    "surviving_edge_retention": (
                        len(predicted & surviving) / len(surviving)
                        if surviving
                        else 1.0
                    ),
                    "predicted_edge_count": len(predicted),
                }
            )
        valid_precision = [
            row["added_edge_precision"]
            for row in rows
            if row["added_edge_precision"] is not None
        ]
        diversity = len(
            {
                frozenset(frozenset(edge) for edge in edge_set)
                for edge_set in edge_sets
            }
        )
        pairwise_distances = []
        for first, second in itertools.combinations(edge_sets, 2):
            first_set = {frozenset(edge) for edge in first}
            second_set = {frozenset(edge) for edge in second}
            union = first_set | second_set
            pairwise_distances.append(
                0.0 if not union else 1.0 - len(first_set & second_set) / len(union)
            )
        return {
            "book_stability_score": score_recovery(removed, edge_sets).scenario_score,
            "mean_added_edge_precision": (
                float(np.mean(valid_precision)) if valid_precision else None
            ),
            "mean_false_positive_rate": float(
                np.mean([row["false_positive_rate"] for row in rows])
            ),
            "mean_false_positive_edges": float(
                np.mean([row["false_positive_edges"] for row in rows])
            ),
            "mean_surviving_edge_retention": float(
                np.mean([row["surviving_edge_retention"] for row in rows])
            ),
            "mean_predicted_edge_count": float(
                np.mean([row["predicted_edge_count"] for row in rows])
            ),
            "distinct_reconstructions": diversity,
            "unique_reconstruction_fraction": diversity / len(edge_sets),
            "mean_pairwise_jaccard_distance": (
                float(np.mean(pairwise_distances))
                if pairwise_distances
                else None
            ),
            "per_reconstruction": rows,
            "true_nonedge_count": len(nonedges),
        }

    partial = metrics(case["partial_probabilities"])
    baseline = metrics([case["baseline_probabilities"]])
    baseline["unique_reconstruction_fraction"] = None
    baseline["mean_pairwise_jaccard_distance"] = None
    return {"partial_recovery": partial, "no_diffusion_baseline": baseline}


def summarize_split(graph_results, tau):
    attack_level_rows = []
    for graph in graph_results:
        for attack_type in ("targeted_degree_product", "uniform_random"):
            for fraction in DAMAGE_FRACTIONS:
                cases = [
                    case
                    for case in graph["cases"]
                    if case["attack_type"] == attack_type
                    and case["damage_fraction_requested"] == fraction
                ]
                scored = [
                    score_case(case, graph["original_edges"], graph["node_ids"], tau)
                    for case in cases
                ]
                method_rows = {}
                for method in ("partial_recovery", "no_diffusion_baseline"):
                    metrics = [
                        row[method]
                        for row in scored
                    ]
                    names = (
                        "book_stability_score",
                        "mean_added_edge_precision",
                        "mean_false_positive_rate",
                        "mean_false_positive_edges",
                        "mean_surviving_edge_retention",
                        "mean_predicted_edge_count",
                    )
                    method_rows[method] = {
                        name: (
                            float(np.mean(values))
                            if (values := [
                                metric[name]
                                for metric in metrics
                                if metric[name] is not None
                            ])
                            else None
                        )
                        for name in names
                    }
                    method_rows[method]["mask_count"] = len(metrics)
                    method_rows[method]["masks"] = [
                        {
                            "mask_index": case["mask_index"],
                            "mask_seed": case["mask_seed"],
                            "edges_removed": case["edges_removed"],
                            "false_positive_edges": metric["mean_false_positive_edges"],
                            "false_positive_rate": metric["mean_false_positive_rate"],
                            "book_stability_score": metric["book_stability_score"],
                            "added_edge_precision": metric["mean_added_edge_precision"],
                            "surviving_edge_retention": metric[
                                "mean_surviving_edge_retention"
                            ],
                            "distinct_reconstructions": metric[
                                "distinct_reconstructions"
                            ],
                            "mean_pairwise_jaccard_distance": metric[
                                "mean_pairwise_jaccard_distance"
                            ],
                        }
                        for case, metric in zip(cases, metrics)
                    ]
                attack_level_rows.append(
                    {
                        "graph_id": graph["graph_id"],
                        "attack_type": attack_type,
                        "damage_fraction_requested": fraction,
                        "mask_count": len(cases),
                        "metrics": method_rows,
                    }
                )
    return attack_level_rows


def macro_attack_balanced(rows):
    output = {}
    for method in ("partial_recovery", "no_diffusion_baseline"):
        output[method] = {}
        for metric in (
            "book_stability_score",
            "mean_added_edge_precision",
            "mean_false_positive_rate",
            "mean_false_positive_edges",
            "mean_surviving_edge_retention",
            "mean_predicted_edge_count",
        ):
            attack_means = []
            for attack_type in ("targeted_degree_product", "uniform_random"):
                values = [
                    row["metrics"][method][metric]
                    for row in rows
                    if row["attack_type"] == attack_type
                    and row["metrics"][method][metric] is not None
                ]
                if values:
                    attack_means.append(float(np.mean(values)))
            output[method][metric] = (
                float(np.mean(attack_means)) if attack_means else None
            )
    return output


def selection_manifest(calibration, evaluation, eligible_count, distinct_count):
    def graph_row(graph):
        return {
            "dataset_index": graph.dataset_index,
            "graph_id": graph.graph_id,
            "content_sha256": graph.content_sha256,
            "node_count": len(graph.node_ids),
            "edge_count": len(graph.edges),
            "feature_dim": graph.features.shape[1],
        }

    return {
        "selection_seed": SELECTION_SEED,
        "excluded_graph_ids": sorted(USED_GRAPH_IDS),
        "eligible_fresh_graph_count": eligible_count,
        "distinct_fresh_graph_count": distinct_count,
        "calibration": [graph_row(graph) for graph in calibration],
        "evaluation": [graph_row(graph) for graph in evaluation],
        "disjoint_content_hashes": not bool(
            {graph.content_sha256 for graph in calibration}
            & {graph.content_sha256 for graph in evaluation}
        ),
    }


def removal_count_for_fraction(edge_count: int, fraction: float) -> int:
    return math.floor(edge_count * fraction + 0.5)
