from __future__ import annotations

import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stability.imdb_binary_dataset import (
    IMDBBinaryGraph,
    load_imdb_binary_graphs,
    with_node_identity_features,
)
from stability.partial_recovery import decode_without_diffusion, reconstruct_k
from stability.pilot_config import PilotConfig
from stability.proteins_common_fpr_replay import (
    FALSE_POSITIVE_RATE_CAP,
    select_method_threshold,
    summarize_at_threshold,
    summarize_attack_balanced,
)
from stability.synthetic_instances import SyntheticGraph
from stability.wp7_training import (
    RECONSTRUCTIONS,
    T_STAR,
    _edge_split_validation_first,
    make_adjacency,
    train_graph_models,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tu"
OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "imdb_binary_pilot_20261110"
    / "recovery_20261110_01"
)
SELECTION_SEED = 20261110
MODEL_SEED_OFFSET = 3_000_000
CAP_SECONDS = 20 * 60
DAMAGE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
RANDOM_MASKS_PER_LEVEL = 3
DEVELOPMENT_GRAPH_ID = "IMDB-BINARY-0255"
CALIBRATION_GRAPH_IDS = (
    "IMDB-BINARY-0961",
    "IMDB-BINARY-0502",
    "IMDB-BINARY-0493",
    "IMDB-BINARY-0733",
    "IMDB-BINARY-0737",
)
EVALUATION_GRAPH_IDS = (
    "IMDB-BINARY-0377",
    "IMDB-BINARY-0419",
    "IMDB-BINARY-0027",
    "IMDB-BINARY-0773",
    "IMDB-BINARY-0369",
)
METHODS = ("partial_recovery", "no_diffusion_baseline")


@dataclass(frozen=True)
class PilotGraph:
    graph: SyntheticGraph
    split: str
    source: IMDBBinaryGraph
    graph_sha256: str


def _save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as artifact:
        json.dump(payload, artifact, indent=2, sort_keys=True, allow_nan=False)
        artifact.write("\n")


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _indexed_edges(edges, node_ids):
    indices = {node: index for index, node in enumerate(node_ids)}
    return tuple((indices[source], indices[target]) for source, target in edges)


def _as_synthetic_graph(source: IMDBBinaryGraph) -> SyntheticGraph:
    graph = with_node_identity_features(source)
    return SyntheticGraph(
        graph_id=graph.graph_id,
        seed=graph.dataset_index,
        node_ids=graph.node_ids,
        features=graph.features,
        edges=graph.edges,
    )


def _ordered_unique_graphs(graphs, seed: int):
    ordered = sorted(
        graphs, key=lambda graph: (graph.content_sha256, graph.dataset_index)
    )
    unique = []
    seen_hashes = set()
    for graph in ordered:
        if graph.content_sha256 not in seen_hashes:
            unique.append(graph)
            seen_hashes.add(graph.content_sha256)
    permutation = np.random.default_rng(seed).permutation(len(unique))
    return [unique[index] for index in permutation]


def _split_supported(graph: IMDBBinaryGraph) -> bool:
    try:
        _edge_split_validation_first(
            graph.edges,
            _as_synthetic_graph(graph),
            MODEL_SEED_OFFSET + graph.dataset_index + 1,
        )
    except ValueError:
        return False
    return True


def _select_graphs(graphs):
    ordered = _ordered_unique_graphs(graphs, SELECTION_SEED)
    first_ten = [graph.graph_id for graph in ordered[:10]]
    expected_first_ten = (
        DEVELOPMENT_GRAPH_ID,
        "IMDB-BINARY-0961",
        "IMDB-BINARY-0502",
        "IMDB-BINARY-0493",
        "IMDB-BINARY-0733",
        "IMDB-BINARY-0377",
        "IMDB-BINARY-0419",
        "IMDB-BINARY-0027",
        "IMDB-BINARY-0773",
        "IMDB-BINARY-0369",
    )
    if tuple(first_ten) != expected_first_ten:
        raise RuntimeError(
            "seeded IMDB-BINARY graph order differs from the approved manifest"
        )
    by_id = {graph.graph_id: graph for graph in graphs}
    if DEVELOPMENT_GRAPH_ID not in by_id:
        raise ValueError(f"development graph is unavailable: {DEVELOPMENT_GRAPH_ID}")

    calibration_ids = list(CALIBRATION_GRAPH_IDS)
    evaluation_ids = list(EVALUATION_GRAPH_IDS)
    replacements = []
    used_ids = {DEVELOPMENT_GRAPH_ID, *calibration_ids, *evaluation_ids}
    candidates = ordered[10:]
    candidate_index = 0
    for split_name, graph_ids in (
        ("calibration", calibration_ids),
        ("evaluation", evaluation_ids),
    ):
        for index, graph_id in enumerate(tuple(graph_ids)):
            graph = by_id.get(graph_id)
            if graph is not None and _split_supported(graph):
                continue
            reason = "missing" if graph is None else "validation-first split unsupported"
            while candidate_index < len(candidates):
                candidate = candidates[candidate_index]
                candidate_index += 1
                if candidate.graph_id in used_ids or not _split_supported(candidate):
                    continue
                graph_ids[index] = candidate.graph_id
                used_ids.add(candidate.graph_id)
                replacements.append(
                    {
                        "partition": split_name,
                        "replaced_graph_id": graph_id,
                        "replacement_graph_id": candidate.graph_id,
                        "reason": reason,
                    }
                )
                break
            else:
                raise ValueError(
                    f"could not select five {split_name} graphs with nonempty "
                    "validation-first splits"
                )

    selected_ids = set(calibration_ids) | set(evaluation_ids)
    if len(calibration_ids) != 5 or len(evaluation_ids) != 5:
        raise RuntimeError("pilot requires five calibration and five evaluation graphs")
    if len(selected_ids) != 10 or DEVELOPMENT_GRAPH_ID in selected_ids:
        raise RuntimeError("development/calibration/evaluation graphs overlap")
    selected = {
        graph_id: by_id[graph_id]
        for graph_id in selected_ids
    }
    return calibration_ids, evaluation_ids, selected, replacements, ordered


def _removal_count(edge_count: int, fraction: float) -> int:
    return min(edge_count, max(1, int(math.floor(edge_count * fraction + 0.5))))


def _case_plan(
    graph: IMDBBinaryGraph,
    split: str,
    split_index: int,
):
    node_order = {node: index for index, node in enumerate(graph.node_ids)}
    canonical_edges = tuple(
        sorted(
            graph.edges,
            key=lambda edge: (node_order[edge[0]], node_order[edge[1]]),
        )
    )
    degrees = {node: 0 for node in graph.node_ids}
    for source, target in canonical_edges:
        degrees[source] += 1
        degrees[target] += 1
    targeted_order = tuple(
        sorted(
            canonical_edges,
            key=lambda edge: (
                -(degrees[edge[0]] * degrees[edge[1]]),
                node_order[edge[0]],
                node_order[edge[1]],
            ),
        )
    )
    plans = []
    split_code = 1 if split == "calibration" else 2
    for level_index, fraction in enumerate(DAMAGE_FRACTIONS, start=1):
        removal_count = _removal_count(len(canonical_edges), fraction)
        reconstruction_seed_base = (
            7_000_000
            + split_code * 1_000_000
            + split_index * 10_000
            + level_index * 100
        )
        masks = [
            {
                "attack_type": "targeted_degree_product",
                "mask_index": 0,
                "mask_seed": None,
                "removed_edges": targeted_order[:removal_count],
                "reconstruction_seed_base": reconstruction_seed_base,
            }
        ]
        used_masks = {
            frozenset(frozenset(edge) for edge in targeted_order[:removal_count])
        }
        for mask_index in range(1, RANDOM_MASKS_PER_LEVEL + 1):
            base_seed = (
                SELECTION_SEED
                + graph.dataset_index * 10_000
                + level_index * 100
                + mask_index
            )
            mask_seed = base_seed
            attempt = 0
            while True:
                removed = tuple(
                    sorted(
                        random.Random(mask_seed).sample(
                            canonical_edges, removal_count
                        ),
                        key=lambda edge: (
                            node_order[edge[0]],
                            node_order[edge[1]],
                        ),
                    )
                )
                mask_key = frozenset(frozenset(edge) for edge in removed)
                if mask_key not in used_masks:
                    break
                attempt += 1
                if attempt > 10_000:
                    raise RuntimeError(
                        f"could not create distinct masks for {graph.graph_id}"
                    )
                mask_seed += RANDOM_MASKS_PER_LEVEL
            used_masks.add(mask_key)
            masks.append(
                {
                    "attack_type": "uniform_random",
                    "mask_index": mask_index,
                    "mask_seed": mask_seed,
                    "mask_seed_attempt": attempt,
                    "removed_edges": removed,
                    "reconstruction_seed_base": (
                        reconstruction_seed_base + mask_index * 10
                    ),
                }
            )
        for mask in masks:
            removed_set = {frozenset(edge) for edge in mask["removed_edges"]}
            damaged_edges = tuple(
                edge
                for edge in canonical_edges
                if frozenset(edge) not in removed_set
            )
            plans.append(
                {
                    **mask,
                    "damage_fraction_requested": fraction,
                    "damage_fraction_realized": (
                        removal_count / len(canonical_edges)
                    ),
                    "edges_removed": removal_count,
                    "damaged_edges": damaged_edges,
                }
            )
    return plans


def _candidate_pairs(node_ids):
    return tuple(itertools.combinations(node_ids, 2))


def _embedding_diversity(embeddings):
    rounded = torch.round(embeddings * 1e7) / 1e7
    distances = torch.pdist(embeddings)
    return {
        "distinct_rows_at_1e-7": int(torch.unique(rounded, dim=0).shape[0]),
        "minimum_pairwise_distance": float(distances.min()),
        "mean_pairwise_distance": float(distances.mean()),
        "maximum_pairwise_distance": float(distances.max()),
        "maximum_norm": float(torch.linalg.vector_norm(embeddings, dim=1).max()),
        "finite": bool(torch.isfinite(embeddings).all()),
    }


def _score_ties(probability_rows, candidate_labels):
    rows = []
    for values in probability_rows:
        rounded = np.round(np.asarray(values, dtype=np.float64), 7)
        unique, counts = np.unique(rounded, return_counts=True)
        tied_groups = []
        for value, count in zip(unique, counts):
            if count <= 1:
                continue
            positions = np.flatnonzero(rounded == value)
            labels = candidate_labels[positions]
            tied_groups.append(
                {
                    "score_rounded_1e-7": float(value),
                    "count": int(count),
                    "removed_edge_positives": int(np.count_nonzero(labels == 1)),
                    "true_nonedge_negatives": int(np.count_nonzero(labels == 0)),
                }
            )
        rows.append(
            {
                "distinct_scores_at_1e-7": int(len(unique)),
                "tie_excess_pairs": int(np.maximum(counts - 1, 0).sum()),
                "tied_score_groups": tied_groups,
            }
        )
    return rows


def _case_result(graph, plan, encoder, denoiser, context, deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("20-minute pilot CPU budget reached between masks")
    candidates_all = _candidate_pairs(graph.node_ids)
    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    original_keys = {frozenset(edge) for edge in graph.edges}
    removed_edges = tuple(plan["removed_edges"])
    damaged_edges = tuple(plan["damaged_edges"])
    damaged_keys = {frozenset(edge) for edge in damaged_edges}
    candidates = tuple(
        edge
        for edge in candidates_all
        if frozenset(edge) not in damaged_keys
    )
    if not candidates or not removed_edges:
        raise RuntimeError(f"{graph.graph_id}: empty recovery candidate or positive set")
    candidate_index = {frozenset(edge): index for index, edge in enumerate(candidates)}
    if any(frozenset(edge) not in candidate_index for edge in removed_edges):
        raise RuntimeError(f"{graph.graph_id}: a removed edge is absent from candidates")
    true_nonedges = {
        frozenset(edge)
        for edge in candidates
        if frozenset(edge) not in original_keys
    }
    if not true_nonedges:
        raise RuntimeError(f"{graph.graph_id}: candidate set has no true nonedges")

    adjacency = make_adjacency(
        len(graph.node_ids), _indexed_edges(damaged_edges, graph.node_ids)
    )
    if adjacency.shape != (len(graph.node_ids), len(graph.node_ids)):
        raise RuntimeError(f"{graph.graph_id}: damaged adjacency shape changed")
    features_before = graph.features.clone()
    ids_before = tuple(graph.node_ids)
    with torch.no_grad():
        damaged_embeddings = encoder.encode(graph.features, adjacency)[2]
        baseline_edges, baseline_probabilities = decode_without_diffusion(
            encoder, damaged_embeddings, graph.node_ids, candidates, tau=0.0
        )
    if tuple(graph.node_ids) != ids_before or not torch.equal(
        graph.features, features_before
    ):
        raise RuntimeError(f"{graph.graph_id}: IDs/features mutated during encoding")
    if damaged_embeddings.shape[0] != len(graph.node_ids):
        raise RuntimeError(f"{graph.graph_id}: damaged embedding rows do not align")
    if context.node_ids != graph.node_ids:
        raise RuntimeError(f"{graph.graph_id}: context node IDs are misaligned")
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
        num_reconstructions=RECONSTRUCTIONS,
        seed=int(plan["reconstruction_seed_base"]),
        config=PilotConfig(curvature=1.0, tau=0.0),
    )
    partial_probabilities = np.stack(
        [row.detach().cpu().numpy() for row in batch.probabilities]
    ).astype(np.float64, copy=False)
    baseline_values = baseline_probabilities.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    if not np.isfinite(partial_probabilities).all() or not np.isfinite(
        baseline_values
    ).all():
        raise RuntimeError(f"{graph.graph_id}: decoder produced non-finite scores")

    candidate_labels = np.asarray(
        [
            1 if frozenset(edge) in original_keys else 0
            for edge in candidates
        ],
        dtype=np.int8,
    )
    removed_indices = [candidate_index[frozenset(edge)] for edge in removed_edges]
    nonedge_indices = [
        index
        for index, edge in enumerate(candidates)
        if frozenset(edge) not in original_keys
    ]
    return {
        **{
            key: value
            for key, value in plan.items()
            if key not in {"damaged_edges"}
        },
        "damaged_edges": [list(edge) for edge in damaged_edges],
        "removed_edges": [list(edge) for edge in removed_edges],
        "candidate_edges": [list(edge) for edge in candidates],
        "removed_indices": removed_indices,
        "nonedge_indices": nonedge_indices,
        "baseline_probabilities": baseline_values.tolist(),
        "partial_probabilities": partial_probabilities.tolist(),
        "damaged_embedding_diversity": _embedding_diversity(
            damaged_embeddings.detach()
        ),
        "score_ties": {
            "no_diffusion_baseline": _score_ties(
                baseline_values[None, :], candidate_labels
            )[0],
            "partial_recovery_per_draw": _score_ties(
                partial_probabilities, candidate_labels
            ),
        },
        "identity_alignment": {
            "node_ids_preserved": tuple(graph.node_ids) == ids_before,
            "feature_rows_preserved": torch.equal(graph.features, features_before),
            "feature_shape": list(graph.features.shape),
            "adjacency_shape": list(adjacency.shape),
            "embedding_rows": int(damaged_embeddings.shape[0]),
            "community_context_node_ids_match": context.node_ids == graph.node_ids,
        },
        "baseline_predicted_edges_at_tau_zero": [list(edge) for edge in baseline_edges],
    }


def _train_and_score(
    pilot_graph: PilotGraph,
    plans,
    model_seed: int,
    checkpoint_dir: Path,
    deadline: float,
):
    graph = pilot_graph.graph
    if time.perf_counter() >= deadline:
        raise TimeoutError("20-minute pilot CPU budget reached before graph")
    (
        encoder,
        denoiser,
        context,
        original_adjacency,
        training,
    ) = train_graph_models(
        graph,
        PilotConfig(curvature=1.0, tau=0.0),
        model_seed,
        checkpoint_dir,
        validation_first_negative_split=True,
    )
    with torch.no_grad():
        original_embeddings = encoder.encode(
            graph.features, original_adjacency
        )[2]
    if not torch.isfinite(original_embeddings).all():
        raise RuntimeError(f"{graph.graph_id}: intact embeddings are non-finite")
    if original_embeddings.shape[0] != len(graph.node_ids):
        raise RuntimeError(f"{graph.graph_id}: intact embedding rows are misaligned")

    results = []
    for plan in plans:
        results.append(
            _case_result(
                graph,
                plan,
                encoder,
                denoiser,
                context,
                deadline,
            )
        )
    return {
        "graph_id": graph.graph_id,
        "split": pilot_graph.split,
        "dataset_index": graph.seed,
        "model_seed": model_seed,
        "graph_sha256": pilot_graph.graph_sha256,
        "identity_feature_graph_sha256": with_node_identity_features(
            pilot_graph.source
        ).content_sha256,
        "node_ids": list(graph.node_ids),
        "features": graph.features.tolist(),
        "original_edges": [list(edge) for edge in graph.edges],
        "training": training,
        "original_embedding_diversity": _embedding_diversity(
            original_embeddings.detach()
        ),
        "community_context": {
            "labels": context.labels.cpu().tolist(),
            "centers": context.centers.cpu().tolist(),
            "curvature": context.curvature,
            "computed_from_intact_graph": True,
        },
        "cases": results,
    }


def _threshold_graph(graph_result):
    return {
        "graph_id": graph_result["graph_id"],
        "node_ids": graph_result["node_ids"],
        "original_edges": graph_result["original_edges"],
        "cases": [
            {
                **{
                    key: case[key]
                    for key in (
                        "attack_type",
                        "damage_fraction_requested",
                        "damage_fraction_realized",
                        "edges_removed",
                        "mask_index",
                        "mask_seed",
                        "removed_edges",
                        "candidate_edges",
                        "removed_indices",
                        "nonedge_indices",
                    )
                },
                "baseline_probabilities": np.asarray(
                    case["baseline_probabilities"], dtype=np.float64
                ),
                "partial_probabilities": np.asarray(
                    case["partial_probabilities"], dtype=np.float64
                ),
            }
            for case in graph_result["cases"]
        ],
    }


def _sampling_intervals(graph_results, thresholds, seed):
    rng = np.random.default_rng(seed)
    grouped = {}
    for graph in graph_results:
        original = {frozenset(edge) for edge in graph["original_edges"]}
        for case in graph["cases"]:
            group_key = (
                graph["graph_id"],
                case["attack_type"],
                case["damage_fraction_requested"],
            )
            group = grouped.setdefault(
                group_key,
                {
                    method: {"recovery": [], "false_positives": [], "fpr": []}
                    for method in METHODS
                },
            )
            nonedge_count = len(case["nonedge_indices"])
            for method in METHODS:
                probabilities = np.asarray(
                    case["partial_probabilities"]
                    if method == "partial_recovery"
                    else np.asarray(
                        case["baseline_probabilities"], dtype=np.float64
                    )[None, :],
                    dtype=np.float64,
                )
                removed_indices = np.asarray(case["removed_indices"], dtype=np.int64)
                nonedge_indices = np.asarray(case["nonedge_indices"], dtype=np.int64)
                row_metrics = []
                for row in probabilities:
                    predicted = row >= thresholds[method]
                    recovered = int(np.count_nonzero(predicted[removed_indices]))
                    false_positives = int(np.count_nonzero(predicted[nonedge_indices]))
                    row_metrics.append(
                        (
                            recovered / len(removed_indices),
                            false_positives,
                            false_positives / nonedge_count,
                        )
                    )
                if method == "partial_recovery":
                    for col, metric in enumerate(("recovery", "false_positives", "fpr")):
                        group[method][metric].append(
                            [row[col] for row in row_metrics]
                        )
                else:
                    for col, metric in enumerate(("recovery", "false_positives", "fpr")):
                        group[method][metric].append(
                            [row_metrics[0][col]] * RECONSTRUCTIONS
                        )

    summaries = []
    for (graph_id, attack, damage), methods in sorted(grouped.items()):
        row = {
            "graph_id": graph_id,
            "attack_type": attack,
            "damage_fraction_requested": damage,
            "methods": {},
        }
        for method in METHODS:
            row["methods"][method] = {}
            for metric, masks in methods[method].items():
                values = np.asarray(masks, dtype=np.float64)
                if method != "partial_recovery":
                    row["methods"][method][metric] = {
                        "mean": float(values.mean()),
                        "ci95": None,
                        "scope": "deterministic baseline; no reconstruction sampling",
                    }
                    continue
                draws = np.empty(2000, dtype=np.float64)
                for index in range(len(draws)):
                    sample_indices = rng.integers(
                        0, RECONSTRUCTIONS, size=values.shape
                    )
                    draws[index] = np.take_along_axis(
                        values, sample_indices, axis=1
                    ).mean()
                low, high = np.quantile(draws, (0.025, 0.975))
                row["methods"][method][metric] = {
                    "mean": float(values.mean()),
                    "ci95": [float(low), float(high)],
                    "scope": (
                        "conditional on these evaluation graphs and fixed masks; "
                        "resamples only the eight reconstruction draws"
                    ),
                }
        summaries.append(row)
    return summaries


def _make_manifest(
    calibration_ids,
    evaluation_ids,
    selected,
    replacements,
    ordered,
    raw_hashes,
    dataset_metadata,
):
    return {
        "artifact_type": "IMDB-BINARY paired known-graph recovery pilot",
        "status": "predefined_before_pilot_outcomes",
        "dataset": "IMDB-BINARY",
        "raw_file_sha256": raw_hashes,
        "dataset_metadata": dataset_metadata,
        "eligibility": {
            "bounds": "12-50 nodes, at most 200 unique edges, connected simple graph",
            "eligible_graph_count": len(ordered),
            "distinct_graph_count": len(
                {graph.content_sha256 for graph in ordered}
            ),
            "selection_seed": SELECTION_SEED,
            "selection_order_prefix": [graph.graph_id for graph in ordered[:12]],
            "development_graph_id": DEVELOPMENT_GRAPH_ID,
            "development_role": "separate development evidence; excluded from calibration and evaluation",
            "calibration_graph_ids": calibration_ids,
            "evaluation_graph_ids": evaluation_ids,
            "split_replacements": replacements,
            "graph_hashes": {
                graph_id: selected[graph_id].content_sha256
                for graph_id in sorted(selected)
            },
            "identity_feature_graph_hashes": {
                graph_id: with_node_identity_features(selected[graph_id]).content_sha256
                for graph_id in sorted(selected)
            },
        },
        "features": (
            "constant all-ones first column plus one-hot local node identity; "
            "node_ids remain in graph-indicator order; no graph labels used"
        ),
        "edge_training_validation_split": {
            "model_seed": "3000000 + dataset_index",
            "split_seed": "model_seed + 1",
            "positive_split": (
                "numpy default_rng(split_seed); permute unique intact edges; "
                "reserve max(1, floor(E/5)) for validation"
            ),
            "negative_split": (
                "enumerate true nonedges from all unordered pairs; permute with "
                "default_rng(split_seed + 1); reserve validation negatives first "
                "(same count as validation positives), then use remaining "
                "nonedges for training up to the training-positive count; no replacement"
            ),
            "unsupported_graph_rule": (
                "require nonempty train positives, train negatives, validation "
                "positives, and validation negatives; otherwise replace with the "
                "next unused distinct eligible graph in seeded order within the "
                "same partition and record the reason"
            ),
            "validation_use": (
                "training-loop validation loss only; fixed epochs, no "
                "validation-based checkpoint or threshold selection"
            ),
        },
        "training": {
            "device": "CPU",
            "encoder_epochs": 30,
            "encoder_optimizer": "Adam",
            "encoder_learning_rate": 0.01,
            "denoiser_epochs": 50,
            "denoiser_optimizer": "Adam",
            "denoiser_learning_rate": 0.001,
            "curvature": 1.0,
            "activation": "ReLU",
            "diffusion_schedule_length": 200,
            "t_star": T_STAR,
            "partial_reconstructions_per_mask": RECONSTRUCTIONS,
            "same_graph_specific_model_for_both_methods": True,
        },
        "perturbations": {
            "damage_fractions": DAMAGE_FRACTIONS,
            "rounding": "max(1, floor(fraction * unique_undirected_edge_count + 0.5))",
            "uniform_random": {
                "masks_per_graph_and_level": RANDOM_MASKS_PER_LEVEL,
                "seed": "20261110 + dataset_index*10000 + level_index*100 + mask_index; deterministic collision retry",
            },
            "targeted_degree_product": (
                "one deterministic mask per graph and level; descending "
                "intact degree(u)*degree(v), then ascending endpoint node-ID order"
            ),
            "candidate_pairs": (
                "all unordered pairs absent from the damaged input; identical "
                "candidate order for both methods"
            ),
        },
        "methods": {
            "partial_recovery": (
                "encode damaged adjacency, sample eight reconstructions at t*=7, "
                "decode each candidate set"
            ),
            "no_diffusion_baseline": (
                "decode the same damaged-input embeddings and candidate set without diffusion"
            ),
            "paired_inputs": "identical masks, damaged inputs, node order, and candidate pairs",
            "thresholds": (
                "independently maximize mean removed-edge recovery under macro "
                "mean true-nonedge FPR <= 0.05 on calibration graphs only; "
                "macro-average graph x attack x damage cells, averaging masks "
                "and reconstruction draws within each cell; inclusive >= threshold"
            ),
            "evaluation_gate": (
                "freeze both method-specific thresholds before evaluation; "
                "evaluate both methods only if both calibration gates pass; "
                "otherwise report both calibration trade-offs and skip both"
            ),
        },
        "metrics": {
            "positives": "removed intact edges",
            "false_positives": "predicted pairs absent from the intact graph",
            "reported": (
                "removed-edge recovery, absolute false positives, realized FPR, "
                "embedding duplicate counts/distances, score tie groups"
            ),
            "uncertainty": (
                "conditional on tested evaluation graphs and fixed masks; "
                "resample eight reconstruction draws only"
            ),
            "labels": "used only for reporting/calibration metrics, never to filter predictions",
        },
        "cpu_budget_seconds": CAP_SECONDS,
        "preservation": (
            "new unused artifact directory; all prior IMDB-BINARY, PROTEINS, "
            "and synthetic outputs remain unchanged"
        ),
    }


def run_pilot() -> str:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite pilot directory: {OUTPUT_DIR}")
    started = time.perf_counter()
    deadline = started + CAP_SECONDS
    torch.set_num_threads(1)
    graphs, raw_hashes, dataset_metadata = load_imdb_binary_graphs(DATA_ROOT)
    calibration_ids, evaluation_ids, selected, replacements, ordered = _select_graphs(
        graphs
    )
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=False)

    split_graphs = {}
    plans_by_graph = {}
    model_seeds = {}
    split_indices = {}
    for split_name, graph_ids in (
        ("calibration", calibration_ids),
        ("evaluation", evaluation_ids),
    ):
        split_indices[split_name] = {}
        for split_index, graph_id in enumerate(graph_ids):
            graph = _as_synthetic_graph(selected[graph_id])
            split_graphs[graph_id] = PilotGraph(
                graph=graph,
                split=split_name,
                source=selected[graph_id],
                graph_sha256=selected[graph_id].content_sha256,
            )
            plans_by_graph[graph_id] = _case_plan(
                selected[graph_id], split_name, split_index
            )
            model_seeds[graph_id] = MODEL_SEED_OFFSET + graph.seed
            split_indices[split_name][graph_id] = split_index

    manifest = _make_manifest(
        calibration_ids,
        evaluation_ids,
        selected,
        replacements,
        ordered,
        raw_hashes,
        dataset_metadata,
    )
    manifest_path = output_dir / "selection_and_masks.json"
    _save(manifest_path, manifest)
    manifest_hash = _sha256(manifest_path.read_bytes())
    _save(
        output_dir / "work_in_progress.json",
        {
            "status": "running",
            "current_stage": "calibration_training_and_scoring",
            "manifest_sha256": manifest_hash,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )

    calibration_results = []
    calibration_dir = output_dir / "models" / "calibration"
    try:
        for graph_id in calibration_ids:
            pilot_graph = split_graphs[graph_id]
            result = _train_and_score(
                pilot_graph,
                plans_by_graph[graph_id],
                model_seeds[graph_id],
                calibration_dir,
                deadline,
            )
            calibration_results.append(result)
            _save(
                output_dir / "calibration" / "graphs" / f"{graph_id}.json",
                result,
            )
            _save(
                output_dir / "work_in_progress.json",
                {
                    "status": "running",
                    "current_stage": "calibration_training_and_scoring",
                    "completed_calibration_graphs": [
                        item["graph_id"] for item in calibration_results
                    ],
                    "manifest_sha256": manifest_hash,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

        calibration_threshold_graphs = [
            _threshold_graph(item) for item in calibration_results
        ]
        thresholds = {
            method: select_method_threshold(calibration_threshold_graphs, method)
            for method in METHODS
        }
        gate_passed = all(
            thresholds[method]["status"] == "threshold_selected"
            and thresholds[method].get("nonzero_recovery_feasible", False)
            for method in METHODS
        )
        frozen_thresholds = (
            {method: float(thresholds[method]["tau"]) for method in METHODS}
            if gate_passed
            else None
        )
        calibration_report = {
            "status": "calibration_complete",
            "manifest_sha256": manifest_hash,
            "method_thresholds": thresholds,
            "thresholds_frozen_before_evaluation": gate_passed,
            "evaluation_gate_passed_for_both_methods": gate_passed,
            "calibration_graphs": calibration_ids,
            "calibration_duration_seconds": time.perf_counter() - started,
        }
        if frozen_thresholds is not None:
            calibration_report["metrics_at_selected_thresholds"] = {
                "thresholds": frozen_thresholds,
                "per_graph_attack_damage": summarize_at_threshold(
                    calibration_threshold_graphs, frozen_thresholds
                ),
                "macro_mean_by_attack": summarize_attack_balanced(
                    summarize_at_threshold(
                        calibration_threshold_graphs, frozen_thresholds
                    )
                ),
            }
        _save(output_dir / "calibration" / "summary.json", calibration_report)

        if not gate_passed:
            status = "evaluation_skipped_calibration_gate_failed"
            _save(
                output_dir / "summary.json",
                {
                    "status": status,
                    "elapsed_seconds": time.perf_counter() - started,
                    "cpu_budget_seconds": CAP_SECONDS,
                    "calibration": calibration_report,
                    "evaluation": "skipped for both methods because at least one calibration gate failed",
                    "preserved_prior_artifacts": True,
                },
            )
            return status

        evaluation_results = []
        evaluation_dir = output_dir / "models" / "evaluation"
        for graph_id in evaluation_ids:
            pilot_graph = split_graphs[graph_id]
            result = _train_and_score(
                pilot_graph,
                plans_by_graph[graph_id],
                model_seeds[graph_id],
                evaluation_dir,
                deadline,
            )
            evaluation_results.append(result)
            _save(
                output_dir / "evaluation" / "graphs" / f"{graph_id}.json",
                result,
            )
            _save(
                output_dir / "work_in_progress.json",
                {
                    "status": "running",
                    "current_stage": "evaluation_training_and_scoring",
                    "completed_calibration_graphs": calibration_ids,
                    "completed_evaluation_graphs": [
                        item["graph_id"] for item in evaluation_results
                    ],
                    "frozen_thresholds": frozen_thresholds,
                    "manifest_sha256": manifest_hash,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

        evaluation_threshold_graphs = [
            _threshold_graph(item) for item in evaluation_results
        ]
        evaluation_rows = summarize_at_threshold(
            evaluation_threshold_graphs, frozen_thresholds
        )
        evaluation_sampling = _sampling_intervals(
            evaluation_results,
            frozen_thresholds,
            seed=SELECTION_SEED + 99,
        )
        evaluation_report = {
            "status": "evaluation_complete",
            "manifest_sha256": manifest_hash,
            "thresholds_frozen_before_evaluation": True,
            "calibration_summary": "calibration/summary.json",
            "frozen_method_thresholds": frozen_thresholds,
            "macro_mean_by_attack": summarize_attack_balanced(evaluation_rows),
            "per_graph_attack_damage_results": evaluation_rows,
            "conditional_reconstruction_sampling_intervals": evaluation_sampling,
            "evaluation_graphs": evaluation_ids,
            "uncertainty_scope": (
                "conditional on the tested evaluation graphs and fixed masks; "
                "resamples only the eight reconstruction draws"
            ),
            "evaluation_duration_seconds": time.perf_counter() - started,
        }
        _save(output_dir / "evaluation" / "summary.json", evaluation_report)
        status = (
            "complete"
            if time.perf_counter() < deadline
            else "complete_at_cpu_budget_boundary"
        )
        _save(
            output_dir / "summary.json",
            {
                "status": status,
                "elapsed_seconds": time.perf_counter() - started,
                "cpu_budget_seconds": CAP_SECONDS,
                "calibration": calibration_report,
                "evaluation": {
                    "status": evaluation_report["status"],
                    "frozen_method_thresholds": frozen_thresholds,
                    "macro_mean_by_attack": evaluation_report[
                        "macro_mean_by_attack"
                    ],
                },
                "preserved_prior_artifacts": True,
            },
        )
        return status
    except TimeoutError as error:
        _save(
            output_dir / "work_in_progress.json",
            {
                "status": "incomplete_cpu_budget_exceeded",
                "reason": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "completed_calibration_graphs": [
                    item["graph_id"] for item in calibration_results
                ],
                "preserved_partial_outputs": True,
                "manifest_sha256": manifest_hash,
            },
        )
        _save(
            output_dir / "summary.json",
            {
                "status": "incomplete_cpu_budget_exceeded",
                "reason": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "cpu_budget_seconds": CAP_SECONDS,
                "preserved_partial_outputs": True,
                "preserved_prior_artifacts": True,
            },
        )
        return "incomplete_cpu_budget_exceeded"
    except Exception as error:
        _save(
            output_dir / "work_in_progress.json",
            {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "completed_calibration_graphs": [
                    item["graph_id"] for item in calibration_results
                ],
                "preserved_partial_outputs": True,
                "manifest_sha256": manifest_hash,
            },
        )
        _save(
            output_dir / "summary.json",
            {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "cpu_budget_seconds": CAP_SECONDS,
                "preserved_partial_outputs": True,
                "preserved_prior_artifacts": True,
            },
        )
        raise


def finalize_existing_pilot() -> str:
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f"pilot artifact directory is missing: {OUTPUT_DIR}")
    evaluation_summary_path = OUTPUT_DIR / "evaluation" / "summary.json"
    if evaluation_summary_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing evaluation summary: "
            f"{evaluation_summary_path}"
        )
    wip_path = OUTPUT_DIR / "work_in_progress.json"
    previous_wip = json.loads(wip_path.read_text(encoding="utf-8"))
    previous_summary_path = OUTPUT_DIR / "summary.json"
    previous_summary = json.loads(
        previous_summary_path.read_text(encoding="utf-8")
    )
    if previous_wip.get("status") != "failed":
        raise ValueError("finalization is only allowed for a failed partial run")
    manifest_path = OUTPUT_DIR / "selection_and_masks.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = _sha256(manifest_path.read_bytes())
    if manifest_hash != previous_wip.get("manifest_sha256"):
        raise RuntimeError("saved selection manifest hash changed")

    calibration = json.loads(
        (OUTPUT_DIR / "calibration" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if calibration.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("calibration summary references a different manifest")
    if not calibration.get("evaluation_gate_passed_for_both_methods"):
        raise RuntimeError("calibration gate did not pass for both methods")
    frozen_thresholds = {
        method: float(calibration["method_thresholds"][method]["tau"])
        for method in METHODS
    }

    evaluation_ids = manifest["eligibility"]["evaluation_graph_ids"]
    evaluation_results = []
    for graph_id in evaluation_ids:
        graph_path = OUTPUT_DIR / "evaluation" / "graphs" / f"{graph_id}.json"
        if not graph_path.is_file():
            raise FileNotFoundError(f"saved evaluation graph output missing: {graph_path}")
        graph_result = json.loads(graph_path.read_text(encoding="utf-8"))
        if graph_result.get("graph_id") != graph_id:
            raise RuntimeError(f"evaluation graph ID mismatch in {graph_path}")
        evaluation_results.append(graph_result)

    if not (OUTPUT_DIR / "initial_failure.json").exists():
        _save(
            OUTPUT_DIR / "initial_failure.json",
            {
                "work_in_progress_at_failure": previous_wip,
                "summary_at_failure": previous_summary,
            },
        )
    threshold_graphs = [_threshold_graph(result) for result in evaluation_results]
    rows = summarize_at_threshold(threshold_graphs, frozen_thresholds)
    sampling = _sampling_intervals(
        evaluation_results, frozen_thresholds, seed=SELECTION_SEED + 99
    )
    evaluation_report = {
        "status": "evaluation_complete",
        "manifest_sha256": manifest_hash,
        "thresholds_frozen_before_evaluation": True,
        "calibration_summary": "calibration/summary.json",
        "frozen_method_thresholds": frozen_thresholds,
        "macro_mean_by_attack": summarize_attack_balanced(rows),
        "per_graph_attack_damage_results": rows,
        "conditional_reconstruction_sampling_intervals": sampling,
        "evaluation_graphs": evaluation_ids,
        "uncertainty_scope": (
            "conditional on the tested evaluation graphs and fixed masks; "
            "resamples only the eight reconstruction draws"
        ),
        "aggregation_resumed_from_saved_scores": True,
    }
    _save(evaluation_summary_path, evaluation_report)
    elapsed_seconds = float(previous_wip.get("elapsed_seconds", 0.0))
    _save(
        OUTPUT_DIR / "work_in_progress.json",
        {
            "status": "complete_resumed_aggregation",
            "initial_aggregation_failure": previous_wip,
            "evaluation_scores_reused": True,
            "elapsed_seconds_at_original_failure": elapsed_seconds,
            "manifest_sha256": manifest_hash,
        },
    )
    _save(
        previous_summary_path,
        {
            "status": "complete_resumed_aggregation",
            "elapsed_seconds_at_original_failure": elapsed_seconds,
            "cpu_budget_seconds": CAP_SECONDS,
            "calibration": {
                "status": calibration["status"],
                "method_thresholds": calibration["method_thresholds"],
                "evaluation_gate_passed_for_both_methods": True,
            },
            "evaluation": {
                "status": evaluation_report["status"],
                "frozen_method_thresholds": frozen_thresholds,
                "macro_mean_by_attack": evaluation_report[
                    "macro_mean_by_attack"
                ],
            },
            "initial_aggregation_failure": previous_wip.get("failure"),
            "evaluation_scores_reused": True,
            "preserved_prior_artifacts": True,
        },
    )
    return "complete_resumed_aggregation"


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--finalize-existing"]:
        print(finalize_existing_pilot())
    elif not sys.argv[1:]:
        print(run_pilot())
    else:
        raise SystemExit("usage: python -m stability.imdb_binary_recovery_pilot [--finalize-existing]")
