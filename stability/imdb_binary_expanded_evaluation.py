from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from stability.imdb_binary_dataset import (
    load_imdb_binary_graphs,
    with_node_identity_features,
)
from stability.imdb_binary_f1 import (
    feature_distance_matrix,
    graph_feature_histograms,
    prdc_from_distances,
)
from stability.imdb_binary_recovery_pilot import (
    METHODS,
    PilotGraph,
    _as_synthetic_graph,
    _candidate_pairs,
    _threshold_graph,
    _train_and_score,
    _save,
    _sha256,
)
from stability.proteins_common_fpr_replay import (
    summarize_attack_balanced,
    summarize_at_threshold,
)
from stability.wp7_training import RECONSTRUCTIONS, T_STAR


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tu"
OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "imdb_binary_pilot_20261110"
    / "expanded_evaluation_20261110_01"
)
PREVIOUS_RECOVERY_DIR = (
    ROOT
    / "artifacts"
    / "imdb_binary_pilot_20261110"
    / "recovery_20261110_01"
)
PREVIOUS_DISTRIBUTION_DIR = (
    ROOT
    / "artifacts"
    / "imdb_binary_pilot_20261110"
    / "distribution_metrics_20261110_01"
)

SELECTION_SEED = 20261111
MODEL_SEED_OFFSET = 3_000_000
MASK_SEED_BASE = 20261110
RECONSTRUCTION_SEED_BASE = 7_000_000
EXPANDED_SPLIT_CODE = 3
TRAINING_BUDGET_SECONDS = 600
METRICS_BUDGET_SECONDS = 300
BOOTSTRAP_SEED = 20261113
F1_SAMPLE_SEED = 20261112
F1_POINT_REPLICATES = 1_000
F1_BOOTSTRAP_REPLICATES = 2_000
SCORE_BOOTSTRAP_REPLICATES = 10_000

DAMAGE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
RANDOM_MASKS_PER_LEVEL = 3
FROZEN_THRESHOLDS = {
    "partial_recovery": 0.8807944655418396,
    "no_diffusion_baseline": 0.8806627988815308,
}
EXCLUDED_GRAPH_IDS = (
    "IMDB-BINARY-0255",
    "IMDB-BINARY-0961",
    "IMDB-BINARY-0502",
    "IMDB-BINARY-0493",
    "IMDB-BINARY-0733",
    "IMDB-BINARY-0737",
    "IMDB-BINARY-0377",
    "IMDB-BINARY-0419",
    "IMDB-BINARY-0027",
    "IMDB-BINARY-0773",
    "IMDB-BINARY-0369",
)
EXPECTED_SELECTED_IDS = (
    "IMDB-BINARY-0739",
    "IMDB-BINARY-0276",
    "IMDB-BINARY-0433",
    "IMDB-BINARY-0144",
    "IMDB-BINARY-0578",
    "IMDB-BINARY-0192",
    "IMDB-BINARY-0606",
    "IMDB-BINARY-0656",
    "IMDB-BINARY-0609",
    "IMDB-BINARY-0762",
    "IMDB-BINARY-0177",
    "IMDB-BINARY-0731",
    "IMDB-BINARY-0732",
    "IMDB-BINARY-0761",
    "IMDB-BINARY-0215",
    "IMDB-BINARY-0375",
    "IMDB-BINARY-0162",
    "IMDB-BINARY-0601",
    "IMDB-BINARY-0062",
    "IMDB-BINARY-0445",
)
EXPECTED_GRAPH_HASHES = {
    "IMDB-BINARY-0739": "d57fbd92258f6bf3d9a5e53a341025b9a7352fd85990258983e36a753109411d",
    "IMDB-BINARY-0276": "35bd184e83cb4f661481fb44a07c95a04612d2bcae771612ee496681103ee612",
    "IMDB-BINARY-0433": "16f0ff084b91b71e59ab605e3289c6c7f33b254978ffff40466d85a64fa0f02e",
    "IMDB-BINARY-0144": "f68ae9ec8352a2ca4b19c0b5c34ba0c4d959e9e20c3fd8f7b050cd7fc8d2cb47",
    "IMDB-BINARY-0578": "5a314091e9dc7eb3d706b2e48f57b0e0f3191ccf4f933eca2ca2dd7379cb7f06",
    "IMDB-BINARY-0192": "8bea9a5ff410def6a648a1f2dcfa71365a38c40f3e959aa3e40e33e26149516c",
    "IMDB-BINARY-0606": "b8f800c42ca3a2f8a55059a6c0122eb23887d49cb10e20c0d53d60d60c98522c",
    "IMDB-BINARY-0656": "1aebe9f66e382f7e7d1b68fe5772f7e8c1bd396e86c5bc9292608df315d8fd4f",
    "IMDB-BINARY-0609": "5924084094c4e3f780844eb2a4903aa8c2e5acb907d6157b441b565c5a35deac",
    "IMDB-BINARY-0762": "086b511e4b264f584ad54e4bca193be97c5108dfa5fef2aed9248c5be364903c",
    "IMDB-BINARY-0177": "4f93dc3685d890b655d46e91378707b665d5bd6297617b5d78dd8c56b0395233",
    "IMDB-BINARY-0731": "920aa0bfc643fd707f9d47af188cee49af2f1c9f62fc55114dee1cb5317de23d",
    "IMDB-BINARY-0732": "9fa988f93615641097104e69be544e58339c083d7acf37c28b08178d81173519",
    "IMDB-BINARY-0761": "fdc6ed463937f74d23cd3cd7acc741be34f7b2cf4f25e42fec5a645a44a30bc1",
    "IMDB-BINARY-0215": "2288c952a9e50ecebc0a10d5438cc1df260e50ed0b8d73e45efe5be3284ecb27",
    "IMDB-BINARY-0375": "049427efb37b387cab166f7d012a29ea069ca9e54d9c22861c43730b4e687677",
    "IMDB-BINARY-0162": "cd9fd3a8f8cc0c4763d9d58f0e559b5ca93b99c4ec0e171dced1457da8077f67",
    "IMDB-BINARY-0601": "48227665cdc8f1bfaecd476cce9ff1af009679a1eb1340a1cac0504df238197c",
    "IMDB-BINARY-0062": "a1e8525b9fb8e10334f6a66f4bfa8df18148ddb2d97973e85db48012af4b35f8",
    "IMDB-BINARY-0445": "75812a3a8348ecb5dddff8dc250cd235a25a1964a7d8e53cbcfbfebb5df3ab9c",
}
OLD_EVALUATION_IDS = (
    "IMDB-BINARY-0377",
    "IMDB-BINARY-0419",
    "IMDB-BINARY-0027",
    "IMDB-BINARY-0773",
    "IMDB-BINARY-0369",
)


def _read_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_unique_graphs(graphs):
    ordered = sorted(
        graphs, key=lambda graph: (graph.content_sha256, graph.dataset_index)
    )
    unique = []
    seen = set()
    for graph in ordered:
        if graph.content_sha256 not in seen:
            unique.append(graph)
            seen.add(graph.content_sha256)
    return unique


def _make_expanded_plans(graph, selection_rank: int):
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
    for level_index, fraction in enumerate(DAMAGE_FRACTIONS, start=1):
        removal_count = max(
            1, int(math.floor(len(canonical_edges) * fraction + 0.5))
        )
        seed_base = (
            RECONSTRUCTION_SEED_BASE
            + EXPANDED_SPLIT_CODE * 1_000_000
            + selection_rank * 10_000
            + level_index * 100
        )
        masks = [
            {
                "attack_type": "targeted_degree_product",
                "mask_index": 0,
                "mask_seed": None,
                "removed_edges": targeted_order[:removal_count],
                "reconstruction_seed_base": seed_base,
            }
        ]
        used_masks = {
            frozenset(frozenset(edge) for edge in targeted_order[:removal_count])
        }
        for mask_index in range(1, RANDOM_MASKS_PER_LEVEL + 1):
            mask_seed = (
                MASK_SEED_BASE
                + graph.dataset_index * 10_000
                + level_index * 100
                + mask_index
            )
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
                key = frozenset(frozenset(edge) for edge in removed)
                if key not in used_masks:
                    break
                attempt += 1
                if attempt > 10_000:
                    raise RuntimeError(
                        f"could not create distinct masks for {graph.graph_id}"
                    )
                mask_seed += RANDOM_MASKS_PER_LEVEL
            used_masks.add(key)
            masks.append(
                {
                    "attack_type": "uniform_random",
                    "mask_index": mask_index,
                    "mask_seed": mask_seed,
                    "mask_seed_attempt": attempt,
                    "removed_edges": removed,
                    "reconstruction_seed_base": seed_base + mask_index * 10,
                }
            )

        for mask in masks:
            removed_keys = {
                frozenset(edge) for edge in mask["removed_edges"]
            }
            damaged = tuple(
                edge
                for edge in canonical_edges
                if frozenset(edge) not in removed_keys
            )
            plans.append(
                {
                    **mask,
                    "damage_fraction_requested": fraction,
                    "damage_fraction_realized": (
                        removal_count / len(canonical_edges)
                    ),
                    "edges_removed": removal_count,
                    "damaged_edges": damaged,
                }
            )
    return plans


def prepare_selection_manifest() -> str:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    graphs, raw_hashes, metadata = load_imdb_binary_graphs(
        DATA_ROOT, download=False
    )
    by_id = {graph.graph_id: graph for graph in graphs}
    unique = _ordered_unique_graphs(graphs)
    eligible_candidates = [
        graph
        for graph in unique
        if graph.graph_id not in set(EXCLUDED_GRAPH_IDS)
    ]
    order = np.random.default_rng(SELECTION_SEED).permutation(
        len(eligible_candidates)
    )

    selected = []
    skipped = []
    for index in order:
        candidate = eligible_candidates[int(index)]
        synthetic = _as_synthetic_graph(candidate)
        try:
            from stability.wp7_training import _edge_split_validation_first

            _edge_split_validation_first(
                synthetic.edges,
                synthetic,
                MODEL_SEED_OFFSET + candidate.dataset_index + 1,
            )
        except ValueError as error:
            skipped.append(
                {"graph_id": candidate.graph_id, "reason": str(error)}
            )
            continue
        selected.append(candidate)
        if len(selected) == len(EXPECTED_SELECTED_IDS):
            break
    selected_ids = tuple(graph.graph_id for graph in selected)
    if selected_ids != EXPECTED_SELECTED_IDS:
        raise RuntimeError(
            f"pre-outcome selected IDs changed: {selected_ids}"
        )
    for graph in selected:
        if graph.content_sha256 != EXPECTED_GRAPH_HASHES[graph.graph_id]:
            raise RuntimeError(
                f"pre-outcome graph hash changed: {graph.graph_id}"
            )
    if set(selected_ids) & set(EXCLUDED_GRAPH_IDS):
        raise RuntimeError("an excluded source graph was selected")
    if len({graph.content_sha256 for graph in selected}) != 20:
        raise RuntimeError("selected source graphs are not content-distinct")

    selected_rows = []
    for rank, source in enumerate(selected, start=1):
        model_seed = MODEL_SEED_OFFSET + source.dataset_index
        graph = with_node_identity_features(source)
        plans = _make_expanded_plans(source, rank - 1)
        selected_rows.append(
            {
                "selection_rank": rank,
                "graph_id": source.graph_id,
                "dataset_index": source.dataset_index,
                "graph_sha256": source.content_sha256,
                "identity_feature_graph_sha256": graph.content_sha256,
                "model_seed": model_seed,
                "edge_split_seed": model_seed + 1,
                "node_ids": list(source.node_ids),
                "node_count": len(source.node_ids),
                "unique_undirected_edge_count": len(source.edges),
                "features": graph.features.tolist(),
                "original_edges": [list(edge) for edge in graph.edges],
                "plans": plans,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "pre_outcome_selection_and_masks_frozen",
        "selection_seed": SELECTION_SEED,
        "raw_file_sha256": raw_hashes,
        "dataset_metadata": metadata,
        "selection_algorithm": (
            "sort distinct eligible graphs by (content SHA-256, dataset index), "
            "permute with NumPy default_rng(20261111), skip only graphs whose "
            "approved validation-first train/validation split is unsupported, "
            "then take the first 20"
        ),
        "eligibility": {
            "bounds": "12-50 nodes, at most 200 unique edges, connected simple graph",
            "unique_eligible_graph_count": len(unique),
            "nonexcluded_candidate_count": len(eligible_candidates),
            "excluded_previously_used_graph_ids": list(EXCLUDED_GRAPH_IDS),
            "validation_split_rejections_before_selection": skipped,
            "selected_ids": list(selected_ids),
        },
        "features": (
            "constant all-ones node feature concatenated with one-hot local "
            "node identity; graph labels unused"
        ),
        "training": {
            "role": "known-graph recovery; each model trains on its intact source graph before perturbation",
            "device": "CPU",
            "encoder_epochs": 30,
            "denoiser_epochs": 50,
            "activation": "ReLU",
            "curvature": 1.0,
            "diffusion_schedule_steps": 200,
            "t_star": T_STAR,
            "partial_reconstructions_per_mask": RECONSTRUCTIONS,
            "model_seed": "3000000 + dataset_index",
            "validation_first_split_seed": "model_seed + 1",
            "checkpoint_root": "models/",
        },
        "perturbations": {
            "damage_fractions": list(DAMAGE_FRACTIONS),
            "rounding": "max(1, floor(fraction * unique undirected edge count + 0.5))",
            "targeted": "one intact-degree-product ranking mask per graph and level; endpoint order breaks ties",
            "uniform_random_masks_per_graph_level": RANDOM_MASKS_PER_LEVEL,
            "uniform_random_seed": (
                "20261110 + dataset_index*10000 + level_index*100 + mask_index; "
                "collision retry increments by 3"
            ),
            "reconstruction_seed": (
                "7000000 + 3000000 + zero_based_selection_rank*10000 + "
                "level_index*100; random mask_index adds mask_index*10"
            ),
            "paired_methods": "same damaged input, node order, candidates, and mask",
        },
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "metric_policy": {
            "labels": "used only after prediction for exact-edge recovery and false-positive scoring",
            "distribution_features_distance_and_ties": (
                "retain the approved degree/clustering/spectral histograms, "
                "equal-weight mean total variation, inclusive neighborhood ties; "
                "main k=2, sensitivity k=1 and k=3"
            ),
            "f1dc_correction": (
                "use standard PRDC coverage: a real point is covered when its "
                "nearest generated point lies within that real point's real-set "
                "kNN radius; previous five-source F1 results will be recomputed "
                "offline for consistent supplementary pooling"
            ),
        },
        "budgets_seconds": {
            "training_and_scoring_hard_cap": TRAINING_BUDGET_SECONDS,
            "offline_metrics_hard_cap": METRICS_BUDGET_SECONDS,
            "expected_total": "about 2 minutes measured-scale estimate; 15-minute combined cap",
        },
        "selected_graphs": selected_rows,
        "validation": {
            "selected_graph_count": len(selected_rows),
            "all_selected_graphs_pass_validation_first_split": True,
            "all_graphs_content_distinct": True,
            "excluded_graph_ids_absent": True,
            "model_outcomes_inspected": False,
        },
        "preservation": "all prior pilot and diagnostic artifacts remain unchanged",
    }
    manifest_path = OUTPUT_DIR / "selection_manifest.json"
    _save(manifest_path, manifest)
    digest = _sha256(manifest_path.read_bytes())
    (OUTPUT_DIR / "selection_manifest.sha256").write_text(
        digest + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sha256": digest,
                "selected_ids": list(selected_ids),
                "rejected_before_selection": skipped,
            },
            indent=2,
        )
    )
    return digest


def _load_and_validate_manifest():
    manifest_path = OUTPUT_DIR / "selection_manifest.json"
    hash_path = OUTPUT_DIR / "selection_manifest.sha256"
    if not manifest_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError("pre-outcome selection manifest is missing")
    digest = _sha256(manifest_path.read_bytes())
    if digest != hash_path.read_text(encoding="ascii").strip():
        raise RuntimeError("selection manifest hash mismatch")
    manifest = _read_json(manifest_path)
    if manifest["status"] != "pre_outcome_selection_and_masks_frozen":
        raise RuntimeError("selection manifest has unexpected status")
    if tuple(manifest["eligibility"]["selected_ids"]) != EXPECTED_SELECTED_IDS:
        raise RuntimeError("selection manifest IDs differ from preselected IDs")
    if set(manifest["eligibility"]["selected_ids"]) & set(EXCLUDED_GRAPH_IDS):
        raise RuntimeError("selection manifest includes a previously used graph")
    calibration_path = (
        PREVIOUS_RECOVERY_DIR / "calibration" / "summary.json"
    )
    calibration = _read_json(calibration_path)
    calibrated = {
        method: float(row["tau"])
        for method, row in calibration["method_thresholds"].items()
    }
    if calibrated != FROZEN_THRESHOLDS:
        raise RuntimeError(
            "frozen thresholds differ from the approved calibration summary"
        )
    return manifest, digest


def run_training_and_scoring() -> str:
    manifest, manifest_hash = _load_and_validate_manifest()
    if (OUTPUT_DIR / "evaluation" / "summary.json").exists():
        raise FileExistsError("refusing to overwrite the expanded evaluation")
    graphs, raw_hashes, _ = load_imdb_binary_graphs(DATA_ROOT, download=False)
    if raw_hashes != manifest["raw_file_sha256"]:
        raise RuntimeError("raw dataset hashes changed after pre-outcome selection")
    by_id = {graph.graph_id: graph for graph in graphs}
    started = time.perf_counter()
    deadline = started + TRAINING_BUDGET_SECONDS
    torch.set_num_threads(1)
    completed = []
    try:
        for row in manifest["selected_graphs"]:
            if time.perf_counter() >= deadline:
                raise TimeoutError("10-minute training/scoring budget reached")
            source = by_id.get(row["graph_id"])
            if source is None or source.content_sha256 != row["graph_sha256"]:
                raise RuntimeError(
                    f"source graph changed after selection: {row['graph_id']}"
                )
            graph = _as_synthetic_graph(source)
            if graph.features.tolist() != row["features"]:
                raise RuntimeError(
                    f"feature rows changed after selection: {row['graph_id']}"
                )
            plans = []
            for item in row["plans"]:
                plan = dict(item)
                for key in ("removed_edges", "damaged_edges"):
                    plan[key] = tuple(tuple(edge) for edge in plan[key])
                plans.append(plan)
            pilot_graph = PilotGraph(
                graph=graph,
                split="expanded_evaluation",
                source=source,
                graph_sha256=source.content_sha256,
            )
            result = _train_and_score(
                pilot_graph,
                plans,
                row["model_seed"],
                OUTPUT_DIR / "models",
                deadline,
            )
            if result["graph_id"] != row["graph_id"]:
                raise RuntimeError("training output graph ID mismatch")
            completed.append(result)
            _save(
                OUTPUT_DIR / "evaluation" / "graphs" / f"{row['graph_id']}.json",
                result,
            )
            _save(
                OUTPUT_DIR / "work_in_progress.json",
                {
                    "status": "running",
                    "manifest_sha256": manifest_hash,
                    "completed_graph_ids": [
                        item["graph_id"] for item in completed
                    ],
                    "elapsed_seconds": time.perf_counter() - started,
                    "remaining_budget_seconds": max(
                        0.0, deadline - time.perf_counter()
                    ),
                },
            )

        threshold_graphs = [_threshold_graph(item) for item in completed]
        rows = summarize_at_threshold(threshold_graphs, FROZEN_THRESHOLDS)
        score_summary = _source_bootstrap_scores(
            rows, seed=BOOTSTRAP_SEED, replicates=SCORE_BOOTSTRAP_REPLICATES
        )
        report = {
            "status": "evaluation_complete",
            "manifest_sha256": manifest_hash,
            "evaluation_role": "known-graph recovery",
            "frozen_thresholds": FROZEN_THRESHOLDS,
            "thresholds_adjusted_or_selected_on_new_graphs": False,
            "graph_count": len(completed),
            "attack_damage_group_count": len(rows),
            "partial_reconstruction_count": len(completed)
            * len(DAMAGE_FRACTIONS)
            * (1 + RANDOM_MASKS_PER_LEVEL)
            * RECONSTRUCTIONS,
            "no_diffusion_output_count": len(completed)
            * len(DAMAGE_FRACTIONS)
            * (1 + RANDOM_MASKS_PER_LEVEL),
            "per_graph_attack_damage_results": rows,
            "macro_mean_by_attack": summarize_attack_balanced(rows),
            "source_level_uncertainty": score_summary,
            "uncertainty_scope": (
                "paired bootstrap resamples of the 20 source graphs; masks and "
                "draws are nested within each source"
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _save(OUTPUT_DIR / "evaluation" / "summary.json", report)
        status = (
            "complete"
            if time.perf_counter() < deadline
            else "complete_at_budget_boundary"
        )
        _save(
            OUTPUT_DIR / "summary.json",
            {
                "status": status,
                "manifest_sha256": manifest_hash,
                "elapsed_seconds": time.perf_counter() - started,
                "training_and_scoring_budget_seconds": TRAINING_BUDGET_SECONDS,
                "prior_artifacts_preserved": True,
            },
        )
        _save(
            OUTPUT_DIR / "work_in_progress.json",
            {
                "status": "complete",
                "manifest_sha256": manifest_hash,
                "completed_graph_ids": [
                    item["graph_id"] for item in completed
                ],
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        return status
    except Exception as error:
        _save(
            OUTPUT_DIR / "work_in_progress.json",
            {
                "status": "failed_or_stopped",
                "failure_type": type(error).__name__,
                "failure": str(error),
                "manifest_sha256": manifest_hash,
                "completed_graph_ids": [
                    item["graph_id"] for item in completed
                ],
                "elapsed_seconds": time.perf_counter() - started,
                "preserved_partial_outputs": True,
                "prior_artifacts_preserved": True,
            },
        )
        raise


def _source_bootstrap_scores(rows, seed: int, replicates: int):
    graph_ids = sorted({row["graph_id"] for row in rows})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(graph_ids), size=(replicates, len(graph_ids)))
    metrics = (
        "book_stability_score",
        "absolute_false_positive_edges",
        "realized_fpr",
    )
    strata = sorted(
        {
            (row["attack_type"], row["damage_fraction_requested"])
            for row in rows
        }
    )
    output = {"replicates": replicates, "seed": seed, "by_stratum": []}
    by_graph = {
        (row["graph_id"], row["attack_type"], row["damage_fraction_requested"]): row
        for row in rows
    }
    for attack, damage in strata:
        group = {
            method: {
                graph_id: [
                    by_graph[(graph_id, attack, damage)]["method_metrics"][method][
                        metric
                    ]
                    for metric in metrics
                ]
                for graph_id in graph_ids
            }
            for method in METHODS
        }
        boot = {
            method: np.empty((replicates, len(metrics)), dtype=np.float64)
            for method in METHODS
        }
        for method in METHODS:
            values = np.asarray(
                [group[method][graph_id] for graph_id in graph_ids],
                dtype=np.float64,
            )
            boot[method] = values[draws].mean(axis=1)
        item = {
            "attack_type": attack,
            "damage_fraction_requested": damage,
            "metrics": {},
        }
        for metric_index, metric in enumerate(metrics):
            item["metrics"][metric] = {}
            for method in METHODS:
                values = np.asarray(
                    [group[method][graph_id][metric_index] for graph_id in graph_ids]
                )
                sample = boot[method][:, metric_index]
                item["metrics"][metric][method] = {
                    "mean": float(values.mean()),
                    "ci95": [
                        float(np.quantile(sample, 0.025)),
                        float(np.quantile(sample, 0.975)),
                    ],
                }
            delta = boot["partial_recovery"][:, metric_index] - boot[
                "no_diffusion_baseline"
            ][:, metric_index]
            item["metrics"][metric]["paired_difference_partial_minus_no_diffusion"] = {
                "mean": float(
                    np.mean(
                        [
                            group["partial_recovery"][gid][metric_index]
                            - group["no_diffusion_baseline"][gid][metric_index]
                            for gid in graph_ids
                        ]
                    )
                ),
                "ci95": [
                    float(np.quantile(delta, 0.025)),
                    float(np.quantile(delta, 0.975)),
                ],
            }
        output["by_stratum"].append(item)

    overall = {}
    for method in METHODS:
        values_by_graph = []
        for graph_id in graph_ids:
            per_metric = []
            for metric in metrics:
                cell_values = [
                    by_graph[(graph_id, attack, damage)]["method_metrics"][method][
                        metric
                    ]
                    for attack, damage in strata
                ]
                per_metric.append(float(np.mean(cell_values)))
            values_by_graph.append(per_metric)
        values = np.asarray(values_by_graph, dtype=np.float64)
        boot = values[draws].mean(axis=1)
        overall[method] = {}
        for metric_index, metric in enumerate(metrics):
            overall[method][metric] = {
                "mean": float(values[:, metric_index].mean()),
                "ci95": [
                    float(np.quantile(boot[:, metric_index], 0.025)),
                    float(np.quantile(boot[:, metric_index], 0.975)),
                ],
            }
    delta = (
        np.asarray(
            [
                [
                    np.mean(
                        [
                            by_graph[(gid, attack, damage)]["method_metrics"][
                                "partial_recovery"
                            ][metric]
                            - by_graph[(gid, attack, damage)]["method_metrics"][
                                "no_diffusion_baseline"
                            ][metric]
                            for attack, damage in strata
                        ]
                    )
                    for metric in metrics
                ]
                for gid in graph_ids
            ],
            dtype=np.float64,
        )[draws]
        .mean(axis=1)
    )
    overall["paired_difference_partial_minus_no_diffusion"] = {
        metric: {
            "mean": float(delta[:, metric_index].mean()),
            "ci95": [
                float(np.quantile(delta[:, metric_index], 0.025)),
                float(np.quantile(delta[:, metric_index], 0.975)),
            ],
        }
        for metric_index, metric in enumerate(metrics)
    }
    output["overall_source_balanced"] = overall
    return output


def _graphs_from_saved_outputs():
    result_paths = sorted(
        (OUTPUT_DIR / "evaluation" / "graphs").glob("IMDB-BINARY-*.json")
    )
    if len(result_paths) != 20:
        raise FileNotFoundError("all 20 expanded graph outputs are required")
    expanded = []
    for path in result_paths:
        result = _read_json(path)
        if result["graph_id"] not in EXPECTED_SELECTED_IDS:
            raise RuntimeError(f"unexpected graph in expanded outputs: {path.name}")
        source = {
            "graph_id": result["graph_id"],
            "node_ids": result["node_ids"],
            "original_edges": result["original_edges"],
            "cases": [],
        }
        for case in result["cases"]:
            if len(case["candidate_edges"]) != len(
                case["partial_probabilities"][0]
            ) or len(case["candidate_edges"]) != len(
                case["baseline_probabilities"]
            ):
                raise RuntimeError(f"{result['graph_id']}: probability shape mismatch")
            original = {frozenset(edge) for edge in result["original_edges"]}
            damaged = {frozenset(edge) for edge in case["damaged_edges"]}
            removed = {frozenset(edge) for edge in case["removed_edges"]}
            candidates = [tuple(edge) for edge in case["candidate_edges"]]
            candidate_keys = {frozenset(edge) for edge in candidates}
            expected_candidates = {
                frozenset(edge)
                for edge in _candidate_pairs(result["node_ids"])
                if frozenset(edge) not in damaged
            }
            if candidate_keys != expected_candidates:
                raise RuntimeError(f"{result['graph_id']}: candidate set mismatch")
            if damaged | removed != original or damaged & removed:
                raise RuntimeError(f"{result['graph_id']}: mask partition mismatch")
            nonedges = {
                frozenset(edge) for edge in candidates if frozenset(edge) not in original
            }
            case_out = dict(case)
            for method in METHODS:
                probability_rows = (
                    case["partial_probabilities"]
                    if method == "partial_recovery"
                    else [case["baseline_probabilities"]]
                )
                predicted_by_draw = []
                for probabilities in probability_rows:
                    predicted_by_draw.append(
                        [
                            edge
                            for edge, probability in zip(candidates, probabilities)
                            if probability >= FROZEN_THRESHOLDS[method]
                        ]
                    )
                case_out.setdefault("predicted_edge_sets", {})[method] = (
                    predicted_by_draw
                )
                case_out.setdefault("method_outputs", {})[method] = [
                    {
                        "draw_index": draw_index,
                        "decoded_edges": [list(edge) for edge in predicted],
                        "completed_edges": [
                            *case["damaged_edges"],
                            *[list(edge) for edge in predicted],
                        ],
                    }
                    for draw_index, predicted in enumerate(predicted_by_draw)
                ]
            for method in METHODS:
                for prediction in case_out["predicted_edge_sets"][method]:
                    predicted = {frozenset(edge) for edge in prediction}
                    if predicted - candidate_keys:
                        raise RuntimeError("prediction contains non-candidate pair")
                    if predicted & damaged:
                        raise RuntimeError("prediction includes a surviving edge")
                    if len(predicted) != len(prediction):
                        raise RuntimeError("duplicate predicted edge")
            if not removed or not nonedges:
                raise RuntimeError(f"{result['graph_id']}: missing score class")
            source["cases"].append(case_out)
        expanded.append(source)
    return expanded


def _old_five_from_saved_outputs():
    path = PREVIOUS_DISTRIBUTION_DIR / "validation_and_reconstructed_graphs.json"
    data = _read_json(path)
    if data["status"] != "all_saved_evaluation_metric_checks_passed":
        raise RuntimeError("previous five-graph outputs were not fully validated")
    graphs = [row for row in data["graphs"] if row["graph_id"] in OLD_EVALUATION_IDS]
    if {row["graph_id"] for row in graphs} != set(OLD_EVALUATION_IDS):
        raise RuntimeError("previous five evaluation graphs are incomplete")
    return graphs


def _prepare_feature_pools(sources, attack, damage):
    reference_features = {}
    pools = {method: {} for method in METHODS}
    duplicate_rows = []
    for source in sources:
        graph_id = source["graph_id"]
        reference_features[graph_id] = graph_feature_histograms(
            source["node_ids"], source["original_edges"]
        )
        cases = [
            case
            for case in source["cases"]
            if case["attack_type"] == attack
            and math.isclose(
                float(case["damage_fraction_requested"]),
                damage,
                abs_tol=1e-12,
            )
        ]
        cases.sort(key=lambda case: int(case["mask_index"]))
        if len(cases) != (1 if attack == "targeted_degree_product" else 3):
            raise RuntimeError(f"{graph_id}: unexpected mask count")
        for method in METHODS:
            outputs = []
            for case in cases:
                rows = case["method_outputs"][method]
                for draw in rows:
                    edges = draw["completed_edges"]
                    features = graph_feature_histograms(source["node_ids"], edges)
                    canonical_edges = sorted(
                        tuple(sorted(map(str, edge))) for edge in edges
                    )
                    exact_hash = hashlib.sha256(
                        json.dumps(
                            {
                                "nodes": sorted(map(str, source["node_ids"])),
                                "edges": canonical_edges,
                            },
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    feature_hash = hashlib.sha256(
                        b"".join(
                            np.asarray(block, dtype="<f8").tobytes()
                            for block in features
                        )
                    ).hexdigest()
                    outputs.append(
                        {
                            "mask_index": int(case["mask_index"]),
                            "draw_index": int(draw["draw_index"]),
                            "features": features,
                            "exact_hash": exact_hash,
                            "feature_hash": feature_hash,
                        }
                    )
            pools[method][graph_id] = outputs
            exact = [row["exact_hash"] for row in outputs]
            feature = [row["feature_hash"] for row in outputs]
            duplicate_rows.append(
                {
                    "graph_id": graph_id,
                    "method": method,
                    "attack_type": attack,
                    "damage_fraction_requested": damage,
                    "output_count": len(outputs),
                    "distinct_exact_graphs": len(set(exact)),
                    "exact_duplicate_outputs": len(exact) - len(set(exact)),
                    "distinct_feature_vectors": len(set(feature)),
                    "duplicate_feature_vectors": len(feature) - len(set(feature)),
                }
            )
    return reference_features, pools, duplicate_rows


def _f1_for_group(sources, attack, damage, group_name, rng, bootstrap_rng):
    source_ids = [source["graph_id"] for source in sources]
    references, pools, duplicates = _prepare_feature_pools(
        sources, attack, damage
    )
    all_results = {}
    distances_by_method = {}
    offsets_by_method = {}
    for method in METHODS:
        feature_list = [references[graph_id] for graph_id in source_ids]
        output_offsets = {}
        for graph_id in source_ids:
            output_offsets[graph_id] = {}
            for index, output in enumerate(pools[method][graph_id]):
                output_offsets[graph_id][
                    (output["mask_index"], output["draw_index"])
                ] = len(feature_list)
                feature_list.append(output["features"])
        distances_by_method[method] = feature_distance_matrix(feature_list)
        offsets_by_method[method] = output_offsets
    reference_indices = np.arange(len(source_ids), dtype=np.int64)

    point_choices = []
    selection_preview = []
    for replicate in range(F1_POINT_REPLICATES):
        choices = []
        for graph_id in source_ids:
            outputs = pools["partial_recovery"][graph_id]
            masks = sorted({row["mask_index"] for row in outputs})
            mask_index = masks[int(rng.integers(len(masks)))]
            draw_index = int(rng.integers(RECONSTRUCTIONS))
            choices.append((graph_id, mask_index, draw_index))
        point_choices.append(choices)
        if replicate < 5:
            selection_preview.append(
                [
                    {
                        "graph_id": graph_id,
                        "mask_index": mask_index,
                        "partial_draw_index": draw_index,
                    }
                    for graph_id, mask_index, draw_index in choices
                ]
            )
    bootstrap_choices = []
    for _ in range(F1_BOOTSTRAP_REPLICATES):
        choices = []
        for source_index in bootstrap_rng.integers(
            0, len(source_ids), size=len(source_ids)
        ):
            graph_id = source_ids[int(source_index)]
            outputs = pools["partial_recovery"][graph_id]
            masks = sorted({row["mask_index"] for row in outputs})
            mask_index = masks[int(bootstrap_rng.integers(len(masks)))]
            draw_index = int(bootstrap_rng.integers(RECONSTRUCTIONS))
            choices.append((graph_id, mask_index, draw_index))
        bootstrap_choices.append(choices)

    for method in METHODS:
        distances = distances_by_method[method]
        output_offsets = offsets_by_method[method]
        point_rows = {k: [] for k in (1, 2, 3)}
        saturated = {k: {"f1pr": 0, "f1dc": 0} for k in (1, 2, 3)}
        above_one = {k: 0 for k in (1, 2, 3)}
        for choices in point_choices:
            generated_indices = []
            for graph_id, mask_index, draw_index in choices:
                output_index = output_offsets[graph_id].get(
                    (mask_index, draw_index)
                )
                if output_index is None:
                    output_index = output_offsets[graph_id][(mask_index, 0)]
                generated_indices.append(output_index)
            metrics = prdc_from_distances(
                distances[np.ix_(reference_indices, reference_indices)],
                distances[np.ix_(generated_indices, generated_indices)],
                distances[np.ix_(reference_indices, generated_indices)],
                (1, 2, 3),
            )
            for k, values in metrics.items():
                point_rows[k].append(values)
                saturated[k]["f1pr"] += int(math.isclose(values["f1pr"], 1.0))
                saturated[k]["f1dc"] += int(math.isclose(values["f1dc"], 1.0))
                above_one[k] += int(values["f1dc"] > 1.0)

        bootstrap_rows = {k: [] for k in (1, 2, 3)}
        for choices in bootstrap_choices:
            sampled_sources = [
                source_ids.index(graph_id) for graph_id, _, _ in choices
            ]
            sampled_reference_indices = reference_indices[sampled_sources]
            generated_indices = []
            for graph_id, mask_index, draw_index in choices:
                output_index = output_offsets[graph_id].get(
                    (mask_index, draw_index)
                )
                if output_index is None:
                    output_index = output_offsets[graph_id][(mask_index, 0)]
                generated_indices.append(output_index)
            metrics = prdc_from_distances(
                distances[
                    np.ix_(sampled_reference_indices, sampled_reference_indices)
                ],
                distances[np.ix_(generated_indices, generated_indices)],
                distances[
                    np.ix_(sampled_reference_indices, generated_indices)
                ],
                (1, 2, 3),
            )
            for k, values in metrics.items():
                bootstrap_rows[k].append(values)

        method_results = {}
        for k in (1, 2, 3):
            method_results[str(k)] = {}
            for metric in (
                "precision",
                "recall",
                "f1pr",
                "density",
                "coverage",
                "f1dc",
            ):
                values = np.asarray(
                    [row[metric] for row in point_rows[k]], dtype=np.float64
                )
                boot = np.asarray(
                    [row[metric] for row in bootstrap_rows[k]],
                    dtype=np.float64,
                )
                method_results[str(k)][metric] = {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "conditional_output_q05": float(np.quantile(values, 0.05)),
                    "conditional_output_q95": float(np.quantile(values, 0.95)),
                    "source_bootstrap_ci95": [
                        float(np.quantile(boot, 0.025)),
                        float(np.quantile(boot, 0.975)),
                    ],
                    "exactly_one_saturation_fraction": (
                        float(saturated[k][metric] / F1_POINT_REPLICATES)
                        if metric in ("f1pr", "f1dc")
                        else None
                    ),
                    "f1dc_above_one_fraction": (
                        float(above_one[k] / F1_POINT_REPLICATES)
                        if metric == "f1dc"
                        else None
                    ),
                }
        all_results[method] = {
            "metrics": method_results,
            "duplicates": [row for row in duplicates if row["method"] == method],
        }

    return {
        "group": group_name,
        "attack_type": attack,
        "damage_fraction_requested": damage,
        "source_count": len(source_ids),
        "source_graph_ids": source_ids,
        "main_k": 2,
        "sensitivity_k": [1, 3],
        "point_replicates": F1_POINT_REPLICATES,
        "source_bootstrap_replicates": F1_BOOTSTRAP_REPLICATES,
        "methods": all_results,
        "selection_preview": selection_preview,
        "duplicate_diagnostics": duplicates,
    }


def compute_distribution_metrics() -> str:
    if not (OUTPUT_DIR / "evaluation" / "summary.json").is_file():
        raise FileNotFoundError("complete expanded evaluation is required")
    manifest, manifest_hash = _load_and_validate_manifest()
    report_path = OUTPUT_DIR / "distribution" / "f1_metrics.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    started = time.perf_counter()
    deadline = started + METRICS_BUDGET_SECONDS
    expanded = _graphs_from_saved_outputs()
    old_graphs = _old_five_from_saved_outputs()
    groups = {
        "new20_primary": expanded,
        "previous5_corrected_secondary": old_graphs,
        "pooled25_supplementary": old_graphs + expanded,
    }
    distribution_dir = OUTPUT_DIR / "distribution"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    score_summary_path = OUTPUT_DIR / "evaluation" / "summary.json"
    previous_validation_path = (
        PREVIOUS_DISTRIBUTION_DIR / "validation_and_reconstructed_graphs.json"
    )
    input_hashes = {
        "expanded_evaluation_summary_sha256": _sha256_file(score_summary_path),
        "previous_five_validation_sha256": _sha256_file(previous_validation_path),
        "selection_manifest_sha256": manifest_hash,
    }
    metrics = []
    for group_index, (group_name, sources) in enumerate(groups.items()):
        for attack_index, attack in enumerate(
            ("targeted_degree_product", "uniform_random")
        ):
            for damage_index, damage in enumerate(DAMAGE_FRACTIONS):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("5-minute offline-metrics budget reached")
                output_path = distribution_dir / (
                    f"{group_name}_{attack}_damage_{int(damage * 100):02d}.json"
                )
                if output_path.exists():
                    saved = _read_json(output_path)
                    if saved.get("input_hashes") != input_hashes:
                        raise RuntimeError(
                            f"stale metric checkpoint exists: {output_path}"
                        )
                    metrics.append(saved["result"])
                    continue
                stratum_index = (
                    group_index * 100 + attack_index * 10 + damage_index
                )
                result = _f1_for_group(
                        sources,
                        attack,
                        damage,
                        group_name,
                        np.random.default_rng(
                            F1_SAMPLE_SEED + stratum_index
                        ),
                        np.random.default_rng(
                            BOOTSTRAP_SEED + 100 + stratum_index
                        ),
                    )
                metrics.append(result)
                _save(
                    output_path,
                    {"input_hashes": input_hashes, "result": result},
                )
                _save(
                    distribution_dir / "work_in_progress.json",
                    {
                        "status": "running",
                        "completed_strata": len(metrics),
                        "total_strata": len(groups) * 10,
                        "elapsed_seconds": time.perf_counter() - started,
                        "input_hashes": input_hashes,
                    },
                )
    result = {
        "status": "complete",
        "book_f1_operationalization": (
            "The book names precision-recall F1pr and density-coverage F1dc "
            "but does not specify their operational definitions. This run uses "
            "the approved graph-histogram feature space and standard PRDC "
            "coverage, correcting the previous five-source implementation "
            "whose coverage duplicated recall."
        ),
        "features_and_distance": (
            "normalized degree, 100-bin clustering, and 200-bin normalized-"
            "Laplacian histograms; equal-weight mean of block total-variation "
            "distances"
        ),
        "tie_policy": "inclusive <= neighborhood-radius comparisons",
        "thresholds": FROZEN_THRESHOLDS,
        "groups": list(groups),
        "metrics": metrics,
        "input_hashes": input_hashes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _save(report_path, result)
    _save(
        OUTPUT_DIR / "distribution" / "manifest.json",
        {
            "status": "complete",
            "expanded_selection_manifest_sha256": manifest_hash,
            "input_hashes": input_hashes,
            "frozen_thresholds": FROZEN_THRESHOLDS,
            "main_k": 2,
            "sensitivity_k": [1, 3],
            "point_selection_seed": F1_SAMPLE_SEED,
            "point_replicates_per_stratum": F1_POINT_REPLICATES,
            "source_bootstrap_seed": BOOTSTRAP_SEED + 100,
            "source_bootstrap_replicates": F1_BOOTSTRAP_REPLICATES,
            "source_balance": (
                "one generated graph per source per replicate; each source "
                "equally weighted; same mask for paired methods"
            ),
            "duplicates": "retained in sampling and reported; no deduplication",
            "coverage_correction": (
                "coverage uses each real sample's own k-nearest-neighbor radius, "
                "not the generated-set radius used by recall"
            ),
            "metrics": "F1pr and F1dc are structural-distribution diagnostics and do not replace exact-edge recovery or MMD",
        },
    )
    _save(
        distribution_dir / "work_in_progress.json",
        {
            "status": "complete",
            "completed_strata": len(metrics),
            "total_strata": len(groups) * 10,
            "elapsed_seconds": time.perf_counter() - started,
            "input_hashes": input_hashes,
        },
    )
    return "complete"


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--run-training", action="store_true")
    group.add_argument("--run-metrics", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare_only:
        prepare_selection_manifest()
    elif arguments.run_training:
        print(run_training_and_scoring())
    else:
        print(compute_distribution_metrics())


if __name__ == "__main__":
    main()
