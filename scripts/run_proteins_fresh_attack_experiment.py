from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stability.pilot_artifacts import save_pilot_artifact
from stability.proteins_dataset import load_proteins_graphs
from stability.proteins_fresh_attack_experiment import (
    DAMAGE_FRACTIONS,
    FALSE_POSITIVE_RATE_CAP,
    RANDOM_MASKS_PER_LEVEL,
    RECONSTRUCTIONS_PER_MASK,
    SELECTION_SEED,
    T_STAR,
    _make_attack_cases,
    macro_attack_balanced,
    run_graph_cases,
    score_case,
    select_calibration_threshold,
    select_fresh_splits,
    selection_manifest,
    summarize_split,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tu"
RUN_DIR = (
    ROOT
    / "artifacts"
    / "proteins_pilot"
    / "fresh_disjoint_random_targeted_20261105"
)
SELECTION_FILE = RUN_DIR / "selection.json"
CALIBRATION_FILE = RUN_DIR / "calibration.json"
EVALUATION_FILE = RUN_DIR / "evaluation.json"
OFFICIAL_ARTIFACTS = {
    "calibration": ROOT
    / "artifacts"
    / "proteins_pilot"
    / "calibration_stable_clustering"
    / "calibration.json",
    "evaluation": ROOT
    / "artifacts"
    / "proteins_pilot"
    / "evaluation_stable_clustering"
    / "evaluation.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _planned_case_manifest(cases):
    return [
        {
            key: value
            for key, value in case.items()
            if key not in {"candidate_edges"}
        }
        for case in cases
    ]


def _graph_summary(graph, rows, tau):
    return {
        "graph_id": graph["graph_id"],
        "dataset_index": graph["dataset_index"],
        "content_sha256": graph["content_sha256"],
        "node_count": len(graph["node_ids"]),
        "edge_count": len(graph["original_edges"]),
        "training": graph["training"],
        "attack_levels": [row for row in rows if row["graph_id"] == graph["graph_id"]],
        "scoring_threshold": tau,
    }


def _diagnostic_threshold(selection):
    if selection["tau"] is not None:
        return float(selection["tau"])
    first_nonzero = selection["first_nonzero_recovery_threshold"]
    if first_nonzero is not None:
        return float(first_nonzero["tau"])
    best_under_cap = selection["best_threshold_under_fpr_cap"]
    return float(best_under_cap["tau"]) if best_under_cap else 0.0


def _score_rows(graph_results, tau):
    rows = summarize_split(graph_results, tau)
    return {
        "rows": rows,
        "attack_balanced_macro_metrics": macro_attack_balanced(rows),
    }


def _format_graph_results(graph_results, rows, tau):
    return [
        _graph_summary(graph, rows, tau)
        for graph in graph_results
    ]


def _method_definition():
    return {
        "scope": "recovery of known graphs; one model pair is trained separately on each intact graph",
        "attack_types": {
            "targeted_degree_product": (
                "remove an intact-graph edge prefix ranked by descending "
                "degree(u) * degree(v), ties by ascending canonical endpoint "
                "pair in fixed node order"
            ),
            "uniform_random": (
                "eight distinct edge subsets sampled uniformly without "
                "replacement from the intact graph at each graph and damage level"
            ),
        },
        "damage_fractions": list(DAMAGE_FRACTIONS),
        "removal_count_rounding": "floor(p * original_edge_count + 0.5), nearest integer, half up",
        "training": {
            "autoencoder_epochs_per_graph": 30,
            "denoiser_epochs_per_graph": 50,
            "curvature": 1.0,
            "t_star": T_STAR,
            "diffusion_schedule_length": 200,
            "reconstructions_per_mask": RECONSTRUCTIONS_PER_MASK,
            "per_graph_training": True,
            "training_target": "each intact graph",
        },
        "paired_comparison": (
            "partial recovery and no-diffusion decoding use the same trained "
            "graph model, damaged adjacency, candidate pairs, and threshold"
        ),
        "metrics": {
            "book_stability_score": (
                "mean removed-edge recovery frequency across reconstructions"
            ),
            "added_edge_precision": (
                "recovered removed edges divided by all predicted additions "
                "(recovered edges plus false positives)"
            ),
            "false_positive_rate": (
                "false positives divided by all true nonedges in the intact graph"
            ),
            "surviving_edge_retention": (
                "surviving original edges present in the decoded graph divided "
                "by surviving original edges"
            ),
            "partial_recovery_diversity": (
                "number of unique decoded edge sets among eight samples and "
                "mean pairwise edge-set Jaccard distance; no-diffusion is a "
                "single deterministic output"
            ),
        },
        "calibration_aggregation": (
            "average reconstructions within each mask, masks within each graph "
            "x attack x damage-level cell, then graph/damage cells within each "
            "attack; give targeted and random attacks equal weight"
        ),
        "threshold_rule": (
            "choose one threshold from observed decoder probabilities in the "
            "attainable range that maximizes the macro partial-recovery book "
            "score, requires nonzero recovery, and keeps each method's "
            "attack-balanced macro mean FPR at or below 0.05; ties prefer "
            "lower mean FPR across both methods, then higher threshold"
        ),
        "evaluation": (
            "run only after calibration passes; freeze selected tau before "
            "evaluation and do not use evaluation results to alter it"
        ),
        "calibration_fpr_cap": FALSE_POSITIVE_RATE_CAP,
        "official_tau_1_results": "separate and unchanged",
    }


def main():
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite fresh run: {RUN_DIR}")
    official_hashes_before = {
        name: _sha256(path) for name, path in OFFICIAL_ARTIFACTS.items()
    }
    started = time.perf_counter()
    graphs, raw_hashes = load_proteins_graphs(DATA_ROOT, download=False)
    calibration, evaluation, eligible_count, distinct_count = select_fresh_splits(
        graphs
    )
    manifest = selection_manifest(
        calibration, evaluation, eligible_count, distinct_count
    )

    calibration_plans = {
        graph.graph_id: _make_attack_cases(graph, index, 1)
        for index, graph in enumerate(calibration)
    }
    evaluation_plans = {
        graph.graph_id: _make_attack_cases(graph, index, 2)
        for index, graph in enumerate(evaluation)
    }
    seed_plan = {
        "model_seeds": {
            graph.graph_id: 3_000_000 + 100_000 + graph.dataset_index
            for graph in calibration
        }
        | {
            graph.graph_id: 3_000_000 + 200_000 + graph.dataset_index
            for graph in evaluation
        },
        "calibration_attack_masks": {
            graph_id: _planned_case_manifest(cases)
            for graph_id, cases in calibration_plans.items()
        },
        "evaluation_attack_masks": {
            graph_id: _planned_case_manifest(cases)
            for graph_id, cases in evaluation_plans.items()
        },
    }
    selection_payload = {
        "artifact_type": "fresh disjoint PROTEINS random-targeted selection and seed plan",
        "status": "selected_before_training_or_outcome_inspection",
        "dataset": "PROTEINS",
        "loader": "torch_geometric.datasets.TUDataset(use_node_attr=True)",
        "raw_file_sha256": raw_hashes,
        "selection": manifest,
        "method": _method_definition(),
        "seed_plan": seed_plan,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    save_pilot_artifact(SELECTION_FILE, selection_payload)
    torch.set_num_threads(1)

    calibration_dir = RUN_DIR / "models" / "calibration"
    calibration_results = []
    for index, graph in enumerate(calibration):
        print(
            f"Training/calibrating graph {index + 1}/{len(calibration)}: "
            f"{graph.graph_id}"
        )
        calibration_results.append(
            run_graph_cases(
                graph=graph,
                graph_index=index,
                split_code=1,
                checkpoint_dir=calibration_dir,
                planned_cases=calibration_plans[graph.graph_id],
            )
        )
    threshold_selection = select_calibration_threshold(calibration_results)
    diagnostic_tau = _diagnostic_threshold(threshold_selection)
    calibration_scoring = _score_rows(calibration_results, diagnostic_tau)
    calibration_payload = {
        "artifact_type": "fresh disjoint PROTEINS calibration",
        "status": (
            "calibration_complete"
            if threshold_selection["tau"] is not None
            else "calibration_gate_failed"
        ),
        "selection_artifact": str(SELECTION_FILE.relative_to(ROOT)),
        "selection_artifact_sha256": _sha256(SELECTION_FILE),
        "dataset_raw_file_sha256": raw_hashes,
        "method": _method_definition(),
        "threshold_selection": threshold_selection,
        "diagnostic_tau_for_tradeoff_only": diagnostic_tau,
        "calibration_attack_balanced_metrics_at_diagnostic_tau": calibration_scoring[
            "attack_balanced_macro_metrics"
        ],
        "calibration_graphs": _format_graph_results(
            calibration_results, calibration_scoring["rows"], diagnostic_tau
        ),
        "calibration_duration_seconds": time.perf_counter() - started,
    }
    save_pilot_artifact(CALIBRATION_FILE, calibration_payload)
    print(
        json.dumps(
            {
                "calibration_artifact": str(CALIBRATION_FILE),
                "status": calibration_payload["status"],
                "threshold_selection": threshold_selection,
                "diagnostic_tau": diagnostic_tau,
                "calibration_metrics": calibration_payload[
                    "calibration_attack_balanced_metrics_at_diagnostic_tau"
                ],
            },
            indent=2,
        )
    )
    if threshold_selection["tau"] is None:
        if {name: _sha256(path) for name, path in OFFICIAL_ARTIFACTS.items()} != official_hashes_before:
            raise RuntimeError("an official artifact changed during calibration")
        return

    frozen_tau = float(threshold_selection["tau"])
    calibration_hash = _sha256(CALIBRATION_FILE)
    evaluation_results = []
    evaluation_dir = RUN_DIR / "models" / "evaluation"
    evaluation_started = time.perf_counter()
    for index, graph in enumerate(evaluation):
        print(
            f"Training/evaluating graph {index + 1}/{len(evaluation)}: "
            f"{graph.graph_id}"
        )
        evaluation_results.append(
            run_graph_cases(
                graph=graph,
                graph_index=index,
                split_code=2,
                checkpoint_dir=evaluation_dir,
                planned_cases=evaluation_plans[graph.graph_id],
            )
        )
    evaluation_scoring = _score_rows(evaluation_results, frozen_tau)
    evaluation_payload = {
        "artifact_type": "fresh disjoint PROTEINS evaluation at frozen threshold",
        "status": "evaluation_complete",
        "selection_artifact": str(SELECTION_FILE.relative_to(ROOT)),
        "selection_artifact_sha256": _sha256(SELECTION_FILE),
        "calibration_artifact": str(CALIBRATION_FILE.relative_to(ROOT)),
        "calibration_artifact_sha256": calibration_hash,
        "dataset_raw_file_sha256": raw_hashes,
        "frozen_tau": frozen_tau,
        "threshold_changed_during_evaluation": False,
        "method": _method_definition(),
        "evaluation_attack_balanced_metrics": evaluation_scoring[
            "attack_balanced_macro_metrics"
        ],
        "evaluation_graphs": _format_graph_results(
            evaluation_results, evaluation_scoring["rows"], frozen_tau
        ),
        "evaluation_duration_seconds": time.perf_counter() - evaluation_started,
    }
    if _sha256(CALIBRATION_FILE) != calibration_hash:
        raise RuntimeError("frozen calibration artifact changed during evaluation")
    if {name: _sha256(path) for name, path in OFFICIAL_ARTIFACTS.items()} != official_hashes_before:
        raise RuntimeError("an official artifact changed during evaluation")
    if EVALUATION_FILE.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {EVALUATION_FILE}")
    save_pilot_artifact(EVALUATION_FILE, evaluation_payload)
    print(
        json.dumps(
            {
                "evaluation_artifact": str(EVALUATION_FILE),
                "frozen_tau": frozen_tau,
                "evaluation_metrics": evaluation_payload[
                    "evaluation_attack_balanced_metrics"
                ],
                "duration_seconds": evaluation_payload[
                    "evaluation_duration_seconds"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
