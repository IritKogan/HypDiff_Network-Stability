from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stability.pilot_artifacts import save_pilot_artifact
from stability.proteins_common_fpr_replay import (
    regenerate_graph_scores,
    select_method_threshold,
    summarize_attack_balanced,
    summarize_at_threshold,
)
from stability.proteins_dataset import (
    load_proteins_graphs,
)
from stability.proteins_fresh_attack_experiment import selection_manifest
from stability.proteins_fresh_attack_experiment import select_fresh_splits


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tu"
RUN_ROOT = (
    ROOT
    / "artifacts"
    / "proteins_pilot"
    / "fresh_disjoint_random_targeted_20261105"
)
OUTPUT_DIR = (
    ROOT / "artifacts" / "proteins_pilot" / "common_fpr_replay_20261105"
)
SELECTION_ARTIFACT = RUN_ROOT / "selection.json"
SHARED_CALIBRATION = RUN_ROOT / "calibration.json"
SHARED_EVALUATION = RUN_ROOT / "evaluation.json"
CALIBRATION_OUTPUT = OUTPUT_DIR / "calibration_thresholds.json"
EVALUATION_OUTPUT = OUTPUT_DIR / "evaluation.json"
OFFICIAL_ARTIFACTS = {
    "official_calibration": ROOT
    / "artifacts"
    / "proteins_pilot"
    / "calibration_stable_clustering"
    / "calibration.json",
    "official_evaluation": ROOT
    / "artifacts"
    / "proteins_pilot"
    / "evaluation_stable_clustering"
    / "evaluation.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_same_metrics(replay_rows, saved_graphs, tau):
    saved = {
        (row["graph_id"], row["attack_type"], row["damage_fraction_requested"]): row
        for graph in saved_graphs
        for row in graph["attack_levels"]
    }
    checked = 0
    for row in replay_rows:
        old = saved[
            (
                row["graph_id"],
                row["attack_type"],
                row["damage_fraction_requested"],
            )
        ]
        for method in ("partial_recovery", "no_diffusion_baseline"):
            new_metrics = row["method_metrics"][method]
            old_metrics = old["metrics"][method]
            comparisons = (
                ("book_stability_score", "book_stability_score"),
                ("realized_fpr", "mean_false_positive_rate"),
                (
                    "absolute_false_positive_edges",
                    "mean_false_positive_edges",
                ),
            )
            for new_key, old_key in comparisons:
                if not np.isclose(
                    new_metrics[new_key],
                    old_metrics[old_key],
                    rtol=0.0,
                    atol=1e-8,
                ):
                    raise RuntimeError(
                        f"shared-threshold replay mismatch at {row['graph_id']} "
                        f"{row['attack_type']} {row['damage_fraction_requested']} "
                        f"{method} {new_key}: {new_metrics[new_key]} != "
                        f"{old_metrics[old_key]}"
                    )
                checked += 1
    return {"shared_tau": tau, "cell_metrics_checked": checked}


def _load_inputs():
    selection = json.loads(SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    calibration_shared = json.loads(SHARED_CALIBRATION.read_text(encoding="utf-8"))
    evaluation_shared = json.loads(SHARED_EVALUATION.read_text(encoding="utf-8"))
    graphs, raw_hashes = load_proteins_graphs(DATA_ROOT, download=False)
    calibration, evaluation, eligible_count, distinct_count = select_fresh_splits(
        graphs,
        seed=int(selection["selection"]["selection_seed"]),
    )
    manifest = selection_manifest(
        calibration,
        evaluation,
        eligible_count,
        distinct_count,
    )
    if manifest != selection["selection"]:
        raise RuntimeError("fresh graph manifest differs from the saved selection")
    if raw_hashes != selection["raw_file_sha256"]:
        raise RuntimeError("cached PROTEINS raw data hashes differ from selection")
    if calibration_shared["selection_artifact_sha256"] != _sha256(SELECTION_ARTIFACT):
        raise RuntimeError("shared calibration does not match the saved selection")
    if evaluation_shared["selection_artifact_sha256"] != _sha256(SELECTION_ARTIFACT):
        raise RuntimeError("shared evaluation does not match the saved selection")
    return selection, calibration_shared, evaluation_shared, calibration, evaluation, raw_hashes


def _replay_split(split_graphs, split_name, selection, run_started):
    split_key = f"{split_name}_attack_masks"
    split_code = 1 if split_name == "calibration" else 2
    checkpoint_dir = RUN_ROOT / "models" / split_name
    graph_results = []
    for graph in split_graphs:
        model_seed = int(selection["seed_plan"]["model_seeds"][graph.graph_id])
        planned = selection["seed_plan"][split_key][graph.graph_id]
        graph_results.append(
            regenerate_graph_scores(
                graph=graph,
                planned_cases=planned,
                checkpoint_dir=checkpoint_dir,
                model_seed=model_seed,
            )
        )
    return graph_results


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite supplementary results: {OUTPUT_DIR}")
    official_hashes_before = {
        name: _sha256(path) for name, path in OFFICIAL_ARTIFACTS.items()
    }
    source_hashes = {
        "selection": _sha256(SELECTION_ARTIFACT),
        "shared_calibration": _sha256(SHARED_CALIBRATION),
        "shared_evaluation": _sha256(SHARED_EVALUATION),
    }
    started = time.perf_counter()
    (
        selection,
        calibration_shared,
        evaluation_shared,
        calibration_graphs,
        evaluation_graphs,
        raw_hashes,
    ) = _load_inputs()
    torch.set_num_threads(1)
    calibration_scores = _replay_split(
        calibration_graphs, "calibration", selection, started
    )
    partial_selection = select_method_threshold(
        calibration_scores, "partial_recovery"
    )
    baseline_selection = select_method_threshold(
        calibration_scores, "no_diffusion_baseline"
    )
    thresholds = {
        "partial_recovery": float(partial_selection["tau"]),
        "no_diffusion_baseline": float(baseline_selection["tau"]),
    }
    calibration_rows = summarize_at_threshold(calibration_scores, thresholds)
    calibration_payload = {
        "artifact_type": "method-specific common-FPR calibration thresholds",
        "status": "thresholds_frozen_before_evaluation",
        "source_artifacts": {
            "selection": str(SELECTION_ARTIFACT.relative_to(ROOT)),
            "selection_sha256": source_hashes["selection"],
            "shared_calibration": str(SHARED_CALIBRATION.relative_to(ROOT)),
            "shared_calibration_sha256": source_hashes["shared_calibration"],
            "official_tau_1_results_unchanged": True,
        },
        "dataset_raw_file_sha256": raw_hashes,
        "calibration_graph_ids": [graph.graph_id for graph in calibration_graphs],
        "evaluation_graph_ids_reserved": [
            graph.graph_id for graph in evaluation_graphs
        ],
        "calibration_rule": {
            "false_positive_rate_cap": 0.05,
            "aggregation": (
                "average reconstructions within each mask, masks within each "
                "graph x attack x damage-level cell, graph/damage cells within "
                "attack, then equal weight for random and targeted attacks"
            ),
            "threshold_objective": (
                "maximize each method's own book stability score subject to "
                "attack-balanced mean FPR <= 0.05"
            ),
            "score_ties": (
                "all candidates at an equal probability are included together "
                "using probability >= tau; no candidate-order tie break"
            ),
            "threshold_ties": (
                "among equal book scores, prefer lower FPR then higher tau"
            ),
            "thresholds": {
                "partial_recovery": partial_selection,
                "no_diffusion_baseline": baseline_selection,
            },
        },
        "calibration_attack_balanced_metrics_at_selected_thresholds": summarize_attack_balanced(
            calibration_rows
        ),
        "calibration_graph_attack_damage_breakdown": calibration_rows,
        "calibration_runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    save_pilot_artifact(CALIBRATION_OUTPUT, calibration_payload)

    frozen_calibration_hash = _sha256(CALIBRATION_OUTPUT)
    shared_tau = float(evaluation_shared["frozen_tau"])
    shared_rows = summarize_at_threshold(
        calibration_scores,
        {
            "partial_recovery": shared_tau,
            "no_diffusion_baseline": shared_tau,
        },
    )
    replay_validation = _assert_same_metrics(
        shared_rows, calibration_shared["calibration_graphs"], shared_tau
    )

    evaluation_started = time.perf_counter()
    evaluation_scores = _replay_split(
        evaluation_graphs, "evaluation", selection, evaluation_started
    )
    evaluation_rows = summarize_at_threshold(evaluation_scores, thresholds)
    if _sha256(CALIBRATION_OUTPUT) != frozen_calibration_hash:
        raise RuntimeError("frozen thresholds changed before or during evaluation")
    shared_evaluation_rows = summarize_at_threshold(
        evaluation_scores,
        {
            "partial_recovery": shared_tau,
            "no_diffusion_baseline": shared_tau,
        },
    )
    evaluation_replay_validation = _assert_same_metrics(
        shared_evaluation_rows, evaluation_shared["evaluation_graphs"], shared_tau
    )
    if {name: _sha256(path) for name, path in OFFICIAL_ARTIFACTS.items()} != official_hashes_before:
        raise RuntimeError("an official tau=1 artifact changed during replay")
    if {
        "selection": _sha256(SELECTION_ARTIFACT),
        "shared_calibration": _sha256(SHARED_CALIBRATION),
        "shared_evaluation": _sha256(SHARED_EVALUATION),
    } != source_hashes:
        raise RuntimeError("an existing shared-threshold artifact changed")
    evaluation_payload = {
        "artifact_type": "evaluation at method-specific calibration thresholds",
        "status": "evaluation_complete",
        "calibration_threshold_artifact": str(
            CALIBRATION_OUTPUT.relative_to(ROOT)
        ),
        "calibration_threshold_artifact_sha256": frozen_calibration_hash,
        "source_artifacts": {
            "shared_calibration_sha256": source_hashes["shared_calibration"],
            "shared_evaluation_sha256": source_hashes["shared_evaluation"],
            "official_tau_1_results_unchanged": True,
        },
        "frozen_thresholds": thresholds,
        "thresholds_changed_during_evaluation": False,
        "evaluation_graph_ids": [graph.graph_id for graph in evaluation_graphs],
        "evaluation_attack_balanced_metrics": summarize_attack_balanced(
            evaluation_rows
        ),
        "evaluation_graph_attack_damage_breakdown": evaluation_rows,
        "shared_threshold_replay_validation": {
            "calibration": replay_validation,
            "evaluation": evaluation_replay_validation,
        },
        "evaluation_runtime_seconds": time.perf_counter() - evaluation_started,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    save_pilot_artifact(EVALUATION_OUTPUT, evaluation_payload)
    print(
        json.dumps(
            {
                "calibration_thresholds": thresholds,
                "calibration_metrics": calibration_payload[
                    "calibration_attack_balanced_metrics_at_selected_thresholds"
                ],
                "evaluation_metrics": evaluation_payload[
                    "evaluation_attack_balanced_metrics"
                ],
                "evaluation_runtime_seconds": evaluation_payload[
                    "evaluation_runtime_seconds"
                ],
                "total_runtime_seconds": evaluation_payload[
                    "total_runtime_seconds"
                ],
                "output": str(EVALUATION_OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
