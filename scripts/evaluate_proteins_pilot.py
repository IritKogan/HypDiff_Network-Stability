from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stability.pilot_artifacts import load_pilot_artifact, save_pilot_artifact
from stability.pilot_config import PilotConfig
from stability.proteins_dataset import (
    EVALUATION_GRAPH_COUNT,
    eligible_graph_id_manifest,
    load_proteins_graphs,
    select_graph_splits,
)
from stability.proteins_pilot import macro_metrics, run_proteins_graph


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tu"
ARTIFACT_ROOT = ROOT / "artifacts" / "proteins_pilot"
CALIBRATION_ARTIFACT = (
    ARTIFACT_ROOT / "calibration_stable_clustering" / "calibration.json"
)
EVALUATION_DIR = ARTIFACT_ROOT / "evaluation_stable_clustering"
MODEL_SEED_OFFSET = 980000
PERTURBATION_SEED_OFFSET = 990000
RECONSTRUCTION_SEED_OFFSET = 1000000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not CALIBRATION_ARTIFACT.is_file():
        raise FileNotFoundError(
            "run the calibration command and inspect its frozen threshold first"
        )
    calibration_hash_before = _sha256(CALIBRATION_ARTIFACT)
    calibration = load_pilot_artifact(CALIBRATION_ARTIFACT)
    if calibration.get("status") != "calibration_complete":
        raise ValueError("evaluation requires completed calibration")
    selection = calibration["selection"]
    threshold = calibration["frozen_threshold_selection"]
    tau = float(threshold["tau"])
    if not calibration["method"].get("threshold_frozen_before_evaluation"):
        raise ValueError("calibration threshold is not marked frozen")
    if threshold["calibration_mean_false_positive_rate"] > threshold[
        "false_positive_rate_cap"
    ]:
        raise ValueError("frozen threshold does not satisfy the calibration FPR cap")

    started = time.perf_counter()
    graphs, raw_hashes = load_proteins_graphs(DATA_ROOT, download=False)
    split = select_graph_splits(graphs)
    manifest = eligible_graph_id_manifest(split)
    if manifest != selection:
        raise RuntimeError("PROTEINS split does not match the frozen calibration manifest")
    if raw_hashes != calibration["dataset"]["raw_file_sha256"]:
        raise RuntimeError("PROTEINS raw data checksum changed since calibration")
    if len(split.evaluation) != EVALUATION_GRAPH_COUNT:
        raise RuntimeError("evaluation split has an unexpected graph count")

    evaluation_results = []
    config = PilotConfig(curvature=1.0, tau=tau)
    for index, graph in enumerate(split.evaluation):
        print(
            f"Training PROTEINS evaluation graph {index + 1}/"
            f"{len(split.evaluation)}: {graph.graph_id}"
        )
        evaluation_results.append(
            run_proteins_graph(
                graph=graph,
                config=config,
                model_seed=graph.dataset_index + MODEL_SEED_OFFSET,
                perturbation_seed=graph.dataset_index + PERTURBATION_SEED_OFFSET,
                reconstruction_seed=graph.dataset_index + RECONSTRUCTION_SEED_OFFSET,
                checkpoint_dir=EVALUATION_DIR / "models",
            )
        )

    if _sha256(CALIBRATION_ARTIFACT) != calibration_hash_before:
        raise RuntimeError("frozen calibration artifact changed during evaluation")
    payload = {
        "work_package": "bounded PROTEINS CPU pilot",
        "artifact_type": "evaluation at frozen calibration threshold",
        "status": "evaluation_complete",
        "study_scope": "recovery of known graphs; per-graph models train on intact graph",
        "dataset": calibration["dataset"],
        "selection": {
            **manifest,
            "calibration_graph_hashes": [
                row["content_sha256"] for row in selection["calibration"]
            ],
        },
        "calibration_artifact": str(CALIBRATION_ARTIFACT.relative_to(ROOT)),
        "calibration_artifact_sha256": calibration_hash_before,
        "frozen_tau": tau,
        "threshold_changed_during_evaluation": False,
        "method": {
            "curvature": 1.0,
            "schedule_length": 200,
            "timestep_conditioning": "normalized t/T",
            "autoencoder_epochs_per_graph": 30,
            "denoiser_epochs_per_graph": 50,
            "total_autoencoder_epochs": 30 * len(evaluation_results),
            "total_denoiser_epochs": 50 * len(evaluation_results),
            "removed_edges_per_graph": 3,
            "reconstructions_per_graph": 8,
            "t_star": 7,
            "models_trained_separately_per_graph": True,
            "partial_recovery_and_baseline_share_graph_model_damaged_input_and_candidates": True,
            "evaluation_metrics_not_used_for_selection": True,
        },
        "macro_mean_metrics": {
            "partial_recovery": macro_metrics(
                evaluation_results, "partial_recovery"
            ),
            "no_diffusion_baseline": macro_metrics(
                evaluation_results, "no_diffusion_baseline"
            ),
        },
        "evaluation_duration_seconds": time.perf_counter() - started,
        "evaluation_graphs": evaluation_results,
    }
    destination = EVALUATION_DIR / "evaluation.json"
    save_pilot_artifact(destination, payload)
    print(
        json.dumps(
            {
                "artifact": str(destination),
                "frozen_tau": tau,
                "partial_recovery": payload["macro_mean_metrics"][
                    "partial_recovery"
                ],
                "no_diffusion_baseline": payload["macro_mean_metrics"][
                    "no_diffusion_baseline"
                ],
                "per_graph": [
                    {
                        "graph_id": graph["graph_id"],
                        "partial_recovery": graph["partial_recovery"],
                        "no_diffusion_baseline": graph["no_diffusion_baseline"],
                    }
                    for graph in evaluation_results
                ],
                "duration_seconds": payload["evaluation_duration_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
