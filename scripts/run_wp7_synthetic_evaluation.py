from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stability.pilot_artifacts import load_pilot_artifact, save_pilot_artifact
from stability.pilot_config import PilotConfig
from stability.synthetic_instances import (
    EVALUATION_SEEDS,
    generate_synthetic_graph,
    validate_split_seeds,
)
from stability.wp7_pipeline import run_graph


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ARTIFACT = (
    ROOT / "artifacts" / "wp7_synthetic_calibration" / "calibration.json"
)
ARTIFACT_DIR = ROOT / "artifacts" / "wp7_synthetic_evaluation"
MODEL_SEED_OFFSET = 83000
PERTURBATION_SEED_OFFSET = 84000
RECONSTRUCTION_SEED_OFFSET = 85000


def main():
    validate_split_seeds()
    calibration = load_pilot_artifact(CALIBRATION_ARTIFACT)
    if calibration.get("status") != "calibration_complete":
        raise ValueError("evaluation requires a completed calibration artifact")
    if not calibration.get("method", {}).get("threshold_frozen_before_evaluation"):
        raise ValueError("calibration artifact does not declare a frozen threshold")
    tau = float(calibration["frozen_threshold_selection"]["tau"])
    if not 0.0 <= tau <= 1.0:
        raise ValueError("frozen calibration threshold is outside [0, 1]")
    if calibration["method"].get("curvature") != 1.0:
        raise ValueError("calibration curvature does not match the approved c=1.0")
    config = PilotConfig(curvature=1.0, tau=tau)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for index, seed in enumerate(EVALUATION_SEEDS):
        graph = generate_synthetic_graph(seed, "evaluation")
        print(f"Training evaluation graph {index + 1}/{len(EVALUATION_SEEDS)}: {graph.graph_id}")
        results.append(
            run_graph(
                graph=graph,
                config=config,
                model_seed=seed + MODEL_SEED_OFFSET,
                perturbation_seed=seed + PERTURBATION_SEED_OFFSET,
                reconstruction_seed=seed + RECONSTRUCTION_SEED_OFFSET,
                checkpoint_dir=ARTIFACT_DIR / "models",
            )
        )

    def macro(path):
        return {
            metric: float(np.mean([graph[path][metric] for graph in results]))
            for metric in (
                "book_stability_score",
                "mean_removed_edges_recovered",
                "mean_false_positive_edges",
                "mean_predicted_edge_count",
                "mean_surviving_edge_retention",
                "distinct_thresholded_reconstructions",
            )
        }

    payload = {
        "work_package": "WP7",
        "artifact_type": "synthetic evaluation using frozen calibration threshold",
        "status": "evaluation_complete",
        "calibration_artifact": str(CALIBRATION_ARTIFACT.relative_to(ROOT)),
        "frozen_tau": tau,
        "threshold_changed_during_evaluation": False,
        "split": {
            "graph_count": len(results),
            "graph_seeds": list(EVALUATION_SEEDS),
            "node_count_per_graph": 12,
            "calibration_seeds": [71001, 71002, 71003],
            "seeds_disjoint": True,
            "graph_instances_distinct": True,
        },
        "method": {
            "curvature": 1.0,
            "schedule_length": 200,
            "timestep_conditioning": "normalized t/T",
            "autoencoder_epochs_per_graph": 30,
            "denoiser_epochs_per_graph": 50,
            "total_autoencoder_epochs": 30 * len(results),
            "total_denoiser_epochs": 50 * len(results),
            "removed_edges_per_graph": 3,
            "reconstructions_per_graph": 8,
            "t_star": 7,
            "models_trained_separately_per_graph": True,
            "comparison_uses_same_graph_model_damaged_input_and_candidates": True,
        },
        "macro_mean_metrics": {
            "partial_recovery": macro("partial_recovery"),
            "no_diffusion_baseline": macro("no_diffusion_baseline"),
        },
        "graphs": results,
    }
    destination = ARTIFACT_DIR / "evaluation.json"
    save_pilot_artifact(destination, payload)
    print(json.dumps({
        "evaluation_artifact": str(destination),
        "frozen_tau": tau,
        "partial_recovery": payload["macro_mean_metrics"]["partial_recovery"],
        "no_diffusion_baseline": payload["macro_mean_metrics"][
            "no_diffusion_baseline"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
