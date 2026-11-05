from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stability.pilot_artifacts import save_pilot_artifact
from stability.pilot_config import PilotConfig
from stability.proteins_dataset import (
    CALIBRATION_GRAPH_COUNT,
    EVALUATION_GRAPH_COUNT,
    SELECTION_SEED,
    eligible_graph_id_manifest,
    load_proteins_graphs,
    select_graph_splits,
)
from stability.proteins_pilot import run_proteins_graph
from stability.wp7_calibration import choose_threshold


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tu"
ARTIFACT_ROOT = ROOT / "artifacts" / "proteins_pilot"
CALIBRATION_DIR = ARTIFACT_ROOT / "calibration_stable_clustering"
DOWNLOAD_DATA = False
MODEL_SEED_OFFSET = 950000
PERTURBATION_SEED_OFFSET = 960000
RECONSTRUCTION_SEED_OFFSET = 970000


def main():
    started = time.perf_counter()
    graphs, raw_hashes = load_proteins_graphs(DATA_ROOT, download=DOWNLOAD_DATA)
    split = select_graph_splits(graphs)
    manifest = eligible_graph_id_manifest(split)
    if len(split.calibration) != CALIBRATION_GRAPH_COUNT:
        raise RuntimeError("calibration split has an unexpected graph count")
    if len(split.evaluation) != EVALUATION_GRAPH_COUNT:
        raise RuntimeError("evaluation split has an unexpected graph count")

    calibration_results = []
    config = PilotConfig(curvature=1.0, tau=0.5)
    for index, graph in enumerate(split.calibration):
        print(
            f"Training PROTEINS calibration graph {index + 1}/"
            f"{len(split.calibration)}: {graph.graph_id}"
        )
        calibration_results.append(
            run_proteins_graph(
                graph=graph,
                config=config,
                model_seed=graph.dataset_index + MODEL_SEED_OFFSET,
                perturbation_seed=graph.dataset_index + PERTURBATION_SEED_OFFSET,
                reconstruction_seed=graph.dataset_index + RECONSTRUCTION_SEED_OFFSET,
                checkpoint_dir=CALIBRATION_DIR / "models",
            )
        )

    threshold_inputs = [
        {
            "original_edges": graph["original_edges"],
            "removed_edges": graph["removed_edges"],
            "candidate_edges": graph["candidate_edges"],
            "reconstructions": graph["partial_recovery"]["reconstructions"],
        }
        for graph in calibration_results
    ]
    selection = choose_threshold(threshold_inputs)
    payload = {
        "work_package": "bounded PROTEINS CPU pilot",
        "artifact_type": "calibration and frozen threshold",
        "status": "calibration_complete",
        "study_scope": "recovery of known graphs; per-graph models train on intact graph",
        "dataset": {
            "name": "PROTEINS",
            "loader": "torch_geometric.datasets.TUDataset(use_node_attr=True)",
            "raw_file_sha256": raw_hashes,
            "graph_labels_used": False,
        },
        "selection": manifest,
        "method": {
            "curvature": 1.0,
            "schedule_length": 200,
            "timestep_conditioning": "normalized t/T",
            "autoencoder_epochs_per_graph": 30,
            "denoiser_epochs_per_graph": 50,
            "total_autoencoder_epochs": 30 * len(calibration_results),
            "total_denoiser_epochs": 50 * len(calibration_results),
            "removed_edges_per_graph": 3,
            "reconstructions_per_graph": 8,
            "t_star": 7,
            "threshold_objective": selection["selection_objective"],
            "threshold_frozen_before_evaluation": True,
            "evaluation_split_metrics_not_used_for_selection": True,
        },
        "frozen_threshold_selection": selection,
        "calibration_graphs": calibration_results,
        "calibration_duration_seconds": time.perf_counter() - started,
    }
    destination = CALIBRATION_DIR / "calibration.json"
    save_pilot_artifact(destination, payload)
    print(
        json.dumps(
            {
                "artifact": str(destination),
                "tau": selection["tau"],
                "calibration_book_stability_score": selection[
                    "calibration_book_stability_score"
                ],
                "calibration_mean_false_positive_rate": selection[
                    "calibration_mean_false_positive_rate"
                ],
                "calibration_graphs": manifest["calibration"],
                "evaluation_graphs_reserved": manifest["evaluation"],
                "duration_seconds": payload["calibration_duration_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
