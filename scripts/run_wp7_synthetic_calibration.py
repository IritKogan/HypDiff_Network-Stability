from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from stability.pilot_artifacts import load_pilot_artifact, save_pilot_artifact
from stability.pilot_config import PilotConfig
from stability.synthetic_instances import (
    CALIBRATION_SEEDS,
    generate_synthetic_graph,
    validate_split_seeds,
)
from stability.wp7_calibration import choose_threshold
from stability.wp7_pipeline import run_graph


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "wp7_synthetic_calibration"
WP6_ARTIFACT = ROOT / "artifacts" / "wp5_synthetic_pilot" / "results.json"
CALIBRATION_MODEL_SEED_OFFSET = 73000
PERTURBATION_SEED_OFFSET = 74000
RECONSTRUCTION_SEED_OFFSET = 75000


def describe_wp6_ranking(path: Path):
    artifact = load_pilot_artifact(path)
    graph = artifact["graph"]
    node_ids = graph["node_ids"]
    edge_order = [
        (node_ids[source], node_ids[target])
        for source in range(len(node_ids))
        for target in range(source + 1, len(node_ids))
    ]
    removed = {frozenset(edge) for edge in artifact["removed_edges"]}
    original = {frozenset(edge) for edge in graph["original_edges"]}
    true_nonedges = {
        frozenset(edge) for edge in edge_order
    } - original
    labels = []
    scores = []
    for reconstruction in artifact["reconstructions"]:
        probabilities = reconstruction["probabilities"]
        if len(probabilities) != len(edge_order):
            raise ValueError("WP6 probability vector does not match candidate-edge order")
        for edge, probability in zip(edge_order, probabilities):
            key = frozenset(edge)
            if key in removed:
                labels.append(1)
                scores.append(float(probability))
            elif key in true_nonedges:
                labels.append(0)
                scores.append(float(probability))
    if not labels or len(set(labels)) != 2:
        raise ValueError("WP6 artifact must include removed edges and true nonedges")
    positives = np.asarray(scores)[np.asarray(labels) == 1]
    negatives = np.asarray(scores)[np.asarray(labels) == 0]
    return {
        "source": str(path.relative_to(ROOT)),
        "description": (
            "Descriptive ranking only; WP6 probabilities are from one synthetic "
            "graph and are not used to select the WP7 threshold."
        ),
        "positive_scores": int(len(positives)),
        "true_nonedge_scores": int(len(negatives)),
        "removed_edge_probability_mean": float(positives.mean()),
        "true_nonedge_probability_mean": float(negatives.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "pairwise_positive_above_negative_fraction": float(
            np.mean(
                (positives[:, None] > negatives[None, :])
                + 0.5 * (positives[:, None] == negatives[None, :])
            )
        ),
    }


def main():
    validate_split_seeds()
    ranking = describe_wp6_ranking(WP6_ARTIFACT)
    config = PilotConfig(curvature=1.0, tau=0.5)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    graph_results = []
    for index, seed in enumerate(CALIBRATION_SEEDS):
        graph = generate_synthetic_graph(seed, "calibration")
        print(f"Training calibration graph {index + 1}/{len(CALIBRATION_SEEDS)}: {graph.graph_id}")
        result = run_graph(
            graph=graph,
            config=config,
            model_seed=seed + CALIBRATION_MODEL_SEED_OFFSET,
            perturbation_seed=seed + PERTURBATION_SEED_OFFSET,
            reconstruction_seed=seed + RECONSTRUCTION_SEED_OFFSET,
            checkpoint_dir=ARTIFACT_DIR / "models",
        )
        result["partial_recovery"] = {
            "reconstructions": [
                {
                    "seed": reconstruction["seed"],
                    "probabilities": reconstruction["probabilities"],
                }
                for reconstruction in result["partial_recovery"]["reconstructions"]
            ],
            "t_star": result["partial_recovery"]["t_star"],
            "reconstruction_count": result["partial_recovery"][
                "reconstruction_count"
            ],
        }
        result.pop("no_diffusion_baseline")
        graph_results.append(result)

    threshold_inputs = [
        {
            "original_edges": graph["original_edges"],
            "removed_edges": graph["removed_edges"],
            "candidate_edges": graph["candidate_edges"],
            "reconstructions": graph["partial_recovery"]["reconstructions"],
        }
        for graph in graph_results
    ]
    frozen_selection = choose_threshold(threshold_inputs)
    payload = {
        "work_package": "WP7",
        "artifact_type": "synthetic calibration; frozen threshold source",
        "status": "calibration_complete",
        "split": {
            "graph_count": len(graph_results),
            "graph_seeds": list(CALIBRATION_SEEDS),
            "node_count_per_graph": 12,
            "evaluation_seeds_reserved": [72001, 72002, 72003, 72004, 72005],
            "seeds_disjoint": True,
            "graph_instances_distinct": True,
        },
        "method": {
            "curvature": 1.0,
            "schedule_length": 200,
            "timestep_conditioning": "normalized t/T",
            "autoencoder_epochs_per_graph": 30,
            "denoiser_epochs_per_graph": 50,
            "total_autoencoder_epochs": 30 * len(graph_results),
            "total_denoiser_epochs": 50 * len(graph_results),
            "removed_edges_per_graph": 3,
            "reconstructions_per_graph": 8,
            "t_star": 7,
            "threshold_frozen_before_evaluation": True,
        },
        "wp6_descriptive_ranking": ranking,
        "frozen_threshold_selection": frozen_selection,
        "graphs": graph_results,
    }
    destination = ARTIFACT_DIR / "calibration.json"
    save_pilot_artifact(destination, payload)
    print(json.dumps({
        "calibration_artifact": str(destination),
        "tau": frozen_selection["tau"],
        "calibration_book_stability_score": frozen_selection[
            "calibration_book_stability_score"
        ],
        "calibration_mean_false_positive_rate": frozen_selection[
            "calibration_mean_false_positive_rate"
        ],
        "wp6_ranking_roc_auc": ranking["roc_auc"],
        "wp6_ranking_average_precision": ranking["average_precision"],
    }, indent=2))


if __name__ == "__main__":
    main()
