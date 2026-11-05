from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stability.encoder_damage_experiment import run_graph
from stability.pilot_artifacts import save_pilot_artifact
from stability.synthetic_instances import (
    ENCODER_DAMAGE_SEEDS,
    generate_synthetic_graph,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "encoder_damage_diagnostic"
MODEL_SEED_OFFSET = 20000


def main():
    results = []
    for index, seed in enumerate(ENCODER_DAMAGE_SEEDS):
        graph = generate_synthetic_graph(seed, "encoder_damage")
        print(
            f"Running paired encoder diagnostic {index + 1}/"
            f"{len(ENCODER_DAMAGE_SEEDS)}: {graph.graph_id}"
        )
        results.append(run_graph(graph, MODEL_SEED_OFFSET))
    payload = {
        "work_package": "paired damaged-input encoder diagnostic",
        "status": "descriptive_recovery_of_known_graphs",
        "graph_seeds": list(ENCODER_DAMAGE_SEEDS),
        "graph_count": len(results),
        "nodes_per_graph": 12,
        "edges_per_graph": 15,
        "variants": {
            "intact_input": "encode intact adjacency, reconstruct intact graph",
            "randomly_damaged_input": (
                "encode a fresh random three-edge-deleted adjacency each epoch, "
                "reconstruct the intact graph"
            ),
        },
        "architecture": "same HGCN LPModel and Fermi-Dirac decoder architecture",
        "features": "same fixed four-dimensional node features per paired graph",
        "reconstruction_target": (
            "all 15 original edges and the same 15 fixed true nonedges per graph"
        ),
        "training_epochs_per_variant_per_graph": 30,
        "evaluation": {
            "diffusion": False,
            "removed_edges_per_graph": 3,
            "candidate_pairs": "three removed edges plus all 51 true nonedges",
            "reported_metrics": [
                "average precision",
                "ROC-AUC",
                "removed edges among top three candidates",
            ],
            "training_corruptions_exclude_exact_evaluation_mask": True,
            "wp7_threshold_used": False,
        },
        "graphs": results,
    }
    destination = ARTIFACT_DIR / "results.json"
    save_pilot_artifact(destination, payload)
    summary = [
        {
            "graph_id": graph["graph_id"],
            "intact_input": graph["metrics"]["intact_input"],
            "randomly_damaged_input": graph["metrics"][
                "randomly_damaged_input"
            ],
            "training_seconds": {
                variant: graph["training"][variant]["duration_seconds"]
                for variant in ("intact_input", "randomly_damaged_input")
            },
        }
        for graph in results
    ]
    print(json.dumps({"artifact": str(destination), "graphs": summary}, indent=2))


if __name__ == "__main__":
    main()
