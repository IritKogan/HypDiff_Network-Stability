from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stability.posthoc_baselines import (
    candidate_additions,
    common_neighbor_scores,
    ranking_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ARTIFACT = (
    ROOT
    / "artifacts"
    / "proteins_pilot"
    / "evaluation_stable_clustering"
    / "evaluation.json"
)
OUTPUT_DIR = ROOT / "artifacts" / "proteins_pilot" / "posthoc_common_neighbors"
OUTPUT_FILE = OUTPUT_DIR / "comparison.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_score_map(candidate_pairs, scores):
    if len(candidate_pairs) != len(scores):
        raise ValueError("candidate pair and probability counts do not match")
    return {
        frozenset(pair): float(score)
        for pair, score in zip(candidate_pairs, scores)
    }


def build_comparison(evaluation: dict) -> dict:
    if evaluation.get("status") != "evaluation_complete":
        raise ValueError("expected the completed official evaluation artifact")
    results = []
    for graph in evaluation["evaluation_graphs"]:
        candidate_pairs = [tuple(pair) for pair in graph["candidate_edges"]]
        removed = [tuple(edge) for edge in graph["removed_edges"]]
        additions, labels = candidate_additions(
            graph["original_edges"], removed, candidate_pairs
        )
        addition_keys = [frozenset(pair) for pair in additions]
        if len(set(addition_keys)) != len(addition_keys):
            raise ValueError(f"{graph['graph_id']}: duplicate candidate addition")
        if sum(labels) != len(removed):
            raise ValueError(f"{graph['graph_id']}: removed edges missing from candidates")

        baseline_map = _canonical_score_map(
            candidate_pairs, graph["no_diffusion_baseline"]["probabilities"]
        )
        partial_records = graph["partial_recovery"]["reconstructions"]
        if not partial_records:
            raise ValueError(f"{graph['graph_id']}: missing partial reconstructions")
        partial_maps = [
            _canonical_score_map(candidate_pairs, row["probabilities"])
            for row in partial_records
        ]
        partial_mean = {
            key: sum(row[key] for row in partial_maps) / len(partial_maps)
            for key in addition_keys
        }

        cn_values = common_neighbor_scores(graph["damaged_edges"], additions)
        method_scores = {
            "common_neighbors": [int(score) for score in cn_values],
            "no_diffusion": [baseline_map[key] for key in addition_keys],
            "partial_recovery_mean_probability": [
                partial_mean[key] for key in addition_keys
            ],
        }
        method_metrics = {
            method: ranking_metrics(scores, labels)
            for method, scores in method_scores.items()
        }
        edge_ranks = []
        for edge in removed:
            key = frozenset(edge)
            index = addition_keys.index(key)
            edge_ranks.append(
                {
                    "edge": list(edge),
                    "scores": {
                        method: values[index]
                        for method, values in method_scores.items()
                    },
                    "competition_ranks": {
                        method: metrics["competition_ranks"][index]
                        for method, metrics in method_metrics.items()
                    },
                    "tie_counts_at_edge_score": {
                        method: int(
                            sum(value == values[index] for value in values)
                        )
                        for method, values in method_scores.items()
                    },
                }
            )
        results.append(
            {
                "graph_id": graph["graph_id"],
                "node_count": len(graph["node_ids"]),
                "original_edge_count": len(graph["original_edges"]),
                "removed_edges": [list(edge) for edge in removed],
                "candidate_addition_count": len(additions),
                "true_nonedge_count": len(additions) - len(removed),
                "partial_reconstruction_count": len(partial_records),
                "methods": {
                    method: {
                        key: value
                        for key, value in metrics.items()
                        if key != "competition_ranks"
                    }
                    for method, metrics in method_metrics.items()
                },
                "removed_edge_ranks": edge_ranks,
                "candidate_additions": [list(pair) for pair in additions],
                "scores_by_candidate_addition": method_scores,
            }
        )
    return {
        "artifact_type": "post-hoc ranking comparison",
        "status": "complete",
        "official_evaluation_status": evaluation["status"],
        "official_frozen_tau_context_only": evaluation["frozen_tau"],
        "method_scope": (
            "Common Neighbors uses damaged adjacency only; no-diffusion uses saved "
            "probabilities; partial recovery uses mean saved probabilities across "
            "the existing reconstructions. Candidate additions are removed edges "
            "plus true nonedges, excluding surviving edges."
        ),
        "ranking_ties": (
            "Competition rank is 1 + the number of strictly higher scores. "
            "Guaranteed top-three counts positives strictly above the third-place "
            "score; possible counts those plus positive ties that could occupy "
            "remaining top-three slots."
        ),
        "evaluation_graphs": results,
    }


def main():
    if not EVALUATION_ARTIFACT.is_file():
        raise FileNotFoundError(f"evaluation artifact not found: {EVALUATION_ARTIFACT}")
    if OUTPUT_DIR.exists() or OUTPUT_FILE.exists():
        raise FileExistsError(f"refusing to overwrite post-hoc results: {OUTPUT_DIR}")
    source_hash_before = _sha256(EVALUATION_ARTIFACT)
    evaluation = json.loads(EVALUATION_ARTIFACT.read_text(encoding="utf-8"))
    result = build_comparison(evaluation)
    if _sha256(EVALUATION_ARTIFACT) != source_hash_before:
        raise RuntimeError("official evaluation artifact changed during comparison")
    result["source_evaluation_artifact"] = str(
        EVALUATION_ARTIFACT.relative_to(ROOT)
    )
    result["source_evaluation_sha256"] = source_hash_before
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result_file": str(OUTPUT_FILE), "graphs": len(result["evaluation_graphs"])}, indent=2))


if __name__ == "__main__":
    main()
