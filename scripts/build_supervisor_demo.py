from __future__ import annotations

import base64
import gzip
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_single_recovery import score_saved_predictions


OUTPUT = ROOT / "artifacts" / "recovery_runs" / "hypdiff_phaseb_supervisor_demo.html"
RUNS = {
    "PROTEINS-0080": ROOT
    / "artifacts"
    / "recovery_runs"
    / "acceptance_PROTEINS-0080_targeted_20261111_03",
    "IMDB-BINARY-0377": ROOT
    / "artifacts"
    / "recovery_runs"
    / "acceptance_IMDB-BINARY-0377_random_20261111_03",
}
PROTEINS = ROOT / "artifacts" / "proteins_pilot" / "score_scale_diagnostic_20261110"
IMDB_PILOT = (
    ROOT
    / "artifacts"
    / "imdb_binary_pilot_20261110"
    / "recovery_20261110_01"
)
IMDB_EXPANDED = (
    ROOT
    / "artifacts"
    / "imdb_binary_pilot_20261110"
    / "expanded_evaluation_20261110_01"
)
SYNTHETIC = ROOT / "artifacts" / "synthetic_pilot" / "sbm_ba_recovery_20261105_retry03"
CORA_RUNS = ROOT / "artifacts" / "cora_edge_recovery"

ATTACK_LABELS = {
    "uniform_random": "Random removal",
    "targeted_degree_product": "Targeted removal",
}
DAMAGE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required demo artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _edge_key(edge) -> tuple[str, str]:
    if len(edge) != 2 or edge[0] == edge[1]:
        raise ValueError(f"invalid undirected edge: {edge}")
    return tuple(sorted((str(edge[0]), str(edge[1]))))


def _edge_map(edges) -> dict[tuple[str, str], list[str]]:
    result = {}
    for edge in edges:
        key = _edge_key(edge)
        if key in result:
            raise ValueError(f"duplicate edge in saved data: {edge}")
        result[key] = [key[0], key[1]]
    return result


def _same_metric(actual: float, expected: float, description: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(
            f"{description} differs from the saved evaluation: "
            f"{actual!r} != {expected!r}"
        )


def _method_summary(summary: dict, method_name: str, mask_index: int) -> dict:
    method = summary["method_metrics"][method_name]
    matches = [
        row for row in method["masks"] if int(row["mask_index"]) == int(mask_index)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method_name} summary for mask {mask_index}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _evaluate_edges(
    graph_id: str,
    node_ids: list[str],
    original_edges: list[list[str]],
    damaged_edges: list[list[str]],
    removed_edges: list[list[str]],
    candidate_edges: list[list[str]],
    reconstructions: list[list[list[str]]],
) -> dict:
    original = _edge_map(original_edges)
    damaged = _edge_map(damaged_edges)
    removed = _edge_map(removed_edges)
    candidates = _edge_map(candidate_edges)
    if set(damaged) & set(removed) or set(damaged) | set(removed) != set(original):
        raise ValueError(f"{graph_id}: damaged and removed edges do not partition original")
    if not set(removed) <= set(candidates):
        raise ValueError(f"{graph_id}: removed edges are absent from candidate pairs")
    if any(not set(_edge_map(edges)) <= set(candidates) for edges in reconstructions):
        raise ValueError(f"{graph_id}: predicted edges are absent from candidate pairs")

    predictions = {
        "graph_id": graph_id,
        "candidate_edges": candidate_edges,
        "reconstructions": [{"edges": edges} for edges in reconstructions],
    }
    labels = {
        "graph_id": graph_id,
        "node_ids": node_ids,
        "original_edges": original_edges,
        "damaged_edges": damaged_edges,
        "removed_edges": removed_edges,
    }
    score = score_saved_predictions(predictions, labels)
    true_nonedges = {
        _edge_key((node_ids[i], node_ids[j]))
        for i in range(len(node_ids))
        for j in range(i + 1, len(node_ids))
    } - set(original)
    removed_keys = set(removed)
    sample_metrics = []
    for edge_set in reconstructions:
        predicted = {_edge_key(edge) for edge in edge_set}
        sample_metrics.append(
            {
                "recovered": len(predicted & removed_keys),
                "false_positives": len(predicted & true_nonedges),
            }
        )
    return {
        "score": float(score["book_stability_score"]),
        "mean_false_positives": float(score["mean_false_positive_edges"]),
        "true_nonedge_count": int(score["true_nonedge_count"]),
        "sample_metrics": sample_metrics,
        "reconstructions": reconstructions,
    }


def _check_saved_metrics(
    record: dict,
    saved: dict,
    graph_id: str,
    method_name: str,
) -> None:
    _same_metric(
        record["score"],
        saved["book_stability_score"],
        f"{graph_id} {method_name} stability",
    )
    _same_metric(
        record["mean_false_positives"],
        saved.get("absolute_false_positive_edges", saved.get("mean_false_positive_edges")),
        f"{graph_id} {method_name} false positives",
    )
    saved_fpr = saved.get("realized_fpr", saved.get("mean_realized_fpr"))
    if saved_fpr is not None:
        computed_fpr = record["mean_false_positives"] / record["true_nonedge_count"]
        _same_metric(computed_fpr, saved_fpr, f"{graph_id} {method_name} realized FPR")
    if len(record["sample_metrics"]) != int(saved["reconstruction_count"]):
        raise ValueError(f"{graph_id} {method_name} reconstruction count differs")
    for index, (computed, expected) in enumerate(
        zip(record["sample_metrics"], saved.get("per_reconstruction", []))
    ):
        if computed["recovered"] != int(expected["recovered_removed_edges"]):
            raise ValueError(
                f"{graph_id} {method_name} sample {index + 1} recovery differs"
            )
        if computed["false_positives"] != int(expected["false_positive_edges"]):
            raise ValueError(
                f"{graph_id} {method_name} sample {index + 1} false positives differ"
            )


def _summary_only_method(saved: dict, threshold: float | None) -> dict:
    return {
        "threshold": threshold,
        "score": float(saved["book_stability_score"]),
        "mean_false_positives": float(saved["absolute_false_positive_edges"]),
        "sample_metrics": [
            {
                "recovered": int(row["recovered_removed_edges"]),
                "false_positives": int(row["false_positive_edges"]),
            }
            for row in saved["per_reconstruction"]
        ],
        "reconstructions": None,
    }


def _case_record(
    *,
    graph_id: str,
    dataset: str,
    attack: str,
    requested_damage: float,
    mask_index: int,
    mask_seed,
    node_ids: list[str],
    original_edges: list[list[str]],
    damaged_edges: list[list[str]],
    removed_edges: list[list[str]],
    partial: dict,
    no_diffusion: dict | None,
    source: str,
    realized_damage: float | None = None,
) -> dict:
    original = _edge_map(original_edges)
    damaged = _edge_map(damaged_edges)
    removed = _edge_map(removed_edges)
    if not original or not removed:
        raise ValueError(f"{graph_id}: saved case has no original or removed edges")
    if set(damaged) & set(removed) or set(damaged) | set(removed) != set(original):
        raise ValueError(f"{graph_id}: graph, damage, and labels do not agree")
    actual = len(removed) / len(original)
    if realized_damage is not None:
        _same_metric(actual, realized_damage, f"{graph_id} actual damage")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"{graph_id}: node IDs are not unique")
    node_set = set(node_ids)
    if any(not set(edge) <= node_set for edge in original.values()):
        raise ValueError(f"{graph_id}: an original edge references an unknown node")
    return {
        "graph_id": graph_id,
        "dataset": dataset,
        "attack": ATTACK_LABELS[attack],
        "requested_damage": float(requested_damage),
        "actual_damage": actual,
        "mask_index": int(mask_index),
        "mask_seed": mask_seed,
        "node_ids": node_ids,
        "original_edges": list(original.values()),
        "damaged_edges": list(damaged.values()),
        "removed_edges": list(removed.values()),
        "partial": partial,
        "no_diffusion": no_diffusion,
        "graph_available": partial["reconstructions"] is not None,
        "source": source,
    }


def _threshold_predictions(candidate_edges: list, score_rows, threshold: float):
    results = []
    for scores in score_rows:
        if len(scores) != len(candidate_edges):
            raise ValueError("saved score vector does not match candidate-edge count")
        results.append(
            [
                edge
                for edge, score in zip(candidate_edges, scores)
                if float(score) >= threshold
            ]
        )
    return results


def _load_proteins_cases() -> list[dict]:
    manifest = _read_json(PROTEINS / "manifest.json")
    labels_by_graph = _read_json(PROTEINS / "evaluation_labels.json")
    candidates_by_graph = _read_json(PROTEINS / "candidate_pairs.json")
    summary = _read_json(PROTEINS / "summary.json")
    scores_file = np.load(PROTEINS / "decoder_scores.npz", allow_pickle=False)
    threshold = float(manifest["threshold"])
    summary_by_case = {row["case_id"]: row for row in summary["per_mask_results"]}
    records = []
    for graph_id in manifest["graph_ids"]:
        graph_labels = labels_by_graph[graph_id]
        graph_candidates = candidates_by_graph[graph_id]["candidate_pairs"]
        if len(graph_candidates) != int(graph_labels["candidate_count"]):
            raise ValueError(f"{graph_id}: PROTEINS candidate count mismatch")
        all_candidates = _edge_map(graph_candidates)
        candidate_list = list(all_candidates.values())
        true_nonedge_indices = set(
            int(index) for index in graph_labels["true_nonedge_candidate_indices"]
        )
        if any(index < 0 or index >= len(candidate_list) for index in true_nonedge_indices):
            raise ValueError(f"{graph_id}: invalid true-nonedge candidate index")
        original = [
            edge for index, edge in enumerate(candidate_list) if index not in true_nonedge_indices
        ]
        node_ids = sorted(
            {node for edge in candidate_list for node in edge},
            key=lambda node: int(node.rsplit("n", 1)[1]),
        )
        score_array = scores_file[f"scores__{graph_id}"]
        if score_array.shape[0] != len(graph_labels["cases"]):
            raise ValueError(f"{graph_id}: score rows do not match case manifest")
        if score_array.shape[2] != len(candidate_list):
            raise ValueError(f"{graph_id}: score columns do not match candidate pairs")
        for case_index, case in enumerate(graph_labels["cases"]):
            parsed = re.fullmatch(
                r"(.+)\|([^|]+)\|(\d+)%\|mask(\d+)", case["case_id"]
            )
            if not parsed or parsed.group(1) != graph_id:
                raise ValueError(f"unrecognized PROTEINS case ID: {case['case_id']}")
            attack, damage_percent, mask_index = (
                parsed.group(2),
                int(parsed.group(3)),
                int(parsed.group(4)),
            )
            removed_indices = set(
                int(index) for index in case["removed_edge_candidate_indices"]
            )
            if not removed_indices or removed_indices & true_nonedge_indices:
                raise ValueError(f"{case['case_id']}: removed-edge labels are invalid")
            removed = [candidate_list[index] for index in sorted(removed_indices)]
            damaged = [edge for edge in original if _edge_key(edge) not in set(map(_edge_key, removed))]
            predicted = _threshold_predictions(
                candidate_list, score_array[case_index], threshold
            )
            computed = _evaluate_edges(
                graph_id, node_ids, original, damaged, removed, candidate_list, predicted
            )
            summary_row = summary_by_case.get(case["case_id"])
            if summary_row is None:
                raise ValueError(f"{case['case_id']}: score-scale summary is missing")
            _check_saved_metrics(computed, summary_row, graph_id, "partial recovery")
            records.append(
                _case_record(
                    graph_id=graph_id,
                    dataset="PROTEINS",
                    attack=attack,
                    requested_damage=damage_percent / 100,
                    mask_index=mask_index,
                    mask_seed=case.get("mask_seed"),
                    node_ids=node_ids,
                    original_edges=original,
                    damaged_edges=damaged,
                    removed_edges=removed,
                    partial={**computed, "threshold": threshold},
                    no_diffusion=None,
                    source="PROTEINS score-scale diagnostic",
                    realized_damage=summary_row["damage_fraction_realized"],
                )
            )
    return records


def _index_summary_rows(rows: list[dict]) -> dict[tuple, dict]:
    indexed = {}
    for row in rows:
        key = (
            row["graph_id"],
            row["attack_type"],
            round(float(row["damage_fraction_requested"]), 8),
        )
        if key in indexed:
            raise ValueError(f"duplicate saved evaluation summary row: {key}")
        indexed[key] = row
    return indexed


def _load_imdb_cases(
    graph_dir: Path,
    evaluation_summary: dict,
    thresholds: dict[str, float],
    dataset: str,
) -> list[dict]:
    summary_rows = _index_summary_rows(
        evaluation_summary["per_graph_attack_damage_results"]
    )
    records = []
    for path in sorted(graph_dir.glob("*.json")):
        graph = _read_json(path)
        graph_id = graph["graph_id"]
        node_ids = graph["node_ids"]
        if len(node_ids) != len(graph.get("features", node_ids)):
            raise ValueError(f"{graph_id}: feature rows are not aligned to node IDs")
        for case in graph["cases"]:
            summary_key = (
                graph_id,
                case["attack_type"],
                round(float(case["damage_fraction_requested"]), 8),
            )
            summary_row = summary_rows.get(summary_key)
            if summary_row is None:
                raise ValueError(f"{graph_id}: evaluation summary row is missing")
            mask_index = int(case["mask_index"])
            original = graph["original_edges"]
            damaged = case["damaged_edges"]
            removed = case["removed_edges"]
            candidates = case["candidate_edges"]
            original_keys = set(_edge_map(original))
            damaged_keys = set(_edge_map(damaged))
            if original_keys - damaged_keys != set(_edge_map(removed)):
                raise ValueError(f"{graph_id}: saved removed edges do not match the mask")
            candidate_keys = set(_edge_map(candidates))
            if not (original_keys - damaged_keys) <= candidate_keys:
                raise ValueError(f"{graph_id}: candidate list omits removed edges")
            partial_edges = _threshold_predictions(
                candidates,
                case["partial_probabilities"],
                thresholds["partial_recovery"],
            )
            partial = _evaluate_edges(
                graph_id, node_ids, original, damaged, removed, candidates, partial_edges
            )
            saved_partial = _method_summary(summary_row, "partial_recovery", mask_index)
            _check_saved_metrics(partial, saved_partial, graph_id, "partial recovery")

            baseline_scores = case["baseline_probabilities"]
            baseline_edges = _threshold_predictions(
                candidates, [baseline_scores], thresholds["no_diffusion_baseline"]
            )
            baseline = _evaluate_edges(
                graph_id, node_ids, original, damaged, removed, candidates, baseline_edges
            )
            saved_baseline = _method_summary(
                summary_row, "no_diffusion_baseline", mask_index
            )
            _check_saved_metrics(baseline, saved_baseline, graph_id, "no diffusion")

            if int(saved_partial["reconstruction_count"]) != len(partial_edges):
                raise ValueError(f"{graph_id}: partial reconstruction count mismatch")
            if case.get("mask_seed") is not None:
                if int(saved_partial["mask_seed"]) != int(case["mask_seed"]):
                    raise ValueError(f"{graph_id}: partial mask seed mismatch")
                if int(saved_baseline["mask_seed"]) != int(case["mask_seed"]):
                    raise ValueError(f"{graph_id}: baseline mask seed mismatch")
            records.append(
                _case_record(
                    graph_id=graph_id,
                    dataset=dataset,
                    attack=case["attack_type"],
                    requested_damage=case["damage_fraction_requested"],
                    mask_index=mask_index,
                    mask_seed=case.get("mask_seed"),
                    node_ids=node_ids,
                    original_edges=original,
                    damaged_edges=damaged,
                    removed_edges=removed,
                    partial={**partial, "threshold": thresholds["partial_recovery"]},
                    no_diffusion={
                        **baseline,
                        "threshold": thresholds["no_diffusion_baseline"],
                    },
                    source=f"{dataset} saved evaluation probabilities",
                    realized_damage=case["damage_fraction_realized"],
                )
            )
    return records


def _load_synthetic_cases() -> list[dict]:
    manifest = _read_json(SYNTHETIC / "selection_and_masks.json")
    records = []
    for family in ("SBM", "BA"):
        evaluation = _read_json(SYNTHETIC / family / "evaluation.json")
        if evaluation.get("thresholds_frozen_before_evaluation") is not True:
            raise ValueError(f"{family}: evaluation does not certify frozen thresholds")
        thresholds = evaluation["method_thresholds"]
        summary_rows = _index_summary_rows(
            evaluation["per_graph_attack_damage_results"]
        )
        graph_metadata = {
            row["graph_id"]: row
            for row in manifest["families"][family]["evaluation"]
        }
        for graph_id, masks in manifest["families"][family]["attack_masks"].items():
            if graph_id not in graph_metadata or "-evaluation-" not in graph_id:
                continue
            node_count = int(graph_metadata[graph_id]["node_count"])
            node_ids = [f"{graph_id}:n{index:02d}" for index in range(node_count)]
            original_edges = None
            prepared = []
            for mask in masks:
                if mask["attack_type"] not in ATTACK_LABELS or not mask.get("removed_edges"):
                    continue
                if mask.get("damage_fraction_requested") not in DAMAGE_LEVELS:
                    continue
                damaged = mask["damaged_edges"]
                removed = mask["removed_edges"]
                original = list(_edge_map([*damaged, *removed]).values())
                if original_edges is None:
                    original_edges = original
                elif set(_edge_map(original)) != set(_edge_map([*damaged, *removed])):
                    raise ValueError(f"{graph_id}: masks do not share an original graph")
                prepared.append((mask, original))
            if original_edges is None:
                raise ValueError(f"{graph_id}: synthetic masks are missing")
            for mask, original in prepared:
                summary_key = (
                    graph_id,
                    mask["attack_type"],
                    round(float(mask["damage_fraction_requested"]), 8),
                )
                summary_row = summary_rows.get(summary_key)
                if summary_row is None:
                    raise ValueError(f"{graph_id}: synthetic summary row is missing")
                partial_saved = _method_summary(
                    summary_row, "partial_recovery", mask["mask_index"]
                )
                baseline_saved = _method_summary(
                    summary_row, "no_diffusion_baseline", mask["mask_index"]
                )
                for saved in (partial_saved, baseline_saved):
                    if saved.get("mask_seed") is not None and int(saved["mask_seed"]) != int(
                        mask["mask_seed"]
                    ):
                        raise ValueError(f"{graph_id}: synthetic mask seed does not match")
                partial = _summary_only_method(
                    partial_saved, thresholds["partial_recovery"]
                )
                baseline = _summary_only_method(
                    baseline_saved, thresholds["no_diffusion_baseline"]
                )
                records.append(
                    _case_record(
                        graph_id=graph_id,
                        dataset=family,
                        attack=mask["attack_type"],
                        requested_damage=mask["damage_fraction_requested"],
                        mask_index=mask["mask_index"],
                        mask_seed=mask["mask_seed"],
                        node_ids=node_ids,
                        original_edges=original,
                        damaged_edges=mask["damaged_edges"],
                        removed_edges=mask["removed_edges"],
                        partial=partial,
                        no_diffusion=baseline,
                        source=f"{family} saved summary metrics; predicted edge sets not saved",
                        realized_damage=mask["damage_fraction_realized"],
                    )
                )
    return records


def _load_cora_gui_results() -> dict | None:
    summaries = sorted(
        CORA_RUNS.glob("run_*/summary.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not summaries:
        return None
    result = _read_json(summaries[0])
    if (
        result.get("status") != "complete"
        or result.get("artifact_type") != "Cora cumulative edge recovery experiment"
    ):
        raise ValueError(f"{summaries[0]} is not a completed Cora edge-recovery run")
    reconstruction_count = int(result["reconstruction_count"])
    attack_series = []
    for series in result["attack_series"]:
        levels = []
        for level in series["damage_levels"]:
            recovery = level["removed_edge_recovery"]
            histogram = recovery["histogram"]
            if [row["recovered_count"] for row in histogram] != list(
                range(reconstruction_count + 1)
            ):
                raise ValueError("Cora histogram must contain every 0-through-K bin")
            levels.append(
                {
                    key: level[key]
                    for key in (
                        "damage_fraction_requested",
                        "original_edge_count",
                        "edges_removed",
                        "removed_edge_recovery",
                        "false_positive_additions",
                    )
                }
            )
        attack_series.append(
            {"attack_type": series["attack_type"], "damage_levels": levels}
        )
    return {
        "dataset": "Cora",
        "primary_evaluation_damage_fraction": float(
            result["primary_evaluation_damage_fraction"]
        ),
        "reconstruction_count": reconstruction_count,
        "attack_series": attack_series,
    }


def _cora_graph_marker(cora_results: dict) -> list[dict]:
    cases = []
    for series in cora_results["attack_series"]:
        for level in series["damage_levels"]:
            cases.append(
                {
                    "graph_id": "Cora",
                    "dataset": "Cora",
                    "attack": ATTACK_LABELS[series["attack_type"]],
                    "requested_damage": level["damage_fraction_requested"],
                    "mask_index": 0,
                    "partial": {
                        "score": None,
                        "sample_metrics": [],
                        "reconstructions": [],
                    },
                }
            )
    return cases


def _check_saved_run(case_id: str, run_dir: Path) -> dict:
    config = _read_json(run_dir / "run_config.json")
    predictions = _read_json(run_dir / "predictions.json")
    labels = _read_json(run_dir / "evaluation_labels.json")
    saved_score = _read_json(run_dir / "score.json")
    recomputed = score_saved_predictions(predictions, labels)
    for key in (
        "actual_damage",
        "book_stability_score",
        "mean_false_positive_edges",
        "per_reconstruction_false_positive_edges",
        "removed_edge_recovery_frequencies",
        "reconstruction_count",
    ):
        if recomputed[key] != saved_score[key]:
            raise ValueError(f"{case_id}: recomputed {key} differs from score.json")
    if config["graph_id"] != case_id or predictions["graph_id"] != case_id:
        raise ValueError(f"{case_id}: saved run artifacts identify another graph")
    expected = {
        "PROTEINS-0080": (0.75, 4.875),
        "IMDB-BINARY-0377": (1.0, 0.0),
    }[case_id]
    _same_metric(recomputed["book_stability_score"], expected[0], case_id + " stability")
    _same_metric(
        recomputed["mean_false_positive_edges"], expected[1], case_id + " false positives"
    )
    return {"predictions": predictions, "labels": labels, "score": recomputed}


def _check_acceptance_cases(records: list[dict]) -> None:
    indexed = {
        (
            row["graph_id"],
            row["attack"],
            round(row["requested_damage"], 8),
            row["mask_index"],
        ): row
        for row in records
    }
    checks = (
        ("PROTEINS-0080", "Targeted removal", 0.1, 0, "PROTEINS-0080"),
        ("IMDB-BINARY-0377", "Random removal", 0.1, 1, "IMDB-BINARY-0377"),
    )
    for graph_id, attack, damage, mask_index, run_id in checks:
        record = indexed.get((graph_id, attack, round(damage, 8), mask_index))
        if record is None:
            raise ValueError(f"{graph_id}: exact accepted demo case was not loaded")
        run = _check_saved_run(run_id, RUNS[run_id])
        labels = run["labels"]
        if set(_edge_map(record["original_edges"])) != set(
            _edge_map(labels["original_edges"])
        ):
            raise ValueError(f"{graph_id}: demo original graph differs from saved run")
        if set(_edge_map(record["removed_edges"])) != set(
            _edge_map(labels["removed_edges"])
        ):
            raise ValueError(f"{graph_id}: demo removed-edge mask differs from saved run")
        if record["partial"]["reconstructions"] is None:
            raise ValueError(f"{graph_id}: accepted case has no reconstruction edge sets")
        if len(record["partial"]["reconstructions"]) != len(
            run["predictions"]["reconstructions"]
        ):
            raise ValueError(f"{graph_id}: accepted run reconstruction count differs")
        for demo_edges, run_row in zip(
            record["partial"]["reconstructions"],
            run["predictions"]["reconstructions"],
        ):
            if set(_edge_map(demo_edges)) != set(_edge_map(run_row["edges"])):
                raise ValueError(f"{graph_id}: demo predictions differ from saved run")


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Network Stability with HypDiff</title>
<style>
:root{color-scheme:light;--ink:#17212f;--muted:#526174;--paper:#f2f5f8;--panel:#fff;--line:#d6dde7;--survive:#526174;--removed:#bd2638;--recovered:#00836f;--false:#d97500}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.35 "Segoe UI",Arial,sans-serif}
[hidden]{display:none!important}
header{background:#10233f;color:#fff;padding:14px 28px}header h1{margin:0;font-size:25px}
main{max-width:1900px;margin:auto;padding:20px 28px}
.controls{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.selector label{display:block;font-weight:700;margin-bottom:5px;font-size:15px}
select{width:100%;font:inherit;padding:9px 10px;border:1px solid #8794a7;border-radius:6px;background:#fff;color:var(--ink)}
.cards{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:12px;margin:0 0 16px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 15px;min-height:88px;box-shadow:0 3px 12px #12233b0b}
.metric-label{font-size:14px;font-weight:700;color:var(--muted)}
.metric-value{font-size:26px;font-weight:750;margin-top:5px;line-height:1.1}
.metric.stability{background:#10233f;color:white;border:1px solid #10233f;border-radius:10px;padding:10px 14px;box-shadow:0 6px 18px #10233f24}
.metric.stability .metric-label{font-size:16px;color:#d7e5f7}
.metric.stability .metric-value{font-size:40px;color:#fff}
.metric-meta{font-size:13px;color:#d7e5f7;margin-top:3px}
.chart-panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 16px;box-shadow:0 3px 12px #12233b0b}
.chart-panel h2{font-size:19px;margin:0 0 4px}
.chart-caption{font-size:14px;color:var(--muted);margin:0 0 8px}
.chart-wrap{width:100%;overflow-x:auto}
.chart-wrap svg{display:block;width:100%;height:345px}
.chart-unavailable{height:345px;display:grid;place-items:center;text-align:center;padding:24px;color:var(--muted);font-size:18px;font-weight:700}
.axis-text{font:15px "Segoe UI",Arial,sans-serif;fill:var(--muted)}
.axis-title{font:15px "Segoe UI",Arial,sans-serif;fill:var(--ink);font-weight:700}
.grid-line{stroke:#dbe2eb;stroke-width:1}
.chart-line{fill:none;stroke:#2864a8;stroke-width:4;stroke-linecap:round}
.chart-point{fill:#2864a8;stroke:#fff;stroke-width:3}
.chart-point.selected{fill:#d97706;stroke:#7c3d00;stroke-width:3}
.selected-level{stroke:#d97706;stroke-width:2;stroke-dasharray:6 6;opacity:.75}
.selected-tick{fill:#a64b00;font-weight:800}
.graphs{display:grid;grid-template-columns:repeat(3,minmax(320px,1fr));gap:15px;overflow-x:auto}
.graph-card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;box-shadow:0 4px 14px #12233b0b}
.graph-card h2{margin:0;padding:12px 15px;border-bottom:1px solid var(--line);font-size:19px}
.graph-card svg{display:block;width:100%;height:auto;min-height:340px}
.graph-card .unavailable{height:340px;display:grid;place-items:center;padding:30px;text-align:center;color:var(--muted);font-weight:650}
.legend{display:flex;align-items:center;justify-content:center;gap:12px 28px;flex-wrap:wrap;margin:13px 0;font-size:14px}
.key{display:inline-flex;align-items:center;gap:8px}.swatch{display:inline-block;width:30px;height:0;border-top:3px solid}
.swatch.removed{border-top:3px dashed var(--removed)}.swatch.recovered{border-color:var(--recovered);border-width:4px}.swatch.false{border-top:3px dotted var(--false)}
.selected-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px 18px;margin:12px 0 14px;box-shadow:0 4px 14px #12233b0b}
.selected-card h2{font-size:18px;margin:0 0 10px}
.selected-counters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.selected-counter{border-left:5px solid var(--line);padding:4px 12px;min-height:75px}
.selected-counter.recovered{border-color:var(--recovered)}.selected-counter.incorrect{border-color:var(--false)}
.graph-level-score{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;border-left:4px solid #8794a7;padding:2px 10px;margin:0 0 12px}
.graph-level-label{font-size:15px;font-weight:750;color:var(--muted)}
.graph-level-value{font-size:23px;font-weight:800}
.graph-level-meta{font-size:13px;color:var(--muted)}
.counter-label{font-size:15px;font-weight:700;color:var(--muted)}
.counter-value{font-size:32px;line-height:1.15;font-weight:800;margin-top:3px}
.selected-counter.recovered .counter-value{color:var(--recovered)}.selected-counter.incorrect .counter-value{color:var(--false)}
.frequency-heading{font-size:16px;font-weight:750;margin:14px 0 6px}
.frequency-table{width:100%;border-collapse:collapse;font-size:15px}
.frequency-table th,.frequency-table td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line)}
.frequency-table th{background:#edf1f6;color:#25344a}
.frequency-table td:nth-child(2),.frequency-table td:nth-child(3){white-space:nowrap}
.frequency-table tr.zero-return td{background:#fde8e8;color:#8f1c2b}
.frequency-status{font-size:15px;color:var(--muted);padding:10px 0}
.cora-results{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
.cora-results h2{margin:0 0 6px;font-size:21px}
.cora-results h3{margin:20px 0 7px;font-size:17px}
.cora-note{color:var(--muted);margin:0 0 12px}
.cora-chart{display:block;width:100%;height:300px;max-width:100%;border:1px solid var(--line);border-radius:8px}
.cora-table-wrap{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:6px}
.cora-table{width:100%;border-collapse:collapse;font-size:14px}
.cora-table th,.cora-table td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
.cora-table th{position:sticky;top:0;background:#edf1f6;color:#25344a}
.cora-table td:nth-child(2),.cora-table td:nth-child(3){font-variant-numeric:tabular-nums}
.cora-fp-details{max-width:650px;white-space:normal}
.cora-fp-details summary{cursor:pointer}
.cora-status{color:var(--muted);padding:12px 0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-top:12px}
.panel h2{font-size:17px;margin:0 0 10px}
.comparison{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.method{border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.method h3{margin:0 0 4px;font-size:16px}
.method .method-meta{font-size:13px;color:var(--muted);margin-bottom:7px}
.method .method-metrics{font-weight:700}
.status{font-size:13px;color:var(--muted);margin-top:8px}
.status{text-align:center;font-size:16px;font-weight:700}
.no-data{color:#8f1c2b}
@media(max-width:1050px){main{padding:15px}.controls{grid-template-columns:repeat(6,minmax(150px,1fr));overflow-x:auto}.cards{grid-template-columns:repeat(5,minmax(150px,1fr));overflow-x:auto}.graphs{grid-template-columns:repeat(3,minmax(310px,1fr))}}
@media(max-width:720px){header{padding:12px 16px}main{padding:12px}.controls{grid-template-columns:repeat(2,minmax(145px,1fr))}.cards{grid-template-columns:repeat(5,minmax(150px,1fr))}.selected-counters{gap:7px}.selected-counter{padding:4px 8px}.counter-label{font-size:13px}.counter-value{font-size:27px}.graph-card svg,.graph-card .unavailable{min-height:0;height:300px}}
</style>
</head>
<body>
<header><h1>Network Stability with HypDiff</h1></header>
<main>
<section class="controls" aria-label="Saved experiment selectors">
  <div class="selector"><label for="datasetSelect">Dataset</label><select id="datasetSelect"><option value="IMDB-BINARY">IMDB-BINARY</option><option value="SBM">SBM</option><option value="PROTEINS">PROTEINS</option><option value="BA">BA</option></select></div>
  <div class="selector" id="graphSelector"><label for="graphSelect">Graph</label><select id="graphSelect"></select></div>
  <div class="selector"><label for="attackSelect">Attack type</label><select id="attackSelect"><option>Random removal</option><option>Targeted removal</option></select></div>
  <div class="selector"><label for="damageSelect">Requested damage</label><select id="damageSelect"><option value="0.1">10%</option><option value="0.2">20%</option><option value="0.3">30%</option><option value="0.4">40%</option><option value="0.5">50%</option></select></div>
  <div class="selector" id="maskSelector"><label for="maskSelect">Saved mask</label><select id="maskSelect"></select></div>
  <div class="selector" id="reconSelector"><label for="reconSelect">Partial recovery sample</label><select id="reconSelect"></select></div>
</section>
<section class="cards" aria-label="Selected case metrics">
  <article class="metric"><div class="metric-label">Requested damage</div><div class="metric-value" id="requestedMetric">-</div></article>
  <article class="metric"><div class="metric-label">Actual damage</div><div class="metric-value" id="actualMetric">-</div></article>
  <article class="metric"><div class="metric-label">Removed edges</div><div class="metric-value" id="removedMetric">-</div></article>
  <article class="metric"><div class="metric-label">Reconstructions</div><div class="metric-value" id="reconMetric">-</div></article>
  <article class="metric stability"><div class="metric-label">Final Stability Score</div><div class="metric-value" id="finalStabilityMetric">Unavailable</div><div class="metric-meta" id="finalStabilityMeta">No saved mask scores for this setting.</div></article>
</section>
<div class="status no-data" id="caseStatus" aria-live="polite"></div>
<section class="graphs" aria-label="Original, damaged, and recovered graph views">
  <article class="graph-card"><h2>Original</h2><svg id="originalGraph" viewBox="0 0 600 500" role="img" aria-label="Original graph"></svg><div class="unavailable" id="originalUnavailable" hidden></div></article>
  <article class="graph-card"><h2>Damaged</h2><svg id="damagedGraph" viewBox="0 0 600 500" role="img" aria-label="Damaged graph"></svg><div class="unavailable" id="damagedUnavailable" hidden></div></article>
  <article class="graph-card"><h2>After recovery</h2><svg id="completedGraph" viewBox="0 0 600 500" role="img" aria-label="Graph after recovery"></svg><div class="unavailable" id="completedUnavailable" hidden></div></article>
</section>
<section class="legend" aria-label="Shared graph legend">
  <span class="key"><i class="swatch removed"></i>Removed or still missing</span>
  <span class="key"><i class="swatch recovered"></i>Correctly recovered</span>
  <span class="key"><i class="swatch false"></i>Incorrect addition</span>
</section>
<section class="selected-card" aria-label="Selected reconstruction summary">
  <h2>Selected reconstruction</h2>
  <div class="graph-level-score">
    <span class="graph-level-label">Graph Level Stability</span>
    <span class="graph-level-value" id="graphLevelMetric">Unavailable</span>
    <span class="graph-level-meta" id="graphLevelMeta">Selected mask · across all saved reconstructions</span>
  </div>
  <div class="selected-counters">
    <article class="selected-counter"><div class="counter-label">Sample</div><div class="counter-value" id="sampleValue">-</div></article>
    <article class="selected-counter recovered"><div class="counter-label">Removed edges recovered</div><div class="counter-value" id="recoveredValue">-</div></article>
    <article class="selected-counter incorrect"><div class="counter-label">Incorrect edges added</div><div class="counter-value" id="incorrectValue">-</div></article>
  </div>
  <h3 class="frequency-heading">Edge Level Recovery</h3>
  <table class="frequency-table">
    <thead><tr><th>Removed edge</th><th>Returned</th><th>Frequency</th></tr></thead>
    <tbody id="frequencyBody"></tbody>
  </table>
  <div class="frequency-status" id="frequencyStatus"></div>
</section>
<section class="chart-panel" aria-label="Final Stability Score by Damage Level">
  <h2>Final Stability Score by Damage Level</h2>
  <p class="chart-caption" id="chartCaption"></p>
  <div class="chart-wrap">
    <svg id="stabilityChart" viewBox="0 0 1500 360" role="img" aria-label="Final Stability Score from zero to one hundred percent by requested damage level"></svg>
    <div class="chart-unavailable" id="chartUnavailable" hidden></div>
  </div>
</section>
<section class="cora-results" id="coraResults" aria-label="Cora edge-level results" hidden>
  <h2>Cora edge-level recovery</h2>
  <p class="cora-note" id="coraNote">30% original-edge removal is the primary evaluation point. Masks are cumulative within each attack series. Only per-edge recovery and false-positive additions are shown.</p>
  <div class="cora-status" id="coraStatus" aria-live="polite"></div>
  <h3 id="coraHistogramHeading">Removed-edge recovery histogram</h3>
  <svg class="cora-chart" id="coraHistogramSvg" viewBox="0 0 900 300" role="img" aria-label="Histogram of removed edges by recovery count"></svg>
  <div class="cora-table-wrap">
    <table class="cora-table"><thead><tr><th>Recovered appearances (r/K)</th><th>Removed edges</th><th>Percentage</th></tr></thead><tbody id="coraHistogramBody"></tbody></table>
  </div>
  <h3 id="coraEdgesHeading">Removed edges</h3>
  <div class="cora-table-wrap">
    <table class="cora-table"><thead><tr><th>Removed edge endpoints</th><th>Recovered</th></tr></thead><tbody id="coraEdgesBody"></tbody></table>
  </div>
  <h3>False-positive additions (reported separately)</h3>
  <div class="cora-table-wrap">
    <table class="cora-table"><thead><tr><th>Reconstruction</th><th>Added false-positive edges</th><th>Endpoints</th></tr></thead><tbody id="coraFalsePositivesBody"></tbody></table>
  </div>
</section>
</main>
<script id="demo-data" type="application/octet-stream">__PAYLOAD__</script>
<script>
"use strict";
let DATA;
const byId = id => document.getElementById(id);
const edgeKey = edge => [...edge].sort().join("\u0000");
const setOf = edges => new Set(edges.map(edgeKey));
const shortId = id => id.includes(":") ? id.split(":").pop() : id;
const NS = "http://www.w3.org/2000/svg";
const DAMAGE_LEVELS = [0.1,0.2,0.3,0.4,0.5];

function svgEl(tag, attrs={}) {
  const el=document.createElementNS(NS,tag);
  for(const [key,value] of Object.entries(attrs)) el.setAttribute(key,String(value));
  return el;
}
function positions(nodeIds) {
  const cx=300, cy=250, radius=Math.min(202,132+nodeIds.length*3);
  return nodeIds.map((id,i)=>{
    const angle=-Math.PI/2+2*Math.PI*i/nodeIds.length;
    return {id,x:cx+radius*Math.cos(angle),y:cy+radius*Math.sin(angle)};
  });
}
function renderGraph(svgId, nodeIds, edgeGroups) {
  const svg=byId(svgId); svg.replaceChildren();
  const pos=positions(nodeIds), lookup=new Map(pos.map(p=>[p.id,p]));
  for(const group of edgeGroups) for(const edge of group.edges) {
    const a=lookup.get(edge[0]),b=lookup.get(edge[1]); if(!a||!b) continue;
    const line=svgEl("line",{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:group.kind});
    const title=svgEl("title"); title.textContent=`${group.label}: ${edge[0]} - ${edge[1]}`; line.appendChild(title); svg.appendChild(line);
  }
  for(const p of pos) {
    const circle=svgEl("circle",{cx:p.x,cy:p.y,r:19,class:"node"});
    const title=svgEl("title"); title.textContent=p.id; circle.appendChild(title); svg.appendChild(circle);
    const text=svgEl("text",{x:p.x,y:p.y+4,class:"node-label"}); text.textContent=shortId(p.id); svg.appendChild(text);
  }
}
function graphState(available, svgId, emptyId, nodeIds, edgeGroups, message) {
  byId(emptyId).hidden=available;
  byId(svgId).hidden=!available;
  byId(emptyId).textContent=message;
  if(available) renderGraph(svgId,nodeIds,edgeGroups);
}
function selectedGraph() {
  return DATA.graphs.find(graph=>graph.dataset===byId("datasetSelect").value &&
    graph.graph_id===byId("graphSelect").value);
}
function matchingCases() {
  const graph=selectedGraph();
  if(!graph) return [];
  return graph.cases.filter(c=>c.attack===byId("attackSelect").value &&
    Math.abs(c.requested_damage-Number(byId("damageSelect").value))<1e-9);
}
function selectedCase() {
  const mask=Number(byId("maskSelect").value);
  return matchingCases().find(c=>c.mask_index===mask);
}
function graphLevelStability(c) {
  const removed=c?.removed_edges||[];
  const reconstructions=c?.partial?.reconstructions;
  if(Array.isArray(reconstructions)&&reconstructions.length&&removed.length) {
    const predictionSets=reconstructions.map(setOf);
    return removed.reduce((sum,edge)=>{
      const key=edgeKey(edge);
      return sum+predictionSets.reduce((count,prediction)=>count+Number(prediction.has(key)),0)/predictionSets.length;
    },0)/removed.length;
  }
  const saved=c?.partial?.score;
  return typeof saved==="number"&&Number.isFinite(saved)?saved:null;
}
function averageGraphLevel(cases) {
  if(!cases.length) return {score:null,maskCount:0,missingCount:0};
  const scores=cases.map(graphLevelStability);
  const missingCount=scores.filter(score=>score===null).length;
  return {
    score:missingCount?null:scores.reduce((sum,score)=>sum+score,0)/scores.length,
    maskCount:cases.length,
    missingCount
  };
}
function updateFinalStability() {
  const cases=matchingCases();
  const average=averageGraphLevel(cases);
  byId("finalStabilityMetric").textContent=average.score===null
    ?"Unavailable":(100*average.score).toFixed(2)+"%";
  if(!average.maskCount) {
    byId("finalStabilityMeta").textContent="No saved masks for this setting.";
  } else if(average.missingCount) {
    byId("finalStabilityMeta").textContent=`Unavailable · saved scores missing for ${average.missingCount} of ${average.maskCount} masks.`;
  } else if(byId("attackSelect").value==="Targeted removal"&&average.maskCount===1) {
    byId("finalStabilityMeta").textContent="1 targeted mask";
  } else {
    const attack=byId("attackSelect").value==="Targeted removal"?"targeted":"random";
    byId("finalStabilityMeta").textContent=`Average across ${average.maskCount} ${attack} ${average.maskCount===1?"mask":"masks"}.`;
  }
}
function fillReconstructions(c,preferredIndex=0) {
  const selector=byId("reconSelect");
  selector.replaceChildren();
  if(!c) {
    selector.add(new Option("No saved result",""));
    selector.disabled=true;
    return;
  }
  const count=c.partial.sample_metrics.length;
  for(let i=0;i<count;i++) selector.add(new Option(`Sample ${i+1} of ${count}`,String(i)));
  selector.disabled=count<2;
  const index=Number(preferredIndex);
  selector.value=Number.isInteger(index)&&index>=0&&index<count?String(index):"0";
}
function fillMasks(preferredMask,preferredRecon=0) {
  const selector=byId("maskSelect");
  selector.replaceChildren();
  const cases=matchingCases();
  if(!cases.length) {
    selector.add(new Option("No saved mask",""));
    selector.disabled=true;
    fillReconstructions(null);
    drawCase();
    return;
  }
  for(const c of cases) {
    const seed=c.mask_seed===null||c.mask_seed===undefined?"":" | seed "+c.mask_seed;
    selector.add(new Option(`Mask ${c.mask_index}${seed}`,String(c.mask_index)));
  }
  selector.disabled=cases.length===1;
  const selected=cases.find(c=>c.mask_index===Number(preferredMask))||cases[0];
  selector.value=String(selected.mask_index);
  fillReconstructions(selected,preferredRecon);
  drawCase();
}
function edgeLabel(edge) {
  return `${shortId(edge[0])} - ${shortId(edge[1])}`;
}
function updateEdgeDetails(c) {
  const body=byId("frequencyBody");
  body.replaceChildren();
  byId("frequencyStatus").textContent="";
  if(c.partial.reconstructions===null) {
    byId("frequencyStatus").textContent="Per-edge return frequencies are unavailable in the saved artifacts for this case.";
    return;
  }
  const reconstructions=c.partial.reconstructions;
  if(!reconstructions.length) {
    byId("frequencyStatus").textContent="No reconstruction edge sets are saved for this case.";
    return;
  }
  for(const edge of c.removed_edges) {
    const returned=reconstructions.reduce((count,edges)=>count+Number(setOf(edges).has(edgeKey(edge))),0);
    const row=document.createElement("tr");
    const edgeCell=document.createElement("td");
    edgeCell.textContent=edgeLabel(edge);
    edgeCell.title=edge.join(" - ");
    const returnedCell=document.createElement("td");
    returnedCell.textContent=`${returned}/${reconstructions.length}`;
    returnedCell.title=`Returned in ${returned} of ${reconstructions.length} saved reconstructions`;
    const frequencyCell=document.createElement("td");
    frequencyCell.textContent=`${(100*returned/reconstructions.length).toFixed(2)}%`;
    if(returned===0) row.className="zero-return";
    row.append(edgeCell,returnedCell,frequencyCell);
    body.appendChild(row);
  }
  if(!c.removed_edges.length) byId("frequencyStatus").textContent="No removed edges in this case.";
}
function drawCoraResults() {
  const active=byId("datasetSelect").value==="Cora";
  for(const id of ["graphSelector","maskSelector","reconSelector"]) byId(id).hidden=active;
  for(const selector of [".cards",".graphs",".legend",".selected-card",".chart-panel"]) {
    document.querySelector(selector).hidden=active;
  }
  byId("caseStatus").hidden=active;
  byId("coraResults").hidden=!active;
  if(!active) return;

  const histogramBody=byId("coraHistogramBody");
  const edgesBody=byId("coraEdgesBody");
  const falsePositivesBody=byId("coraFalsePositivesBody");
  const svg=byId("coraHistogramSvg");
  histogramBody.replaceChildren();
  edgesBody.replaceChildren();
  falsePositivesBody.replaceChildren();
  svg.replaceChildren();

  const attack=byId("attackSelect").value==="Targeted removal"
    ?"targeted_degree_product":"uniform_random";
  const damage=Number(byId("damageSelect").value);
  const series=DATA.cora_results?.attack_series.find(row=>row.attack_type===attack);
  const level=series?.damage_levels.find(row=>
    Math.abs(row.damage_fraction_requested-damage)<1e-9
  );
  if(!level) {
    byId("coraStatus").textContent=DATA.cora_results
      ?"No saved Cora result for this attack and damage level."
      :"No completed Cora edge-recovery run is saved. Run scripts\\run_cora_experiment.py to create one.";
    svg.hidden=true;
    return;
  }
  byId("coraStatus").textContent=`${attack==="uniform_random"?"Random":"Targeted"} removal · ${Math.round(100*damage)}% requested damage · ${level.edges_removed}/${level.original_edge_count} original edges removed · ${DATA.cora_results.reconstruction_count} reconstructions`;
  const recovery=level.removed_edge_recovery;
  const bins=recovery.histogram;
  byId("coraHistogramHeading").textContent=`Removed-edge recovery histogram · ${recovery.reconstruction_count} reconstructions`;
  byId("coraEdgesHeading").textContent=`Removed edges · ${level.edges_removed} endpoints`;

  for(const bin of bins) {
    const row=document.createElement("tr");
    for(const text of [
      `${bin.recovered_count}/${recovery.reconstruction_count}`,
      String(bin.edge_count),
      `${bin.percentage.toFixed(2)}%`
    ]) {
      const cell=document.createElement("td");
      cell.textContent=text;
      row.appendChild(cell);
    }
    histogramBody.appendChild(row);
  }

  const left=70,right=875,top=20,bottom=235;
  const maxPercent=Math.max(1,...bins.map(bin=>bin.percentage));
  const slot=(right-left)/Math.max(1,bins.length);
  bins.forEach((bin,index)=>{
    const x=left+index*slot+slot*0.18;
    const height=bin.percentage/maxPercent*(bottom-top);
    const bar=svgEl("rect",{
      x,y:bottom-height,width:slot*0.64,height,
      fill:"#2864a8",rx:3
    });
    const title=svgEl("title");
    title.textContent=`Recovered ${bin.recovered_count} of ${recovery.reconstruction_count}: ${bin.edge_count} edges (${bin.percentage.toFixed(2)}%)`;
    bar.appendChild(title);
    svg.appendChild(bar);
    const label=svgEl("text",{x:x+slot*0.32,y:bottom+25,"text-anchor":"middle",class:"axis-text"});
    label.textContent=String(bin.recovered_count);
    svg.appendChild(label);
    if(bin.edge_count) {
      const value=svgEl("text",{
        x:x+slot*0.32,y:Math.max(top+13,bottom-height-7),
        "text-anchor":"middle",class:"axis-text"
      });
      value.textContent=String(bin.edge_count);
      svg.appendChild(value);
    }
  });
  svg.appendChild(svgEl("line",{x1:left,y1:bottom,x2:right,y2:bottom,class:"grid-line"}));
  const xTitle=svgEl("text",{x:(left+right)/2,y:285,"text-anchor":"middle",class:"axis-title"});
  xTitle.textContent="Recovered appearances per removed edge (r)";
  svg.appendChild(xTitle);

  for(const edge of recovery.edges) {
    const row=document.createElement("tr");
    const endpoints=document.createElement("td");
    endpoints.textContent=`${edge.source} - ${edge.target}`;
    const count=document.createElement("td");
    count.textContent=edge.r_over_K;
    row.append(endpoints,count);
    edgesBody.appendChild(row);
  }
  for(const reconstruction of level.false_positive_additions) {
    const row=document.createElement("tr");
    const index=document.createElement("td");
    index.textContent=String(reconstruction.reconstruction);
    const count=document.createElement("td");
    count.textContent=String(reconstruction.count);
    const endpoints=document.createElement("td");
    const details=document.createElement("details");
    details.className="cora-fp-details";
    const summary=document.createElement("summary");
    summary.textContent=`Show ${reconstruction.count} added edge endpoints`;
    const edgeList=document.createElement("div");
    edgeList.textContent=reconstruction.edges.map(edge=>`${edge[0]} - ${edge[1]}`).join(", ");
    details.append(summary,edgeList);
    if(reconstruction.count) endpoints.appendChild(details);
    else endpoints.textContent="None";
    row.append(index,count,endpoints);
    falsePositivesBody.appendChild(row);
  }
}
function drawCase(){
  const isCora=byId("datasetSelect").value==="Cora";
  drawCoraResults();
  if(isCora) return;
  const c=selectedCase(), reconIndex=Math.max(0,Number(byId("reconSelect").value)||0);
  const dimensionsUnavailable="No saved result for this graph, attack, and requested damage.";
  updateFinalStability();
  if(!c) {
    for(const id of ["requestedMetric","actualMetric","removedMetric","reconMetric"]) byId(id).textContent="-";
    const dataset=byId("datasetSelect").value;
    const hasDatasetCases=DATA.graphs.some(graph=>graph.dataset===dataset);
    const message=hasDatasetCases?dimensionsUnavailable:`No saved cases are available for ${dataset||"the selected dataset"}.`;
    byId("caseStatus").textContent=message;
    byId("graphLevelMetric").textContent="Unavailable";
    byId("graphLevelMeta").textContent="Selected mask · across all saved reconstructions · score unavailable.";
    byId("sampleValue").textContent="-";
    byId("recoveredValue").textContent="-";
    byId("incorrectValue").textContent="-";
    byId("frequencyBody").replaceChildren();
    byId("frequencyStatus").textContent="No saved removed-edge return data.";
    graphState(false,"originalGraph","originalUnavailable",[],[],message);
    graphState(false,"damagedGraph","damagedUnavailable",[],[],message);
    graphState(false,"completedGraph","completedUnavailable",[],[],message);
    return;
  }
  const sample=c.partial.sample_metrics[reconIndex]||c.partial.sample_metrics[0];
  const count=c.partial.sample_metrics.length;
  const original=setOf(c.original_edges), damaged=setOf(c.damaged_edges), removed=setOf(c.removed_edges);
  const prediction=c.partial.reconstructions===null?null:c.partial.reconstructions[reconIndex];
  byId("requestedMetric").textContent=(100*c.requested_damage).toFixed(2)+"%";
  byId("actualMetric").textContent=`${c.removed_edges.length}/${c.original_edges.length} (${(100*c.actual_damage).toFixed(2)}%)`;
  byId("removedMetric").textContent=String(c.removed_edges.length);
  byId("reconMetric").textContent=String(count);
  const graphScore=graphLevelStability(c);
  byId("graphLevelMetric").textContent=graphScore===null?"Unavailable":(100*graphScore).toFixed(2)+"%";
  byId("graphLevelMeta").textContent=graphScore===null
    ?"Selected mask · across all saved reconstructions · score unavailable in saved data."
    :"Selected mask · across all saved reconstructions";
  byId("sampleValue").textContent=`${reconIndex+1} of ${count}`;
  byId("recoveredValue").textContent=`${sample.recovered} of ${c.removed_edges.length}`;
  byId("incorrectValue").textContent=String(sample.false_positives);
  byId("caseStatus").textContent=c.graph_available?"":`${c.dataset}: summary metrics are saved, but individual predicted edge sets are not; the recovery graph view is unavailable.`;
  graphState(true,"originalGraph","originalUnavailable",c.node_ids,[
    {kind:"surviving",label:"Surviving edge",edges:c.damaged_edges},
    {kind:"removed",label:"Removed edge",edges:c.removed_edges}
  ],"");
  graphState(true,"damagedGraph","damagedUnavailable",c.node_ids,[
    {kind:"surviving",label:"Surviving edge",edges:c.damaged_edges},
    {kind:"removed ghost",label:"Removed edge, shown for reference",edges:c.removed_edges}
  ],"");
  if(prediction===null) {
    graphState(false,"completedGraph","completedUnavailable",[],[],`${c.dataset} saved metrics do not include reconstructed edge sets.`);
  } else {
    const predicted=setOf(prediction);
    const completed=[...new Map([...c.damaged_edges,...prediction].map(e=>[edgeKey(e),e])).values()];
    const groups=[
      {kind:"surviving",label:"Surviving edge",edges:completed.filter(e=>damaged.has(edgeKey(e)))},
      {kind:"recovered",label:"Correctly recovered edge",edges:completed.filter(e=>removed.has(edgeKey(e))&&predicted.has(edgeKey(e)))},
      {kind:"removed",label:"Removed edge still missing",edges:completed.filter(e=>removed.has(edgeKey(e))&&!predicted.has(edgeKey(e)))},
      {kind:"falseadd",label:"Incorrect addition",edges:completed.filter(e=>!original.has(edgeKey(e)))}
    ];
    graphState(true,"completedGraph","completedUnavailable",c.node_ids,groups,"");
  }
  updateEdgeDetails(c);
}
function drawStabilityChart() {
  const svg=byId("stabilityChart");
  svg.replaceChildren();
  if(byId("datasetSelect").value==="Cora") {
    svg.hidden=true;
    byId("chartUnavailable").hidden=true;
    return;
  }
  const graph=selectedGraph();
  const attack=byId("attackSelect").value;
  const selectedDamage=Number(byId("damageSelect").value);
  const cases=graph?graph.cases.filter(c=>c.attack===attack):[];
  const points=DAMAGE_LEVELS.map(damage=>{
    const masks=cases.filter(c=>Math.abs(c.requested_damage-damage)<1e-9);
    const average=averageGraphLevel(masks);
    return {...average,damage};
  });
  const available=points.some(point=>point.score!==null);
  svg.hidden=!available;
  byId("chartUnavailable").hidden=available;
  if(!available) {
    byId("chartUnavailable").textContent=`No saved Final Stability Score sweep is available for ${graph?graph.graph_id:"the selected graph"} under ${attack}.`;
    byId("chartCaption").textContent="No saved mask-level stability scores are available for this graph and attack type.";
    return;
  }
  const missing=points.filter(point=>point.score===null).map(point=>`${Math.round(point.damage*100)}%`);
  const incomplete=points.filter(point=>point.missingCount>0).map(point=>
    `${Math.round(point.damage*100)}% (${point.missingCount} of ${point.maskCount} mask scores unavailable)`
  );
  const sweepCaption=attack==="Targeted removal"
    ? "Targeted removal averages Graph Level Stability across saved targeted masks."
    : "Random removal averages Graph Level Stability equally across saved random masks.";
  byId("chartCaption").textContent=`${sweepCaption} The Final Stability Score card uses the same calculation for the selected damage level.${missing.length?` Missing levels are gaps: ${missing.join(", ")}.`:""}${incomplete.length?` Missing mask scores: ${incomplete.join("; ")}.`:""}`;

  const left=115,right=1470,top=30,bottom=292,width=right-left,height=bottom-top;
  const xAt=score=>left+score*width;
  const yAt=damage=>bottom-(damage-DAMAGE_LEVELS[0])/(DAMAGE_LEVELS.at(-1)-DAMAGE_LEVELS[0])*height;
  for(const percent of [0,25,50,75,100]) {
    const x=xAt(percent/100);
    const grid=svgEl("line",{x1:x,y1:top,x2:x,y2:bottom,class:"grid-line"});
    svg.appendChild(grid);
    const label=svgEl("text",{x,y:bottom+28,"text-anchor":"middle",class:"axis-text"});
    label.textContent=`${percent}%`;
    svg.appendChild(label);
  }
  DAMAGE_LEVELS.forEach(damage=>{
    const y=yAt(damage);
    svg.appendChild(svgEl("line",{x1:left,y1:y,x2:right,y2:y,class:"grid-line"}));
  });
  const selectedIndex=DAMAGE_LEVELS.findIndex(level=>Math.abs(level-selectedDamage)<1e-9);
  if(selectedIndex>=0) {
    const selectedY=yAt(selectedDamage);
    svg.appendChild(svgEl("line",{x1:left,y1:selectedY,x2:right,y2:selectedY,class:"selected-level"}));
  }
  for(let index=0;index<points.length-1;index++) {
    if(points[index].score===null||points[index+1].score===null) continue;
    svg.appendChild(svgEl("line",{
      x1:xAt(points[index].score),y1:yAt(points[index].damage),
      x2:xAt(points[index+1].score),y2:yAt(points[index+1].damage),
      class:"chart-line"
    }));
  }
  points.forEach((point,index)=>{
    if(point.score===null) return;
    const selected=Math.abs(point.damage-selectedDamage)<1e-9;
    const circle=svgEl("circle",{
      cx:xAt(point.score),cy:yAt(point.damage),r:selected?9:7,
      class:selected?"chart-point selected":"chart-point"
    });
    const title=svgEl("title");
    const maskKind=attack==="Targeted removal"?"targeted":"random";
    title.textContent=`Final Stability Score ${(100*point.score).toFixed(2)}% at ${Math.round(point.damage*100)}% requested damage; average across ${point.maskCount} saved ${maskKind} ${point.maskCount===1?"mask":"masks"}.`;
    circle.appendChild(title);
    svg.appendChild(circle);
  });
  svg.appendChild(svgEl("line",{x1:left,y1:bottom,x2:right,y2:bottom,class:"grid-line"}));
  DAMAGE_LEVELS.forEach((damage,index)=>{
    const y=yAt(damage);
    const label=svgEl("text",{
      x:left-15,y:y+5,"text-anchor":"end",
      class:Math.abs(damage-selectedDamage)<1e-9?"axis-text selected-tick":"axis-text"
    });
    label.textContent=`${Math.round(damage*100)}%`;
    svg.appendChild(label);
  });
  const yTitle=svgEl("text",{
    x:22,y:(top+bottom)/2,"text-anchor":"middle",class:"axis-title",
    transform:`rotate(-90 22 ${(top+bottom)/2})`
  });
  yTitle.textContent="Requested damage (%)";
  svg.appendChild(yTitle);
  const xTitle=svgEl("text",{
    x:(left+right)/2,y:bottom+58,"text-anchor":"middle",class:"axis-title"
  });
  xTitle.textContent="Final Stability Score (%)";
  svg.appendChild(xTitle);
}
function refreshSelectors(preferred=null){
  if(!preferred||typeof preferred!=="object"||!("attack" in preferred)) {
    preferred={
      attack:byId("attackSelect").value,
      damage:Number(byId("damageSelect").value),
      mask:byId("maskSelect").value,
      recon:byId("reconSelect").value
    };
  }
  const graph=selectedGraph();
  if(graph) {
    let attack=preferred.attack;
    if(!graph.cases.some(c=>c.attack===attack)) {
      attack=["Random removal","Targeted removal"].find(value=>graph.cases.some(c=>c.attack===value))||"";
    }
    if(attack) byId("attackSelect").value=attack;
    const availableDamage=DAMAGE_LEVELS.filter(level=>graph.cases.some(c=>
      c.attack===byId("attackSelect").value&&Math.abs(c.requested_damage-level)<1e-9
    ));
    const damage=availableDamage.includes(preferred.damage)?preferred.damage:availableDamage[0];
    if(damage!==undefined) byId("damageSelect").value=String(damage);
    fillMasks(preferred.mask,preferred.recon);
  } else {
    fillMasks();
  }
  drawStabilityChart();
}
function populateGraphOptions(preferredGraphId=null){
  const selector=byId("graphSelect");
  selector.replaceChildren();
  const graphs=DATA.graphs
    .filter(graph=>graph.dataset===byId("datasetSelect").value)
    .sort((left,right)=>left.graph_id.localeCompare(right.graph_id));
  if(!graphs.length) {
    selector.add(new Option("No saved graphs",""));
    selector.disabled=true;
    selector.value="";
    return graphs;
  }
  selector.disabled=false;
  for(const graph of graphs) selector.add(new Option(graph.graph_id,graph.graph_id));
  selector.value=graphs.some(graph=>graph.graph_id===preferredGraphId)
    ?preferredGraphId
    :graphs[0].graph_id;
  return graphs;
}
function initialize(){
  const datasets=["IMDB-BINARY","SBM","PROTEINS","BA","Cora"];
  const datasetSelector=byId("datasetSelect");
  datasetSelector.replaceChildren(...datasets.map(dataset=>new Option(dataset,dataset)));
  datasetSelector.value=DATA.graphs.some(graph=>graph.dataset==="PROTEINS")
    ?"PROTEINS"
    :datasets.find(dataset=>DATA.graphs.some(graph=>graph.dataset===dataset))||datasets[0];
  populateGraphOptions(datasetSelector.value==="PROTEINS"?"PROTEINS-0080":null);
  datasetSelector.addEventListener("change",()=>{
    const preferred={
      attack:byId("attackSelect").value,
      damage:datasetSelector.value==="Cora"?0.3:Number(byId("damageSelect").value),
      mask:byId("maskSelect").value,
      recon:byId("reconSelect").value
    };
    populateGraphOptions();
    refreshSelectors(preferred);
  });
  byId("graphSelect").addEventListener("change",()=>refreshSelectors());
  for(const id of ["attackSelect","damageSelect"]) byId(id).addEventListener("change",()=>refreshSelectors());
  byId("maskSelect").addEventListener("change",()=>{
    fillReconstructions(selectedCase(),byId("reconSelect").value);
    drawCase();
  });
  byId("reconSelect").addEventListener("change",drawCase);
  byId("attackSelect").value="Targeted removal";
  byId("damageSelect").value="0.1";
  refreshSelectors();
  window.demoSnapshot=()=>({
    dataset:byId("datasetSelect").value,
    graph:byId("graphSelect").value,
    graphOptions:[...byId("graphSelect").options].map(option=>option.value),
    attack:byId("attackSelect").value,
    damage:byId("damageSelect").value,
    mask:byId("maskSelect").value,
    sample:byId("reconSelect").value,
    requested:byId("requestedMetric").textContent,
    actual:byId("actualMetric").textContent,
    removed:byId("removedMetric").textContent,
    reconstructions:byId("reconMetric").textContent,
    finalStability:byId("finalStabilityMetric").textContent,
    finalStabilityMeta:byId("finalStabilityMeta").textContent,
    graphLevelStability:byId("graphLevelMetric").textContent,
    graphLevelMeta:byId("graphLevelMeta").textContent,
    sampleLabel:byId("sampleValue").textContent,
    recovered:byId("recoveredValue").textContent,
    incorrect:byId("incorrectValue").textContent,
    frequencies:[...byId("frequencyBody").rows].map(row=>[...row.cells].map(cell=>cell.textContent)),
    completedGraphAvailable:!byId("completedGraph").hidden,
    status:byId("caseStatus").textContent,
    chartCaption:byId("chartCaption").textContent,
    chartVisible:!byId("stabilityChart").hidden,
    chartPointCount:byId("stabilityChart").querySelectorAll("circle.chart-point").length,
    chartPointTitles:[...byId("stabilityChart").querySelectorAll("circle.chart-point title")].map(title=>title.textContent)
  });
}
async function loadPayload(){
  const base64=document.getElementById("demo-data").textContent.trim();
  const bytes=Uint8Array.from(atob(base64),character=>character.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  DATA=JSON.parse(await new Response(stream).text());
  initialize();
}
window.demoReady=loadPayload().catch(error=>{
  document.querySelector("main").textContent="Unable to load the embedded saved demo data: "+error.message;
  throw error;
});
</script>
<style>
svg line{fill:none;stroke-linecap:round}.surviving{stroke:var(--survive);stroke-width:2}.removed{stroke:var(--removed);stroke-width:2.6;stroke-dasharray:8 6}.ghost{opacity:.55}.recovered{stroke:var(--recovered);stroke-width:4}.falseadd{stroke:var(--false);stroke-width:3;stroke-dasharray:2 5}
.node{fill:#fff;stroke:#10233f;stroke-width:2.5}.node-label{fill:#10233f;font-size:12px;text-anchor:middle;font-weight:700;pointer-events:none}
</style>
</body>
</html>
"""


def build() -> Path:
    proteins = _load_proteins_cases()
    pilot_thresholds = {
        method: float(value["tau"])
        for method, value in _read_json(IMDB_PILOT / "calibration" / "summary.json")[
            "method_thresholds"
        ].items()
    }
    pilot_summary = _read_json(IMDB_PILOT / "evaluation" / "summary.json")
    for method, value in pilot_thresholds.items():
        _same_metric(
            pilot_summary["frozen_method_thresholds"][method],
            value,
            f"IMDB pilot frozen {method} threshold",
        )
    pilot_cases = _load_imdb_cases(
        IMDB_PILOT / "evaluation" / "graphs",
        pilot_summary,
        pilot_thresholds,
        "IMDB-BINARY pilot",
    )

    expanded_manifest = _read_json(IMDB_EXPANDED / "selection_manifest.json")
    expanded_summary = _read_json(IMDB_EXPANDED / "evaluation" / "summary.json")
    expanded_thresholds = {
        method: float(value)
        for method, value in expanded_manifest["frozen_thresholds"].items()
    }
    for method, value in expanded_thresholds.items():
        _same_metric(
            expanded_summary["frozen_thresholds"][method],
            value,
            f"IMDB expanded frozen {method} threshold",
        )
        _same_metric(
            pilot_thresholds[method],
            value,
            f"IMDB expanded reused {method} threshold",
        )
    expanded_cases = _load_imdb_cases(
        IMDB_EXPANDED / "evaluation" / "graphs",
        expanded_summary,
        expanded_thresholds,
        "IMDB-BINARY expanded",
    )
    synthetic_cases = _load_synthetic_cases()
    cora_results = _load_cora_gui_results()
    cases = proteins + pilot_cases + expanded_cases + synthetic_cases
    if cora_results is not None:
        cases.extend(_cora_graph_marker(cora_results))
    _check_acceptance_cases(cases)
    graphs = {}
    for case in cases:
        if case["dataset"] in ("IMDB-BINARY pilot", "IMDB-BINARY expanded"):
            case["dataset"] = "IMDB-BINARY"
        graph_id = case["graph_id"]
        graph = graphs.setdefault(
            graph_id, {"graph_id": graph_id, "dataset": case["dataset"], "cases": []}
        )
        if graph["dataset"] != case["dataset"]:
            raise ValueError(f"{graph_id}: conflicting dataset labels")
        graph["cases"].append(case)
    for graph in graphs.values():
        graph["cases"].sort(
            key=lambda case: (
                case["attack"] != "Random removal",
                case["requested_damage"],
                case["mask_index"],
            )
        )
    payload = {
        "graphs": sorted(graphs.values(), key=lambda graph: (graph["dataset"], graph["graph_id"])),
        "cora_results": cora_results,
    }
    plain = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    compressed = gzip.compress(plain.encode("utf-8"), compresslevel=9, mtime=0)
    embedded = base64.b64encode(compressed).decode("ascii")
    html = HTML.replace("__PAYLOAD__", embedded)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8", newline="\n")
    print(
        f"Built {OUTPUT.relative_to(ROOT)} with {len(graphs)} graphs, "
        f"{len(cases)} exact saved cases, {len(compressed):,} compressed data bytes."
    )
    return OUTPUT


if __name__ == "__main__":
    print(build())
