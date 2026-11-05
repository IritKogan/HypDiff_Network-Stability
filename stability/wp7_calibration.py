from __future__ import annotations

from typing import Any

import numpy as np

from stability.scoring import score_recovery


FALSE_POSITIVE_RATE_CAP = 0.05


def calibration_metrics(graph_results: list[dict[str, Any]], tau: float):
    recovery_scores = []
    false_positive_rates = []
    for graph in graph_results:
        original = {frozenset(edge) for edge in graph["original_edges"]}
        removed = tuple(tuple(edge) for edge in graph["removed_edges"])
        true_nonedges = {
            frozenset(edge) for edge in graph["candidate_edges"]
        } - original
        reconstruction_sets = []
        graph_fprs = []
        for reconstruction in graph["reconstructions"]:
            probabilities = np.asarray(reconstruction["probabilities"])
            predicted = tuple(
                tuple(edge)
                for edge, probability in zip(
                    graph["candidate_edges"], probabilities
                )
                if probability >= tau
            )
            predicted_set = {frozenset(edge) for edge in predicted}
            reconstruction_sets.append(predicted)
            graph_fprs.append(
                len(predicted_set & true_nonedges) / len(true_nonedges)
            )
        recovery_scores.append(
            score_recovery(removed, reconstruction_sets).scenario_score
        )
        false_positive_rates.extend(graph_fprs)
    return {
        "book_stability_score": float(np.mean(recovery_scores)),
        "mean_false_positive_rate": float(np.mean(false_positive_rates)),
    }


def choose_threshold(
    graph_results: list[dict[str, Any]],
    false_positive_rate_cap: float = FALSE_POSITIVE_RATE_CAP,
):
    if not graph_results:
        raise ValueError("calibration requires at least one graph")
    if not 0.0 <= false_positive_rate_cap <= 1.0:
        raise ValueError("false_positive_rate_cap must be in [0, 1]")
    candidates = {0.0, 1.0}
    for graph in graph_results:
        for reconstruction in graph["reconstructions"]:
            candidates.update(float(p) for p in reconstruction["probabilities"])

    feasible = []
    for tau in sorted(candidates):
        metrics = calibration_metrics(graph_results, tau)
        if metrics["mean_false_positive_rate"] <= false_positive_rate_cap:
            feasible.append(
                {
                    "tau": tau,
                    **metrics,
                }
            )
    if not feasible:
        raise ValueError(
            "no calibration threshold meets the false-positive-rate cap; "
            "evaluation must not proceed"
        )
    selected = max(
        feasible,
        key=lambda row: (
            row["book_stability_score"],
            -row["mean_false_positive_rate"],
            row["tau"],
        ),
    )
    return {
        "tau": selected["tau"],
        "selection_objective": (
            "maximize mean book stability score subject to mean false-positive "
            "rate <= cap; ties: lower false-positive rate, then higher tau"
        ),
        "false_positive_rate_cap": false_positive_rate_cap,
        "calibration_book_stability_score": selected["book_stability_score"],
        "calibration_mean_false_positive_rate": selected[
            "mean_false_positive_rate"
        ],
        "candidate_threshold_count": len(candidates),
    }
