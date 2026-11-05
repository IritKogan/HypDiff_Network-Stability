import itertools

import numpy as np

from stability.proteins_common_fpr_replay import (
    select_method_threshold,
    summarize_at_threshold,
)


def _graph_result(include_tied_false_positive):
    nodes = ("a", "b", "c", "d")
    candidates = tuple(itertools.combinations(nodes, 2))
    removed = (("a", "b"),)
    original = (("a", "b"), ("b", "c"))
    partial = np.full((8, len(candidates)), 0.1)
    baseline = np.full(len(candidates), 0.1)
    removed_index = candidates.index(removed[0])
    partial[:, removed_index] = 0.9
    baseline[removed_index] = 0.9
    if include_tied_false_positive:
        false_positive_index = candidates.index(("a", "c"))
        partial[:, false_positive_index] = 0.9
        baseline[false_positive_index] = 0.9
    nonedges = np.asarray(
        [
            index
            for index, edge in enumerate(candidates)
            if frozenset(edge) not in {frozenset(e) for e in original}
        ],
        dtype=np.int64,
    )
    case = {
        "attack_type": "targeted_degree_product",
        "damage_fraction_requested": 0.1,
        "removed_edges": [list(removed[0])],
        "mask_index": 0,
        "mask_seed": None,
        "candidate_edges": candidates,
        "removed_indices": np.asarray([removed_index]),
        "nonedge_indices": nonedges,
        "partial_probabilities": partial,
        "baseline_probabilities": baseline,
    }
    random_case = {**case, "attack_type": "uniform_random", "mask_index": 1}
    return {
        "graph_id": "test",
        "node_ids": nodes,
        "original_edges": original,
        "cases": [case, random_case],
    }


def test_method_threshold_selects_maximum_recovery_under_fpr_cap():
    graph = _graph_result(include_tied_false_positive=False)

    selected = select_method_threshold([graph], "partial_recovery")
    rows = summarize_at_threshold(
        [graph],
        {
            "partial_recovery": selected["tau"],
            "no_diffusion_baseline": selected["tau"],
        },
    )

    assert selected["tau"] == 0.9
    assert selected["calibration_book_stability_score"] == 1.0
    assert selected["calibration_realized_fpr"] == 0.0
    assert rows[0]["method_metrics"]["partial_recovery"]["book_stability_score"] == 1.0


def test_equal_probability_false_positive_group_is_included_together():
    graph = _graph_result(include_tied_false_positive=True)

    selected = select_method_threshold([graph], "partial_recovery")
    rows = summarize_at_threshold(
        [graph],
        {
            "partial_recovery": selected["tau"],
            "no_diffusion_baseline": selected["tau"],
        },
    )

    assert selected["status"] == "no_nonzero_recovery_under_fpr_cap"
    assert selected["tau"] > 0.9
    assert selected["calibration_book_stability_score"] == 0.0
    assert selected["calibration_realized_fpr"] == 0.0
    assert rows[0]["method_metrics"]["partial_recovery"]["book_stability_score"] == 0.0
