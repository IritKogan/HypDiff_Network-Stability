from stability.proteins_fresh_activation_experiment import _ranking_metrics


def test_top_three_reports_guaranteed_and_possible_hits_under_ties():
    inputs = {
        "labels": [1, 0, 1, 0, 0],
        "candidate_keys": (
            frozenset(("a", "b")),
            frozenset(("a", "c")),
            frozenset(("b", "c")),
            frozenset(("c", "d")),
            frozenset(("d", "e")),
        ),
    }

    metrics = _ranking_metrics([0.9, 0.8, 0.8, 0.8, 0.1], inputs)

    assert metrics["removed_edges_guaranteed_in_top_three"] == 1
    assert metrics["removed_edges_possible_in_top_three"] == 2
    assert metrics["top_three_cutoff_tie_count"] == 3


def test_top_three_tied_all_candidates_reports_no_guaranteed_hits():
    inputs = {
        "labels": [1, 0, 1, 0, 0],
        "candidate_keys": (
            frozenset(("a", "b")),
            frozenset(("a", "c")),
            frozenset(("b", "c")),
            frozenset(("c", "d")),
            frozenset(("d", "e")),
        ),
    }

    metrics = _ranking_metrics([0.5, 0.5, 0.5, 0.5, 0.5], inputs)

    assert metrics["removed_edges_guaranteed_in_top_three"] == 0
    assert metrics["removed_edges_possible_in_top_three"] == 2
    assert metrics["top_three_cutoff_tie_count"] == 5
