from stability.posthoc_baselines import (
    candidate_additions,
    common_neighbor_scores,
    ranking_metrics,
)


def test_common_neighbors_uses_only_damaged_edges():
    damaged = (("a", "b"), ("b", "c"), ("c", "d"))
    pairs = (("a", "c"), ("a", "d"), ("a", "b"))

    assert common_neighbor_scores(damaged, pairs) == [1, 0, 0]


def test_candidate_additions_excludes_surviving_edges():
    original = (("a", "b"), ("b", "c"))
    removed = (("a", "b"),)
    candidates = (("a", "b"), ("b", "c"), ("a", "c"))

    pairs, labels = candidate_additions(original, removed, candidates)

    assert pairs == [("a", "b"), ("a", "c")]
    assert labels == [1, 0]


def test_ranking_metrics_report_guaranteed_and_possible_hits_under_ties():
    metrics = ranking_metrics(
        [0.9, 0.8, 0.8, 0.8, 0.1],
        [1, 0, 1, 0, 0],
    )

    assert metrics["removed_edges_guaranteed_in_top_three"] == 1
    assert metrics["removed_edges_possible_in_top_three"] == 2
    assert metrics["top_three_cutoff_tie_count"] == 3
    assert metrics["competition_ranks"] == {0: 1, 1: 2, 2: 2, 3: 2, 4: 5}
