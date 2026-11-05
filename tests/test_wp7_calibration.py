import pytest

from stability.synthetic_instances import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    generate_synthetic_graph,
    validate_split_seeds,
)
from stability.wp7_calibration import choose_threshold


def _calibration_graph(probability_rows):
    return {
        "original_edges": [["a", "b"], ["b", "c"]],
        "removed_edges": [["a", "b"]],
        "candidate_edges": [["a", "b"], ["a", "c"], ["b", "c"]],
        "reconstructions": [
            {"probabilities": row} for row in probability_rows
        ],
    }


def test_synthetic_calibration_and_evaluation_instances_are_disjoint():
    validate_split_seeds()
    calibration = [
        generate_synthetic_graph(seed, "calibration") for seed in CALIBRATION_SEEDS
    ]
    evaluation = [
        generate_synthetic_graph(seed, "evaluation") for seed in EVALUATION_SEEDS
    ]
    assert len({graph.graph_id for graph in calibration + evaluation}) == 8
    assert len({graph.edges for graph in calibration + evaluation}) == 8


def test_threshold_maximizes_stability_subject_to_false_positive_cap():
    graphs = [_calibration_graph([[0.8, 0.4, 0.9], [0.7, 0.2, 0.9]])]
    selection = choose_threshold(graphs, false_positive_rate_cap=0.0)
    assert selection["tau"] == 0.7
    assert selection["calibration_book_stability_score"] == 1.0
    assert selection["calibration_mean_false_positive_rate"] == 0.0


def test_threshold_tie_breaks_by_lower_false_positive_rate_then_higher_tau():
    graphs = [_calibration_graph([[0.8, 0.7, 0.9]])]
    selection = choose_threshold(graphs, false_positive_rate_cap=1.0)
    assert selection["tau"] == 0.8
    assert selection["calibration_mean_false_positive_rate"] == 0.0


def test_threshold_rejects_empty_calibration_and_invalid_cap():
    with pytest.raises(ValueError, match="at least one graph"):
        choose_threshold([])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        choose_threshold([_calibration_graph([[0.8, 0.7, 0.9]])], 1.1)
