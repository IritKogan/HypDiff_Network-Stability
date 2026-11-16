import numpy as np

from stability.imdb_binary_f1 import (
    feature_distance_matrix,
    graph_feature_histograms,
    prdc_from_distances,
)


def test_coverage_uses_real_neighborhoods_not_generated_radii():
    real_real = np.array(
        [
            [0.0, 1.0, 10.0],
            [1.0, 0.0, 9.0],
            [10.0, 9.0, 0.0],
        ]
    )
    generated_generated = np.array(
        [
            [0.0, 1.0, 1.1],
            [1.0, 0.0, 0.1],
            [1.1, 0.1, 0.0],
        ]
    )
    real_generated = np.array(
        [
            [5.0, 6.0, 6.1],
            [4.0, 5.0, 5.1],
            [5.0, 4.0, 3.9],
        ]
    )

    result = prdc_from_distances(
        real_real, generated_generated, real_generated, (1,)
    )[1]

    assert result["recall"] == 0.0
    assert result["coverage"] == 1.0 / 3.0
    assert result["f1dc"] > 0.0


def test_prdc_includes_boundary_ties_and_reports_density_above_one():
    distances = np.zeros((3, 3), dtype=np.float64)

    result = prdc_from_distances(
        distances, distances, distances, (1, 2)
    )

    assert result[1]["precision"] == 1.0
    assert result[1]["recall"] == 1.0
    assert result[1]["density"] == 3.0
    assert result[1]["coverage"] == 1.0
    assert result[1]["f1dc"] == 1.5


def test_edgeless_graph_features_preserve_isolates_and_duplicate_distance():
    first = graph_feature_histograms(("a", "b", "c"), ())
    duplicate = graph_feature_histograms(("x", "y", "z"), ())

    distance = feature_distance_matrix((first, duplicate))

    assert distance.shape == (2, 2)
    assert np.array_equal(distance, np.zeros((2, 2)))
