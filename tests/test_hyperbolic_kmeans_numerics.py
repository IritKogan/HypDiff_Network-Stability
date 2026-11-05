import numpy as np
import pytest

from hyperbolic_learning.hyperbolic_kmeans.hkmeans import (
    minkowski_distance_gradient,
    minkowski_loss_gradient,
    update_theta,
)
from hyperbolic_learning.hyperbolic_kmeans.util_hk import (
    hyperboloid_pt_to_poincare,
)


def test_centroid_update_bounds_large_steps_and_stays_inside_unit_ball():
    theta = np.asarray([0.0, 0.0, 1.0])
    gradient = np.asarray([1000.0, -2000.0, 0.0])
    updated = update_theta(theta, gradient, alpha=0.1)
    point = hyperboloid_pt_to_poincare(updated)
    assert np.isfinite(updated).all()
    assert np.isfinite(point).all()
    assert np.linalg.norm(point) < 1.0
    assert np.isclose(updated[-1] ** 2 - np.dot(updated[:-1], updated[:-1]), 1.0)


def test_distance_gradient_is_finite_for_coincident_hyperboloid_points():
    point = np.asarray([0.0, 0.0, 1.0])
    gradient = minkowski_distance_gradient(point, point)
    assert np.isfinite(gradient).all()


def test_nonfinite_centroid_update_and_frechet_gradient_fail_explicitly():
    theta = np.asarray([0.0, 0.0, 1.0])
    with pytest.raises(FloatingPointError, match="centroid step"):
        update_theta(theta, np.asarray([np.inf, 0.0, 0.0]))
    with pytest.raises(FloatingPointError, match="Fréchet-mean gradient"):
        minkowski_loss_gradient(theta, np.asarray([[np.nan, 0.0, 1.0]]))
