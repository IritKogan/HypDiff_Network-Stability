import numpy as np
import pytest

from stability.hyperbolic_kmeans.hkmeans import (
    minkowski_distance_gradient,
    minkowski_loss_gradient,
    update_theta,
)
from stability.hyperbolic_kmeans.util_hk import (
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


# ---- exp_map uses the Minkowski norm (fix 2026-11-16) ----------------------

from stability.hyperbolic_kmeans.hkmeans import (  # noqa: E402
    HyperbolicKMeans,
    compute_mean,
    exp_map,
    frechet_loss,
    project_to_tangent,
)
from stability.hyperbolic_kmeans.util_hk import (  # noqa: E402
    hyperboloid_dot,
    poincare_pt_to_hyperboloid,
    poincare_pts_to_hyperboloid,
)


def _near_boundary_cloud(seed=0, count=200, dim=8):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.3, 0.99, size=(count, 1))
    shift = np.zeros(dim)
    shift[0] = 0.3
    points = 0.6 * directions * radii + shift
    return points / np.maximum(1.0, np.linalg.norm(points, axis=1, keepdims=True) / 0.99)


def test_exp_map_stays_on_hyperboloid_for_far_base_points():
    theta = poincare_pt_to_hyperboloid(np.array([0.9, 0.3, 0.0]))
    tangent = project_to_tangent(theta, np.array([0.5, -1.0, 0.2, 3.0]))
    tangent *= 0.8 / np.sqrt(hyperboloid_dot(tangent, tangent))   # geodesic length 0.8
    point = exp_map(tangent, theta)
    assert np.isclose(hyperboloid_dot(point, point), -1.0)
    # distance travelled equals the Minkowski norm of the step
    assert np.isclose(np.arccosh(-hyperboloid_dot(point, theta)), 0.8, atol=1e-4)  # theta built with eps=1e-6


def test_frechet_mean_descends_and_stays_inside_ball_with_near_boundary_points():
    points = poincare_pts_to_hyperboloid(_near_boundary_cloud())
    theta = poincare_pt_to_hyperboloid(np.array([0.2] + [0.0] * 7))
    losses = [frechet_loss(theta, points)]
    for _ in range(5):
        theta = compute_mean(theta, points, alpha=0.1)
        losses.append(frechet_loss(theta, points))
        assert np.isclose(hyperboloid_dot(theta, theta), -1.0)
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))


def test_kmeans_centroids_do_not_drift_to_the_boundary():
    points = _near_boundary_cloud(seed=1, count=150)
    np.random.seed(0)
    centroids, labels = HyperbolicKMeans(n_clusters=3).fit(points, max_epochs=10)
    assert np.all(np.linalg.norm(centroids, axis=1) < 0.95)
    assert len(np.unique(labels)) == 3
