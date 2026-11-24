"""Tests for hypstab/geometry/poincare.py (Stage 1).

Three kinds of checks:
1. closed forms (1-D distances, conformal factor, known identities);
2. internal consistency (inverses, metric axioms, gyro-group identities, gradients);
3. agreement with upstream HypDiff's PoincareBall inside the ball. Upstream is used
   ONLY here, as a reference; hypstab itself never imports it.
"""

from __future__ import annotations

import math

import pytest
import torch

from hypstab.geometry import poincare as pb

CURVATURES = [0.5, 1.0, 2.0]
DTYPE = torch.float64


def random_points(n: int, dim: int, c: float, max_radius: float = 0.9, seed: int = 0) -> torch.Tensor:
    """n points inside the ball, norms uniform in [0, max_radius / sqrt(c)]."""
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(n, dim, generator=g, dtype=DTYPE)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    radius = torch.rand(n, 1, generator=g, dtype=DTYPE) * max_radius / math.sqrt(c)
    return direction * radius


def random_tangent(n: int, dim: int, scale: float = 1.0, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g, dtype=DTYPE) * scale


# --------------------------------------------------------------------------- closed forms


@pytest.mark.parametrize("c", CURVATURES)
def test_distance_from_origin_closed_form(c):
    x = random_points(50, 5, c)
    expected = 2 / math.sqrt(c) * torch.atanh(math.sqrt(c) * x.norm(dim=-1))
    torch.testing.assert_close(pb.distance(torch.zeros_like(x), x, c), expected)


@pytest.mark.parametrize("c", CURVATURES)
def test_one_dimensional_distance(c):
    # On a diameter: d(a, b) = 2/sqrt(c) |artanh(sqrt(c) b) - artanh(sqrt(c) a)|
    a = torch.tensor([[-0.3], [0.1], [0.5]], dtype=DTYPE) / math.sqrt(c)
    b = torch.tensor([[0.6], [-0.7], [0.5]], dtype=DTYPE) / math.sqrt(c)
    s = math.sqrt(c)
    expected = 2 / s * (torch.atanh(s * b) - torch.atanh(s * a)).abs().squeeze(-1)
    torch.testing.assert_close(pb.distance(a, b, c), expected)


@pytest.mark.parametrize("c", CURVATURES)
def test_expmap0_norm_closed_form(c):
    v = random_tangent(40, 4, scale=0.4)  # interior (no projection)
    y = pb.expmap0(v, c)
    torch.testing.assert_close(y.norm(dim=-1), torch.tanh(math.sqrt(c) * v.norm(dim=-1)) / math.sqrt(c))
    # direction is preserved
    torch.testing.assert_close(y / y.norm(dim=-1, keepdim=True), v / v.norm(dim=-1, keepdim=True))


def test_conformal_factor():
    x = torch.tensor([[0.0, 0.0], [0.6, 0.0]], dtype=DTYPE)
    torch.testing.assert_close(pb.conformal_factor(x, 1.0), torch.tensor([[2.0], [2 / 0.64]], dtype=DTYPE))


def test_distance_origin_and_expmap0_are_consistent():
    # d(0, expmap0(v)) = 2 ||v||  (tangent space at 0 has metric lambda_0^2 = 4)
    c = 1.0
    v = random_tangent(30, 3, scale=0.5)
    torch.testing.assert_close(pb.distance(torch.zeros_like(v), pb.expmap0(v, c), c), 2 * v.norm(dim=-1))


# --------------------------------------------------------------------------- consistency


@pytest.mark.parametrize("c", CURVATURES)
def test_expmap0_logmap0_inverse(c):
    y = random_points(100, 8, c)
    torch.testing.assert_close(pb.expmap0(pb.logmap0(y, c), c), y)
    v = random_tangent(100, 8, scale=0.7)
    torch.testing.assert_close(pb.logmap0(pb.expmap0(v, c), c), v)


@pytest.mark.parametrize("c", CURVATURES)
def test_expmap_logmap_inverse_at_any_point(c):
    x = random_points(100, 6, c, seed=2)
    y = random_points(100, 6, c, seed=3)
    torch.testing.assert_close(pb.expmap(x, pb.logmap(x, y, c), c), y)
    # ||logmap_x(y)||_x = lambda_x ||logmap_x(y)|| = d(x, y)
    v = pb.logmap(x, y, c)
    torch.testing.assert_close((pb.conformal_factor(x, c) * v.norm(dim=-1, keepdim=True)).squeeze(-1),
                               pb.distance(x, y, c))


@pytest.mark.parametrize("c", CURVATURES)
def test_expmap_at_origin_equals_expmap0(c):
    v = random_tangent(20, 5, scale=0.5)
    zero = torch.zeros_like(v)
    torch.testing.assert_close(pb.expmap(zero, v, c), pb.expmap0(v, c))
    y = random_points(20, 5, c)
    torch.testing.assert_close(pb.logmap(zero, y, c), pb.logmap0(y, c))


@pytest.mark.parametrize("c", CURVATURES)
def test_metric_axioms(c):
    x, y, z = (random_points(60, 4, c, seed=s) for s in (4, 5, 6))
    dxy = pb.distance(x, y, c)
    torch.testing.assert_close(dxy, pb.distance(y, x, c))
    assert torch.all(dxy >= 0)
    torch.testing.assert_close(pb.distance(x, x, c), torch.zeros(60, dtype=DTYPE), atol=1e-7, rtol=0)
    assert torch.all(pb.distance(x, z, c) <= dxy + pb.distance(y, z, c) + 1e-10)


@pytest.mark.parametrize("c", CURVATURES)
def test_mobius_add_identities(c):
    x = random_points(50, 3, c, seed=7)
    y = random_points(50, 3, c, seed=8)
    zero = torch.zeros_like(x)
    torch.testing.assert_close(pb.mobius_add(x, zero, c), x)
    torch.testing.assert_close(pb.mobius_add(zero, x, c), x)
    torch.testing.assert_close(pb.mobius_add(-x, x, c), zero, atol=1e-12, rtol=0)
    # left cancellation: (-x) (+) (x (+) y) = y
    torch.testing.assert_close(pb.mobius_add(-x, pb.mobius_add(x, y, c), c), y)
    # Mobius addition is an isometry: d(x (+) a, x (+) b) = d(a, b)
    a, b = random_points(50, 3, c, 0.5, 9), random_points(50, 3, c, 0.5, 10)
    torch.testing.assert_close(
        pb.distance(pb.mobius_add(x, a, c), pb.mobius_add(x, b, c), c), pb.distance(a, b, c)
    )


@pytest.mark.parametrize("c", CURVATURES)
def test_mobius_matvec(c):
    x = random_points(40, 4, c, seed=11)
    identity = torch.eye(4, dtype=DTYPE)
    torch.testing.assert_close(pb.mobius_matvec(identity, x, c), x)
    m = torch.randn(3, 4, dtype=DTYPE, generator=torch.Generator().manual_seed(12))
    # M (x) x = expmap0(M logmap0(x))
    torch.testing.assert_close(pb.mobius_matvec(m, x, c), pb.expmap0(pb.logmap0(x, c) @ m.T, c))
    # rows with Mx = 0 map to the origin
    zero_rows = pb.mobius_matvec(torch.zeros(3, 4, dtype=DTYPE), x, c)
    assert torch.equal(zero_rows, torch.zeros_like(zero_rows))


def test_tangent_mean0():
    c = 1.0
    y = random_points(10, 3, c, seed=13)
    torch.testing.assert_close(pb.tangent_mean0(y, c), pb.expmap0(pb.logmap0(y, c).mean(0), c))
    # mean of a point with itself is the point; symmetric pair averages to the origin
    torch.testing.assert_close(pb.tangent_mean0(torch.stack([y[0], y[0]]), c), y[0])
    torch.testing.assert_close(pb.tangent_mean0(torch.stack([y[0], -y[0]]), c), torch.zeros(3, dtype=DTYPE))
    # weights: all weight on one point returns it; weights are normalised
    w = torch.zeros(10, dtype=DTYPE)
    w[4] = 3.0
    torch.testing.assert_close(pb.tangent_mean0(y, c, weights=w), y[4])
    # batched: mean over dim=1
    batch = torch.stack([y, y.flip(0)])
    out = pb.tangent_mean0(batch, c, dim=1)
    torch.testing.assert_close(out[0], out[1])


# --------------------------------------------------------------------------- numerics


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_project_keeps_points_inside(dtype):
    c = 1.0
    x = torch.tensor([[2.0, 0.0], [0.0, 0.99999], [0.1, 0.2]], dtype=dtype)
    p = pb.project(x, c)
    max_norm = 1 - pb.BOUNDARY_EPS[dtype]
    assert torch.all(p.norm(dim=-1) <= max_norm + 1e-6)
    assert torch.equal(p[2], x[2])  # interior points untouched


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_inf_or_nan_near_boundary(dtype):
    c = 1.0
    big = torch.tensor([[50.0, 0.0], [0.0, -80.0]], dtype=dtype)  # expmap0 saturates tanh to 1
    y = pb.expmap0(big, c)
    assert torch.all(y.norm(dim=-1) < 1)
    for out in (pb.logmap0(y, c), pb.distance(y[:1], y[1:], c), pb.logmap(y[:1], y[1:], c)):
        assert torch.isfinite(out).all()
    # artanh at exactly 1 stays finite (upstream clamp 1 - 1e-15 rounds to 1.0 in float32)
    assert torch.isfinite(pb.artanh(torch.ones(1, dtype=dtype))).all()


def test_zero_vectors_have_finite_values_and_gradients():
    c = 1.0
    v = torch.zeros(2, 3, dtype=DTYPE, requires_grad=True)
    out = pb.expmap0(v, c).sum() + pb.logmap0(v, c).sum()
    out.backward()
    assert torch.isfinite(v.grad).all()
    x = torch.zeros(2, 3, dtype=DTYPE, requires_grad=True)
    m = torch.randn(4, 3, dtype=DTYPE, requires_grad=True)
    pb.mobius_matvec(m, x, c).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(m.grad).all()


@pytest.mark.parametrize("c", CURVATURES)
def test_distance_gradient_matches_finite_differences(c):
    x = random_points(3, 2, c, max_radius=0.7, seed=14).requires_grad_(True)
    y = random_points(3, 2, c, max_radius=0.7, seed=15)
    assert torch.autograd.gradcheck(lambda a: pb.distance(a, y, c), (x,))
    assert torch.autograd.gradcheck(lambda a: pb.logmap(a, y, c), (x,))


def test_curvature_as_tensor_and_broadcasting():
    c = torch.tensor(2.0, dtype=DTYPE)
    x = random_points(5, 3, 2.0)
    torch.testing.assert_close(pb.distance(x, x[:1], c), pb.distance(x, x[:1].expand(5, 3), 2.0))


# --------------------------------------------------------------------------- upstream reference


@pytest.fixture(scope="module")
def upstream():
    module = pytest.importorskip("manifolds.poincare")  # upstream HypDiff, reference only
    return module.PoincareBall()


@pytest.mark.parametrize("c", CURVATURES)
def test_matches_upstream_poincare_ball(upstream, c):
    x = random_points(80, 6, c, seed=16)
    y = random_points(80, 6, c, seed=17)
    v = random_tangent(80, 6, scale=0.15, seed=18)  # images stay in the interior (no projection)
    m = torch.randn(4, 6, dtype=DTYPE, generator=torch.Generator().manual_seed(19))
    close = torch.testing.assert_close
    close(pb.mobius_add(x, y, c), upstream.mobius_add(x, y, c))
    close(pb.expmap0(v, c), upstream.expmap0(v, c))
    close(pb.logmap0(y, c), upstream.logmap0(y, c))
    close(pb.expmap(x, v, c), upstream.expmap(v, x, c))  # upstream order: (tangent, point)
    close(pb.logmap(x, y, c), upstream.logmap(x, y, c))
    close(pb.sqdistance(x, y, c), upstream.sqdist(x, y, c))
    close(pb.mobius_matvec(m, x, c), upstream.mobius_matvec(m, x, c))
    close(pb.conformal_factor(x, c), upstream._lambda_x(x, c))
    # near the boundary the only difference is that hypstab projects (upstream callers do it themselves)
    big = random_tangent(80, 6, scale=3.0, seed=21)
    close(pb.expmap0(big, c), upstream.proj(upstream.expmap0(big, c), c))
    close(pb.mobius_matvec(3 * m, x, c), upstream.proj(upstream.mobius_matvec(3 * m, x, c), c))
    far = random_tangent(80, 6, scale=10.0, seed=20)  # outside -> both project to the same margin
    for dtype in (torch.float32, torch.float64):
        close(pb.project(far.to(dtype), c), upstream.proj(far.to(dtype), c))
