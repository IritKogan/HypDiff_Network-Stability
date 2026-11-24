"""Poincare ball of curvature -c (c > 0): the set {x : c * ||x||^2 < 1}, radius 1/sqrt(c).

Same convention as HypDiff / HGCN (Chami et al. 2019, Ganea et al. 2018), so
results stay comparable to the old code. All functions work on the last
dimension and broadcast over leading dimensions. Argument order is always
(point, tangent vector, c) or (point, point, c).

Formulas (lambda_x = 2 / (1 - c ||x||^2) is the conformal factor):
    x (+)_c y      = ((1 + 2c<x,y> + c||y||^2) x + (1 - c||x||^2) y) / (1 + 2c<x,y> + c^2 ||x||^2 ||y||^2)
    expmap0(v)     = tanh(sqrt(c) ||v||) v / (sqrt(c) ||v||)
    logmap0(y)     = artanh(sqrt(c) ||y||) y / (sqrt(c) ||y||)
    expmap_x(v)    = x (+)_c [ tanh(sqrt(c) lambda_x ||v|| / 2) v / (sqrt(c) ||v||) ]
    logmap_x(y)    = 2 / (sqrt(c) lambda_x) artanh(sqrt(c) ||w||) w / ||w||,   w = (-x) (+)_c y
    d_c(x, y)      = 2 / sqrt(c) artanh(sqrt(c) ||(-x) (+)_c y||)
    M (x)_c x      = tanh(||Mx|| / ||x|| artanh(sqrt(c) ||x||)) Mx / (sqrt(c) ||Mx||)

Numerics (differences from the upstream code, all deliberate):
- ``artanh`` clamps its argument to 1 - eps(dtype). Upstream clamps to 1 - 1e-15,
  which is exactly 1.0 in float32 and gives inf if a point reaches the boundary.
- Every function that returns a point (expmap0, expmap, mobius_add, mobius_matvec)
  projects it back inside the ball with ``project``. Upstream leaves this to the
  caller (HGCN layers call proj after each op), so inside the ball the results
  are identical (tested against upstream in tests/hypstab_tests/test_poincare.py).
- ``project`` keeps points at norm <= (1 - BOUNDARY_EPS[dtype]) / sqrt(c), the
  same margins as upstream (float32 4e-3, float64 1e-5).
- Zero vectors: norms are clamped at MIN_NORM so expmap0(0) = 0, logmap0(0) = 0,
  mobius_matvec(M, x) = 0 when Mx = 0, without NaN in values or gradients.
"""

from __future__ import annotations

import torch
from torch import Tensor

MIN_NORM = 1e-15
BOUNDARY_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


def _sqrt_c(c: float | Tensor) -> float | Tensor:
    return c**0.5


def _norm(x: Tensor) -> Tensor:
    return x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)


def artanh(x: Tensor) -> Tensor:
    """Inverse hyperbolic tangent with the argument kept strictly inside (-1, 1)."""
    limit = 1.0 - torch.finfo(x.dtype).eps
    return torch.atanh(x.clamp(-limit, limit))


def conformal_factor(x: Tensor, c: float | Tensor) -> Tensor:
    """lambda_x = 2 / (1 - c ||x||^2), shape (..., 1)."""
    sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - c * sqnorm).clamp_min(MIN_NORM)


def project(x: Tensor, c: float | Tensor) -> Tensor:
    """Pull points that are too close to (or beyond) the boundary back inside the ball."""
    norm = _norm(x)
    max_norm = (1.0 - BOUNDARY_EPS[x.dtype]) / _sqrt_c(c)
    return torch.where(norm > max_norm, x / norm * max_norm, x)


def mobius_add(x: Tensor, y: Tensor, c: float | Tensor) -> Tensor:
    """Mobius addition x (+)_c y."""
    return project(_mobius_add_raw(x, y, c), c)


def expmap0(v: Tensor, c: float | Tensor) -> Tensor:
    """Tangent vector at the origin -> point in the ball."""
    sqrt_c = _sqrt_c(c)
    v_norm = _norm(v)
    return project(torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm), c)


def logmap0(y: Tensor, c: float | Tensor) -> Tensor:
    """Point in the ball -> tangent vector at the origin (inverse of expmap0)."""
    sqrt_c = _sqrt_c(c)
    y_norm = _norm(y)
    return artanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)


def expmap(x: Tensor, v: Tensor, c: float | Tensor) -> Tensor:
    """Tangent vector v at point x -> point in the ball."""
    sqrt_c = _sqrt_c(c)
    v_norm = _norm(v)
    second = torch.tanh(sqrt_c / 2 * conformal_factor(x, c) * v_norm) * v / (sqrt_c * v_norm)
    return mobius_add(x, second, c)


def logmap(x: Tensor, y: Tensor, c: float | Tensor) -> Tensor:
    """Point y -> tangent vector at point x (inverse of expmap at x)."""
    sqrt_c = _sqrt_c(c)
    w = _mobius_add_raw(-x, y, c)
    w_norm = _norm(w)
    return 2 / (sqrt_c * conformal_factor(x, c)) * artanh(sqrt_c * w_norm) * w / w_norm


def distance(x: Tensor, y: Tensor, c: float | Tensor, keepdim: bool = False) -> Tensor:
    """Geodesic distance d_c(x, y)."""
    sqrt_c = _sqrt_c(c)
    w_norm = _mobius_add_raw(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim)
    return 2 / sqrt_c * artanh(sqrt_c * w_norm)


def sqdistance(x: Tensor, y: Tensor, c: float | Tensor, keepdim: bool = False) -> Tensor:
    """Squared geodesic distance (used by the Fermi-Dirac decoder)."""
    return distance(x, y, c, keepdim=keepdim) ** 2


def mobius_matvec(m: Tensor, x: Tensor, c: float | Tensor) -> Tensor:
    """Mobius matrix-vector product M (x)_c x for weight m of shape (out, in), x of shape (..., in)."""
    sqrt_c = _sqrt_c(c)
    x_norm = _norm(x)
    mx = x @ m.transpose(-1, -2)
    mx_norm = _norm(mx)
    res = torch.tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    is_zero = (mx == 0).all(dim=-1, keepdim=True)
    return project(torch.where(is_zero, torch.zeros_like(res), res), c)


def tangent_mean0(
    y: Tensor, c: float | Tensor, weights: Tensor | None = None, dim: int = 0
) -> Tensor:
    """Mean taken in the tangent space at the origin: expmap0(weighted mean of logmap0(y)).

    ``weights`` (optional) has y's shape without the last dimension; it is
    normalised to sum to 1 along ``dim``. Not the Frechet mean (cheaper; HGCN
    aggregation uses the same tangent-space averaging).
    """
    v = logmap0(y, c)
    if weights is None:
        mean = v.mean(dim=dim)
    else:
        w = weights / weights.sum(dim=dim, keepdim=True)
        mean = (v * w.unsqueeze(-1)).sum(dim=dim)
    return expmap0(mean, c)


def _mobius_add_raw(x: Tensor, y: Tensor, c: float | Tensor) -> Tensor:
    """Mobius addition without the final projection (for logmap / distance)."""
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)
