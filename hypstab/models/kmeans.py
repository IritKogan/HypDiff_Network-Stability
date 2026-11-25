"""Hyperbolic K-means on the Poincare ball (book section 2: communities for the angular noise).

Written from scratch (replaces the upstream hyperboloid implementation, which had
three defects we fixed in the old code: Euclidean norm in exp_map, a gradient
precedence bug, and a random init that collapses clusters; bug log 2026-11-16).

Algorithm (Lloyd iterations with the geodesic distance):
1. k-means++ seeding with the squared Poincare distance (own numpy Generator).
2. Assign every point to its nearest centre (geodesic distance).
3. Empty cluster -> re-seeded with the point farthest from its own centre
   (taken from a cluster with more than one point).
4. New centre = Frechet (Karcher) mean of the cluster:
       mu <- expmap_mu( mean_i logmap_mu(x_i) )
   which is Riemannian gradient descent on (1/2n) sum_i d(mu, x_i)^2 with step 1,
   started from the previous centre, until the step is below ``tol``.
5. Stop when assignments no longer change or after ``max_iter`` iterations.

Differences from the old code: the Frechet mean is iterated to convergence (old:
10 steps of size 0.1, i.e. only partly converged); any curvature c > 0 (old: c = 1
only); the seed is an explicit Generator (old: global numpy state).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from hypstab.geometry import get_geometry
from hypstab.geometry import poincare as pb


@dataclass(frozen=True)
class KMeansResult:
    centers: Tensor          # (k, dim), inside the ball
    labels: Tensor           # (n,) long
    inertia: float           # sum of squared geodesic distances to the own centre
    iterations: int
    converged: bool          # assignments stable before max_iter


def pairwise_distance(points: Tensor, centers: Tensor, c: float, geo=pb) -> Tensor:
    """Geodesic distances (n, k) between points (n, d) and centres (k, d)."""
    return geo.distance(points[:, None, :], centers[None, :, :], c)


def frechet_mean(points: Tensor, c: float, init: Tensor, max_steps: int = 100, tol: float = 1e-9,
                 geo=pb) -> Tensor:
    """Karcher mean of points (m, d) on the Poincare ball, started at ``init`` (d,)."""
    mu = init
    for _ in range(max_steps):
        step = geo.logmap(mu.expand_as(points), points, c).mean(dim=0)
        mu_next = geo.expmap(mu, step, c)
        moved = float(geo.distance(mu, mu_next, c))
        mu = mu_next
        if moved < tol:
            break
    return mu


def kmeanspp_init(points: Tensor, k: int, c: float, rng: np.random.Generator, geo=pb) -> Tensor:
    n = points.shape[0]
    if k > n:
        raise ValueError(f"more clusters ({k}) than points ({n})")
    chosen = [int(rng.integers(n))]
    closest = geo.distance(points, points[chosen[0]], c).pow(2).cpu().numpy()
    for _ in range(1, k):
        total = closest.sum()
        index = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=closest / total))
        chosen.append(index)
        closest = np.minimum(closest, geo.distance(points, points[index], c).pow(2).cpu().numpy())
    return points[chosen].clone()


def hyperbolic_kmeans(
    points: Tensor,
    k: int,
    c: float,
    seed: int,
    max_iter: int,
    frechet_steps: int = 100,
    frechet_tol: float = 1e-9,
    manifold: str = "poincare",
) -> KMeansResult:
    """Cluster points (n, d) inside the Poincare ball (or in flat space, manifold "euclidean"). float64."""
    geo = get_geometry(manifold)
    x = points.detach().to(torch.float64)
    if not torch.isfinite(x).all():
        raise ValueError("hyperbolic_kmeans got non-finite points")
    if manifold == "poincare" and (x.norm(dim=-1) * c**0.5 >= 1).any():
        raise ValueError("hyperbolic_kmeans: points must lie inside the ball")
    rng = np.random.default_rng(seed)
    centers = kmeanspp_init(x, k, c, rng, geo)
    labels = pairwise_distance(x, centers, c, geo).argmin(dim=1)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        labels = _reseed_empty(x, centers, labels, k, c, geo)
        centers = torch.stack(
            [frechet_mean(x[labels == j], c, centers[j], frechet_steps, frechet_tol, geo) for j in range(k)]
        )
        new_labels = pairwise_distance(x, centers, c, geo).argmin(dim=1)
        if torch.equal(new_labels, labels):
            converged = True
            break
        labels = new_labels
    distances = pairwise_distance(x, centers, c, geo)
    inertia = float(distances.gather(1, labels[:, None]).pow(2).sum())
    return KMeansResult(
        centers=centers.to(points.dtype),
        labels=labels,
        inertia=inertia,
        iterations=iterations,
        converged=converged,
    )


def _reseed_empty(x: Tensor, centers: Tensor, labels: Tensor, k: int, c: float, geo=pb) -> Tensor:
    labels = labels.clone()
    for j in range(k):
        if (labels == j).any():
            continue
        own = geo.distance(x, centers[labels], c)
        sizes = torch.bincount(labels, minlength=k)[labels]
        own[sizes <= 1] = -1.0  # never empty another cluster
        farthest = int(own.argmax())
        labels[farthest] = j
        centers[j] = x[farthest]
    return labels
