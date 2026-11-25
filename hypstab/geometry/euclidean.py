"""Flat (Euclidean) geometry with the same function names and signatures as poincare.py.

Used only for the hyperbolic-vs-Euclidean check (geometry.manifold = "euclidean"): the whole
pipeline stays the same, only the space changes. The curvature argument ``c`` is accepted and
ignored. With these maps the HGCN layer becomes a plain GCN layer (linear -> A_hat aggregation ->
activation), the Fermi-Dirac decoder uses the squared Euclidean distance, the Karcher mean is the
ordinary mean and the tangent space at a centre is "point minus centre".
"""

from __future__ import annotations

import torch
from torch import Tensor


def project(x: Tensor, c=None) -> Tensor:
    return x


def mobius_add(x: Tensor, y: Tensor, c=None) -> Tensor:
    return x + y


def expmap0(v: Tensor, c=None) -> Tensor:
    return v


def logmap0(y: Tensor, c=None) -> Tensor:
    return y


def expmap(x: Tensor, v: Tensor, c=None) -> Tensor:
    return x + v


def logmap(x: Tensor, y: Tensor, c=None) -> Tensor:
    return y - x


def sqdistance(x: Tensor, y: Tensor, c=None, keepdim: bool = False) -> Tensor:
    return (x - y).pow(2).sum(dim=-1, keepdim=keepdim)


def distance(x: Tensor, y: Tensor, c=None, keepdim: bool = False) -> Tensor:
    return sqdistance(x, y, c, keepdim).clamp_min(0).sqrt()


def mobius_matvec(m: Tensor, x: Tensor, c=None) -> Tensor:
    return x @ m.transpose(-1, -2)
