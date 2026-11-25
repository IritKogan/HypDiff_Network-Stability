"""Geometry back-ends: "poincare" (the model) and "euclidean" (hyperbolic-vs-Euclidean check only)."""

from __future__ import annotations

from types import ModuleType

MANIFOLDS = ("poincare", "euclidean")


def get_geometry(manifold: str) -> ModuleType:
    if manifold == "poincare":
        from hypstab.geometry import poincare

        return poincare
    if manifold == "euclidean":
        from hypstab.geometry import euclidean

        return euclidean
    raise ValueError(f"unknown manifold {manifold!r}; choose from {MANIFOLDS}")
