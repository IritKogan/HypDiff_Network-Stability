from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence

from stability.edge_perturbation import Edge


@dataclass(frozen=True)
class StabilityScore:
    edge_recovery_frequencies: Mapping[Edge, float]
    scenario_score: float


def score_recovery(
    removed_edges: Sequence[Edge],
    reconstructions: Sequence[Iterable[Edge]],
) -> StabilityScore:
    """Score removed-edge recovery using binary presence in each reconstruction."""
    if not removed_edges:
        raise ValueError("removed_edges must not be empty")
    if not reconstructions:
        raise ValueError("reconstructions must not be empty")

    requested_edges = {}
    for edge in removed_edges:
        key = _edge_key(edge)
        if key in requested_edges:
            raise ValueError(f"removed_edges contains a duplicate undirected edge: {edge!r}")
        requested_edges[key] = edge

    recovered_counts = {key: 0 for key in requested_edges}
    for reconstruction in reconstructions:
        present = {_edge_key(edge) for edge in reconstruction}
        for key in recovered_counts:
            recovered_counts[key] += key in present

    frequencies = {
        requested_edges[key]: count / len(reconstructions)
        for key, count in recovered_counts.items()
    }
    return StabilityScore(
        edge_recovery_frequencies=frequencies,
        scenario_score=sum(frequencies.values()) / len(frequencies),
    )


def _edge_key(edge: Edge) -> frozenset[Hashable]:
    if len(edge) != 2:
        raise ValueError("each edge must contain exactly two node IDs")
    source, target = edge
    if source == target:
        raise ValueError(f"self-loops are not supported: {edge!r}")
    return frozenset((source, target))
