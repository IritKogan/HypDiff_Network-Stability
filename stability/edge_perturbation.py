from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Hashable, Sequence

import torch


Edge = tuple[Hashable, Hashable]


@dataclass(frozen=True)
class PerturbedGraph:
    node_ids: tuple[Hashable, ...]
    features: torch.Tensor
    edges: tuple[Edge, ...]
    removed_edges: tuple[Edge, ...]
    is_connected: bool


def remove_edges(
    node_ids: Sequence[Hashable],
    features: torch.Tensor,
    edges: Sequence[Edge],
    num_edges: int,
    seed: int,
) -> PerturbedGraph:
    """Remove an exact seeded number of edges from an undirected simple graph."""
    nodes = tuple(node_ids)
    if len(set(nodes)) != len(nodes):
        raise ValueError("node_ids must be unique")
    if features.ndim == 0 or features.shape[0] != len(nodes):
        raise ValueError("features must have one row per node_id")
    if num_edges < 0:
        raise ValueError("num_edges must be non-negative")

    node_order = {node: index for index, node in enumerate(nodes)}
    normalized_edges = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each edge must contain exactly two node IDs")
        source, target = edge
        if source not in node_order or target not in node_order:
            raise ValueError(f"edge endpoint missing from node_ids: {edge!r}")
        if source == target:
            raise ValueError(f"self-loops are not supported: {edge!r}")
        if node_order[source] < node_order[target]:
            normalized_edges.add((source, target))
        else:
            normalized_edges.add((target, source))

    ordered_edges = tuple(
        sorted(normalized_edges, key=lambda edge: (node_order[edge[0]], node_order[edge[1]]))
    )
    if num_edges > len(ordered_edges):
        raise ValueError(
            f"cannot remove {num_edges} edges from a graph with {len(ordered_edges)} edges"
        )

    removed_edges = tuple(random.Random(seed).sample(ordered_edges, num_edges))
    removed_set = set(removed_edges)
    remaining_edges = tuple(edge for edge in ordered_edges if edge not in removed_set)
    return PerturbedGraph(
        node_ids=nodes,
        features=features.clone(),
        edges=remaining_edges,
        removed_edges=removed_edges,
        is_connected=_is_connected(nodes, remaining_edges),
    )


def _is_connected(node_ids: tuple[Hashable, ...], edges: tuple[Edge, ...]) -> bool:
    if not node_ids:
        return False
    adjacency = {node: set() for node in node_ids}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)

    visited = {node_ids[0]}
    frontier = [node_ids[0]]
    while frontier:
        for neighbor in adjacency[frontier.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == len(node_ids)
