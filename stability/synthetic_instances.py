from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
import torch


CALIBRATION_SEEDS = (71001, 71002, 71003)
EVALUATION_SEEDS = (72001, 72002, 72003, 72004, 72005)
DIAGNOSTIC_SEEDS = (73001, 73002, 73003)
ENCODER_DAMAGE_SEEDS = (74001, 74002, 74003)
NODE_COUNT = 12


@dataclass(frozen=True)
class SyntheticGraph:
    graph_id: str
    seed: int
    node_ids: tuple[str, ...]
    features: torch.Tensor
    edges: tuple[tuple[str, str], ...]


def generate_synthetic_graph(seed: int, split: str) -> SyntheticGraph:
    if split not in {"calibration", "evaluation", "diagnostic", "encoder_damage"}:
        raise ValueError(
            "split must be 'calibration', 'evaluation', 'diagnostic', "
            "or 'encoder_damage'"
        )
    if split == "calibration" and seed not in CALIBRATION_SEEDS:
        raise ValueError("seed is not assigned to the calibration split")
    if split == "evaluation" and seed not in EVALUATION_SEEDS:
        raise ValueError("seed is not assigned to the evaluation split")
    if split == "diagnostic" and seed not in DIAGNOSTIC_SEEDS:
        raise ValueError("seed is not assigned to the diagnostic split")
    if split == "encoder_damage" and seed not in ENCODER_DAMAGE_SEEDS:
        raise ValueError("seed is not assigned to the encoder_damage split")

    node_ids = tuple(f"n{index:02d}" for index in range(NODE_COUNT))
    edges = []
    for start in (0, 6):
        community = tuple(range(start, start + 6))
        edges.extend(
            (community[index], community[(index + 1) % len(community)])
            for index in range(len(community))
        )
    rng = np.random.default_rng(seed)
    possible_bridges = tuple(itertools.product(range(6), range(6, 12)))
    bridge_indices = rng.choice(len(possible_bridges), size=3, replace=False)
    edges.extend(possible_bridges[index] for index in bridge_indices)
    named_edges = tuple(
        (node_ids[source], node_ids[target]) for source, target in edges
    )
    angles = 2 * np.pi * np.arange(NODE_COUNT) / NODE_COUNT
    features = torch.tensor(
        np.column_stack(
            (
                np.cos(angles),
                np.sin(angles),
                np.arange(NODE_COUNT) < 6,
                np.arange(NODE_COUNT) >= 6,
            )
        ),
        dtype=torch.float32,
    )
    return SyntheticGraph(
        graph_id=f"{split}-{seed}",
        seed=seed,
        node_ids=node_ids,
        features=features,
        edges=named_edges,
    )


def all_candidate_edges(node_ids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(itertools.combinations(node_ids, 2))


def validate_split_seeds() -> None:
    if set(CALIBRATION_SEEDS) & set(EVALUATION_SEEDS):
        raise ValueError("calibration and evaluation seed ranges must be disjoint")
    calibration_graphs = [
        generate_synthetic_graph(seed, "calibration") for seed in CALIBRATION_SEEDS
    ]
    evaluation_graphs = [
        generate_synthetic_graph(seed, "evaluation") for seed in EVALUATION_SEEDS
    ]
    if {graph.graph_id for graph in calibration_graphs} & {
        graph.graph_id for graph in evaluation_graphs
    }:
        raise ValueError("calibration and evaluation graph IDs must be disjoint")
    calibration_edge_sets = {frozenset(graph.edges) for graph in calibration_graphs}
    evaluation_edge_sets = {frozenset(graph.edges) for graph in evaluation_graphs}
    if len(calibration_edge_sets) != len(calibration_graphs):
        raise ValueError("calibration graph instances are not distinct")
    if len(evaluation_edge_sets) != len(evaluation_graphs):
        raise ValueError("evaluation graph instances are not distinct")
    if calibration_edge_sets & evaluation_edge_sets:
        raise ValueError("calibration and evaluation graph instances overlap")
