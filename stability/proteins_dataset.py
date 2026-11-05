from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset


MIN_NODES = 12
MAX_NODES = 50
MAX_EDGES = 200
SELECTION_SEED = 20261105
CALIBRATION_GRAPH_COUNT = 3
EVALUATION_GRAPH_COUNT = 5


@dataclass(frozen=True)
class ProteinsGraph:
    dataset_index: int
    content_sha256: str
    node_ids: tuple[str, ...]
    features: torch.Tensor
    edges: tuple[tuple[str, str], ...]

    @property
    def graph_id(self) -> str:
        return f"PROTEINS-{self.dataset_index:04d}"

    @property
    def seed(self) -> int:
        return self.dataset_index


@dataclass(frozen=True)
class GraphSplit:
    calibration: tuple[ProteinsGraph, ...]
    evaluation: tuple[ProteinsGraph, ...]
    eligible_graph_count: int
    distinct_eligible_graph_count: int


def _simple_edges(data: Data, node_count: int) -> tuple[tuple[int, int], ...] | None:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return None
    edges = set()
    for source, target in edge_index.t().tolist():
        source, target = int(source), int(target)
        if source < 0 or target < 0 or source >= node_count or target >= node_count:
            return None
        if source == target:
            return None
        edges.add((min(source, target), max(source, target)))
    return tuple(sorted(edges))


def _is_connected(node_count: int, edges: Iterable[tuple[int, int]]) -> bool:
    if node_count <= 0:
        return False
    adjacency = [set() for _ in range(node_count)]
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == node_count


def _content_hash(features: torch.Tensor, edges: tuple[tuple[int, int], ...]) -> str:
    digest = hashlib.sha256()
    values = features.detach().cpu().contiguous().numpy()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(values.tobytes())
    digest.update(np.asarray(edges, dtype=np.int64).reshape(-1, 2).tobytes())
    return digest.hexdigest()


def adapt_graph(data: Data, dataset_index: int) -> ProteinsGraph | None:
    node_count = int(data.num_nodes or 0)
    if node_count < MIN_NODES or node_count > MAX_NODES:
        return None
    features = getattr(data, "x", None)
    if (
        features is None
        or features.ndim != 2
        or features.shape[0] != node_count
        or features.shape[1] == 0
        or not torch.isfinite(features).all()
    ):
        return None
    simple_edges = _simple_edges(data, node_count)
    if simple_edges is None or not simple_edges:
        return None
    if len(simple_edges) > MAX_EDGES or not _is_connected(node_count, simple_edges):
        return None
    node_ids = tuple(
        f"PROTEINS-{dataset_index:04d}:n{node_index:03d}"
        for node_index in range(node_count)
    )
    named_edges = tuple(
        (node_ids[source], node_ids[target]) for source, target in simple_edges
    )
    features = features.detach().cpu().to(dtype=torch.float32).clone()
    return ProteinsGraph(
        dataset_index=dataset_index,
        content_sha256=_content_hash(features, simple_edges),
        node_ids=node_ids,
        features=features,
        edges=named_edges,
    )


def select_graph_splits(graphs: Iterable[ProteinsGraph]) -> GraphSplit:
    eligible = sorted(graphs, key=lambda graph: (graph.content_sha256, graph.dataset_index))
    unique = []
    seen_hashes = set()
    for graph in eligible:
        if graph.content_sha256 not in seen_hashes:
            unique.append(graph)
            seen_hashes.add(graph.content_sha256)
    required = CALIBRATION_GRAPH_COUNT + EVALUATION_GRAPH_COUNT
    if len(unique) < required:
        raise ValueError(
            f"only {len(unique)} distinct eligible PROTEINS graphs; "
            f"{required} are required"
        )
    permutation = np.random.default_rng(SELECTION_SEED).permutation(len(unique))
    selected = [unique[index] for index in permutation[:required]]
    calibration = tuple(selected[:CALIBRATION_GRAPH_COUNT])
    evaluation = tuple(selected[CALIBRATION_GRAPH_COUNT:])
    if {graph.content_sha256 for graph in calibration} & {
        graph.content_sha256 for graph in evaluation
    }:
        raise RuntimeError("calibration and evaluation graph content overlaps")
    return GraphSplit(
        calibration=calibration,
        evaluation=evaluation,
        eligible_graph_count=len(eligible),
        distinct_eligible_graph_count=len(unique),
    )


def load_proteins_graphs(
    root: Path,
    download: bool,
) -> tuple[tuple[ProteinsGraph, ...], dict[str, str]]:
    raw_dir = root / "PROTEINS" / "raw"
    if not download and not raw_dir.is_dir():
        raise FileNotFoundError(
            "PROTEINS is not cached; rerun the approved calibration command "
            "with download enabled"
        )
    dataset = TUDataset(
        root=str(root),
        name="PROTEINS",
        use_node_attr=True,
    )
    graphs = tuple(
        graph
        for index, data in enumerate(dataset)
        if (graph := adapt_graph(data, index)) is not None
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_dir.glob("*"))
        if path.is_file()
    }
    return graphs, hashes


def eligible_graph_id_manifest(split: GraphSplit) -> dict[str, object]:
    return {
        "selection_seed": SELECTION_SEED,
        "eligible_graph_count": split.eligible_graph_count,
        "distinct_eligible_graph_count": split.distinct_eligible_graph_count,
        "eligibility": {
            "connected": True,
            "simple_undirected": True,
            "minimum_nodes": MIN_NODES,
            "maximum_nodes": MAX_NODES,
            "maximum_unique_edges": MAX_EDGES,
            "finite_node_features_required": True,
            "self_loops": "excluded",
            "duplicate_and_reciprocal_edges": "canonicalized and deduplicated",
            "graph_labels_used": False,
        },
        "calibration": [
            {
                "dataset_index": graph.dataset_index,
                "graph_id": graph.graph_id,
                "content_sha256": graph.content_sha256,
                "node_count": len(graph.node_ids),
                "edge_count": len(graph.edges),
                "feature_dim": graph.features.shape[1],
            }
            for graph in split.calibration
        ],
        "evaluation": [
            {
                "dataset_index": graph.dataset_index,
                "graph_id": graph.graph_id,
                "content_sha256": graph.content_sha256,
                "node_count": len(graph.node_ids),
                "edge_count": len(graph.edges),
                "feature_dim": graph.features.shape[1],
            }
            for graph in split.evaluation
        ],
    }
