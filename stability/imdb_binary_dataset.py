from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

import numpy as np
import torch

from stability.proteins_dataset import (
    MAX_EDGES,
    MAX_NODES,
    MIN_NODES,
    _content_hash,
    _is_connected,
)


DATASET_NAME = "IMDB-BINARY"
DOWNLOAD_URL = "https://chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip"
REQUIRED_RAW_FILES = (
    "IMDB-BINARY_A.txt",
    "IMDB-BINARY_graph_indicator.txt",
)


@dataclass(frozen=True)
class IMDBBinaryGraph:
    dataset_index: int
    content_sha256: str
    node_ids: tuple[str, ...]
    features: torch.Tensor
    edges: tuple[tuple[str, str], ...]

    @property
    def graph_id(self) -> str:
        return f"{DATASET_NAME}-{self.dataset_index:04d}"

    @property
    def seed(self) -> int:
        return self.dataset_index


@dataclass(frozen=True)
class IMDBBinarySplit:
    calibration: tuple[IMDBBinaryGraph, ...]
    evaluation: tuple[IMDBBinaryGraph, ...]
    eligible_graph_count: int
    distinct_eligible_graph_count: int


def _read_integer_lines(path: Path) -> list[int]:
    try:
        return [
            int(line.strip())
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    except ValueError as error:
        raise ValueError(f"{path.name} must contain one integer per line") from error


def _download_raw_files(raw_dir: Path) -> None:
    with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = {
            Path(name).name: name
            for name in archive.namelist()
            if name.startswith(f"{DATASET_NAME}/") and name.endswith(".txt")
        }
        missing = [name for name in REQUIRED_RAW_FILES if name not in members]
        if missing:
            raise ValueError(f"IMDB-BINARY archive is missing required files: {missing}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for filename, member in members.items():
            path = raw_dir / filename
            if path.exists():
                continue
            path.write_bytes(archive.read(member))


def _raw_file_hashes(raw_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_dir.glob("*.txt"))
        if path.is_file()
    }


def load_imdb_binary_graphs(
    root: Path,
    download: bool = False,
) -> tuple[tuple[IMDBBinaryGraph, ...], dict[str, str], dict[str, object]]:
    raw_dir = root / DATASET_NAME / "raw"
    if not all((raw_dir / name).is_file() for name in REQUIRED_RAW_FILES):
        if not download:
            raise FileNotFoundError(
                f"IMDB-BINARY raw files are not cached under {raw_dir}"
            )
        _download_raw_files(raw_dir)

    graph_indicator = _read_integer_lines(raw_dir / "IMDB-BINARY_graph_indicator.txt")
    if not graph_indicator or min(graph_indicator) != 1:
        raise ValueError("IMDB-BINARY graph_indicator must use positive 1-based IDs")
    graph_ids = sorted(set(graph_indicator))
    if graph_ids != list(range(1, graph_ids[-1] + 1)):
        raise ValueError("IMDB-BINARY graph_indicator IDs must be contiguous")

    nodes_by_graph: dict[int, list[int]] = {graph_id: [] for graph_id in graph_ids}
    for node_index, graph_id in enumerate(graph_indicator):
        nodes_by_graph[graph_id].append(node_index)
    node_to_graph = graph_indicator
    local_indices = {
        graph_id: {node: index for index, node in enumerate(nodes)}
        for graph_id, nodes in nodes_by_graph.items()
    }
    edges_by_graph: dict[int, set[tuple[int, int]]] = {
        graph_id: set() for graph_id in graph_ids
    }
    edge_path = raw_dir / "IMDB-BINARY_A.txt"
    for line_number, line in enumerate(edge_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            source_raw, target_raw = (int(value.strip()) for value in line.split(",", 1))
        except ValueError as error:
            raise ValueError(
                f"{edge_path.name}:{line_number} must contain two integer endpoints"
            ) from error
        source, target = source_raw - 1, target_raw - 1
        if not (0 <= source < len(graph_indicator) and 0 <= target < len(graph_indicator)):
            raise ValueError(f"{edge_path.name}:{line_number} endpoint is out of range")
        graph_id = node_to_graph[source]
        if node_to_graph[target] != graph_id:
            raise ValueError(f"{edge_path.name}:{line_number} connects different graphs")
        if source == target:
            raise ValueError(f"{edge_path.name}:{line_number} contains a self-loop")
        local_index = local_indices[graph_id]
        edge = tuple(sorted((local_index[source], local_index[target])))
        edges_by_graph[graph_id].add(edge)

    graphs = []
    for graph_id in graph_ids:
        global_nodes = nodes_by_graph[graph_id]
        node_count = len(global_nodes)
        edges = tuple(sorted(edges_by_graph[graph_id]))
        features = torch.ones((node_count, 1), dtype=torch.float32)
        if (
            node_count < MIN_NODES
            or node_count > MAX_NODES
            or len(edges) > MAX_EDGES
            or not edges
            or not _is_connected(node_count, edges)
        ):
            continue
        node_ids = tuple(
            f"{DATASET_NAME}-{graph_id:04d}:n{local_index:03d}"
            for local_index in range(node_count)
        )
        named_edges = tuple(
            (node_ids[source], node_ids[target]) for source, target in edges
        )
        graphs.append(
            IMDBBinaryGraph(
                dataset_index=graph_id,
                content_sha256=_content_hash(features, edges),
                node_ids=node_ids,
                features=features,
                edges=named_edges,
            )
        )

    raw_hashes = _raw_file_hashes(raw_dir)
    raw_names = set(raw_hashes)
    feature_files = [
        name
        for name in raw_names
        if "node_attributes" in name.lower() or "node_labels" in name.lower()
    ]
    return (
        tuple(graphs),
        raw_hashes,
        {
            "graph_count_in_raw_indicator": len(graph_ids),
            "eligible_graph_count": len(graphs),
            "node_attribute_files": sorted(
                name for name in feature_files if "node_attributes" in name.lower()
            ),
            "node_label_files": sorted(
                name for name in feature_files if "node_labels" in name.lower()
            ),
            "feature_policy": "constant all-ones node features with shape (N, 1)",
            "graph_labels_used": False,
        },
    )


def with_node_identity_features(graph: IMDBBinaryGraph) -> IMDBBinaryGraph:
    identity = torch.eye(len(graph.node_ids), dtype=graph.features.dtype)
    features = torch.cat((graph.features, identity), dim=1)
    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    indexed_edges = tuple(
        (node_index[source], node_index[target]) for source, target in graph.edges
    )
    return replace(
        graph,
        content_sha256=_content_hash(features, indexed_edges),
        features=features,
    )


def select_imdb_binary_splits(
    graphs: tuple[IMDBBinaryGraph, ...],
    seed: int = 20261110,
) -> IMDBBinarySplit:
    ordered = sorted(
        graphs, key=lambda graph: (graph.content_sha256, graph.dataset_index)
    )
    unique = []
    seen = set()
    for graph in ordered:
        if graph.content_sha256 not in seen:
            unique.append(graph)
            seen.add(graph.content_sha256)
    required = 10
    if len(unique) < required:
        raise ValueError(
            f"only {len(unique)} distinct eligible IMDB-BINARY graphs; "
            f"{required} are required"
        )
    order = np.random.default_rng(seed).permutation(len(unique))
    selected = [unique[index] for index in order[:required]]
    calibration, evaluation = tuple(selected[:5]), tuple(selected[5:])
    if {graph.content_sha256 for graph in calibration} & {
        graph.content_sha256 for graph in evaluation
    }:
        raise RuntimeError("IMDB-BINARY calibration and evaluation graphs overlap")
    return IMDBBinarySplit(
        calibration=calibration,
        evaluation=evaluation,
        eligible_graph_count=len(graphs),
        distinct_eligible_graph_count=len(unique),
    )
