"""Graph datasets for the edge-recovery experiments.

Moved from stability/graph_data.py (phase-b-work, bec9111) without changes.
Class labels are kept for analysis only and are never used in training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from hypstab.data.split import canonical_edges


@dataclass(frozen=True)
class GraphData:
    name: str
    features: torch.Tensor          # (n, f) float32
    edges: np.ndarray               # (m, 2) int64, canonical (source < target), sorted
    class_labels: np.ndarray | None  # (n,) used ONLY for analysis, never for training

    @property
    def node_count(self) -> int:
        return int(self.features.shape[0])

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(tuple(self.features.shape)).encode("ascii"))
        digest.update(self.features.numpy().tobytes())
        digest.update(self.edges.tobytes())
        if self.class_labels is not None:
            digest.update(self.class_labels.astype(np.int64).tobytes())
        return digest.hexdigest()


def load_planetoid(name: str, root: str | Path, download: bool) -> GraphData:
    """Planetoid graph (e.g. Cora) with row-normalised features, undirected edges."""
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    root = Path(root)
    processed = root / name / "processed" / "data.pt"
    raw = root / name / "raw"
    if not download and not processed.is_file() and not raw.is_dir():
        raise FileNotFoundError(f"{name} is not cached under {root}; set data.download = true")
    data = Planetoid(root=str(root), name=name, transform=NormalizeFeatures())[0]
    edge_index = data.edge_index.t().numpy()
    edge_index = edge_index[edge_index[:, 0] != edge_index[:, 1]]
    keys = np.unique(np.sort(edge_index, axis=1), axis=0)
    features = data.x.detach().cpu().to(torch.float32).contiguous()
    if not torch.isfinite(features).all():
        raise ValueError(f"{name}: non-finite node features")
    return GraphData(
        name=name,
        features=features,
        edges=canonical_edges(keys, features.shape[0]),
        class_labels=data.y.numpy().astype(np.int64),
    )


def with_identity_features(graph: GraphData) -> GraphData:
    """The same graph with one one-hot vector per node: the model sees the structure only.

    Advisor (2026-11-26): run Cora without its bag-of-words features. Edges and class labels are unchanged;
    the features carry no information about any edge, so no leakage is possible through them.
    """
    n = graph.node_count
    return GraphData(name=graph.name, features=torch.eye(n, dtype=torch.float32), edges=graph.edges,
                     class_labels=graph.class_labels)


def load_graph(dataset: str, root: str | Path, download: bool, synthetic=None,
               features: str = "original") -> GraphData:
    """Planetoid graphs from disk; "SBM" / "BA" generated from the config section [synthetic].

    features: "original" (the dataset's own features; SBM / BA have identity features anyway) or
    "identity" (one-hot per node, config data.features).
    """
    if features not in {"original", "identity"}:
        raise ValueError(f"unsupported feature mode: {features}")
    if dataset in {"Cora", "CiteSeer", "PubMed"}:
        graph = load_planetoid(dataset, root, download)
    elif dataset in {"SBM", "BA"}:
        if synthetic is None:
            raise ValueError(f"{dataset} needs the [synthetic] config section")
        from hypstab.data.synthetic import synthetic_graph

        graph = synthetic_graph(dataset, synthetic)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return with_identity_features(graph) if features == "identity" else graph
