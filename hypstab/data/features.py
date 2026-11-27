"""Structural node features for graphs without (or without their) node attributes.

Students, 2026-11-27: "Cora without text features" means replacing the text with STRUCTURAL
features, not removing every feature (Claude's identity-only implementation of 2026-11-26 was a
mistake; mode "identity" is kept for reference). Recipe copied from the upstream HypDiff code:

- "identity_degree" (upstream data.py, Graph dataset class, lines ~297-306): one-hot identity plus
  ONE column (degree + 1) / n. Upstream sums the adjacency AFTER adding self-loops, hence the + 1.

Upstream's other recipe (data.py node_festure_creator: deg / n, deg / n, diag(A^3) / n^2,
diag(A^4) / n^3) was implemented and REMOVED (students, 2026-11-27): with upstream's scaling the
last two columns are ~1e-6 .. 1e-10 on Cora, so it carries only the degree, which identity_degree
already has; a rescaled version would have no source.

NO LEAKAGE: A is built from the TRAIN edges only (split.train); validation and pool edges never
influence the features. The features are computed once and stay fixed (also for damaged graphs).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from hypstab.data.graph import GraphData

TRAIN_EDGE_FEATURE_MODES = ("identity_degree",)


def train_adjacency(train_edges: np.ndarray, node_count: int) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency (no self-loops) of the train edges."""
    edges = np.asarray(train_edges, dtype=np.int64).reshape(-1, 2)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(node_count, node_count))
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def identity_degree_features(train_edges: np.ndarray, node_count: int) -> np.ndarray:
    """Upstream HypDiff featureless-graph recipe: [I_n, (degree + 1) / n]."""
    degree = np.asarray(train_adjacency(train_edges, node_count).sum(axis=1)).reshape(-1)
    column = ((degree + 1.0) / node_count).reshape(-1, 1)
    return np.concatenate([np.eye(node_count), column], axis=1)


def with_train_edge_features(graph: GraphData, train_edges: np.ndarray, mode: str) -> GraphData:
    """Replace the node features by a train-edge structural recipe; other modes: graph unchanged."""
    if mode not in TRAIN_EDGE_FEATURE_MODES:
        return graph
    values = identity_degree_features(train_edges, graph.node_count)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite structural features ({mode})")
    return GraphData(name=graph.name, features=torch.as_tensor(values, dtype=torch.float32).contiguous(),
                     edges=graph.edges, class_labels=graph.class_labels)
