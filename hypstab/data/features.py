"""Structural node features for graphs without (or without their) node attributes.

Students, 2026-11-27: "Cora without text features" means replacing the text with STRUCTURAL
features, not removing every feature (Claude's identity-only implementation of 2026-11-26 was a
mistake; mode "identity" is kept for reference). Two recipes, both copied from the upstream HypDiff
code (data.py; featureless graph datasets), to be compared on the validation protocol:

- "identity_degree" (upstream data.py, Graph dataset class, lines ~297-306): one-hot identity plus
  ONE column (degree + 1) / n. Upstream sums the adjacency AFTER adding self-loops, hence the + 1.
- "structural" (upstream data.py, node_festure_creator(adj, steps=3), lines 903-931, without its
  optional random columns and without identity): columns
      deg / n,  diag(A A^T) / n,  diag(A^3) / n^2,  diag(A^4) / n^3
  diag(A A^T) equals the degree again; diag(A^3) = 2 x (triangles through the node); diag(A^4) =
  closed walks of length 4. The scaling is upstream's (it makes the last columns very small).

NO LEAKAGE: A is built from the TRAIN edges only (split.train); validation and pool edges never
influence the features. The features are computed once and stay fixed (also for damaged graphs).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from hypstab.data.graph import GraphData

TRAIN_EDGE_FEATURE_MODES = ("identity_degree", "structural")


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


def structural_features(train_edges: np.ndarray, node_count: int) -> np.ndarray:
    """Upstream node_festure_creator(adj, steps=3, rand_dim=0, Use_identity=False)."""
    adj = train_adjacency(train_edges, node_count)
    n = float(node_count)
    columns = [np.asarray(adj.sum(axis=1)).reshape(-1) / n]
    walk = adj
    for step in range(3):
        walk = walk @ adj.T
        columns.append(walk.diagonal() / n ** (step + 1))
    return np.stack(columns, axis=1)


def with_train_edge_features(graph: GraphData, train_edges: np.ndarray, mode: str) -> GraphData:
    """Replace the node features by a train-edge structural recipe; other modes: graph unchanged."""
    if mode not in TRAIN_EDGE_FEATURE_MODES:
        return graph
    n = graph.node_count
    values = identity_degree_features(train_edges, n) if mode == "identity_degree" else structural_features(
        train_edges, n)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite structural features ({mode})")
    return GraphData(name=graph.name, features=torch.as_tensor(values, dtype=torch.float32).contiguous(),
                     edges=graph.edges, class_labels=graph.class_labels)
