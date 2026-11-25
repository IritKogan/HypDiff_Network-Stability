"""Synthetic single graphs for the stability experiment: SBM (community structure) and BA (scale-free).

Both are generated from the config section [synthetic] (all parameters and the seed are saved with the run):
- "SBM": ``sbm_blocks`` blocks of ``sbm_block_size`` nodes; a pair is an edge with probability ``sbm_p_in``
  inside a block and ``sbm_p_out`` between blocks. The blocks are the class labels, used ONLY for analysis
  (same as the Cora classes); the model never sees them.
- "BA": Barabasi-Albert graph (networkx) with ``ba_nodes`` nodes, each new node attached to ``ba_m``
  existing nodes. No class labels.
Synthetic graphs have no node features: every node gets its own one-hot vector (identity features), the
usual choice for featureless graphs.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from hypstab.data.graph import GraphData
from hypstab.data.split import canonical_edges


def sbm_graph(blocks: int, block_size: int, p_in: float, p_out: float, seed: int) -> GraphData:
    rng = np.random.default_rng([seed, 1])
    labels = np.repeat(np.arange(blocks), block_size).astype(np.int64)
    n = len(labels)
    i, j = np.triu_indices(n, 1)
    probability = np.where(labels[i] == labels[j], p_in, p_out)
    keep = rng.random(len(i)) < probability
    edges = canonical_edges(np.stack([i[keep], j[keep]], axis=1), n)
    return GraphData(name="SBM", features=torch.eye(n, dtype=torch.float32), edges=edges, class_labels=labels)


def ba_graph(nodes: int, m: int, seed: int) -> GraphData:
    graph = nx.barabasi_albert_graph(nodes, m, seed=int(np.random.SeedSequence([seed, 2]).generate_state(1)[0]))
    edges = canonical_edges(np.asarray(graph.edges(), dtype=np.int64), nodes)
    return GraphData(name="BA", features=torch.eye(nodes, dtype=torch.float32), edges=edges, class_labels=None)


def synthetic_graph(dataset: str, synthetic) -> GraphData:
    if dataset == "SBM":
        return sbm_graph(synthetic.sbm_blocks, synthetic.sbm_block_size, synthetic.sbm_p_in, synthetic.sbm_p_out,
                         synthetic.seed)
    if dataset == "BA":
        return ba_graph(synthetic.ba_nodes, synthetic.ba_m, synthetic.seed)
    raise ValueError(f"not a synthetic dataset: {dataset}")
