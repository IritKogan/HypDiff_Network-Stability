"""Synthetic single graphs for the stability experiment: SBM (community structure) and BA (scale-free).

Generated with the standard torch_geometric functions (the library HypDiff itself uses), from the config
section [synthetic]; all parameters and the seed are saved with the run. Default parameters follow the
node-level synthetic datasets of the HypDiff paper (section 4.1):
- "SBM": torch_geometric.utils.stochastic_blockmodel_graph; ``sbm_blocks`` equal blocks of
  ``sbm_block_size`` nodes, edge probability ``sbm_p_in`` inside a block and ``sbm_p_out`` between blocks
  (paper: 1000 nodes, 5 communities, p = 0.21, q = 0.025). The blocks are the class labels, used ONLY for
  analysis (as the Cora classes); the model never sees them.
- "BA": torch_geometric.utils.barabasi_albert_graph with ``ba_nodes`` nodes and a FIXED number ``ba_m`` of
  edges per new node. The paper attaches "between 1 and 10" edges per new node, which no standard generator
  offers; we use the fixed mean-like value 5 (students, 2026-11-25). In this generator a new node may draw the
  same neighbour twice (merged), so a node can get fewer than ``ba_m`` distinct edges. No class labels.
The torch_geometric functions draw from the global torch / numpy generators: they are seeded from
``synthetic.seed`` inside a saved-and-restored RNG state, so the graph is reproducible (same library
versions) and the rest of the run's random streams are not affected.
Synthetic graphs have no node features: every node gets its own one-hot vector (identity features), so the
experiment tests the graph structure only (torch_geometric's optional SBM features are derived from the block
labels and are not used).
"""

from __future__ import annotations

import numpy as np
import torch

from hypstab.data.graph import GraphData
from hypstab.data.split import canonical_edges


def _seeded(seed: int, generate):
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            return generate()
        finally:
            np.random.set_state(numpy_state)


def _canonical(edge_index: torch.Tensor, n: int) -> np.ndarray:
    pairs = edge_index.t().cpu().numpy().astype(np.int64)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return canonical_edges(np.unique(np.sort(pairs, axis=1), axis=0), n)


def sbm_graph(blocks: int, block_size: int, p_in: float, p_out: float, seed: int) -> GraphData:
    from torch_geometric.utils import stochastic_blockmodel_graph

    probs = torch.full((blocks, blocks), float(p_out))
    probs.fill_diagonal_(float(p_in))
    edge_index = _seeded(seed, lambda: stochastic_blockmodel_graph([block_size] * blocks, probs, directed=False))
    n = blocks * block_size
    labels = np.repeat(np.arange(blocks), block_size).astype(np.int64)
    return GraphData(name="SBM", features=torch.eye(n, dtype=torch.float32), edges=_canonical(edge_index, n),
                     class_labels=labels)


def ba_graph(nodes: int, m: int, seed: int) -> GraphData:
    from torch_geometric.utils import barabasi_albert_graph

    edge_index = _seeded(seed, lambda: barabasi_albert_graph(nodes, m))
    return GraphData(name="BA", features=torch.eye(nodes, dtype=torch.float32), edges=_canonical(edge_index, nodes),
                     class_labels=None)


def synthetic_graph(dataset: str, synthetic) -> GraphData:
    if dataset == "SBM":
        return sbm_graph(synthetic.sbm_blocks, synthetic.sbm_block_size, synthetic.sbm_p_in, synthetic.sbm_p_out,
                         synthetic.seed)
    if dataset == "BA":
        return ba_graph(synthetic.ba_nodes, synthetic.ba_m, synthetic.seed)
    raise ValueError(f"not a synthetic dataset: {dataset}")
