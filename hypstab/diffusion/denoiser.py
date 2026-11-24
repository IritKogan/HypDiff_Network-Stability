"""Denoising network: residual GCN that predicts x_0 from (x_t, graph, t / T).

    h   = W_in x_t + MLP_time(t / T)
    h  <- h + GELU( GCN( LayerNorm(h) ) )        (``layers`` times)
    x_0 = x_t + W_out h                           (W_out starts at zero: identity at init)

GCN(h) = A_hat (h W) + b with A_hat = D^-1/2 (A + I) D^-1/2 of the (damaged) graph,
the same propagation as torch_geometric's GCNConv (tested), written here so the model
does not depend on torch_geometric. Chosen on validation in the old code (Decision 8:
"residual_gcn"); the upstream HypDiff GCN decoder was not better.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: Tensor, adjacency: Tensor) -> Tensor:
        return torch.sparse.mm(adjacency, h @ self.weight) + self.bias


class ResidualGCNDenoiser(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, layers: int):
        super().__init__()
        if layers < 1 or hidden_dim < 1:
            raise ValueError("layers and hidden_dim must be >= 1")
        self.input = nn.Linear(dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.convs = nn.ModuleList(GraphConv(hidden_dim, hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x_t: Tensor, adjacency: Tensor, t_normalized: Tensor) -> Tensor:
        h = self.input(x_t) + self.time(t_normalized.reshape(-1, 1).to(x_t.dtype))
        for norm, conv in zip(self.norms, self.convs):
            h = h + F.gelu(conv(norm(h), adjacency))
        return x_t + self.output(h)
