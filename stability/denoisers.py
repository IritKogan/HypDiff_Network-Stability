"""Project denoiser networks (alternatives to upstream ``gcndec``).

``ResidualGCNDenoiser`` predicts x0 as ``x_t + correction(x_t, A, t)``. The
output layer starts at zero, so the untrained network is the identity map:
at small t (partial-noise recovery, t* = 7) the input is already close to x0
and the network only has to learn the correction. Upstream ``gcndec`` has to
re-create x0 through four GCN layers without skip connections and, on Cora,
was measurably worse than returning its input (MSE 0.023 vs 0.001 at t = 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ResidualGCNDenoiser(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, layers: int):
        super().__init__()
        if layers < 1 or hidden_dim < 1:
            raise ValueError("layers and hidden_dim must be positive")
        self.input = nn.Linear(dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, edge_index, t):
        """Same call signature as upstream gcndec: (x_t, edge_index, t / T)."""
        x = x.to(torch.float32)
        h = self.input(x) + self.time(t.to(torch.float32).reshape(-1, 1))
        for norm, conv in zip(self.norms, self.convs):
            h = h + F.gelu(conv(norm(h), edge_index))
        return x + self.output(h)
