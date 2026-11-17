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


class ConditionalGCNDenoiser(nn.Module):
    """Conditional version of upstream HypDiff ``gcndec`` (documented copy).

    Copy of ``models.direction_diffusion.gcndec`` (same layers, widths and GELU
    activations); the only change is the input: ``[x_t, condition, t]`` (2 * dim + 1)
    instead of ``[x_t, t]`` (dim + 1). ``condition`` is the standardised tangent
    embedding of the damaged graph (Decision 9, conditional diffusion).
    """

    conditional = True

    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = GCNConv(2 * dim + 1, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, dim)

    def forward(self, x, edge_index, t, condition):
        x = torch.cat((x.to(torch.float32), condition.to(torch.float32),
                       t.to(torch.float32).reshape(-1, 1)), dim=1)
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.conv2(x, edge_index))
        x = F.gelu(self.conv3(x, edge_index))
        x = F.gelu(self.conv4(x, edge_index))
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


class ConditionalResidualGCNDenoiser(nn.Module):
    """x0_hat = condition + correction(x_t, condition, A, t).

    The output layer starts at zero, so the untrained network returns the damaged
    embedding (the condition); training learns what to add. Same block structure as
    ``ResidualGCNDenoiser``.
    """

    conditional = True

    def __init__(self, dim: int, hidden_dim: int, layers: int):
        super().__init__()
        if layers < 1 or hidden_dim < 1:
            raise ValueError("layers and hidden_dim must be positive")
        self.input = nn.Linear(2 * dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, edge_index, t, condition):
        condition = condition.to(torch.float32)
        h = self.input(torch.cat((x.to(torch.float32), condition), dim=1))
        h = h + self.time(t.to(torch.float32).reshape(-1, 1))
        for norm, conv in zip(self.norms, self.convs):
            h = h + F.gelu(conv(norm(h), edge_index))
        return condition + self.output(h)


GEOMETRIC_COORDINATES = 2   # per node: community offset (distance to its prototype), popularity (radial)


class GeometricPriorNet(nn.Module):
    """Per-node Gaussian prior over the geometric coefficients, from the damaged graph:
    returns (mean, log_variance), each (nodes, 2). Starts at mean 0, variance ``init_variance``."""

    def __init__(self, dim: int, hidden_dim: int, layers: int, init_variance: float):
        super().__init__()
        if init_variance <= 0:
            raise ValueError("init_variance must be positive")
        self.input = nn.Linear(dim, hidden_dim)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, 2 * GEOMETRIC_COORDINATES)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.zero_()
            self.output.bias[GEOMETRIC_COORDINATES:] = float(torch.log(torch.tensor(init_variance)))

    def forward(self, condition, edge_index):
        h = self.input(condition.to(torch.float32))
        for norm, conv in zip(self.norms, self.convs):
            h = h + F.gelu(conv(norm(h), edge_index))
        out = self.output(h)
        return out[:, :GEOMETRIC_COORDINATES], out[:, GEOMETRIC_COORDINATES:].clamp(-12.0, 4.0)


class GeometricDenoiserNet(nn.Module):
    """Correction F(innovation, condition, A, t) for the preconditioned 2-D diffusion over the
    geometric coefficients. Output layer zero-initialised (F = 0: exact Gaussian posterior)."""

    def __init__(self, dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.input = nn.Linear(dim + GEOMETRIC_COORDINATES, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, GEOMETRIC_COORDINATES)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, innovation, condition, edge_index, t):
        h = self.input(torch.cat((innovation.to(torch.float32), condition.to(torch.float32)), dim=1))
        h = h + self.time(t.to(torch.float32).reshape(-1, 1))
        for norm, conv in zip(self.norms, self.convs):
            h = h + F.gelu(conv(norm(h), edge_index))
        return self.output(h)
