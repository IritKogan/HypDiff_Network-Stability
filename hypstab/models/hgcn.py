"""Hyperbolic graph convolutional encoder (HGCN, Chami et al. 2019) on the Poincare ball.

Book section 2 (Figure 3): each layer maps node points to the tangent space at
the origin, mixes neighbours there, and maps back to the ball. Per layer:

    h   = W (x)_c x  (+)_c  expmap0(b)          (HypLinear; bias optional)
    h   = expmap0( A_hat @ logmap0(h) )         (HypAgg; A_hat = D^-1/2 (A + I) D^-1/2)
    out = expmap0( act( logmap0(h) ) )          (HypAct)

The input features are first placed in the ball with expmap0. Dropout is applied
to the weight matrix (as in HGCN / HypDiff), not to the features. Curvature c is
fixed (config geometry.curvature), the same in every layer.

Not taken over from the upstream code: attention / local aggregation (off in all
our runs, not in the book), trainable curvature, and the unused MLP / GCN heads
of upstream LPModel.

manifold = "euclidean" (hyperbolic-vs-Euclidean check only): all maps become identities, so the
same code is a plain GCN with the same layers, dropout and activations.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from hypstab.geometry import get_geometry

ACTIVATIONS = ("relu", "leaky_relu", "elu", "tanh", "identity")


def activation_fn(name: str):
    if name == "relu":
        return F.relu
    if name == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=0.01)
    if name == "elu":
        return F.elu
    if name == "tanh":
        return torch.tanh
    if name == "identity":
        return lambda x: x
    raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")


ADJACENCY_NORMS = ("symmetric", "row")


def normalized_adjacency(node_count: int, edges: np.ndarray, device: torch.device | str,
                         mode: str = "symmetric") -> Tensor:
    """Sparse normalised A + I of an undirected edge list (m, 2).

    mode "symmetric": D^-1/2 (A + I) D^-1/2 (GCN; our default since the old code).
    mode "row": D^-1 (A + I), as the upstream HypDiff / HGCN code (utils/data_utils.py process(): normalize(adj +
    sp.eye)), used by the "paper" encoder package (2026-11-27).
    """
    if mode not in ADJACENCY_NORMS:
        raise ValueError(f"unknown adjacency normalisation: {mode}")
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    loops = np.arange(node_count, dtype=np.int64)
    rows = np.concatenate([loops, edges[:, 0], edges[:, 1]])
    cols = np.concatenate([loops, edges[:, 1], edges[:, 0]])
    indices = torch.as_tensor(np.stack([rows, cols]), device=device)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    adjacency = torch.sparse_coo_tensor(indices, values, (node_count, node_count)).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1)
    index = adjacency.indices()
    if mode == "row":
        normalized = adjacency.values() / degree[index[0]]
    else:
        normalized = adjacency.values() / torch.sqrt(degree[index[0]] * degree[index[1]])
    return torch.sparse_coo_tensor(index, normalized, adjacency.shape).coalesce()


class HypLinear(nn.Module):
    """Mobius linear map W (x)_c x, plus optional hyperbolic bias expmap0(b)."""

    def __init__(self, in_features: int, out_features: int, c: float, dropout: float, bias: bool,
                 manifold: str = "poincare"):
        super().__init__()
        self.c = c
        self.geo = get_geometry(manifold)
        self.dropout = dropout
        self.use_bias = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))  # same init as HGCN / HypDiff

    def forward(self, x: Tensor) -> Tensor:
        weight = F.dropout(self.weight, self.dropout, training=self.training)
        geo = self.geo
        out = geo.mobius_matvec(weight, x, self.c)
        if self.bias is not None:
            out = geo.mobius_add(out, geo.expmap0(self.bias.view(1, -1), self.c), self.c)
        return out


class HGCNLayer(nn.Module):
    """One hyperbolic graph convolution: HypLinear -> HypAgg -> HypAct."""

    def __init__(self, in_features: int, out_features: int, c: float, dropout: float, bias: bool, act: str,
                 manifold: str = "poincare"):
        super().__init__()
        self.c = c
        self.geo = get_geometry(manifold)
        self.linear = HypLinear(in_features, out_features, c, dropout, bias, manifold)
        self.act_name = act
        self.act = activation_fn(act)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        geo = self.geo
        h = self.linear(x)
        h = geo.expmap0(torch.sparse.mm(adjacency, geo.logmap0(h, self.c)), self.c)  # HypAgg
        return geo.expmap0(self.act(geo.logmap0(h, self.c)), self.c)  # HypAct


class HGCNEncoder(nn.Module):
    """Features (n, f) + normalised adjacency -> node points in the Poincare ball (n, dim)."""

    def __init__(
        self,
        feature_dim: int,
        dim: int,
        num_layers: int,
        c: float,
        dropout: float,
        bias: bool,
        activation: str,
        final_activation: str,
        manifold: str = "poincare",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.c = c
        self.geo = get_geometry(manifold)
        dims = [feature_dim] + [dim] * num_layers
        acts = [activation] * (num_layers - 1) + [final_activation]
        self.layers = nn.ModuleList(
            HGCNLayer(dims[i], dims[i + 1], c, dropout, bias, acts[i], manifold) for i in range(num_layers)
        )

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        x = self.geo.expmap0(features, self.c)
        for layer in self.layers:
            x = layer(x, adjacency)
        return x
