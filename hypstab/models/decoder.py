"""Fermi-Dirac edge decoder (book section 2, HypDiff / HGCN).

    P(edge i-j) = 1 / (exp((d_ij - r) / t) + 1) = sigmoid(-(d_ij - r) / t)

As in HGCN and the HypDiff code, d_ij is the SQUARED Poincare distance (the book
writes the distance x_st; the code of both papers uses its square). Training and
evaluation work with the logit -(d_ij - r) / t: it is numerically safer than the
probability and ranks pairs identically.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hypstab.geometry import poincare as pb


class FermiDiracDecoder(nn.Module):
    def __init__(self, r: float, t: float, c: float):
        super().__init__()
        if t <= 0:
            raise ValueError("Fermi-Dirac temperature t must be > 0")
        self.r = r
        self.t = t
        self.c = c

    def logits(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        """Edge logits for index pairs (m, 2) of node embeddings (n, dim)."""
        sqdist = pb.sqdistance(embeddings[pairs[:, 0]], embeddings[pairs[:, 1]], self.c)
        return -(sqdist - self.r) / self.t

    def probabilities(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(embeddings, pairs))
