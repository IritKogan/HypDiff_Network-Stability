"""Leakage-free training of the hyperbolic autoencoder (HGCN encoder + Fermi-Dirac decoder).

Uses ONLY ``split.train`` edges: message passing (adjacency) and positives.
Negatives are resampled every epoch from non-edges that are neither validation
negatives nor control pairs (EdgeSplit.forbidden_negative_keys). Validation
edges vs validation negatives drive early stopping (Decision 4). Pool edges and
control pairs never receive a training signal (Decision 6).

Loss (book L_HAE with the Fermi-Dirac likelihood): binary cross-entropy of the
edge logits, positives vs sampled negatives.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from hypstab.config import HypstabConfig
from hypstab.data.graph import GraphData
from hypstab.data.split import EdgeSplit, pair_keys, sample_non_edges
from hypstab.metrics import roc_auc
from hypstab.models.decoder import FermiDiracDecoder
from hypstab.models.hgcn import HGCNEncoder, normalized_adjacency

NEGATIVE_STREAM = 1  # negatives use np.random.default_rng([encoder.seed, 1]) (same as the old code)


class HyperbolicAutoencoder(nn.Module):
    def __init__(self, encoder: HGCNEncoder, decoder: FermiDiracDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, features: Tensor, adjacency: Tensor) -> Tensor:
        return self.encoder(features, adjacency)

    def edge_logits(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        return self.decoder.logits(embeddings, pairs)


def build_autoencoder(feature_dim: int, config: HypstabConfig) -> HyperbolicAutoencoder:
    enc, c = config.encoder, config.geometry.curvature
    encoder = HGCNEncoder(
        feature_dim=feature_dim,
        dim=enc.dim,
        num_layers=enc.num_layers,
        c=c,
        dropout=enc.dropout,
        bias=enc.bias,
        activation=enc.activation,
        final_activation=enc.final_activation,
    )
    decoder = FermiDiracDecoder(config.decoder.fermi_dirac_r, config.decoder.fermi_dirac_t, c)
    return HyperbolicAutoencoder(encoder, decoder)


@dataclass
class AutoencoderResult:
    model: HyperbolicAutoencoder       # best epoch (validation AUC), eval mode
    train_adjacency: Tensor            # sparse, train edges only
    history: dict[str, Any]
    audit: dict[str, np.ndarray]       # pair keys actually used (leakage tests); empty unless audit=True


def link_loss(model: HyperbolicAutoencoder, embeddings: Tensor, positives: Tensor, negatives: Tensor) -> Tensor:
    pos = model.edge_logits(embeddings, positives)
    neg = model.edge_logits(embeddings, negatives)
    return F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + F.binary_cross_entropy_with_logits(
        neg, torch.zeros_like(neg)
    )


@torch.no_grad()
def validation_auc(model: HyperbolicAutoencoder, embeddings: Tensor, val_pos: Tensor, val_neg: Tensor) -> float:
    pos = model.edge_logits(embeddings, val_pos).cpu().numpy()
    neg = model.edge_logits(embeddings, val_neg).cpu().numpy()
    return roc_auc(pos, neg)


def _pairs(edges: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(np.asarray(edges, dtype=np.int64).reshape(-1, 2), device=device)


def train_autoencoder(
    graph: GraphData,
    split: EdgeSplit,
    config: HypstabConfig,
    device: torch.device,
    audit: bool = False,
    log_every: int = 0,
) -> AutoencoderResult:
    enc = config.encoder
    torch.manual_seed(enc.seed)
    negative_rng = np.random.default_rng([enc.seed, NEGATIVE_STREAM])

    model = build_autoencoder(graph.features.shape[1], config).to(device)
    features = graph.features.to(device)
    adjacency = normalized_adjacency(graph.node_count, split.train, device)
    train_pos = _pairs(split.train, device)
    val_pos = _pairs(split.val, device)
    val_neg = _pairs(split.val_negatives, device)
    forbidden = split.forbidden_negative_keys()
    optimizer = torch.optim.Adam(model.parameters(), lr=enc.learning_rate, weight_decay=enc.weight_decay)

    train_loss: list[float] = []
    val_auc: list[float] = []
    negative_keys: list[np.ndarray] = []
    best_auc, best_epoch, best_state = -np.inf, -1, None
    for epoch in range(enc.max_epochs):
        model.train()
        negatives = sample_non_edges(
            len(split.train) * enc.negatives_per_positive, graph.node_count, forbidden, negative_rng
        )
        if audit:
            negative_keys.append(pair_keys(negatives, graph.node_count))
        optimizer.zero_grad()
        loss = link_loss(model, model.encode(features, adjacency), train_pos, _pairs(negatives, device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite autoencoder loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        train_loss.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            auc = validation_auc(model, model.encode(features, adjacency), val_pos, val_neg)
        val_auc.append(auc)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if log_every and epoch % log_every == 0:
            print(f"[autoencoder] epoch {epoch} loss {train_loss[-1]:.4f} val_auc {auc:.4f}", flush=True)
        if epoch - best_epoch >= enc.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    history = {
        "train_loss": train_loss,
        "val_auc": val_auc,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "epochs_run": len(train_loss),
        "early_stopped": len(train_loss) < enc.max_epochs,
    }
    audit_record: dict[str, np.ndarray] = {}
    if audit:
        index = adjacency.indices().cpu().numpy()
        off_diagonal = index[0] != index[1]
        audit_record = {
            "negative_keys": np.unique(np.concatenate(negative_keys)),
            "positive_keys": np.unique(pair_keys(split.train, graph.node_count)),
            "adjacency_keys": np.unique(pair_keys(index[:, off_diagonal].T, graph.node_count)),
        }
    return AutoencoderResult(model=model, train_adjacency=adjacency, history=history, audit=audit_record)
