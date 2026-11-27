"""Leakage-free training of the hyperbolic autoencoder (HGCN encoder + Fermi-Dirac decoder).

Uses ONLY ``split.train`` edges: message passing (adjacency) and positives.
Negatives are resampled every epoch from non-edges that are neither validation
negatives nor control pairs (EdgeSplit.forbidden_negative_keys). Validation
edges vs validation negatives drive early stopping (Decision 4). Pool edges and
control pairs never receive a training signal (Decision 6).

Loss (book L_HAE with the Fermi-Dirac likelihood): binary cross-entropy of the
edge logits, positives vs sampled negatives.

Training on damaged train graphs (encoder.edge_drop_mode, students 2026-11-27): in every
epoch each train edge is dropped from the encoder's INPUT with probability q ("fixed": q =
edge_drop_fraction, the paper's 2% edge dropping; "uniform": q ~ U(0, edge_drop_fraction), as
the denoiser). edge_drop_loss "dropped": the positives are the dropped edges only (the task of
the experiment: recover edges missing from the input); "dropped_and_kept": mean BCE of the
dropped edges + mean BCE of the kept edges (as the denoiser). Each group's BCE is a mean, so a
small group weighs as much as a large one. An epoch in which no edge was dropped makes no update
(loss recorded as NaN, epoch listed in edge_drop_skipped_epochs). Validation always encodes the
full train graph. Mode "none" = the original training, bit for bit (the drop stream is unused).
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
EDGE_DROP_STREAM = 2  # dropped train edges use np.random.default_rng([encoder.seed, 2])


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
        manifold=config.geometry.manifold,
    )
    decoder = FermiDiracDecoder(config.decoder.fermi_dirac_r, config.decoder.fermi_dirac_t, c,
                                config.geometry.manifold)
    return HyperbolicAutoencoder(encoder, decoder)


@dataclass
class AutoencoderResult:
    model: HyperbolicAutoencoder       # best epoch (validation AUC), eval mode
    train_adjacency: Tensor            # sparse, train edges only
    history: dict[str, Any]
    audit: dict[str, np.ndarray]       # pair keys actually used (leakage tests); empty unless audit=True


def grouped_link_loss(model: HyperbolicAutoencoder, embeddings: Tensor, positive_groups: list[Tensor],
                      negatives: Tensor) -> Tensor:
    """Sum of the mean BCE of every positive group (target 1) and of the negatives (target 0).

    With one positive group this is exactly ``link_loss``.
    """
    terms = []
    for pairs in positive_groups:
        logits = model.edge_logits(embeddings, pairs)
        terms.append(F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)))
    neg = model.edge_logits(embeddings, negatives)
    terms.append(F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
    return torch.stack(terms).sum()


def draw_edge_drop(mode: str, fraction: float, edge_count: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Drop probability q of one epoch and the boolean mask of dropped train edges."""
    if mode == "none":
        return 0.0, np.zeros(edge_count, dtype=bool)
    q = fraction if mode == "fixed" else float(rng.uniform(0.0, fraction))
    return q, rng.random(edge_count) < q


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
    drop_rng = np.random.default_rng([enc.seed, EDGE_DROP_STREAM])
    dropping = enc.edge_drop_mode != "none"

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
    drop_fractions: list[float] = []
    drop_counts: list[int] = []
    skipped: list[int] = []
    epoch_adjacency_keys: list[np.ndarray] = []
    epoch_positive_keys: list[np.ndarray] = []
    best_auc, best_epoch, best_state = -np.inf, -1, None
    for epoch in range(enc.max_epochs):
        model.train()
        negatives = sample_non_edges(
            len(split.train) * enc.negatives_per_positive, graph.node_count, forbidden, negative_rng
        )
        if audit:
            negative_keys.append(pair_keys(negatives, graph.node_count))
        optimizer.zero_grad()
        if not dropping:
            loss = link_loss(model, model.encode(features, adjacency), train_pos, _pairs(negatives, device))
            if audit:
                epoch_adjacency_keys.append(pair_keys(split.train, graph.node_count))
                epoch_positive_keys.append(pair_keys(split.train, graph.node_count))
        else:
            q, dropped = draw_edge_drop(enc.edge_drop_mode, enc.edge_drop_fraction, len(split.train), drop_rng)
            drop_fractions.append(q)
            drop_counts.append(int(dropped.sum()))
            kept_edges, dropped_edges = split.train[~dropped], split.train[dropped]
            if audit:
                epoch_adjacency_keys.append(pair_keys(kept_edges, graph.node_count))
                epoch_positive_keys.append(pair_keys(
                    dropped_edges if enc.edge_drop_loss == "dropped" else split.train, graph.node_count))
            if len(dropped_edges) == 0:
                loss = None
            else:
                epoch_adjacency = normalized_adjacency(graph.node_count, kept_edges, device)
                groups = [_pairs(dropped_edges, device)]
                if enc.edge_drop_loss == "dropped_and_kept" and len(kept_edges):
                    groups.append(_pairs(kept_edges, device))
                loss = grouped_link_loss(model, model.encode(features, epoch_adjacency), groups,
                                         _pairs(negatives, device))
        if loss is None:
            skipped.append(epoch)
            train_loss.append(float("nan"))
        else:
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
        "edge_drop_fractions": drop_fractions,
        "edge_drop_counts": drop_counts,
        "edge_drop_skipped_epochs": skipped,
    }
    audit_record: dict[str, np.ndarray] = {}
    if audit:
        index = adjacency.indices().cpu().numpy()
        off_diagonal = index[0] != index[1]
        audit_record = {
            "negative_keys": np.unique(np.concatenate(negative_keys)),
            "positive_keys": np.unique(np.concatenate(epoch_positive_keys)),
            "adjacency_keys": np.unique(pair_keys(index[:, off_diagonal].T, graph.node_count)),
            # per epoch: the train edges in the encoder's input / the positives of the loss
            "epoch_adjacency_keys": epoch_adjacency_keys,
            "epoch_positive_keys": epoch_positive_keys,
        }
    return AutoencoderResult(model=model, train_adjacency=adjacency, history=history, audit=audit_record)
