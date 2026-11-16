"""Leakage-free training of the encoder, community context and denoiser.

Only ``split.train`` edges are used for message passing, positives, the denoiser
input graph and the community centres; ``split.val`` / ``split.val_negatives``
drive encoder early stopping. Pool edges and control pairs never appear in any
training signal (see stability/edge_split.py).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
import torch

from stability.edge_split import EdgeSplit, pair_keys, sample_non_edges
from stability.experiment_config import ExperimentConfig
from stability.graph_data import GraphData
from stability.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.partial_recovery import CommunityContext, compute_community_context
from stability.pilot_config import PilotConfig
from stability.wp7_training import _loss as link_loss

DENOISER_SEED_OFFSET = 1  # torch is reseeded with model.seed + 1 before denoiser training


@dataclass
class TrainedModels:
    encoder: StabilityLPModel
    denoiser: StabilityDiffusion
    context: CommunityContext
    train_adjacency: torch.Tensor
    history: dict[str, Any]
    audit: dict[str, Any] = field(default_factory=dict)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device = 'cuda' but CUDA is not available")
    return torch.device(name)


def normalized_adjacency(node_count: int, edges: np.ndarray, device) -> torch.Tensor:
    """Symmetric D^-1/2 (A + I) D^-1/2 as a sparse tensor (same as wp7 make_adjacency)."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    loops = np.arange(node_count, dtype=np.int64)
    rows = np.concatenate([loops, edges[:, 0], edges[:, 1]])
    cols = np.concatenate([loops, edges[:, 1], edges[:, 0]])
    indices = torch.as_tensor(np.stack([rows, cols]), device=device)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    adjacency = torch.sparse_coo_tensor(indices, values, (node_count, node_count)).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1)
    index = adjacency.indices()
    normalized = adjacency.values() / torch.sqrt(degree[index[0]] * degree[index[1]])
    return torch.sparse_coo_tensor(index, normalized, adjacency.shape).coalesce()


def build_model_args(graph: GraphData, config: ExperimentConfig, device: torch.device):
    model = config.model
    args = SimpleNamespace(
        manifold="PoincareBall",
        cuda=0 if device.type == "cuda" else -1,
        device=str(device),
        feat_dim=graph.features.shape[1],
        dim=model.dim,
        model="HGCN",
        n_nodes=graph.node_count,
        num_layers=model.num_layers,
        act=model.activation,
        dropout=model.dropout,
        bias=int(model.bias),
        use_att=int(model.use_attention),
        local_agg=0,
        r=model.fermi_dirac_r,
        t=model.fermi_dirac_t,
        nb_false_edges=1,
        nb_edges=1,
        # Upstream LPModel builds an MLP(dim, hid1, hid2, dim) head that is never
        # used by encode/decode; values match the previous pilots (wp7_training).
        hid1=8,
        hid2=4,
        final_activation=model.final_activation,
    )
    pilot = PilotConfig(
        curvature=model.curvature,
        tau=config.recovery.tau,
        noise_formula=config.denoiser.noise_formula,
    )
    return pilot.apply_to_args(args), pilot


def _edge_tensor(edges: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(edges, dtype=np.int64).reshape(-1, 2), device=device)


def _validation_auc(encoder, embeddings, val_pos, val_neg) -> float:
    with torch.no_grad():
        scores = torch.cat([encoder.decode(embeddings, val_pos), encoder.decode(embeddings, val_neg)])
    labels = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))])
    return float(roc_auc_score(labels, scores.detach().cpu().numpy()))


def train_models(
    graph: GraphData,
    split: EdgeSplit,
    config: ExperimentConfig,
    device: torch.device,
    audit: bool = False,
    log_every: int = 0,
) -> TrainedModels:
    seed = config.model.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    negative_rng = np.random.default_rng([seed, 1])

    args, _ = build_model_args(graph, config, device)
    encoder = StabilityLPModel(args).to(device)
    features = graph.features.to(device)
    train_adjacency = normalized_adjacency(graph.node_count, split.train, device)
    train_pos = _edge_tensor(split.train, device)
    val_pos = _edge_tensor(split.val, device)
    val_neg = _edge_tensor(split.val_negatives, device)
    forbidden = split.forbidden_negative_keys()
    seen_negative_keys = [] if audit else None

    # ---- encoder: link prediction on train edges, early stopping on val AUC ----
    ae = config.autoencoder
    optimizer = torch.optim.Adam(encoder.parameters(), lr=ae.learning_rate)
    train_loss, val_auc = [], []
    best_auc, best_epoch, best_state = -np.inf, -1, None
    for epoch in range(ae.max_epochs):
        encoder.train()
        negatives = sample_non_edges(
            len(split.train) * ae.negatives_per_positive, graph.node_count, forbidden, negative_rng
        )
        if audit:
            seen_negative_keys.append(pair_keys(negatives, graph.node_count))
        optimizer.zero_grad()
        embeddings = encoder.encode(features, train_adjacency)[2]
        loss = link_loss(encoder, embeddings, train_pos, _edge_tensor(negatives, device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite encoder loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        train_loss.append(float(loss.detach()))

        encoder.eval()
        with torch.no_grad():
            embeddings = encoder.encode(features, train_adjacency)[2]
        auc = _validation_auc(encoder, embeddings, val_pos, val_neg)
        val_auc.append(auc)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = copy.deepcopy(encoder.state_dict())
        if log_every and epoch % log_every == 0:
            print(f"[encoder] epoch {epoch} loss {train_loss[-1]:.4f} val_auc {auc:.4f}", flush=True)
        if epoch - best_epoch >= ae.patience:
            break
    encoder.load_state_dict(best_state)
    encoder.eval()

    # ---- community context on the TRAIN graph ----
    np.random.seed(config.community.seed)
    clusterer = HyperbolicKMeans(
        n_clusters=config.community.n_clusters, curvature=config.model.curvature
    )
    context = compute_community_context(
        encoder,
        features,
        train_adjacency,
        clusterer,
        tuple(range(graph.node_count)),
        max_epochs=config.community.max_epochs,
    )
    with torch.no_grad():
        train_embeddings = encoder.encode(features, train_adjacency)[2].detach()

    # ---- denoiser on train-graph embeddings ----
    torch.manual_seed(seed + DENOISER_SEED_OFFSET)
    denoiser = StabilityDiffusion(
        args, dim=args.dim, curvature=config.model.curvature, device=device
    ).to(device)
    den = config.denoiser
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=den.learning_rate)
    denoiser_loss = []
    for epoch in range(den.epochs):
        denoiser.train()
        optimizer.zero_grad()
        loss = denoiser(
            train_embeddings.clone(),
            train_adjacency,
            context.labels,
            context.centers,
            restrict=True,
            graphset=False,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite denoiser loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        denoiser_loss.append(float(loss.detach()))
        if log_every and epoch % log_every == 0:
            print(f"[denoiser] epoch {epoch} loss {denoiser_loss[-1]:.4f}", flush=True)
    denoiser.eval()
    for model in (encoder, denoiser):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    labels = context.labels.cpu().numpy()
    history = {
        "autoencoder_train_loss": train_loss,
        "autoencoder_val_auc": val_auc,
        "autoencoder_best_epoch": best_epoch,
        "autoencoder_best_val_auc": best_auc,
        "autoencoder_epochs_run": len(train_loss),
        "autoencoder_early_stopped": len(train_loss) < ae.max_epochs,
        "denoiser_loss": denoiser_loss,
        "community_cluster_sizes": np.bincount(labels, minlength=config.community.n_clusters).tolist(),
        "community_nmi_vs_classes": (
            float(normalized_mutual_info_score(graph.class_labels, labels))
            if graph.class_labels is not None
            else None
        ),
    }
    audit_record = {}
    if audit:
        # Edges actually present in the sparse adjacency used for encoder message
        # passing, community context and denoiser training (self-loops dropped).
        index = train_adjacency.indices().cpu().numpy()
        off_diagonal = index[0] != index[1]
        audit_record = {
            "negative_keys": np.unique(np.concatenate(seen_negative_keys)),
            "positive_keys": np.unique(pair_keys(train_pos.cpu().numpy(), graph.node_count)),
            "adjacency_keys": np.unique(
                pair_keys(index[:, off_diagonal].T, graph.node_count)
            ),
        }
    return TrainedModels(encoder, denoiser, context, train_adjacency, history, audit_record)
