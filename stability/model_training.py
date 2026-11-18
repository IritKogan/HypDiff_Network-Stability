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
        denoiser_architecture=config.denoiser.architecture,
        denoiser_hidden_dim=config.denoiser.hidden_dim,
        denoiser_layers=config.denoiser.layers,
        clip_x0=config.denoiser.clip_x0,
        radial_delta=config.denoiser.radial_delta,
        angular_noise=config.denoiser.angular_noise,
        denoiser_parameterization=config.denoiser.parameterization,
        prior_variance=config.denoiser.prior_variance,
        recovery_sampler=config.recovery.sampler,
        recovery_eta=config.recovery.eta,
        recovery_steps=config.recovery.steps,
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


def train_encoder(
    graph: GraphData,
    split: EdgeSplit,
    config: ExperimentConfig,
    device: torch.device,
    audit: bool = False,
    log_every: int = 0,
):
    """Link-prediction training on train edges, early stopping on val AUC.

    Returns (encoder, args, train_adjacency, history, audit_record)."""
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
    seen_negative_keys = []

    ae = config.autoencoder
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=ae.learning_rate, weight_decay=ae.weight_decay
    )
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
    history = {
        "autoencoder_train_loss": train_loss,
        "autoencoder_val_auc": val_auc,
        "autoencoder_best_epoch": best_epoch,
        "autoencoder_best_val_auc": best_auc,
        "autoencoder_epochs_run": len(train_loss),
        "autoencoder_early_stopped": len(train_loss) < ae.max_epochs,
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
    return encoder, args, train_adjacency, history, audit_record


def build_context(graph, config, encoder, train_adjacency, device) -> CommunityContext:
    """Hyperbolic K-means on train-graph embeddings (train edges only)."""
    np.random.seed(config.community.seed)
    clusterer = HyperbolicKMeans(
        n_clusters=config.community.n_clusters,
        curvature=config.model.curvature,
        init=config.community.init,
    )
    return compute_community_context(
        encoder,
        graph.features.to(device),
        train_adjacency,
        clusterer,
        tuple(range(graph.node_count)),
        max_epochs=config.community.max_epochs,
    )


def train_denoiser(graph, split, config, device, encoder, args, train_adjacency, context,
                   log_every: int = 0):
    """Train the denoiser on the TRAIN graph only.

    generative          : upstream objective (noised clean embeddings -> clean).
    damage_aware        : edge-dropped train graph (+ noise) -> intact train graph (MSE).
    edge_reconstruction : edge-dropped train graph -> one denoising step -> decoder;
                          BCE(dropped train edges) + BCE(kept train edges) + BCE(negatives).
    conditional         : (Decision 9) noised CLEAN train-graph embedding at t < train_t_max,
                          conditioned on the edge-dropped train graph -> x0; MSE
                          + edge_loss_weight * the edge_reconstruction BCE on the prediction.
                          parameterization "geometric": the core loss is MSE(mu) + prior NLL
                          + 2-D denoising loss (history key "denoiser_mse" holds this core
                          loss) and the decoder BCE acts on the deterministic mean mu.
                          Negatives use the encoder's sampler (never val negatives or
                          control pairs); the encoder and decoder stay frozen.
    """
    seed = config.model.seed
    den = config.denoiser
    features = graph.features.to(device)
    with torch.no_grad():
        train_embeddings = encoder.encode(features, train_adjacency)[2].detach()
    torch.manual_seed(seed + DENOISER_SEED_OFFSET)
    denoiser = StabilityDiffusion(
        args, dim=args.dim, curvature=config.model.curvature, device=device
    ).to(device)
    # fail before training if the configured stochastic sampler cannot run at t*
    denoiser.check_stochastic_range(config.recovery.t_star)
    prior_variance = None
    if den.parameterization in ("preconditioned", "geometric"):
        prior_variance = den.prior_variance or estimate_prior_variance(
            graph, split, config, device, encoder, train_embeddings, context)
        denoiser.prior_variance.fill_(prior_variance)
        if den.parameterization == "geometric":
            denoiser.initialise_geometric_prior(prior_variance)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=den.learning_rate)
    drop_rng = np.random.default_rng([seed, 2])
    negative_rng = np.random.default_rng([seed, 3])
    forbidden = split.forbidden_negative_keys()
    encoder_requires_grad = [p.requires_grad for p in encoder.parameters()]
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    losses, drops, mses, skipped = [], [], [], []

    def skip(epoch: int, reason: str) -> None:
        """Non-finite loss / gradients: no parameter update for this epoch (denoiser.nonfinite_max_skips)."""
        skipped.append({"epoch": epoch, "reason": reason})
        if len(skipped) > den.nonfinite_max_skips:
            raise RuntimeError(f"non-finite denoiser {reason} at epoch {epoch} "
                               f"({len(skipped)} skipped epochs > nonfinite_max_skips = {den.nonfinite_max_skips})")
        optimizer.zero_grad()
        losses.append(float("nan"))
        if log_every:
            print(f"[denoiser] epoch {epoch}: non-finite {reason}, step skipped ({len(skipped)} so far)", flush=True)

    for epoch in range(den.epochs):
        denoiser.train()
        optimizer.zero_grad()
        if den.training == "generative":
            loss = denoiser(
                train_embeddings.clone(), train_adjacency, context.labels, context.centers,
                restrict=True, graphset=False,
            )
        else:
            fraction = float(drop_rng.uniform(0.0, den.train_max_drop))
            keep = drop_rng.random(len(split.train)) >= fraction
            damaged_adjacency = normalized_adjacency(graph.node_count, split.train[keep], device)
            with torch.no_grad():
                damaged_embeddings = encoder.encode(features, damaged_adjacency)[2].detach()
            if den.training == "conditional":
                mse, recovered = denoiser.conditional_loss(
                    train_embeddings, damaged_embeddings, damaged_adjacency,
                    context.labels, context.centers, den.train_t_max, den.t_sampling,
                )
                loss = mse
                mses.append(float(mse.detach()))
                if den.edge_loss_weight > 0:
                    negatives = _edge_tensor(
                        sample_non_edges(len(split.train), graph.node_count, forbidden, negative_rng), device
                    )
                    loss = loss + den.edge_loss_weight * _edge_reconstruction_loss(
                        encoder, recovered,
                        dropped=_edge_tensor(split.train[~keep], device),
                        kept=_edge_tensor(split.train[keep], device),
                        negatives=negatives,
                    )
            elif den.training == "damage_aware":
                loss = denoiser.restoration_loss(
                    train_embeddings, damaged_embeddings, damaged_adjacency,
                    context.labels, context.centers, den.train_t_max,
                )
            else:
                t = torch.randint(0, den.train_t_max, (graph.node_count,), device=device).long()
                recovered = denoiser.denoise_once(
                    damaged_embeddings, damaged_adjacency, context.labels, context.centers, t
                )
                negatives = _edge_tensor(
                    sample_non_edges(len(split.train), graph.node_count, forbidden, negative_rng), device
                )
                loss = _edge_reconstruction_loss(
                    encoder, recovered,
                    dropped=_edge_tensor(split.train[~keep], device),
                    kept=_edge_tensor(split.train[keep], device),
                    negatives=negatives,
                )
            drops.append(fraction)
        if not torch.isfinite(loss):
            skip(epoch, "loss")
            continue
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in denoiser.parameters() if p.grad is not None):
            skip(epoch, "gradients")
            continue
        optimizer.step()
        losses.append(float(loss.detach()))
        if log_every and epoch % log_every == 0:
            print(f"[denoiser] epoch {epoch} loss {losses[-1]:.4f}", flush=True)
    denoiser.eval()
    for parameter in denoiser.parameters():
        parameter.requires_grad_(False)
    for parameter, flag in zip(encoder.parameters(), encoder_requires_grad):
        parameter.requires_grad_(flag)
    if not all(torch.isfinite(p).all() for p in denoiser.parameters()):
        raise RuntimeError("denoiser parameters are non-finite after training")
    history = {"denoiser_loss": losses, "denoiser_drop_fractions": drops, "denoiser_skipped_epochs": skipped}
    if mses:
        history["denoiser_mse"] = mses
    if prior_variance is not None:
        history["prior_variance"] = float(prior_variance)
    return denoiser, history


def estimate_prior_variance(graph, split, config, device, encoder, train_embeddings, context,
                            draws: int = 16) -> float:
    """v = mean Var(x0 - c) for the preconditioned model: x0 / c = standardised tangent
    embeddings of the intact / edge-dropped TRAIN graph (same standardisation as
    conditional_loss). Uses train edges only and its own RNG stream."""
    from manifolds.poincare import PoincareBall

    manifold = PoincareBall()
    rng = np.random.default_rng([config.model.seed, 4])
    features = graph.features.to(device)
    labels = context.labels.to(device)
    prototypes = context.centers.to(device=device, dtype=train_embeddings.dtype)[labels]
    curvature = config.model.curvature
    values = []
    with torch.no_grad():
        tangent_clean = manifold.logmap(prototypes, train_embeddings, c=curvature)
        for _ in range(draws):
            fraction = float(rng.uniform(0.0, config.denoiser.train_max_drop))
            keep = rng.random(len(split.train)) >= fraction
            damaged = encoder.encode(features, normalized_adjacency(graph.node_count, split.train[keep], device))[2]
            tangent_damaged = manifold.logmap(prototypes, damaged, c=curvature)
            std = torch.std(tangent_damaged)
            values.append(float(torch.mean(((tangent_clean - tangent_damaged) / std) ** 2)))
    return float(np.mean(values))


def _edge_reconstruction_loss(encoder, embeddings, dropped, kept, negatives):
    """Mean BCE (Fermi-Dirac logits) of three equally weighted groups."""
    def logits(pairs):
        distance = encoder.manifold.sqdist(embeddings[pairs[:, 0]], embeddings[pairs[:, 1]], encoder.c)
        return -(distance - encoder.dc.r) / encoder.dc.t

    import torch.nn.functional as F

    terms = []
    for pairs, target in ((dropped, 1.0), (kept, 1.0), (negatives, 0.0)):
        if pairs.shape[0] == 0:
            continue
        values = logits(pairs)
        terms.append(F.binary_cross_entropy_with_logits(values, torch.full_like(values, target)))
    return torch.stack(terms).sum()


def train_models(
    graph: GraphData,
    split: EdgeSplit,
    config: ExperimentConfig,
    device: torch.device,
    audit: bool = False,
    log_every: int = 0,
) -> TrainedModels:
    encoder, args, train_adjacency, history, audit_record = train_encoder(
        graph, split, config, device, audit=audit, log_every=log_every
    )
    context = build_context(graph, config, encoder, train_adjacency, device)
    denoiser, denoiser_history = train_denoiser(
        graph, split, config, device, encoder, args, train_adjacency, context, log_every
    )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    labels = context.labels.cpu().numpy()
    history.update(denoiser_history)
    history.update({
        "community_cluster_sizes": np.bincount(labels, minlength=config.community.n_clusters).tolist(),
        "community_nmi_vs_classes": (
            float(normalized_mutual_info_score(graph.class_labels, labels))
            if graph.class_labels is not None
            else None
        ),
    })
    return TrainedModels(encoder, denoiser, context, train_adjacency, history, audit_record)
