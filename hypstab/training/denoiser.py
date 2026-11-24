"""Edge-reconstruction training of the diffusion denoiser (Decision 8), TRAIN edges only.

Per epoch (training = "edge_reconstruction"):
1. drop a fraction f ~ U(0, train_max_drop) of the train edges;
2. encode the remaining train graph with the FROZEN encoder;
3. one noise + denoise step at random per-node steps t in [0, train_t_max);
4. loss = BCE(dropped train edges -> 1) + BCE(kept train edges -> 1)
          + BCE(sampled non-edges -> 0), through the FROZEN Fermi-Dirac decoder.
training = "unrolled" (experiment E2): step 3 is replaced by noising to t* and running the
iterative reverse chain (recovery.steps strided steps, differentiable), i.e. exactly the
recovery procedure; the loss is on its final output.
Non-edges never include validation negatives or control pairs; validation and
pool edges are never used (Decision 6). Communities come from the train graph.

An epoch with a non-finite loss or gradients makes no update and is recorded;
more than ``nonfinite_max_skips`` such epochs stop training (Decision 4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from hypstab.config import HypstabConfig
from hypstab.data.graph import GraphData
from hypstab.data.split import EdgeSplit, pair_keys, sample_non_edges
from hypstab.diffusion.model import HyperbolicDiffusion, sampling_steps
from hypstab.diffusion.schedule import DiffusionSchedule
from hypstab.models.hgcn import normalized_adjacency
from hypstab.training.autoencoder import HyperbolicAutoencoder
from hypstab.training.communities import CommunityContext

DROP_STREAM, NEGATIVE_STREAM, NOISE_STREAM = 2, 3, 4


def build_diffusion(config: HypstabConfig) -> HyperbolicDiffusion:
    d, den = config.diffusion, config.denoiser
    schedule = DiffusionSchedule(d.timesteps, d.radial_delta, config.geometry.curvature)
    return HyperbolicDiffusion(
        dim=config.encoder.dim, hidden_dim=den.hidden_dim, layers=den.layers, schedule=schedule,
        curvature=config.geometry.curvature, angular_noise=d.angular_noise, clip_x0=d.clip_x0,
    )


def edge_reconstruction_loss(model: HyperbolicAutoencoder, embeddings: Tensor,
                             dropped: Tensor, kept: Tensor, negatives: Tensor) -> Tensor:
    """Sum of the mean BCE of three groups (empty groups skipped)."""
    terms = []
    for pairs, target in ((dropped, 1.0), (kept, 1.0), (negatives, 0.0)):
        if pairs.shape[0] == 0:
            continue
        logits = model.edge_logits(embeddings, pairs)
        terms.append(F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target)))
    return torch.stack(terms).sum()


def train_denoiser(
    graph: GraphData,
    split: EdgeSplit,
    config: HypstabConfig,
    device: torch.device,
    autoencoder: HyperbolicAutoencoder,
    context: CommunityContext,
    audit: bool = False,
    log_every: int = 0,
) -> tuple[HyperbolicDiffusion, dict[str, Any]]:
    den = config.denoiser
    torch.manual_seed(den.seed)
    diffusion = build_diffusion(config).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=den.learning_rate)
    drop_rng = np.random.default_rng([den.seed, DROP_STREAM])
    negative_rng = np.random.default_rng([den.seed, NEGATIVE_STREAM])
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(int(np.random.SeedSequence([den.seed, NOISE_STREAM]).generate_state(1)[0]))
    t_generator = torch.Generator(device=device)
    t_generator.manual_seed(int(np.random.SeedSequence([den.seed, NOISE_STREAM, 1]).generate_state(1)[0]))
    forbidden = split.forbidden_negative_keys()
    features = graph.features.to(device)
    labels, centers = context.labels.to(device), context.centers.to(device)
    chain = sampling_steps(config.recovery.t_star, config.recovery.steps) if den.training == "unrolled" else None

    autoencoder.eval()
    frozen = [p.requires_grad for p in autoencoder.parameters()]
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    losses: list[float] = []
    drops: list[float] = []
    skipped: list[dict[str, Any]] = []
    seen: dict[str, list[np.ndarray]] = {"adjacency": [], "positive": [], "negative": []}
    try:
        for epoch in range(den.epochs):
            diffusion.train()
            optimizer.zero_grad()
            fraction = float(drop_rng.uniform(0.0, den.train_max_drop))
            keep = drop_rng.random(len(split.train)) >= fraction
            kept_edges, dropped_edges = split.train[keep], split.train[~keep]
            adjacency = normalized_adjacency(graph.node_count, kept_edges, device)
            with torch.no_grad():
                damaged = autoencoder.encode(features, adjacency)
            if den.training == "unrolled":
                recovered = diffusion.denoise_chain(damaged, adjacency, labels, centers, chain, noise_generator)
            else:
                t = torch.randint(0, den.train_t_max, (graph.node_count,), device=device, generator=t_generator)
                recovered = diffusion.denoise_once(damaged, adjacency, labels, centers, t, noise_generator)
            negatives = sample_non_edges(len(split.train), graph.node_count, forbidden, negative_rng)
            if audit:
                seen["adjacency"].append(pair_keys(kept_edges, graph.node_count))
                seen["positive"].append(pair_keys(split.train, graph.node_count))
                seen["negative"].append(pair_keys(negatives, graph.node_count))
            loss = edge_reconstruction_loss(
                autoencoder, recovered,
                dropped=torch.as_tensor(dropped_edges, device=device),
                kept=torch.as_tensor(kept_edges, device=device),
                negatives=torch.as_tensor(negatives, device=device),
            )
            drops.append(fraction)
            reason = None
            if not torch.isfinite(loss):
                reason = "loss"
            else:
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in diffusion.parameters() if p.grad is not None):
                    reason = "gradients"
            if reason is not None:
                skipped.append({"epoch": epoch, "reason": reason})
                optimizer.zero_grad()
                losses.append(float("nan"))
                if len(skipped) > den.nonfinite_max_skips:
                    raise RuntimeError(f"non-finite denoiser {reason} at epoch {epoch}: {len(skipped)} skipped "
                                       f"epochs > nonfinite_max_skips = {den.nonfinite_max_skips}")
                continue
            optimizer.step()
            losses.append(float(loss.detach()))
            if log_every and epoch % log_every == 0:
                print(f"[denoiser] epoch {epoch} loss {losses[-1]:.4f} drop {fraction:.2f}", flush=True)
    finally:
        for p, flag in zip(autoencoder.parameters(), frozen):
            p.requires_grad_(flag)
    diffusion.eval()
    if not all(torch.isfinite(p).all() for p in diffusion.parameters()):
        raise RuntimeError("denoiser parameters are non-finite after training")
    history: dict[str, Any] = {"loss": losses, "drop_fractions": drops, "skipped_epochs": skipped}
    if audit:
        history["audit"] = {k: np.unique(np.concatenate(v)) for k, v in seen.items()}
    return diffusion, history
