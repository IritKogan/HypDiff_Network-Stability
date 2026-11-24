"""Validation-recovery protocol (Decision 7), used for ALL model selection.

Input = the train graph only; score validation edges vs validation negatives.
Pool edges and control pairs are never touched here. Reported per method as
threshold-free ROC AUC of the edge logits:
- no_diffusion : decode the encoder's embeddings directly;
- clamp_only   : the clamp step alone (hypstab/models/clamp.py);
- diffusion    : K reconstructions from t* (recovery.reconstructions); AUC of the mean
                 probability over the K (as in experiment E1) and the mean per-reconstruction AUC.
Recovery noise seeds: recovery_seed(recovery.seed, VALIDATION_SERIES, k), disjoint from the
evaluation series (same convention as the old code).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hypstab.config import HypstabConfig
from hypstab.diffusion.model import HyperbolicDiffusion
from hypstab.data.split import EdgeSplit
from hypstab.metrics import roc_auc
from hypstab.models.clamp import clamp_to_centers
from hypstab.training.autoencoder import HyperbolicAutoencoder
from hypstab.training.communities import CommunityContext


VALIDATION_SERIES = 1_000_000


def recovery_seed(base_seed: int, series: int, reconstruction: int) -> int:
    """Noise seed of reconstruction k in series s (common random numbers across levels / attacks)."""
    return int(np.random.SeedSequence([base_seed, series, reconstruction]).generate_state(1)[0])


@torch.no_grad()
def validation_report(
    model: HyperbolicAutoencoder,
    features: torch.Tensor,
    train_adjacency: torch.Tensor,
    split: EdgeSplit,
    context: CommunityContext,
    config: HypstabConfig,
    diffusion: HyperbolicDiffusion | None = None,
) -> dict[str, Any]:
    model.eval()
    device = features.device
    pos = torch.as_tensor(np.asarray(split.val), device=device)
    neg = torch.as_tensor(np.asarray(split.val_negatives), device=device)
    embeddings = model.encode(features, train_adjacency)
    clamped = clamp_to_centers(embeddings, context.labels, context.centers,
                               config.geometry.curvature, config.clamp.clip)
    methods = {"no_diffusion": embeddings, "clamp_only": clamped}
    report: dict[str, Any] = {"protocol": "train graph as input; validation edges vs validation negatives"}
    for name, points in methods.items():
        report[name] = {"auc": roc_auc(model.edge_logits(points, pos).cpu().numpy(),
                                       model.edge_logits(points, neg).cpu().numpy())}
    if diffusion is not None:
        diffusion.eval()
        rec = config.recovery
        prob_pos, prob_neg = [], []
        for k in range(rec.reconstructions):
            generator = torch.Generator(device=device)
            generator.manual_seed(recovery_seed(rec.seed, VALIDATION_SERIES, k))
            recovered = diffusion.recover(embeddings, train_adjacency, context.labels, context.centers,
                                          rec.t_star, generator)
            prob_pos.append(torch.sigmoid(model.edge_logits(recovered, pos)).cpu().numpy())
            prob_neg.append(torch.sigmoid(model.edge_logits(recovered, neg)).cpu().numpy())
        prob_pos, prob_neg = np.stack(prob_pos), np.stack(prob_neg)
        report["diffusion"] = {
            "auc": roc_auc(prob_pos.mean(0), prob_neg.mean(0)),
            "auc_per_reconstruction_mean": float(np.mean([roc_auc(p, n) for p, n in zip(prob_pos, prob_neg)])),
            "probability_std_across_K_mean": float(np.concatenate([prob_pos, prob_neg], 1).std(0).mean()),
            "K": rec.reconstructions,
            "t_star": rec.t_star,
        }
    return report
