"""Validation-recovery protocol (Decision 7), used for ALL model selection.

Input = the train graph only; score validation edges vs validation negatives.
Pool edges and control pairs are never touched here. Reported per method as
threshold-free ROC AUC of the edge logits:
- no_diffusion : decode the encoder's embeddings directly;
- clamp_only   : the clamp step alone (hypstab/models/clamp.py);
- diffusion    : added in Stage 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hypstab.config import HypstabConfig
from hypstab.data.split import EdgeSplit
from hypstab.metrics import roc_auc
from hypstab.models.clamp import clamp_to_centers
from hypstab.training.autoencoder import HyperbolicAutoencoder
from hypstab.training.communities import CommunityContext


@torch.no_grad()
def validation_report(
    model: HyperbolicAutoencoder,
    features: torch.Tensor,
    train_adjacency: torch.Tensor,
    split: EdgeSplit,
    context: CommunityContext,
    config: HypstabConfig,
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
    return report
