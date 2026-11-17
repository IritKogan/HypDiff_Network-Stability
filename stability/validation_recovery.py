"""Leakage-free validation of the recovery step and threshold calibration.

The validation graph contains the TRAIN edges only: validation edges are the
"removed" edges, validation negatives are the non-edges. Pool edges and control
pairs are not used at all, so model selection and tau calibration never see the
evaluation set.

For each method (``diffusion`` = partial-noise recovery, ``no_diffusion`` =
decode the damaged embeddings directly, ``clamp_only`` = the clamp step of the
recovery without network or noise) we report the threshold-free AUC of
validation edges vs validation negatives and, for tau_mode = "val_fpr", the
threshold at which the false-positive rate on validation negatives equals
``recovery.target_fpr``. Each method gets its own tau, so methods are compared
at the same false-positive rate.

Decision 10: every method also gets a Platt calibrator fitted on the same validation
pairs, weighted to the real base rate (validation edges stand for all edges missing from
the validation input graph, validation negatives for all non-edges), used by
edge_decision = "bernoulli"; plus cross-fitted log-loss / Brier / ECE at that base rate
(probability-quality criterion for model selection).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from stability.model_training import normalized_adjacency
from stability.probability_calibration import base_rate_weights, cross_fitted_quality, fit_platt

VALIDATION_SERIES_INDEX = 1_000_000  # recovery seeds disjoint from evaluation series


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return float(roc_auc_score(labels, np.r_[positive, negative]))


def calibrate_tau(negative_probabilities: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold with mean(negative >= tau) <= target_fpr (exact with ties)."""
    values = np.sort(np.asarray(negative_probabilities, dtype=np.float64))[::-1]
    allowed = int(np.floor(target_fpr * len(values)))
    if allowed >= len(values):
        return float(values[-1])
    return float(np.nextafter(values[allowed], np.inf))


def _method_summary(positive: np.ndarray, negative: np.ndarray, config) -> dict[str, Any]:
    """positive / negative: (samples, pairs) probabilities."""
    # float64: numpy 2 compares float32 arrays with a Python float in float32,
    # which would round the calibrated tau back onto a tied probability.
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    rec = config.recovery
    if rec.tau_mode == "val_fpr":
        tau = calibrate_tau(negative.reshape(-1), rec.target_fpr)
    else:
        tau = rec.tau
    return {
        "tau": tau,
        "auc_mean_probability": _auc(positive.mean(axis=0), negative.mean(axis=0)),
        "hit_rate": float((positive >= tau).mean()),
        "false_positive_rate": float((negative >= tau).mean()),
        "hit_rate_at_tau_0_5": float((positive >= 0.5).mean()),
        "false_positive_rate_at_tau_0_5": float((negative >= 0.5).mean()),
    }


def validation_base_rate(graph, split) -> dict[str, int]:
    """Missing edges / non-edges of the validation input graph (train edges only)."""
    n = graph.node_count
    return {"missing_edges": int(len(split.val) + len(split.pool)),
            "non_edges": int(n * (n - 1) // 2 - len(split.all_edges))}


def _calibration(positive, negative, counts, config) -> dict[str, Any]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    w_pos, w_neg = base_rate_weights(counts["missing_edges"], counts["non_edges"],
                                     positive.shape[-1], negative.shape[-1])
    calibrator = fit_platt(positive, negative, w_pos, w_neg)
    return {
        "calibrator": calibrator.to_dict(),
        "cross_fitted": cross_fitted_quality(positive, negative, w_pos, w_neg,
                                             config.recovery.calibration_folds, config.recovery.seed),
    }


def validation_recovery(graph, split, models, config, device) -> dict[str, Any]:
    from stability.edge_recovery_experiment import recovery_seed

    rec = config.recovery
    adjacency = normalized_adjacency(graph.node_count, split.train, device)
    features = graph.features.to(device)
    positive_pairs = torch.as_tensor(split.val, device=device)
    negative_pairs = torch.as_tensor(split.val_negatives, device=device)
    labels = models.context.labels.to(device)
    centers = models.context.centers.to(device)
    with torch.no_grad():
        embeddings = models.encoder.encode(features, adjacency)[2]
        base_pos = models.encoder.decode(embeddings, positive_pairs).cpu().numpy()[None]
        base_neg = models.encoder.decode(embeddings, negative_pairs).cpu().numpy()[None]
        shrunk = models.denoiser.shrink_to_prototype(embeddings, labels, centers)
        clamp_pos = models.encoder.decode(shrunk, positive_pairs).cpu().numpy()[None]
        clamp_neg = models.encoder.decode(shrunk, negative_pairs).cpu().numpy()[None]
        diff_pos, diff_neg = [], []
        for k in range(rec.reconstructions):
            generator = torch.Generator(device=device)
            generator.manual_seed(recovery_seed(rec.seed, VALIDATION_SERIES_INDEX, k))
            recovered = models.denoiser.sample_from_tstar(
                damaged_embeddings=embeddings, damaged_adjacency=adjacency, labels=labels,
                prototypes=centers, t_star=rec.t_star, generator=generator,
            )
            diff_pos.append(models.encoder.decode(recovered, positive_pairs).cpu().numpy())
            diff_neg.append(models.encoder.decode(recovered, negative_pairs).cpu().numpy())
    counts = validation_base_rate(graph, split)
    methods = {"no_diffusion": (base_pos, base_neg), "clamp_only": (clamp_pos, clamp_neg),
               "diffusion": (np.stack(diff_pos), np.stack(diff_neg))}
    result = {
        "protocol": "train edges only as input; validation edges vs validation negatives",
        "tau_mode": rec.tau_mode,
        "target_fpr": rec.target_fpr if rec.tau_mode == "val_fpr" else None,
        "base_rate": counts,
    }
    for name, (positive, negative) in methods.items():
        result[name] = {**_method_summary(positive, negative, config),
                        "calibration": _calibration(positive, negative, counts, config)}
    return result
