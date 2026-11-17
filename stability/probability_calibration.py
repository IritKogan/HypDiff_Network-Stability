"""Leakage-free calibration of decoded edge probabilities (Decision 10, 2026-11-17).

The Fermi-Dirac decoder is trained on balanced edges / non-edges, so its probabilities are
far too high for real graphs (Cora: ~700 non-edges per missing edge). Reconstructions now
draw every pair from a Bernoulli with the CALIBRATED probability, so calibration matters.

* Platt scaling on the logit of the decoder probability, fitted on the VALIDATION pairs only
  (validation edges vs validation negatives; pool edges and control pairs never used), with
  sample weights that restore the real base rate: validation edges stand for all edges
  missing from the validation input graph, validation negatives for all non-edges.
* Prior shift: the base rate at evaluation differs (R removed edges among all candidate
  pairs); the intercept is moved by the difference in log-odds (Saerens et al., 2002).
  The NUMBER of removed edges is known by construction; which edges were removed is not used.
* ``cross_fitted_quality``: log-loss, Brier and ECE at the real base rate with the calibrator
  fitted on other folds of validation pairs (no pair scores itself): the model-selection
  criterion for probability quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from stability.sample_diversity import calibration_table

EPSILON = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def log_odds(prevalence: float) -> float:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    return float(np.log(prevalence / (1 - prevalence)))


@dataclass(frozen=True)
class PlattCalibrator:
    slope: float
    intercept: float
    fit_prevalence: float

    def apply(self, probabilities: np.ndarray, prevalence: float | None = None) -> np.ndarray:
        z = self.slope * logit(probabilities) + self.intercept
        if prevalence is not None:
            z = z + log_odds(prevalence) - log_odds(self.fit_prevalence)
        return sigmoid(z)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def base_rate_weights(missing_edges: int, non_edges: int, positives: int, negatives: int) -> tuple[float, float]:
    """Weight of one validation edge / one validation negative so that the weighted sample
    has the real ratio of missing edges to non-edges."""
    if min(missing_edges, non_edges, positives, negatives) <= 0:
        raise ValueError("all counts must be positive")
    return missing_edges / positives, non_edges / negatives


def fit_platt(positive: np.ndarray, negative: np.ndarray, positive_weight: float,
              negative_weight: float, max_iter: int = 200) -> PlattCalibrator:
    """Weighted logistic regression y ~ sigmoid(a * logit(p) + b) (Newton's method, exact)."""
    x = np.r_[logit(np.ravel(positive)), logit(np.ravel(negative))]
    y = np.r_[np.ones(np.size(positive)), np.zeros(np.size(negative))]
    w = np.r_[np.full(np.size(positive), positive_weight), np.full(np.size(negative), negative_weight)]
    w = w / w.mean()
    design = np.c_[x, np.ones_like(x)]
    beta = np.zeros(2)
    for _ in range(max_iter):
        p = sigmoid(design @ beta)
        gradient = design.T @ (w * (y - p))
        hessian = (design * (w * p * (1 - p))[:, None]).T @ design + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    prevalence = positive_weight * np.size(positive) / (
        positive_weight * np.size(positive) + negative_weight * np.size(negative))
    return PlattCalibrator(float(beta[0]), float(beta[1]), float(prevalence))


def _weighted_scores(probability: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    p = np.clip(probability, EPSILON, 1 - EPSILON)
    loss = -(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    return {
        "log_loss": float(np.sum(weights * loss) / weights.sum()),
        "brier": float(np.sum(weights * (p - labels) ** 2) / weights.sum()),
        "ece": calibration_table(p, labels, weights)["ece"],
        "mean_probability_edges": float(np.mean(p[labels == 1])),
        "mean_probability_non_edges": float(np.mean(p[labels == 0])),
    }


def cross_fitted_quality(positive: np.ndarray, negative: np.ndarray, positive_weight: float,
                         negative_weight: float, folds: int, seed: int) -> dict[str, Any]:
    """positive / negative: (samples, pairs) or (pairs,) probabilities. Pairs are assigned to
    folds (all samples of a pair stay together); each fold is scored by a calibrator fitted on
    the others. Also reports the prior log-loss (constant base-rate prediction) for reference."""
    positive = np.atleast_2d(np.asarray(positive, dtype=np.float64))
    negative = np.atleast_2d(np.asarray(negative, dtype=np.float64))
    if folds < 2:
        raise ValueError("folds must be >= 2")
    rng = np.random.default_rng(seed)
    fold_pos = rng.permutation(positive.shape[1]) % folds
    fold_neg = rng.permutation(negative.shape[1]) % folds
    probs, labels, weights = [], [], []
    for fold in range(folds):
        calibrator = fit_platt(positive[:, fold_pos != fold], negative[:, fold_neg != fold],
                               positive_weight, negative_weight)
        for array, mask, label, weight in ((positive, fold_pos == fold, 1.0, positive_weight),
                                           (negative, fold_neg == fold, 0.0, negative_weight)):
            values = calibrator.apply(array[:, mask]).ravel()
            probs.append(values)
            labels.append(np.full(values.size, label))
            weights.append(np.full(values.size, weight))
    probs, labels, weights = np.concatenate(probs), np.concatenate(labels), np.concatenate(weights)
    scores = _weighted_scores(probs, labels, weights)
    prevalence = float(np.sum(weights * labels) / weights.sum())
    scores["log_loss_base_rate_only"] = _weighted_scores(np.full_like(probs, prevalence), labels, weights)["log_loss"]
    scores["prevalence"] = prevalence
    return scores
