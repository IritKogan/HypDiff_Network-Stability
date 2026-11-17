"""Diversity of the K stochastic reconstructions (diagnostic, Stage 1 of the
conditional-diffusion proposal, 2026-11-17).

The project's score is the per-edge recovery frequency p_e over K reconstructions.
p_e can only be informative if the K reconstructions actually differ. These helpers
quantify that, in two places:

* in the denoiser's standardised tangent space: how much spread the forward noise
  puts into x_{t*} (input) and how much of it survives in the predicted x0 (output);
  output / input ~ 0 means the denoiser removes the noise and the samples collapse;
* after decoding: per-pair spread of edge probabilities and of the binarised
  decisions (p_e exactly 0, exactly 1, or in between).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def tangent_spread(samples: np.ndarray) -> float:
    """samples: (K, nodes, dim). Mean over nodes/dims of the std across the K samples."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[0] < 2:
        raise ValueError("samples must have shape (K >= 2, nodes, dim)")
    return float(samples.std(axis=0, ddof=0).mean())


def recovery_frequency(probabilities: np.ndarray, tau: float) -> np.ndarray:
    """probabilities: (K, pairs) -> per-pair fraction of the K samples with prob >= tau."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (K, pairs)")
    return (probabilities >= tau).mean(axis=0)


def frequency_profile(frequency: np.ndarray) -> dict[str, float]:
    """Share of pairs with frequency exactly 0, exactly 1, or strictly in between."""
    frequency = np.asarray(frequency, dtype=np.float64)
    if frequency.size == 0:
        raise ValueError("empty frequency array")
    zero = float(np.mean(frequency == 0.0))
    one = float(np.mean(frequency == 1.0))
    return {
        "share_zero": zero,
        "share_one": one,
        "share_intermediate": 1.0 - zero - one,
        "mean": float(frequency.mean()),
        "mean_binomial_variance": float(np.mean(frequency * (1.0 - frequency))),
    }


def stratified_auc(score: np.ndarray, labels: np.ndarray, strata_score: np.ndarray,
                   n_strata: int = 10) -> float | None:
    """AUC of ``score`` for ``labels`` computed inside quantile strata of ``strata_score``
    (pair-count weighted mean over strata that contain both classes).

    ~0.5 means ``score`` carries no information about the label beyond ``strata_score``.
    """
    from sklearn.metrics import roc_auc_score

    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    strata_score = np.asarray(strata_score, dtype=np.float64)
    if not (score.shape == labels.shape == strata_score.shape):
        raise ValueError("score, labels and strata_score must have the same shape")
    edges = np.unique(np.quantile(strata_score, np.linspace(0, 1, n_strata + 1)))
    strata = np.clip(np.searchsorted(edges, strata_score, side="right") - 1, 0, len(edges) - 2)
    total, weighted = 0, 0.0
    for stratum in np.unique(strata):
        mask = strata == stratum
        y, s = labels[mask], score[mask]
        if y.min() == y.max() or s.min() == s.max():
            continue
        weighted += roc_auc_score(y, s) * mask.sum()
        total += mask.sum()
    return weighted / total if total else None


def incremental_log_likelihood(base_score: np.ndarray, extra: np.ndarray, labels: np.ndarray,
                               folds: int = 5, seed: int = 0) -> dict[str, float]:
    """Cross-validated mean log-likelihood of logistic models label ~ logit(base) and
    label ~ logit(base) + extra. A positive gain means ``extra`` adds information."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    eps = 1e-6
    base = np.clip(np.asarray(base_score, dtype=np.float64), eps, 1 - eps)
    x_base = np.log(base / (1 - base))[:, None]
    x_full = np.c_[x_base, np.asarray(extra, dtype=np.float64)]
    labels = np.asarray(labels).astype(int)
    results = {"base": 0.0, "with_extra": 0.0}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(x_base, labels):
        for name, x in (("base", x_base), ("with_extra", x_full)):
            model = LogisticRegression().fit(x[train], labels[train])
            p = np.clip(model.predict_proba(x[test])[:, 1], eps, 1 - eps)
            y = labels[test]
            results[name] += float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    n = len(labels)
    results = {k: v / n for k, v in results.items()}
    results["gain_per_pair"] = results["with_extra"] - results["base"]
    return results


def probability_diversity(positive: np.ndarray, negative: np.ndarray, tau: float) -> dict[str, Any]:
    """positive / negative: (K, pairs) decoded probabilities of edges / non-edges."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    return {
        "probability_std_positive": float(positive.std(axis=0).mean()),
        "probability_std_negative": float(negative.std(axis=0).mean()),
        "positive": frequency_profile(recovery_frequency(positive, tau)),
        "negative": frequency_profile(recovery_frequency(negative, tau)),
    }
