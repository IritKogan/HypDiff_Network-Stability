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


def calibration_table(score: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None,
                      bins: int = 10) -> dict[str, Any]:
    """Reliability of ``score`` as a probability that the pair is an edge.

    ``weights`` re-weight pairs to the real base rate (validation sets are balanced while
    real candidate pairs are mostly non-edges). Bins are equal-width on [0, 1]; scores of
    exactly 1 fall in the last bin. ECE = weighted mean |observed - mean score|.
    """
    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.ones_like(score) if weights is None else np.asarray(weights, dtype=np.float64)
    if not (score.shape == labels.shape == weights.shape):
        raise ValueError("score, labels and weights must have the same shape")
    index = np.minimum((score * bins).astype(int), bins - 1)
    rows, ece, total = [], 0.0, weights.sum()
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        w = weights[mask]
        observed = float(np.sum(w * labels[mask]) / w.sum())
        predicted = float(np.sum(w * score[mask]) / w.sum())
        ece += w.sum() / total * abs(observed - predicted)
        rows.append({"bin": b, "pairs": int(mask.sum()), "weight": float(w.sum()),
                     "mean_score": predicted, "observed": observed})
    return {"ece": float(ece), "bins": rows}


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


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return float(roc_auc_score(y, np.r_[positive, negative]))


def schedule_variant_metrics(prob_pos: np.ndarray, prob_neg: np.ndarray, clamp_pos: np.ndarray,
                             clamp_neg: np.ndarray, target_fpr: float, strata: int = 10,
                             folds: int = 5) -> dict[str, Any]:
    """Validation metrics of one diffusion variant (experiment E1, 2026-11-24).

    prob_pos / prob_neg: (K, pairs) decoded probabilities of validation edges / validation
    negatives in K reconstructions; clamp_pos / clamp_neg: (pairs,) deterministic clamp_only
    probabilities. Criteria of claude/plan-from-scratch-2026-11-24.md:
      (a) quality         : auc_mean_probability vs auc_clamp_only;
      (b) informativeness : stratified AUC of p_e (and of the per-pair spread across K) inside
                            deciles of the model's own mean probability (~0.5 = no information
                            beyond the mean);
      (c) log-likelihood gain of p_e (and of the spread) over the mean.
    p_e here = share of the K reconstructions with prob >= tau (tau at target_fpr on the pooled
    validation negatives), as in the Stage 1 diagnostic, so the numbers are comparable to it.
    The pairwise rule (Decision 10) is reported through the per-reconstruction expected share
    (share of validation negatives scored lower than the edge in that reconstruction).
    """
    from stability.validation_recovery import calibrate_tau

    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    prob_neg = np.asarray(prob_neg, dtype=np.float64)
    if prob_pos.ndim != 2 or prob_neg.ndim != 2 or prob_pos.shape[0] != prob_neg.shape[0]:
        raise ValueError("prob_pos / prob_neg must have shape (K, pairs) with the same K")
    K = prob_pos.shape[0]
    probs = np.concatenate([prob_pos, prob_neg], axis=1)
    y = np.r_[np.ones(prob_pos.shape[1]), np.zeros(prob_neg.shape[1])].astype(int)
    clamp = np.r_[clamp_pos, clamp_neg].astype(np.float64)
    tau = calibrate_tau(prob_neg.reshape(-1), target_fpr)
    p_e = recovery_frequency(probs, tau)
    mean_p = probs.mean(axis=0)
    spread = _logit(probs).std(axis=0) if K > 1 else np.zeros(probs.shape[1])
    sorted_neg = np.sort(prob_neg, axis=1)
    expected_share = np.stack([np.searchsorted(sorted_neg[k], prob_pos[k], side="left") / prob_neg.shape[1]
                               for k in range(K)])                       # (K, positives)
    clamp_auc = _auc(clamp_pos, clamp_neg)
    mean_auc = _auc(mean_p[y == 1], mean_p[y == 0])
    tau_clamp = calibrate_tau(np.asarray(clamp_neg, dtype=np.float64), target_fpr)

    def gain(base, extra):
        if K < 2 or np.ptp(extra) == 0:
            return 0.0
        return incremental_log_likelihood(base, extra, y, folds=folds)["gain_per_pair"]

    return {
        "K": K,
        "tau": tau,
        "auc_mean_probability": mean_auc,
        "auc_per_reconstruction_mean": float(np.mean([_auc(prob_pos[k], prob_neg[k]) for k in range(K)])),
        "auc_clamp_only": clamp_auc,
        "auc_gain_over_clamp_only": mean_auc - clamp_auc,
        "hit_rate": float((prob_pos >= tau).mean()),
        "hit_rate_clamp_only": float((np.asarray(clamp_pos) >= tau_clamp).mean()),
        "probability_std_positive": float(prob_pos.std(axis=0).mean()),
        "probability_std_negative": float(prob_neg.std(axis=0).mean()),
        "logit_std_mean": float(spread.mean()),
        "expected_share_mean": float(expected_share.mean()),
        "expected_share_std_across_K": float(expected_share.std(axis=0).mean()),
        "p_e": {
            **frequency_profile(p_e),
            "auc": _auc(p_e[y == 1], p_e[y == 0]),
            "stratified_auc_given_own_mean": stratified_auc(p_e, y, mean_p, strata),
            "stratified_auc_given_clamp_only": stratified_auc(p_e, y, clamp, strata),
            "ll_gain_over_own_mean": gain(mean_p, p_e),
        },
        "spread": {
            "stratified_auc_given_own_mean": stratified_auc(spread, y, mean_p, strata) if K > 1 else None,
            "ll_gain_over_own_mean": gain(mean_p, spread),
        },
    }
