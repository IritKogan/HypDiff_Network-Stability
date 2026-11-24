"""Do the K reconstructions carry information beyond their mean? (experiments E1 / E2)

Same definitions as stability/sample_diversity.schedule_variant_metrics (E1), so E1 and
E2 numbers are comparable; written without scikit-learn (tested against it).

Criteria (claude/plan-from-scratch-2026-11-24.md), on the validation protocol:
  (a) quality         : AUC of the mean probability over K  >=  clamp_only AUC;
  (b) informativeness : AUC of p_e INSIDE deciles of the model's own mean probability
                        (0.5 = p_e says nothing beyond the mean), clearly > 0.5;
  (c) cross-validated log-likelihood gain of p_e over the mean, > 0.
Diagnostic p_e here = share of the K reconstructions with probability >= tau, tau at
``target_fpr`` on the pooled validation negatives (as in E1; NOT the main pairwise rule).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from hypstab.metrics import roc_auc


def calibrate_tau(negative_probabilities: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold with mean(negative >= tau) <= target_fpr (exact with ties)."""
    values = np.sort(np.asarray(negative_probabilities, dtype=np.float64))[::-1]
    allowed = int(np.floor(target_fpr * len(values)))
    if allowed >= len(values):
        return float(values[-1])
    return float(np.nextafter(values[allowed], np.inf))


def frequency_profile(frequency: np.ndarray) -> dict[str, float]:
    f = np.asarray(frequency, dtype=np.float64)
    zero, one = float(np.mean(f == 0.0)), float(np.mean(f == 1.0))
    return {"share_zero": zero, "share_one": one, "share_intermediate": 1.0 - zero - one,
            "mean": float(f.mean())}


def stratified_auc(score, labels, strata_score, n_strata: int = 10) -> float | None:
    """AUC of ``score`` inside quantile strata of ``strata_score`` (pair-weighted mean)."""
    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    strata_score = np.asarray(strata_score, dtype=np.float64)
    edges = np.unique(np.quantile(strata_score, np.linspace(0, 1, n_strata + 1)))
    strata = np.clip(np.searchsorted(edges, strata_score, side="right") - 1, 0, len(edges) - 2)
    total, weighted = 0, 0.0
    for stratum in np.unique(strata):
        mask = strata == stratum
        y, s = labels[mask], score[mask]
        if y.min() == y.max() or s.min() == s.max():
            continue
        weighted += roc_auc(s[y == 1], s[y == 0]) * mask.sum()
        total += mask.sum()
    return weighted / total if total else None


def fit_logistic(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """L2-regularised logistic regression (= sklearn defaults: C = 1, unpenalised intercept).
    Returns [intercept, w_1, ..., w_d]."""
    x1 = np.c_[np.ones(len(x)), x]

    def objective(theta):
        z = x1 @ theta
        loss = c * np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(theta[1:] ** 2)
        grad = c * x1.T @ (expit(z) - y)
        grad[1:] += theta[1:]
        return loss, grad

    result = minimize(objective, np.zeros(x1.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "gtol": 1e-10})
    return result.x


def stratified_folds(labels: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    """Test-index sets of a class-stratified, shuffled K-fold split."""
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(labels), dtype=int)
    for cls in np.unique(labels):
        index = rng.permutation(np.flatnonzero(labels == cls))
        assignment[index] = np.arange(len(index)) % folds
    return [np.flatnonzero(assignment == f) for f in range(folds)]


def incremental_log_likelihood(base_score, extra, labels, folds: int = 5, seed: int = 0,
                               test_sets: list[np.ndarray] | None = None) -> dict[str, float]:
    """Cross-validated mean log-likelihood of label ~ logit(base) vs label ~ logit(base) + extra."""
    eps = 1e-6
    base = np.clip(np.asarray(base_score, dtype=np.float64), eps, 1 - eps)
    x_base = np.log(base / (1 - base))[:, None]
    x_full = np.c_[x_base, np.asarray(extra, dtype=np.float64)]
    labels = np.asarray(labels).astype(int)
    test_sets = test_sets or stratified_folds(labels, folds, seed)
    totals = {"base": 0.0, "with_extra": 0.0}
    for test in test_sets:
        train = np.setdiff1d(np.arange(len(labels)), test)
        for name, x in (("base", x_base), ("with_extra", x_full)):
            theta = fit_logistic(x[train], labels[train])
            p = np.clip(expit(np.c_[np.ones(len(test)), x[test]] @ theta), eps, 1 - eps)
            y = labels[test]
            totals[name] += float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    n = len(labels)
    out = {k: v / n for k, v in totals.items()}
    out["gain_per_pair"] = out["with_extra"] - out["base"]
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def variant_metrics(prob_pos, prob_neg, clamp_pos, clamp_neg, target_fpr: float, quality_tolerance: float,
                    informative_auc: float, min_ll_gain: float, strata: int = 10,
                    folds: int = 5) -> dict[str, Any]:
    """prob_pos / prob_neg: (K, pairs) probabilities of validation edges / negatives over K
    reconstructions; clamp_pos / clamp_neg: (pairs,) clamp_only probabilities."""
    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    prob_neg = np.asarray(prob_neg, dtype=np.float64)
    if prob_pos.ndim != 2 or prob_neg.ndim != 2 or prob_pos.shape[0] != prob_neg.shape[0]:
        raise ValueError("prob_pos / prob_neg must have shape (K, pairs) with the same K")
    K = prob_pos.shape[0]
    probs = np.concatenate([prob_pos, prob_neg], axis=1)
    y = np.r_[np.ones(prob_pos.shape[1]), np.zeros(prob_neg.shape[1])].astype(int)
    tau = calibrate_tau(prob_neg.reshape(-1), target_fpr)
    p_e = (probs >= tau).mean(axis=0)
    mean_p = probs.mean(axis=0)
    spread = _logit(probs).std(axis=0) if K > 1 else np.zeros(probs.shape[1])
    sorted_neg = np.sort(prob_neg, axis=1)
    expected_share = np.stack([np.searchsorted(sorted_neg[k], prob_pos[k], side="left") / prob_neg.shape[1]
                               for k in range(K)])
    mean_auc = roc_auc(mean_p[y == 1], mean_p[y == 0])
    clamp_auc = roc_auc(clamp_pos, clamp_neg)

    def gain(base, extra):
        if K < 2 or np.ptp(extra) == 0:
            return 0.0
        return incremental_log_likelihood(base, extra, y, folds=folds)["gain_per_pair"]

    p_e_strat = stratified_auc(p_e, y, mean_p, strata)
    p_e_gain = gain(mean_p, p_e)
    return {
        "K": K,
        "tau": tau,
        "auc_mean_probability": mean_auc,
        "auc_per_reconstruction_mean": float(np.mean([roc_auc(p, n) for p, n in zip(prob_pos, prob_neg)])),
        "auc_clamp_only": clamp_auc,
        "auc_gain_over_clamp_only": mean_auc - clamp_auc,
        "probability_std_positive": float(prob_pos.std(axis=0).mean()),
        "expected_share_std_across_K": float(expected_share.std(axis=0).mean()),
        "p_e": {**frequency_profile(p_e), "stratified_auc_given_own_mean": p_e_strat, "ll_gain_over_own_mean": p_e_gain},
        "spread": {
            "stratified_auc_given_own_mean": stratified_auc(spread, y, mean_p, strata) if K > 1 else None,
            "ll_gain_over_own_mean": gain(mean_p, spread),
        },
        # criteria as in E1 (configs/e1_paper_schedule.toml: tolerance 0, informative_auc 0.55, min_ll_gain 0.001)
        "quality_pass": bool(mean_auc >= clamp_auc - quality_tolerance),
        "informative": bool(p_e_strat is not None and p_e_strat >= informative_auc and p_e_gain >= min_ll_gain),
    }
