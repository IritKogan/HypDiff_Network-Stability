"""Evaluation metrics (no scikit-learn dependency)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def roc_auc(positive_scores, negative_scores) -> float:
    """Area under the ROC curve = P(score_pos > score_neg) + 0.5 P(tie).

    Mann-Whitney form with average ranks for ties (same value as
    sklearn.metrics.roc_auc_score; tested).
    """
    pos = np.asarray(positive_scores, dtype=np.float64).ravel()
    neg = np.asarray(negative_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError("roc_auc needs at least one positive and one negative score")
    if not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        raise ValueError("roc_auc got non-finite scores")
    ranks = rankdata(np.concatenate([pos, neg]))
    rank_sum = ranks[: pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
