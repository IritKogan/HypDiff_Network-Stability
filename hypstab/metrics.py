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


def adjusted_rand_index(labels_a, labels_b) -> float:
    """Adjusted Rand index of two partitions (Hubert & Arabie 1985; same value as sklearn's
    adjusted_rand_score). 1 = identical partitions, ~0 = agreement expected by chance."""
    a = np.unique(np.asarray(labels_a).ravel(), return_inverse=True)[1]
    b = np.unique(np.asarray(labels_b).ravel(), return_inverse=True)[1]
    if a.size != b.size:
        raise ValueError("partitions must have the same length")
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)

    def pairs(x):
        x = np.asarray(x, dtype=np.float64)
        return float((x * (x - 1) / 2).sum())

    index = pairs(table)
    rows, cols, total = pairs(table.sum(1)), pairs(table.sum(0)), a.size * (a.size - 1) / 2
    expected = rows * cols / total
    maximum = (rows + cols) / 2
    if maximum == expected:
        return 1.0
    return float((index - expected) / (maximum - expected))
