from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


Pair = frozenset[Hashable]


def common_neighbor_scores(
    damaged_edges: Sequence[Sequence[Hashable]],
    candidate_pairs: Sequence[Sequence[Hashable]],
) -> list[int]:
    """Score candidate pairs using neighbors from the damaged edges only."""
    neighbors: dict[Hashable, set[Hashable]] = {}
    for source, target in damaged_edges:
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)
    return [
        len(neighbors.get(source, set()) & neighbors.get(target, set()))
        for source, target in candidate_pairs
    ]


def ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or targets.shape != values.shape or values.size == 0:
        raise ValueError("scores and labels must be nonempty vectors of equal length")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not np.isin(targets, (0, 1)).all() or targets.sum() == 0:
        raise ValueError("labels must be binary and include at least one positive")

    ordered = np.sort(values)[::-1]
    cutoff = float(ordered[min(2, len(ordered) - 1)])
    strictly_above = values > cutoff
    at_cutoff = values == cutoff
    guaranteed = int(targets[strictly_above].sum())
    slots_at_cutoff = max(0, min(3, len(values)) - int(strictly_above.sum()))
    possible = guaranteed + min(
        slots_at_cutoff, int(targets[at_cutoff].sum())
    )
    ranks = {
        int(index): int(1 + np.sum(values > values[index]))
        for index in range(len(values))
    }
    return {
        "average_precision": float(average_precision_score(targets, values)),
        "removed_edges_guaranteed_in_top_three": guaranteed,
        "removed_edges_possible_in_top_three": possible,
        "top_three_cutoff_score": cutoff,
        "top_three_cutoff_tie_count": int(at_cutoff.sum()),
        "competition_ranks": ranks,
    }


def candidate_additions(
    original_edges: Sequence[Sequence[Hashable]],
    removed_edges: Sequence[Sequence[Hashable]],
    candidate_pairs: Sequence[Sequence[Hashable]],
) -> tuple[list[tuple[Hashable, Hashable]], list[int]]:
    original = {frozenset(edge) for edge in original_edges}
    removed = {frozenset(edge) for edge in removed_edges}
    additions = [
        tuple(pair)
        for pair in candidate_pairs
        if frozenset(pair) in removed or frozenset(pair) not in original
    ]
    labels = [int(frozenset(pair) in removed) for pair in additions]
    return additions, labels
