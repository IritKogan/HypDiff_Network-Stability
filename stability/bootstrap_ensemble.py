"""Model-uncertainty (epistemic) reconstructions: an ensemble of models trained on resampled
train edges (idea A, 2026-11-18; docs/diffusion-diagnosis-and-literature.md section 5).

Reconstruction k of a damaged graph goes through ensemble member k instead of drawing new
diffusion noise from one model. Members differ only in which TRAIN edges they learn from
(and their seeds); validation, pool and control pairs are untouched, so the leakage rules of
Decision 6 hold: train edges a member does not learn from are still never sampled as its
training negatives.

Resampling modes:
  * "seed_only": all train edges, different model / community seeds (training randomness only);
  * "subsample": a fraction of the train edges without replacement;
  * "bootstrap": len(train) draws with replacement, duplicates collapsed (~63% unique edges).
Member 0 of "seed_only" uses the config seeds unchanged, i.e. it is the current single model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stability.edge_split import EdgeSplit, pair_keys

RESAMPLE_MODES = ("seed_only", "subsample", "bootstrap")
_MODE_ID = {name: i for i, name in enumerate(RESAMPLE_MODES)}


@dataclass(frozen=True)
class MemberSplit(EdgeSplit):
    """EdgeSplit whose ``train`` is a member's resample; ``held_out_train`` are the train edges
    the member does not learn from. They stay forbidden as training negatives."""

    held_out_train: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))

    def forbidden_negative_keys(self) -> np.ndarray:
        return np.unique(np.concatenate([
            super().forbidden_negative_keys(),
            pair_keys(self.held_out_train, self.node_count),
        ]))


def member_seed(base_seed: int, mode: str, member: int) -> int:
    """Seed of ensemble member ``member``; seed_only member 0 keeps ``base_seed``."""
    if mode not in _MODE_ID:
        raise ValueError(f"unknown resample mode {mode!r}; expected one of {RESAMPLE_MODES}")
    if mode == "seed_only" and member == 0:
        return int(base_seed)
    return int(np.random.SeedSequence([base_seed, 31, _MODE_ID[mode], member]).generate_state(1)[0] % 2**31)


def member_split(split: EdgeSplit, mode: str, member: int, fraction: float, seed: int) -> MemberSplit:
    """Resample the train edges of ``split`` for one ensemble member."""
    if mode not in _MODE_ID:
        raise ValueError(f"unknown resample mode {mode!r}; expected one of {RESAMPLE_MODES}")
    n = len(split.train)
    rng = np.random.default_rng([seed, 32, _MODE_ID[mode], member])
    if mode == "seed_only":
        keep = np.ones(n, dtype=bool)
    elif mode == "subsample":
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        keep = np.zeros(n, dtype=bool)
        keep[rng.choice(n, size=int(round(fraction * n)), replace=False)] = True
    else:
        keep = np.zeros(n, dtype=bool)
        keep[np.unique(rng.integers(0, n, size=n))] = True
    return MemberSplit(
        node_count=split.node_count, train=split.train[keep], val=split.val, pool=split.pool,
        val_negatives=split.val_negatives, control_pairs=split.control_pairs,
        held_out_train=split.train[~keep],
    )


def rank_against(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Share of ``reference`` scores strictly below each score (expected pairwise p_e)."""
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, np.asarray(scores, dtype=np.float64), side="left") / len(ordered)


def pairwise_outcomes(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(K, pairs) 0/1: pair scored strictly above one negative drawn uniformly per reconstruction
    (Decision 10). Negatives are compared with other negatives, never themselves."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    negative_index = np.flatnonzero(labels == 0)
    if len(negative_index) < 2:
        raise ValueError("need at least two negatives")
    k, p = scores.shape
    draw = rng.integers(0, len(negative_index), size=(k, p))
    reference_index = negative_index[draw]
    own = reference_index == np.arange(p)[None, :]
    while own.any():  # a negative never serves as its own reference
        draw[own] = rng.integers(0, len(negative_index), size=int(own.sum()))
        reference_index = negative_index[draw]
        own = reference_index == np.arange(p)[None, :]
    reference = scores[np.arange(k)[:, None], reference_index]
    return (scores > reference).astype(np.int8)
