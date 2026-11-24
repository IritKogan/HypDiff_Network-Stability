"""Leakage-free edge split for edge-recovery stability experiments.

Moved from stability/edge_split.py (phase-b-work, bec9111) without changes to the
algorithm, so the same seed gives the same split as the old code (tested).

The undirected edges of the graph are split once into

* ``train``  — used to train the encoder (positives + message passing), the
  denoiser and the community centres;
* ``val``    — encoder early stopping only; never removed;
* ``pool``   — the ONLY edges that perturbations may remove; never seen in training.

Non-edge pairs are split into ``val_negatives`` (early stopping) and
``control_pairs`` (false-positive reference group). Training negatives are
resampled every epoch from non-edges that are neither validation negatives nor
control pairs. Pool edges are true edges, so they are never sampled as
negatives either: pool edges and control pairs are treated symmetrically —
neither ever receives a training signal.

Edges are stored as int64 arrays of shape (m, 2) with source < target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EdgeSplit:
    node_count: int
    train: np.ndarray
    val: np.ndarray
    pool: np.ndarray
    val_negatives: np.ndarray
    control_pairs: np.ndarray

    @property
    def all_edges(self) -> np.ndarray:
        return np.concatenate([self.train, self.val, self.pool])

    def forbidden_negative_keys(self) -> np.ndarray:
        """Sorted pair keys that must never be sampled as training negatives."""
        return np.unique(
            np.concatenate(
                [
                    pair_keys(self.all_edges, self.node_count),
                    pair_keys(self.val_negatives, self.node_count),
                    pair_keys(self.control_pairs, self.node_count),
                ]
            )
        )


def canonical_edges(edges, node_count: int) -> np.ndarray:
    array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if array.size and (array.min() < 0 or array.max() >= node_count):
        raise ValueError("edge endpoint outside [0, node_count)")
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError("self-loops are not supported")
    array = np.sort(array, axis=1)
    keys = pair_keys(array, node_count)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate undirected edges")
    return array[np.argsort(keys, kind="stable")]


def pair_keys(pairs: np.ndarray, node_count: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    low = np.minimum(pairs[:, 0], pairs[:, 1])
    high = np.maximum(pairs[:, 0], pairs[:, 1])
    return low * node_count + high


def sample_non_edges(
    count: int,
    node_count: int,
    forbidden_keys: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample ``count`` distinct pairs i<j whose keys are not forbidden."""
    total_pairs = node_count * (node_count - 1) // 2
    if count > total_pairs - len(forbidden_keys):
        raise ValueError("not enough non-edge pairs to sample from")
    forbidden = np.unique(np.asarray(forbidden_keys, dtype=np.int64))
    chosen = np.empty(0, dtype=np.int64)
    while len(chosen) < count:
        need = count - len(chosen)
        source = rng.integers(0, node_count, size=2 * need + 16)
        target = rng.integers(0, node_count, size=2 * need + 16)
        keep = source != target
        keys = pair_keys(np.stack([source[keep], target[keep]], axis=1), node_count)
        keys = keys[~np.isin(keys, forbidden)]
        keys = keys[~np.isin(keys, chosen)]
        _, first = np.unique(keys, return_index=True)
        keys = keys[np.sort(first)]
        chosen = np.concatenate([chosen, keys[:need]])
    return np.stack([chosen // node_count, chosen % node_count], axis=1)


def split_edges(
    edges,
    node_count: int,
    train_fraction: float,
    val_fraction: float,
    pool_fraction: float,
    control_pairs: int,
    seed: int,
) -> EdgeSplit:
    if not np.isclose(train_fraction + val_fraction + pool_fraction, 1.0):
        raise ValueError("split fractions must sum to 1")
    edges = canonical_edges(edges, node_count)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(edges))
    val_count = int(round(val_fraction * len(edges)))
    pool_count = int(round(pool_fraction * len(edges)))
    train_count = len(edges) - val_count - pool_count
    if min(train_count, val_count, pool_count) <= 0:
        raise ValueError("every edge group must be non-empty")
    val = edges[np.sort(order[:val_count])]
    pool = edges[np.sort(order[val_count : val_count + pool_count])]
    train = edges[np.sort(order[val_count + pool_count :])]

    edge_keys = pair_keys(edges, node_count)
    negatives = sample_non_edges(val_count + control_pairs, node_count, edge_keys, rng)
    return EdgeSplit(
        node_count=node_count,
        train=train,
        val=val,
        pool=pool,
        val_negatives=negatives[:val_count],
        control_pairs=negatives[val_count:],
    )
