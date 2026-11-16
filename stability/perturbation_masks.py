"""Cumulative damage series drawn from the held-out edge pool.

A series is one ordering of the pool edges. Damage level L removes the first
``round(L * |E|)`` edges of the ordering, so the removed sets are nested
(10% subset of 20% subset of 30%) inside a series — cumulative damage. Independent
series give N different removed sets per level, from which the per-edge
recovery probability p_e is estimated (project_context.md, Decisions 1 and 6).

Attacks:
* ``random``                 uniform random ordering;
* ``targeted_weighted``      ordering sampled without replacement with
                             P(edge) ~ k_i * k_j (Efraimidis–Spirakis keys);
* ``targeted_deterministic`` k_i * k_j descending, ties by pool index; one series.

Degrees k are taken on the full original graph (the attacker sees the graph).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ATTACK_CODES = {"random": 0, "targeted_weighted": 1, "targeted_deterministic": 2}


@dataclass(frozen=True)
class DamageSeries:
    attack: str
    series_index: int
    order: np.ndarray            # permutation of pool indices
    removal_counts: tuple[int, ...]

    def removed_pool_indices(self, level_index: int) -> np.ndarray:
        return self.order[: self.removal_counts[level_index]]


def removal_count(total_edges: int, fraction: float) -> int:
    return int(np.floor(total_edges * fraction + 0.5))


def degree_products(all_edges: np.ndarray, node_count: int, pool: np.ndarray) -> np.ndarray:
    degrees = np.bincount(all_edges.reshape(-1), minlength=node_count).astype(np.float64)
    return degrees[pool[:, 0]] * degrees[pool[:, 1]]


def _series_rng(seed: int, attack: str, series_index: int) -> np.random.Generator:
    return np.random.default_rng([seed, ATTACK_CODES[attack], series_index])


def make_damage_series(
    pool: np.ndarray,
    all_edges: np.ndarray,
    node_count: int,
    damage_fractions,
    attacks,
    series: int,
    seed: int,
) -> dict[str, list[DamageSeries]]:
    counts = tuple(removal_count(len(all_edges), fraction) for fraction in damage_fractions)
    if max(counts) > len(pool):
        raise ValueError(
            f"largest damage level removes {max(counts)} edges but the pool has {len(pool)}"
        )
    if any(count <= 0 for count in counts):
        raise ValueError("every damage level must remove at least one edge")
    weights = degree_products(all_edges, node_count, pool)
    result: dict[str, list[DamageSeries]] = {}
    for attack in attacks:
        if attack == "targeted_deterministic":
            order = np.lexsort((np.arange(len(pool)), -weights))
            result[attack] = [DamageSeries(attack, 0, order, counts)]
            continue
        items = []
        for series_index in range(series):
            rng = _series_rng(seed, attack, series_index)
            if attack == "random":
                order = rng.permutation(len(pool))
            elif attack == "targeted_weighted":
                # Efraimidis–Spirakis: sort by log(u) / w descending.
                keys = np.log(rng.random(len(pool))) / np.maximum(weights, 1e-12)
                order = np.argsort(-keys, kind="stable")
            else:
                raise ValueError(f"unknown attack: {attack}")
            items.append(DamageSeries(attack, series_index, order, counts))
        result[attack] = items
    return result
