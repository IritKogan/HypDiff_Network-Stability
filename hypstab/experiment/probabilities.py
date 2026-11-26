"""Analysis of experiment.output = "probabilities" (advisor meeting, 2026-11-26).

The result is the decoded probabilities themselves: no control pairs, no threshold, no mean over edges.
Inputs are the arrays of recovery.run_attack_probabilities:
  removed (L,S,P), prob_diffusion (L,S,K,P), prob_no_diffusion (L,S,P).

Equal weight per edge
  Edge e is removed in n_e of the S series (n_e differs between edges, e.g. 3-17 on Cora at 20%), so it
  has n_e * K probabilities at a level. Each of them gets weight 1 / (n_e * K): every edge counts once in
  the pooled histogram, however often it was drawn.

How hard the model works for an edge (no threshold anywhere):
  option 1 "lift"   = prob_diffusion - prob_no_diffusion, per (series, reconstruction, edge): how much, and
                      in which direction, the diffusion moved the edge's probability on the same damaged graph;
  option 3 "change" = prob_diffusion at the LAST damage level - at the FIRST level at which the edge was
                      removed in that series, per (series, reconstruction, edge). Levels are nested and the
                      noise seed depends on (series, reconstruction) only, so a trajectory compares the same
                      noise draw on more and more damaged graphs.

Probability ceiling: the Fermi-Dirac decoder gives at most sigmoid(r / t) (distance 0); 0.881 for r=2, t=1.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

HIST_BINS = 50   # bins of the probability histograms on [0, 1] (width 0.02)


def probability_ceiling(r: float, t: float) -> float:
    """Largest probability the Fermi-Dirac decoder can give: 1 / (1 + exp(-r / t))."""
    return 1.0 / (1.0 + math.exp(-r / t))


def _point_weights(removed_level: np.ndarray, K: int) -> np.ndarray:
    """(S, P) weight of every probability of a removed edge so that each edge sums to 1 (0 elsewhere)."""
    n_e = removed_level.sum(axis=0)
    per_edge = np.zeros(n_e.shape, dtype=np.float64)
    per_edge[n_e > 0] = 1.0 / (n_e[n_e > 0] * K)
    return np.where(removed_level, per_edge[None, :], 0.0)


def level_probabilities(arrays: dict[str, np.ndarray], level: int) -> tuple[np.ndarray, np.ndarray]:
    """All diffusion probabilities of removed edges at one level and their equal-edge weights (flat)."""
    removed = arrays["removed"][level]                      # (S, P)
    prob = arrays["prob_diffusion"][level]                  # (S, K, P)
    K = prob.shape[1]
    weights = np.broadcast_to(_point_weights(removed, K)[:, None, :], prob.shape)
    mask = np.broadcast_to(removed[:, None, :], prob.shape)
    return prob[mask].astype(np.float64), weights[mask]


def level_lift(arrays: dict[str, np.ndarray], level: int) -> tuple[np.ndarray, np.ndarray]:
    """Option 1: prob_diffusion - prob_no_diffusion of every removed edge / series / reconstruction."""
    removed = arrays["removed"][level]
    prob = arrays["prob_diffusion"][level]
    lift = prob - arrays["prob_no_diffusion"][level][:, None, :]
    K = prob.shape[1]
    weights = np.broadcast_to(_point_weights(removed, K)[:, None, :], prob.shape)
    mask = np.broadcast_to(removed[:, None, :], prob.shape)
    return lift[mask].astype(np.float64), weights[mask]


def trajectories(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Option 3: probability of every (series, edge) on every level from its first removal on.

    Returns
      paths (M, K, L) float  probabilities (NaN before the first removal level)
      first (M,) int         first level at which the edge was removed in that series
      edge (M,) int          pool index
      series (M,) int
      change (M, K) float    last level - first removal level (only rows with first < L - 1 are returned)
      weight (M, K) float    equal weight per edge over all its (series, reconstruction) changes
    """
    removed = arrays["removed"]                              # (L, S, P)
    prob = arrays["prob_diffusion"]                          # (L, S, K, P)
    L, S, K, P = prob.shape
    if L < 2:
        raise ValueError("trajectories need at least two damage levels")
    if not np.all(removed[:-1] <= removed[1:]):
        raise ValueError("damage levels must be nested (cumulative series)")
    ever = removed[-1]                                       # (S, P)
    first = np.where(ever, np.argmax(removed, axis=0), L)    # (S, P)
    rows = np.argwhere(first < L - 1)                        # (M, 2) series, edge
    series, edge = rows[:, 0], rows[:, 1]
    first_level = first[series, edge]
    # advanced indices separated by a slice go first: prob[:, series, :, edge] is (M, L, K) -> (M, K, L)
    paths = prob[:, series, :, edge].transpose(0, 2, 1).astype(np.float64)
    for level in range(L):
        paths[first_level > level, :, level] = np.nan
    change = paths[:, :, -1] - paths[np.arange(len(rows)), :, first_level]
    counts = np.bincount(edge, minlength=P)
    weight = np.repeat((1.0 / (counts[edge] * K))[:, None], K, axis=1)
    return {"paths": paths, "first": first_level, "edge": edge, "series": series, "change": change,
            "weight": weight}


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs) -> list[float]:
    values, weights = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return [float(values[order][min(np.searchsorted(cumulative, q, side="left"), len(values) - 1)]) for q in qs]


def weighted_histogram(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of the total weight in every bin (sums to 1 when all values fall inside the bins)."""
    counts, _ = np.histogram(np.asarray(values).ravel(), bins=edges, weights=np.asarray(weights).ravel())
    return counts / np.asarray(weights).sum()


QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def _describe(values, weights) -> dict[str, Any]:
    q = weighted_quantiles(values, weights, QUANTILES)
    return {"quantiles": dict(zip([f"q{int(x * 100):02d}" for x in QUANTILES], q)),
            "share_negative": float(np.asarray(weights)[np.asarray(values) < 0].sum() / np.sum(weights))}


def summarize_attack(arrays: dict[str, np.ndarray], fractions) -> dict[str, Any]:
    """Distributions per level (equal weight per edge): probabilities, option 1, and option 3."""
    prob_bins = np.linspace(0.0, 1.0, HIST_BINS + 1)
    levels = []
    for level, fraction in enumerate(fractions):
        values, weights = level_probabilities(arrays, level)
        lift, lift_w = level_lift(arrays, level)
        n_e = arrays["removed"][level].sum(axis=0)
        levels.append({
            "damage_fraction": float(fraction),
            "edges_removed_at_least_once": int((n_e > 0).sum()),
            "times_removed_min_max": [int(n_e[n_e > 0].min()), int(n_e.max())] if (n_e > 0).any() else None,
            "probabilities": {
                "quantiles": _describe(values, weights)["quantiles"],
                "histogram_bins": prob_bins.tolist(),
                "histogram_share": weighted_histogram(values, weights, prob_bins).tolist(),
            },
            "lift_diffusion_minus_no_diffusion": _describe(lift, lift_w),
        })
    out: dict[str, Any] = {"levels": levels}
    if len(fractions) >= 2 and arrays["removed"].shape[1] > 0:
        t = trajectories(arrays)
        if len(t["edge"]):
            out["change_first_to_last_level"] = _describe(t["change"], t["weight"])
            out["change_first_to_last_level"]["edges"] = int(len(np.unique(t["edge"])))
    return out


def edge_rows(arrays, fractions, attack: str, pool: np.ndarray, degrees: np.ndarray, same_class=None):
    """One CSV row per (attack, level, pool edge that was removed at least once): its own distribution."""
    rows = []
    removed_all = arrays["removed"]
    prob_all = arrays["prob_diffusion"]
    for level, fraction in enumerate(fractions):
        removed = removed_all[level]
        n_e = removed.sum(axis=0)
        for p in np.flatnonzero(n_e):
            mask = removed[:, p]
            values = prob_all[level][mask][:, :, p].ravel().astype(np.float64)
            lift = (prob_all[level][mask][:, :, p] - arrays["prob_no_diffusion"][level][mask, p][:, None]).ravel()
            q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
            u, v = int(pool[p, 0]), int(pool[p, 1])
            row = {"attack": attack, "damage_fraction": float(fraction), "pool_index": int(p), "u": u, "v": v,
                   "degree_u": int(degrees[u]), "degree_v": int(degrees[v]), "times_removed": int(n_e[p]),
                   "probabilities": int(values.size)}
            row.update({f"prob_q{int(x * 100):02d}": float(val)
                        for x, val in zip((0, .05, .25, .5, .75, .95, 1), q)})
            row["lift_median"] = float(np.median(lift))
            if same_class is not None:
                row["same_class"] = bool(same_class[p])
            rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- figures


def plot_attack(directory: Path, attack: str, arrays: dict[str, np.ndarray], fractions, ceiling: float,
                trajectory_sample: int = 300, seed: int = 0) -> list[Path]:
    """Wide figures for one attack: probability histograms, option 1, option 3, per-edge ranges."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    L = len(fractions)
    prob_bins = np.linspace(0.0, 1.0, HIST_BINS + 1)

    # 1. probabilities: one histogram per level (equal weight per edge) + ECDF overlay
    fig, axes = plt.subplots(1, L + 1, figsize=(4.2 * (L + 1), 3.8))
    for level, fraction in enumerate(fractions):
        values, weights = level_probabilities(arrays, level)
        share = weighted_histogram(values, weights, prob_bins)
        ax = axes[level]
        ax.bar(prob_bins[:-1], share * 100, width=np.diff(prob_bins), align="edge", color="#2a7ab9",
               edgecolor="white", linewidth=0.3)
        ax.axvline(ceiling, color="gray", linestyle="--", linewidth=1)
        ax.text(ceiling, ax.get_ylim()[1] * 0.95, f" max {ceiling:.3f}", color="gray", fontsize=8, ha="right")
        ax.set_title(f"{attack}: removed {fraction:.0%}")
        ax.set_xlabel("decoded probability (diffusion)")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order]) / weights.sum()
        axes[-1].step(values[order], cumulative, where="post", label=f"removed {fraction:.0%}")
    axes[0].set_ylabel("% of edges (each edge weight 1)")
    axes[-1].axvline(ceiling, color="gray", linestyle="--", linewidth=1)
    axes[-1].set_title("ECDF, all levels")
    axes[-1].set_xlabel("decoded probability")
    axes[-1].set_ylabel("share of edges with probability <= x")
    axes[-1].set_xlim(0, 1)
    axes[-1].legend(loc="upper left")
    axes[-1].grid(alpha=0.3)
    fig.tight_layout()
    path = directory / f"probabilities_{attack}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 2. option 1: lift = diffusion - no_diffusion
    lift_bins = np.linspace(-1.0, 1.0, 81)
    fig, axes = plt.subplots(1, L, figsize=(4.2 * L, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for level, fraction in enumerate(fractions):
        values, weights = level_lift(arrays, level)
        share = weighted_histogram(values, weights, lift_bins)
        ax = axes[level]
        ax.bar(lift_bins[:-1], share * 100, width=np.diff(lift_bins), align="edge", color="#e07b39",
               edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{attack}: removed {fraction:.0%}")
        ax.set_xlabel("probability with diffusion - without")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of edges (each edge weight 1)")
    fig.tight_layout()
    path = directory / f"lift_{attack}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 3. option 3: trajectories over damage levels + histogram of the change
    if L >= 2:
        t = trajectories(arrays)
        if len(t["edge"]):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            full = np.flatnonzero(t["first"] == 0)
            rng = np.random.default_rng(seed)
            pick = rng.choice(full, size=min(trajectory_sample, len(full)), replace=False) if len(full) else []
            x = [f"{f:.0%}" for f in fractions]
            for i in pick:
                axes[0].plot(x, t["paths"][i, 0], color="#2a7ab9", alpha=0.12, linewidth=0.8)
            axes[0].axhline(ceiling, color="gray", linestyle="--", linewidth=1)
            axes[0].set_ylim(0, 1)
            axes[0].set_xlabel("damage level")
            axes[0].set_ylabel("decoded probability (reconstruction 0)")
            axes[0].set_title(f"{attack}: {len(pick)} edges removed from the first level")
            axes[0].grid(alpha=0.3)
            change_bins = np.linspace(-1.0, 1.0, 81)
            share = weighted_histogram(t["change"], t["weight"], change_bins)
            axes[1].bar(change_bins[:-1], share * 100, width=np.diff(change_bins), align="edge",
                        color="#6a51a3", edgecolor="white", linewidth=0.3)
            axes[1].axvline(0, color="black", linewidth=0.8)
            axes[1].set_xlabel(f"probability at {fractions[-1]:.0%} - at first removal level")
            axes[1].set_ylabel("% of edges (each edge weight 1)")
            axes[1].set_title("change as damage grows")
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            path = directory / f"trajectories_{attack}.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            written.append(path)

    # 4. per-edge distributions at the middle level: edges sorted by median, 5-95% and 25-75% bands
    level = L // 2
    removed = arrays["removed"][level]
    edges = np.flatnonzero(removed.any(axis=0))
    if len(edges):
        stats = []
        for p in edges:
            values = arrays["prob_diffusion"][level][removed[:, p]][:, :, p].ravel()
            stats.append(np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]))
        stats = np.array(stats)
        stats = stats[np.argsort(stats[:, 2])]
        x = np.arange(len(stats))
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(x, stats[:, 0], stats[:, 4], color="#2a7ab9", alpha=0.2, linewidth=0, label="5-95%")
        ax.fill_between(x, stats[:, 1], stats[:, 3], color="#2a7ab9", alpha=0.45, linewidth=0, label="25-75%")
        ax.plot(x, stats[:, 2], color="#1b4f7a", linewidth=1, label="median")
        ax.axhline(ceiling, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("removed edges, sorted by their median probability")
        ax.set_ylabel("decoded probability (diffusion)")
        ax.set_title(f"{attack}, removed {fractions[level]:.0%}: the probabilities of every edge")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = directory / f"per_edge_{attack}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written
