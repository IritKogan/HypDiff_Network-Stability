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
# Approved by the students 2026-11-26 (one file per figure and attack):
#   probabilities_<attack>   wide histogram per damage level + the levels together on one graph
#   examples_<attack>        six example edges (2 hard, 2 medium, 2 easy): every probability each received
#   easy_vs_varies_<attack>  one dot per edge: typical probability (median) vs how much it varies (5-95%)
#   lift_<attack>            option 1: histogram of diffusion - no_diffusion per level
#   lift_scatter_<attack>    option 1: one dot per (series, edge): without vs with diffusion
#   trajectories_<attack>    option 3: first vs last damage level (one dot per (series, edge)) + change histogram
# "Hard / medium / easy" in the examples is relative: edges at fixed ranks of the per-edge median.

EXAMPLE_RANKS = (("hard", (0.05, 0.15)), ("medium", (0.45, 0.55)), ("easy", (0.85, 0.95)))
EXAMPLE_COLORS = {"hard": "#c0392b", "medium": "#e0a039", "easy": "#2e8b57"}
LEVEL_COLORS = ("#2a7ab9", "#e07b39", "#3a9a5b", "#8e44ad", "#7f8c8d")


def per_edge_values(arrays: dict[str, np.ndarray], level: int) -> dict[int, np.ndarray]:
    """Every diffusion probability of each removed edge at one level (n_e * K values per edge)."""
    removed = arrays["removed"][level]
    prob = arrays["prob_diffusion"][level]
    return {int(p): prob[removed[:, p]][:, :, p].ravel() for p in np.flatnonzero(removed.any(axis=0))}


def per_edge_summary(values: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(edges, median, spread = 95th - 5th percentile) of each edge's own probabilities."""
    edges = np.array(sorted(values), dtype=np.int64)
    q = np.array([np.quantile(values[p], [0.05, 0.5, 0.95]) for p in edges]).reshape(-1, 3)
    return edges, q[:, 1], q[:, 2] - q[:, 0]


def example_edges(edges: np.ndarray, medians: np.ndarray) -> list[tuple[str, int]]:
    """Two edges per group at fixed ranks of the per-edge median (deterministic, no threshold)."""
    order = edges[np.argsort(medians, kind="stable")]
    picks = []
    for label, ranks in EXAMPLE_RANKS:
        for r in ranks:
            picks.append((label, int(order[min(int(r * len(order)), len(order) - 1)])))
    return picks


def _line_y(ax, ceiling):
    ax.axvline(ceiling, color="gray", linestyle="--", linewidth=1)


def _diagonal(ax, top):
    ax.plot([0, top], [0, top], color="black", linewidth=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def plot_attack(directory: Path, attack: str, arrays: dict[str, np.ndarray], fractions, ceiling: float,
                seed: int = 0) -> list[Path]:
    """The approved figures for one attack (see the list above). Returns the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    del seed  # figures are deterministic (fixed ranks, reconstruction 0)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    L = len(fractions)
    middle = L // 2
    top = min(1.0, ceiling + 0.02)
    prob_bins = np.linspace(0.0, 1.0, HIST_BINS + 1)
    centres = (prob_bins[:-1] + prob_bins[1:]) / 2

    def save(fig, name):
        fig.tight_layout()
        path = directory / f"{name}_{attack}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    # 1 + 2. probabilities: one histogram per level, then all levels on one graph
    fig, axes = plt.subplots(1, L + 1, figsize=(4.2 * (L + 1), 3.8))
    for level, fraction in enumerate(fractions):
        values, weights = level_probabilities(arrays, level)
        share = weighted_histogram(values, weights, prob_bins) * 100
        ax = axes[level]
        ax.bar(prob_bins[:-1], share, width=np.diff(prob_bins), align="edge", color="#2a7ab9",
               edgecolor="white", linewidth=0.3)
        _line_y(ax, ceiling)
        ax.set_title(f"{attack}: {fraction:.0%} of edges removed")
        ax.set_xlabel("probability")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        axes[-1].plot(centres, share, marker="o", markersize=3, color=LEVEL_COLORS[level % len(LEVEL_COLORS)],
                      label=f"{fraction:.0%} removed")
    axes[0].set_ylabel("% of edges")
    _line_y(axes[-1], ceiling)
    axes[-1].set_title(f"all levels (max possible {ceiling:.3f})")
    axes[-1].set_xlabel("probability")
    axes[-1].set_xlim(0, 1)
    axes[-1].legend(loc="upper left")
    axes[-1].grid(alpha=0.3)
    save(fig, "probabilities")

    # 3 + 4. per edge, middle level
    values = per_edge_values(arrays, middle)
    if values:
        edges, medians, spreads = per_edge_summary(values)
        picks = example_edges(edges, medians)
        fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
        for i, (label, p) in enumerate(picks):
            ax = axes[i % 2, i // 2]
            ax.hist(values[p], bins=prob_bins, color=EXAMPLE_COLORS[label])
            _line_y(ax, ceiling)
            ax.set_title(f"{label} edge #{p} ({len(values[p])} probabilities)", fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel("probability")
        for ax in axes[:, 0]:
            ax.set_ylabel("count")
        fig.suptitle(f"{attack}, {fractions[middle]:.0%} removed: six example edges")
        save(fig, "examples")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(medians, spreads, s=6, alpha=0.4, color="#2a7ab9")
        for label, p in picks:
            j = int(np.searchsorted(edges, p))
            ax.scatter([medians[j]], [spreads[j]], s=60, color=EXAMPLE_COLORS[label], edgecolor="black", zorder=5)
        ax.set_xlabel("typical probability of the edge")
        ax.set_ylabel("how much its probability varies")
        ax.set_xlim(0, top)
        ax.set_title(f"{attack}, {fractions[middle]:.0%} removed: one dot per edge")
        ax.grid(alpha=0.3)
        save(fig, "easy_vs_varies")

    # 5. option 1, histogram of the lift per level
    lift_bins = np.linspace(-1.0, 1.0, 81)
    fig, axes = plt.subplots(1, L, figsize=(4.2 * L, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for level, fraction in enumerate(fractions):
        lift, weights = level_lift(arrays, level)
        ax = axes[level]
        ax.bar(lift_bins[:-1], weighted_histogram(lift, weights, lift_bins) * 100, width=np.diff(lift_bins),
               align="edge", color="#e07b39", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{attack}: {fraction:.0%} removed")
        ax.set_xlabel("with diffusion - without")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of edges")
    save(fig, "lift")

    # 6. option 1, without vs with diffusion (reconstruction 0), middle level
    removed = arrays["removed"][middle]
    if removed.any():
        without = arrays["prob_no_diffusion"][middle][removed]
        with_diffusion = arrays["prob_diffusion"][middle][:, 0, :][removed]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(without, with_diffusion, s=4, alpha=0.25, color="#e07b39")
        _diagonal(ax, top)
        ax.set_xlabel("probability without diffusion")
        ax.set_ylabel("probability with diffusion")
        ax.set_title(f"{attack}, {fractions[middle]:.0%} removed: one dot per edge and series")
        save(fig, "lift_scatter")

    # 7. option 3, first vs last level (reconstruction 0) + histogram of the change (all reconstructions)
    if L >= 2:
        t = trajectories(arrays)
        full = t["first"] == 0
        if full.any():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            ax = axes[0]
            ax.scatter(t["paths"][full, 0, 0], t["paths"][full, 0, -1], s=4, alpha=0.25, color="#6a51a3")
            _diagonal(ax, top)
            ax.set_xlabel(f"probability when {fractions[0]:.0%} of edges are removed")
            ax.set_ylabel(f"probability when {fractions[-1]:.0%} of edges are removed")
            ax.set_title(f"{attack}: one dot per edge and series")
            change_bins = np.linspace(-1.0, 1.0, 81)
            share = weighted_histogram(t["change"], t["weight"], change_bins) * 100
            axes[1].bar(change_bins[:-1], share, width=np.diff(change_bins), align="edge", color="#6a51a3",
                        edgecolor="white", linewidth=0.3)
            axes[1].axvline(0, color="black", linewidth=0.8)
            axes[1].set_xlabel(f"probability at {fractions[-1]:.0%} - at first removal")
            axes[1].set_ylabel("% of edges")
            axes[1].set_title("change as the damage grows")
            axes[1].grid(alpha=0.3)
            save(fig, "trajectories")
    return written
