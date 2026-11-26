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
# Titles and axis labels approved by the students (2026-11-26). One file per figure and attack:
#   probabilities_<attack>   one histogram per damage level + all damage levels together
#   examples_<attack>        six example edges (2 hard, 2 medium, 2 easy), middle damage level
#   easy_vs_varies_<attack>  one dot per edge: typical probability vs how much it varies, middle level
#   lift_<attack>            option 1: with diffusion minus without, one histogram per level
#   lift_scatter_<attack>    option 1: without vs with diffusion, middle level
#   trajectories_<attack>    option 3: same edge at the first vs the last level + histogram of the change
# Only edges removed already at the first level are used in trajectories_ (first vs last level).
# "Hard / medium / easy" is relative: edges at fixed ranks of the per-edge median (no threshold).
# Dots in the scatter plots use reconstruction 0 (one dot per edge and series).

EXAMPLE_RANKS = (("hard", (0.05, 0.15)), ("medium", (0.45, 0.55)), ("easy", (0.85, 0.95)))
EXAMPLE_COLORS = {"hard": "#c0392b", "medium": "#e0a039", "easy": "#2e8b57"}
LEVEL_COLORS = ("#2a7ab9", "#e07b39", "#3a9a5b", "#8e44ad", "#7f8c8d")
PROB = "Probability the edge was recovered"


def dataset_label(dataset: str, features: str) -> str:
    """Name used in figure titles, e.g. "Cora" or "Cora-structure" (identity features)."""
    return f"{dataset}-structure" if features == "identity" and dataset not in {"SBM", "BA"} else dataset


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


def first_level_changes(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Edges removed already at the first level: probability at the first and the last level.

    Returns first (M, K), last (M, K), change = last - first (M, K), weight (M, K; each edge sums to 1),
    edge (M,), series (M,).
    """
    t = trajectories(arrays)
    keep = t["first"] == 0
    edge = t["edge"][keep]
    paths = t["paths"][keep]
    K = paths.shape[1]
    counts = np.bincount(edge, minlength=int(edge.max()) + 1 if len(edge) else 0)
    weight = np.repeat((1.0 / (counts[edge] * K))[:, None], K, axis=1) if len(edge) else np.zeros((0, K))
    return {"first": paths[:, :, 0], "last": paths[:, :, -1], "change": paths[:, :, -1] - paths[:, :, 0],
            "weight": weight, "edge": edge, "series": t["series"][keep]}


def plot_attack(directory: Path, attack: str, arrays: dict[str, np.ndarray], fractions, ceiling: float,
                dataset: str = "") -> list[Path]:
    """The approved figures for one attack, from the arrays of a run. Returns the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    L = len(fractions)
    middle = L // 2
    mid = f"{fractions[middle]:.0%}"
    top = min(1.0, ceiling + 0.02)
    name = f"{dataset}, {attack}" if dataset else attack
    max_label = f"max possible ({ceiling:.3f})"
    prob_bins = np.linspace(0.0, 1.0, HIST_BINS + 1)
    centres = (prob_bins[:-1] + prob_bins[1:]) / 2

    def ceiling_line(ax, vertical=True, label=True):
        kwargs = dict(color="gray", linestyle="--", linewidth=1, label=max_label if label else None)
        (ax.axvline if vertical else ax.axhline)(ceiling, **kwargs)

    def diagonal(ax):
        ax.plot([0, top], [0, top], color="black", linewidth=1)
        ceiling_line(ax, vertical=True, label=True)
        ceiling_line(ax, vertical=False, label=False)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.text(0.04, 0.93, "above the line: went up", transform=ax.transAxes, va="top")
        ax.text(0.96, 0.04, "below the line: went down", transform=ax.transAxes, ha="right")
        ax.legend(loc="center left", fontsize=8)
        ax.grid(alpha=0.3)

    def save(fig, stem):
        fig.tight_layout()
        path = directory / f"{stem}_{attack}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    # 1. probabilities: one histogram per level, then all damage levels together
    fig, axes = plt.subplots(1, L + 1, figsize=(4.4 * (L + 1), 4.0))
    for level, fraction in enumerate(fractions):
        values, weights = level_probabilities(arrays, level)
        share = weighted_histogram(values, weights, prob_bins) * 100
        ax = axes[level]
        ax.bar(prob_bins[:-1], share, width=np.diff(prob_bins), align="edge", color="#2a7ab9",
               edgecolor="white", linewidth=0.3)
        ceiling_line(ax)
        ax.set_title(f"{name}: {fraction:.0%} of edges removed", fontsize=10)
        ax.set_xlabel(PROB)
        ax.set_ylabel("% of edges")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        axes[-1].plot(centres, share, marker="o", markersize=3, color=LEVEL_COLORS[level % len(LEVEL_COLORS)],
                      label=f"{fraction:.0%} removed")
    ceiling_line(axes[-1])
    axes[-1].set_title("All damage levels together", fontsize=10)
    axes[-1].set_xlabel(PROB)
    axes[-1].set_ylabel("% of edges")
    axes[-1].set_xlim(0, 1)
    axes[-1].legend(loc="upper left", fontsize=8)
    axes[-1].grid(alpha=0.3)
    save(fig, "probabilities")

    # 2 + 3. per edge, middle level
    values = per_edge_values(arrays, middle)
    if values:
        edges, medians, spreads = per_edge_summary(values)
        picks = example_edges(edges, medians)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for i, (label, p) in enumerate(picks):
            ax = axes[i % 2, i // 2]
            ax.hist(values[p], bins=prob_bins, color=EXAMPLE_COLORS[label])
            ceiling_line(ax, label=False)
            ax.set_title(f"{label.capitalize()} edge #{p} ({len(values[p])} probabilities)", fontsize=9)
            ax.set_xlabel(PROB, fontsize=8)
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)
        fig.suptitle(f"{name}, {mid} removed: six example edges (dashed: {max_label})")
        save(fig, "examples")

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(medians, spreads, s=6, alpha=0.4, color="#2a7ab9")
        for label, p in picks:
            j = int(np.searchsorted(edges, p))
            ax.scatter([medians[j]], [spreads[j]], s=60, color=EXAMPLE_COLORS[label], edgecolor="black", zorder=5)
        ceiling_line(ax)
        ax.set_xlabel(f"Typical {PROB.lower()}")
        ax.set_ylabel("How much it varies")
        ax.set_xlim(0, top)
        ax.set_title(f"{name}, {mid} removed: one dot per edge")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "easy_vs_varies")

    # 4. option 1, histogram per level
    lift_bins = np.linspace(-1.0, 1.0, 81)
    fig, axes = plt.subplots(1, L, figsize=(4.4 * L, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for level, fraction in enumerate(fractions):
        lift, weights = level_lift(arrays, level)
        ax = axes[level]
        ax.bar(lift_bins[:-1], weighted_histogram(lift, weights, lift_bins) * 100, width=np.diff(lift_bins),
               align="edge", color="#e07b39", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{fraction:.0%} of edges removed", fontsize=10)
        ax.set_xlabel("With diffusion minus without")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of edges")
    fig.suptitle(f"{name}: how much the diffusion changed the probability")
    save(fig, "lift")

    # 5. option 1, without vs with diffusion (reconstruction 0), middle level
    removed = arrays["removed"][middle]
    if removed.any():
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(arrays["prob_no_diffusion"][middle][removed], arrays["prob_diffusion"][middle][:, 0, :][removed],
                   s=4, alpha=0.25, color="#e07b39")
        diagonal(ax)
        ax.set_xlabel(f"{PROB}, without diffusion")
        ax.set_ylabel(f"{PROB}, with diffusion")
        ax.set_title(f"{name}, {mid} removed: with vs without diffusion", fontsize=10)
        save(fig, "lift_scatter")

    # 6. option 3: edges removed already at the first level, first vs last level
    if L >= 2:
        c = first_level_changes(arrays)
        if len(c["edge"]):
            first, last = f"{fractions[0]:.0%}", f"{fractions[-1]:.0%}"
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            ax = axes[0]
            ax.scatter(c["first"][:, 0], c["last"][:, 0], s=4, alpha=0.25, color="#6a51a3")
            diagonal(ax)
            ax.set_xlabel(f"{PROB}, {first} removed")
            ax.set_ylabel(f"{PROB}, {last} removed")
            ax.set_title(f"{name}: same edge, more damage", fontsize=10)
            change_bins = np.linspace(-1.0, 1.0, 81)
            share = weighted_histogram(c["change"], c["weight"], change_bins) * 100
            axes[1].bar(change_bins[:-1], share, width=np.diff(change_bins), align="edge", color="#6a51a3",
                        edgecolor="white", linewidth=0.3)
            axes[1].axvline(0, color="black", linewidth=0.8)
            axes[1].set_xlabel(f"Change in the {PROB.lower()} ({last} minus {first})")
            axes[1].set_ylabel("% of edges")
            axes[1].set_title("Change as the damage grows", fontsize=10)
            axes[1].grid(alpha=0.3)
            save(fig, "trajectories")
    return written


def plot_run(run_dir: Path, config, results: dict[str, dict[str, np.ndarray]] | None = None) -> list[Path]:
    """All figures of a finished "probabilities" run, from its results/noise__<attack>.npz files."""
    run_dir = Path(run_dir)
    if results is None:
        results = {}
        for attack in config.perturbation.attacks:
            path = run_dir / "results" / f"noise__{attack}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing results file: {path}")
            results[attack] = dict(np.load(path))
    ceiling = probability_ceiling(config.decoder.fermi_dirac_r, config.decoder.fermi_dirac_t)
    label = dataset_label(config.data.dataset, config.data.features)
    written = []
    for attack, arrays in results.items():
        written += plot_attack(run_dir / "figures", attack, arrays, config.perturbation.damage_fractions,
                               ceiling, dataset=label)
    return written
