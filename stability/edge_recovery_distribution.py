"""Per-edge recovery probability p_e and its distribution (Decision 1).

For one attack with S cumulative series, L damage levels, P pool edges and K
reconstructions per damaged graph:

* ``removed[l, s, e]``   edge e was removed at level l in series s;
* ``recovered[l, s, e]`` number of the K reconstructions in which it reappeared;
* n_e(l) = sum_s removed,  r_e(l) = sum_s recovered,  p_e(l) = r_e / (n_e * K).

Control pairs (true non-edges never used in training) get the same frequency
p_c(l) = sum_s hits / (S * K): the false-positive reference distribution.

Levels inside a series are nested (cumulative damage), so p_e at different
levels are NOT independent samples; the distances reported between levels are
descriptive only. Comparisons between attacks or against the control group do
not have this dependence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

HISTOGRAM_BINS = np.linspace(0.0, 1.0, 11)


def per_edge_probability(removed: np.ndarray, recovered: np.ndarray, reconstructions: int):
    """Return n_e, r_e, p_e with shape (L, P); p_e is NaN where n_e == 0."""
    if removed.shape != recovered.shape:
        raise ValueError("removed and recovered must have the same shape")
    if np.any(recovered[~removed] != 0):
        raise ValueError("an edge cannot be recovered in a series where it was not removed")
    if np.any(recovered > reconstructions):
        raise ValueError("recovered counts cannot exceed the number of reconstructions")
    n_e = removed.sum(axis=1)
    r_e = recovered.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_e = np.where(n_e > 0, r_e / (n_e * reconstructions), np.nan)
    return n_e, r_e, p_e


def control_probability(hits: np.ndarray, reconstructions: int) -> np.ndarray:
    """hits: (L, S, C) counts in [0, K] -> (L, C) frequency."""
    series = hits.shape[1]
    return hits.sum(axis=1) / (series * reconstructions)


def describe(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    counts, _ = np.histogram(values, bins=HISTOGRAM_BINS)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            str(q): float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "fraction_zero": float(np.mean(values == 0.0)),
        "fraction_one": float(np.mean(values == 1.0)),
        "histogram": {
            "bin_edges": HISTOGRAM_BINS.tolist(),
            "counts": counts.tolist(),
        },
    }


def distance(a: np.ndarray, b: np.ndarray) -> dict[str, float] | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return None
    return {
        "ks_statistic": float(ks_2samp(a, b).statistic),
        "wasserstein": float(wasserstein_distance(a, b)),
        "mean_difference": float(a.mean() - b.mean()),
    }
