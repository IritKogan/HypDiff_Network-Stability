"""Per-edge recovery probability p_e, its distribution, and the run summary (Decisions 1, 10, 11).

From the arrays of hypstab/experiment/recovery.py (one source, one attack), per level l and method m:
  n_e        = number of series in which pool edge e was removed;
  way A      p_e = r_e / (n_e * K), r_e = reconstructions (over series) in which e reappeared;
             expected_p_e = mean expected share (the same without reference-draw noise);
  way B      p_e = hits / (n_e * R) of the averaged score vs R reference control pairs;
  control    the same frequency for control pairs (~0.5 by construction in both ways);
  AUC        removed vs control of the mean decoded probability (over series and reconstructions),
             also for the first 1, 2, 5, ... reconstructions (ensemble size) and per reconstruction.

Levels inside a series are nested (cumulative damage), so level comparisons are descriptive only.
describe / distance are the old stability/edge_recovery_distribution.py functions (moved).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from hypstab.metrics import roc_auc

HISTOGRAM_BINS = np.linspace(0.0, 1.0, 11)
ENSEMBLE_SIZES = (1, 2, 5, 10, 20)


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
        "quantiles": {str(q): float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "fraction_below_half": float(np.mean(values < 0.5)),
        "fraction_zero": float(np.mean(values == 0.0)),
        "fraction_one": float(np.mean(values == 1.0)),
        "histogram": {"bin_edges": HISTOGRAM_BINS.tolist(), "counts": counts.tolist()},
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


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), np.nan)


def per_edge(arrays: dict[str, np.ndarray], method: str, references: int) -> dict[str, np.ndarray]:
    """(L, P) per-edge quantities (NaN where never removed) and (L, C) control frequencies."""
    removed = arrays["removed"]
    hit = arrays[f"hit_{method}"]
    L, S, K, P = hit.shape
    if np.any(hit & ~removed[:, :, None, :]):
        raise ValueError("an edge cannot reappear in a series where it was not removed")
    n_e = removed.sum(axis=1)
    share = arrays[f"share_{method}"]
    prob = arrays[f"prob_{method}"]
    with np.errstate(invalid="ignore"):
        return {
            "n_e": n_e,
            "p_e": _ratio(hit.sum(axis=(1, 2)), n_e * K),
            "expected_p_e": _ratio(np.nansum(share, axis=(1, 2)), n_e * K),
            "mean_probability": _ratio(np.nansum(prob, axis=(1, 2)), n_e * K),
            "p_e_way_b": _ratio(arrays[f"wayb_hits_{method}"].sum(axis=1), n_e * references),
            "expected_p_e_way_b": _ratio(np.nansum(arrays[f"wayb_share_{method}"], axis=1), n_e),
            "control": arrays[f"control_hits_{method}"].sum(axis=1) / (S * K),
            "control_expected": arrays[f"control_share_sum_{method}"].sum(axis=1) / (S * K),
            "control_way_b": arrays[f"control_wayb_hits_{method}"].sum(axis=1) / (S * references),
            "control_mean_probability": arrays[f"control_prob_k_sum_{method}"].sum(axis=1) / (S * K),
        }


def auc_by_size(arrays: dict[str, np.ndarray], method: str) -> dict[str, Any]:
    """AUC (removed vs control) of the mean probability over the first m reconstructions, and per k."""
    prob = arrays[f"prob_{method}"]                     # (L, S, K, P)
    control_sum = arrays[f"control_prob_k_sum_{method}"]  # (L, K, C)
    removed = arrays["removed"]
    L, S, K, P = prob.shape
    n_e = removed.sum(axis=1)
    sizes = [m for m in ENSEMBLE_SIZES if m <= K]
    if K not in sizes:
        sizes.append(K)
    result: dict[str, Any] = {"sizes": sizes, "by_size": [], "per_reconstruction": []}
    for level in range(L):
        evaluated = n_e[level] > 0
        by_size = []
        for m in sizes:
            edge = np.nansum(prob[level, :, :m], axis=(0, 1))[evaluated] / (n_e[level][evaluated] * m)
            ctrl = control_sum[level, :m].sum(axis=0) / (S * m)
            by_size.append(roc_auc(edge, ctrl))
        per_k = []
        for k in range(K):
            edge = np.nansum(prob[level, :, k], axis=0)[evaluated] / n_e[level][evaluated]
            per_k.append(roc_auc(edge, control_sum[level, k] / S))
        result["by_size"].append(by_size)
        result["per_reconstruction"].append(per_k)
    return result


def summarize_attack(arrays: dict[str, np.ndarray], config, methods, pool_same_class=None) -> list[dict]:
    """Summary per damage level: every method, ways A and B, control, AUCs, class split."""
    references = config.experiment.way_b_references
    fractions = config.perturbation.damage_fractions
    per_method = {m: per_edge(arrays, m, references) for m in methods}
    auc = {m: auc_by_size(arrays, m) for m in methods}
    removed = arrays["removed"]
    levels = []
    for level, fraction in enumerate(fractions):
        evaluated = per_method[methods[0]]["n_e"][level] > 0
        entry: dict[str, Any] = {
            "damage_fraction": fraction,
            "series": int(removed.shape[1]),
            "edges_removed_per_series": int(removed[level, 0].sum()),
            "pool_edges_evaluated": int(evaluated.sum()),
            "n_e": describe(per_method[methods[0]]["n_e"][level][evaluated]),
            "methods": {},
        }
        for m in methods:
            q = per_method[m]
            item = {
                "p_e": describe(q["p_e"][level]),
                "expected_p_e": describe(q["expected_p_e"][level]),
                "control": describe(q["control"][level]),
                "p_e_vs_control": distance(q["p_e"][level], q["control"][level]),
                "p_e_way_b": describe(q["p_e_way_b"][level]),
                "expected_p_e_way_b": describe(q["expected_p_e_way_b"][level]),
                "control_way_b": describe(q["control_way_b"][level]),
                "p_e_way_b_vs_control": distance(q["p_e_way_b"][level], q["control_way_b"][level]),
                "auc_mean_probability": roc_auc(q["mean_probability"][level][evaluated],
                                                q["control_mean_probability"][level]),
                "auc_by_size": dict(zip(map(str, auc[m]["sizes"]), auc[m]["by_size"][level])),
                "auc_per_reconstruction": auc[m]["per_reconstruction"][level],
            }
            if pool_same_class is not None:
                same = pool_same_class
                item["p_e_same_class"] = describe(q["p_e"][level][same])
                item["p_e_different_class"] = describe(q["p_e"][level][~same])
                item["p_e_way_b_same_class"] = describe(q["p_e_way_b"][level][same])
                item["p_e_way_b_different_class"] = describe(q["p_e_way_b"][level][~same])
            entry["methods"][m] = item
        if "diffusion" in methods:
            for other in methods:
                if other != "diffusion":
                    entry[f"diffusion_vs_{other}"] = {
                        "way_a": distance(per_method["diffusion"]["p_e"][level], per_method[other]["p_e"][level]),
                        "way_b": distance(per_method["diffusion"]["p_e_way_b"][level],
                                          per_method[other]["p_e_way_b"][level]),
                    }
        entry["way_a_vs_way_b"] = {m: distance(per_method[m]["p_e"][level], per_method[m]["p_e_way_b"][level])
                                   for m in methods}
        levels.append(entry)
    return levels


def edge_rows(arrays, config, methods, source: str, attack: str, pool: np.ndarray, degrees: np.ndarray,
              pool_same_class=None) -> list[dict]:
    """One CSV row per (level, removed pool edge) with every method's way A / way B values."""
    references = config.experiment.way_b_references
    per_method = {m: per_edge(arrays, m, references) for m in methods}
    rows = []
    n_e = per_method[methods[0]]["n_e"]
    for level, fraction in enumerate(config.perturbation.damage_fractions):
        for index in np.flatnonzero(n_e[level] > 0):
            a, b = (int(v) for v in pool[index])
            row = {
                "source": source, "attack": attack, "damage_fraction": fraction, "pool_index": int(index),
                "node_a": a, "node_b": b, "degree_a": int(degrees[a]), "degree_b": int(degrees[b]),
                "same_class": None if pool_same_class is None else bool(pool_same_class[index]),
                "n_e": int(n_e[level, index]),
            }
            for m in methods:
                q = per_method[m]
                row[f"p_e_{m}"] = float(q["p_e"][level, index])
                row[f"expected_p_e_{m}"] = float(q["expected_p_e"][level, index])
                row[f"p_e_way_b_{m}"] = float(q["p_e_way_b"][level, index])
                row[f"expected_p_e_way_b_{m}"] = float(q["expected_p_e_way_b"][level, index])
                row[f"mean_probability_{m}"] = float(q["mean_probability"][level, index])
            rows.append(row)
    return rows
