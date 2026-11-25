"""The full stability experiment of one config (Stage 5), resumable.

load graph -> split (train / val / pool + val negatives + control pairs) -> damage series (pool only)
-> models (member 0; members 1 .. K-1 if the "ensemble" source is on), trained on TRAIN edges only
-> for every source and attack: recovery of every damaged graph (hypstab/experiment/recovery.py)
-> per-edge p_e (ways A and B), summary, CSV, figures.

Run directory:
  config.toml, run_info.json                 (hypstab/runinfo.py; saved before anything runs)
  split.npz, series.npz                      split and damage orderings
  models/member_XX/                          autoencoder.pt, communities.pt, denoiser.pt, record.json, done
  models.json                                member records (seeds, validation AUCs, times; no histories)
  results/<source>__<attack>.npz             raw arrays of recovery.run_attack (written when an attack ends)
  summary.json, edges.csv, figures/          derived; recomputed from results/ on every (re)run
Resume: finished members and finished results/*.npz are loaded, not recomputed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from hypstab.config import HypstabConfig
from hypstab.data.graph import GraphData, load_graph
from hypstab.data.split import EdgeSplit, split_edges
from hypstab.experiment.damage import series_from_config
from hypstab.experiment.models import get_bundles
from hypstab.experiment.pe import edge_rows, per_edge, summarize_attack
from hypstab.experiment.recovery import METHODS, run_attack


def make_split(graph: GraphData, config: HypstabConfig) -> EdgeSplit:
    s = config.split
    return split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction, s.pool_fraction,
                       s.control_pairs, s.seed)


def _check_or_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Save; on resume the saved arrays must be identical (same split / series)."""
    if path.exists():
        saved = np.load(path)
        for key, value in arrays.items():
            if not np.array_equal(saved[key], value):
                raise RuntimeError(f"resume: {path.name}[{key}] differs from the saved run")
        return
    np.savez_compressed(path, **arrays)


def run_experiment(config: HypstabConfig, run_dir: Path, device: torch.device, graph: GraphData | None = None,
                   log_every: int = 0) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if graph is None:
        graph = load_graph(config.data.dataset, config.data.root, config.data.download)
    split = make_split(graph, config)
    _check_or_save_npz(run_dir / "split.npz", {
        "train": split.train, "val": split.val, "pool": split.pool,
        "val_negatives": split.val_negatives, "control_pairs": split.control_pairs})
    all_edges = split.all_edges
    series = series_from_config(split, all_edges, graph.node_count, config)
    _check_or_save_npz(run_dir / "series.npz", {
        f"{attack}__{item.series_index}": item.order for attack, items in series.items() for item in items})

    K = config.recovery.reconstructions
    sources = config.experiment.sources
    members = K if "ensemble" in sources else 1
    bundles = get_bundles(graph, split, config, members, run_dir, device, log_every=log_every)
    records = [{k: v for k, v in b.record.items() if k != "histories"} for b in bundles]
    (run_dir / "models.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    timings: dict[str, float] = {}
    results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for source in sources:
        models = [bundles[0]] * K if source == "noise" else bundles
        results[source] = {}
        for attack, items in series.items():
            path = results_dir / f"{source}__{attack}.npz"
            if path.exists():
                results[source][attack] = dict(np.load(path))
                print(f"[recovery] {source} / {attack}: loaded", flush=True)
                continue
            started = time.perf_counter()
            arrays = run_attack(graph, split, items, models, config, device, log_prefix=f"{source} / {attack}")
            timings[f"{source}__{attack}"] = round(time.perf_counter() - started, 1)
            np.savez_compressed(path, **arrays)
            results[source][attack] = arrays

    summary = summarize(graph, split, results, config, records)
    summary["recovery_seconds_this_session"] = timings
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_edges_csv(run_dir / "edges.csv", graph, split, results, config)
    plot_ecdf(run_dir / "figures", results, config)
    return summary


def _same_class(graph: GraphData, pairs: np.ndarray):
    if graph.class_labels is None:
        return None
    labels = np.asarray(graph.class_labels)
    return labels[pairs[:, 0]] == labels[pairs[:, 1]]


def summarize(graph, split, results, config, records) -> dict[str, Any]:
    same = _same_class(graph, split.pool)
    out: dict[str, Any] = {
        "graph": {"name": graph.name, "nodes": graph.node_count, "edges": int(len(graph.edges))},
        "split_sizes": {k: int(len(getattr(split, k)))
                        for k in ("train", "val", "pool", "val_negatives", "control_pairs")},
        "K": config.recovery.reconstructions,
        "way_b_references": config.experiment.way_b_references,
        "members": [{"member": r["member"], "validation_auc": r["validation_auc"]} for r in records],
        "sources": {},
    }
    for source, attacks in results.items():
        out["sources"][source] = {attack: summarize_attack(arrays, config, METHODS, same)
                                  for attack, arrays in attacks.items()}
    # headline table: mean p_e (ways A / B) and AUC of the mean probability, per source / attack / level / method
    headline = []
    for source, attacks in out["sources"].items():
        for attack, levels in attacks.items():
            for entry in levels:
                for m, item in entry["methods"].items():
                    headline.append({
                        "source": source, "attack": attack, "damage_fraction": entry["damage_fraction"],
                        "method": m,
                        "mean_p_e_way_a": item["p_e"].get("mean"),
                        "mean_p_e_way_b": item["p_e_way_b"].get("mean"),
                        "fragile_way_a": item["p_e"].get("fraction_below_half"),
                        "fragile_way_b": item["p_e_way_b"].get("fraction_below_half"),
                        "control_way_a": item["control"].get("mean"),
                        "control_way_b": item["control_way_b"].get("mean"),
                        "auc_mean_probability": item["auc_mean_probability"],
                    })
    out["headline"] = headline
    return out


def write_edges_csv(path: Path, graph, split, results, config) -> None:
    degrees = np.bincount(split.all_edges.reshape(-1), minlength=graph.node_count)
    same = _same_class(graph, split.pool)
    rows = []
    for source, attacks in results.items():
        for attack, arrays in attacks.items():
            rows.extend(edge_rows(arrays, config, METHODS, source, attack, split.pool, degrees, same))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_ecdf(directory: Path, results, config) -> None:
    """ECDF of p_e (diffusion; ways A and B) per source / attack, one line per damage level + control."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    references = config.experiment.way_b_references
    for source, attacks in results.items():
        for attack, arrays in attacks.items():
            q = per_edge(arrays, "diffusion", references)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, (key, control_key, title) in zip(axes, (("p_e", "control", "way A (vote)"),
                                                            ("p_e_way_b", "control_way_b", "way B (average)"))):
                for level, fraction in enumerate(config.perturbation.damage_fractions):
                    values = np.sort(q[key][level][np.isfinite(q[key][level])])
                    ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                            label=f"removed {fraction:.0%}")
                control = np.sort(q[control_key][-1])
                ax.step(control, np.arange(1, len(control) + 1) / len(control), where="post", color="gray",
                        linestyle="--", label="control")
                ax.set_xlabel("p_e")
                ax.set_title(f"{source} / {attack}: {title}")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel("share of edges with p_e <= x")
            axes[0].legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(directory / f"ecdf_{source}__{attack}.png", dpi=120)
            plt.close(fig)
