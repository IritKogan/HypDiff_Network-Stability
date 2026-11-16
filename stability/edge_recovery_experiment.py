"""Leakage-free cumulative edge-recovery experiment driven by a config file.

Pipeline: load graph -> split edges (train / val / pool + val negatives +
control pairs) -> train encoder, community context and denoiser on train edges
only -> for every attack, series and damage level remove pool edges, encode the
damaged graph, run K partial-noise recoveries from t*, decode removed edges and
control pairs -> p_e distributions.

Everything needed to reproduce a run is written to its directory:
config.toml / config.json, environment.json, split.npz, series.npz,
checkpoints/, training_history.json, recovery_counts.npz, edge_recovery.csv,
control_recovery.csv, summary.json, figures/.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import torch

from stability.edge_recovery_distribution import (
    control_probability,
    describe,
    distance,
    per_edge_probability,
)
from stability.edge_split import EdgeSplit, split_edges
from stability.experiment_config import ExperimentConfig
from stability.graph_data import GraphData, load_graph
from stability.model_training import (
    TrainedModels,
    normalized_adjacency,
    resolve_device,
    train_models,
)
from stability.perturbation_masks import DamageSeries, make_damage_series

ROOT = Path(__file__).resolve().parents[1]


def recovery_seed(base_seed: int, series_index: int, reconstruction_index: int) -> int:
    """Noise seed for reconstruction k of series s.

    Shared across attacks and damage levels (common random numbers), so level
    and attack comparisons are not confounded by different noise draws.
    """
    return int(np.random.SeedSequence([base_seed, series_index, reconstruction_index]).generate_state(1)[0])


def environment_record(device: torch.device) -> dict[str, Any]:
    import networkx
    import scipy
    import sklearn
    import torch_geometric

    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - git may be unavailable (e.g. zip download)
            return None

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "packages": {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "networkx": networkx.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _decode_hits(encoder, embeddings, pairs: torch.Tensor, tau: float) -> np.ndarray:
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    probabilities = encoder.decode(embeddings, pairs)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("decoder produced non-finite probabilities")
    return (probabilities >= tau).cpu().numpy()


def run_recoveries(
    graph: GraphData,
    split: EdgeSplit,
    series: dict[str, list[DamageSeries]],
    models: TrainedModels,
    config: ExperimentConfig,
    device: torch.device,
    log: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    rec = config.recovery
    levels = len(config.perturbation.damage_fractions)
    features = graph.features.to(device)
    labels = models.context.labels.to(device)
    centers = models.context.centers.to(device)
    control = torch.as_tensor(split.control_pairs, device=device)
    kept_edges = np.concatenate([split.train, split.val])
    results = {}
    for attack, items in series.items():
        shape = (levels, len(items), len(split.pool))
        removed = np.zeros(shape, dtype=bool)
        recovered = np.zeros(shape, dtype=np.int16)
        no_diffusion = np.zeros(shape, dtype=bool)
        control_hits = np.zeros((levels, len(items), len(split.control_pairs)), dtype=np.int16)
        control_no_diffusion = np.zeros(control_hits.shape, dtype=bool)
        for item in items:
            s = item.series_index
            started = time.time()
            for level in range(levels):
                pool_index = item.removed_pool_indices(level)
                keep = np.ones(len(split.pool), dtype=bool)
                keep[pool_index] = False
                damaged = np.concatenate([kept_edges, split.pool[keep]])
                adjacency = normalized_adjacency(graph.node_count, damaged, device)
                removed_pairs = torch.as_tensor(split.pool[pool_index], device=device)
                with torch.no_grad():
                    embeddings = models.encoder.encode(features, adjacency)[2]
                    no_diffusion[level, s, pool_index] = _decode_hits(
                        models.encoder, embeddings, removed_pairs, rec.tau
                    )
                    control_no_diffusion[level, s] = _decode_hits(
                        models.encoder, embeddings, control, rec.tau
                    )
                    for k in range(rec.reconstructions):
                        generator = torch.Generator(device=device)
                        generator.manual_seed(recovery_seed(rec.seed, s, k))
                        recovered_embeddings = models.denoiser.sample_from_tstar(
                            damaged_embeddings=embeddings,
                            damaged_adjacency=adjacency,
                            labels=labels,
                            prototypes=centers,
                            t_star=rec.t_star,
                            generator=generator,
                        )
                        recovered[level, s, pool_index] += _decode_hits(
                            models.encoder, recovered_embeddings, removed_pairs, rec.tau
                        )
                        control_hits[level, s] += _decode_hits(
                            models.encoder, recovered_embeddings, control, rec.tau
                        )
                removed[level, s, pool_index] = True
            if log:
                print(f"[recovery] {attack} series {s + 1}/{len(items)} "
                      f"({time.time() - started:.1f}s)", flush=True)
        results[attack] = {
            "removed": removed,
            "recovered": recovered,
            "no_diffusion": no_diffusion,
            "control_hits": control_hits,
            "control_no_diffusion": control_no_diffusion,
        }
    return results


def summarize(
    graph: GraphData,
    split: EdgeSplit,
    results: dict[str, dict[str, np.ndarray]],
    config: ExperimentConfig,
) -> tuple[dict[str, Any], list[dict], list[dict]]:
    K = config.recovery.reconstructions
    fractions = config.perturbation.damage_fractions
    degrees = np.bincount(split.all_edges.reshape(-1), minlength=graph.node_count)
    classes = graph.class_labels

    def same_class(pairs):
        return None if classes is None else classes[pairs[:, 0]] == classes[pairs[:, 1]]

    pool_same = same_class(split.pool)
    control_same = same_class(split.control_pairs)
    summary: dict[str, Any] = {}
    edge_rows, control_rows = [], []
    for attack, arrays in results.items():
        n_e, r_e, p_e = per_edge_probability(arrays["removed"], arrays["recovered"], K)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_no_diffusion = np.where(n_e > 0, arrays["no_diffusion"].sum(axis=1) / n_e, np.nan)
        p_control = control_probability(arrays["control_hits"], K)
        p_control_no_diffusion = arrays["control_no_diffusion"].mean(axis=1)
        attack_summary = []
        for level, fraction in enumerate(fractions):
            evaluated = n_e[level] > 0
            entry = {
                "damage_fraction": fraction,
                "series": int(arrays["removed"].shape[1]),
                "edges_removed_per_series": int(arrays["removed"][level, 0].sum()),
                "pool_edges_evaluated": int(evaluated.sum()),
                "n_e_distribution": describe(n_e[level][evaluated]),
                "p_e": describe(p_e[level]),
                "p_e_no_diffusion": describe(p_no_diffusion[level]),
                "control_false_positive": describe(p_control[level]),
                "control_false_positive_no_diffusion": describe(p_control_no_diffusion[level]),
                "p_e_vs_control": distance(p_e[level], p_control[level]),
                "p_e_vs_no_diffusion": distance(p_e[level], p_no_diffusion[level]),
            }
            if pool_same is not None:
                entry["p_e_same_class"] = describe(p_e[level][pool_same])
                entry["p_e_different_class"] = describe(p_e[level][~pool_same])
                entry["same_vs_different_class"] = distance(
                    p_e[level][pool_same], p_e[level][~pool_same]
                )
            attack_summary.append(entry)
            for index in np.flatnonzero(evaluated):
                source, target = (int(v) for v in split.pool[index])
                edge_rows.append({
                    "attack": attack,
                    "damage_fraction": fraction,
                    "pool_index": int(index),
                    "source": source,
                    "target": target,
                    "degree_source": int(degrees[source]),
                    "degree_target": int(degrees[target]),
                    "same_class": None if pool_same is None else bool(pool_same[index]),
                    "n_e": int(n_e[level, index]),
                    "r_e": int(r_e[level, index]),
                    "p_e": float(p_e[level, index]),
                    "p_no_diffusion": float(p_no_diffusion[level, index]),
                })
            for index, (source, target) in enumerate(split.control_pairs):
                control_rows.append({
                    "attack": attack,
                    "damage_fraction": fraction,
                    "control_index": index,
                    "source": int(source),
                    "target": int(target),
                    "same_class": None if control_same is None else bool(control_same[index]),
                    "p_false_positive": float(p_control[level, index]),
                    "p_false_positive_no_diffusion": float(p_control_no_diffusion[level, index]),
                })
        summary[attack] = attack_summary

    comparisons = {}
    attacks = list(results)
    for i, first in enumerate(attacks):
        for second in attacks[i + 1:]:
            _, _, p_first = per_edge_probability(results[first]["removed"], results[first]["recovered"], K)
            _, _, p_second = per_edge_probability(results[second]["removed"], results[second]["recovered"], K)
            comparisons[f"{first}_vs_{second}"] = [
                {"damage_fraction": fraction, **(distance(p_first[l], p_second[l]) or {})}
                for l, fraction in enumerate(fractions)
            ]
    return {"attacks": summary, "attack_comparisons": comparisons}, edge_rows, control_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(run_dir: Path, results, models: TrainedModels, config: ExperimentConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = run_dir / "figures"
    figures.mkdir()
    history = models.history
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(history["autoencoder_train_loss"])
    axes[0].set_title("encoder train loss")
    axes[1].plot(history["autoencoder_val_auc"])
    axes[1].axvline(history["autoencoder_best_epoch"], color="grey", linestyle="--")
    axes[1].set_title("encoder validation AUC")
    axes[2].plot(history["denoiser_loss"])
    axes[2].set_title("denoiser loss")
    for axis in axes:
        axis.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(figures / "training_curves.png", dpi=120)
    plt.close(fig)

    K = config.recovery.reconstructions
    fractions = config.perturbation.damage_fractions
    fig, axes = plt.subplots(
        len(results), len(fractions), figsize=(4 * len(fractions), 3 * len(results)), squeeze=False
    )
    for row, (attack, arrays) in enumerate(results.items()):
        _, _, p_e = per_edge_probability(arrays["removed"], arrays["recovered"], K)
        p_control = control_probability(arrays["control_hits"], K)
        for col, fraction in enumerate(fractions):
            axis = axes[row][col]
            values = p_e[col][np.isfinite(p_e[col])]
            bins = np.linspace(0, 1, 21)
            for data, label, alpha in (
                (values, "removed edges p_e", 0.7),
                (p_control[col], "control non-edges", 0.5),
            ):
                axis.hist(data, bins=bins, weights=np.full(len(data), 1.0 / max(len(data), 1)),
                          alpha=alpha, label=label)
            axis.set_yscale("log")
            axis.set_title(f"{attack}, damage {fraction:.0%}")
            axis.set_xlabel("recovery frequency")
            axis.set_ylabel("fraction of pairs")
        axes[row][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "p_e_distributions.png", dpi=120)
    plt.close(fig)


def run_experiment(
    config: ExperimentConfig,
    config_text: str | None = None,
    output_dir: Path | None = None,
    graph: GraphData | None = None,
    log_every: int = 0,
    make_plots: bool = True,
) -> Path:
    started = datetime.now(timezone.utc)
    device = resolve_device(config.run.device)
    if output_dir is None:
        root = Path(config.run.output_root)
        if not root.is_absolute():
            root = ROOT / root
        output_dir = root / f"{config.run.name}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
    output_dir.mkdir(parents=True)

    _write_json(output_dir / "config.json", config.to_dict())
    if config_text is not None:
        (output_dir / "config.toml").write_text(config_text, encoding="utf-8")
    environment = environment_record(device)
    environment["started_utc"] = started.isoformat()
    _write_json(output_dir / "environment.json", environment)

    if graph is None:
        data_root = Path(config.data.root)
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        graph = load_graph(config.data.dataset, data_root, config.data.download)
    split = split_edges(
        graph.edges,
        graph.node_count,
        config.split.train_fraction,
        config.split.val_fraction,
        config.split.pool_fraction,
        config.split.control_pairs,
        config.split.seed,
    )
    np.savez_compressed(
        output_dir / "split.npz",
        train=split.train, val=split.val, pool=split.pool,
        val_negatives=split.val_negatives, control_pairs=split.control_pairs,
    )
    series = make_damage_series(
        split.pool,
        split.all_edges,
        graph.node_count,
        config.perturbation.damage_fractions,
        config.perturbation.attacks,
        config.perturbation.series,
        config.perturbation.seed,
    )
    np.savez_compressed(
        output_dir / "series.npz",
        **{attack: np.stack([item.order for item in items]) for attack, items in series.items()},
        removal_counts=np.asarray(next(iter(series.values()))[0].removal_counts),
    )

    models = train_models(graph, split, config, device, log_every=log_every)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir()
    torch.save(models.encoder.state_dict(), checkpoints / "encoder.pt")
    torch.save(models.denoiser.state_dict(), checkpoints / "denoiser.pt")
    torch.save(
        {"labels": models.context.labels.cpu(), "centers": models.context.centers.cpu()},
        checkpoints / "community_context.pt",
    )
    _write_json(output_dir / "training_history.json", models.history)

    results = run_recoveries(graph, split, series, models, config, device, log=bool(log_every))
    np.savez_compressed(
        output_dir / "recovery_counts.npz",
        **{f"{attack}__{name}": array for attack, arrays in results.items() for name, array in arrays.items()},
    )
    summary, edge_rows, control_rows = summarize(graph, split, results, config)
    _write_csv(output_dir / "edge_recovery.csv", edge_rows)
    _write_csv(output_dir / "control_recovery.csv", control_rows)
    finished = datetime.now(timezone.utc)
    summary = {
        "dataset": graph.name,
        "graph_content_sha256": graph.content_sha256(),
        "node_count": graph.node_count,
        "edge_count": int(len(graph.edges)),
        "split_sizes": {
            "train": int(len(split.train)), "val": int(len(split.val)), "pool": int(len(split.pool)),
            "val_negatives": int(len(split.val_negatives)), "control_pairs": int(len(split.control_pairs)),
        },
        "training": {
            key: value for key, value in models.history.items()
            if key not in {"autoencoder_train_loss", "autoencoder_val_auc", "denoiser_loss"}
        },
        "notes": [
            "p_e = r_e / (n_e * K) over independent cumulative series (Decision 1, 6).",
            "Levels are nested within a series: between-level distances are descriptive only.",
            "Recovery noise seeds are shared across attacks and levels (common random numbers).",
        ],
        **summary,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    _write_json(output_dir / "summary.json", summary)
    if make_plots:
        _plot(output_dir, results, models, config)
    return output_dir
