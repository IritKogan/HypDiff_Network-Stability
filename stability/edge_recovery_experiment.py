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

Decision 11: recovery.reconstruction_sources = ["noise", "ensemble"] runs both reconstruction sources
paired (same split, series, noise seeds and reference draws). "noise" = one model with K noise draws
(files above); "ensemble" = K models, reconstruction k through model k (stability/ensemble_training.py;
models in checkpoints/member_XX/, ensemble_members.json, files *_ensemble.*, summary["ensemble"]).
recovery_counts*.npz also keep every reconstruction's outcome, expected share and probability.
Runs are resumable (run_experiment(..., resume=True)): finished models are loaded, not retrained.
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
from sklearn.metrics import roc_auc_score
import torch

from stability.edge_recovery_distribution import (
    control_probability,
    describe,
    distance,
    per_edge_probability,
)
from stability.edge_split import EdgeSplit, split_edges
from stability.ensemble_training import is_saved, load_models, save_models, train_ensemble
from stability.experiment_config import ExperimentConfig
from stability.graph_data import GraphData, load_graph
from stability.model_training import (
    TrainedModels,
    normalized_adjacency,
    resolve_device,
    train_models,
)
from stability.perturbation_masks import DamageSeries, make_damage_series
from stability.probability_calibration import PlattCalibrator
from stability.validation_recovery import validation_recovery

METHOD_CODES = {"diffusion": 0, "no_diffusion": 1, "clamp_only": 2}
BERNOULLI_STREAM = 10  # SeedSequence tag separating edge draws from recovery noise
PAIRWISE_STREAM = 11   # SeedSequence tag for the reference-pair draws (edge_decision "pairwise")

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


def _decode(encoder, embeddings, pairs: torch.Tensor) -> np.ndarray:
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    probabilities = encoder.decode(embeddings, pairs)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("decoder produced non-finite probabilities")
    # float64 so that comparisons with the calibrated tau are exact (see validation_recovery).
    return probabilities.cpu().numpy().astype(np.float64)


def run_recoveries(
    graph: GraphData,
    split: EdgeSplit,
    series: dict[str, list[DamageSeries]],
    models: TrainedModels,
    config: ExperimentConfig,
    device: torch.device,
    taus: dict[str, float],
    log: bool = False,
    calibrators: dict[str, PlattCalibrator] | None = None,
    keep_reconstructions: bool = False,
    member_models: list[TrainedModels] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """taus: {"diffusion", "no_diffusion", "clamp_only"} -> per-method thresholds.

    edge_decision "threshold": a pair is present in a reconstruction if p >= tau.
    edge_decision "bernoulli" (Decision 10): present with probability q = calibrated p,
    prior-shifted to the damaged graph's base rate (removed edges among its non-edges).
    Uniforms come from SeedSequence([seed, BERNOULLI_STREAM, method, series, level, k]) over
    ALL pool / control pairs, so draws are shared across attacks (common random numbers).
    Deterministic methods (no_diffusion, clamp_only) then also get K draws per damaged graph.

    edge_decision "pairwise" (Decision 10, chosen 2026-11-17): relative recoverability. In each
    reconstruction a removed edge "reappears" if its decoded probability is strictly higher than that
    of a reference pair drawn uniformly from the control pairs (true non-edges, never used in training)
    of the SAME reconstruction; a control pair is compared with another control pair (never itself),
    so control frequencies are ~0.5 by construction. Base-rate and threshold free. Reference draws:
    SeedSequence([seed, PAIRWISE_STREAM, method, series, level, k]) (common random numbers).
    The expected value per draw (no draw noise) is the share of control pairs scored strictly lower.

    keep_reconstructions: also return the outcome of every single reconstruction, used by the method
    comparison and the viewers: "recovered_by_reconstruction" (diffusion) and "<method>_by_reconstruction"
    (levels, series, draws, pool) bool (they sum over draws to the counts); "<method>_expected_by_reconstruction"
    (same shape, float16; drawn edge decisions only); "<method>_probability_by_reconstruction_sum"
    (levels, draws, pool) and "control_<method>_probability_by_reconstruction_sum" (levels, draws, control):
    decoded probabilities of reconstruction k summed over series.

    member_models (Decision 11, "ensemble" source): K trained models; reconstruction k of every damaged
    graph goes through model k (its encoder, community context and denoiser) with noise draw k, also for
    no_diffusion and clamp_only. Noise seeds and reference draws are the same as for one model, so a run
    with and without member_models is paired. None = one model for all K reconstructions (as before).
    """
    rec = config.recovery
    bernoulli = rec.edge_decision == "bernoulli"
    pairwise = rec.edge_decision == "pairwise"
    if bernoulli and not calibrators:
        raise ValueError("edge_decision 'bernoulli' needs calibrators from validation_recovery")
    if pairwise and len(split.control_pairs) < 2:
        raise ValueError("edge_decision 'pairwise' needs at least two control pairs")
    draws = rec.reconstructions if (bernoulli or pairwise) else 1
    control_count = len(split.control_pairs)
    control_range = np.arange(control_count)
    n = graph.node_count
    non_edges = n * (n - 1) // 2 - len(split.all_edges)

    def decide(method, p_removed, p_control, pool_index, s, level, k):
        if pairwise:
            rng = np.random.default_rng(np.random.SeedSequence(
                [rec.seed, PAIRWISE_STREAM, METHOD_CODES[method], s, level, k]))
            reference_pool = rng.integers(0, control_count, size=len(split.pool))[pool_index]
            reference_control = rng.integers(0, control_count - 1, size=control_count)
            reference_control = reference_control + (reference_control >= control_range)   # skip itself
            ordered = np.sort(p_control)
            share_removed = np.searchsorted(ordered, p_removed, side="left") / control_count
            share_control = np.searchsorted(ordered, p_control, side="left") / (control_count - 1)
            return (p_removed > p_control[reference_pool], p_control > p_control[reference_control],
                    share_removed, share_control)
        if not bernoulli:
            return p_removed >= taus[method], p_control >= taus[method], None, None
        prevalence = len(pool_index) / (len(pool_index) + non_edges)
        q_removed = calibrators[method].apply(p_removed, prevalence)
        q_control = calibrators[method].apply(p_control, prevalence)
        rng = np.random.default_rng(np.random.SeedSequence(
            [rec.seed, BERNOULLI_STREAM, METHOD_CODES[method], s, level, k]))
        u_pool = rng.random(len(split.pool))
        u_control = rng.random(len(split.control_pairs))
        return u_pool[pool_index] < q_removed, u_control < q_control, q_removed, q_control

    levels = len(config.perturbation.damage_fractions)
    K = rec.reconstructions
    if member_models is not None:
        if not pairwise:
            raise ValueError("ensemble reconstructions need edge_decision 'pairwise' (Decision 11)")
        if len(member_models) != K:
            raise ValueError(f"need one model per reconstruction: {len(member_models)} models, K = {K}")
    reconstruction_models = list(member_models) if member_models is not None else [models] * K
    # one model for every reconstruction (the "noise" source): deterministic methods are computed once
    single_model = all(m is reconstruction_models[0] for m in reconstruction_models)
    distinct = {id(m): m for m in reconstruction_models}
    features = graph.features.to(device)
    context = {key: (m.context.labels.to(device), m.context.centers.to(device)) for key, m in distinct.items()}
    control = torch.as_tensor(split.control_pairs, device=device)
    kept_edges = np.concatenate([split.train, split.val])
    deterministic = ("no_diffusion", "clamp_only")
    results = {}
    for attack, items in series.items():
        shape = (levels, len(items), len(split.pool))
        control_shape = (levels, len(items), len(split.control_pairs))
        removed = np.zeros(shape, dtype=bool)
        recovered = np.zeros(shape, dtype=np.int16)
        recovered_probability = np.zeros(shape, dtype=np.float32)    # sum over K
        control_hits = np.zeros(control_shape, dtype=np.int16)
        control_probability_sum = np.zeros(control_shape, dtype=np.float32)
        # deterministic methods: counts over draws, probability = mean over draws (one value for one model)
        counts = {m: np.zeros(shape, dtype=np.int16) for m in deterministic}
        probability = {m: np.zeros(shape, dtype=np.float32) for m in deterministic}
        control_counts = {m: np.zeros(control_shape, dtype=np.int16) for m in deterministic}
        control_probability = {m: np.zeros(control_shape, dtype=np.float32) for m in deterministic}
        # Decision 10: expected decision per draw summed over draws (no draw noise):
        # bernoulli -> calibrated probability; pairwise -> share of control pairs scored lower
        expected = {m: np.zeros(shape, dtype=np.float32) for m in METHOD_CODES}
        control_expected = {m: np.zeros(control_shape, dtype=np.float32) for m in METHOD_CODES}
        # keep_reconstructions: per-reconstruction outcomes of every method (k = model k for an ensemble),
        # expected shares per (series, k) and decoded probabilities per k summed over series
        method_draws = {"diffusion": K, "no_diffusion": draws, "clamp_only": draws}
        if keep_reconstructions:
            outcome_k = {m: np.zeros((levels, len(items), d, len(split.pool)), dtype=bool)
                         for m, d in method_draws.items()}
            expected_k = ({m: np.zeros((levels, len(items), d, len(split.pool)), dtype=np.float16)
                           for m, d in method_draws.items()} if (bernoulli or pairwise) else None)
            probability_k = {m: np.zeros((levels, d, len(split.pool)), dtype=np.float32)
                             for m, d in method_draws.items()}
            control_probability_k = {m: np.zeros((levels, d, len(split.control_pairs)), dtype=np.float32)
                                     for m, d in method_draws.items()}

        def record(method, level, s, k, pool_index, hit, q_r, p_removed, p_control):
            if not keep_reconstructions:
                return
            outcome_k[method][level, s, k, pool_index] = hit
            if expected_k is not None:
                expected_k[method][level, s, k, pool_index] = q_r
            probability_k[method][level, k, pool_index] += p_removed
            control_probability_k[method][level, k] += p_control

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
                scored = {}

                def scores(model):
                    """Embedding of the damaged graph + deterministic-method probabilities, once per model."""
                    key = id(model)
                    if key not in scored:
                        labels, centers = context[key]
                        embeddings = model.encoder.encode(features, adjacency)[2]
                        shrunk = model.denoiser.shrink_to_prototype(embeddings, labels, centers)
                        scored[key] = {
                            "embeddings": embeddings,
                            "no_diffusion": (_decode(model.encoder, embeddings, removed_pairs),
                                             _decode(model.encoder, embeddings, control)),
                            "clamp_only": (_decode(model.encoder, shrunk, removed_pairs),
                                           _decode(model.encoder, shrunk, control)),
                        }
                    return scored[key]

                with torch.no_grad():
                    for method in deterministic:
                        p_sum = q_sum = None
                        for k in range(draws):
                            p_removed, p_control = scores(reconstruction_models[k])[method]
                            hit, control_hit, q_r, q_c = decide(method, p_removed, p_control,
                                                                pool_index, s, level, k)
                            counts[method][level, s, pool_index] += hit
                            control_counts[method][level, s] += control_hit
                            record(method, level, s, k, pool_index, hit, q_r, p_removed, p_control)
                            if single_model:
                                if k == 0:
                                    probability[method][level, s, pool_index] = p_removed
                                    control_probability[method][level, s] = p_control
                                    if bernoulli or pairwise:
                                        expected[method][level, s, pool_index] = q_r * draws
                                        control_expected[method][level, s] = q_c * draws
                                continue
                            p_sum = (p_removed, p_control) if p_sum is None else (
                                p_sum[0] + p_removed, p_sum[1] + p_control)
                            q_sum = (q_r, q_c) if q_sum is None else (q_sum[0] + q_r, q_sum[1] + q_c)
                        if not single_model:
                            probability[method][level, s, pool_index] = p_sum[0] / draws
                            control_probability[method][level, s] = p_sum[1] / draws
                            expected[method][level, s, pool_index] = q_sum[0]
                            control_expected[method][level, s] = q_sum[1]
                    for k in range(K):
                        model = reconstruction_models[k]
                        labels, centers = context[id(model)]
                        generator = torch.Generator(device=device)
                        generator.manual_seed(recovery_seed(rec.seed, s, k))
                        recovered_embeddings = model.denoiser.sample_from_tstar(
                            damaged_embeddings=scores(model)["embeddings"],
                            damaged_adjacency=adjacency,
                            labels=labels,
                            prototypes=centers,
                            t_star=rec.t_star,
                            generator=generator,
                        )
                        p_removed = _decode(model.encoder, recovered_embeddings, removed_pairs)
                        p_control = _decode(model.encoder, recovered_embeddings, control)
                        hit, control_hit, q_r, q_c = decide("diffusion", p_removed, p_control,
                                                            pool_index, s, level, k)
                        recovered[level, s, pool_index] += hit
                        record("diffusion", level, s, k, pool_index, hit, q_r, p_removed, p_control)
                        recovered_probability[level, s, pool_index] += p_removed
                        control_hits[level, s] += control_hit
                        control_probability_sum[level, s] += p_control
                        if bernoulli or pairwise:
                            expected["diffusion"][level, s, pool_index] += q_r
                            control_expected["diffusion"][level, s] += q_c
                removed[level, s, pool_index] = True
            if log:
                print(f"[recovery{'' if single_model else ' ensemble'}] {attack} series {s + 1}/{len(items)} "
                      f"({time.time() - started:.1f}s)", flush=True)
        kept = {}
        if keep_reconstructions:
            kept["recovered_by_reconstruction"] = outcome_k["diffusion"]
            kept.update({f"{m}_by_reconstruction": outcome_k[m] for m in deterministic})
            if expected_k is not None:
                kept.update({f"{m}_expected_by_reconstruction": expected_k[m] for m in METHOD_CODES})
            kept.update({f"{m}_probability_by_reconstruction_sum": probability_k[m] for m in METHOD_CODES})
            kept.update({f"control_{m}_probability_by_reconstruction_sum": control_probability_k[m]
                         for m in METHOD_CODES})
        results[attack] = {
            "removed": removed,
            "recovered": recovered,
            "recovered_probability_sum": recovered_probability,
            "no_diffusion": counts["no_diffusion"],
            "no_diffusion_probability": probability["no_diffusion"],
            "control_hits": control_hits,
            "control_probability_sum": control_probability_sum,
            "control_no_diffusion": control_counts["no_diffusion"],
            "control_no_diffusion_probability": control_probability["no_diffusion"],
            "clamp_only": counts["clamp_only"],
            "clamp_only_probability": probability["clamp_only"],
            "control_clamp_only": control_counts["clamp_only"],
            "control_clamp_only_probability": control_probability["clamp_only"],
            "deterministic_draws": np.asarray(draws),
            **kept,
            **{f"{m}_expected_sum": expected[m] for m in METHOD_CODES},
            **{f"control_{m}_expected_sum": control_expected[m] for m in METHOD_CODES},
        }
    return results


def _auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    positive = positive[np.isfinite(positive)]
    if positive.size == 0 or negative.size == 0:
        return None
    labels = np.r_[np.ones(positive.size), np.zeros(negative.size)]
    return float(roc_auc_score(labels, np.r_[positive, negative]))


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
        draws = int(arrays.get("deterministic_draws", 1))   # K draws per damaged graph if bernoulli
        with np.errstate(invalid="ignore", divide="ignore"):
            p_no_diffusion = np.where(n_e > 0, arrays["no_diffusion"].sum(axis=1) / (n_e * draws), np.nan)
        p_control = control_probability(arrays["control_hits"], K)
        p_control_no_diffusion = control_probability(arrays["control_no_diffusion"], draws)
        drawn = config.recovery.edge_decision in ("bernoulli", "pairwise")
        with np.errstate(invalid="ignore", divide="ignore"):
            expected_p_e = (np.where(n_e > 0, arrays["diffusion_expected_sum"].sum(axis=1) / (n_e * K), np.nan)
                            if drawn else None)
        series_count = arrays["removed"].shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            score_e = np.where(n_e > 0, arrays["recovered_probability_sum"].sum(axis=1) / (n_e * K), np.nan)
            score_e_no_diffusion = np.where(
                n_e > 0, arrays["no_diffusion_probability"].sum(axis=1) / n_e, np.nan
            )
        score_control = arrays["control_probability_sum"].sum(axis=1) / (series_count * K)
        score_control_no_diffusion = arrays["control_no_diffusion_probability"].mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_clamp = np.where(n_e > 0, arrays["clamp_only"].sum(axis=1) / (n_e * draws), np.nan)
            score_e_clamp = np.where(n_e > 0, arrays["clamp_only_probability"].sum(axis=1) / n_e, np.nan)
        p_control_clamp = control_probability(arrays["control_clamp_only"], draws)
        score_control_clamp = arrays["control_clamp_only_probability"].mean(axis=1)
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
                "auc_removed_vs_control": _auc(score_e[level], score_control[level]),
                "auc_removed_vs_control_no_diffusion": _auc(
                    score_e_no_diffusion[level], score_control_no_diffusion[level]
                ),
                "auc_removed_vs_control_clamp_only": _auc(score_e_clamp[level], score_control_clamp[level]),
                "p_e_clamp_only": describe(p_clamp[level]),
                "control_false_positive_clamp_only": describe(p_control_clamp[level]),
                "p_e_vs_control": distance(p_e[level], p_control[level]),
                "p_e_vs_no_diffusion": distance(p_e[level], p_no_diffusion[level]),
            }
            if drawn:
                entry["expected_p_e"] = describe(expected_p_e[level])
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
                    "mean_probability": float(score_e[level, index]),
                    "mean_probability_no_diffusion": float(score_e_no_diffusion[level, index]),
                    "expected_p_e": None if expected_p_e is None else float(expected_p_e[level, index]),
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


def _plot(run_dir: Path, results, models: TrainedModels, config: ExperimentConfig, suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
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
    fig.savefig(figures / f"p_e_distributions{suffix}.png", dpi=120)
    plt.close(fig)


def run_experiment(
    config: ExperimentConfig,
    config_text: str | None = None,
    output_dir: Path | None = None,
    graph: GraphData | None = None,
    log_every: int = 0,
    make_plots: bool = True,
    resume: bool = False,
) -> Path:
    """Run (or, with resume=True and an existing output_dir, continue) one experiment."""
    started = datetime.now(timezone.utc)
    device = resolve_device(config.run.device)
    if output_dir is None:
        root = Path(config.run.output_root)
        if not root.is_absolute():
            root = ROOT / root
        output_dir = root / f"{config.run.name}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = Path(output_dir)
    resuming = output_dir.exists() and (output_dir / "config.json").is_file()
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir} (use resume to continue it)")
    if resuming:
        saved = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if saved != json.loads(json.dumps(config.to_dict())):
            raise ValueError(f"cannot resume {output_dir}: its config.json differs from the given config")
        if (output_dir / "summary.json").is_file():
            return output_dir   # already finished
        _write_json(output_dir / f"environment_resumed_{started.strftime('%Y%m%dT%H%M%SZ')}.json",
                    environment_record(device))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
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

    checkpoints = output_dir / "checkpoints"
    if is_saved(checkpoints):
        models = load_models(graph, split.train, config, device, checkpoints)
        if log_every:
            print("[training] model 1 loaded from checkpoints/", flush=True)
    else:
        training_started = time.time()
        models = train_models(graph, split, config, device, log_every=log_every)
        models.history["training_seconds"] = round(time.time() - training_started, 1)
        save_models(models, checkpoints, {"member": 0, "model_seed": config.model.seed,
                                          "community_seed": config.community.seed,
                                          "train_edges_used": int(len(split.train)),
                                          "training_seconds": models.history["training_seconds"]})
    _write_json(output_dir / "training_history.json", models.history)
    sources = config.recovery.reconstruction_sources
    members, member_records = None, None
    if "ensemble" in sources:
        def validate_member(member, member_models, directory):
            directory.mkdir(parents=True, exist_ok=True)
            _write_json(directory / "validation_recovery.json",
                        validation_recovery(graph, split, member_models, config, device))

        members, member_records = train_ensemble(graph, split, config, device, checkpoints, models,
                                                 log_every=log_every, after_member=validate_member)

    validation = validation_recovery(graph, split, models, config, device)
    _write_json(output_dir / "validation_recovery.json", validation)
    taus = {method: validation[method]["tau"] for method in ("diffusion", "no_diffusion", "clamp_only")}
    if log_every:
        print(f"[validation] {json.dumps(validation)}", flush=True)
    calibrators = {method: PlattCalibrator(**validation[method]["calibration"]["calibrator"])
                   for method in ("diffusion", "no_diffusion", "clamp_only")}
    source_results, source_summaries, recovery_seconds = {}, {}, {}
    for source in sources:
        suffix = "" if source == "noise" else "_ensemble"
        recovery_started = time.time()
        results = run_recoveries(graph, split, series, models, config, device, taus, log=bool(log_every),
                                 calibrators=calibrators, keep_reconstructions=True,
                                 member_models=members if source == "ensemble" else None)
        recovery_seconds[source] = round(time.time() - recovery_started, 1)
        np.savez_compressed(
            output_dir / f"recovery_counts{suffix}.npz",
            **{f"{attack}__{name}": array for attack, arrays in results.items() for name, array in arrays.items()},
        )
        source_summary, edge_rows, control_rows = summarize(graph, split, results, config)
        _write_csv(output_dir / f"edge_recovery{suffix}.csv", edge_rows)
        _write_csv(output_dir / f"control_recovery{suffix}.csv", control_rows)
        source_results[source], source_summaries[source] = results, source_summary
    ensemble_summary = None
    if members is not None:
        member_validation = []
        for record in member_records:
            directory = output_dir / record["checkpoint"]
            validation_file = directory / "validation_recovery.json"
            member_val = validation if record["checkpoint"] == checkpoints.name else json.loads(
                validation_file.read_text(encoding="utf-8"))
            member_validation.append({m: member_val[m]["auc_mean_probability"]
                                      for m in ("diffusion", "no_diffusion", "clamp_only")})
        _write_json(output_dir / "ensemble_members.json", {
            "resample_mode": config.ensemble.resample_mode, "members": member_records,
            "validation_auc_mean_probability": member_validation,
        })
        ensemble_summary = {
            **source_summaries["ensemble"],
            "resample_mode": config.ensemble.resample_mode,
            "members": len(members),
            "member_validation_auc_mean_probability": member_validation,
            "training_seconds_total": float(sum(r["training_seconds"] or 0.0 for r in member_records)),
        }
    summary = source_summaries.get("noise", {})
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
            if key not in {"autoencoder_train_loss", "autoencoder_val_auc", "denoiser_loss",
                           "denoiser_drop_fractions"}
        },
        "edge_decision": config.recovery.edge_decision,
        "thresholds": taus,
        "calibrators": {m: c.to_dict() for m, c in calibrators.items()},
        "validation_recovery": validation,
        "notes": [
            "p_e = r_e / (n_e * K) over independent cumulative series (Decision 1, 6).",
            "edge_decision 'bernoulli' (Decision 10): each reconstruction draws each pair with the calibrated, "
            "prior-shifted probability; expected_p_e = mean calibrated probability (no draw noise).",
            "edge_decision 'pairwise' (Decision 10): a pair reappears when it scores above a random control "
            "pair of the same reconstruction; control ~0.5 by construction; expected_p_e = share of control "
            "pairs scored lower.",
            "Thresholds per method from validation (tau_mode); AUC uses mean probabilities (threshold-free).",
            "Levels are nested within a series: between-level distances are descriptive only.",
            "Recovery noise seeds are shared across attacks and levels (common random numbers).",
            "Decision 11: 'ensemble' = reconstruction k through independently trained model k (summary['ensemble'],"
            " files *_ensemble); paired with 'noise' (same series, noise seeds and reference draws; model 1 shared).",
        ],
        **summary,
        "reconstruction_sources": list(sources),
        **({"ensemble": ensemble_summary} if ensemble_summary is not None else {}),
        "training_seconds_model_1": models.history.get("training_seconds"),
        "recovery_seconds": recovery_seconds,
        "resumed": resuming,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    _write_json(output_dir / "summary.json", summary)
    if make_plots:
        for source, results in source_results.items():
            _plot(output_dir, results, models, config, suffix="" if source == "noise" else "_ensemble")
    return output_dir
