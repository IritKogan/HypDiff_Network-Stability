"""Decision 14, round 1: compare encoder variants per network on the validation protocol (no denoiser).

Usage (from the repository root):
    python scripts/hypstab_round1.py --plan configs/hypstab/round1.toml --run-dir artifacts/round1 [--zip]
        [--resume] [--smoke] [--only NETWORK ...] [--device auto|cpu|cuda]

Every variant = configs/hypstab/<network>.toml + one feature mode + one encoder package + one edge-drop
setting, trained with `seeds` seeds. For each variant (validation edges vs validation negatives only):
- no_diffusion AUC (the selection criterion; no communities, so independent of k);
- clamp_only AUC for every k in round.k_values (K-means on the train-graph embeddings + clamp, no denoiser);
- BA check: quantiles of the Poincare distance and of the edge probability for validation edges and for
  validation negatives, quantiles of the embedding radius, Spearman correlation radius vs train degree.
Pool edges and control pairs are never read. No model weights are kept (small zip).

Outputs: <run_dir>/plan.toml, run_info.json, round1_summary.csv (one row per variant x seed) and
variants/<id>/{config.toml, history.json, metrics.json}. --zip writes <run_dir>.zip next to the run dir.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import tomllib
from typing import Any

import numpy as np
from scipy.stats import spearmanr
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypstab.config import HypstabConfig, config_from_dict, config_to_toml  # noqa: E402
from hypstab.data.features import with_train_edge_features  # noqa: E402
from hypstab.data.graph import GraphData, load_graph  # noqa: E402
from hypstab.data.split import EdgeSplit, split_edges  # noqa: E402
from hypstab.evaluation.validation import validation_report  # noqa: E402
from hypstab.experiment.models import member_seed  # noqa: E402
from hypstab.geometry import get_geometry  # noqa: E402
from hypstab.metrics import roc_auc  # noqa: E402
from hypstab.runinfo import resolve_device, run_info, seed_everything  # noqa: E402
from hypstab.training.autoencoder import train_autoencoder  # noqa: E402
from hypstab.training.communities import fit_communities  # noqa: E402

SMOKE = {"encoder.max_epochs": 4, "encoder.patience": 4}


# --------------------------------------------------------------------------- plan


def set_key(raw: dict, dotted: str, value: Any) -> None:
    section, key = dotted.split(".", 1)
    if section not in raw or key not in raw[section]:
        raise KeyError(f"unknown config key {dotted!r}")
    raw[section][key] = value


def load_plan(path: Path) -> dict:
    plan = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for name in ("round", "network", "package", "edge_drop"):
        if name not in plan:
            raise ValueError(f"plan: missing [{name}]")
    if plan["round"]["seeds"] < 1 or not plan["round"]["k_values"]:
        raise ValueError("plan: seeds >= 1 and a non-empty k_values are required")
    return plan


def build_variants(plan: dict, configs_dir: Path, only: list[str] | None = None, smoke: bool = False,
                   device: str | None = None) -> list[dict]:
    """All (network, features, package, edge_drop, seed) variants with their validated raw configs."""
    variants = []
    for network in plan["network"]:
        if only and network["name"] not in only:
            continue
        base = tomllib.loads((configs_dir / f"{network['name']}.toml").read_text(encoding="utf-8"))
        combos = itertools.product(network["features"], plan["package"], plan["edge_drop"],
                                   range(plan["round"]["seeds"]))
        for features, package, drop, replicate in combos:
            raw = copy.deepcopy(base)
            set_key(raw, "data.features", features)
            for key, value in {**package["overrides"], **drop["overrides"]}.items():
                set_key(raw, key, value)
            set_key(raw, "encoder.seed", member_seed(base["encoder"]["seed"], replicate))
            set_key(raw, "community.seed", member_seed(base["community"]["seed"], replicate))
            if smoke:
                for key, value in SMOKE.items():
                    set_key(raw, key, value)
            if device is not None:
                set_key(raw, "run.device", device)
            config_from_dict(raw)  # validate now, before any training
            variant_id = f"{network['name']}__{features}__{package['name']}__{drop['name']}__s{replicate}"
            variants.append({"id": variant_id, "network": network["name"], "features": features,
                             "package": package["name"], "edge_drop": drop["name"], "seed_index": replicate,
                             "raw": raw})
    return variants


# --------------------------------------------------------------------------- metrics


def quantile_dict(values: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    return {f"q{round(q * 100):02d}": float(np.quantile(values, q)) for q in quantiles}


def train_degree(split: EdgeSplit, node_count: int) -> np.ndarray:
    return np.bincount(np.asarray(split.train).ravel(), minlength=node_count)


@torch.no_grad()
def variant_metrics(graph: GraphData, split: EdgeSplit, config: HypstabConfig, device: torch.device,
                    k_values: list[int], quantiles: list[float]) -> dict[str, Any]:
    """Train the encoder (train edges only) and measure everything on validation pairs only."""
    started = time.perf_counter()
    with torch.enable_grad():
        result = train_autoencoder(graph, split, config, device)
    train_seconds = time.perf_counter() - started
    model, adjacency = result.model, result.train_adjacency
    features = graph.features.to(device)
    embeddings = model.encode(features, adjacency)
    pos = torch.as_tensor(np.asarray(split.val), device=device)
    neg = torch.as_tensor(np.asarray(split.val_negatives), device=device)

    no_diffusion = roc_auc(model.edge_logits(embeddings, pos).cpu().numpy(),
                           model.edge_logits(embeddings, neg).cpu().numpy())
    clamp: dict[str, float] = {}
    for k in k_values:
        if k > graph.node_count:
            continue
        raw = copy.deepcopy(_raw_of(config))
        raw["community"]["n_clusters"] = k
        config_k = config_from_dict(raw)
        context = fit_communities(model, features, adjacency, config_k)
        report = validation_report(model, features, adjacency, split, context, config_k)
        clamp[f"clamp_k{k}"] = report["clamp_only"]["auc"]
        if abs(report["no_diffusion"]["auc"] - no_diffusion) > 1e-9:
            raise RuntimeError("validation_report and the direct no_diffusion AUC disagree")

    geo = get_geometry(config.geometry.manifold)
    c = config.geometry.curvature
    check: dict[str, Any] = {}
    for name, pairs in (("val_edges", pos), ("val_non_edges", neg)):
        distance = geo.distance(embeddings[pairs[:, 0]], embeddings[pairs[:, 1]], c).cpu().numpy()
        probability = torch.sigmoid(model.edge_logits(embeddings, pairs)).cpu().numpy()
        check[f"distance_{name}"] = quantile_dict(distance, quantiles)
        check[f"probability_{name}"] = quantile_dict(probability, quantiles)
    radius = geo.distance(embeddings, torch.zeros_like(embeddings), c).cpu().numpy()
    check["radius"] = quantile_dict(radius, quantiles)
    rho = spearmanr(radius, train_degree(split, graph.node_count)).statistic
    check["spearman_radius_vs_train_degree"] = float(rho) if np.isfinite(rho) else None

    h = result.history
    return {
        "no_diffusion_auc": no_diffusion,
        "best_val_auc_training": h["best_val_auc"],
        "clamp_only_auc": clamp,
        "ba_check": check,
        "epochs_run": h["epochs_run"],
        "best_epoch": h["best_epoch"],
        "edge_drop_skipped_epochs": len(h["edge_drop_skipped_epochs"]),
        "train_seconds": round(train_seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "history": h,
    }


def _raw_of(config: HypstabConfig) -> dict:
    return tomllib.loads(config_to_toml(config))


def summary_row(variant: dict, metrics: dict, quantiles: list[float]) -> dict[str, Any]:
    row = {key: variant[key] for key in ("id", "network", "features", "package", "edge_drop", "seed_index")}
    row["no_diffusion_auc"] = metrics["no_diffusion_auc"]
    row.update(metrics["clamp_only_auc"])
    check = metrics["ba_check"]
    for group in ("distance_val_edges", "distance_val_non_edges", "probability_val_edges",
                  "probability_val_non_edges", "radius"):
        for q in quantiles:
            name = f"q{round(q * 100):02d}"
            row[f"{group}_{name}"] = check[group][name]
    row["spearman_radius_vs_train_degree"] = check["spearman_radius_vs_train_degree"]
    for key in ("epochs_run", "best_epoch", "edge_drop_skipped_epochs", "train_seconds", "total_seconds"):
        row[key] = metrics[key]
    return row


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default=str(REPO_ROOT / "configs" / "hypstab" / "round1.toml"))
    parser.add_argument("--configs-dir", default=str(REPO_ROOT / "configs" / "hypstab"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="continue a run dir; finished variants are kept")
    parser.add_argument("--smoke", action="store_true", help="4 encoder epochs per variant (pipeline check)")
    parser.add_argument("--only", nargs="*", default=None, help="network names to run (default: all)")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    plan_text = Path(args.plan).read_text(encoding="utf-8")
    if (run_dir / "plan.toml").exists():
        if not args.resume:
            raise SystemExit(f"{run_dir} already contains a run; use --resume to continue it")
        if (run_dir / "plan.toml").read_text(encoding="utf-8") != plan_text:
            raise SystemExit("--resume: the plan differs from the saved plan.toml")
    plan = load_plan(Path(args.plan))
    variants = build_variants(plan, Path(args.configs_dir), args.only, args.smoke, args.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.toml").write_text(plan_text, encoding="utf-8")
    device = resolve_device(variants[0]["raw"]["run"]["device"])
    info = run_info(device) | {"smoke": args.smoke, "only": args.only, "variants": len(variants)}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"run dir: {run_dir} | device: {device} | variants: {len(variants)} | smoke: {args.smoke}", flush=True)

    k_values = plan["round"]["k_values"]
    quantiles = plan["round"]["quantiles"]
    graphs: dict[tuple[str, str], tuple[GraphData, EdgeSplit]] = {}
    rows = []
    started = time.perf_counter()
    for index, variant in enumerate(variants, 1):
        out = run_dir / "variants" / variant["id"]
        metrics_path = out / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(summary_row(variant, metrics, quantiles))
            print(f"[{index}/{len(variants)}] {variant['id']}: loaded", flush=True)
            continue
        config = config_from_dict(variant["raw"])
        key = (variant["network"], variant["features"])
        if key not in graphs:
            d, s = config.data, config.split
            graph = load_graph(d.dataset, d.root, d.download, config.synthetic, d.features)
            split = split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction,
                                s.pool_fraction, s.control_pairs, s.seed)
            graphs[key] = (with_train_edge_features(graph, split.train, d.features), split)
        graph, split = graphs[key]
        seed_everything(config.run.seed)
        metrics = variant_metrics(graph, split, config, device, k_values, quantiles)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.toml").write_text(config_to_toml(config), encoding="utf-8")
        (out / "history.json").write_text(json.dumps(metrics.pop("history")), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        rows.append(summary_row(variant, metrics, quantiles))
        elapsed = time.perf_counter() - started
        print(f"[{index}/{len(variants)}] {variant['id']}: no_diffusion {metrics['no_diffusion_auc']:.4f} "
              f"({metrics['total_seconds']:.0f} s; elapsed {elapsed / 60:.1f} min)", flush=True)

    with (run_dir / "round1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {run_dir / 'round1_summary.csv'}", flush=True)
    if args.zip:
        archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir.parent, base_dir=run_dir.name)
        print(f"zip: {archive}", flush=True)
    return rows


if __name__ == "__main__":
    main()
