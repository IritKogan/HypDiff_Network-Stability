"""Denoiser selection on the validation-recovery protocol only.

Trains the encoder once (config as given) and the community context once per
community.n_clusters value, then trains a denoiser for every grid point (named
--variants and/or a --set grid over [denoiser] / [community] keys) and evaluates
validation recovery (train edges as input, validation edges vs validation
negatives) at several t*. Pool edges and control pairs are never used.

Examples:
    python scripts/sweep_denoiser.py --config configs/cora.toml --out sweep.json \
        --variants upstream_generative_clip residual_damage_noclip
    python scripts/sweep_denoiser.py --config configs/cora.toml --out sweep.json \
        --set community.n_clusters=7,20,50 --set denoiser.radial_delta=0,0.4,1
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = {
    "upstream_generative_clip": {"architecture": "upstream_gcn", "training": "generative", "clip_x0": True},
    "upstream_generative_noclip": {"architecture": "upstream_gcn", "training": "generative", "clip_x0": False},
    "residual_generative_noclip": {"architecture": "residual_gcn", "training": "generative", "clip_x0": False},
    "residual_damage_noclip": {"architecture": "residual_gcn", "training": "damage_aware", "clip_x0": False},
    "residual_damage_noclip_t50": {"architecture": "residual_gcn", "training": "damage_aware",
                                   "clip_x0": False, "train_t_max": 50},
    "residual_damage_clip": {"architecture": "residual_gcn", "training": "damage_aware", "clip_x0": True},
}


def _setup(raw):
    import torch

    from stability.edge_split import split_edges
    from stability.experiment_config import config_from_dict
    from stability.graph_data import load_graph

    torch.set_num_threads(1)
    config = config_from_dict(raw)
    root = Path(config.data.root)
    graph = load_graph(config.data.dataset, root if root.is_absolute() else ROOT / root, config.data.download)
    s = config.split
    split = split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction,
                        s.pool_fraction, s.control_pairs, s.seed)
    return config, graph, split


def _variant(job):
    import torch

    from stability.experiment_config import config_from_dict
    from stability.model_training import TrainedModels, build_model_args, train_denoiser
    from stability.validation_recovery import validation_recovery
    from stability.hypdiff_models import StabilityLPModel
    from stability.partial_recovery import CommunityContext

    raw, name, overrides, encoder_state, context_tensors, t_stars = job
    raw = json.loads(json.dumps(raw))
    for key, value in overrides.items():
        section, field = key.split(".") if "." in key else ("denoiser", key)
        raw[section][field] = value
    config, graph, split = _setup(raw)
    device = torch.device("cpu")
    args, _ = build_model_args(graph, config, device)
    encoder = StabilityLPModel(args)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    context = CommunityContext(tuple(range(graph.node_count)), context_tensors["labels"],
                               context_tensors["centers"], config.model.curvature)
    train_adjacency = None
    from stability.model_training import normalized_adjacency
    train_adjacency = normalized_adjacency(graph.node_count, split.train, device)
    started = time.time()
    denoiser, history = train_denoiser(graph, split, config, device, encoder, args, train_adjacency, context)
    seconds = time.time() - started
    models = TrainedModels(encoder, denoiser, context, train_adjacency, history)
    rows = []
    for t_star in t_stars:
        raw_t = json.loads(json.dumps(raw))
        raw_t["recovery"]["t_star"] = t_star
        result = validation_recovery(graph, split, models, config_from_dict(raw_t), device)
        rows.append({
            "variant": name, **overrides, "t_star": t_star,
            "val_auc_diffusion": result["diffusion"]["auc_mean_probability"],
            "val_auc_no_diffusion": result["no_diffusion"]["auc_mean_probability"],
            "val_auc_clamp_only": result["clamp_only"]["auc_mean_probability"],
            "hit_at_target_fpr_diffusion": result["diffusion"]["hit_rate"],
            "hit_at_target_fpr_no_diffusion": result["no_diffusion"]["hit_rate"],
            "hit_at_target_fpr_clamp_only": result["clamp_only"]["hit_rate"],
            "final_denoiser_loss": history["denoiser_loss"][-1],
            "train_seconds": round(seconds, 1),
        })
    return rows


def _parse_value(text: str):
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main() -> None:
    import numpy as np
    import torch

    from stability.model_training import build_context, train_encoder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="*", default=[], choices=list(VARIANTS))
    parser.add_argument("--set", action="append", default=[], help="section.key=v1,v2,... (denoiser/community)")
    parser.add_argument("--t-stars", type=int, nargs="+", default=[0, 7, 15])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    grid = {}
    for item in args.set:
        key, values = item.split("=", 1)
        if key.split(".")[0] not in {"denoiser", "community"}:
            parser.error("--set only supports [denoiser] and [community] keys (encoder is trained once)")
        grid[key] = [_parse_value(v) for v in values.split(",")]
    points = [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())] if grid else [{}]
    named = [(name, dict(VARIANTS[name])) for name in args.variants]
    jobs_spec = named + [(",".join(f"{k}={v}" for k, v in p.items()) or "config", p) for p in points
                         if not named or p]

    config, graph, split = _setup(raw)
    device = torch.device("cpu")
    encoder, _, train_adjacency, history, _ = train_encoder(graph, split, config, device)
    print(f"encoder best val AUC {history['autoencoder_best_val_auc']:.4f}", flush=True)
    contexts = {}
    for _, overrides in jobs_spec:
        k = overrides.get("community.n_clusters", config.community.n_clusters)
        if k not in contexts:
            raw_k = json.loads(json.dumps(raw))
            raw_k["community"]["n_clusters"] = k
            from stability.experiment_config import config_from_dict
            context = build_context(graph, config_from_dict(raw_k), encoder, train_adjacency, device)
            sizes = np.bincount(context.labels.numpy(), minlength=k)
            print(f"k={k}: cluster sizes min {sizes.min()} max {sizes.max()} empty {(sizes == 0).sum()}", flush=True)
            contexts[k] = {"labels": context.labels, "centers": context.centers, "sizes": sizes.tolist()}
    jobs = []
    for name, overrides in jobs_spec:
        k = overrides.get("community.n_clusters", config.community.n_clusters)
        jobs.append((raw, name, overrides, encoder.state_dict(),
                     {"labels": contexts[k]["labels"], "centers": contexts[k]["centers"]}, args.t_stars))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for result in pool.map(_variant, jobs) for row in result]
    rows.sort(key=lambda row: -row["val_auc_diffusion"])
    args.out.write_text(json.dumps({
        "config": str(args.config),
        "encoder_best_val_auc": history["autoencoder_best_val_auc"],
        "cluster_sizes": {str(k): v["sizes"] for k, v in contexts.items()},
        "rows": rows,
    }, indent=2))
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
