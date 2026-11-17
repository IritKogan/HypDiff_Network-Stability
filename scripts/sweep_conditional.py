"""Stage 2 selection sweep for the conditional geometric diffusion (Decision 9).

Validation-recovery protocol only (input = train edges; validation edges vs validation
negatives; pool and control pairs never used). For every denoiser variant (a --set grid
over [denoiser] keys, on top of training = "conditional") the denoiser is trained once
and evaluated with the stochastic sampler at every (t*, eta). Reported per point:

  quality         : tau at target FPR, hit rate, AUC of the per-pair mean probability
  diversity       : share of pairs with recovery frequency p_e strictly in (0, 1),
                    mean per-pair probability std across the K samples
  informativeness : stratified AUC of p_e within deciles of the model's own mean
                    probability; CV log-likelihood gain of p_e over the own mean and
                    of the mean over the deterministic clamp_only score
  calibration     : ECE of p_e and of the mean probability, balanced and re-weighted
                    to the real base rate (missing edges vs all non-edges)

eta = 0 is the internal control (same network, no sampling randomness).
Every setting is an argument and is written to the output JSON with the resolved config.

Example:
    python scripts/sweep_conditional.py --config configs/cora.toml --out-dir artifacts/sweeps/conditional \
        --encoder-checkpoint artifacts/diagnostics/sample_diversity/encoder_context.pt \
        --set denoiser.architecture=conditional_gcn,conditional_residual_gcn \
        --set denoiser.angular_noise=upstream,folded
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

BASE_DENOISER = {"training": "conditional", "architecture": "conditional_residual_gcn",
                 "train_t_max": 200}


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


def evaluate_point(denoiser, encoder, embeddings, adjacency, labels, centers, pairs, y, weights,
                   clamp, config, t_star, eta, steps, reconstructions):
    import numpy as np
    import torch

    from stability.edge_recovery_experiment import recovery_seed
    from stability.sample_diversity import (calibration_table, incremental_log_likelihood,
                                            recovery_frequency, stratified_auc)
    from stability.validation_recovery import VALIDATION_SERIES_INDEX, _auc, calibrate_tau

    denoiser.sampler, denoiser.eta, denoiser.sampler_steps = "stochastic", eta, steps
    denoiser.check_stochastic_range(t_star)
    probs = []
    with torch.no_grad():
        for k in range(reconstructions):
            g = torch.Generator().manual_seed(recovery_seed(config.recovery.seed, VALIDATION_SERIES_INDEX, k))
            rec = denoiser.sample_from_tstar(embeddings, adjacency, labels, centers, t_star, g)
            probs.append(encoder.decode(rec, pairs).numpy().astype(np.float64))
    probs = np.stack(probs)
    tau = calibrate_tau(probs[:, y == 0].reshape(-1), config.recovery.target_fpr)
    p_e = recovery_frequency(probs, tau)
    mean_p = probs.mean(axis=0)
    logit = np.log(np.clip(mean_p, 1e-6, 1 - 1e-6) / np.clip(1 - mean_p, 1e-6, 1))
    return {
        "t_star": t_star, "eta": eta, "steps": steps, "K": reconstructions, "tau": tau,
        "hit_rate": float((probs[:, y == 1] >= tau).mean()),
        "false_positive_rate": float((probs[:, y == 0] >= tau).mean()),
        "auc_mean_probability": _auc(mean_p[y == 1], mean_p[y == 0]),
        "auc_p_e": _auc(p_e[y == 1], p_e[y == 0]),
        "share_intermediate_edges": float(np.mean((p_e[y == 1] > 0) & (p_e[y == 1] < 1))),
        "share_intermediate_all": float(np.mean((p_e > 0) & (p_e < 1))),
        "probability_std": float(probs.std(axis=0).mean()),
        "stratified_auc_p_e_given_own_mean": stratified_auc(p_e, y, mean_p),
        "ll_gain_p_e_over_own_mean": incremental_log_likelihood(mean_p, p_e, y)["gain_per_pair"],
        "ll_gain_mean_over_clamp": incremental_log_likelihood(clamp, logit, y)["gain_per_pair"],
        "ece_p_e_balanced": calibration_table(p_e, y)["ece"],
        "ece_p_e_base_rate": calibration_table(p_e, y, weights)["ece"],
        "ece_mean_balanced": calibration_table(mean_p, y)["ece"],
        "ece_mean_base_rate": calibration_table(mean_p, y, weights)["ece"],
        "reliability_p_e_balanced": calibration_table(p_e, y)["bins"],
    }


def _variant(job):
    import numpy as np
    import torch

    from stability.hypdiff_models import StabilityLPModel
    from stability.model_training import build_model_args, normalized_adjacency, train_denoiser
    from stability.partial_recovery import CommunityContext

    raw, name, overrides, saved, opts = job
    raw = json.loads(json.dumps(raw))
    raw["denoiser"].update(BASE_DENOISER)
    for key, value in overrides.items():
        section, field = key.split(".")
        raw[section][field] = value
    # sampler settings used for the pre-training range check; evaluation sets its own
    raw["recovery"].update(sampler="stochastic", eta=0.0, t_star=min(opts["t_stars"]), steps=opts["steps"])
    config, graph, split = _setup(raw)
    device = torch.device("cpu")
    args, _ = build_model_args(graph, config, device)
    encoder = StabilityLPModel(args)
    encoder.load_state_dict(saved["encoder"])
    encoder.eval()
    context = CommunityContext(tuple(range(graph.node_count)), saved["labels"], saved["centers"],
                               config.model.curvature)
    adjacency = normalized_adjacency(graph.node_count, split.train, device)
    started = time.time()
    denoiser, history = train_denoiser(graph, split, config, device, encoder, args, adjacency, context)
    train_seconds = time.time() - started
    torch.save(denoiser.state_dict(), Path(opts["out_dir"]) / f"denoiser_{name}.pt")
    print(f"[{name}] trained in {train_seconds:.0f}s, final loss {history['denoiser_loss'][-1]:.4f}",
          flush=True)

    pairs = torch.as_tensor(np.r_[split.val, split.val_negatives])
    y = np.r_[np.ones(len(split.val)), np.zeros(len(split.val_negatives))].astype(int)
    n = graph.node_count
    missing_edges = len(split.val) + len(split.pool)             # edges absent from the train graph
    non_edges = n * (n - 1) // 2 - len(graph.edges)
    weights = np.where(y == 1, missing_edges / len(split.val), non_edges / len(split.val_negatives))
    with torch.no_grad():
        embeddings = encoder.encode(graph.features, adjacency)[2]
        clamp = encoder.decode(denoiser.shrink_to_prototype(embeddings, saved["labels"], saved["centers"]),
                               pairs).numpy().astype(np.float64)
    rows = []
    for t_star, eta in itertools.product(opts["t_stars"], opts["etas"]):
        try:
            row = evaluate_point(denoiser, encoder, embeddings, adjacency, saved["labels"], saved["centers"],
                                 pairs, y, weights, clamp, config, t_star, eta, opts["steps"],
                                 opts["reconstructions"])
        except ValueError as error:
            row = {"t_star": t_star, "eta": eta, "skipped": str(error)}
        rows.append(row)
        if "skipped" not in row:
            strat = row["stratified_auc_p_e_given_own_mean"]
            print(f"  [{name}] t*={t_star} eta={eta}: hit {row['hit_rate']:.3f} auc {row['auc_mean_probability']:.3f} "
                  f"inter {row['share_intermediate_edges']:.2f} strat "
                  f"{'n/a' if strat is None else format(strat, '.2f')} ece_pe {row['ece_p_e_balanced']:.3f}",
                  flush=True)
    return {"variant": name, "overrides": overrides, "resolved_denoiser": raw["denoiser"],
            "train_seconds": round(train_seconds, 1), "final_loss": history["denoiser_loss"][-1],
            "final_mse": history.get("denoiser_mse", [None])[-1], "rows": rows}


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
    import torch

    from stability.edge_recovery_experiment import environment_record
    from stability.model_training import build_context, train_encoder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, default=None,
                        help="reuse encoder + community context (must come from the same config)")
    parser.add_argument("--set", action="append", default=[], help="denoiser.key=v1,v2,...")
    parser.add_argument("--t-stars", type=int, nargs="+", default=[30, 60, 100, 150])
    parser.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--reconstructions", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    grid = {}
    for item in args.set:
        key, values = item.split("=", 1)
        if key.split(".")[0] != "denoiser":
            parser.error("--set supports [denoiser] keys only")
        grid[key] = [_parse_value(v) for v in values.split(",")]
    points = [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())] if grid else [{}]

    config, graph, split = _setup(raw)
    device = torch.device("cpu")
    if args.encoder_checkpoint:
        saved = torch.load(args.encoder_checkpoint, weights_only=False)
        encoder_auc = None
    else:
        encoder, _, adjacency, history, _ = train_encoder(graph, split, config, device)
        context = build_context(graph, config, encoder, adjacency, device)
        saved = {"encoder": encoder.state_dict(), "labels": context.labels, "centers": context.centers}
        torch.save(saved, args.out_dir / "encoder_context.pt")
        encoder_auc = history["autoencoder_best_val_auc"]
    opts = {"out_dir": str(args.out_dir), "t_stars": args.t_stars, "etas": args.etas,
            "steps": args.steps, "reconstructions": args.reconstructions}
    jobs = [(raw, ",".join(f"{k.split('.')[1]}={v}" for k, v in p.items()) or "base", p, saved, opts)
            for p in points]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        variants = list(pool.map(_variant, jobs))
    (args.out_dir / "sweep_conditional.json").write_text(json.dumps({
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "base_denoiser": BASE_DENOISER,
        "resolved_config": raw,
        "environment": environment_record(device),
        "encoder_best_val_auc": encoder_auc,
        "validation_pairs": {"positive": len(split.val), "negative": len(split.val_negatives)},
        "variants": variants,
    }, indent=2))
    print(f"wrote {args.out_dir / 'sweep_conditional.json'}")


if __name__ == "__main__":
    main()
