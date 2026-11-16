"""Ablation of the partial-noise recovery step on a finished edge-recovery run.

Loads the trained models, split and series of a run directory and, for one
damaged graph, measures each component of StabilityDiffusion.sample_from_tstar
separately:

  no_diffusion          decode the damaged embeddings directly (baseline)
  roundtrip             logmap -> normalize -> renormal -> expmap (no network, no clamp)
  clamp_only            as roundtrip but with x0.clamp(-1, 1) (no network)
  network_t{t}          full recovery from t* = t (current pipeline: clamp on)
  network_t{t}_noclamp  same without the clamp on predicted x0
  ..._upstream          noise_formula = "upstream"

Metrics: hit rate on removed edges and false-positive rate on control pairs at
tau, plus threshold-free AUC (removed edges vs control pairs) and the tau that
would give the same control FP rate as the no-diffusion baseline.
Nothing in the pipeline is changed; results go to <run>/diagnostics/.

Example:
    python scripts/diagnose_diffusion_recovery.py --run artifacts/runs/cora_edge_recovery_...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stability.edge_recovery_experiment import recovery_seed  # noqa: E402
from stability.experiment_config import config_from_dict  # noqa: E402
from stability.graph_data import load_graph  # noqa: E402
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel  # noqa: E402
from stability.model_training import build_model_args, normalized_adjacency  # noqa: E402


def load_run(run_dir: Path):
    config = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    root = Path(config.data.root)
    graph = load_graph(config.data.dataset, root if root.is_absolute() else ROOT / root, False)
    split = np.load(run_dir / "split.npz")
    series = np.load(run_dir / "series.npz")
    device = torch.device("cpu")
    args, _ = build_model_args(graph, config, device)
    encoder = StabilityLPModel(args)
    encoder.load_state_dict(torch.load(run_dir / "checkpoints" / "encoder.pt", map_location=device))
    denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.model.curvature, device=device)
    denoiser.load_state_dict(torch.load(run_dir / "checkpoints" / "denoiser.pt", map_location=device))
    context = torch.load(run_dir / "checkpoints" / "community_context.pt")
    encoder.eval(), denoiser.eval()
    return config, graph, split, series, encoder, denoiser, context


def recover(denoiser, embeddings, adjacency, labels, centers, t_star, generator, clamp, use_network):
    """Mirror of StabilityDiffusion.sample_from_tstar with switchable parts."""
    prototypes = centers[labels]
    tangent = denoiser.manifold.logmap(prototypes, embeddings, c=denoiser.c)
    normalized = denoiser.normalization(tangent)
    if not use_network:
        x0 = normalized.clamp(-1.0, 1.0) if clamp else normalized
    else:
        direction = torch.stack([denoiser.manifold.logmap0(p, c=denoiser.c) for p in centers])
        edge_index = denoiser.edge(adjacency.coalesce())
        noise = denoiser.draw_noise(normalized, generator)
        batch = lambda t: torch.full((normalized.shape[0],), t, dtype=torch.long)
        x_t = denoiser.q_sample(normalized, batch(t_star), direction, noise, labels, None, restrict=True)
        for t in range(t_star, -1, -1):
            x0 = denoiser.gcn(x_t, edge_index, batch(t).float() / denoiser.num_timesteps)
            if clamp:
                x0 = x0.clamp(-1.0, 1.0)
            if t == 0:
                break
            eps = denoiser.predict_noise_from_start(x_t, batch(t), x0)
            x_t = denoiser.q_sample(x0, batch(t - 1), direction, eps, labels, None, restrict=False)
    recovered_tangent = denoiser.renormal(x0, tangent)
    return denoiser.manifold.expmap(recovered_tangent, prototypes, c=denoiser.c), normalized, x0


def metrics(encoder, embeddings_list, removed, control, tau, baseline_fp=None):
    removed_p = torch.stack([encoder.decode(e, removed) for e in embeddings_list]).double()
    control_p = torch.stack([encoder.decode(e, control) for e in embeddings_list]).double()
    labels = np.r_[np.ones(removed_p.shape[1]), np.zeros(control_p.shape[1])]
    aucs = [roc_auc_score(labels, np.r_[r.numpy(), c.numpy()]) for r, c in zip(removed_p, control_p)]
    result = {
        "hit_rate": float((removed_p >= tau).double().mean()),
        "false_positive_rate": float((control_p >= tau).double().mean()),
        "auc_removed_vs_control": float(np.mean(aucs)),
        "mean_prob_removed": float(removed_p.mean()),
        "mean_prob_control": float(control_p.mean()),
    }
    if baseline_fp is not None:
        # tau giving the same control FP rate as the baseline -> hit rate at matched FP
        matched_tau = float(torch.quantile(control_p.flatten(), 1 - baseline_fp))
        result["matched_fp_tau"] = matched_tau
        result["hit_rate_at_matched_fp"] = float((removed_p >= matched_tau).double().mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attack", default="random")
    parser.add_argument("--series", type=int, default=0)
    parser.add_argument("--level", type=int, default=-1, help="damage level index (default: last)")
    parser.add_argument("--reconstructions", type=int, default=3)
    parser.add_argument("--t-stars", type=int, nargs="+", default=[0, 1, 3, 7, 15, 30])
    args = parser.parse_args()

    config, graph, split, series, encoder, denoiser, context = load_run(args.run)
    tau = config.recovery.tau
    pool, train, val = split["pool"], split["train"], split["val"]
    counts = series["removal_counts"]
    level = args.level % len(counts)
    order = series[args.attack][args.series]
    removed_index = order[: counts[level]]
    keep = np.ones(len(pool), dtype=bool)
    keep[removed_index] = False
    damaged = np.concatenate([train, val, pool[keep]])
    adjacency = normalized_adjacency(graph.node_count, damaged, "cpu")
    removed = torch.as_tensor(pool[removed_index])
    control = torch.as_tensor(split["control_pairs"])
    labels, centers = context["labels"], context["centers"]

    out = {"run": str(args.run), "attack": args.attack, "series": args.series,
           "damage_fraction": config.perturbation.damage_fractions[level], "tau": tau, "variants": {}}
    with torch.no_grad():
        damaged_embeddings = encoder.encode(graph.features, adjacency)[2]
        base = metrics(encoder, [damaged_embeddings], removed, control, tau)
        out["variants"]["no_diffusion"] = base
        fp = base["false_positive_rate"]

        def run(name, **kwargs):
            embeddings, stats = [], {}
            for k in range(args.reconstructions):
                generator = torch.Generator().manual_seed(recovery_seed(config.recovery.seed, args.series, k))
                rec, normalized, x0 = recover(denoiser, damaged_embeddings, adjacency, labels, centers,
                                              generator=generator, **kwargs)
                embeddings.append(rec)
                stats = {
                    "fraction_normalized_outside_unit": float((normalized.abs() > 1).double().mean()),
                    "x0_std": float(x0.std()), "normalized_std": float(normalized.std()),
                    "embedding_shift_mean": float((rec - damaged_embeddings).norm(dim=1).mean()),
                    "embedding_norm_mean": float(rec.norm(dim=1).mean()),
                }
            out["variants"][name] = {**metrics(encoder, embeddings, removed, control, tau, fp), **stats}

        run("roundtrip", t_star=0, clamp=False, use_network=False)
        run("clamp_only", t_star=0, clamp=True, use_network=False)
        for formula in ("exact", "upstream"):
            denoiser.noise_formula = formula
            suffix = "" if formula == "exact" else "_upstream"
            for t_star in args.t_stars:
                if formula == "upstream" and t_star not in (7,):
                    continue
                run(f"network_t{t_star}{suffix}", t_star=t_star, clamp=True, use_network=True)
                run(f"network_t{t_star}_noclamp{suffix}", t_star=t_star, clamp=False, use_network=True)
        denoiser.noise_formula = config.denoiser.noise_formula
        out["no_diffusion_embedding_norm_mean"] = float(damaged_embeddings.norm(dim=1).mean())

    target = args.run / "diagnostics"
    target.mkdir(exist_ok=True)
    path = target / f"diffusion_ablation_{args.attack}_s{args.series}_l{level}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'variant':28s} {'hit@tau':>8s} {'FP@tau':>8s} {'AUC':>6s} {'hit@FP_base':>11s} {'clip%':>6s} {'shift':>6s}")
    for name, v in out["variants"].items():
        print(f"{name:28s} {v['hit_rate']:8.3f} {v['false_positive_rate']:8.4f} {v['auc_removed_vs_control']:6.3f} "
              f"{v.get('hit_rate_at_matched_fp', float('nan')):11.3f} "
              f"{100 * v.get('fraction_normalized_outside_unit', float('nan')):6.1f} {v.get('embedding_shift_mean', 0):6.3f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
