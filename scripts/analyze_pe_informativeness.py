"""Stage 1 diagnostic, part 2: is the spread of the K reconstructions informative?

Uses the checkpoints written by scripts/diagnose_sample_diversity.py (same config,
validation protocol only: train edges as input, validation edges vs validation
negatives). For each (variant, sampler, t*):

  * AUC of p_e (recovery frequency at the 5%-FP tau), of the mean probability, and
    of the deterministic clamp_only score;
  * stratified AUC of p_e inside deciles of (a) the clamp_only score and (b) the
    model's own mean probability. ~0.5 in (b) means p_e is only a monotone transform
    of a point score: the *distribution* adds nothing beyond its mean;
  * cross-validated log-likelihood gain of adding p_e to a logistic model on the
    clamp_only logit / on the mean-probability logit;
  * reliability table of p_e (share of true edges per p_e value).

Example:
    python scripts/analyze_pe_informativeness.py --config configs/cora.toml \
        --checkpoints artifacts/diagnostics/sample_diversity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_sample_diversity import VARIANTS, _setup  # noqa: E402

DEFAULT_GRID = [
    "current_residual_edgerec:one_step:7", "current_residual_edgerec:one_step:15",
    "current_residual_edgerec:one_step:30", "current_residual_edgerec:one_step:50",
    "current_residual_edgerec:iterative:7", "current_residual_edgerec:iterative:30",
    "upstream_generative:one_step:50", "upstream_generative:one_step:150",
    "upstream_generative:iterative:100", "upstream_generative:iterative:150",
]


def main() -> None:
    import numpy as np
    import torch

    from stability.edge_recovery_experiment import recovery_seed
    from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
    from stability.model_training import build_model_args, normalized_adjacency
    from stability.sample_diversity import incremental_log_likelihood, recovery_frequency, stratified_auc
    from stability.validation_recovery import VALIDATION_SERIES_INDEX, _auc, calibrate_tau

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--grid", nargs="+", default=DEFAULT_GRID, help="variant:sampler:t_star")
    parser.add_argument("--reconstructions", type=int, default=None, help="default: config K")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    config, graph, split = _setup(raw)
    device = torch.device("cpu")
    K = args.reconstructions or config.recovery.reconstructions
    model_args, _ = build_model_args(graph, config, device)
    saved = torch.load(args.checkpoints / "encoder_context.pt", weights_only=False)
    encoder = StabilityLPModel(model_args)
    encoder.load_state_dict(saved["encoder"])
    encoder.eval()
    labels, centers = saved["labels"], saved["centers"]
    adjacency = normalized_adjacency(graph.node_count, split.train, device)
    pairs = torch.as_tensor(np.r_[split.val, split.val_negatives])
    y = np.r_[np.ones(len(split.val)), np.zeros(len(split.val_negatives))].astype(int)

    denoisers = {}
    rows = []
    with torch.no_grad():
        embeddings = encoder.encode(graph.features, adjacency)[2]
        for item in args.grid:
            name, sampler, t_star = item.split(":")
            t_star = int(t_star)
            if name not in denoisers:
                raw_v = json.loads(json.dumps(raw))
                raw_v["denoiser"].update(VARIANTS[name])
                cfg_v, _, _ = _setup(raw_v)
                margs, _ = build_model_args(graph, cfg_v, device)
                den = StabilityDiffusion(margs, dim=margs.dim, curvature=cfg_v.model.curvature, device=device)
                den.load_state_dict(torch.load(args.checkpoints / f"denoiser_{name}.pt"))
                den.eval()
                denoisers[name] = den
            den = denoisers[name]
            den.sampler = sampler
            probs = []
            for k in range(K):
                g = torch.Generator().manual_seed(recovery_seed(config.recovery.seed, VALIDATION_SERIES_INDEX, k))
                rec = den.sample_from_tstar(embeddings, adjacency, labels, centers, t_star, g)
                probs.append(encoder.decode(rec, pairs).numpy().astype(np.float64))
            probs = np.stack(probs)
            neg = probs[:, y == 0]
            tau = calibrate_tau(neg.reshape(-1), config.recovery.target_fpr)
            p_e = recovery_frequency(probs, tau)
            mean_p = probs.mean(axis=0)
            if "clamp" not in locals():
                clamp = encoder.decode(den.shrink_to_prototype(embeddings, labels, centers), pairs).numpy()
                clamp = clamp.astype(np.float64)
            reliability = {f"{v:.1f}": [int((p_e == v).sum()), float(y[p_e == v].mean()) if (p_e == v).any()
                                        else None] for v in np.unique(p_e)}
            row = {
                "variant": name, "sampler": sampler, "t_star": t_star, "K": K, "tau": tau,
                "share_intermediate_all_pairs": float(np.mean((p_e > 0) & (p_e < 1))),
                "auc_p_e": _auc(p_e[y == 1], p_e[y == 0]),
                "auc_mean_probability": _auc(mean_p[y == 1], mean_p[y == 0]),
                "auc_clamp_only": _auc(clamp[y == 1], clamp[y == 0]),
                "stratified_auc_p_e_given_clamp": stratified_auc(p_e, y, clamp),
                "stratified_auc_p_e_given_own_mean": stratified_auc(p_e, y, mean_p),
                "ll_gain_p_e_over_clamp": incremental_log_likelihood(clamp, p_e, y)["gain_per_pair"],
                "ll_gain_p_e_over_own_mean": incremental_log_likelihood(mean_p, p_e, y)["gain_per_pair"],
                "ll_gain_mean_over_clamp": incremental_log_likelihood(
                    clamp, np.log(np.clip(mean_p, 1e-6, 1 - 1e-6) / np.clip(1 - mean_p, 1e-6, 1)), y
                )["gain_per_pair"],
                "reliability_p_e": reliability,
            }
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                   if k != "reliability_p_e"}, flush=True)
    out = args.out or args.checkpoints / "pe_informativeness.json"
    out.write_text(json.dumps({"arguments": {k: str(v) for k, v in vars(args).items()},
                               "validation_pairs": {"positive": int(y.sum()), "negative": int((1 - y).sum())},
                               "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
