"""Decision 10 model selection: quality of CALIBRATED edge probabilities (validation only).

For each method the decoded probabilities of validation edges and validation negatives
(input = train edges only; pool and control never used) are Platt-calibrated with the
real base-rate weights and scored by 2-fold cross-fitting: log-loss, Brier and ECE at the
real base rate, against the log-loss of predicting the base rate alone. Diffusion methods
are scored per sample (what a Bernoulli reconstruction uses) and on the mean over K.
Also reports the distribution of the expected p_e (mean calibrated probability) of the
validation edges at an evaluation-like prevalence (``--eval-removed`` removed edges).

Methods: no_diffusion, clamp_only, and every denoiser checkpoint given as
    name=path/to/denoiser.pt:sampler:t_star:eta[:key=value,...]   (key=value -> [denoiser] overrides)
All settings and results go to the output JSON.

Example:
    python scripts/compare_calibrated_methods.py --config configs/cora.toml \
        --encoder-checkpoint artifacts/diagnostics/sample_diversity/encoder_context.pt \
        --denoiser "edge_rec=artifacts/diagnostics/sample_diversity/denoiser_current_residual_edgerec.pt:one_step:7:0" \
        --out artifacts/sweeps/calibrated_methods.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _value(text):
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

    from scripts.sweep_conditional import _setup
    from stability.edge_recovery_experiment import environment_record, recovery_seed
    from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
    from stability.model_training import build_model_args, normalized_adjacency
    from stability.probability_calibration import base_rate_weights, cross_fitted_quality, fit_platt
    from stability.validation_recovery import VALIDATION_SERIES_INDEX, _auc, validation_base_rate

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--denoiser", action="append", default=[])
    parser.add_argument("--reconstructions", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-removed", type=int, default=1056, help="removed edges for the prevalence shift (20%% of Cora)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    config, graph, split = _setup(raw)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    model_args, _ = build_model_args(graph, config, device)
    saved = torch.load(args.encoder_checkpoint, weights_only=False)
    encoder = StabilityLPModel(model_args)
    encoder.load_state_dict(saved["encoder"])
    encoder.eval()
    labels, centers = saved["labels"], saved["centers"]
    adjacency = normalized_adjacency(graph.node_count, split.train, device)
    pairs = torch.as_tensor(np.r_[split.val, split.val_negatives])
    y = np.r_[np.ones(len(split.val)), np.zeros(len(split.val_negatives))].astype(int)
    counts = validation_base_rate(graph, split)
    w_pos, w_neg = base_rate_weights(counts["missing_edges"], counts["non_edges"], len(split.val),
                                     len(split.val_negatives))
    eval_prevalence = args.eval_removed / (args.eval_removed + counts["non_edges"])

    def score(name, probs):
        probs = np.atleast_2d(probs)
        pos, neg = probs[:, y == 1], probs[:, y == 0]
        per_sample = cross_fitted_quality(pos, neg, w_pos, w_neg, config.recovery.calibration_folds,
                                          config.recovery.seed)
        mean_quality = cross_fitted_quality(pos.mean(0), neg.mean(0), w_pos, w_neg,
                                            config.recovery.calibration_folds, config.recovery.seed)
        calibrator = fit_platt(pos, neg, w_pos, w_neg)
        expected = calibrator.apply(pos, eval_prevalence).mean(axis=0)
        row = {
            "method": name, "samples": int(probs.shape[0]),
            "auc_mean_probability": _auc(probs.mean(0)[y == 1], probs.mean(0)[y == 0]),
            "per_sample": per_sample, "mean_over_samples": mean_quality,
            "calibrator": calibrator.to_dict(),
            "expected_p_e_validation_edges_at_eval_prevalence": {
                "mean": float(expected.mean()),
                "quantiles": {q: float(np.quantile(expected, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            },
        }
        print(f"{name:28s} AUC {row['auc_mean_probability']:.3f} | log-loss per-sample {per_sample['log_loss']:.5f} "
              f"(base rate {per_sample['log_loss_base_rate_only']:.5f}) mean-K {mean_quality['log_loss']:.5f} | "
              f"ECE {per_sample['ece']:.5f} | expected p_e median "
              f"{row['expected_p_e_validation_edges_at_eval_prevalence']['quantiles'][0.5]:.4f}", flush=True)
        return row

    rows = []
    with torch.no_grad():
        embeddings = encoder.encode(graph.features, adjacency)[2]
        rows.append(score("no_diffusion", encoder.decode(embeddings, pairs).numpy().astype(np.float64)))
        reference = StabilityDiffusion(model_args, dim=model_args.dim, device=device)
        rows.append(score("clamp_only", encoder.decode(
            reference.shrink_to_prototype(embeddings, labels, centers), pairs).numpy().astype(np.float64)))
        for spec in args.denoiser:
            name, rest = spec.split("=", 1)
            parts = rest.split(":")
            path, sampler, t_star, eta = parts[0], parts[1], int(parts[2]), float(parts[3])
            overrides = dict(kv.split("=") for kv in parts[4].split(",")) if len(parts) > 4 else {}
            raw_v = json.loads(json.dumps(raw))
            raw_v["denoiser"].update({k: _value(v) for k, v in overrides.items()})
            raw_v["recovery"].update(sampler=sampler, eta=eta, steps=args.steps)
            cfg_v, _, _ = _setup(raw_v)
            margs, _ = build_model_args(graph, cfg_v, device)
            denoiser = StabilityDiffusion(margs, dim=margs.dim, device=device)
            denoiser.load_state_dict(torch.load(path))
            denoiser.eval()
            probs = []
            for k in range(args.reconstructions):
                g = torch.Generator().manual_seed(recovery_seed(config.recovery.seed, VALIDATION_SERIES_INDEX, k))
                recovered = denoiser.sample_from_tstar(embeddings, adjacency, labels, centers, t_star, g)
                probs.append(encoder.decode(recovered, pairs).numpy().astype(np.float64))
            rows.append(score(name, np.stack(probs)) | {"spec": spec})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "base_rate": counts, "eval_prevalence": eval_prevalence,
        "environment": environment_record(device), "rows": rows,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
