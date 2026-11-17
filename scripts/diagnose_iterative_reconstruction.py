"""Does structural (edge-level) sampling give the diffusion an informative role? (validation only)

Input graph = train edges (validation protocol); unknown = validation + pool edges (R of them; the
COUNT is known, which ones is not). For every setting and reconstruction k:
  round 0: encode the input graph, denoise (config denoiser, its own noise seed), score all candidate
           pairs (all non-edges of the input graph);
  sample:  add R' = add_fraction * R pairs, Gumbel-top-R' with the given temperature (0 = top-R');
  round 1: encode input + added pairs, denoise, score validation edges / validation negatives.
Reported: AUC of the mean round-1 score; per-pair std across K; share of validation edges / negatives
added by the sampling step and precision of the added set (labels used for evaluation only);
pairwise p_e (validation edge vs a random validation negative per reconstruction) and whether it is
informative beyond the model's own mean score (stratified AUC, CV log-likelihood gain).
"one_round" = current pipeline (no re-encoding). All settings + results go to the JSON.

Example:
    python scripts/diagnose_iterative_reconstruction.py --config configs/cora.toml \
        --encoder-checkpoint artifacts/diagnostics/sample_diversity/encoder_context.pt \
        --denoiser-checkpoint artifacts/diagnostics/sample_diversity/denoiser_current_residual_edgerec.pt \
        --out artifacts/diagnostics/iterative_reconstruction.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import numpy as np
    import torch

    from scripts.sweep_conditional import _setup
    from stability.edge_recovery_experiment import environment_record, recovery_seed
    from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
    from stability.iterative_reconstruction import candidate_pairs, gumbel_top_k, score_pairs
    from stability.model_training import build_model_args, normalized_adjacency
    from stability.sample_diversity import incremental_log_likelihood, stratified_auc
    from stability.validation_recovery import VALIDATION_SERIES_INDEX, _auc

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--denoiser-checkpoint", type=Path, required=True)
    parser.add_argument("--add-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--reconstructions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261117)
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
    denoiser = StabilityDiffusion(model_args, dim=model_args.dim, device=device)
    denoiser.load_state_dict(torch.load(args.denoiser_checkpoint))
    denoiser.eval()
    labels, centers = saved["labels"], saved["centers"]
    rec = config.recovery

    candidates = candidate_pairs(graph.node_count, split.train)
    missing = len(split.val) + len(split.pool)
    val_pairs = np.r_[split.val, split.val_negatives]
    y = np.r_[np.ones(len(split.val)), np.zeros(len(split.val_negatives))].astype(int)
    key = lambda pairs: np.sort(pairs, axis=1)[:, 0] * graph.node_count + np.sort(pairs, axis=1)[:, 1]
    val_keys = key(val_pairs)
    unknown_keys = set(key(np.r_[split.val, split.pool]).tolist())
    print(f"candidates {len(candidates)}, missing edges R = {missing}", flush=True)

    def denoised(edges, k):
        adjacency = normalized_adjacency(graph.node_count, edges, device)
        with torch.no_grad():
            embeddings = encoder.encode(graph.features, adjacency)[2]
            g = torch.Generator().manual_seed(recovery_seed(rec.seed, VALIDATION_SERIES_INDEX, k))
            return denoiser.sample_from_tstar(embeddings, adjacency, labels, centers, rec.t_star, g)

    def summarize(name, scores, added=None, extra=None):
        scores = np.stack(scores)                          # (K, val pairs)
        mean = scores.mean(axis=0)
        rng = np.random.default_rng([args.seed, 99])
        negatives = scores[:, y == 0]
        reference = negatives[np.arange(len(scores))[:, None], rng.integers(0, negatives.shape[1], size=(len(scores), len(y)))]
        p_e = (scores > reference).mean(axis=0)
        row = {
            "setting": name,
            "auc_mean_score": _auc(mean[y == 1], mean[y == 0]),
            "auc_p_e": _auc(p_e[y == 1], p_e[y == 0]),
            "score_std_across_K": float(scores.std(axis=0).mean()),
            "p_e_std_validation_edges": float(p_e[y == 1].std()),
            "stratified_auc_p_e_given_own_mean": stratified_auc(p_e, y, mean),
            "ll_gain_p_e_over_own_mean": incremental_log_likelihood(mean, p_e, y)["gain_per_pair"],
        }
        if added is not None:
            added = np.stack(added)                        # (K, val pairs) bool
            row.update({
                "share_validation_edges_added": float(added[:, y == 1].mean()),
                "share_validation_negatives_added": float(added[:, y == 0].mean()),
                "validation_edges_added_in_some_but_not_all_K": float(
                    np.mean((added[:, y == 1].mean(0) > 0) & (added[:, y == 1].mean(0) < 1))),
            })
        row.update(extra or {})
        s = row["stratified_auc_p_e_given_own_mean"]
        print(f"{name:34s} AUC {row['auc_mean_score']:.3f} | std_K {row['score_std_across_K']:.4f} | "
              f"strat {'n/a' if s is None else round(s, 3)} | LL {row['ll_gain_p_e_over_own_mean']:+.4f}"
              + (f" | val edges added {row['share_validation_edges_added']:.3f} prec {row['precision_added']:.3f}"
                 if added is not None else ""), flush=True)
        return row

    rows = []
    started = time.time()
    round0 = []
    for k in range(args.reconstructions):
        recovered = denoised(split.train, k)
        round0.append((recovered, score_pairs(encoder, recovered, candidates)))
    rows.append(summarize("one_round", [score_pairs(encoder, r, val_pairs) for r, _ in round0]))
    candidate_index = {v: i for i, v in enumerate(key(candidates).tolist())}
    val_in_candidates = np.array([candidate_index[v] for v in val_keys.tolist()])

    for fraction, temperature in itertools.product(args.add_fractions, args.temperatures):
        count = int(round(fraction * missing))
        scores, added_flags, precisions = [], [], []
        for k in range(args.reconstructions):
            _, candidate_scores = round0[k]
            rng = np.random.default_rng([args.seed, k, int(fraction * 1000), int(temperature * 1000)])
            chosen = gumbel_top_k(candidate_scores, count, temperature, rng)
            added_pairs = candidates[chosen]
            precisions.append(np.mean([v in unknown_keys for v in key(added_pairs).tolist()]))
            flag = np.zeros(len(candidates), dtype=bool)
            flag[chosen] = True
            added_flags.append(flag[val_in_candidates])
            recovered = denoised(np.r_[split.train, added_pairs], k)
            scores.append(score_pairs(encoder, recovered, val_pairs))
        rows.append(summarize(f"iterate add={fraction} T={temperature}", scores, added_flags,
                              {"add_fraction": fraction, "added_pairs": count, "temperature": temperature,
                               "precision_added": float(np.mean(precisions))}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "missing_edges": missing, "candidates": int(len(candidates)),
        "environment": environment_record(device), "seconds": round(time.time() - started, 1), "rows": rows,
    }, indent=2))
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
