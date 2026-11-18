"""Idea A (2026-11-18): do reconstructions from an ensemble of models trained on resampled train
edges carry information that one model's diffusion noise does not? (validation protocol only)

Each member = full training (encoder, community context, denoiser; configs as given) on a resample
of the TRAIN edges (stability/bootstrap_ensemble.py). Input graph for scoring = all train edges;
validation edges vs validation negatives. Per member the script stores the diffusion scores of
config-K noise draws, no_diffusion and clamp_only scores (one .npz per member, so runs resume).

Compared, per resample mode:
  * single model, K noise draws (= current pipeline; member 0);
  * ensemble, reconstruction k = member k (its noise draw k).
Reported: AUC of each single member (training variance), AUC of the ensemble mean probability and
of the mean expected pairwise p_e (rank of the score among validation negatives, member by member),
pairwise p_e (Decision 10) and whether it is informative beyond the own mean (stratified AUC,
CV log-likelihood gain), whether the ensemble beats a single model (LL gain over member 0), and
whether member disagreement is informative (spread of the per-member ranks).

Example (both CPUs, members trained two at a time):
    python scripts/diagnose_bootstrap_ensemble.py --config configs/cora.toml \
        --modes seed_only subsample --members 5 --jobs 2 \
        --out artifacts/diagnostics/bootstrap_ensemble
GPU (one member at a time; scripts/run_gpu_checks.py calls it this way):
    python scripts/diagnose_bootstrap_ensemble.py --config configs/cora.toml --device cuda \
        --modes seed_only --members 10 --jobs 1 --out artifacts/diagnostics/bootstrap_ensemble_gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _member_file(out: Path, mode: str, member: int) -> Path:
    return out / f"{mode}_member{member}.npz"


def train_member(args) -> None:
    import numpy as np
    import torch

    from scripts.sweep_conditional import _setup
    from stability.bootstrap_ensemble import member_seed, member_split
    from stability.edge_recovery_experiment import recovery_seed
    from stability.model_training import normalized_adjacency, resolve_device, train_models
    from stability.validation_recovery import VALIDATION_SERIES_INDEX

    started = time.time()
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    config, graph, split = _setup(raw)
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)  # members train in parallel processes (--jobs)
    config = config.with_overrides(
        model={"seed": member_seed(config.model.seed, args.mode, args.member)},
        community={"seed": member_seed(config.community.seed, args.mode, args.member)},
    )
    msplit = member_split(split, args.mode, args.member, args.fraction, config.split.seed)
    models = train_models(graph, msplit, config, device)
    rec = config.recovery
    adjacency = normalized_adjacency(graph.node_count, split.train, device)  # input: ALL train edges
    pairs = torch.as_tensor(np.r_[split.val, split.val_negatives], device=device)
    features = graph.features.to(device)
    labels, centers = models.context.labels, models.context.centers
    with torch.no_grad():
        embeddings = models.encoder.encode(features, adjacency)[2]
        base = models.encoder.decode(embeddings, pairs).cpu().numpy()
        clamp = models.encoder.decode(
            models.denoiser.shrink_to_prototype(embeddings, labels, centers), pairs).cpu().numpy()
        diffusion = []
        for k in range(rec.reconstructions):
            generator = torch.Generator(device=device).manual_seed(recovery_seed(rec.seed, VALIDATION_SERIES_INDEX, k))
            recovered = models.denoiser.sample_from_tstar(
                damaged_embeddings=embeddings, damaged_adjacency=adjacency, labels=labels,
                prototypes=centers, t_star=rec.t_star, generator=generator)
            diffusion.append(models.encoder.decode(recovered, pairs).cpu().numpy())
    history = {k: v for k, v in models.history.items() if not isinstance(v, list)}
    np.savez_compressed(
        _member_file(args.out, args.mode, args.member),
        diffusion=np.stack(diffusion).astype(np.float64), no_diffusion=base.astype(np.float64),
        clamp_only=clamp.astype(np.float64), train_edges_used=len(msplit.train),
        model_seed=config.model.seed, community_seed=config.community.seed,
        seconds=time.time() - started, history=json.dumps(history, default=str), device=str(device))
    print(f"{args.mode} member {args.member}: {len(msplit.train)} train edges, "
          f"{time.time() - started:.0f}s", flush=True)


def analyze(args, y) -> dict:
    import numpy as np

    from stability.bootstrap_ensemble import pairwise_outcomes, rank_against
    from stability.sample_diversity import incremental_log_likelihood, stratified_auc
    from stability.validation_recovery import _auc

    def summary(name, scores, single_reference):
        """scores: (K, pairs), reconstruction k = row k."""
        rng = np.random.default_rng([args.seed, 99])
        outcomes = pairwise_outcomes(scores, y, rng)
        p_e = outcomes.mean(axis=0)
        mean_prob = scores.mean(axis=0)
        ranks = np.stack([rank_against(row, row[y == 0]) for row in scores])  # expected pairwise p_e
        mean_rank, spread = ranks.mean(axis=0), ranks.std(axis=0)
        positives = p_e[y == 1]
        row = {
            "setting": name,
            "auc_mean_probability": _auc(mean_prob[y == 1], mean_prob[y == 0]),
            "auc_mean_expected_p_e": _auc(mean_rank[y == 1], mean_rank[y == 0]),
            "mean_p_e_validation_edges": float(positives.mean()),
            "share_val_edges_p_e_strictly_between_0_1": float(np.mean((positives > 0) & (positives < 1))),
            "probability_std_across_K": float(scores.std(axis=0).mean()),
            "expected_p_e_std_across_K_val_edges": float(spread[y == 1].mean()),
            "stratified_auc_p_e_given_own_mean_probability": stratified_auc(p_e, y, mean_prob),
            "ll_gain_p_e_over_own_mean_probability": incremental_log_likelihood(mean_prob, p_e, y)["gain_per_pair"],
            "ll_gain_mean_expected_p_e_over_single_model": incremental_log_likelihood(
                single_reference, mean_rank, y)["gain_per_pair"],
            "stratified_auc_disagreement_given_mean_expected_p_e": stratified_auc(-spread, y, mean_rank),
            "ll_gain_disagreement_over_mean_expected_p_e": incremental_log_likelihood(
                np.clip(mean_rank, 1e-4, 1 - 1e-4), spread, y)["gain_per_pair"],
        }
        s = row["stratified_auc_p_e_given_own_mean_probability"]
        print(f"{name:30s} AUC(mean prob) {row['auc_mean_probability']:.3f} | AUC(mean exp p_e) "
              f"{row['auc_mean_expected_p_e']:.3f} | p_e in (0,1) {row['share_val_edges_p_e_strictly_between_0_1']:.2f} | "
              f"strat {s:.3f} | LL p_e {row['ll_gain_p_e_over_own_mean_probability']:+.4f} | "
              f"LL vs single {row['ll_gain_mean_expected_p_e_over_single_model']:+.4f} | "
              f"disagree strat {row['stratified_auc_disagreement_given_mean_expected_p_e']:.3f}", flush=True)
        return row

    result = {}
    for mode in args.modes:
        members = [np.load(_member_file(args.out, mode, m)) for m in range(args.members)]
        single = members[0]["diffusion"]                       # (K noise draws, pairs)
        single_mean = single.mean(axis=0)
        member_aucs = {
            method: [_auc(m[method].mean(axis=0)[y == 1], m[method].mean(axis=0)[y == 0])
                     if method == "diffusion" else _auc(m[method][y == 1], m[method][y == 0])
                     for m in members]
            for method in ("diffusion", "clamp_only", "no_diffusion")
        }
        print(f"\n[{mode}] member AUCs (diffusion): {np.round(member_aucs['diffusion'], 3).tolist()}")
        rows = [summary(f"{mode}: single model, K noise", single, single_mean)]
        ensemble = np.stack([m["diffusion"][k % m["diffusion"].shape[0]] for k, m in enumerate(members)])
        rows.append(summary(f"{mode}: ensemble, k = member k", ensemble, single_mean))
        result[mode] = {
            "members": [{"train_edges_used": int(m["train_edges_used"]), "model_seed": int(m["model_seed"]),
                         "community_seed": int(m["community_seed"]), "seconds": float(m["seconds"]),
                         "history": json.loads(str(m["history"]))} for m in members],
            "member_auc": member_aucs,
            "member_auc_summary": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                   for k, v in member_aucs.items()},
            "rows": rows,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["seed_only", "subsample"])
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=0.8, help="subsample mode only")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261118, help="pairwise reference draws")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu",
                        help="training device of every member (cpu: one thread per member)")
    parser.add_argument("--mode", help=argparse.SUPPRESS)       # worker: train one member
    parser.add_argument("--member", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode is not None:
        train_member(args)
        return

    started = time.time()
    todo = [(mode, m) for mode in args.modes for m in range(args.members)
            if not _member_file(args.out, mode, m).exists()]
    running = []
    while todo or running:
        while todo and len(running) < args.jobs:
            mode, m = todo.pop(0)
            log = open(args.out / f"{mode}_member{m}.log", "w")
            running.append(subprocess.Popen(
                [sys.executable, __file__, "--config", str(args.config), "--out", str(args.out),
                 "--fraction", str(args.fraction), "--device", args.device, "--mode", mode, "--member", str(m)],
                stdout=log, stderr=subprocess.STDOUT))
        time.sleep(5)
        for process in [p for p in running if p.poll() is not None]:
            if process.returncode != 0:
                raise RuntimeError(f"member failed: {process.args}")
            running.remove(process)
            print(f"done: {process.args[-3]} member {process.args[-1]} ({time.time() - started:.0f}s)", flush=True)

    import numpy as np

    from scripts.sweep_conditional import _setup
    from stability.edge_recovery_experiment import environment_record
    from stability.model_training import resolve_device

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    config, _, split = _setup(raw)
    y = np.r_[np.ones(len(split.val)), np.zeros(len(split.val_negatives))].astype(int)
    result = analyze(args, y)
    (args.out / "bootstrap_ensemble.json").write_text(json.dumps({
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "config": config.to_dict(), "environment": environment_record(resolve_device(args.device)), "results": result,
    }, indent=2, default=str))
    print(f"wrote {args.out / 'bootstrap_ensemble.json'}")


if __name__ == "__main__":
    main()
