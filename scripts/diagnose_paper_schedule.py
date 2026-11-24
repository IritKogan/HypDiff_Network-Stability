"""Experiment E1 on the old code: paper diffusion schedule + t* sweep (validation protocol only).

Plan: configs/e1_paper_schedule.toml (variants = overrides of base_config). For every variant:
train a denoiser (shared encoder + community context, train edges only), run K reconstructions
of the validation input graph (train edges), decode validation edges and validation negatives,
and report quality (vs clamp_only) and whether the K samples carry information beyond their mean
(stability/sample_diversity.schedule_variant_metrics). Pool and control pairs are never used.

Outputs in --out-dir (resumable: rerun with the same --out-dir; finished variants are skipped,
trained denoisers are loaded):
  plan.toml, base_config.toml        copies of the inputs
  encoder_context.pt                 shared encoder + community labels / centres
  variants/<name>/config.toml        resolved config of the variant
  variants/<name>/denoiser.pt        trained denoiser
  variants/<name>/probabilities.npz  (K, pairs) validation probabilities + clamp_only
  variants/<name>/result.json        metrics, training history summary, timings
  e1_results.json, e1_summary.csv    all variants

Examples:
  python scripts/diagnose_paper_schedule.py --device cuda --out-dir artifacts/e1
  python scripts/diagnose_paper_schedule.py --device cpu --smoke --variants baseline_T200_t7 paper_T1000_t250
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHARED_SECTIONS_EDITABLE = ("denoiser", "recovery")
SUMMARY_COLUMNS = (
    "variant", "timesteps", "radial_delta", "angular_noise", "t_star", "train_t_max",
    "noise_coefficient", "signal_coefficient", "auc_mean_probability", "auc_clamp_only",
    "auc_gain_over_clamp_only", "hit_rate", "hit_rate_clamp_only", "sensitivity",
    "probability_std_positive", "expected_share_std_across_K", "p_e_share_intermediate",
    "p_e_stratified_auc_given_own_mean", "p_e_ll_gain_over_own_mean",
    "spread_stratified_auc_given_own_mean", "spread_ll_gain_over_own_mean",
    "quality_pass", "informative", "train_seconds", "skipped_epochs",
)


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Copy of a config dict with "section.key" = value overrides; keys must already exist."""
    result = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if not key or section not in result or key not in result[section]:
            raise KeyError(f"override {dotted!r} does not name an existing config key")
        result[section][key] = value
    return result


def load_plan(path: Path, smoke: bool = False) -> dict[str, Any]:
    """Read and validate the E1 plan; returns run settings, criteria, base config and variant configs."""
    from stability.experiment_config import config_from_dict

    plan = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for section, keys in (("run", ("base_config", "reconstructions", "strata", "ll_folds")),
                          ("criteria", ("quality_tolerance", "informative_auc", "min_ll_gain"))):
        for key in keys:
            if key not in plan.get(section, {}):
                raise ValueError(f"[{section}].{key} missing in {path}")
    variants = plan.get("variant", [])
    if not variants:
        raise ValueError("no [[variant]] in the plan")
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate variant names: {names}")
    base = tomllib.loads((ROOT / plan["run"]["base_config"]).read_text(encoding="utf-8"))
    reconstructions = int(plan["run"]["reconstructions"])
    if smoke:
        base = apply_overrides(base, plan["smoke"]["overrides"])
        reconstructions = int(plan["smoke"]["reconstructions"])
    if reconstructions < 2:
        raise ValueError("reconstructions must be >= 2 (spread across K)")
    resolved = {}
    for variant in variants:
        overrides = variant.get("overrides", {})
        bad = [k for k in overrides if k.partition(".")[0] not in SHARED_SECTIONS_EDITABLE]
        if bad:
            raise ValueError(f"variant {variant['name']}: only {SHARED_SECTIONS_EDITABLE} keys may change "
                             f"(the encoder and community context are shared), got {bad}")
        raw = apply_overrides(base, overrides)
        raw["recovery"]["reconstructions"] = reconstructions
        config_from_dict(raw)   # validates, incl. the t* guard (stability/diffusion_schedule.py)
        resolved[variant["name"]] = {"raw": raw, "overrides": overrides,
                                     "description": variant.get("description", "")}
    return {"plan": plan, "base": base, "variants": resolved, "reconstructions": reconstructions}


def dump_toml(raw: dict[str, Any]) -> str:
    from scripts.run_gpu_checks import dump_toml as dump

    return dump(raw)


def _setup(raw, device):
    import torch

    from stability.edge_split import split_edges
    from stability.experiment_config import config_from_dict
    from stability.graph_data import load_graph

    config = config_from_dict(raw)
    root = Path(config.data.root)
    graph = load_graph(config.data.dataset, root if root.is_absolute() else ROOT / root, config.data.download)
    s = config.split
    split = split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction,
                        s.pool_fraction, s.control_pairs, s.seed)
    return config, graph, split


def evaluate_variant(denoiser, encoder, context, graph, split, config, device, K) -> dict[str, Any]:
    """K reconstructions of the validation input (train edges only). Returns probabilities + spreads."""
    import numpy as np
    import torch

    from stability.edge_recovery_experiment import recovery_seed
    from stability.model_training import normalized_adjacency
    from stability.sample_diversity import tangent_spread
    from stability.validation_recovery import VALIDATION_SERIES_INDEX

    rec = config.recovery
    t_star = rec.t_star
    adjacency = normalized_adjacency(graph.node_count, split.train, device)
    features = graph.features.to(device)
    pos = torch.as_tensor(split.val, device=device)
    neg = torch.as_tensor(split.val_negatives, device=device)
    labels = context.labels.to(device)
    centers = context.centers.to(device)
    with torch.no_grad():
        embeddings = encoder.encode(features, adjacency)[2]
        node_prototypes = centers[labels]
        tangent = denoiser.manifold.logmap(node_prototypes, embeddings, c=denoiser.c)
        x_start = denoiser.normalization(tangent)
        direction = torch.stack([denoiser.manifold.logmap0(p, c=denoiser.c) for p in centers])
        edge_index = denoiser.edge(adjacency.coalesce())
        xts, x0s, prob_pos, prob_neg = [], [], [], []
        for k in range(K):
            seed = recovery_seed(rec.seed, VALIDATION_SERIES_INDEX, k)
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = denoiser.draw_noise(x_start, generator)
            t = torch.full((x_start.shape[0],), t_star, dtype=torch.long, device=device)
            xts.append(denoiser.q_sample(x_start, t, direction, noise, labels, None, restrict=True).cpu().numpy())
            generator = torch.Generator(device=device).manual_seed(seed)
            x0 = denoiser._denoise_from_tstar(x_start, edge_index, direction, labels, t_star, generator)
            x0s.append(x0.cpu().numpy())
            recovered = denoiser.manifold.expmap(denoiser.renormal(x0, tangent), node_prototypes, c=denoiser.c)
            generator = torch.Generator(device=device).manual_seed(seed)
            reference = denoiser.sample_from_tstar(embeddings, adjacency, labels, centers, t_star, generator)
            if not torch.allclose(reference, recovered, atol=1e-5):   # same path as the pipeline
                raise RuntimeError("diagnostic path differs from sample_from_tstar")
            prob_pos.append(encoder.decode(recovered, pos).cpu().numpy().astype(np.float64))
            prob_neg.append(encoder.decode(recovered, neg).cpu().numpy().astype(np.float64))
        shrunk = denoiser.shrink_to_prototype(embeddings, labels, centers)
        clamp_pos = encoder.decode(shrunk, pos).cpu().numpy().astype(np.float64)
        clamp_neg = encoder.decode(shrunk, neg).cpu().numpy().astype(np.float64)
        base_pos = encoder.decode(embeddings, pos).cpu().numpy().astype(np.float64)
        base_neg = encoder.decode(embeddings, neg).cpu().numpy().astype(np.float64)
    xts, x0s = np.stack(xts), np.stack(x0s)
    input_spread, output_spread = tangent_spread(xts), tangent_spread(x0s)
    return {
        "prob_pos": np.stack(prob_pos), "prob_neg": np.stack(prob_neg),
        "clamp_pos": clamp_pos, "clamp_neg": clamp_neg, "base_pos": base_pos, "base_neg": base_neg,
        "input_spread": input_spread, "output_spread": output_spread,
        "sensitivity": output_spread / input_spread if input_spread > 0 else None,
        "displacement": float(np.abs(x0s.mean(axis=0) - x_start.cpu().numpy()).mean()),
    }


def summary_row(name: str, result: dict[str, Any]) -> dict[str, Any]:
    m, den, rec = result["metrics"], result["denoiser"], result["recovery"]
    return {
        "variant": name, "timesteps": den["timesteps"], "radial_delta": den["radial_delta"],
        "angular_noise": den["angular_noise"], "t_star": rec["t_star"], "train_t_max": den["train_t_max"],
        "noise_coefficient": result["noise_coefficient"], "signal_coefficient": result["signal_coefficient"],
        "auc_mean_probability": m["auc_mean_probability"], "auc_clamp_only": m["auc_clamp_only"],
        "auc_gain_over_clamp_only": m["auc_gain_over_clamp_only"], "hit_rate": m["hit_rate"],
        "hit_rate_clamp_only": m["hit_rate_clamp_only"], "sensitivity": result["sensitivity"],
        "probability_std_positive": m["probability_std_positive"],
        "expected_share_std_across_K": m["expected_share_std_across_K"],
        "p_e_share_intermediate": m["p_e"]["share_intermediate"],
        "p_e_stratified_auc_given_own_mean": m["p_e"]["stratified_auc_given_own_mean"],
        "p_e_ll_gain_over_own_mean": m["p_e"]["ll_gain_over_own_mean"],
        "spread_stratified_auc_given_own_mean": m["spread"]["stratified_auc_given_own_mean"],
        "spread_ll_gain_over_own_mean": m["spread"]["ll_gain_over_own_mean"],
        "quality_pass": result["criteria"]["quality_pass"], "informative": result["criteria"]["informative"],
        "train_seconds": result["train_seconds"], "skipped_epochs": result["skipped_epochs"],
    }


def criteria_flags(metrics: dict[str, Any], criteria: dict[str, Any]) -> dict[str, bool]:
    quality = metrics["auc_mean_probability"] >= metrics["auc_clamp_only"] - float(criteria["quality_tolerance"])
    strat = metrics["p_e"]["stratified_auc_given_own_mean"]
    informative = (strat is not None and strat >= float(criteria["informative_auc"])
                   and metrics["p_e"]["ll_gain_over_own_mean"] >= float(criteria["min_ll_gain"]))
    return {"quality_pass": bool(quality), "informative": bool(informative)}


def main() -> None:
    import numpy as np
    import torch

    from stability.edge_recovery_experiment import environment_record
    from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
    from stability.model_training import (
        build_context, build_model_args, normalized_adjacency, resolve_device, train_denoiser, train_encoder,
    )
    from stability.partial_recovery import CommunityContext
    from stability.sample_diversity import schedule_variant_metrics

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs" / "e1_paper_schedule.toml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--variants", nargs="+", help="subset of the plan's variant names")
    parser.add_argument("--smoke", action="store_true", help="apply the plan's [smoke] settings")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    loaded = load_plan(args.plan, smoke=args.smoke)
    plan, criteria = loaded["plan"], loaded["plan"]["criteria"]
    names = args.variants or list(loaded["variants"])
    unknown = set(names) - set(loaded["variants"])
    if unknown:
        raise SystemExit(f"unknown variants {sorted(unknown)}; plan has {list(loaded['variants'])}")
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    (out / "variants").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.plan, out / "plan.toml")
    (out / "base_config.toml").write_text(dump_toml(loaded["base"]), encoding="utf-8")
    device = resolve_device(args.device)
    print(f"device {device}; variants {names}; K = {loaded['reconstructions']}; smoke = {args.smoke}", flush=True)

    config, graph, split = _setup(loaded["base"], device)
    model_args, _ = build_model_args(graph, config, device)
    shared = out / "encoder_context.pt"
    started = time.time()
    if shared.exists():
        saved = torch.load(shared, map_location=device, weights_only=False)
        encoder = StabilityLPModel(model_args).to(device)
        encoder.load_state_dict(saved["encoder"])
        encoder.eval()
        encoder_history = saved["history"]
        context = CommunityContext(tuple(range(graph.node_count)), saved["labels"], saved["centers"],
                                   config.model.curvature)
        print("encoder + context loaded", flush=True)
    else:
        encoder, _, train_adjacency, encoder_history, _ = train_encoder(graph, split, config, device,
                                                                        log_every=args.log_every)
        context = build_context(graph, config, encoder, train_adjacency, device)
        torch.save({"encoder": encoder.state_dict(), "labels": context.labels, "centers": context.centers,
                    "history": encoder_history}, shared)
    encoder_seconds = round(time.time() - started, 1)
    print(f"encoder best val AUC {encoder_history['autoencoder_best_val_auc']:.4f}", flush=True)
    train_adjacency = normalized_adjacency(graph.node_count, split.train, device)

    results = {}
    for name in names:
        variant = loaded["variants"][name]
        vdir = out / "variants" / name
        vdir.mkdir(exist_ok=True)
        result_file = vdir / "result.json"
        if result_file.exists():
            results[name] = json.loads(result_file.read_text(encoding="utf-8"))
            print(f"\n[{name}] done earlier, skipped", flush=True)
            continue
        print(f"\n[{name}] {variant['description']}", flush=True)
        (vdir / "config.toml").write_text(dump_toml(variant["raw"]), encoding="utf-8")
        vconfig, _, _ = _setup(variant["raw"], device)
        vargs, _ = build_model_args(graph, vconfig, device)
        checkpoint = vdir / "denoiser.pt"
        started = time.time()
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            denoiser = StabilityDiffusion(vargs, dim=vargs.dim, curvature=vconfig.model.curvature, device=device)
            denoiser.load_state_dict(saved["state"])
            denoiser.eval()
            history = saved["history"]
            print(f"[{name}] denoiser loaded", flush=True)
        else:
            denoiser, history = train_denoiser(graph, split, vconfig, device, encoder, vargs, train_adjacency,
                                               context, log_every=args.log_every)
            torch.save({"state": denoiser.state_dict(), "history": history}, checkpoint)
        train_seconds = round(time.time() - started, 1)
        started = time.time()
        evaluation = evaluate_variant(denoiser, encoder, context, graph, split, vconfig, device,
                                      vconfig.recovery.reconstructions)
        metrics = schedule_variant_metrics(evaluation["prob_pos"], evaluation["prob_neg"],
                                           evaluation["clamp_pos"], evaluation["clamp_neg"],
                                           vconfig.recovery.target_fpr, int(plan["run"]["strata"]),
                                           int(plan["run"]["ll_folds"]))
        np.savez_compressed(vdir / "probabilities.npz", **{k: evaluation[k] for k in (
            "prob_pos", "prob_neg", "clamp_pos", "clamp_neg", "base_pos", "base_neg")})
        t_star = vconfig.recovery.t_star
        losses = [x for x in history["denoiser_loss"] if x == x]
        result = {
            "variant": name, "description": variant["description"], "overrides": variant["overrides"],
            "denoiser": variant["raw"]["denoiser"], "recovery": variant["raw"]["recovery"],
            "noise_coefficient": denoiser.noise_coefficient(t_star),
            "signal_coefficient": denoiser.signal_coefficient(t_star),
            "input_spread": evaluation["input_spread"], "output_spread": evaluation["output_spread"],
            "sensitivity": evaluation["sensitivity"], "displacement": evaluation["displacement"],
            "auc_no_diffusion": float(metrics_auc(evaluation["base_pos"], evaluation["base_neg"])),
            "metrics": metrics, "criteria": criteria_flags(metrics, criteria),
            "train_seconds": train_seconds, "evaluation_seconds": round(time.time() - started, 1),
            "final_denoiser_loss": losses[-1] if losses else None,
            "skipped_epochs": len(history.get("denoiser_skipped_epochs", [])),
        }
        result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results[name] = result
        m = metrics
        print(f"[{name}] AUC {m['auc_mean_probability']:.4f} (clamp_only {m['auc_clamp_only']:.4f}) "
              f"hit {m['hit_rate']:.3f} sens {result['sensitivity']:.3f} p_e strat-AUC "
              f"{m['p_e']['stratified_auc_given_own_mean']} LL gain {m['p_e']['ll_gain_over_own_mean']:.4f} "
              f"| train {train_seconds:.0f}s", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = [summary_row(n, results[n]) for n in names]
    with open(out / "e1_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "e1_results.json").write_text(json.dumps({
        "protocol": "validation only: train edges as input; validation edges vs validation negatives",
        "smoke": args.smoke, "device": str(device), "environment": environment_record(device),
        "encoder_best_val_auc": encoder_history["autoencoder_best_val_auc"],
        "encoder_seconds": encoder_seconds, "criteria": criteria, "variants": results,
    }, indent=2), encoding="utf-8")
    print("\nvariant                 AUC    clamp  sens   pe_strat  pe_LL    spread_strat")
    for r in rows:
        print(f"{r['variant']:<22} {r['auc_mean_probability']:.4f} {r['auc_clamp_only']:.4f} "
              f"{(r['sensitivity'] or 0):.3f}  {r['p_e_stratified_auc_given_own_mean'] or float('nan'):.3f}     "
              f"{r['p_e_ll_gain_over_own_mean']:.4f}  {r['spread_stratified_auc_given_own_mean'] or float('nan'):.3f}")
    print(f"\nwrote {out / 'e1_results.json'} and {out / 'e1_summary.csv'}")


def metrics_auc(positive, negative) -> float:
    from stability.sample_diversity import _auc

    return _auc(positive, negative)


if __name__ == "__main__":
    main()
