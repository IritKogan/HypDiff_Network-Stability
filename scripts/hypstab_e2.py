"""Experiment E2 (Stage 4b): does multi-step denoising make the K reconstructions informative?

Usage (from the repository root):
    python scripts/hypstab_e2.py --plan configs/hypstab/e2.toml [--config configs/hypstab/cora.toml]
        [--run-dir DIR] [--autoencoder-from RUN_DIR] [--smoke] [--set SECTION.KEY=VALUE ...]

Validation protocol only (pool edges and control pairs are never touched). One shared
autoencoder + communities; one denoiser per trained variant (variants with ``denoiser_from``
only change the sampler and reuse that denoiser). Resumable: a variant whose folder has
``metrics.json`` is not recomputed. --smoke: 20 denoiser epochs, K = 2 (pipeline check only).

Writes <run_dir>/e2_results.json, <run_dir>/e2_summary.csv and per variant
variants/<name>/{config.toml, denoiser.pt, denoiser_history.json, metrics.json}.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
import tomllib

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hypstab.config import config_from_dict, config_to_dict, config_to_toml  # noqa: E402
from hypstab.data.features import with_train_edge_features  # noqa: E402
from hypstab.data.graph import load_graph  # noqa: E402
from hypstab.data.split import split_edges  # noqa: E402
from hypstab.evaluation.informativeness import variant_metrics  # noqa: E402
from hypstab.evaluation.validation import diffusion_probabilities, validation_report  # noqa: E402
from hypstab.models.hgcn import normalized_adjacency  # noqa: E402
from hypstab.runinfo import start_run  # noqa: E402
from hypstab.training.autoencoder import build_autoencoder, train_autoencoder  # noqa: E402
from hypstab.training.communities import fit_communities  # noqa: E402
from hypstab.training.denoiser import build_diffusion, train_denoiser  # noqa: E402
from hypstab_train import AUTOENCODER_SECTIONS, apply_overrides  # noqa: E402

SUMMARY_COLUMNS = (
    "variant", "training", "sampler", "steps", "t_star", "train_t_max", "auc_mean_probability",
    "auc_clamp_only", "auc_gain_over_clamp_only", "probability_std_positive", "expected_share_std_across_K",
    "p_e_share_intermediate", "p_e_stratified_auc_given_own_mean", "p_e_ll_gain_over_own_mean",
    "spread_stratified_auc_given_own_mean", "spread_ll_gain_over_own_mean", "quality_pass", "informative",
    "train_seconds", "skipped_epochs",
)


def variant_config(base_raw: dict, overrides: dict, smoke: bool):
    raw = copy.deepcopy(base_raw)
    apply_overrides(raw, [f"{key}={json.dumps(value)}" for key, value in overrides.items()])
    if smoke:
        raw["denoiser"]["epochs"] = 20
        raw["recovery"]["reconstructions"] = 2
    return config_from_dict(raw)


def summary_row(name: str, config, metrics: dict, train_seconds, skipped) -> dict:
    return {
        "variant": name, "training": config.denoiser.training, "sampler": config.recovery.sampler,
        "steps": config.recovery.steps if config.recovery.sampler == "iterative" else 1,
        "t_star": config.recovery.t_star, "train_t_max": config.denoiser.train_t_max,
        "auc_mean_probability": metrics["auc_mean_probability"], "auc_clamp_only": metrics["auc_clamp_only"],
        "auc_gain_over_clamp_only": metrics["auc_gain_over_clamp_only"],
        "probability_std_positive": metrics["probability_std_positive"],
        "expected_share_std_across_K": metrics["expected_share_std_across_K"],
        "p_e_share_intermediate": metrics["p_e"]["share_intermediate"],
        "p_e_stratified_auc_given_own_mean": metrics["p_e"]["stratified_auc_given_own_mean"],
        "p_e_ll_gain_over_own_mean": metrics["p_e"]["ll_gain_over_own_mean"],
        "spread_stratified_auc_given_own_mean": metrics["spread"]["stratified_auc_given_own_mean"],
        "spread_ll_gain_over_own_mean": metrics["spread"]["ll_gain_over_own_mean"],
        "quality_pass": metrics["quality_pass"], "informative": metrics["informative"],
        "train_seconds": train_seconds, "skipped_epochs": skipped,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default=str(REPO_ROOT / "configs" / "hypstab" / "e2.toml"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "hypstab" / "cora.toml"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--autoencoder-from", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args(argv)

    plan = tomllib.loads(Path(args.plan).read_text(encoding="utf-8"))
    criteria = plan["criteria"]
    variants = plan["variant"]
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise SystemExit("variant names must be unique")
    for v in variants:
        if "denoiser_from" in v and v["denoiser_from"] not in names[: names.index(v["name"])]:
            raise SystemExit(f"{v['name']}: denoiser_from must name an EARLIER variant")
    base_raw = apply_overrides(tomllib.loads(Path(args.config).read_text(encoding="utf-8")), args.set)
    base = config_from_dict(copy.deepcopy(base_raw))

    run_dir = Path(args.run_dir) if args.run_dir else None
    resuming = run_dir is not None and (run_dir / "config.toml").exists()
    if resuming:
        device = torch.device("cuda" if torch.cuda.is_available() and base.run.device != "cpu" else "cpu")
    else:
        run_dir, device = start_run(base, run_dir)
        (run_dir / "plan.toml").write_text(Path(args.plan).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"run dir: {run_dir} | device: {device} | smoke: {args.smoke}", flush=True)

    graph = load_graph(base.data.dataset, base.data.root, base.data.download, base.synthetic,
                       base.data.features)
    s = base.split
    split = split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction, s.pool_fraction,
                        s.control_pairs, s.seed)
    graph = with_train_edge_features(graph, split.train, base.data.features)  # train edges only
    features = graph.features.to(device)
    adjacency = normalized_adjacency(graph.node_count, split.train, device, base.encoder.adjacency_norm)

    # shared autoencoder (trained once, or reused) + communities
    source = Path(args.autoencoder_from) if args.autoencoder_from else (
        run_dir if (run_dir / "autoencoder.pt").exists() else None)
    if source is not None:
        previous = tomllib.loads((source / "config.toml").read_text(encoding="utf-8"))
        differing = [n for n in AUTOENCODER_SECTIONS if previous.get(n) != config_to_dict(base)[n]]
        if differing:
            raise SystemExit(f"autoencoder source differs in {differing}")
        autoencoder = build_autoencoder(graph.features.shape[1], base).to(device)
        autoencoder.load_state_dict(torch.load(source / "autoencoder.pt", map_location=device))
        autoencoder.eval()
        encoder_auc = json.loads((source / "autoencoder_history.json").read_text(encoding="utf-8"))["best_val_auc"]
        if source != run_dir:
            torch.save(autoencoder.state_dict(), run_dir / "autoencoder.pt")
            (run_dir / "autoencoder_history.json").write_text(
                (source / "autoencoder_history.json").read_text(encoding="utf-8"), encoding="utf-8")
    else:
        result = train_autoencoder(graph, split, base, device)
        autoencoder, encoder_auc = result.model, result.history["best_val_auc"]
        torch.save(autoencoder.state_dict(), run_dir / "autoencoder.pt")
        (run_dir / "autoencoder_history.json").write_text(json.dumps(result.history), encoding="utf-8")
    context = fit_communities(autoencoder, features, adjacency, base)
    print(f"encoder val AUC {encoder_auc:.4f} | communities {context.summary()['sizes']}", flush=True)

    pos = torch.as_tensor(split.val, device=device)
    neg = torch.as_tensor(split.val_negatives, device=device)
    with torch.no_grad():
        embeddings = autoencoder.encode(features, adjacency)
    clamp = validation_report(autoencoder, features, adjacency, split, context, base)  # no_diffusion + clamp_only
    from hypstab.models.clamp import clamp_to_centers
    with torch.no_grad():
        clamped = clamp_to_centers(embeddings, context.labels, context.centers, base.geometry.curvature,
                                   base.clamp.clip)
        clamp_pos = torch.sigmoid(autoencoder.edge_logits(clamped, pos)).cpu().numpy()
        clamp_neg = torch.sigmoid(autoencoder.edge_logits(clamped, neg)).cpu().numpy()

    rows, results = [], {}
    denoisers: dict[str, Path] = {}
    for v in variants:
        name = v["name"]
        folder = run_dir / "variants" / name
        folder.mkdir(parents=True, exist_ok=True)
        config = variant_config(base_raw, v.get("overrides", {}), args.smoke)
        if (folder / "metrics.json").exists():
            stored = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            print(f"[{name}] done earlier, skipped", flush=True)
            denoisers[name] = folder / "denoiser.pt" if (folder / "denoiser.pt").exists() else denoisers.get(
                v.get("denoiser_from", ""))
            rows.append(stored["summary"])
            results[name] = stored
            continue
        (folder / "config.toml").write_text(config_to_toml(config), encoding="utf-8")
        print(f"\n[{name}] {v.get('description', '')}", flush=True)
        train_seconds, skipped = 0.0, 0
        if "denoiser_from" in v:
            donor = variants[names.index(v["denoiser_from"])]
            donor_config = variant_config(base_raw, donor.get("overrides", {}), args.smoke)
            if config_to_dict(donor_config)["denoiser"] != config_to_dict(config)["denoiser"] or \
                    config_to_dict(donor_config)["diffusion"] != config_to_dict(config)["diffusion"]:
                raise SystemExit(f"{name}: denoiser/diffusion config differs from {v['denoiser_from']}")
            diffusion = build_diffusion(config).to(device)
            diffusion.load_state_dict(torch.load(denoisers[v["denoiser_from"]], map_location=device))
            diffusion.eval()
        else:
            torch.manual_seed(config.run.seed)
            start = time.perf_counter()
            diffusion, history = train_denoiser(graph, split, config, device, autoencoder, context,
                                                log_every=args.log_every)
            train_seconds = round(time.perf_counter() - start, 1)
            skipped = len(history["skipped_epochs"])
            torch.save(diffusion.state_dict(), folder / "denoiser.pt")
            (folder / "denoiser_history.json").write_text(json.dumps(history), encoding="utf-8")
            denoisers[name] = folder / "denoiser.pt"
        prob_pos, prob_neg = diffusion_probabilities(autoencoder, diffusion, embeddings, adjacency, context,
                                                     config, pos, neg)
        metrics = variant_metrics(prob_pos, prob_neg, clamp_pos, clamp_neg, **criteria)
        row = summary_row(name, config, metrics, train_seconds, skipped)
        stored = {"description": v.get("description", ""), "overrides": v.get("overrides", {}),
                  "denoiser_from": v.get("denoiser_from"), "metrics": metrics, "summary": row}
        (folder / "metrics.json").write_text(json.dumps(stored, indent=2), encoding="utf-8")
        np.savez_compressed(folder / "probabilities.npz", prob_pos=prob_pos, prob_neg=prob_neg)
        rows.append(row)
        results[name] = stored
        print(f"[{name}] AUC {metrics['auc_mean_probability']:.4f} (clamp_only {metrics['auc_clamp_only']:.4f}) "
              f"p_e strat-AUC {metrics['p_e']['stratified_auc_given_own_mean']} "
              f"LL gain {metrics['p_e']['ll_gain_over_own_mean']:.4f} | train {train_seconds}s", flush=True)

    output = {"protocol": "validation only; shared autoencoder + communities", "smoke": args.smoke,
              "device": str(device), "encoder_best_val_auc": encoder_auc, "criteria": criteria,
              "no_diffusion_auc": clamp["no_diffusion"]["auc"], "clamp_only_auc": clamp["clamp_only"]["auc"],
              "variants": results}
    (run_dir / "e2_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (run_dir / "e2_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("\nvariant              AUC    clamp  pe_strat pe_LL   spread_strat informative", flush=True)
    for r in rows:
        strat = r["spread_stratified_auc_given_own_mean"]
        print(f"{r['variant']:<20} {r['auc_mean_probability']:.4f} {r['auc_clamp_only']:.4f} "
              f"{r['p_e_stratified_auc_given_own_mean'] or float('nan'):.3f}    {r['p_e_ll_gain_over_own_mean']:.4f}  "
              f"{strat or float('nan'):.3f}        {r['informative']}", flush=True)
    return output


if __name__ == "__main__":
    main()
