"""Build a self-contained HTML explorer of a pairwise-p_e Cora run (English, for the advisor).

The explorer shows, for every attack and damage level (10/20/30%):
  * a summary over all damage scenarios (series): p_e distribution vs the control group, ECDFs across levels
    with KS / Wasserstein distances, method comparison, scenario-to-scenario variation;
  * a single damage scenario and a single reconstruction: which removed edges returned;
  * a local graph view around a chosen removed edge: original / damaged / reconstruction (or p_e colours).

Per-reconstruction outcomes are not stored by run_experiment.py, so they are recomputed here from the run's
checkpoints with run_recoveries(keep_reconstructions=True) and checked to sum EXACTLY to the saved counts
(the build stops otherwise). The split and damage series are rebuilt from the saved config and compared with
split.npz / series.npz.

Example:
    python scripts/build_cora_explorer.py --run artifacts/runs/<run> --out artifacts/explorer/cora_explorer.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ATTACK_LABELS = {
    "random": "Random removal",
    "targeted_weighted": "Targeted (weighted by degree product)",
    "targeted_deterministic": "Targeted (deterministic, top degree product)",
}


def _bits(values) -> str:
    import numpy as np

    return base64.b64encode(np.packbits(np.asarray(values, dtype=bool), bitorder="little").tobytes()).decode()


def _round(values, digits=4):
    import numpy as np

    out = []
    for v in np.asarray(values, dtype=float).tolist():
        out.append(None if not np.isfinite(v) else round(v, digits))
    return out


def build_data(run: Path, features_csv: Path | None = None) -> dict:
    import networkx as nx
    import numpy as np
    from scipy.stats import ks_2samp, wasserstein_distance
    import torch

    from stability.edge_recovery_distribution import per_edge_probability
    from stability.edge_recovery_experiment import run_recoveries
    from stability.edge_split import split_edges
    from stability.experiment_config import load_config
    from stability.graph_data import load_graph
    from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
    from stability.model_training import TrainedModels, build_model_args, normalized_adjacency
    from stability.partial_recovery import CommunityContext
    from stability.perturbation_masks import make_damage_series
    from stability.probability_calibration import PlattCalibrator

    config = load_config(run / "config.toml")
    if config.recovery.edge_decision != "pairwise":
        raise ValueError("the explorer expects an edge_decision = 'pairwise' run")
    device = torch.device("cpu")
    data_root = Path(config.data.root)
    graph = load_graph(config.data.dataset, data_root if data_root.is_absolute() else ROOT / data_root, False)
    s = config.split
    split = split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction, s.pool_fraction,
                        s.control_pairs, s.seed)
    saved_split = np.load(run / "split.npz")
    for name in ("train", "val", "pool", "val_negatives", "control_pairs"):
        if not np.array_equal(getattr(split, name), saved_split[name]):
            raise RuntimeError(f"rebuilt split differs from the saved one: {name}")
    p = config.perturbation
    series = make_damage_series(split.pool, split.all_edges, graph.node_count, p.damage_fractions, p.attacks,
                                p.series, p.seed)
    saved_series = np.load(run / "series.npz")
    for attack, items in series.items():
        if not np.array_equal(np.stack([item.order for item in items]), saved_series[attack]):
            raise RuntimeError(f"rebuilt damage series differ from the saved ones: {attack}")

    args, _ = build_model_args(graph, config, device)
    encoder = StabilityLPModel(args)
    encoder.load_state_dict(torch.load(run / "checkpoints" / "encoder.pt"))
    encoder.eval()
    denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.model.curvature, device=device)
    denoiser.load_state_dict(torch.load(run / "checkpoints" / "denoiser.pt"))
    denoiser.eval()
    context_saved = torch.load(run / "checkpoints" / "community_context.pt", weights_only=False)
    context = CommunityContext(tuple(range(graph.node_count)), context_saved["labels"], context_saved["centers"],
                               config.model.curvature)
    models = TrainedModels(encoder, denoiser, context, normalized_adjacency(graph.node_count, split.train, device), {})
    print("recomputing per-reconstruction outcomes ...", flush=True)
    validation = json.loads((run / "validation_recovery.json").read_text())
    methods = ("diffusion", "no_diffusion", "clamp_only")
    taus = {m: validation[m]["tau"] for m in methods}
    calibrators = {m: PlattCalibrator(**validation[m]["calibration"]["calibrator"]) for m in methods}
    results = run_recoveries(graph, split, series, models, config, device, taus, calibrators=calibrators,
                             keep_reconstructions=True)
    saved = np.load(run / "recovery_counts.npz")
    for attack, arrays in results.items():
        if not np.array_equal(arrays["recovered"], saved[f"{attack}__recovered"]):
            raise RuntimeError(f"recomputed reconstructions differ from the saved run: {attack}")
        if not np.array_equal(arrays["recovered_by_reconstruction"].sum(axis=2), saved[f"{attack}__recovered"]):
            raise RuntimeError(f"per-reconstruction outcomes do not sum to the saved counts: {attack}")
    print("recomputed outcomes match the saved run exactly", flush=True)

    K = config.recovery.reconstructions
    fractions = list(p.damage_fractions)
    removal_counts = [int(c) for c in saved_series["removal_counts"]]
    pool = split.pool
    degree = np.bincount(graph.edges.reshape(-1), minlength=graph.node_count)
    if features_csv is not None and features_csv.exists():
        import pandas as pd

        table = pd.read_csv(features_csv).sort_values("pool_index")
        if not (np.array_equal(table["source"], pool[:, 0]) and np.array_equal(table["target"], pool[:, 1])):
            raise RuntimeError("features.csv does not match the run's pool edges")
        pool_features = {
            "common_neighbours": table["common_neighbours"].astype(int).tolist(),
            "bridge": table["bridge"].astype(int).tolist(),
            "betweenness": [round(float(v) * 1e4, 3) for v in table["edge_betweenness"]],   # x 1e-4
            "same_community": table["same_community"].astype(int).tolist(),
        }
    else:
        G = nx.Graph()
        G.add_nodes_from(range(graph.node_count))
        G.add_edges_from(map(tuple, graph.edges.tolist()))
        bridges = {tuple(sorted(e)) for e in nx.bridges(G)}
        labels = context_saved["labels"].numpy()
        pool_features = {
            "common_neighbours": [len(list(nx.common_neighbors(G, int(u), int(v)))) for u, v in pool],
            "bridge": [int(tuple(sorted((int(u), int(v)))) in bridges) for u, v in pool],
            "same_community": [int(labels[u] == labels[v]) for u, v in pool],
        }
    attacks = {}
    for attack, arrays in results.items():
        draws_total = saved_series[attack].shape[0] * K
        # with few (series x reconstructions) draws p_e takes only the values i / draws: one bin per value
        bins = np.linspace(0, 1, 21) if draws_total > 20 else (np.arange(draws_total + 2) - 0.5) / draws_total
        n_e, _, p_e = per_edge_probability(saved[f"{attack}__removed"], saved[f"{attack}__recovered"], K)
        draws = int(saved[f"{attack}__deterministic_draws"])
        control = saved[f"{attack}__control_hits"].sum(axis=1) / (saved[f"{attack}__control_hits"].shape[1] * K)
        levels = []
        for level, fraction in enumerate(fractions):
            evaluated = n_e[level] > 0
            values = p_e[level][evaluated]
            with np.errstate(invalid="ignore", divide="ignore"):
                clamp = saved[f"{attack}__clamp_only"][level].sum(axis=0) / (n_e[level] * draws)
                nodiff = saved[f"{attack}__no_diffusion"][level].sum(axis=0) / (n_e[level] * draws)
            control_hist, _ = np.histogram(control[level], bins=bins)
            scenarios = []
            orders = saved_series[attack]
            for s_index in range(orders.shape[0]):
                removed_idx = orders[s_index][: removal_counts[level]]
                kept = arrays["recovered_by_reconstruction"][level, s_index][:, removed_idx]   # (K, R)
                scenarios.append({
                    "bits": [_bits(kept[k]) for k in range(K)],
                    "mean_recovery": round(float(kept.mean()), 4),
                    "control_mean": round(float(saved[f"{attack}__control_hits"][level, s_index].mean() / K), 4),
                })
            levels.append({
                "fraction": fraction,
                "removed_per_scenario": removal_counts[level],
                "p_e": _round(p_e[level], 3),
                "n_e": [int(v) for v in n_e[level]],
                "edges_evaluated": int(evaluated.sum()),
                "mean": {"diffusion": round(float(values.mean()), 4),
                         "clamp_only": round(float(np.nanmean(clamp[evaluated])), 4),
                         "no_diffusion": round(float(np.nanmean(nodiff[evaluated])), 4),
                         "control": round(float(control[level].mean()), 4)},
                "control_hist": [round(float(v) / control.shape[1], 5) for v in control_hist],
                "vs_control": {"ks": round(float(ks_2samp(values, control[level]).statistic), 4),
                               "wasserstein": round(float(wasserstein_distance(values, control[level])), 4)},
                "scenarios": scenarios,
            })
        between = []
        for a in range(len(fractions)):
            for b in range(a + 1, len(fractions)):
                va = p_e[a][n_e[a] > 0]
                vb = p_e[b][n_e[b] > 0]
                between.append({"from": fractions[a], "to": fractions[b],
                                "ks": round(float(ks_2samp(va, vb).statistic), 4),
                                "wasserstein": round(float(wasserstein_distance(va, vb)), 4),
                                "mean_change": round(float(vb.mean() - va.mean()), 4)})
        attacks[attack] = {"label": ATTACK_LABELS.get(attack, attack), "orders": saved_series[attack].tolist(),
                           "hist_edges": [round(float(b), 6) for b in bins],
                           "levels": levels, "between_levels": between}

    summary = json.loads((run / "summary.json").read_text())
    return {
        "run": run.name,
        "dataset": graph.name,
        "nodes": graph.node_count,
        "classes": [int(c) for c in graph.class_labels],
        "communities": [int(c) for c in context_saved["labels"].tolist()],
        "degree": [int(d) for d in degree],
        "edges": graph.edges.astype(int).ravel().tolist(),
        "pool": pool.astype(int).ravel().tolist(),
        "pool_features": pool_features,
        "K": K,
        "series": p.series,
        "attacks": attacks,
        "validation_auc": {m: round(summary["validation_recovery"][m]["auc_mean_probability"], 3)
                           for m in ("diffusion", "clamp_only", "no_diffusion")},
        "split_sizes": summary["split_sizes"],
        "control_pairs": int(len(split.control_pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/analysis/cora_pairwise/features.csv",
                        help="per-edge features from analyze_edge_stability.py (recomputed if missing)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="JSON cache of the computed data; reused if it exists (skips the recomputation)")
    args = parser.parse_args()
    if args.cache is not None and args.cache.exists():
        data = json.loads(args.cache.read_text())
    else:
        data = build_data(args.run, args.features)
        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(data, separators=(",", ":")))
    template = (ROOT / "scripts" / "cora_explorer_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("/*__DATA__*/null", payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html.encode()) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
