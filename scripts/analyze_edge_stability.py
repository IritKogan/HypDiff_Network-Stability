"""Which edges are stable? Analysis of a pairwise-p_e run (Decision 10) against classical measures.

Reads a run directory written by run_experiment.py (edge_decision = "pairwise") and the graph, and
writes <out-dir>/analysis.json plus small CSVs for plotting:

  overview.csv        mean / median / quantiles of p_e per attack, level and method; control
  histograms.csv      p_e histograms (20 bins) of removed edges (diffusion) and control pairs
  features.csv        per pool edge: classical edge measures on the ORIGINAL graph + expected p_e
                      (random attack, every level)
  feature_bins.csv    mean expected p_e per quintile of each classical measure (random, 20%)
  dynamics.csv        paired per-edge change of expected p_e between damage levels (random)
  attacks.csv         per-edge expected p_e under random vs targeted_weighted at 20%
  correlations.csv    Spearman correlation of expected p_e (random, 20%) with each classical measure
  explained_variance.csv  CV R^2 of expected p_e from structural / community measures
  network_curves.csv  classical network-level robustness (largest connected component, sampled
                      global efficiency) of the same damaged graphs, next to mean p_e

Classical edge measures: degree product, min degree, edge betweenness (exact), common neighbours,
Jaccard, bridge, same class (labels, analysis only), same community (the run's K-means context).
"How much of p_e do classical measures explain?": 5-fold cross-validated R^2 of a gradient-boosting
regressor of expected p_e on the classical measures (with and without the class label).
Expected p_e (share of control pairs scored lower, no draw noise) is used for per-edge analyses;
the drawn p_e for the distributions.

Example:
    python scripts/analyze_edge_stability.py --run artifacts/runs/<run> --out-dir artifacts/analysis/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHODS = ("diffusion", "clamp_only", "no_diffusion")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import networkx as nx
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    import torch

    from stability.edge_recovery_distribution import per_edge_probability
    from stability.experiment_config import load_config
    from stability.graph_data import load_graph

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--efficiency-sources", type=int, default=300)
    parser.add_argument("--efficiency-series", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.run / "config.toml")
    if config.recovery.edge_decision != "pairwise":
        raise ValueError("this analysis expects an edge_decision = 'pairwise' run")
    graph = load_graph(config.data.dataset, ROOT / config.data.root, False)
    split = np.load(args.run / "split.npz")
    counts = np.load(args.run / "recovery_counts.npz")
    summary = json.loads((args.run / "summary.json").read_text())
    context = torch.load(args.run / "checkpoints" / "community_context.pt", weights_only=False)
    communities = np.asarray(context["labels"])
    pool = split["pool"]
    fractions = list(config.perturbation.damage_fractions)
    K = config.recovery.reconstructions
    attacks = list(config.perturbation.attacks)
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- per-edge expected / drawn p_e
    def edge_values(attack):
        removed, recovered = counts[f"{attack}__removed"], counts[f"{attack}__recovered"]
        n_e, _, p_e = per_edge_probability(removed, recovered, K)
        draws = int(counts[f"{attack}__deterministic_draws"])
        expected = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for method in METHODS:
                denominator = n_e * (K if method == "diffusion" else draws)
                expected[method] = np.where(n_e > 0, counts[f"{attack}__{method}_expected_sum"].sum(axis=1) / denominator, np.nan)
            drawn = {"diffusion": p_e,
                     "clamp_only": np.where(n_e > 0, counts[f"{attack}__clamp_only"].sum(axis=1) / (n_e * draws), np.nan),
                     "no_diffusion": np.where(n_e > 0, counts[f"{attack}__no_diffusion"].sum(axis=1) / (n_e * draws), np.nan)}
        control = counts[f"{attack}__control_hits"].sum(axis=1) / (counts[f"{attack}__control_hits"].shape[1] * K)
        return n_e, drawn, expected, control

    values = {attack: edge_values(attack) for attack in attacks}

    # ---------------------------------------------------------------- overview + histograms
    overview, histograms = [], []
    bins = np.linspace(0, 1, 21)
    for attack in attacks:
        n_e, drawn, expected, control = values[attack]
        for level, fraction in enumerate(fractions):
            for method in METHODS:
                v = drawn[method][level][np.isfinite(drawn[method][level])]
                overview.append({"attack": attack, "damage": fraction, "method": method, "edges": int(v.size),
                                 "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                                 "q10": round(float(np.quantile(v, 0.1)), 4), "q90": round(float(np.quantile(v, 0.9)), 4)})
            overview.append({"attack": attack, "damage": fraction, "method": "control", "edges": int(control.shape[1]),
                             "mean": round(float(control[level].mean()), 4), "median": round(float(np.median(control[level])), 4),
                             "q10": round(float(np.quantile(control[level], 0.1)), 4),
                             "q90": round(float(np.quantile(control[level], 0.9)), 4)})
            v = drawn["diffusion"][level][np.isfinite(drawn["diffusion"][level])]
            h_edges, _ = np.histogram(v, bins=bins)
            h_control, _ = np.histogram(control[level], bins=bins)
            for b in range(len(bins) - 1):
                histograms.append({"attack": attack, "damage": fraction, "bin_low": round(bins[b], 2),
                                   "bin_high": round(bins[b + 1], 2),
                                   "removed_edges_share": round(h_edges[b] / v.size, 5),
                                   "control_share": round(h_control[b] / control.shape[1], 5)})
    _write_csv(args.out_dir / "overview.csv", overview)
    _write_csv(args.out_dir / "histograms.csv", histograms)
    print("overview done", flush=True)

    # ---------------------------------------------------------------- classical edge measures
    G = nx.Graph()
    G.add_nodes_from(range(graph.node_count))
    G.add_edges_from(map(tuple, graph.edges.tolist()))
    degree = np.array([G.degree(i) for i in range(graph.node_count)])
    betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    bridges = {tuple(sorted(e)) for e in nx.bridges(G)}
    print("betweenness done", flush=True)
    classes = graph.class_labels
    feature_rows = []
    n_e, _, expected, _ = values["random"]
    for index, (u, v) in enumerate(pool.tolist()):
        cn = len(list(nx.common_neighbors(G, u, v)))
        union = len(set(G[u]) | set(G[v])) - 2
        row = {
            "pool_index": index, "source": u, "target": v,
            "degree_product": int(degree[u] * degree[v]), "min_degree": int(min(degree[u], degree[v])),
            "edge_betweenness": float(betweenness.get((u, v), betweenness.get((v, u)))),
            "common_neighbours": cn, "jaccard": cn / union if union > 0 else 0.0,
            "bridge": int(tuple(sorted((u, v))) in bridges),
            "same_class": int(classes[u] == classes[v]), "same_community": int(communities[u] == communities[v]),
        }
        for level, fraction in enumerate(fractions):
            for method in METHODS:
                value = expected[method][level, index]
                row[f"expected_p_e_{method}_{int(fraction * 100)}"] = None if not np.isfinite(value) else round(float(value), 5)
        feature_rows.append(row)
    _write_csv(args.out_dir / "features.csv", feature_rows)

    level20 = fractions.index(0.2) if 0.2 in fractions else 0
    target = expected["diffusion"][level20]
    mask = np.isfinite(target)
    names = ["degree_product", "min_degree", "edge_betweenness", "common_neighbours", "jaccard", "bridge",
             "same_community", "same_class"]
    X_all = np.array([[r[name] for name in names] for r in feature_rows], dtype=np.float64)[mask]
    X_all[:, 0] = np.log1p(X_all[:, 0])
    y = target[mask]
    correlations = {name: round(float(spearmanr(X_all[:, i], y).statistic), 3) for i, name in enumerate(names)}
    model = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3)
    r2_without_class = cross_val_score(model, X_all[:, :-1], y, cv=5, scoring="r2")
    r2_with_class = cross_val_score(model, X_all, y, cv=5, scoring="r2")
    structural = [names.index(n) for n in ("degree_product", "min_degree", "edge_betweenness",
                                           "common_neighbours", "jaccard", "bridge")]
    explained = []
    for label, columns in (("structural", structural), ("community", [names.index("same_community")]),
                           ("structural_and_community", structural + [names.index("same_community")])):
        r2 = cross_val_score(model, X_all[:, columns], y, cv=5, scoring="r2")
        explained.append({"measures": label, "cv_r2_mean": round(float(r2.mean()), 4),
                          "unexplained": round(1 - float(r2.mean()), 4)})
    _write_csv(args.out_dir / "explained_variance.csv", explained)
    _write_csv(args.out_dir / "correlations.csv",
               [{"measure": name, "spearman": round(float(spearmanr(X_all[:, i], y).statistic), 4)}
                for i, name in enumerate(names)])
    feature_bins = []
    for i, name in enumerate(names):
        column = X_all[:, i]
        if name in ("bridge", "same_class", "same_community"):
            groups = [("0", column == 0), ("1", column == 1)]
        else:
            edges_q = np.unique(np.quantile(column, [0, .2, .4, .6, .8, 1]))
            index = np.clip(np.searchsorted(edges_q, column, side="right") - 1, 0, len(edges_q) - 2)
            groups = [(f"Q{q + 1} [{edges_q[q]:.3g}, {edges_q[q + 1]:.3g}]", index == q) for q in range(len(edges_q) - 1)]
        for label, g in groups:
            if g.any():
                feature_bins.append({"measure": name, "group": label, "edges": int(g.sum()),
                                     "mean_expected_p_e": round(float(y[g].mean()), 4)})
    _write_csv(args.out_dir / "feature_bins.csv", feature_bins)
    betweenness_col = X_all[:, names.index("edge_betweenness")]
    top_bc = betweenness_col >= np.quantile(betweenness_col, 0.9)
    low_pe = y <= np.quantile(y, 0.1)
    criticality = {
        "mean_expected_p_e_top10pct_betweenness": round(float(y[top_bc].mean()), 4),
        "mean_expected_p_e_other_edges": round(float(y[~top_bc].mean()), 4),
        "share_of_least_recoverable_10pct_that_are_top10pct_betweenness": round(float(top_bc[low_pe].mean()), 4),
        "bridges_mean_expected_p_e": round(float(y[X_all[:, names.index('bridge')] == 1].mean()), 4)
        if (X_all[:, names.index("bridge")] == 1).any() else None,
    }
    print("features done", flush=True)

    # ---------------------------------------------------------------- dynamics (paired, random)
    e = expected["diffusion"]
    all_levels = np.all(np.isfinite(e), axis=0)
    dynamics = []
    same = np.array([r["same_class"] for r in feature_rows], dtype=bool)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        change = e[b, all_levels] - e[a, all_levels]
        for label, g in (("all", np.ones(all_levels.sum(), bool)), ("same_class", same[all_levels]),
                         ("different_class", ~same[all_levels])):
            dynamics.append({"from": fractions[a], "to": fractions[b], "group": label, "edges": int(g.sum()),
                             "mean_change": round(float(change[g].mean()), 5),
                             "share_declining": round(float(np.mean(change[g] < 0)), 4),
                             "share_declining_by_more_than_0_05": round(float(np.mean(change[g] < -0.05)), 4)})
    _write_csv(args.out_dir / "dynamics.csv", dynamics)

    # ---------------------------------------------------------------- attacks (paired at 20%)
    targeted = values["targeted_weighted"][2]["diffusion"][level20]
    both = np.isfinite(target) & np.isfinite(targeted)
    attack_rows = [{"pool_index": int(i), "random": round(float(target[i]), 5), "targeted_weighted": round(float(targeted[i]), 5),
                    "degree_product": feature_rows[i]["degree_product"]} for i in np.flatnonzero(both)]
    _write_csv(args.out_dir / "attacks.csv", attack_rows)

    # ---------------------------------------------------------------- classical network-level curves
    series = np.load(args.run / "series.npz")
    removal_counts = series["removal_counts"]
    pool_keys = [tuple(sorted(p)) for p in pool.tolist()]
    sources = rng.choice(graph.node_count, size=min(args.efficiency_sources, graph.node_count), replace=False)

    def network_measures(H):
        lcc = max(len(c) for c in nx.connected_components(H)) / H.number_of_nodes()
        total = 0.0
        for s in sources:
            lengths = nx.single_source_shortest_path_length(H, int(s))
            total += sum(1.0 / d for d in lengths.values() if d > 0)
        efficiency = total / (len(sources) * (H.number_of_nodes() - 1))
        return lcc, efficiency

    base_lcc, base_eff = network_measures(G)
    curves = []
    for attack in ("random", "targeted_weighted"):
        orders = series[attack][: args.efficiency_series]
        _, drawn, _, _ = values[attack]
        curves.append({"attack": attack, "damage": 0.0, "lcc_fraction": round(base_lcc, 4),
                       "global_efficiency_relative": 1.0, "mean_p_e": None})
        for level, fraction in enumerate(fractions):
            lccs, effs = [], []
            for order in orders:
                H = G.copy()
                H.remove_edges_from(pool_keys[i] for i in order[: removal_counts[level]])
                lcc, eff = network_measures(H)
                lccs.append(lcc)
                effs.append(eff / base_eff)
            curves.append({"attack": attack, "damage": fraction, "lcc_fraction": round(float(np.mean(lccs)), 4),
                           "global_efficiency_relative": round(float(np.mean(effs)), 4),
                           "mean_p_e": round(float(np.nanmean(drawn["diffusion"][level])), 4)})
        print(f"curves {attack} done", flush=True)
    _write_csv(args.out_dir / "network_curves.csv", curves)

    analysis = {
        "run": str(args.run), "arguments": {k: str(v) for k, v in vars(args).items()},
        "validation_auc": {m: summary["validation_recovery"][m]["auc_mean_probability"] for m in METHODS},
        "spearman_expected_p_e_random_20pct": correlations,
        "cv_r2_classical_measures_without_class": [round(float(v), 4) for v in r2_without_class],
        "cv_r2_classical_measures_with_class": [round(float(v), 4) for v in r2_with_class],
        "explained_variance": explained,
        "criticality_vs_recoverability": criticality,
        "edges_evaluated_random_20pct": int(mask.sum()),
        "base_network": {"lcc_fraction": base_lcc, "efficiency_sources": len(sources)},
    }
    (args.out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
