from __future__ import annotations

import hashlib
import itertools
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from stability.partial_recovery import decode_without_diffusion, reconstruct_k
from stability.pilot_config import PilotConfig
from stability.proteins_common_fpr_replay import (
    load_saved_graph_models,
    select_method_threshold,
    summarize_at_threshold,
    summarize_attack_balanced,
)
from stability.synthetic_instances import SyntheticGraph
from stability.wp7_training import make_adjacency, train_graph_models


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "synthetic_pilot" / "sbm_ba_recovery_20261105"
REUSE_CHECKPOINT_ROOT: Path | None = None
CAP_SECONDS = 20 * 60
DAMAGE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
RECONSTRUCTIONS = 8
T_STAR = 7
FPR_CAP = 0.05
FAMILY_PLANS = {
    "SBM": {
        "calibration": (51001, 51002, 51003),
        "evaluation": (52001, 52002, 52003, 52004, 52005),
        "parameters": {
            "block_sizes": [12, 12],
            "within_block_probability": 0.35,
            "between_block_probability": 0.05,
        },
    },
    "BA": {
        "calibration": (61001, 61002, 61003),
        "evaluation": (62001, 62002, 62003, 62004, 62005),
        "parameters": {"node_count": 24, "attachment_parameter_m": 2},
    },
}
METHODS = ("partial_recovery", "no_diffusion_baseline")


@dataclass(frozen=True)
class PilotGraph:
    graph: SyntheticGraph
    family: str
    split: str
    initial_component_count: int
    initial_lcc_fraction: float
    graph_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _serialize_graph(graph: nx.Graph) -> bytes:
    edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges)
    return json.dumps(
        {"nodes": list(range(graph.number_of_nodes())), "edges": edges},
        separators=(",", ":"),
    ).encode("ascii")


def generate_pilot_graph(family: str, split: str, seed: int) -> PilotGraph:
    if family == "SBM":
        generated = nx.stochastic_block_model(
            FAMILY_PLANS["SBM"]["parameters"]["block_sizes"],
            [
                [0.35, 0.05],
                [0.05, 0.35],
            ],
            seed=seed,
        )
    elif family == "BA":
        generated = nx.barabasi_albert_graph(24, 2, seed=seed)
    else:
        raise ValueError(f"unsupported graph family: {family}")

    node_ids = tuple(f"{family}-{split}-{seed}:n{index:02d}" for index in range(24))
    edges = tuple(
        (node_ids[min(source, target)], node_ids[max(source, target)])
        for source, target in sorted(generated.edges())
    )
    positions = np.arange(24, dtype=np.float64)
    angles = 2.0 * np.pi * positions / 24.0
    features = torch.tensor(
        np.column_stack(
            (
                np.cos(angles),
                np.sin(angles),
                np.cos(2.0 * angles),
                np.sin(2.0 * angles),
            )
        ),
        dtype=torch.float32,
    )
    graph = SyntheticGraph(
        graph_id=f"{family}-{split}-{seed}",
        seed=seed,
        node_ids=node_ids,
        features=features,
        edges=edges,
    )
    component_sizes = [len(component) for component in nx.connected_components(generated)]
    return PilotGraph(
        graph=graph,
        family=family,
        split=split,
        initial_component_count=len(component_sizes),
        initial_lcc_fraction=max(component_sizes) / 24,
        graph_sha256=_sha256(_serialize_graph(generated)),
    )


def _removal_count(edge_count: int, fraction: float) -> int:
    return min(edge_count, int(np.floor(edge_count * fraction + 0.5)))


def make_case_plan(pilot_graph: PilotGraph, graph_index: int, split_code: int):
    graph = pilot_graph.graph
    node_order = {node: index for index, node in enumerate(graph.node_ids)}
    canonical_edges = tuple(
        sorted(
            graph.edges,
            key=lambda edge: (node_order[edge[0]], node_order[edge[1]]),
        )
    )
    degrees = {node: 0 for node in graph.node_ids}
    for source, target in canonical_edges:
        degrees[source] += 1
        degrees[target] += 1
    targeted = tuple(
        sorted(
            canonical_edges,
            key=lambda edge: (
                -(degrees[edge[0]] * degrees[edge[1]]),
                node_order[edge[0]],
                node_order[edge[1]],
            ),
        )
    )
    plans = []
    for level_index, fraction in enumerate(DAMAGE_FRACTIONS, start=1):
        removal_count = _removal_count(len(canonical_edges), fraction)
        base_seed = (
            8_000_000
            + split_code * 1_000_000
            + graph_index * 10_000
            + level_index * 100
        )
        masks = [
            {
                "attack_type": "targeted_degree_product",
                "mask_index": 0,
                "mask_seed": None,
                "removed_edges": targeted[:removal_count],
                "reconstruction_seed_base": base_seed,
            }
        ]
        used = {frozenset(edge) for edge in targeted[:removal_count]}
        random_seed_base = base_seed + 1
        for mask_index in range(8):
            mask_seed = random_seed_base + mask_index
            removed = tuple(
                sorted(
                    random.Random(mask_seed).sample(
                        canonical_edges, removal_count
                    ),
                    key=lambda edge: (node_order[edge[0]], node_order[edge[1]]),
                )
            )
            key = frozenset(frozenset(edge) for edge in removed)
            attempt = 0
            while key in used:
                attempt += 1
                if attempt > 10000:
                    raise RuntimeError(
                        f"could not select eight distinct masks for {graph.graph_id}"
                    )
                mask_seed += 8
                removed = tuple(
                    sorted(
                        random.Random(mask_seed).sample(
                            canonical_edges, removal_count
                        ),
                        key=lambda edge: (node_order[edge[0]], node_order[edge[1]]),
                    )
                )
                key = frozenset(frozenset(edge) for edge in removed)
            used.add(key)
            masks.append(
                {
                    "attack_type": "uniform_random",
                    "mask_index": mask_index,
                    "mask_seed": mask_seed,
                    "removed_edges": removed,
                    "reconstruction_seed_base": base_seed + 100 + mask_index * 10,
                }
            )
        for mask in masks:
            removed_keys = {frozenset(edge) for edge in mask["removed_edges"]}
            damaged_edges = tuple(
                edge
                for edge in canonical_edges
                if frozenset(edge) not in removed_keys
            )
            plans.append(
                {
                    **mask,
                    "damage_fraction_requested": fraction,
                    "damage_fraction_realized": removal_count / len(canonical_edges),
                    "edges_removed": removal_count,
                    "damaged_edges": damaged_edges,
                }
            )
    return plans


def _candidate_pairs(node_ids):
    return tuple(itertools.combinations(node_ids, 2))


def _prepare_threshold_graph(pilot_graph, case_results):
    graph = pilot_graph.graph
    candidates = _candidate_pairs(graph.node_ids)
    original = {frozenset(edge) for edge in graph.edges}
    nonedges = np.asarray(
        [i for i, edge in enumerate(candidates) if frozenset(edge) not in original],
        dtype=np.int64,
    )
    candidate_index = {frozenset(edge): i for i, edge in enumerate(candidates)}
    cases = []
    for case in case_results:
        removed = tuple(tuple(edge) for edge in case["removed_edges"])
        cases.append(
            {
                **{
                    key: case[key]
                    for key in (
                        "attack_type",
                        "damage_fraction_requested",
                        "damage_fraction_realized",
                        "edges_removed",
                        "mask_index",
                        "mask_seed",
                        "reconstruction_seed_base",
                        "removed_edges",
                    )
                },
                "candidate_edges": candidates,
                "removed_indices": np.asarray(
                    [candidate_index[frozenset(edge)] for edge in removed],
                    dtype=np.int64,
                ),
                "nonedge_indices": nonedges,
                "baseline_probabilities": np.asarray(
                    case["baseline_probabilities"], dtype=np.float64
                ),
                "partial_probabilities": np.asarray(
                    case["partial_probabilities"], dtype=np.float64
                ),
            }
        )
    return {
        "graph_id": graph.graph_id,
        "node_ids": graph.node_ids,
        "original_edges": graph.edges,
        "cases": cases,
    }


def _prediction_metrics(probability_rows, case, tau):
    graph = case["graph"]
    original = {frozenset(edge) for edge in graph.edges}
    removed = {frozenset(edge) for edge in case["removed_edges"]}
    candidates = case["candidate_edges"]
    nonedges = {
        frozenset(edge)
        for edge in candidates
        if frozenset(edge) not in original
    }
    rows = []
    for probabilities in probability_rows:
        predicted = {
            frozenset(edge)
            for edge, probability in zip(candidates, probabilities)
            if probability >= tau
        }
        tp = len(predicted & removed)
        fp = len(predicted & nonedges)
        rows.append(
            {
                "recovered_removed_edges": tp,
                "book_stability_score": tp / len(removed),
                "absolute_false_positive_edges": fp,
                "realized_fpr": fp / len(nonedges),
            }
        )
    return rows


def _bootstrap_case_group(mask_samples, rng, replicates=5000):
    values = np.asarray(mask_samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != RECONSTRUCTIONS:
        raise ValueError("sampling uncertainty requires masks x eight draws")
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample_indices = rng.integers(
            0, RECONSTRUCTIONS, size=(len(values), RECONSTRUCTIONS)
        )
        draws[index] = np.take_along_axis(
            values, sample_indices, axis=1
        ).mean()
    quantiles = np.quantile(draws, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "sampling_ci95": [float(quantiles[0]), float(quantiles[1])],
        "scope": "conditional on these graphs and fixed masks; resamples reconstruction draws only",
    }


def _evaluation_sampling_summary(graph_results, thresholds, graph_lookup, seed):
    rng = np.random.default_rng(seed)
    grouped = {}
    for graph_result in graph_results:
        graph_id = graph_result["graph_id"]
        graph = graph_lookup[graph_id].graph
        candidates = _candidate_pairs(graph.node_ids)
        for case in graph_result["cases"]:
            key = (
                graph_id,
                case["attack_type"],
                case["damage_fraction_requested"],
            )
            group = grouped.setdefault(
                key,
                {
                    method: {"score": [], "fp": [], "fpr": []}
                    for method in METHODS
                },
            )
            for method in METHODS:
                probs = (
                    case["partial_probabilities"]
                    if method == "partial_recovery"
                    else np.asarray(
                        case["baseline_probabilities"], dtype=np.float64
                    )[None, :]
                )
                rows = _prediction_metrics(
                    probs,
                    {
                        "graph": graph,
                        "removed_edges": case["removed_edges"],
                        "candidate_edges": candidates,
                    },
                    thresholds[method],
                )
                if method == "partial_recovery":
                    group[method]["score"].append(
                        [row["book_stability_score"] for row in rows]
                    )
                    group[method]["fp"].append(
                        [row["absolute_false_positive_edges"] for row in rows]
                    )
                    group[method]["fpr"].append(
                        [row["realized_fpr"] for row in rows]
                    )
                else:
                    group[method]["score"].append(
                        [rows[0]["book_stability_score"]] * RECONSTRUCTIONS
                    )
                    group[method]["fp"].append(
                        [rows[0]["absolute_false_positive_edges"]] * RECONSTRUCTIONS
                    )
                    group[method]["fpr"].append(
                        [rows[0]["realized_fpr"]] * RECONSTRUCTIONS
                    )

    summaries = []
    for (graph_id, attack, fraction), methods in sorted(grouped.items()):
        row = {
            "graph_id": graph_id,
            "attack_type": attack,
            "damage_fraction_requested": fraction,
            "method_metrics": {},
        }
        for method in METHODS:
            row["method_metrics"][method] = {}
            for metric, samples in methods[method].items():
                row["method_metrics"][method][metric] = (
                    _bootstrap_case_group(samples, rng)
                    if method == "partial_recovery"
                    else {
                        "mean": float(np.asarray(samples).mean()),
                        "sampling_ci95": None,
                        "scope": "deterministic encode-decode output; no reconstruction sampling",
                    }
                )
        summaries.append(row)
    return summaries


def _jsonable_case_plan(plans):
    return [
        {
            **{
                key: value
                for key, value in case.items()
                if key not in {"removed_edges", "damaged_edges"}
            },
            "removed_edges": [list(edge) for edge in case["removed_edges"]],
            "damaged_edges": [list(edge) for edge in case["damaged_edges"]],
        }
        for case in plans
    ]


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as artifact:
        json.dump(payload, artifact, indent=2, sort_keys=True, allow_nan=False)
        artifact.write("\n")


def _threshold_tradeoff(graph_results, method):
    from stability.proteins_common_fpr_replay import _event_curves

    taus, scores, fprs, fp_counts = _event_curves(graph_results, method)
    rows = [
        {
            "tau": float(tau),
            "calibration_book_stability_score": float(score),
            "calibration_mean_fpr": float(fpr),
            "calibration_absolute_false_positive_edges": float(fp),
            "feasible_under_5_percent_cap": bool(fpr <= FPR_CAP),
        }
        for tau, score, fpr, fp in zip(taus, scores, fprs, fp_counts)
    ]
    feasible = [row for row in rows if row["feasible_under_5_percent_cap"]]
    best_under_cap = max(
        feasible,
        key=lambda row: (
            row["calibration_book_stability_score"],
            -row["calibration_mean_fpr"],
            row["tau"],
        ),
        default=None,
    )
    first_nonzero = next(
        (
            row
            for row in rows
            if row["calibration_book_stability_score"] > 0.0
        ),
        None,
    )
    return {
        "best_threshold_under_fpr_cap": best_under_cap,
        "first_nonzero_recovery_threshold": first_nonzero,
        "tradeoff_curve": rows,
    }


def _train_and_score(
    pilot_graph,
    plans,
    model_seed,
    checkpoint_dir,
    deadline,
):
    graph = pilot_graph.graph
    config = PilotConfig(curvature=1.0, tau=0.0)
    saved_checkpoint_dir = None
    if REUSE_CHECKPOINT_ROOT is not None:
        candidate_dir = (
            REUSE_CHECKPOINT_ROOT
            / pilot_graph.family
            / pilot_graph.split
        )
        if (
            (candidate_dir / graph.graph_id / "autoencoder.pt").is_file()
            and (candidate_dir / graph.graph_id / "denoiser.pt").is_file()
        ):
            saved_checkpoint_dir = candidate_dir

    if saved_checkpoint_dir is None:
        encoder, denoiser, _context, _original_adjacency, training = train_graph_models(
            graph, config, model_seed, checkpoint_dir
        )
    else:
        encoder, denoiser, _context = load_saved_graph_models(
            graph,
            saved_checkpoint_dir,
            config,
            model_seed,
        )
        source_graph_dir = saved_checkpoint_dir / graph.graph_id
        output_graph_dir = checkpoint_dir / graph.graph_id
        output_graph_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint_name in ("autoencoder.pt", "denoiser.pt"):
            shutil.copy2(
                source_graph_dir / checkpoint_name,
                output_graph_dir / checkpoint_name,
            )
        training = {
            "checkpoint_reused": True,
            "checkpoint_source": str(source_graph_dir),
            "autoencoder_checkpoint_sha256": _sha256(
                (output_graph_dir / "autoencoder.pt").read_bytes()
            ),
            "denoiser_checkpoint_sha256": _sha256(
                (output_graph_dir / "denoiser.pt").read_bytes()
            ),
        }
    candidates = _candidate_pairs(graph.node_ids)
    node_order = {node: index for index, node in enumerate(graph.node_ids)}
    original = {frozenset(edge) for edge in graph.edges}
    nonedge_indices = np.asarray(
        [i for i, edge in enumerate(candidates) if frozenset(edge) not in original],
        dtype=np.int64,
    )
    candidate_index = {frozenset(edge): i for i, edge in enumerate(candidates)}
    results = []
    for plan in plans:
        if time.perf_counter() >= deadline:
            raise TimeoutError("20-minute pilot CPU budget reached between masks")
        damaged_edges = tuple(plan["damaged_edges"])
        removed_edges = tuple(plan["removed_edges"])
        adjacency = make_adjacency(
            len(graph.node_ids),
            tuple(
                (node_order[source], node_order[target])
                for source, target in damaged_edges
            ),
        )
        with torch.no_grad():
            damaged_embeddings = encoder.encode(graph.features, adjacency)[2]
            _, baseline_probabilities = decode_without_diffusion(
                encoder, damaged_embeddings, graph.node_ids, candidates, tau=0.0
            )
        batch = reconstruct_k(
            encoder=encoder,
            denoiser=denoiser,
            damaged_embeddings=damaged_embeddings,
            damaged_adjacency=adjacency,
            context=_context,
            node_ids=graph.node_ids,
            candidate_edges=candidates,
            t_star=T_STAR,
            num_reconstructions=RECONSTRUCTIONS,
            seed=int(plan["reconstruction_seed_base"]),
            config=config,
        )
        if any(not torch.isfinite(row).all() for row in batch.probabilities):
            raise RuntimeError(f"{graph.graph_id}: non-finite partial probabilities")
        if not torch.isfinite(baseline_probabilities).all():
            raise RuntimeError(f"{graph.graph_id}: non-finite baseline probabilities")
        results.append(
            {
                **{
                    key: plan[key]
                    for key in (
                        "attack_type",
                        "damage_fraction_requested",
                        "damage_fraction_realized",
                        "edges_removed",
                        "mask_index",
                        "mask_seed",
                        "reconstruction_seed_base",
                        "removed_edges",
                    )
                },
                "candidate_edges": [list(edge) for edge in candidates],
                "removed_indices": [
                    candidate_index[frozenset(edge)] for edge in removed_edges
                ],
                "nonedge_indices": nonedge_indices.tolist(),
                "baseline_probabilities": baseline_probabilities.cpu().tolist(),
                "partial_probabilities": [
                    row.cpu().tolist() for row in batch.probabilities
                ],
            }
        )
    return {
        "graph_id": graph.graph_id,
        "family": pilot_graph.family,
        "split": pilot_graph.split,
        "seed": graph.seed,
        "graph_sha256": pilot_graph.graph_sha256,
        "initial_component_count": pilot_graph.initial_component_count,
        "initial_lcc_fraction": pilot_graph.initial_lcc_fraction,
        "node_ids": list(graph.node_ids),
        "features": graph.features.tolist(),
        "original_edges": [list(edge) for edge in graph.edges],
        "training": training,
        "cases": results,
    }


def run_pilot():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite pilot directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    deadline = started + CAP_SECONDS
    version_info = {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "networkx": nx.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }
    all_graphs = {}
    plans_by_graph = {}
    model_seeds = {}
    manifest = {
        "artifact_type": "paired SBM and BA recovery pilot selection and mask plan",
        "status": "predefined_before_outcome_inspection",
        "graph_definitions": FAMILY_PLANS,
        "features": (
            "four deterministic node-index features [cos(2pi*i/24), "
            "sin(2pi*i/24), cos(4pi*i/24), sin(4pi*i/24)]; no SBM block labels"
        ),
        "attack_protocol": {
            "damage_fractions": DAMAGE_FRACTIONS,
            "targeted": "descending intact-graph degree(u)*degree(v), then ascending canonical node-index endpoint pair",
            "random": "eight distinct uniform edge subsets without replacement per graph and level",
            "removal_count_rounding": "floor(p*m + 0.5), half-up",
            "candidate_pairs": "all unordered node pairs in canonical node order, identical for both methods",
            "retain_disconnected_sbm_graphs": True,
        },
        "methods": {
            "partial_recovery": "one graph-specific trained encoder and denoiser; t*=7; eight samples per mask",
            "no_diffusion_baseline": "encode the same damaged input with that graph's trained encoder and decode without diffusion",
            "paired_inputs": "both methods use the identical masks and candidate-pair order for each graph",
            "calibration": "method-specific threshold per family selected using calibration graphs only; maximize attack-balanced book score subject to mean FPR <= 0.05; equal-probability ties included as a group",
            "evaluation": "freeze each method threshold before evaluation; if either method has no nonzero recovery threshold under the cap, report both calibration trade-offs and skip both methods for that family",
        },
        "training": {
            "autoencoder_epochs": 30,
            "denoiser_epochs": 50,
            "curvature": 1.0,
            "schedule_length": 200,
            "t_star": T_STAR,
            "reconstructions_per_mask": RECONSTRUCTIONS,
            "same_graph_model_for_both_methods": True,
        },
        "cpu_budget_seconds": CAP_SECONDS,
        "software_versions": version_info,
        "families": {},
    }
    for family, family_plan in FAMILY_PLANS.items():
        family_payload = {}
        family_graphs = []
        for split_code, split in enumerate(("calibration", "evaluation"), start=1):
            seeds = family_plan[split]
            graphs = [
                generate_pilot_graph(family, split, seed) for seed in seeds
            ]
            if len({graph.graph_sha256 for graph in graphs}) != len(graphs):
                raise RuntimeError(f"{family} {split} graph seeds produced duplicates")
            family_payload[split] = [
                {
                    "graph_id": item.graph.graph_id,
                    "seed": item.graph.seed,
                    "graph_sha256": item.graph_sha256,
                    "node_count": len(item.graph.node_ids),
                    "edge_count": len(item.graph.edges),
                    "initial_component_count": item.initial_component_count,
                    "initial_lcc_fraction": item.initial_lcc_fraction,
                }
                for item in graphs
            ]
            for graph_index, item in enumerate(graphs):
                graph_id = item.graph.graph_id
                all_graphs[graph_id] = item
                family_graphs.append(item)
                plans_by_graph[graph_id] = make_case_plan(
                    item, graph_index, split_code
                )
                model_seeds[graph_id] = item.graph.seed + 3_000_000
        manifest["families"][family] = family_payload
        manifest["families"][family]["attack_masks"] = {
            item.graph.graph_id: _jsonable_case_plan(
                plans_by_graph[item.graph.graph_id]
            )
            for item in family_graphs
        }
        manifest["families"][family]["model_seeds"] = {
            item.graph.graph_id: model_seeds[item.graph.graph_id]
            for item in family_graphs
        }
    manifest_path = OUTPUT_DIR / "selection_and_masks.json"
    _save(manifest_path, manifest)
    manifest_hash = _sha256(manifest_path.read_bytes())

    family_results = {}
    for family in FAMILY_PLANS:
        calibration_results = []
        calibration_dir = OUTPUT_DIR / "models" / family / "calibration"
        try:
            for graph in all_graphs.values():
                if graph.family != family or graph.split != "calibration":
                    continue
                if time.perf_counter() >= deadline:
                    raise TimeoutError("20-minute pilot CPU budget reached before graph")
                model_seed = model_seeds[graph.graph.graph_id]
                result = _train_and_score(
                    graph,
                    plans_by_graph[graph.graph.graph_id],
                    model_seed,
                    calibration_dir,
                    deadline,
                )
                calibration_results.append(result)
                _save(
                    OUTPUT_DIR / "work_in_progress.json",
                    {
                        "status": "running",
                        "completed_calibration_graphs": [
                            item["graph_id"] for item in calibration_results
                        ],
                        "current_family": family,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
            threshold_graphs = [
                _prepare_threshold_graph(
                    all_graphs[item["graph_id"]], item["cases"]
                )
                for item in calibration_results
            ]
            threshold_results = {
                method: select_method_threshold(threshold_graphs, method)
                for method in METHODS
            }
            tradeoffs = {
                method: _threshold_tradeoff(threshold_graphs, method)
                for method in METHODS
            }
            qualifies = all(
                threshold_results[method]["status"] == "threshold_selected"
                and threshold_results[method].get(
                    "nonzero_recovery_feasible", False
                )
                for method in METHODS
            )
            calibration_scoring = None
            available_thresholds = {
                method: float(threshold_results[method]["tau"])
                for method in METHODS
                if threshold_results[method].get("tau") is not None
            }
            if available_thresholds:
                calibration_rows = summarize_at_threshold(
                    threshold_graphs, available_thresholds
                )
                calibration_scoring = {
                    "thresholds_used": available_thresholds,
                    "per_graph_attack_damage_results": calibration_rows,
                    "attack_balanced_aggregate": summarize_attack_balanced(
                        calibration_rows
                    ),
                    "interpretation": (
                        "calibration-only diagnostics at selected feasible "
                        "thresholds; not evaluation results"
                    ),
                }
            calibration_payload = {
                "artifact_type": f"{family} paired recovery calibration",
                "family": family,
                "status": "calibration_complete",
                "thresholds_frozen_before_evaluation": qualifies,
                "selection_manifest_sha256": manifest_hash,
                "method_thresholds": threshold_results,
                "method_tradeoffs": tradeoffs,
                "evaluation_gate_passed_for_both_methods": qualifies,
                "calibration_metrics_at_available_thresholds": calibration_scoring,
                "calibration_graphs": [
                    {
                        "graph_id": result["graph_id"],
                        "seed": result["seed"],
                        "graph_sha256": result["graph_sha256"],
                        "initial_component_count": result[
                            "initial_component_count"
                        ],
                        "initial_lcc_fraction": result["initial_lcc_fraction"],
                    }
                    for result in calibration_results
                ],
                "calibration_duration_seconds": time.perf_counter() - started,
            }
            _save(
                OUTPUT_DIR / family / "calibration.json",
                calibration_payload,
            )
            family_results[family] = {
                "calibration": calibration_payload,
                "calibration_internal": calibration_results,
            }
            if not qualifies:
                family_results[family]["evaluation_status"] = (
                    "skipped_because_either_method_failed_calibration_gate"
                )
                continue

            frozen_thresholds = {
                method: float(threshold_results[method]["tau"])
                for method in METHODS
            }
            evaluation_results = []
            evaluation_dir = OUTPUT_DIR / "models" / family / "evaluation"
            for graph in all_graphs.values():
                if graph.family != family or graph.split != "evaluation":
                    continue
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        "20-minute pilot CPU budget reached before evaluation graph"
                    )
                result = _train_and_score(
                    graph,
                    plans_by_graph[graph.graph.graph_id],
                    model_seeds[graph.graph.graph_id],
                    evaluation_dir,
                    deadline,
                )
                evaluation_results.append(result)
                _save(
                    OUTPUT_DIR / "work_in_progress.json",
                    {
                        "status": "running",
                        "completed_calibration_graphs": [
                            item["graph_id"] for item in calibration_results
                        ],
                        "completed_evaluation_graphs": [
                            item["graph_id"] for item in evaluation_results
                        ],
                        "current_family": family,
                        "frozen_thresholds": frozen_thresholds,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
            evaluation_threshold_graphs = [
                _prepare_threshold_graph(
                    all_graphs[item["graph_id"]], item["cases"]
                )
                for item in evaluation_results
            ]
            rows = summarize_at_threshold(
                evaluation_threshold_graphs, frozen_thresholds
            )
            aggregate = summarize_attack_balanced(rows)
            graph_lookup = {
                graph_id: item
                for graph_id, item in all_graphs.items()
                if item.family == family and item.split == "evaluation"
            }
            sampling = _evaluation_sampling_summary(
                evaluation_results,
                frozen_thresholds,
                graph_lookup,
                seed=family_plan["evaluation"][0],
            )
            evaluation_payload = {
                "artifact_type": f"{family} paired recovery evaluation",
                "family": family,
                "status": "evaluation_complete",
                "thresholds_frozen_before_evaluation": True,
                "selection_manifest_sha256": manifest_hash,
                "calibration_artifact": f"artifacts/synthetic_pilot/sbm_ba_recovery_20261105/{family}/calibration.json",
                "method_thresholds": frozen_thresholds,
                "attack_balanced_aggregate": aggregate,
                "per_graph_attack_damage_results": rows,
                "reconstruction_sampling_uncertainty": sampling,
                "evaluation_graphs": [
                    {
                        "graph_id": result["graph_id"],
                        "seed": result["seed"],
                        "graph_sha256": result["graph_sha256"],
                        "initial_component_count": result[
                            "initial_component_count"
                        ],
                        "initial_lcc_fraction": result["initial_lcc_fraction"],
                    }
                    for result in evaluation_results
                ],
                "uncertainty_scope": (
                    "intervals condition on the tested evaluation graphs and fixed masks; "
                    "only the eight reconstruction draws are resampled"
                ),
                "evaluation_duration_seconds": time.perf_counter() - started,
            }
            _save(
                OUTPUT_DIR / family / "evaluation.json",
                evaluation_payload,
            )
            family_results[family]["evaluation"] = evaluation_payload
            family_results[family]["evaluation_internal"] = evaluation_results
        except TimeoutError as error:
            _save(
                OUTPUT_DIR / "work_in_progress.json",
                {
                    "status": "incomplete_cpu_budget_exceeded",
                    "reason": str(error),
                    "elapsed_seconds": time.perf_counter() - started,
                    "current_family": family,
                    "completed_families": sorted(family_results),
                },
            )
            break
    if time.perf_counter() >= deadline:
        status = "incomplete_cpu_budget_exceeded"
    elif len(family_results) == len(FAMILY_PLANS):
        status = "complete"
    else:
        status = "incomplete"
    _save(
        OUTPUT_DIR / "summary.json",
        {
            "artifact_type": "SBM and BA paired recovery pilot summary",
            "status": status,
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_budget_seconds": CAP_SECONDS,
            "families": {
                family: {
                    "calibration_status": data["calibration"]["status"],
                    "evaluation_status": data.get(
                        "evaluation_status",
                        data.get("evaluation", {}).get("status", "not_run"),
                    ),
                    "method_thresholds": data["calibration"][
                        "method_thresholds"
                    ],
                    "evaluation_aggregate": data.get("evaluation", {}).get(
                        "attack_balanced_aggregate"
                    ),
                }
                for family, data in family_results.items()
            },
            "preservation": (
                "PROTEINS artifacts were not read or modified by this pilot"
            ),
        },
    )
    return status
