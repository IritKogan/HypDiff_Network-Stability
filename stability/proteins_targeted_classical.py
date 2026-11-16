from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import networkx as nx
import numpy as np
import torch

from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.partial_recovery import CommunityContext, reconstruct_k
from stability.pilot_config import PilotConfig
from stability.scoring import score_recovery
from stability.wp7_training import make_adjacency, model_args


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ARTIFACT = (
    ROOT
    / "artifacts"
    / "proteins_pilot"
    / "evaluation_stable_clustering"
    / "evaluation.json"
)
OUTPUT_DIR = ROOT / "artifacts" / "proteins_pilot" / "targeted_attack_classical_comparison"
OUTPUT_FILE = OUTPUT_DIR / "comparison.json"
DAMAGE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
RECONSTRUCTIONS = 8
SEED_LEVEL_STRIDE = 1000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def targeted_edge_order(node_ids, original_edges):
    node_order = {node: index for index, node in enumerate(node_ids)}
    degree = dict.fromkeys(node_ids, 0)
    canonical_edges = []
    seen = set()
    for source, target in original_edges:
        if source not in node_order or target not in node_order or source == target:
            raise ValueError(f"invalid edge {(source, target)!r}")
        pair = (
            (source, target)
            if node_order[source] < node_order[target]
            else (target, source)
        )
        key = frozenset(pair)
        if key in seen:
            raise ValueError(f"duplicate undirected edge: {pair!r}")
        seen.add(key)
        canonical_edges.append(pair)
        degree[source] += 1
        degree[target] += 1
    canonical_edges.sort(key=lambda edge: (node_order[edge[0]], node_order[edge[1]]))
    return tuple(
        sorted(
            canonical_edges,
            key=lambda edge: (
                -(degree[edge[0]] * degree[edge[1]]),
                node_order[edge[0]],
                node_order[edge[1]],
            ),
        )
    ), degree


def rounded_removal_count(edge_count: int, fraction: float) -> int:
    if edge_count < 0 or not 0.0 <= fraction <= 1.0:
        raise ValueError("edge_count must be nonnegative and fraction in [0, 1]")
    return math.floor(edge_count * fraction + 0.5)


def graph_classical_metrics(node_ids, edges):
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edges)
    connected = nx.is_connected(graph) if graph.number_of_nodes() else False
    if connected and graph.number_of_nodes() >= 2:
        laplacian = nx.laplacian_matrix(
            graph, nodelist=list(node_ids), weight=None
        ).toarray()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        algebraic_connectivity = max(0.0, float(eigenvalues[1]))
    else:
        algebraic_connectivity = 0.0
    betweenness = nx.betweenness_centrality(
        graph, normalized=True, endpoints=False, weight=None
    )
    mean_betweenness = (
        float(sum(betweenness[node] for node in node_ids) / len(node_ids))
        if node_ids
        else 0.0
    )
    return {
        "connected": connected,
        "algebraic_connectivity_lambda2": algebraic_connectivity,
        "mean_normalized_node_betweenness": mean_betweenness,
    }


def _indexed_edges(edges, node_ids):
    index = {node: position for position, node in enumerate(node_ids)}
    return tuple((index[source], index[target]) for source, target in edges)


def _load_graph_models(record, checkpoint_dir: Path, config: PilotConfig):
    graph = SimpleNamespace(
        graph_id=record["graph_id"],
        features=torch.tensor(record["features"], dtype=torch.float32),
        node_ids=tuple(record["node_ids"]),
        edges=tuple(tuple(edge) for edge in record["original_edges"]),
    )
    args = model_args(graph, config)
    encoder = StabilityLPModel(args).cpu()
    denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.curvature).cpu()
    encoder_path = checkpoint_dir / graph.graph_id / "autoencoder.pt"
    denoiser_path = checkpoint_dir / graph.graph_id / "denoiser.pt"
    if not encoder_path.is_file() or not denoiser_path.is_file():
        raise FileNotFoundError(f"frozen checkpoints are missing for {graph.graph_id}")
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True)
    )
    denoiser.load_state_dict(
        torch.load(denoiser_path, map_location="cpu", weights_only=True)
    )
    encoder.eval()
    denoiser.eval()
    expected_encoder_hash = record["training"]["autoencoder_checkpoint_sha256"]
    expected_denoiser_hash = record["training"]["denoiser_checkpoint_sha256"]
    if _state_hash(encoder) != expected_encoder_hash:
        raise RuntimeError(f"autoencoder checkpoint hash mismatch for {graph.graph_id}")
    if _state_hash(denoiser) != expected_denoiser_hash:
        raise RuntimeError(f"denoiser checkpoint hash mismatch for {graph.graph_id}")
    return graph, encoder, denoiser, encoder_path, denoiser_path


def _run_graph(record, checkpoint_dir: Path, config: PilotConfig):
    graph, encoder, denoiser, encoder_path, denoiser_path = _load_graph_models(
        record, checkpoint_dir, config
    )
    node_ids = graph.node_ids
    feature_rows = graph.features
    original_edges = graph.edges
    ordered_edges, intact_degree = targeted_edge_order(node_ids, original_edges)
    original_classical = graph_classical_metrics(node_ids, original_edges)
    candidate_edges = tuple(itertools.combinations(node_ids, 2))
    candidate_pairs = _indexed_edges(candidate_edges, node_ids)
    original_adjacency = make_adjacency(
        len(node_ids), _indexed_edges(original_edges, node_ids)
    )
    context = CommunityContext(
        node_ids=node_ids,
        labels=torch.tensor(record["community_context"]["labels"], dtype=torch.long),
        centers=torch.tensor(
            record["community_context"]["centers"], dtype=feature_rows.dtype
        ),
        curvature=float(record["community_context"]["curvature"]),
    )
    with torch.no_grad():
        original_embeddings = encoder.encode(feature_rows, original_adjacency)[2]
    if not torch.isfinite(original_embeddings).all():
        raise RuntimeError(f"{graph.graph_id}: intact embeddings are non-finite")

    level_results = []
    for level_index, fraction in enumerate(DAMAGE_FRACTIONS):
        removal_count = rounded_removal_count(len(original_edges), fraction)
        removed_edges = ordered_edges[:removal_count]
        removed_keys = {frozenset(edge) for edge in removed_edges}
        damaged_edges = tuple(
            edge for edge in original_edges if frozenset(edge) not in removed_keys
        )
        damaged_adjacency = make_adjacency(
            len(node_ids), _indexed_edges(damaged_edges, node_ids)
        )
        with torch.no_grad():
            damaged_embeddings = encoder.encode(feature_rows, damaged_adjacency)[2]
        if not torch.isfinite(damaged_embeddings).all():
            raise RuntimeError(
                f"{graph.graph_id}: damaged embeddings non-finite at {fraction:.0%}"
            )
        reconstruction_seed = (
            int(record["reconstruction_seed_base"])
            + (level_index + 1) * SEED_LEVEL_STRIDE
        )
        batch = reconstruct_k(
            encoder=encoder,
            denoiser=denoiser,
            damaged_embeddings=damaged_embeddings,
            damaged_adjacency=damaged_adjacency,
            context=context,
            node_ids=node_ids,
            candidate_edges=candidate_edges,
            t_star=int(record["partial_recovery"]["t_star"]),
            num_reconstructions=RECONSTRUCTIONS,
            seed=reconstruction_seed,
            config=config,
        )
        score = (
            score_recovery(removed_edges, batch.edge_sets).scenario_score
            if removed_edges
            else None
        )
        damaged_classical = graph_classical_metrics(node_ids, damaged_edges)
        level_results.append(
            {
                "damage_fraction_requested": fraction,
                "edges_removed": removal_count,
                "edge_count_original": len(original_edges),
                "damage_fraction_realized": (
                    removal_count / len(original_edges) if original_edges else 0.0
                ),
                "removed_edges": [list(edge) for edge in removed_edges],
                "perturbation_is_deterministic_nested_prefix": True,
                "reconstruction_seed_base": reconstruction_seed,
                "reconstruction_seeds": [
                    reconstruction_seed + index
                    for index in range(RECONSTRUCTIONS)
                ],
                "reconstruction_count": RECONSTRUCTIONS,
                "book_stability_score": score,
                "edge_recovery_frequencies": (
                    [
                        {"edge": list(edge), "frequency": frequency}
                        for edge, frequency in score_recovery(
                            removed_edges, batch.edge_sets
                        ).edge_recovery_frequencies.items()
                    ]
                    if removed_edges
                    else []
                ),
                "distinct_thresholded_reconstructions": len(
                    {
                        frozenset(frozenset(edge) for edge in edges)
                        for edges in batch.edge_sets
                    }
                ),
                "graph_disconnected": not damaged_classical["connected"],
                "algebraic_connectivity_lambda2": damaged_classical[
                    "algebraic_connectivity_lambda2"
                ],
                "mean_normalized_node_betweenness": damaged_classical[
                    "mean_normalized_node_betweenness"
                ],
            }
        )
    return {
        "graph_id": graph.graph_id,
        "dataset_index": record["dataset_index"],
        "content_sha256": record["content_sha256"],
        "node_count": len(node_ids),
        "original_edge_count": len(original_edges),
        "intact_edge_degree_product_order": [
            {
                "edge": list(edge),
                "degree_product": intact_degree[edge[0]] * intact_degree[edge[1]],
            }
            for edge in ordered_edges
        ],
        "intact_reference": {
            "connected": original_classical["connected"],
            "algebraic_connectivity_lambda2": original_classical[
                "algebraic_connectivity_lambda2"
            ],
            "mean_normalized_node_betweenness": original_classical[
                "mean_normalized_node_betweenness"
            ],
        },
        "checkpoints": {
            "autoencoder": str(encoder_path.relative_to(ROOT)),
            "autoencoder_state_sha256": record["training"][
                "autoencoder_checkpoint_sha256"
            ],
            "denoiser": str(denoiser_path.relative_to(ROOT)),
            "denoiser_state_sha256": record["training"][
                "denoiser_checkpoint_sha256"
            ],
        },
        "damage_levels": level_results,
    }


def run_comparison(evaluation: dict, checkpoint_dir: Path):
    if evaluation.get("status") != "evaluation_complete":
        raise ValueError("targeted comparison requires completed official evaluation")
    if float(evaluation["frozen_tau"]) != 1.0:
        raise ValueError("this approved comparison requires the frozen tau=1.0")
    if len(evaluation["evaluation_graphs"]) != 5:
        raise ValueError("expected the five approved PROTEINS evaluation graphs")
    config = PilotConfig(curvature=1.0, tau=float(evaluation["frozen_tau"]))
    torch.set_num_threads(1)
    started = time.perf_counter()
    graphs = [
        _run_graph(record, checkpoint_dir, config)
        for record in evaluation["evaluation_graphs"]
    ]
    return {
        "artifact_type": "targeted-attack classical stability comparison",
        "status": "complete",
        "study_scope": (
            "post-hoc comparison on five existing PROTEINS evaluation graphs; "
            "one deterministic targeted mask per graph and damage level"
        ),
        "method": {
            "attack": "rank intact-graph edges by degree(u) * degree(v), descending",
            "tie_break": "canonical endpoint pairs in fixed node order, ascending",
            "damage_fractions": list(DAMAGE_FRACTIONS),
            "rounding": "floor(p * original_edge_count + 0.5), nearest integer, half up",
            "nested_masks": True,
            "book_stability_score": (
                "mean removed-edge recovery frequency over K sampled reconstructions"
            ),
            "classical_metrics": {
                "algebraic_connectivity": (
                    "second-smallest eigenvalue of the unweighted combinatorial "
                    "Laplacian over all original nodes; zero when disconnected"
                ),
                "mean_node_betweenness": (
                    "arithmetic mean of normalized node betweenness over all "
                    "original nodes, endpoints=False, unweighted shortest paths"
                ),
            },
            "curvature": 1.0,
            "t_star": 7,
            "reconstruction_count_per_graph_level": RECONSTRUCTIONS,
            "tau": 1.0,
            "threshold_frozen": True,
            "threshold_recalibrated": False,
            "models_retrained": False,
            "perturbation_scenarios_per_graph_level": 1,
        },
        "graphs": graphs,
        "runtime_seconds": time.perf_counter() - started,
    }


def main():
    if OUTPUT_DIR.exists() or OUTPUT_FILE.exists():
        raise FileExistsError(f"refusing to overwrite targeted comparison: {OUTPUT_DIR}")
    source_hash = _sha256(EVALUATION_ARTIFACT)
    evaluation = json.loads(EVALUATION_ARTIFACT.read_text(encoding="utf-8"))
    result = run_comparison(
        evaluation,
        ROOT / "artifacts" / "proteins_pilot" / "evaluation_stable_clustering" / "models",
    )
    if _sha256(EVALUATION_ARTIFACT) != source_hash:
        raise RuntimeError("official evaluation artifact changed during comparison")
    result["source_evaluation_artifact"] = str(
        EVALUATION_ARTIFACT.relative_to(ROOT)
    )
    result["source_evaluation_sha256"] = source_hash
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result_file": str(OUTPUT_FILE), "graphs": len(result["graphs"]), "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
