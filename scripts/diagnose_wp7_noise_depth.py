from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from stability.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.edge_perturbation import remove_edges
from stability.pilot_artifacts import load_pilot_artifact, save_pilot_artifact
from stability.partial_recovery import (
    CommunityContext,
    decode_without_diffusion,
    reconstruct_k,
)
from stability.pilot_config import PilotConfig
from stability.synthetic_instances import DIAGNOSTIC_SEEDS, generate_synthetic_graph
from stability.wp7_training import make_adjacency, model_args


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_ARTIFACT = (
    ROOT / "artifacts" / "wp7_prethreshold_diagnostic" / "prethreshold-diagnostic.json"
)
CALIBRATION_ARTIFACT = (
    ROOT / "artifacts" / "wp7_synthetic_calibration" / "calibration.json"
)
EVALUATION_ARTIFACT = (
    ROOT / "artifacts" / "wp7_synthetic_evaluation" / "evaluation.json"
)
ARTIFACT_DIR = ROOT / "artifacts" / "wp7_noise_depth_diagnostic"
TIMESTEPS = (2, 4, 7, 14, 28)
RECONSTRUCTIONS = 8


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_graph_models(graph, saved_graph, checkpoint_dir: Path):
    config = PilotConfig(curvature=1.0, tau=0.5)
    args = model_args(graph, config)
    encoder = StabilityLPModel(args).cpu()
    denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.curvature).cpu()
    encoder_path = checkpoint_dir / "autoencoder.pt"
    denoiser_path = checkpoint_dir / "denoiser.pt"
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True), strict=True
    )
    denoiser.load_state_dict(
        torch.load(denoiser_path, map_location="cpu", weights_only=True), strict=True
    )
    encoder.eval()
    denoiser.eval()
    for model in (encoder, denoiser):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    training = saved_graph["training"]
    actual_encoder_hash = _state_hash(encoder)
    actual_denoiser_hash = _state_hash(denoiser)
    if actual_encoder_hash != training["autoencoder_checkpoint_sha256"]:
        raise RuntimeError(f"{graph.graph_id}: autoencoder checkpoint hash mismatch")
    if actual_denoiser_hash != training["denoiser_checkpoint_sha256"]:
        raise RuntimeError(f"{graph.graph_id}: denoiser checkpoint hash mismatch")
    if training["autoencoder_epochs"] != 30 or training["denoiser_epochs"] != 50:
        raise RuntimeError(f"{graph.graph_id}: saved training budget is unexpected")
    if not training["encoder_frozen"] or not training["denoiser_frozen"]:
        raise RuntimeError(f"{graph.graph_id}: saved model state was not frozen")
    return encoder, denoiser, actual_encoder_hash, actual_denoiser_hash


def _ranking_metrics(candidate_edges, removed_edges, probabilities):
    removed_keys = {frozenset(edge) for edge in removed_edges}
    labels = np.asarray(
        [int(frozenset(edge) in removed_keys) for edge in candidate_edges],
        dtype=np.int64,
    )
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite and match candidate ordering")
    if len(set(labels.tolist())) != 2:
        raise ValueError("candidate additions must include positives and nonedges")
    order = sorted(
        range(len(candidate_edges)),
        key=lambda index: (-probabilities[index], candidate_edges[index]),
    )
    top = {frozenset(candidate_edges[index]) for index in order[:3]}
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "removed_edges_in_top_three": len(top & removed_keys),
    }


def _diagnose_graph(graph, saved_graph, checkpoint_dir, frozen_tau):
    if graph.node_ids != tuple(saved_graph["node_ids"]):
        raise RuntimeError(f"{graph.graph_id}: saved node IDs differ")
    if [list(edge) for edge in graph.edges] != saved_graph["original_edges"]:
        raise RuntimeError(f"{graph.graph_id}: saved original graph differs")

    encoder, denoiser, encoder_hash, denoiser_hash = _load_graph_models(
        graph, saved_graph, checkpoint_dir
    )
    perturbation_seed = int(saved_graph["perturbation_seed"])
    perturbation = remove_edges(
        graph.node_ids,
        graph.features,
        graph.edges,
        len(saved_graph["removed_edges"]),
        perturbation_seed,
    )
    if [list(edge) for edge in perturbation.removed_edges] != saved_graph["removed_edges"]:
        raise RuntimeError(f"{graph.graph_id}: saved removed-edge set does not reproduce")
    if [list(edge) for edge in perturbation.edges] != saved_graph["damaged_edges"]:
        raise RuntimeError(f"{graph.graph_id}: damaged graph does not reproduce")

    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    original_index_edges = tuple(
        (node_index[source], node_index[target]) for source, target in graph.edges
    )
    damaged_index_edges = tuple(
        (node_index[source], node_index[target])
        for source, target in perturbation.edges
    )
    original_adjacency = make_adjacency(len(graph.node_ids), original_index_edges)
    damaged_adjacency = make_adjacency(len(graph.node_ids), damaged_index_edges)

    # Training used this per-graph seed for NumPy's clustering initialization.
    np.random.seed(saved_graph["model_seed"])
    clusterer = HyperbolicKMeans(n_clusters=2, curvature=1.0)
    with torch.no_grad():
        original_embeddings = encoder.encode(graph.features, original_adjacency)[2]
        damaged_embeddings = encoder.encode(graph.features, damaged_adjacency)[2]
    original_points = original_embeddings.detach().cpu().numpy()
    centers, labels = clusterer.fit(original_points)
    context = CommunityContext(
        node_ids=graph.node_ids,
        labels=torch.as_tensor(labels, dtype=torch.long),
        centers=torch.as_tensor(centers, dtype=original_embeddings.dtype),
        curvature=1.0,
    )
    context_digest = hashlib.sha256(
        context.labels.contiguous().numpy().tobytes()
        + context.centers.contiguous().numpy().tobytes()
    ).hexdigest()

    original_set = {frozenset(edge) for edge in graph.edges}
    candidates = tuple(
        (graph.node_ids[source], graph.node_ids[target])
        for source, target in itertools.combinations(range(len(graph.node_ids)), 2)
        if frozenset((graph.node_ids[source], graph.node_ids[target]))
        not in original_set
        or frozenset((graph.node_ids[source], graph.node_ids[target]))
        in {frozenset(edge) for edge in perturbation.removed_edges}
    )
    if [list(edge) for edge in candidates] != saved_graph["candidate_edges"]:
        raise RuntimeError(f"{graph.graph_id}: candidate pair ordering differs")
    if len(candidates) != 54:
        raise RuntimeError(f"{graph.graph_id}: expected 54 candidate additions")

    baseline_edges, baseline_probabilities = decode_without_diffusion(
        encoder,
        damaged_embeddings,
        graph.node_ids,
        candidates,
        frozen_tau,
    )
    baseline_values = baseline_probabilities.cpu().numpy()
    baseline_metrics = _ranking_metrics(
        candidates, perturbation.removed_edges, baseline_values
    )
    saved_baseline_values = np.asarray(
        saved_graph["paths"]["damaged_graph_encode_decode_no_diffusion"][
            "probabilities"
        ]
    )
    if not np.allclose(baseline_values, saved_baseline_values, rtol=1e-6, atol=1e-7):
        raise RuntimeError(f"{graph.graph_id}: saved no-diffusion output mismatch")

    methods = {
        "no_diffusion_baseline": {
            "mean_probabilities": baseline_values,
            "mean_probability_metrics": baseline_metrics,
            "sample_metrics": [baseline_metrics],
            "probability_vectors": [baseline_values],
        }
    }
    for timestep in TIMESTEPS:
        batch = reconstruct_k(
            encoder=encoder,
            denoiser=denoiser,
            damaged_embeddings=damaged_embeddings,
            damaged_adjacency=damaged_adjacency,
            context=context,
            node_ids=graph.node_ids,
            candidate_edges=candidates,
            t_star=timestep,
            num_reconstructions=RECONSTRUCTIONS,
            seed=int(saved_graph["reconstruction_seed_base"]),
            config=PilotConfig(curvature=1.0, tau=frozen_tau),
        )
        probability_vectors = [
            probabilities.detach().cpu().numpy()
            for probabilities in batch.probabilities
        ]
        if len(probability_vectors) != RECONSTRUCTIONS:
            raise RuntimeError(f"{graph.graph_id}: wrong reconstruction count")
        if not all(np.isfinite(vector).all() for vector in probability_vectors):
            raise RuntimeError(f"{graph.graph_id}: non-finite probability vector")
        sample_metrics = [
            _ranking_metrics(candidates, perturbation.removed_edges, vector)
            for vector in probability_vectors
        ]
        mean_probabilities = np.mean(np.stack(probability_vectors), axis=0)
        methods[f"t_star_{timestep}"] = {
            "mean_probabilities": mean_probabilities,
            "mean_probability_metrics": _ranking_metrics(
                candidates, perturbation.removed_edges, mean_probabilities
            ),
            "sample_metrics": sample_metrics,
            "probability_vectors": probability_vectors,
        }

    outputs = {}
    for name, values in methods.items():
        vectors = values["probability_vectors"]
        sample_metrics = values["sample_metrics"]
        outputs[name] = {
            "aggregation": (
                "mean probabilities per candidate across 8 samples; baseline has "
                "one deterministic vector"
            ),
            "mean_probability_metrics": values["mean_probability_metrics"],
            "sample_metric_means": {
                metric: float(np.mean([row[metric] for row in sample_metrics]))
                for metric in (
                    "roc_auc",
                    "average_precision",
                    "removed_edges_in_top_three",
                )
            },
            "sample_metric_ranges": {
                metric: [
                    float(np.min([row[metric] for row in sample_metrics])),
                    float(np.max([row[metric] for row in sample_metrics])),
                ]
                for metric in (
                    "roc_auc",
                    "average_precision",
                    "removed_edges_in_top_three",
                )
            },
            "probabilities_by_reconstruction": [
                vector.tolist() for vector in vectors
            ],
        }
    return {
        "graph_id": graph.graph_id,
        "graph_seed": graph.seed,
        "model_seed": saved_graph["model_seed"],
        "perturbation_seed": perturbation_seed,
        "reconstruction_seed_base": saved_graph["reconstruction_seed_base"],
        "reconstruction_seeds": [
            int(saved_graph["reconstruction_seed_base"]) + index
            for index in range(RECONSTRUCTIONS)
        ],
        "checkpoint_hashes_verified": {
            "autoencoder_sha256": encoder_hash,
            "denoiser_sha256": denoiser_hash,
        },
        "fixed_inputs_verified": {
            "original_graph_matches_saved_development_graph": True,
            "removed_edges_match_saved_perturbation": True,
            "candidate_order_matches_saved_diagnostic": True,
            "community_context_computed_once_and_reused": True,
            "community_context_sha256": context_digest,
        },
        "removed_edges": [list(edge) for edge in perturbation.removed_edges],
        "candidate_edges": [list(edge) for edge in candidates],
        "paths": outputs,
    }


def main():
    development = load_pilot_artifact(DEVELOPMENT_ARTIFACT)
    calibration = load_pilot_artifact(CALIBRATION_ARTIFACT)
    evaluation = load_pilot_artifact(EVALUATION_ARTIFACT)
    if development.get("status") != "descriptive_only":
        raise ValueError("noise-depth diagnostic requires saved development graphs")
    if calibration.get("status") != "calibration_complete":
        raise ValueError("frozen calibration artifact is missing or incomplete")
    frozen_tau = float(calibration["frozen_threshold_selection"]["tau"])
    if evaluation.get("frozen_tau") != frozen_tau:
        raise ValueError("WP7 evaluation threshold differs from frozen calibration")
    before_hashes = {
        str(path): _file_hash(path)
        for path in (CALIBRATION_ARTIFACT, EVALUATION_ARTIFACT)
    }
    saved_by_seed = {graph["graph_seed"]: graph for graph in development["graphs"]}
    if set(saved_by_seed) != set(DIAGNOSTIC_SEEDS):
        raise ValueError("saved development graph seeds do not match the approved set")

    results = []
    for index, seed in enumerate(DIAGNOSTIC_SEEDS):
        graph = generate_synthetic_graph(seed, "diagnostic")
        saved_graph = saved_by_seed[seed]
        checkpoint_dir = ARTIFACT_DIR.parent / "wp7_prethreshold_diagnostic" / "models" / graph.graph_id
        if not (checkpoint_dir / "autoencoder.pt").is_file() or not (
            checkpoint_dir / "denoiser.pt"
        ).is_file():
            raise FileNotFoundError(f"{graph.graph_id}: saved checkpoints are missing")
        print(
            f"Evaluating development graph {index + 1}/{len(DIAGNOSTIC_SEEDS)}: "
            f"{graph.graph_id}"
        )
        results.append(
            _diagnose_graph(graph, saved_graph, checkpoint_dir, frozen_tau)
        )

    after_hashes = {
        str(path): _file_hash(path)
        for path in (CALIBRATION_ARTIFACT, EVALUATION_ARTIFACT)
    }
    if before_hashes != after_hashes:
        raise RuntimeError("WP7 calibration or evaluation artifact changed")
    payload = {
        "work_package": "WP7 development noise-depth diagnostic",
        "status": "descriptive_only",
        "frozen_wp7_tau": frozen_tau,
        "wp7_evaluation_artifact_unchanged": True,
        "method": {
            "graph_seeds": list(DIAGNOSTIC_SEEDS),
            "settings": ["no_diffusion_baseline", *[f"t_star_{t}" for t in TIMESTEPS]],
            "reconstructions_per_timestep": RECONSTRUCTIONS,
            "matched_random_streams": True,
            "reconstruction_seed_base_is_same_for_all_timesteps_per_graph": True,
            "timestep_indices_valid": all(0 <= timestep < 200 for timestep in TIMESTEPS),
            "community_context_recomputed_once_per_graph_from_saved_encoder_and_original_graph": True,
            "threshold_used_for_ranking": False,
            "metrics": (
                "mean-probability metrics rank the per-candidate mean over samples; "
                "sample metrics are calculated per reconstruction then summarized "
                "separately"
            ),
        },
        "graphs": results,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACT_DIR / "noise-depth-diagnostic.json"
    save_pilot_artifact(destination, payload)

    summary = []
    for graph in results:
        summary.append(
            {
                "graph_id": graph["graph_id"],
                "settings": {
                    name: {
                        "mean_probability_metrics": data["mean_probability_metrics"],
                        "sample_metric_means": data["sample_metric_means"],
                        "sample_metric_ranges": data["sample_metric_ranges"],
                    }
                    for name, data in graph["paths"].items()
                },
            }
        )
    print(json.dumps({"artifact": str(destination), "graphs": summary}, indent=2))


if __name__ == "__main__":
    main()
