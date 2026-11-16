from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from stability.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.edge_perturbation import remove_edges
from stability.partial_recovery import (
    CommunityContext,
    decode_without_diffusion,
    reconstruct_k,
)
from stability.pilot_artifacts import (
    count_unique_edge_sets,
    load_pilot_artifact,
    save_pilot_artifact,
)
from stability.pilot_config import PilotConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "wp5_synthetic_pilot"
RESULTS_PATH = ARTIFACT_DIR / "results.json"
DIAGNOSTIC_PATH = ARTIFACT_DIR / "stochasticity-diagnostic.json"


def make_adjacency(node_count: int, edges: tuple[tuple[int, int], ...]):
    rows = list(range(node_count))
    cols = list(range(node_count))
    for source, target in edges:
        rows.extend((source, target))
        cols.extend((target, source))
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices, values, (node_count, node_count)
    ).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1)
    normalized_values = adjacency.values() / torch.sqrt(
        degree[adjacency.indices()[0]] * degree[adjacency.indices()[1]]
    )
    return torch.sparse_coo_tensor(
        adjacency.indices(), normalized_values, adjacency.shape
    ).coalesce()


def make_args(node_count: int, feature_dim: int, config: PilotConfig):
    return config.apply_to_args(
        SimpleNamespace(
            manifold="PoincareBall",
            cuda=-1,
            device="cpu",
            feat_dim=feature_dim,
            dim=4,
            model="HGCN",
            n_nodes=node_count,
            num_layers=2,
            act="relu",
            dropout=0.0,
            bias=0,
            use_att=0,
            local_agg=0,
            r=2.0,
            t=1.0,
            nb_false_edges=1,
            nb_edges=1,
            hid1=8,
            hid2=4,
        )
    )


def probability_summary(values: np.ndarray, tau: float):
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("probability matrix must have shape (reconstructions, edges)")
    ranges = values.max(axis=0) - values.min(axis=0)
    margins = np.abs(values - tau)
    return {
        "mean_probability": float(values.mean()),
        "standard_deviation_across_all_values": float(values.std()),
        "minimum_probability": float(values.min()),
        "maximum_probability": float(values.max()),
        "mean_per_edge_standard_deviation": float(values.std(axis=0).mean()),
        "maximum_per_edge_range": float(ranges.max()),
        "mean_per_edge_range": float(ranges.mean()),
        "mean_absolute_distance_to_tau": float(margins.mean()),
        "minimum_absolute_distance_to_tau": float(margins.min()),
        "pairs_with_threshold_classification_changes": int(
            np.any(values >= tau, axis=0).sum()
            - np.all(values >= tau, axis=0).sum()
        ),
    }


def edge_metrics(predicted_edges, original_edges, removed_edges):
    predicted = {frozenset(edge) for edge in predicted_edges}
    original = {frozenset(edge) for edge in original_edges}
    removed = {frozenset(edge) for edge in removed_edges}
    surviving = original - removed
    recovered = predicted & removed
    false_positives = predicted - original
    retained = predicted & surviving
    return {
        "predicted_edge_count": len(predicted),
        "removed_edges_recovered_count": len(recovered),
        "removed_edges_recovered_fraction": len(recovered) / len(removed),
        "false_positive_edge_count": len(false_positives),
        "surviving_edges_retained_count": len(retained),
        "surviving_edge_retention_fraction": len(retained) / len(surviving),
    }


def summarize_metric_rows(rows):
    metric_names = tuple(rows[0])
    return {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "minimum": float(np.min([row[name] for row in rows])),
            "maximum": float(np.max([row[name] for row in rows])),
        }
        for name in metric_names
    }


def check_metric_fixture():
    original = (("a", "b"), ("b", "c"), ("c", "d"))
    removed = (("a", "b"),)
    predicted = (("a", "b"), ("b", "c"), ("a", "d"))
    metrics = edge_metrics(predicted, original, removed)
    expected = {
        "predicted_edge_count": 3,
        "removed_edges_recovered_count": 1,
        "removed_edges_recovered_fraction": 1.0,
        "false_positive_edge_count": 1,
        "surviving_edges_retained_count": 1,
        "surviving_edge_retention_fraction": 0.5,
    }
    if metrics != expected:
        raise AssertionError(f"metric fixture mismatch: {metrics!r}")


def main():
    check_metric_fixture()
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"WP5 result artifact not found: {RESULTS_PATH}")
    encoder_path = ARTIFACT_DIR / "autoencoder.pt"
    denoiser_path = ARTIFACT_DIR / "denoiser.pt"
    for checkpoint in (encoder_path, denoiser_path):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"WP5 checkpoint not found: {checkpoint}")

    saved = load_pilot_artifact(RESULTS_PATH)
    config_data = saved["config"]
    if config_data["tau_status"] != "provisional engineering threshold; not calibrated":
        raise ValueError("expected WP5's explicitly provisional threshold metadata")
    config = PilotConfig(
        curvature=float(config_data["curvature"]),
        tau=float(config_data["tau"]),
    )
    if config_data["num_reconstructions"] != 8 or config_data["t_star"] != 7:
        raise ValueError("saved WP5 recovery settings do not match approved diagnostic")

    graph = saved["graph"]
    node_ids = tuple(graph["node_ids"])
    node_index = {node: index for index, node in enumerate(node_ids)}
    if len(node_index) != len(node_ids):
        raise ValueError("saved WP5 node IDs are not unique")
    features = torch.tensor(graph["features"], dtype=torch.float32)
    original_edges = tuple(tuple(edge) for edge in graph["original_edges"])
    removed_edges = tuple(tuple(edge) for edge in saved["removed_edges"])
    if not set(map(frozenset, removed_edges)).issubset(
        set(map(frozenset, original_edges))
    ):
        raise ValueError("removed edges are not a subset of the original graph")
    damaged_edges = tuple(
        edge
        for edge in original_edges
        if frozenset(edge) not in set(map(frozenset, removed_edges))
    )
    indexed_damaged_edges = tuple(
        (node_index[source], node_index[target]) for source, target in damaged_edges
    )
    damaged_adjacency = make_adjacency(len(node_ids), indexed_damaged_edges)
    candidates = tuple(itertools.combinations(node_ids, 2))
    if len(candidates) != len(saved["reconstructions"][0]["probabilities"]):
        raise ValueError("saved probability vectors do not match candidate-pair ordering")
    reproducible_perturbation = remove_edges(
        node_ids,
        features,
        original_edges,
        len(removed_edges),
        seed=int(config_data["perturbation_seed"]),
    )
    if reproducible_perturbation.removed_edges != removed_edges:
        raise ValueError("saved removed-edge set does not match the recorded seed")

    args = make_args(len(node_ids), features.shape[1], config)
    encoder = StabilityLPModel(args).cpu()
    denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.curvature).cpu()
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

    context_data = graph["community_context"]
    context = CommunityContext(
        node_ids=node_ids,
        labels=torch.tensor(context_data["labels"], dtype=torch.long),
        centers=torch.tensor(context_data["centers"], dtype=torch.float32),
        curvature=float(context_data["curvature"]),
    )
    clusterer = HyperbolicKMeans(n_clusters=2, curvature=config.curvature)
    encoder_curvature = float(encoder.c.detach().reshape(-1)[0].item())
    if not np.isclose(encoder_curvature, config.curvature):
        raise ValueError("loaded encoder curvature does not match WP5 metadata")
    if not np.isclose(context.curvature, config.curvature):
        raise ValueError("saved community curvature does not match WP5 metadata")
    if not np.isclose(denoiser.c, config.curvature):
        raise ValueError("loaded diffusion curvature does not match WP5 metadata")
    if not np.isclose(clusterer.curvature, config.curvature):
        raise ValueError("diagnostic cluster curvature does not match WP5 metadata")

    with torch.no_grad():
        damaged_embeddings = encoder.encode(features, damaged_adjacency)[2]
        baseline_edges, baseline_probabilities = decode_without_diffusion(
            encoder, damaged_embeddings, node_ids, candidates, config.tau
        )
    rerun = reconstruct_k(
        encoder=encoder,
        denoiser=denoiser,
        damaged_embeddings=damaged_embeddings,
        damaged_adjacency=damaged_adjacency,
        context=context,
        node_ids=node_ids,
        candidate_edges=candidates,
        t_star=int(config_data["t_star"]),
        num_reconstructions=int(config_data["num_reconstructions"]),
        seed=int(config_data["reconstruction_seed_base"]),
        config=config,
    )

    stored_probabilities = np.asarray(
        [item["probabilities"] for item in saved["reconstructions"]], dtype=np.float64
    )
    rerun_probabilities = np.stack(
        [tensor.detach().cpu().numpy() for tensor in rerun.probabilities]
    )
    if stored_probabilities.shape != rerun_probabilities.shape:
        raise ValueError("recomputed probability matrix shape differs from saved WP5 data")
    np.testing.assert_allclose(
        rerun_probabilities,
        stored_probabilities,
        rtol=1e-6,
        atol=1e-7,
        err_msg="recomputed probabilities differ from WP5 saved results",
    )
    for index, item in enumerate(saved["reconstructions"]):
        expected_seed = int(config_data["reconstruction_seed_base"]) + index
        if item["seed"] != expected_seed:
            raise ValueError(f"saved reconstruction {index} has an unexpected seed")
        stored_edges = {frozenset(edge) for edge in item["edges"]}
        rerun_edges = {frozenset(edge) for edge in rerun.edge_sets[index]}
        if stored_edges != rerun_edges:
            raise ValueError(f"recomputed thresholded edges differ for reconstruction {index}")

    removed_indices = [candidates.index(tuple(edge)) for edge in removed_edges]
    removed_probability_values = rerun_probabilities[:, removed_indices]
    all_probability_values = rerun_probabilities
    removed_probability_report = {
        edge[0] + "--" + edge[1]: {
            "probabilities": removed_probability_values[:, index].tolist(),
            "mean": float(removed_probability_values[:, index].mean()),
            "standard_deviation": float(removed_probability_values[:, index].std()),
            "minimum": float(removed_probability_values[:, index].min()),
            "maximum": float(removed_probability_values[:, index].max()),
            "mean_absolute_distance_to_tau": float(
                np.abs(removed_probability_values[:, index] - config.tau).mean()
            ),
            "threshold_classifications": (
                removed_probability_values[:, index] >= config.tau
            ).astype(int).tolist(),
        }
        for index, edge in enumerate(removed_edges)
    }

    original_set = {frozenset(edge) for edge in original_edges}
    removed_set = {frozenset(edge) for edge in removed_edges}
    baseline_metrics = edge_metrics(baseline_edges, original_set, removed_set)
    partial_rows = [
        edge_metrics(edge_set, original_set, removed_set)
        for edge_set in rerun.edge_sets
    ]
    partial_metrics = [
        {"reconstruction": index, **metrics}
        for index, metrics in enumerate(partial_rows)
    ]
    unique_sets = len({frozenset(map(frozenset, edge_set)) for edge_set in rerun.edge_sets})
    all_pairs_summary = probability_summary(all_probability_values, config.tau)
    probability_interpretation = (
        "Probabilities vary across seeded reconstructions, but no pair changes "
        "threshold class; thresholding maps them to identical edge sets."
        if all_pairs_summary["maximum_per_edge_range"] > 0
        and all_pairs_summary["pairs_with_threshold_classification_changes"] == 0
        and unique_sets == 1
        else "See probability ranges and threshold-classification counts; the "
        "outputs are not uniformly identical across seeds."
    )

    diagnostic = {
        "scope": "descriptive saved-checkpoint diagnostic; not calibration or generalization",
        "source_artifact": str(RESULTS_PATH.relative_to(ROOT)),
        "checkpoints": {
            "autoencoder": str(encoder_path.relative_to(ROOT)),
            "denoiser": str(denoiser_path.relative_to(ROOT)),
            "strict_state_dict_load": True,
            "saved_reconstruction_probabilities_reproduced": True,
            "saved_thresholded_edge_sets_reproduced": True,
        },
        "settings": {
            "curvature": config.curvature,
            "tau": config.tau,
            "tau_status": config_data["tau_status"],
            "t_star": config_data["t_star"],
            "num_reconstructions": config_data["num_reconstructions"],
            "reconstruction_seed_base": config_data["reconstruction_seed_base"],
        },
        "metric_definitions": {
            "false_positive_edges": "predicted edges absent from the original graph",
            "surviving_edge_retention": (
                "predicted original edges that were not removed, divided by "
                "the number of original non-removed edges"
            ),
            "removed_edge_recovery": (
                "predicted removed edges divided by the number of removed edges"
            ),
            "metric_hand_fixture_passed": True,
            "perturbation_seed_reproduced_saved_removed_edges": True,
        },
        "probabilities": {
            "all_valid_unordered_node_pairs": all_pairs_summary,
            "removed_edges": removed_probability_report,
            "interpretation": probability_interpretation,
        },
        "partial_recovery_metrics_at_tau": {
            "per_reconstruction": partial_metrics,
            "summary": summarize_metric_rows(partial_rows),
        },
        "no_diffusion_baseline_metrics_at_tau": {
            **edge_metrics(baseline_edges, original_set, removed_set),
            "decoder_probability_mean": float(
                baseline_probabilities.detach().cpu().mean()
            ),
            "decoder_probability_minimum": float(
                baseline_probabilities.detach().cpu().min()
            ),
            "decoder_probability_maximum": float(
                baseline_probabilities.detach().cpu().max()
            ),
        },
        "output_set_comparison": {
            "partial_unique_edge_sets": unique_sets,
            "partial_reconstructions": len(rerun.edge_sets),
            "all_partial_sets_identical": unique_sets == 1,
            "no_diffusion_predicted_edges": [list(edge) for edge in baseline_edges],
        },
    }
    save_pilot_artifact(DIAGNOSTIC_PATH, diagnostic)
    print("WP5 stochasticity diagnostic: PASS")
    print(
        "Saved probability reproducibility: PASS; "
        f"all-pair max per-edge range="
        f"{diagnostic['probabilities']['all_valid_unordered_node_pairs']['maximum_per_edge_range']:.6g}; "
        f"minimum |p-tau|={diagnostic['probabilities']['all_valid_unordered_node_pairs']['minimum_absolute_distance_to_tau']:.6g}; "
        "threshold-changing pairs="
        f"{diagnostic['probabilities']['all_valid_unordered_node_pairs']['pairs_with_threshold_classification_changes']}"
    )
    print(f"Probability interpretation: {probability_interpretation}")
    print("Removed-edge probabilities:")
    for edge, values in removed_probability_report.items():
        print(
            f"  {edge}: mean={values['mean']:.6f}, std={values['standard_deviation']:.6g}, "
            f"range=[{values['minimum']:.6f}, {values['maximum']:.6f}], "
            f"mean |p-tau|={values['mean_absolute_distance_to_tau']:.6f}"
        )
    print(
        "Partial recovery: "
        f"removed={diagnostic['partial_recovery_metrics_at_tau']['summary']['removed_edges_recovered_count']}, "
        f"false_positive={diagnostic['partial_recovery_metrics_at_tau']['summary']['false_positive_edge_count']}, "
        f"surviving_retention={diagnostic['partial_recovery_metrics_at_tau']['summary']['surviving_edges_retained_count']}, "
        f"predicted={diagnostic['partial_recovery_metrics_at_tau']['summary']['predicted_edge_count']}"
    )
    print(f"No-diffusion baseline: {diagnostic['no_diffusion_baseline_metrics_at_tau']}")
    print(
        f"Thresholded sets: {unique_sets}/{len(rerun.edge_sets)} unique; "
        f"diagnostic={DIAGNOSTIC_PATH}"
    )


if __name__ == "__main__":
    main()
