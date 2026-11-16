from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stability.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.edge_perturbation import remove_edges
from stability.imdb_binary_dataset import (
    load_imdb_binary_graphs,
    with_node_identity_features,
)
from stability.imdb_binary_recovery_pilot import _as_synthetic_graph
from stability.partial_recovery import compute_community_context, reconstruct_k
from stability.pilot_artifacts import save_pilot_artifact
from stability.pilot_config import PilotConfig
from stability.proteins_common_fpr_replay import load_saved_graph_models
from stability.proteins_dataset import load_proteins_graphs
from stability.proteins_targeted_classical import (
    rounded_removal_count,
    targeted_edge_order,
)
from stability.scoring import score_recovery
from stability.synthetic_instances import SyntheticGraph
from stability.wp7_training import (
    T_STAR,
    _state_hash,
    make_adjacency,
    model_args,
)


DATA_ROOT = ROOT / "data" / "tu"
CALIBRATION_ARTIFACT = (
    ROOT
    / "artifacts"
    / "proteins_pilot"
    / "fresh_disjoint_random_targeted_20261105"
    / "calibration.json"
)
SUPPORTED_ATTACKS = ("uniform_random", "targeted_degree_product")
SUPPORTED_DAMAGE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
RECONSTRUCTION_SEED_OFFSET = 1_000_000
IMDB_EXPERIMENT_DIR = (
    ROOT
    / "artifacts"
    / "imdb_binary_pilot_20261110"
    / "recovery_20261110_01"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def score_saved_predictions(
    predictions: dict, evaluation_labels: dict
) -> dict[str, object]:
    node_ids = tuple(evaluation_labels["node_ids"])
    original_edges = tuple(
        tuple(edge) for edge in evaluation_labels["original_edges"]
    )
    removed_edges = tuple(
        tuple(edge) for edge in evaluation_labels["removed_edges"]
    )
    reconstructions = tuple(
        tuple(tuple(edge) for edge in row["edges"])
        for row in predictions["reconstructions"]
    )
    if not reconstructions:
        raise ValueError("saved predictions contain no reconstructions")
    if not removed_edges:
        raise ValueError("evaluation labels contain no removed edges")

    original = {frozenset(edge) for edge in original_edges}
    removed = {frozenset(edge) for edge in removed_edges}
    if len(removed) != len(removed_edges) or not removed <= original:
        raise ValueError("removed edges must be unique edges from the intact graph")
    all_pairs = {
        frozenset((source, target))
        for index, source in enumerate(node_ids)
        for target in node_ids[index + 1 :]
    }
    true_nonedges = all_pairs - original
    score = score_recovery(removed_edges, reconstructions)
    false_positives = [
        len({frozenset(edge) for edge in edge_set} & true_nonedges)
        for edge_set in reconstructions
    ]
    if not original_edges:
        raise ValueError("evaluation labels contain no original edges")
    return {
        "actual_damage": {
            "edges_removed": len(removed_edges),
            "original_edge_count": len(original_edges),
            "fraction": len(removed_edges) / len(original_edges),
        },
        "book_stability_score": score.scenario_score,
        "removed_edge_recovery_frequencies": [
            {"edge": list(edge), "frequency": frequency}
            for edge, frequency in score.edge_recovery_frequencies.items()
        ],
        "mean_false_positive_edges": float(np.mean(false_positives)),
        "per_reconstruction_false_positive_edges": false_positives,
        "true_nonedge_count": len(true_nonedges),
        "reconstruction_count": len(reconstructions),
        "score_source": "recomputed from predictions.json and evaluation_labels.json",
    }


def _load_run_inputs(graph_id: str, calibration_path: Path):
    calibration_path = calibration_path.resolve()
    run_root = calibration_path.parent
    selection_path = run_root / "selection.json"
    evaluation_path = run_root / "evaluation.json"
    for path in (calibration_path, selection_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen run artifact is missing: {path}")

    calibration = _load_json(calibration_path)
    selection = _load_json(selection_path)
    evaluation = _load_json(evaluation_path)
    if calibration.get("status") != "calibration_complete":
        raise ValueError("calibration artifact is not complete")
    if selection.get("status") != "selected_before_training_or_outcome_inspection":
        raise ValueError("selection artifact is not a frozen pre-outcome manifest")
    if _sha256(selection_path) != calibration.get("selection_artifact_sha256"):
        raise ValueError("selection artifact hash does not match calibration")
    if evaluation.get("status") != "evaluation_complete":
        raise ValueError("matching evaluation artifact is not complete")
    if evaluation.get("threshold_changed_during_evaluation") is not False:
        raise ValueError("evaluation does not certify a frozen threshold")

    threshold = calibration["threshold_selection"]["best_threshold_under_fpr_cap"]
    tau = float(threshold["tau"])
    cap = float(calibration["method"]["calibration_fpr_cap"])
    if float(threshold["partial_recovery_mean_fpr"]) > cap:
        raise ValueError("calibrated partial-recovery threshold exceeded its FPR cap")
    if not np.isclose(float(evaluation["frozen_tau"]), tau, rtol=0.0, atol=1e-12):
        raise ValueError("calibration and evaluation thresholds do not match")

    selected_graph = next(
        (
            row
            for row in selection["selection"]["evaluation"]
            if row["graph_id"] == graph_id
        ),
        None,
    )
    if selected_graph is None:
        raise ValueError(f"{graph_id} is not in this run's frozen evaluation split")
    evaluation_graph = next(
        (row for row in evaluation["evaluation_graphs"] if row["graph_id"] == graph_id),
        None,
    )
    if evaluation_graph is None:
        raise ValueError(f"{graph_id} has no matching saved evaluation model record")
    model_seed = int(selection["seed_plan"]["model_seeds"][graph_id])
    return {
        "calibration": calibration,
        "calibration_path": calibration_path,
        "calibration_sha256": _sha256(calibration_path),
        "selection_path": selection_path,
        "selection_sha256": _sha256(selection_path),
        "evaluation_path": evaluation_path,
        "evaluation_sha256": _sha256(evaluation_path),
        "selection": selection,
        "selected_graph": selected_graph,
        "evaluation_graph": evaluation_graph,
        "checkpoint_root": run_root / "models" / "evaluation",
        "model_seed": model_seed,
        "tau": tau,
        "cap": cap,
    }


def _validate_imdb_thresholds(
    calibration: dict, evaluation: dict
) -> tuple[float, float]:
    if calibration.get("status") != "calibration_complete":
        raise ValueError("IMDB calibration summary is not complete")
    if not calibration.get("evaluation_gate_passed_for_both_methods"):
        raise ValueError("IMDB calibration gate did not pass for both methods")
    if evaluation.get("status") != "evaluation_complete":
        raise ValueError("IMDB evaluation summary is not complete")
    if not evaluation.get("thresholds_frozen_before_evaluation"):
        raise ValueError("IMDB evaluation does not certify frozen thresholds")
    if calibration.get("manifest_sha256") != evaluation.get("manifest_sha256"):
        raise ValueError("IMDB calibration and evaluation reference different manifests")

    partial = calibration["method_thresholds"]["partial_recovery"]
    cap = float(partial["false_positive_rate_cap"])
    tau = float(partial["tau"])
    if float(partial["calibration_realized_fpr"]) > cap:
        raise ValueError("IMDB partial-recovery threshold exceeded its calibration cap")
    evaluated_tau = float(
        evaluation["frozen_method_thresholds"]["partial_recovery"]
    )
    if not np.isclose(evaluated_tau, tau, rtol=0.0, atol=1e-12):
        raise ValueError("IMDB calibration and evaluation thresholds do not match")
    return tau, cap


def _load_imdb_run_inputs(graph_id: str, experiment_dir: Path) -> dict:
    experiment_dir = experiment_dir.resolve()
    selection_path = experiment_dir / "selection_and_masks.json"
    calibration_path = experiment_dir / "calibration" / "summary.json"
    evaluation_path = experiment_dir / "evaluation" / "summary.json"
    graph_record_path = experiment_dir / "evaluation" / "graphs" / f"{graph_id}.json"
    for path in (selection_path, calibration_path, evaluation_path, graph_record_path):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen IMDB artifact is missing: {path}")

    selection = _load_json(selection_path)
    calibration = _load_json(calibration_path)
    evaluation = _load_json(evaluation_path)
    graph_record = _load_json(graph_record_path)
    if selection.get("status") != "predefined_before_pilot_outcomes":
        raise ValueError("IMDB selection artifact is not a frozen pre-outcome manifest")
    manifest_hash = _sha256(selection_path)
    if manifest_hash != calibration.get("manifest_sha256"):
        raise ValueError("IMDB selection artifact hash does not match calibration")
    if manifest_hash != evaluation.get("manifest_sha256"):
        raise ValueError("IMDB selection artifact hash does not match evaluation")
    tau, cap = _validate_imdb_thresholds(calibration, evaluation)

    if graph_id not in selection["eligibility"]["evaluation_graph_ids"]:
        raise ValueError(
            f"{graph_id} is not in this IMDB experiment's frozen evaluation split"
        )
    if graph_record.get("graph_id") != graph_id:
        raise ValueError(f"IMDB graph record does not match requested graph {graph_id}")
    if graph_record.get("split") != "evaluation":
        raise ValueError(f"{graph_id} is not recorded as an IMDB evaluation graph")
    return {
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": manifest_hash,
        "calibration_path": calibration_path,
        "calibration_sha256": _sha256(calibration_path),
        "evaluation_path": evaluation_path,
        "evaluation_sha256": _sha256(evaluation_path),
        "graph_record": graph_record,
        "checkpoint_root": experiment_dir / "models" / "evaluation",
        "model_seed": int(graph_record["model_seed"]),
        "tau": tau,
        "cap": cap,
    }


def run_single_recovery(
    graph_id: str,
    attack: str,
    damage_fraction: float,
    reconstruction_count: int,
    seed: int,
    calibration_path: Path = CALIBRATION_ARTIFACT,
    output_dir: Path | None = None,
    dataset: str = "PROTEINS",
    experiment_dir: Path | None = None,
    reconstruction_seed_base: int | None = None,
) -> Path:
    if dataset not in ("PROTEINS", "IMDB-BINARY"):
        raise ValueError("dataset must be PROTEINS or IMDB-BINARY")
    if attack not in SUPPORTED_ATTACKS:
        raise ValueError(f"attack must be one of {SUPPORTED_ATTACKS}")
    if damage_fraction not in SUPPORTED_DAMAGE_FRACTIONS:
        raise ValueError(
            f"damage_fraction must be one of {SUPPORTED_DAMAGE_FRACTIONS}"
        )
    if reconstruction_count <= 0:
        raise ValueError("reconstruction_count must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if reconstruction_seed_base is not None and reconstruction_seed_base < 0:
        raise ValueError("reconstruction_seed_base must be nonnegative")

    if dataset == "PROTEINS":
        if experiment_dir is not None:
            raise ValueError("--experiment-dir is only supported for IMDB-BINARY")
        run_inputs = _load_run_inputs(graph_id, calibration_path)
        graphs, raw_hashes = load_proteins_graphs(DATA_ROOT, download=False)
        expected_hashes = run_inputs["selection"]["raw_file_sha256"]
        if raw_hashes != expected_hashes:
            raise RuntimeError("cached PROTEINS raw-file hashes differ from frozen run")
        graph = next((row for row in graphs if row.graph_id == graph_id), None)
        if graph is None:
            raise ValueError(f"{graph_id} is not an eligible supported PROTEINS graph")
        if graph.content_sha256 != run_inputs["selected_graph"]["content_sha256"]:
            raise RuntimeError("current graph content differs from the frozen selection")
        recovery_graph = graph
        training = run_inputs["evaluation_graph"]["training"]
        expected_feature_hash = graph.content_sha256
    else:
        if calibration_path != CALIBRATION_ARTIFACT:
            raise ValueError("--calibration-artifact is only supported for PROTEINS")
        run_inputs = _load_imdb_run_inputs(
            graph_id, experiment_dir or IMDB_EXPERIMENT_DIR
        )
        graphs, raw_hashes, _ = load_imdb_binary_graphs(DATA_ROOT, download=False)
        expected_hashes = run_inputs["selection"]["raw_file_sha256"]
        if raw_hashes != expected_hashes:
            raise RuntimeError("cached IMDB-BINARY raw-file hashes differ from frozen run")
        source_graph = next((row for row in graphs if row.graph_id == graph_id), None)
        if source_graph is None:
            raise ValueError(f"{graph_id} is not an eligible supported IMDB-BINARY graph")
        selected_hash = run_inputs["selection"]["eligibility"]["graph_hashes"].get(
            graph_id
        )
        if source_graph.content_sha256 != selected_hash:
            raise RuntimeError("IMDB graph content differs from the frozen selection")
        graph = with_node_identity_features(source_graph)
        recovery_graph = _as_synthetic_graph(source_graph)
        expected_feature_hash = run_inputs["selection"]["eligibility"][
            "identity_feature_graph_hashes"
        ].get(graph_id)
        record = run_inputs["graph_record"]
        if int(record["model_seed"]) != 3_000_000 + source_graph.dataset_index:
            raise RuntimeError("IMDB model seed differs from the frozen pilot protocol")
        if graph.content_sha256 != expected_feature_hash:
            raise RuntimeError("IMDB identity-feature graph hash differs from frozen metadata")
        if tuple(record["node_ids"]) != graph.node_ids:
            raise RuntimeError("IMDB node order differs from frozen evaluation metadata")
        if not torch.equal(
            torch.as_tensor(record["features"], dtype=graph.features.dtype),
            graph.features,
        ):
            raise RuntimeError("IMDB node features differ from frozen evaluation metadata")
        if record.get("graph_sha256") != source_graph.content_sha256:
            raise RuntimeError("IMDB source graph hash differs from evaluation metadata")
        if record.get("identity_feature_graph_sha256") != graph.content_sha256:
            raise RuntimeError("IMDB feature graph hash differs from evaluation metadata")
        training = record["training"]

    checkpoint_dir = run_inputs["checkpoint_root"] / graph_id
    checkpoint_paths = {
        "autoencoder": checkpoint_dir / "autoencoder.pt",
        "denoiser": checkpoint_dir / "denoiser.pt",
    }
    for name, path in checkpoint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"saved {name} checkpoint is missing: {path}")

    config = PilotConfig(curvature=1.0, tau=run_inputs["tau"])
    if dataset == "PROTEINS":
        encoder, denoiser, context = load_saved_graph_models(
            graph,
            run_inputs["checkpoint_root"],
            config,
            run_inputs["model_seed"],
        )
    else:
        args = model_args(recovery_graph, config)
        encoder = StabilityLPModel(args).cpu()
        denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.curvature).cpu()
        encoder.load_state_dict(
            torch.load(checkpoint_paths["autoencoder"], map_location="cpu", weights_only=True)
        )
        denoiser.load_state_dict(
            torch.load(checkpoint_paths["denoiser"], map_location="cpu", weights_only=True)
        )
        encoder.eval()
        denoiser.eval()
        original_adjacency = make_adjacency(
            len(graph.node_ids),
            tuple(
                (graph.node_ids.index(source), graph.node_ids.index(target))
                for source, target in graph.edges
            ),
        )
        np.random.seed(run_inputs["model_seed"])
        context = compute_community_context(
            encoder,
            graph.features,
            original_adjacency,
            HyperbolicKMeans(n_clusters=2, curvature=config.curvature),
            graph.node_ids,
        )
    if not 0 <= T_STAR < denoiser.num_timesteps:
        raise ValueError("t_star is outside the saved denoiser schedule")
    for name, model in (("autoencoder", encoder), ("denoiser", denoiser)):
        if _state_hash(model) != training[f"{name}_checkpoint_sha256"]:
            raise RuntimeError(
                f"loaded {name} state hash differs from saved evaluation metadata"
            )
    node_ids = graph.node_ids
    edge_count = len(graph.edges)
    removal_count = rounded_removal_count(edge_count, damage_fraction)
    if removal_count <= 0:
        raise ValueError("configured damage level would remove no edges")

    if attack == "uniform_random":
        perturbed = remove_edges(
            node_ids, graph.features, graph.edges, removal_count, seed
        )
        removed_edges = perturbed.removed_edges
        damaged_edges = perturbed.edges
    else:
        ordered_edges, _ = targeted_edge_order(node_ids, graph.edges)
        removed_edges = ordered_edges[:removal_count]
        removed_keys = {frozenset(edge) for edge in removed_edges}
        damaged_edges = tuple(
            edge for edge in graph.edges if frozenset(edge) not in removed_keys
        )

    node_index = {node: index for index, node in enumerate(node_ids)}
    indexed_edges = tuple(
        (node_index[source], node_index[target]) for source, target in damaged_edges
    )
    damaged_adjacency = make_adjacency(len(node_ids), indexed_edges)
    damaged_keys = {frozenset(edge) for edge in damaged_edges}
    candidate_edges = tuple(
        (node_ids[source], node_ids[target])
        for source, target in itertools.combinations(range(len(node_ids)), 2)
        if dataset == "PROTEINS"
        or frozenset((node_ids[source], node_ids[target])) not in damaged_keys
    )
    with torch.no_grad():
        damaged_embeddings = encoder.encode(graph.features, damaged_adjacency)[2]
        if not torch.isfinite(damaged_embeddings).all():
            raise RuntimeError("damaged embeddings contain non-finite values")
        reconstruction_seed = (
            reconstruction_seed_base
            if reconstruction_seed_base is not None
            else seed + RECONSTRUCTION_SEED_OFFSET
        )
        batch = reconstruct_k(
            encoder=encoder,
            denoiser=denoiser,
            damaged_embeddings=damaged_embeddings,
            damaged_adjacency=damaged_adjacency,
            context=context,
            node_ids=node_ids,
            candidate_edges=candidate_edges,
            t_star=T_STAR,
            num_reconstructions=reconstruction_count,
            seed=reconstruction_seed,
            config=config,
        )

    if any(not torch.isfinite(row).all() for row in batch.probabilities):
        raise RuntimeError("decoder produced non-finite candidate scores")
    predictions = {
        "graph_id": graph_id,
        "candidate_edges": [list(edge) for edge in candidate_edges],
        "reconstructions": [
            {
                "seed": reconstruction_seed + index,
                "candidate_scores": probabilities.detach().cpu().tolist(),
                "edges": [list(edge) for edge in edge_set],
            }
            for index, (probabilities, edge_set) in enumerate(
                zip(batch.probabilities, batch.edge_sets)
            )
        ],
    }
    evaluation_labels = {
        "graph_id": graph_id,
        "node_ids": list(node_ids),
        "original_edges": [list(edge) for edge in graph.edges],
        "removed_edges": [list(edge) for edge in removed_edges],
        "damaged_edges": [list(edge) for edge in damaged_edges],
    }
    score = score_saved_predictions(predictions, evaluation_labels)
    run_config = {
        "status": "complete",
        "method": "partial_recovery",
        "dataset": dataset,
        "graph_id": graph_id,
        "graph_content_sha256": graph.content_sha256,
        "feature_graph_sha256": expected_feature_hash,
        "node_count": len(node_ids),
        "original_edge_count": edge_count,
        "attack": attack,
        "damage_fraction_requested": damage_fraction,
        "edges_removed": removal_count,
        "damage_fraction_realized": removal_count / edge_count,
        "reconstruction_count": reconstruction_count,
        "seed": seed,
        "model_seed": run_inputs["model_seed"],
        "perturbation_seed": seed if attack == "uniform_random" else None,
        "reconstruction_seed_base": reconstruction_seed,
        "t_star": T_STAR,
        "diffusion_schedule_length": denoiser.num_timesteps,
        "reverse_denoiser_calls_per_reconstruction": T_STAR + 1,
        "threshold": {
            "tau": run_inputs["tau"],
            "calibration_fpr_cap": run_inputs["cap"],
            "calibration_artifact": str(
                _artifact_path(run_inputs["calibration_path"])
            ),
            "calibration_artifact_sha256": run_inputs["calibration_sha256"],
        },
        "selection_artifact_sha256": run_inputs["selection_sha256"],
        "evaluation_artifact_sha256": run_inputs["evaluation_sha256"],
        "selection_artifact": str(_artifact_path(run_inputs["selection_path"])),
        "checkpoint_sha256": {
            name: _sha256(path) for name, path in checkpoint_paths.items()
        },
        "training_performed": False,
        "device": "cpu",
    }
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ROOT / "artifacts" / "recovery_runs" / f"{graph_id}_{stamp}"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    save_pilot_artifact(output_dir / "run_config.json", run_config)
    save_pilot_artifact(output_dir / "predictions.json", predictions)
    save_pilot_artifact(output_dir / "evaluation_labels.json", evaluation_labels)
    save_pilot_artifact(output_dir / "score.json", score)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one CPU partial-recovery case with frozen checkpoints."
    )
    parser.add_argument("--dataset", choices=("PROTEINS", "IMDB-BINARY"), default="PROTEINS")
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--attack", choices=SUPPORTED_ATTACKS, required=True)
    parser.add_argument(
        "--damage-fraction",
        type=float,
        choices=SUPPORTED_DAMAGE_FRACTIONS,
        required=True,
    )
    parser.add_argument("--k", type=int, required=True, dest="reconstruction_count")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=CALIBRATION_ARTIFACT,
    )
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--reconstruction-seed-base", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = run_single_recovery(
        graph_id=args.graph_id,
        attack=args.attack,
        damage_fraction=args.damage_fraction,
        reconstruction_count=args.reconstruction_count,
        seed=args.seed,
        calibration_path=args.calibration_artifact,
        output_dir=args.output_dir,
        dataset=args.dataset,
        experiment_dir=args.experiment_dir,
        reconstruction_seed_base=args.reconstruction_seed_base,
    )
    print(f"Saved single recovery run: {output_dir}")
    score = _load_json(output_dir / "score.json")
    print(f"Actual damage: {score['actual_damage']}")
    print((output_dir / "score.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
