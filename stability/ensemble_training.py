"""Ensemble of independently trained models for the "ensemble" reconstruction source (Decision 11).

Member k is a full training (encoder, community context, denoiser) with the experiment config,
member-specific model / community seeds and, for resample modes other than "seed_only", a resample
of the TRAIN edges (stability/bootstrap_ensemble.py). Pool, control and validation pairs are never
used in training; train edges a member does not learn from stay forbidden as its negatives.

With "seed_only", member 0 uses the config seeds and is therefore the single model of the "noise"
source: it is shared, not trained twice.

Every member is saved as soon as it is trained (checkpoints/member_XX/, finished when the file
"done" exists), so an interrupted run resumes from the last finished member.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from stability.bootstrap_ensemble import member_seed, member_split
from stability.edge_split import EdgeSplit
from stability.experiment_config import ExperimentConfig
from stability.graph_data import GraphData
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.model_training import TrainedModels, build_model_args, normalized_adjacency, train_models
from stability.partial_recovery import CommunityContext

DONE = "done"


def member_config(config: ExperimentConfig, member: int) -> ExperimentConfig:
    """Config of ensemble member ``member`` (only the model and community seeds change)."""
    mode = config.ensemble.resample_mode
    return config.with_overrides(
        model={"seed": member_seed(config.model.seed, mode, member)},
        community={"seed": member_seed(config.community.seed, mode, member)},
    )


def member_directory(checkpoints: Path, member: int) -> Path:
    return Path(checkpoints) / f"member_{member:02d}"


def is_saved(directory: Path) -> bool:
    return (Path(directory) / DONE).is_file()


def save_models(models: TrainedModels, directory: Path, record: dict[str, Any]) -> None:
    """Write checkpoints + history + record; the "done" marker is written last."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(models.encoder.state_dict(), directory / "encoder.pt")
    torch.save(models.denoiser.state_dict(), directory / "denoiser.pt")
    torch.save({"labels": models.context.labels.cpu(), "centers": models.context.centers.cpu()},
               directory / "community_context.pt")
    (directory / "training_history.json").write_text(json.dumps(models.history, indent=2) + "\n", encoding="utf-8")
    (directory / "member.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (directory / DONE).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")


def load_models(graph: GraphData, train_edges: np.ndarray, config: ExperimentConfig, device: torch.device,
                directory: Path) -> TrainedModels:
    """Rebuild a trained model set from a directory written by save_models (or a run's checkpoints/)."""
    directory = Path(directory)
    args, _ = build_model_args(graph, config, device)
    encoder = StabilityLPModel(args).to(device)
    encoder.load_state_dict(torch.load(directory / "encoder.pt", map_location=device))
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.model.curvature, device=device)
    denoiser.load_state_dict(torch.load(directory / "denoiser.pt", map_location=device))
    denoiser.eval()
    saved = torch.load(directory / "community_context.pt", map_location=device, weights_only=False)
    context = CommunityContext(tuple(range(graph.node_count)), saved["labels"].to(device),
                               saved["centers"].to(device), config.model.curvature)
    history_file = directory / "training_history.json"
    history = json.loads(history_file.read_text(encoding="utf-8")) if history_file.is_file() else {}
    return TrainedModels(encoder, denoiser, context, normalized_adjacency(graph.node_count, train_edges, device),
                         history)


def train_ensemble(
    graph: GraphData,
    split: EdgeSplit,
    config: ExperimentConfig,
    device: torch.device,
    checkpoints: Path,
    base_models: TrainedModels,
    log_every: int = 0,
    after_member: Callable[[int, TrainedModels, Path], None] | None = None,
) -> tuple[list[TrainedModels], list[dict[str, Any]]]:
    """Train (or load finished) members 0..K-1, K = recovery.reconstructions.

    after_member(k, models, directory) runs once right after member k is trained (not when loaded),
    e.g. to add its validation recovery. Returns the models and one record per member.
    """
    mode = config.ensemble.resample_mode
    members, records = [], []
    for member in range(config.recovery.reconstructions):
        mconfig = member_config(config, member)
        if mode == "seed_only" and member == 0:
            members.append(base_models)
            records.append({
                "member": 0, "shared_with_noise_source": True, "checkpoint": str(Path(checkpoints).name),
                "model_seed": mconfig.model.seed, "community_seed": mconfig.community.seed,
                "train_edges_used": int(len(split.train)), "held_out_train_edges": 0,
                "training_seconds": base_models.history.get("training_seconds"),
                "denoiser_skipped_epochs": base_models.history.get("denoiser_skipped_epochs", []),
            })
            continue
        directory = member_directory(checkpoints, member)
        msplit = member_split(split, mode, member, config.ensemble.fraction, config.split.seed)
        if is_saved(directory):
            models = load_models(graph, msplit.train, mconfig, device, directory)
            record = json.loads((directory / "member.json").read_text(encoding="utf-8"))
            if log_every:
                print(f"[ensemble] member {member + 1}/{config.recovery.reconstructions}: loaded", flush=True)
        else:
            if log_every:
                print(f"[ensemble] training member {member + 1}/{config.recovery.reconstructions}", flush=True)
            started = time.time()
            models = train_models(graph, msplit, mconfig, device, log_every=log_every)
            models.history["training_seconds"] = round(time.time() - started, 1)
            record = {
                "member": member, "shared_with_noise_source": False,
                "checkpoint": f"{Path(checkpoints).name}/{directory.name}",
                "model_seed": mconfig.model.seed, "community_seed": mconfig.community.seed,
                "train_edges_used": int(len(msplit.train)),
                "held_out_train_edges": int(len(msplit.held_out_train)),
                "training_seconds": models.history["training_seconds"],
                "denoiser_skipped_epochs": models.history.get("denoiser_skipped_epochs", []),
            }
            if log_every:
                skipped = len(models.history.get("denoiser_skipped_epochs", []))
                print(f"[ensemble] member {member + 1}/{config.recovery.reconstructions} trained in "
                      f"{models.history['training_seconds']:.0f}s (skipped denoiser epochs: {skipped})", flush=True)
            if after_member is not None:
                after_member(member, models, directory)
            save_models(models, directory, record)
        members.append(models)
        records.append(record)
    return members, records
