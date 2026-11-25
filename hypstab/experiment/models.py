"""Trained model bundles for the experiment: member 0 (the "noise" model) and the ensemble members.

A bundle = autoencoder (HGCN + Fermi-Dirac) + community context + diffusion denoiser, all trained on
the TRAIN edges only (Decision 6). Ensemble member m (Decision 11, "seed_only") uses the same code,
config and train edges with different encoder / community / denoiser seeds; member 0 uses the config
seeds, so it is the same model as the "noise" source (shared, not retrained).

Bundles are saved in <run_dir>/models/member_XX/ with a "done" marker written last, so an
interrupted run resumes without retraining finished members.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from hypstab.config import HypstabConfig
from hypstab.data.graph import GraphData
from hypstab.data.split import EdgeSplit
from hypstab.diffusion.model import HyperbolicDiffusion
from hypstab.evaluation.validation import validation_report
from hypstab.metrics import adjusted_rand_index
from hypstab.models.kmeans import KMeansResult
from hypstab.training.autoencoder import HyperbolicAutoencoder, build_autoencoder, train_autoencoder
from hypstab.training.communities import CommunityContext, fit_communities
from hypstab.training.denoiser import build_diffusion, train_denoiser

DONE = "done"


@dataclass
class ModelBundle:
    member: int
    autoencoder: HyperbolicAutoencoder
    context: CommunityContext
    diffusion: HyperbolicDiffusion
    record: dict[str, Any]


def member_seed(base_seed: int, member: int) -> int:
    """Member 0 keeps the config seed; member m >= 1 gets a seed derived from (base, m)."""
    if member == 0:
        return base_seed
    return int(np.random.SeedSequence([base_seed, member]).generate_state(1)[0])


def member_config(config: HypstabConfig, member: int) -> HypstabConfig:
    """Config of ensemble member ``member``: only the encoder, community and denoiser seeds change."""
    if member == 0:
        return config
    return replace(
        config,
        encoder=replace(config.encoder, seed=member_seed(config.encoder.seed, member)),
        community=replace(config.community, seed=member_seed(config.community.seed, member)),
        denoiser=replace(config.denoiser, seed=member_seed(config.denoiser.seed, member)),
    )


def member_directory(run_dir: Path, member: int) -> Path:
    return Path(run_dir) / "models" / f"member_{member:02d}"


def is_saved(directory: Path) -> bool:
    return (Path(directory) / DONE).is_file()


def train_bundle(graph: GraphData, split: EdgeSplit, config: HypstabConfig, member: int,
                 device: torch.device, log_every: int = 0) -> ModelBundle:
    """Train autoencoder -> communities -> denoiser of one member on the train edges, + validation report."""
    cfg = member_config(config, member)
    started = time.perf_counter()
    result = train_autoencoder(graph, split, cfg, device, log_every=log_every)
    autoencoder_seconds = time.perf_counter() - started
    features = graph.features.to(device)
    context = fit_communities(result.model, features, result.train_adjacency, cfg)
    started = time.perf_counter()
    diffusion, denoiser_history = train_denoiser(graph, split, cfg, device, result.model, context,
                                                 log_every=log_every * 10)
    denoiser_seconds = time.perf_counter() - started
    validation = validation_report(result.model, features, result.train_adjacency, split, context, cfg, diffusion)
    record = {
        "member": member,
        "seeds": {"encoder": cfg.encoder.seed, "community": cfg.community.seed, "denoiser": cfg.denoiser.seed},
        "autoencoder_best_val_auc": result.history["best_val_auc"],
        "autoencoder_best_epoch": result.history["best_epoch"],
        "autoencoder_seconds": round(autoencoder_seconds, 1),
        "denoiser_seconds": round(denoiser_seconds, 1),
        "denoiser_skipped_epochs": len(denoiser_history["skipped_epochs"]),
        "communities": context.summary(),
        # agreement of the model's communities (K-means, no labels) with the class labels / SBM blocks
        "communities_vs_classes_ari": None if graph.class_labels is None else
        adjusted_rand_index(context.labels.cpu().numpy(), graph.class_labels),
        "validation_auc": {k: v["auc"] for k, v in validation.items() if isinstance(v, dict)},
        "histories": {"autoencoder": result.history, "denoiser": denoiser_history},
    }
    return ModelBundle(member, result.model, context, diffusion, record)


def save_bundle(bundle: ModelBundle, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.autoencoder.state_dict(), directory / "autoencoder.pt")
    km = bundle.context.kmeans
    torch.save({"labels": bundle.context.labels.cpu(), "centers": bundle.context.centers.cpu(),
                "inertia": km.inertia, "iterations": km.iterations, "converged": km.converged},
               directory / "communities.pt")
    torch.save(bundle.diffusion.state_dict(), directory / "denoiser.pt")
    (directory / "record.json").write_text(json.dumps(bundle.record, indent=2), encoding="utf-8")
    (directory / DONE).write_text("ok\n", encoding="utf-8")   # written last


def load_bundle(directory: Path, feature_dim: int, config: HypstabConfig, member: int,
                device: torch.device) -> ModelBundle:
    directory = Path(directory)
    if not is_saved(directory):
        raise FileNotFoundError(f"{directory} has no finished model (missing '{DONE}')")
    cfg = member_config(config, member)
    autoencoder = build_autoencoder(feature_dim, cfg).to(device)
    autoencoder.load_state_dict(torch.load(directory / "autoencoder.pt", map_location=device))
    autoencoder.eval()
    saved = torch.load(directory / "communities.pt", map_location=device)
    labels, centers = saved["labels"].to(device), saved["centers"].to(device)
    kmeans = KMeansResult(centers=centers, labels=labels, inertia=saved["inertia"],
                          iterations=saved["iterations"], converged=saved["converged"])
    context = CommunityContext(labels=labels, centers=centers, kmeans=kmeans)
    diffusion = build_diffusion(cfg).to(device)
    diffusion.load_state_dict(torch.load(directory / "denoiser.pt", map_location=device))
    diffusion.eval()
    record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
    return ModelBundle(member, autoencoder, context, diffusion, record)


def get_bundles(graph: GraphData, split: EdgeSplit, config: HypstabConfig, members: int, run_dir: Path,
                device: torch.device, log_every: int = 0) -> list[ModelBundle]:
    """Members 0 .. members-1: load the finished ones, train (and save) the rest."""
    bundles = []
    for member in range(members):
        directory = member_directory(run_dir, member)
        if is_saved(directory):
            bundle = load_bundle(directory, graph.features.shape[1], config, member, device)
            print(f"[models] member {member}: loaded", flush=True)
        else:
            print(f"[models] member {member}: training", flush=True)
            bundle = train_bundle(graph, split, config, member, device, log_every=log_every)
            save_bundle(bundle, directory)
            print(f"[models] member {member}: val AUC {bundle.record['validation_auc']}", flush=True)
        bundles.append(bundle)
    return bundles
