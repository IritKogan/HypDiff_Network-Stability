from __future__ import annotations

import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from stability.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.edge_perturbation import remove_edges
from stability.partial_recovery import (
    compute_community_context,
    reconstruct_k,
    score_reconstructions,
)
from stability.pilot_artifacts import (
    count_unique_edge_sets,
    rescore_pilot_artifact,
    save_pilot_artifact,
    serialized_edges,
)
from stability.pilot_config import PilotConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "wp5_synthetic_pilot"
AUTOENCODER_EPOCHS = 30
DENOISER_EPOCHS = 50
RECONSTRUCTIONS = 8
REMOVED_EDGE_COUNT = 3
T_STAR = 7
SEED = 20261105


def make_graph():
    node_ids = tuple(f"n{index:02d}" for index in range(12))
    edges = []
    for start in (0, 6):
        community = tuple(range(start, start + 6))
        edges.extend(
            (community[index], community[(index + 1) % len(community)])
            for index in range(len(community))
        )
    edges.extend(((0, 6), (2, 8), (4, 10)))
    edges = tuple(
        (node_ids[source], node_ids[target]) for source, target in edges
    )
    features = torch.tensor(
        [
            [
                float(np.cos(2 * np.pi * index / 12)),
                float(np.sin(2 * np.pi * index / 12)),
                float(index < 6),
                float(index >= 6),
            ]
            for index in range(12)
        ],
        dtype=torch.float32,
    )
    return node_ids, features, tuple(edges)


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


def split_edges(edges, seed):
    ordered = tuple(sorted(edges))
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(len(ordered))
    validation_count = max(1, len(ordered) // 5)
    validation_indices = set(permutation[:validation_count].tolist())
    train_edges = tuple(edge for index, edge in enumerate(ordered) if index not in validation_indices)
    validation_edges = tuple(edge for index, edge in enumerate(ordered) if index in validation_indices)
    all_pairs = tuple(itertools.combinations(range(12), 2))
    positive_set = {frozenset(edge) for edge in ordered}
    negative_edges = tuple(
        edge for edge in all_pairs if frozenset(edge) not in positive_set
    )
    negative_order = generator.permutation(len(negative_edges))
    train_negative_count = len(train_edges)
    validation_negative_count = len(validation_edges)
    train_negatives = tuple(
        negative_edges[index] for index in negative_order[:train_negative_count]
    )
    validation_negatives = tuple(
        negative_edges[index]
        for index in negative_order[
            train_negative_count : train_negative_count + validation_negative_count
        ]
    )
    return train_edges, validation_edges, train_negatives, validation_negatives


def edge_tensor(edges, device):
    return torch.tensor(edges, dtype=torch.long, device=device).reshape(-1, 2)


def link_loss(model, embeddings, positive_edges, negative_edges):
    positive = model.decode(embeddings, positive_edges)
    negative = model.decode(embeddings, negative_edges)
    positive_loss = F.binary_cross_entropy(positive, torch.ones_like(positive))
    negative_loss = F.binary_cross_entropy(negative, torch.zeros_like(negative))
    return positive_loss + negative_loss


def train_autoencoder(model, features, train_adjacency, validation, train_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    positive_train = edge_tensor(train_data[0], features.device)
    negative_train = edge_tensor(train_data[1], features.device)
    positive_validation = edge_tensor(validation[0], features.device)
    negative_validation = edge_tensor(validation[1], features.device)
    epoch_losses = []
    validation_losses = []

    for _ in range(AUTOENCODER_EPOCHS):
        model.train()
        optimizer.zero_grad()
        embeddings, _, _, _ = model.encode(features, train_adjacency)
        loss = link_loss(model, embeddings, positive_train, negative_train)
        if not torch.isfinite(loss):
            raise RuntimeError("autoencoder training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            validation_embeddings = model.encode(features, train_adjacency)[2]
            validation_loss = link_loss(
                model,
                validation_embeddings,
                positive_validation,
                negative_validation,
            )
        if not torch.isfinite(validation_loss):
            raise RuntimeError("autoencoder validation produced a non-finite loss")
        validation_losses.append(float(validation_loss))

    return epoch_losses, validation_losses


def train_denoiser(model, embeddings, adjacency, labels, centers):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch_losses = []
    for _ in range(DENOISER_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = model(
            embeddings.clone(),
            adjacency,
            labels,
            centers,
            restrict=True,
            graphset=False,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("denoiser training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.detach()))
    return epoch_losses


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty():
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(result.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config = PilotConfig(curvature=1.0, tau=0.5)
    args = config.apply_to_args(
        SimpleNamespace(
            manifold="PoincareBall",
            cuda=-1,
            device="cpu",
            feat_dim=4,
            dim=4,
            model="HGCN",
            n_nodes=12,
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
    node_ids, features, original_edges = make_graph()
    node_index = {node: index for index, node in enumerate(node_ids)}
    original_index_edges = tuple(
        (node_index[source], node_index[target]) for source, target in original_edges
    )
    train_edges, validation_edges, train_negatives, validation_negatives = split_edges(
        original_index_edges, SEED
    )
    train_adjacency = make_adjacency(len(node_ids), train_edges)
    original_adjacency = make_adjacency(len(node_ids), original_index_edges)
    encoder = StabilityLPModel(args).cpu()
    denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.curvature).cpu()
    autoencoder_losses, autoencoder_validation_losses = train_autoencoder(
        encoder,
        features,
        train_adjacency,
        (validation_edges, validation_negatives),
        (train_edges, train_negatives),
    )

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        original_embeddings = encoder.encode(features, original_adjacency)[2]
    if not torch.isfinite(original_embeddings).all():
        raise RuntimeError("trained encoder produced non-finite original embeddings")
    if not torch.all(torch.linalg.vector_norm(original_embeddings, dim=1) < 1.0):
        raise RuntimeError("original embeddings lie outside the c=1.0 unit ball")

    clusterer = HyperbolicKMeans(n_clusters=2, curvature=config.curvature)
    context = compute_community_context(
        encoder, features, original_adjacency, clusterer, node_ids
    )
    denoiser_losses = train_denoiser(
        denoiser,
        original_embeddings.detach(),
        original_adjacency,
        context.labels,
        context.centers,
    )
    denoiser.eval()
    for parameter in denoiser.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("encoder weights were not frozen")
    if any(parameter.requires_grad for parameter in denoiser.parameters()):
        raise RuntimeError("denoiser weights were not frozen")

    perturbation = remove_edges(
        node_ids,
        features,
        original_edges,
        REMOVED_EDGE_COUNT,
        seed=SEED + 1,
    )
    damaged_index_edges = tuple(
        (node_index[source], node_index[target]) for source, target in perturbation.edges
    )
    damaged_adjacency = make_adjacency(len(node_ids), damaged_index_edges)
    with torch.no_grad():
        damaged_embeddings = encoder.encode(features, damaged_adjacency)[2]
    candidate_edges = tuple(itertools.combinations(node_ids, 2))
    reconstructions = reconstruct_k(
        encoder=encoder,
        denoiser=denoiser,
        damaged_embeddings=damaged_embeddings,
        damaged_adjacency=damaged_adjacency,
        context=context,
        node_ids=node_ids,
        candidate_edges=candidate_edges,
        t_star=T_STAR,
        num_reconstructions=RECONSTRUCTIONS,
        seed=SEED + 2,
        config=config,
    )
    score = score_reconstructions(reconstructions, perturbation.removed_edges)
    payload = {
        "scope": "synthetic CPU plumbing pilot; not a generalization estimate",
        "git_revision": git_revision(),
        "git_worktree_dirty": git_worktree_dirty(),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
        },
        "config": {
            "curvature": config.curvature,
            "tau": config.tau,
            "tau_status": "provisional engineering threshold; not calibrated",
            "schedule_length": denoiser.num_timesteps,
            "t_star": T_STAR,
            "num_reconstructions": RECONSTRUCTIONS,
            "epochs": {
                "autoencoder": AUTOENCODER_EPOCHS,
                "denoiser": DENOISER_EPOCHS,
            },
            "seed": SEED,
            "perturbation_seed": SEED + 1,
            "reconstruction_seed_base": SEED + 2,
        },
        "graph": {
            "node_ids": list(node_ids),
            "features": features.tolist(),
            "original_edges": serialized_edges(original_edges),
            "validation_edges": [list(edge) for edge in validation_edges],
            "train_epochs": {
                "autoencoder_loss": autoencoder_losses,
                "autoencoder_validation_loss": autoencoder_validation_losses,
                "denoiser_loss": denoiser_losses,
            },
            "community_context": {
                "labels": context.labels.tolist(),
                "centers": context.centers.tolist(),
                "curvature": context.curvature,
            },
            "perturbed_graph_connected": perturbation.is_connected,
        },
        "removed_edges": serialized_edges(perturbation.removed_edges),
        "reconstructions": [
            {
                "index": index,
                "seed": SEED + 2 + index,
                "edges": serialized_edges(edge_set),
                "probabilities": probabilities.tolist(),
            }
            for index, (edge_set, probabilities) in enumerate(
                zip(reconstructions.edge_sets, reconstructions.probabilities)
            )
        ],
        "edge_recovery_frequencies": [
            {"edge": list(edge), "frequency": frequency}
            for edge, frequency in score.edge_recovery_frequencies.items()
        ],
        "scenario_score": score.scenario_score,
        "unique_reconstruction_edge_sets": count_unique_edge_sets(
            {
                "reconstructions": [
                    {"edges": serialized_edges(edge_set)}
                    for edge_set in reconstructions.edge_sets
                ]
            }
        ),
    }

    encoder_path = ARTIFACT_DIR / "autoencoder.pt"
    denoiser_path = ARTIFACT_DIR / "denoiser.pt"
    artifact_path = ARTIFACT_DIR / "results.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(denoiser.state_dict(), denoiser_path)
    save_pilot_artifact(artifact_path, payload)

    with artifact_path.open("r", encoding="utf-8") as artifact_file:
        saved_payload = json.load(artifact_file)
    recomputed_score = rescore_pilot_artifact(saved_payload)
    if recomputed_score != score:
        raise RuntimeError("saved pilot artifact does not reproduce its stability score")

    print("WP5 synthetic CPU pilot: PASS")
    print(
        f"epochs: autoencoder={AUTOENCODER_EPOCHS}, denoiser={DENOISER_EPOCHS}; "
        f"removed_edges={REMOVED_EDGE_COUNT}; K={RECONSTRUCTIONS}; t_star={T_STAR}"
    )
    print(
        f"connected_after_removal={perturbation.is_connected}; "
        f"unique_thresholded_edge_sets={payload['unique_reconstruction_edge_sets']}/{RECONSTRUCTIONS}"
    )
    print(f"scenario_score={score.scenario_score:.4f}")
    print(f"artifact={artifact_path}")
    print(f"checkpoints={encoder_path}, {denoiser_path}")


if __name__ == "__main__":
    main()
