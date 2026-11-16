from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from stability.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.partial_recovery import compute_community_context
from stability.pilot_config import PilotConfig
from stability.synthetic_instances import SyntheticGraph


AUTOENCODER_EPOCHS = 30
DENOISER_EPOCHS = 50
REMOVED_EDGE_COUNT = 3
RECONSTRUCTIONS = 8
T_STAR = 7


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


def _edge_split(edges: tuple[tuple[str, str], ...], graph: SyntheticGraph, seed: int):
    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    indexed_edges = tuple(
        (node_index[source], node_index[target]) for source, target in edges
    )
    permutation = np.random.default_rng(seed).permutation(len(indexed_edges))
    validation_count = max(1, len(indexed_edges) // 5)
    validation_indices = set(permutation[:validation_count].tolist())
    train_edges = tuple(
        edge for index, edge in enumerate(indexed_edges) if index not in validation_indices
    )
    validation_edges = tuple(
        edge for index, edge in enumerate(indexed_edges) if index in validation_indices
    )
    all_pairs = tuple(itertools.combinations(range(len(graph.node_ids)), 2))
    positives = {frozenset(edge) for edge in indexed_edges}
    negatives = tuple(edge for edge in all_pairs if frozenset(edge) not in positives)
    negative_order = np.random.default_rng(seed + 1).permutation(len(negatives))
    train_negatives = tuple(
        negatives[index] for index in negative_order[: len(train_edges)]
    )
    validation_negatives = tuple(
        negatives[index]
        for index in negative_order[
            len(train_edges) : len(train_edges) + len(validation_edges)
        ]
    )
    return train_edges, validation_edges, train_negatives, validation_negatives


def _edge_split_validation_first(
    edges: tuple[tuple[str, str], ...], graph: SyntheticGraph, seed: int
):
    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    indexed_edges = tuple(
        (node_index[source], node_index[target]) for source, target in edges
    )
    if len(indexed_edges) < 2:
        raise ValueError(f"{graph.graph_id}: at least two positive edges are required")
    permutation = np.random.default_rng(seed).permutation(len(indexed_edges))
    validation_count = max(1, len(indexed_edges) // 5)
    validation_indices = set(permutation[:validation_count].tolist())
    train_edges = tuple(
        edge
        for index, edge in enumerate(indexed_edges)
        if index not in validation_indices
    )
    validation_edges = tuple(
        edge
        for index, edge in enumerate(indexed_edges)
        if index in validation_indices
    )

    all_pairs = tuple(itertools.combinations(range(len(graph.node_ids)), 2))
    positive_pairs = {frozenset(edge) for edge in indexed_edges}
    true_nonedges = tuple(
        pair for pair in all_pairs if frozenset(pair) not in positive_pairs
    )
    negative_order = np.random.default_rng(seed + 1).permutation(
        len(true_nonedges)
    )
    if len(true_nonedges) < validation_count + 1:
        raise ValueError(
            f"{graph.graph_id}: validation-first split needs at least "
            f"{validation_count + 1} true nonedges, found {len(true_nonedges)}"
        )

    validation_negatives = tuple(
        true_nonedges[index] for index in negative_order[:validation_count]
    )
    remaining_negative_indices = negative_order[validation_count:]
    train_negative_count = min(len(train_edges), len(remaining_negative_indices))
    train_negatives = tuple(
        true_nonedges[index]
        for index in remaining_negative_indices[:train_negative_count]
    )
    if not train_edges or not validation_edges or not train_negatives or not validation_negatives:
        raise ValueError(f"{graph.graph_id}: validation-first split has an empty group")
    return train_edges, validation_edges, train_negatives, validation_negatives


def _edge_tensor(edges, device):
    return torch.tensor(edges, dtype=torch.long, device=device).reshape(-1, 2)


def _loss(model, embeddings, positives, negatives):
    def logits(edges):
        source = embeddings[edges[:, 0]]
        target = embeddings[edges[:, 1]]
        distance = model.manifold.sqdist(source, target, model.c)
        if not torch.isfinite(distance).all():
            raise RuntimeError("link decoder produced non-finite Poincare distances")
        return -(distance - model.dc.r) / model.dc.t

    positive_logits = logits(positives)
    negative_logits = logits(negatives)
    if not torch.isfinite(positive_logits).all() or not torch.isfinite(
        negative_logits
    ).all():
        raise RuntimeError("link decoder produced non-finite logits")
    return F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits)
    ) + F.binary_cross_entropy_with_logits(
        negative_logits, torch.zeros_like(negative_logits)
    )


def model_args(
    graph: SyntheticGraph,
    config: PilotConfig,
    final_activation: str = "relu",
):
    args = SimpleNamespace(
        manifold="PoincareBall",
        cuda=-1,
        device="cpu",
        feat_dim=graph.features.shape[1],
        dim=4,
        model="HGCN",
        n_nodes=len(graph.node_ids),
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
        final_activation=final_activation,
    )
    return config.apply_to_args(args)


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_checkpoint(path: Path | None, model) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)


def train_graph_models(
    graph: SyntheticGraph,
    config: PilotConfig,
    model_seed: int,
    checkpoint_dir: Path | None = None,
    validation_first_negative_split: bool = False,
):
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    args = model_args(graph, config)
    encoder = StabilityLPModel(args).cpu()
    denoiser = StabilityDiffusion(args, dim=args.dim, curvature=config.curvature).cpu()
    edge_splitter = (
        _edge_split_validation_first
        if validation_first_negative_split
        else _edge_split
    )
    train_edges, validation_edges, train_negatives, validation_negatives = edge_splitter(
        graph.edges, graph, model_seed + 1
    )
    train_adjacency = make_adjacency(len(graph.node_ids), train_edges)
    original_index = {node: index for index, node in enumerate(graph.node_ids)}
    original_index_edges = tuple(
        (original_index[source], original_index[target]) for source, target in graph.edges
    )
    original_adjacency = make_adjacency(len(graph.node_ids), original_index_edges)
    train_positive = _edge_tensor(train_edges, graph.features.device)
    train_negative = _edge_tensor(train_negatives, graph.features.device)
    validation_positive = _edge_tensor(validation_edges, graph.features.device)
    validation_negative = _edge_tensor(validation_negatives, graph.features.device)

    autoencoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    ae_losses = []
    ae_validation_losses = []
    for _ in range(AUTOENCODER_EPOCHS):
        encoder.train()
        autoencoder_optimizer.zero_grad()
        embeddings = encoder.encode(graph.features, train_adjacency)[2]
        loss = _loss(encoder, embeddings, train_positive, train_negative)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{graph.graph_id}: non-finite autoencoder loss")
        loss.backward()
        autoencoder_optimizer.step()
        ae_losses.append(float(loss.detach()))

        encoder.eval()
        with torch.no_grad():
            validation_embeddings = encoder.encode(graph.features, train_adjacency)[2]
            validation_loss = _loss(
                encoder,
                validation_embeddings,
                validation_positive,
                validation_negative,
            )
        if not torch.isfinite(validation_loss):
            raise RuntimeError(f"{graph.graph_id}: non-finite autoencoder validation loss")
        ae_validation_losses.append(float(validation_loss))

    encoder.eval()
    with torch.no_grad():
        original_embeddings = encoder.encode(graph.features, original_adjacency)[2]
    if not torch.isfinite(original_embeddings).all():
        raise RuntimeError(f"{graph.graph_id}: non-finite original embeddings")
    if not torch.all(torch.linalg.vector_norm(original_embeddings, dim=1) < 1.0):
        raise RuntimeError(f"{graph.graph_id}: embeddings outside the c=1 unit ball")

    clusterer = HyperbolicKMeans(n_clusters=2, curvature=config.curvature)
    context = compute_community_context(
        encoder,
        graph.features,
        original_adjacency,
        clusterer,
        graph.node_ids,
    )
    denoiser_optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    denoiser_losses = []
    for _ in range(DENOISER_EPOCHS):
        denoiser.train()
        denoiser_optimizer.zero_grad()
        loss = denoiser(
            original_embeddings.detach().clone(),
            original_adjacency,
            context.labels,
            context.centers,
            restrict=True,
            graphset=False,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{graph.graph_id}: non-finite denoiser loss")
        loss.backward()
        denoiser_optimizer.step()
        denoiser_losses.append(float(loss.detach()))

    encoder.eval()
    denoiser.eval()
    for model in (encoder, denoiser):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            if parameter.grad is not None:
                parameter.grad = None

    graph_dir = checkpoint_dir / graph.graph_id if checkpoint_dir is not None else None
    encoder_path = graph_dir / "autoencoder.pt" if graph_dir is not None else None
    denoiser_path = graph_dir / "denoiser.pt" if graph_dir is not None else None
    _save_checkpoint(encoder_path, encoder)
    _save_checkpoint(denoiser_path, denoiser)

    training = {
        "autoencoder_epochs": len(ae_losses),
        "autoencoder_loss": ae_losses,
        "autoencoder_validation_loss": ae_validation_losses,
        "denoiser_epochs": len(denoiser_losses),
        "denoiser_loss": denoiser_losses,
        "autoencoder_checkpoint_sha256": _state_hash(encoder),
        "denoiser_checkpoint_sha256": _state_hash(denoiser),
        "encoder_frozen": all(not p.requires_grad for p in encoder.parameters()),
        "denoiser_frozen": all(not p.requires_grad for p in denoiser.parameters()),
        "train_edges": [
            [graph.node_ids[source], graph.node_ids[target]]
            for source, target in train_edges
        ],
        "validation_edges": [
            [graph.node_ids[source], graph.node_ids[target]]
            for source, target in validation_edges
        ],
        "validation_first_negative_split": validation_first_negative_split,
        "train_negative_edges": [
            [graph.node_ids[source], graph.node_ids[target]]
            for source, target in train_negatives
        ],
        "validation_negative_edges": [
            [graph.node_ids[source], graph.node_ids[target]]
            for source, target in validation_negatives
        ],
    }
    return (
        encoder,
        denoiser,
        context,
        original_adjacency,
        training,
    )
