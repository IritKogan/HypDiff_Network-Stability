from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Callable, Hashable, Iterable, Sequence

import numpy as np
import torch

from hyperbolic_learning.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.edge_perturbation import Edge
from stability.pilot_config import PilotConfig
from stability.scoring import StabilityScore, score_recovery


@dataclass(frozen=True)
class CommunityContext:
    node_ids: tuple[Hashable, ...]
    labels: torch.Tensor
    centers: torch.Tensor
    curvature: float


@dataclass(frozen=True)
class ReconstructionBatch:
    edge_sets: tuple[tuple[Edge, ...], ...]
    probabilities: tuple[torch.Tensor, ...]


def validate_curvature(encoder, denoiser, clusterer, config: PilotConfig) -> None:
    encoder_curvature = float(encoder.c.detach().reshape(-1)[0].item())
    denoiser_curvature = float(denoiser.c)
    cluster_curvature = float(clusterer.curvature)
    values = (encoder_curvature, denoiser_curvature, cluster_curvature)
    if any(not np.isclose(value, config.curvature) for value in values):
        raise ValueError(
            "encoder, diffusion, clustering, and pilot config curvature must match"
        )


def compute_community_context(
    encoder,
    features: torch.Tensor,
    original_adjacency: torch.Tensor,
    clusterer: HyperbolicKMeans,
    node_ids: Sequence[Hashable],
) -> CommunityContext:
    """Encode and cluster the original graph once, before edge removal."""
    if len(node_ids) != features.shape[0] or len(set(node_ids)) != len(node_ids):
        raise ValueError("node_ids must uniquely identify every feature row")
    with torch.no_grad():
        embeddings, _, _, _ = encoder.encode(features, original_adjacency)
    points = embeddings.detach().cpu().numpy()
    if not np.isfinite(points).all():
        raise ValueError("original-graph embeddings must be finite")
    if np.any(np.linalg.norm(points, axis=1) >= 1.0):
        raise ValueError("c=1.0 community embeddings must lie inside the unit ball")
    centers, labels = clusterer.fit(points)
    if not np.isfinite(centers).all() or not np.isfinite(labels).all():
        raise ValueError("community clustering produced non-finite results")
    center_tensor = torch.as_tensor(
        centers, dtype=embeddings.dtype, device=embeddings.device
    )
    center_tensor = clusterer.manifold.proj(
        center_tensor, c=clusterer.curvature
    )
    if not torch.isfinite(center_tensor).all() or not torch.all(
        torch.linalg.vector_norm(center_tensor, dim=1) < 1.0
    ):
        raise ValueError("community centers must lie inside the unit ball")
    return CommunityContext(
        node_ids=tuple(node_ids),
        labels=torch.as_tensor(labels, dtype=torch.long, device=features.device).detach(),
        centers=center_tensor.detach(),
        curvature=float(clusterer.curvature),
    )


def decode_edge_set(
    encoder,
    embeddings: torch.Tensor,
    node_ids: Sequence[object],
    candidate_edges: Iterable[Edge],
    tau: float,
    candidate_chunk_size: int | None = None,
    retain_probabilities: bool = True,
    validate_candidates: bool = True,
) -> tuple[tuple[Edge, ...], torch.Tensor]:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    if candidate_chunk_size is not None and candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    node_order = {node: index for index, node in enumerate(node_ids)}
    if len(node_order) != len(node_ids):
        raise ValueError("node_ids must be unique")
    seen = set()
    chunk_size = candidate_chunk_size or 2**63 - 1
    predicted_edges = []
    probability_chunks = []
    iterator = iter(candidate_edges)
    while chunk := tuple(islice(iterator, chunk_size)):
        pairs = []
        for source, target in chunk:
            if source not in node_order or target not in node_order:
                raise ValueError(
                    f"candidate edge endpoint missing from node_ids: {(source, target)!r}"
                )
            if source == target:
                raise ValueError("candidate edges must not contain self-loops")
            if validate_candidates:
                key = frozenset((source, target))
                if key in seen:
                    raise ValueError(
                        f"candidate edges contain duplicate undirected edge: "
                        f"{(source, target)!r}"
                    )
                seen.add(key)
            pairs.append((node_order[source], node_order[target]))

        edge_index = torch.tensor(
            pairs, dtype=torch.long, device=embeddings.device
        ).reshape(-1, 2)
        probabilities = encoder.decode(embeddings, edge_index)
        if not torch.isfinite(probabilities).all():
            raise ValueError("decoder produced non-finite probabilities")
        predicted_edges.extend(
            edge
            for edge, is_present in zip(chunk, (probabilities >= tau).tolist())
            if is_present
        )
        if retain_probabilities:
            probability_chunks.append(probabilities.detach())
    if retain_probabilities:
        probabilities = (
            torch.cat(probability_chunks)
            if probability_chunks
            else torch.empty(0, dtype=embeddings.dtype, device=embeddings.device)
        )
    else:
        probabilities = torch.empty(0, dtype=embeddings.dtype, device=embeddings.device)
    return tuple(predicted_edges), probabilities


def decode_without_diffusion(
    encoder,
    damaged_embeddings: torch.Tensor,
    node_ids: Sequence[object],
    candidate_edges: Sequence[Edge],
    tau: float,
) -> tuple[tuple[Edge, ...], torch.Tensor]:
    """Directly decode damaged embeddings; this is the no-diffusion baseline."""
    return decode_edge_set(
        encoder, damaged_embeddings, node_ids, candidate_edges, tau
    )


def reconstruct_k(
    encoder,
    denoiser,
    damaged_embeddings: torch.Tensor,
    damaged_adjacency: torch.Tensor,
    context: CommunityContext,
    node_ids: Sequence[object],
    candidate_edges: Iterable[Edge] | Callable[[], Iterable[Edge]],
    t_star: int,
    num_reconstructions: int,
    seed: int,
    config: PilotConfig,
    candidate_chunk_size: int | None = None,
    retain_probabilities: bool = True,
    validate_candidates: bool = True,
) -> ReconstructionBatch:
    if num_reconstructions <= 0:
        raise ValueError("num_reconstructions must be positive")
    if not np.isclose(float(encoder.c.detach().reshape(-1)[0].item()), config.curvature):
        raise ValueError("encoder curvature does not match pilot config")
    if not np.isclose(float(denoiser.c), config.curvature):
        raise ValueError("diffusion curvature does not match pilot config")
    if not np.isclose(context.curvature, config.curvature):
        raise ValueError("community context curvature does not match pilot config")
    if context.labels.shape[0] != damaged_embeddings.shape[0]:
        raise ValueError("community context must have one label per node")
    if tuple(node_ids) != context.node_ids:
        raise ValueError("reconstruction node_ids must match the frozen context order")

    if callable(candidate_edges):
        candidate_factory = candidate_edges
    else:
        first_iterator = iter(candidate_edges)
        if first_iterator is candidate_edges:
            candidate_edges = tuple(first_iterator)
        candidate_factory = lambda: iter(candidate_edges)

    edge_sets = []
    probabilities = []
    labels = context.labels.to(device=damaged_embeddings.device, dtype=torch.long)
    centers = context.centers.to(
        device=damaged_embeddings.device, dtype=damaged_embeddings.dtype
    )
    for reconstruction_index in range(num_reconstructions):
        generator = torch.Generator(device=damaged_embeddings.device)
        generator.manual_seed(seed + reconstruction_index)
        recovered = denoiser.sample_from_tstar(
            damaged_embeddings=damaged_embeddings,
            damaged_adjacency=damaged_adjacency,
            labels=labels,
            prototypes=centers,
            t_star=t_star,
            generator=generator,
        )
        edge_set, edge_probabilities = decode_edge_set(
            encoder,
            recovered,
            node_ids,
            candidate_factory(),
            config.tau,
            candidate_chunk_size=candidate_chunk_size,
            retain_probabilities=retain_probabilities,
            validate_candidates=validate_candidates,
        )
        edge_sets.append(edge_set)
        probabilities.append(edge_probabilities)
    return ReconstructionBatch(tuple(edge_sets), tuple(probabilities))


def score_reconstructions(
    batch: ReconstructionBatch, removed_edges: Iterable[Edge]
) -> StabilityScore:
    return score_recovery(tuple(removed_edges), batch.edge_sets)
