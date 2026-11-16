"""SUPERSEDED by stability/edge_recovery_experiment.py + configs/cora.toml.

Kept only to reproduce earlier pilot artifacts. Known problems (2026-11-16):
removed edges are drawn from all edges (including encoder training edges), the
denoiser and community context use the full original graph, K-means uses 2
clusters, and there is a single cumulative mask series.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from stability.partial_recovery import reconstruct_k
from stability.pilot_config import PilotConfig
from stability.proteins_targeted_classical import (
    rounded_removal_count,
    targeted_edge_order,
)
from stability.synthetic_instances import SyntheticGraph
from stability.wp7_training import T_STAR, make_adjacency, train_graph_models


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
DAMAGE_FRACTIONS = (0.10, 0.20, 0.30)
RECONSTRUCTIONS = 8
CANDIDATE_CHUNK_SIZE = 65_536
MODEL_SEED = 20_261_112
MASK_SEED = 20_261_113
RECONSTRUCTION_SEED = 20_261_114
PLANETOID_CORA_RAW_FILES = (
    "ind.cora.x",
    "ind.cora.tx",
    "ind.cora.allx",
    "ind.cora.y",
    "ind.cora.ty",
    "ind.cora.ally",
    "ind.cora.graph",
    "ind.cora.test.index",
)


@dataclass(frozen=True)
class CoraGraph:
    graph: SyntheticGraph
    content_sha256: str


def load_cora_graph(root: Path = DATA_ROOT, download: bool = False) -> CoraGraph:
    """Load Cora's Planetoid features and canonical undirected edge list."""
    processed_data = root / "Cora" / "processed" / "data.pt"
    raw_dir = root / "Cora" / "raw"
    has_raw_data = all((raw_dir / name).is_file() for name in PLANETOID_CORA_RAW_FILES)
    if not download and not processed_data.is_file() and not has_raw_data:
        raise FileNotFoundError(
            "Cora is not cached; provide the Planetoid files under "
            f"{raw_dir} or rerun with --download"
        )
    dataset = Planetoid(root=str(root), name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    if data.x is None or data.x.ndim != 2 or not torch.isfinite(data.x).all():
        raise ValueError("Cora must provide finite two-dimensional node features")
    node_count = int(data.num_nodes or 0)
    if node_count != data.x.shape[0] or node_count == 0:
        raise ValueError("Cora node features do not match its node count")

    canonical_edges = set()
    for source, target in data.edge_index.t().tolist():
        source, target = int(source), int(target)
        if source == target:
            raise ValueError("Cora contains an unsupported self-loop")
        if not (0 <= source < node_count and 0 <= target < node_count):
            raise ValueError("Cora edge endpoint is outside the node range")
        canonical_edges.add((min(source, target), max(source, target)))
    if not canonical_edges:
        raise ValueError("Cora contains no edges")

    node_ids = tuple(f"Cora:n{index:04d}" for index in range(node_count))
    edges = tuple(
        (node_ids[source], node_ids[target])
        for source, target in sorted(canonical_edges)
    )
    features = data.x.detach().cpu().to(dtype=torch.float32).contiguous()
    graph = SyntheticGraph(
        graph_id="Cora",
        seed=0,
        node_ids=node_ids,
        features=features,
        edges=edges,
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(features.shape)).encode("ascii"))
    digest.update(features.numpy().tobytes())
    digest.update(np.asarray(sorted(canonical_edges), dtype=np.int64).tobytes())
    return CoraGraph(graph=graph, content_sha256=digest.hexdigest())


def cumulative_removal_series(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
    random_seed: int,
) -> dict[str, list[dict[str, object]]]:
    node_order = {node: index for index, node in enumerate(node_ids)}
    canonical_edges = tuple(
        sorted(
            {
                (source, target)
                if node_order[source] < node_order[target]
                else (target, source)
                for source, target in edges
            },
            key=lambda edge: (node_order[edge[0]], node_order[edge[1]]),
        )
    )
    random_order = tuple(
        canonical_edges[index]
        for index in np.random.default_rng(random_seed).permutation(
            len(canonical_edges)
        )
    )
    targeted_order, _ = targeted_edge_order(node_ids, canonical_edges)
    series = {}
    for attack_type, ordered_edges in (
        ("uniform_random", random_order),
        ("targeted_degree_product", targeted_order),
    ):
        levels = []
        prior_removed: set[tuple[str, str]] = set()
        for fraction in DAMAGE_FRACTIONS:
            count = rounded_removal_count(len(canonical_edges), fraction)
            removed_edges = tuple(ordered_edges[:count])
            removed = set(removed_edges)
            if not prior_removed <= removed:
                raise RuntimeError(f"{attack_type} damage masks are not cumulative")
            levels.append(
                {
                    "damage_fraction_requested": fraction,
                    "edges_removed": count,
                    "removed_edges": removed_edges,
                }
            )
            prior_removed = removed
        series[attack_type] = levels
    return series


def edge_recovery_report(
    removed_edges: Sequence[tuple[str, str]],
    reconstructions: Sequence[Iterable[tuple[str, str]]],
    original_edges: Sequence[tuple[str, str]],
) -> dict[str, object]:
    if not removed_edges:
        raise ValueError("removed_edges must not be empty")
    reconstruction_rows = tuple(tuple(row) for row in reconstructions)
    if not reconstruction_rows:
        raise ValueError("at least one reconstruction is required")
    original_keys = {frozenset(edge) for edge in original_edges}
    removed_keys = [frozenset(edge) for edge in removed_edges]
    if len(set(removed_keys)) != len(removed_keys):
        raise ValueError("removed_edges must be unique undirected edges")
    if not set(removed_keys) <= original_keys:
        raise ValueError("removed_edges must be present in original_edges")

    reconstruction_keys = [
        {frozenset(edge) for edge in reconstruction}
        for reconstruction in reconstruction_rows
    ]
    recovered_counts = [
        sum(key in edges for edges in reconstruction_keys) for key in removed_keys
    ]
    count = len(reconstruction_rows)
    histogram = [
        {
            "recovered_count": recovered,
            "edge_count": recovered_counts.count(recovered),
            "percentage": 100.0 * recovered_counts.count(recovered) / len(removed_edges),
        }
        for recovered in range(count + 1)
    ]
    false_positives = [
        {
            "reconstruction": index + 1,
            "count": len(edges - original_keys),
            "edges": [
                list(edge)
                for edge in reconstruction
                if frozenset(edge) not in original_keys
            ],
        }
        for index, (edges, reconstruction) in enumerate(
            zip(reconstruction_keys, reconstruction_rows)
        )
    ]
    return {
        "removed_edge_recovery": {
            "reconstruction_count": count,
            "edges": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "recovered_count": recovered,
                    "r_over_K": f"{recovered}/{count}",
                }
                for edge, recovered in zip(removed_edges, recovered_counts)
            ],
            "histogram": histogram,
        },
        "false_positive_additions": false_positives,
    }


def run_cora_experiment(
    output_dir: Path,
    reconstruction_count: int = RECONSTRUCTIONS,
    tau: float = 0.5,
    download: bool = False,
) -> Path:
    if reconstruction_count <= 0:
        raise ValueError("reconstruction_count must be positive")
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    graph_record = load_cora_graph(download=download)
    graph = graph_record.graph
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Cora run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    config = PilotConfig(curvature=1.0, tau=tau)
    checkpoint_dir = output_dir / "models"
    encoder, denoiser, context, _original_adjacency, training = train_graph_models(
        graph,
        config,
        MODEL_SEED,
        checkpoint_dir,
        validation_first_negative_split=True,
    )
    node_order = {node: index for index, node in enumerate(graph.node_ids)}
    series = cumulative_removal_series(
        graph.node_ids, graph.edges, random_seed=MASK_SEED
    )
    result_series = []
    for attack_index, (attack_type, levels) in enumerate(series.items()):
        attack_levels = []
        for level_index, level in enumerate(levels):
            removed_edges = tuple(level["removed_edges"])
            removed_set = set(removed_edges)
            reconstruction_seed_base = (
                RECONSTRUCTION_SEED
                + attack_index * 100
                + level_index * reconstruction_count
            )
            damaged_edges = tuple(
                edge for edge in graph.edges if edge not in removed_set
            )
            adjacency = make_adjacency(
                len(graph.node_ids),
                tuple(
                    (node_order[source], node_order[target])
                    for source, target in damaged_edges
                ),
            )
            with torch.no_grad():
                damaged_embeddings = encoder.encode(graph.features, adjacency)[2]
                batch = reconstruct_k(
                    encoder=encoder,
                    denoiser=denoiser,
                    damaged_embeddings=damaged_embeddings,
                    damaged_adjacency=adjacency,
                    context=context,
                    node_ids=graph.node_ids,
                    candidate_edges=lambda: itertools.combinations(
                        graph.node_ids, 2
                    ),
                    t_star=T_STAR,
                    num_reconstructions=reconstruction_count,
                    seed=reconstruction_seed_base,
                    config=config,
                    candidate_chunk_size=CANDIDATE_CHUNK_SIZE,
                    retain_probabilities=False,
                    validate_candidates=False,
                )
            if any(probabilities.numel() for probabilities in batch.probabilities):
                raise RuntimeError("Cora run unexpectedly retained decoder probabilities")
            report = edge_recovery_report(
                removed_edges, batch.edge_sets, graph.edges
            )
            attack_levels.append(
                {
                    "damage_fraction_requested": level["damage_fraction_requested"],
                    "damage_fraction_realized": len(removed_edges) / len(graph.edges),
                    "original_edge_count": len(graph.edges),
                    "damaged_edge_count": len(damaged_edges),
                    "edges_removed": len(removed_edges),
                    "reconstruction_seed_base": reconstruction_seed_base,
                    "reconstruction_seeds": [
                        reconstruction_seed_base + index
                        for index in range(reconstruction_count)
                    ],
                    **report,
                }
            )
        result_series.append(
            {"attack_type": attack_type, "damage_levels": attack_levels}
        )

    checkpoint_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("autoencoder", checkpoint_dir / "Cora" / "autoencoder.pt"),
            ("denoiser", checkpoint_dir / "Cora" / "denoiser.pt"),
        )
    }
    summary = {
        "artifact_type": "Cora cumulative edge recovery experiment",
        "status": "complete",
        "dataset": "Cora",
        "graph_content_sha256": graph_record.content_sha256,
        "node_features": "row-normalized Planetoid Cora bag-of-words features",
        "node_count": len(graph.node_ids),
        "original_edge_count": len(graph.edges),
        "primary_evaluation_damage_fraction": 0.30,
        "reconstruction_count": reconstruction_count,
        "threshold_tau": tau,
        "mask_seed": MASK_SEED,
        "reconstruction_seed": RECONSTRUCTION_SEED,
        "damage_fractions": list(DAMAGE_FRACTIONS),
        "cumulative_masks": True,
        "device": "cpu",
        "attack_series": result_series,
        "model": {
            "seed": MODEL_SEED,
            "t_star": T_STAR,
            "diffusion_schedule_length": denoiser.num_timesteps,
            "checkpoint_sha256": checkpoint_hashes,
            "training": training,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary_path
