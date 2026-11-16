from pathlib import Path
import itertools

import numpy as np
import torch

from stability.edge_perturbation import remove_edges
from stability.imdb_binary_dataset import (
    load_imdb_binary_graphs,
    select_imdb_binary_splits,
    with_node_identity_features,
)
from hyperbolic_learning.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from models.base_models import LPModel
from models.direction_diffusion import diffusion
from stability.partial_recovery import compute_community_context, reconstruct_k
from stability.pilot_config import PilotConfig
from stability.imdb_binary_recovery_pilot import _sampling_intervals
from stability.wp7_training import (
    _edge_split_validation_first,
    make_adjacency,
    model_args,
)


def _write_graph(root: Path, indicator: str, edges: str) -> None:
    raw = root / "IMDB-BINARY" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "IMDB-BINARY_graph_indicator.txt").write_text(indicator)
    (raw / "IMDB-BINARY_A.txt").write_text(edges)
    (raw / "IMDB-BINARY_graph_labels.txt").write_text("1\n")


def test_loader_uses_indicator_node_order_and_constant_features(tmp_path):
    _write_graph(
        tmp_path,
        "".join("1\n" for _ in range(12)),
        "1,2\n2,3\n3,4\n4,5\n5,6\n6,7\n7,8\n8,9\n9,10\n10,11\n11,12\n12,1\n",
    )
    graphs, _, metadata = load_imdb_binary_graphs(tmp_path)

    assert len(graphs) == 1
    graph = graphs[0]
    assert graph.node_ids == tuple(
        f"IMDB-BINARY-0001:n{index:03d}" for index in range(12)
    )
    assert graph.features.shape == (12, 1)
    assert torch.equal(graph.features, torch.ones((12, 1)))
    assert len(graph.edges) == 12
    assert metadata["node_attribute_files"] == []
    assert metadata["node_label_files"] == []
    assert metadata["graph_labels_used"] is False


def test_one_hot_identity_features_align_with_node_ids(tmp_path):
    _write_graph(
        tmp_path,
        "".join("1\n" for _ in range(12)),
        "1,2\n2,3\n3,4\n4,5\n5,6\n6,7\n7,8\n8,9\n9,10\n10,11\n11,12\n12,1\n",
    )
    graph = load_imdb_binary_graphs(tmp_path)[0][0]
    enriched = with_node_identity_features(graph)

    assert enriched.node_ids == graph.node_ids
    assert enriched.features.shape == (12, 13)
    assert torch.equal(enriched.features[:, 0], torch.ones(12))
    assert torch.equal(enriched.features[:, 1:], torch.eye(12))
    assert enriched.content_sha256 != graph.content_sha256


def test_validation_first_edge_split_is_disjoint_and_reproducible(tmp_path):
    _write_graph(
        tmp_path,
        "".join("1\n" for _ in range(12)),
        "1,2\n2,3\n3,4\n4,5\n5,6\n6,7\n7,8\n8,9\n9,10\n10,11\n11,12\n12,1\n",
    )
    graph = load_imdb_binary_graphs(tmp_path)[0][0]
    first = _edge_split_validation_first(graph.edges, graph, 29)
    second = _edge_split_validation_first(graph.edges, graph, 29)

    train_edges, validation_edges, train_negatives, validation_negatives = first
    assert first == second
    assert train_edges and validation_edges and train_negatives and validation_negatives
    assert not (set(train_edges) & set(validation_edges))
    assert not (set(train_negatives) & set(validation_negatives))
    assert not (set(train_edges + validation_edges) & set(train_negatives + validation_negatives))
    assert len(validation_edges) == len(validation_negatives)


def test_sampling_summary_accepts_json_list_probabilities():
    graph_results = [
        {
            "graph_id": "fixture",
            "original_edges": [["a", "b"]],
            "cases": [
                {
                    "attack_type": "uniform_random",
                    "damage_fraction_requested": 0.1,
                    "removed_indices": [0],
                    "nonedge_indices": [1, 2],
                    "baseline_probabilities": [0.8, 0.2, 0.9],
                    "partial_probabilities": [
                        [0.8, 0.2, 0.9] for _ in range(8)
                    ],
                }
            ],
        }
    ]

    summary = _sampling_intervals(
        graph_results,
        {"partial_recovery": 0.5, "no_diffusion_baseline": 0.5},
        seed=7,
    )

    metrics = summary[0]["methods"]
    assert metrics["partial_recovery"]["recovery"]["mean"] == 1.0
    assert metrics["partial_recovery"]["fpr"]["mean"] == 0.5
    assert metrics["no_diffusion_baseline"]["recovery"]["mean"] == 1.0
    assert metrics["no_diffusion_baseline"]["fpr"]["mean"] == 0.5


def test_node_ids_and_feature_rows_survive_perturbation_with_isolated_node(tmp_path):
    _write_graph(
        tmp_path,
        "".join("1\n" for _ in range(12)),
        "1,2\n2,3\n3,4\n4,5\n5,6\n6,7\n7,8\n8,9\n9,10\n10,11\n11,12\n12,1\n",
    )
    graph = with_node_identity_features(load_imdb_binary_graphs(tmp_path)[0][0])
    incident = tuple(edge for edge in graph.edges if graph.node_ids[0] in edge)
    perturbed = None
    for seed in range(1000):
        trial = remove_edges(
            graph.node_ids, graph.features, graph.edges, len(incident), seed
        )
        if not any(graph.node_ids[0] in edge for edge in trial.edges):
            perturbed = trial
            break
    assert perturbed is not None
    assert perturbed.node_ids == graph.node_ids
    assert torch.equal(perturbed.features, graph.features)
    assert perturbed.features.shape == (len(graph.node_ids), len(graph.node_ids) + 1)
    assert not any(graph.node_ids[0] in edge for edge in perturbed.edges)

    np.random.seed(11)
    torch.manual_seed(11)
    config = PilotConfig(curvature=1.0, tau=0.5)
    args = model_args(graph, config)
    encoder = LPModel(args).cpu().eval()
    denoiser = diffusion(args, dim=args.dim, curvature=config.curvature).cpu().eval()
    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    original_edges = tuple(
        (node_index[source], node_index[target]) for source, target in graph.edges
    )
    damaged_edges = tuple(
        (node_index[source], node_index[target])
        for source, target in perturbed.edges
    )
    original_adjacency = make_adjacency(len(graph.node_ids), original_edges)
    damaged_adjacency = make_adjacency(len(graph.node_ids), damaged_edges)
    context = compute_community_context(
        encoder,
        graph.features,
        original_adjacency,
        HyperbolicKMeans(n_clusters=2, curvature=1.0),
        graph.node_ids,
    )
    assert context.node_ids == graph.node_ids
    assert context.labels.shape[0] == len(graph.node_ids)
    assert damaged_adjacency.shape == (len(graph.node_ids), len(graph.node_ids))
    with torch.no_grad():
        embeddings = encoder.encode(perturbed.features, damaged_adjacency)[2]
    candidates = tuple(itertools.combinations(graph.node_ids, 2))
    batch = reconstruct_k(
        encoder,
        denoiser,
        embeddings,
        damaged_adjacency,
        context,
        graph.node_ids,
        candidates,
        t_star=0,
        num_reconstructions=1,
        seed=29,
        config=config,
    )
    assert embeddings.shape[0] == len(graph.node_ids)
    assert len(batch.probabilities) == 1
    assert batch.probabilities[0].shape[0] == len(candidates)
    assert torch.isfinite(batch.probabilities[0]).all()
    assert all(
        source in graph.node_ids and target in graph.node_ids
        for source, target in batch.edge_sets[0]
    )


def test_split_is_deterministic_disjoint_and_uses_graphs_not_labels(tmp_path):
    raw = tmp_path / "IMDB-BINARY" / "raw"
    raw.mkdir(parents=True)
    indicators = []
    edges = []
    offset = 0
    for graph_index in range(10):
        node_count = 12 + graph_index
        indicators.extend([str(graph_index + 1)] * node_count)
        edges.extend(
            f"{offset + index + 1},{offset + (index + 1) % node_count + 1}"
            for index in range(node_count)
        )
        offset += node_count
    _write_graph(
        tmp_path,
        "\n".join(indicators) + "\n",
        "\n".join(edges) + "\n",
    )
    (raw / "IMDB-BINARY_graph_labels.txt").write_text(
        "\n".join(str(index % 2) for index in range(10)) + "\n"
    )
    graphs = load_imdb_binary_graphs(tmp_path)[0]
    first = select_imdb_binary_splits(graphs)
    second = select_imdb_binary_splits(graphs)

    assert [g.graph_id for g in first.calibration] == [
        g.graph_id for g in second.calibration
    ]
    assert [g.graph_id for g in first.evaluation] == [
        g.graph_id for g in second.evaluation
    ]
    assert len(first.calibration) == len(first.evaluation) == 5
    assert not (
        {g.graph_id for g in first.calibration}
        & {g.graph_id for g in first.evaluation}
    )
