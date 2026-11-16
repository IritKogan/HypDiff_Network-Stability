from __future__ import annotations

import hashlib
import itertools
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from stability.hypdiff_models import StabilityLPModel
from stability.edge_perturbation import remove_edges
from stability.pilot_config import PilotConfig
from stability.synthetic_instances import SyntheticGraph
from stability.wp7_training import (
    AUTOENCODER_EPOCHS,
    REMOVED_EDGE_COUNT,
    _edge_tensor,
    _loss,
    make_adjacency,
    model_args,
)


VARIANTS = ("intact_input", "randomly_damaged_input")
NEGATIVE_EDGE_COUNT = 15


def _mask_key(edges):
    return frozenset(frozenset(edge) for edge in edges)


def sample_training_mask(
    edges: tuple[tuple[str, str], ...],
    seed: int,
    excluded_mask,
    count: int = REMOVED_EDGE_COUNT,
):
    rng = random.Random(seed)
    for _ in range(1000):
        candidate = tuple(rng.sample(edges, count))
        if _mask_key(candidate) != excluded_mask:
            return candidate
    raise RuntimeError("could not sample a training mask distinct from evaluation mask")


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rank_metrics(candidate_edges, removed_edges, probabilities):
    removed_keys = {frozenset(edge) for edge in removed_edges}
    labels = np.asarray(
        [int(frozenset(edge) in removed_keys) for edge in candidate_edges],
        dtype=np.int64,
    )
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("candidate probabilities must be finite and aligned")
    if len(set(labels.tolist())) != 2:
        raise ValueError("candidate pairs must include removed edges and true nonedges")
    order = sorted(
        range(len(candidate_edges)),
        key=lambda index: (-probabilities[index], candidate_edges[index]),
    )
    top_three = {frozenset(candidate_edges[index]) for index in order[:3]}
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "removed_edges_in_top_three": len(top_three & removed_keys),
        "top_three_candidates": [
            list(candidate_edges[index]) for index in order[:3]
        ],
    }


def _train_variant(
    graph: SyntheticGraph,
    variant: str,
    model_seed: int,
    training_mask_seed: int,
    excluded_evaluation_mask,
    original_edges_indexed,
    positive_edges,
    negative_edges,
):
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    config = PilotConfig(curvature=1.0, tau=0.5)
    model = StabilityLPModel(model_args(graph, config)).cpu()
    initial_hash = _state_hash(model)
    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    original_adjacency = make_adjacency(len(graph.node_ids), original_edges_indexed)
    training_masks = []
    losses = []
    started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(AUTOENCODER_EPOCHS):
        if variant == "intact_input":
            input_adjacency = original_adjacency
        elif variant == "randomly_damaged_input":
            mask = sample_training_mask(
                graph.edges,
                seed=training_mask_seed + epoch,
                excluded_mask=excluded_evaluation_mask,
            )
            if _mask_key(mask) == excluded_evaluation_mask:
                raise RuntimeError("training used the reserved evaluation mask")
            training_masks.append([list(edge) for edge in mask])
            damaged_edges = tuple(
                edge for edge in graph.edges if frozenset(edge) not in _mask_key(mask)
            )
            indexed_damaged = tuple(
                (node_index[source], node_index[target])
                for source, target in damaged_edges
            )
            input_adjacency = make_adjacency(len(graph.node_ids), indexed_damaged)
        else:
            raise ValueError(f"unknown encoder training variant: {variant}")

        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(graph.features, input_adjacency)[2]
        loss = _loss(model, embeddings, positive_edges, negative_edges)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{graph.graph_id}/{variant}: non-finite loss")
        loss.backward()
        if any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"{graph.graph_id}/{variant}: non-finite gradient")
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.grad is not None:
            parameter.grad = None
    return {
        "model": model,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": _state_hash(model),
        "epochs": len(losses),
        "losses": losses,
        "training_masks": training_masks,
        "training_mask_count": len(training_masks),
        "training_mask_unique_count": len(
            {_mask_key(tuple(tuple(edge) for edge in mask)) for mask in training_masks}
        ),
        "encoder_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "duration_seconds": time.perf_counter() - started,
    }


def _decode(model, embeddings, node_ids, candidate_edges):
    index = {node: position for position, node in enumerate(node_ids)}
    pairs = torch.tensor(
        [(index[source], index[target]) for source, target in candidate_edges],
        dtype=torch.long,
    ).reshape(-1, 2)
    with torch.no_grad():
        probabilities = model.decode(embeddings, pairs)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("decoder returned non-finite probabilities")
    return probabilities.cpu().numpy()


def run_graph(graph: SyntheticGraph, seed_offset: int):
    node_index = {node: index for index, node in enumerate(graph.node_ids)}
    original_edges_indexed = tuple(
        (node_index[source], node_index[target]) for source, target in graph.edges
    )
    node_ids = graph.node_ids
    features = graph.features
    evaluation_perturbation_seed = graph.seed + 11000
    evaluation_mask = remove_edges(
        node_ids,
        features,
        graph.edges,
        REMOVED_EDGE_COUNT,
        evaluation_perturbation_seed,
    )
    excluded_mask = _mask_key(evaluation_mask.removed_edges)
    intact_keys = {frozenset(edge) for edge in graph.edges}
    all_pairs = tuple(
        (node_ids[source], node_ids[target])
        for source, target in itertools.combinations(range(len(node_ids)), 2)
    )
    candidate_edges = tuple(
        edge
        for edge in all_pairs
        if frozenset(edge) not in intact_keys
        or frozenset(edge) in excluded_mask
    )
    remaining_keys = {frozenset(edge) for edge in evaluation_mask.edges}
    damaged_edges = tuple(
        edge for edge in graph.edges if frozenset(edge) in remaining_keys
    )
    removed_edges = tuple(
        edge for edge in graph.edges if frozenset(edge) not in remaining_keys
    )
    if _mask_key(removed_edges) != excluded_mask or len(candidate_edges) != 54:
        raise RuntimeError("evaluation perturbation or candidate set is invalid")
    damaged_indexed = tuple(
        (node_index[source], node_index[target]) for source, target in damaged_edges
    )
    damaged_adjacency = make_adjacency(len(node_ids), damaged_indexed)

    # Both variants reconstruct all intact positives and the same fixed nonedges.
    nonedges = tuple(edge for edge in all_pairs if frozenset(edge) not in intact_keys)
    negative_rng = np.random.default_rng(graph.seed + seed_offset + 1)
    negative_indices = negative_rng.choice(
        len(nonedges), size=NEGATIVE_EDGE_COUNT, replace=False
    )
    negative_edges = tuple(nonedges[index] for index in negative_indices)
    positive_edges = tuple(
        (node_index[source], node_index[target]) for source, target in graph.edges
    )
    negative_edges_indexed = tuple(
        (node_index[source], node_index[target]) for source, target in negative_edges
    )
    positive_tensor = _edge_tensor(positive_edges, torch.device("cpu"))
    negative_tensor = _edge_tensor(negative_edges_indexed, torch.device("cpu"))
    model_seed = graph.seed + seed_offset
    baseline = _train_variant(
        graph,
        "intact_input",
        model_seed,
        graph.seed + seed_offset + 100,
        excluded_mask,
        original_edges_indexed,
        positive_tensor,
        negative_tensor,
    )
    augmented = _train_variant(
        graph,
        "randomly_damaged_input",
        model_seed,
        graph.seed + seed_offset + 200,
        excluded_mask,
        original_edges_indexed,
        positive_tensor,
        negative_tensor,
    )
    if baseline["initial_state_sha256"] != augmented["initial_state_sha256"]:
        raise RuntimeError(f"{graph.graph_id}: variants did not start identically")
    with torch.no_grad():
        damaged_embeddings = {
            name: trained["model"].encode(features, damaged_adjacency)[2]
            for name, trained in (
                ("intact_input", baseline),
                ("randomly_damaged_input", augmented),
            )
        }
    if not all(torch.isfinite(embedding).all() for embedding in damaged_embeddings.values()):
        raise RuntimeError(f"{graph.graph_id}: evaluation embeddings are non-finite")
    scores = {
        name: _decode(trained["model"], damaged_embeddings[name], node_ids, candidate_edges)
        for name, trained in (
            ("intact_input", baseline),
            ("randomly_damaged_input", augmented),
        )
    }
    metrics = {
        name: _rank_metrics(candidate_edges, removed_edges, probabilities)
        for name, probabilities in scores.items()
    }
    return {
        "graph_id": graph.graph_id,
        "graph_seed": graph.seed,
        "model_seed": model_seed,
        "evaluation_perturbation_seed": evaluation_perturbation_seed,
        "evaluation_mask": [list(edge) for edge in removed_edges],
        "training_masks_exclude_evaluation_mask": True,
        "features_identical_between_variants": True,
        "reconstruction_target": "all 15 original graph edges plus the same 15 fixed true nonedges",
        "candidate_edges": [list(edge) for edge in candidate_edges],
        "candidate_edge_count": len(candidate_edges),
        "removed_edge_count": len(removed_edges),
        "training": {
            "intact_input": {
                key: value
                for key, value in baseline.items()
                if key != "model"
            },
            "randomly_damaged_input": {
                key: value
                for key, value in augmented.items()
                if key != "model"
            },
            "initial_weights_matched": True,
            "architecture_and_objective_matched": True,
            "epochs_per_variant": AUTOENCODER_EPOCHS,
        },
        "metrics": metrics,
        "probabilities": {
            name: values.tolist() for name, values in scores.items()
        },
        "removed_edge_probabilities": {
            name: [
                {
                    "edge": list(edge),
                    "probability": float(
                        scores[name][
                            next(
                                index
                                for index, candidate in enumerate(candidate_edges)
                                if frozenset(candidate) == frozenset(edge)
                            )
                        ]
                    ),
                }
                for edge in removed_edges
            ]
            for name in scores
        },
    }
