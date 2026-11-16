from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from stability.hypdiff_models import StabilityLPModel
from stability.pilot_config import PilotConfig
from stability.wp7_training import _edge_split, _edge_tensor, _loss, make_adjacency, model_args


EPOCHS = 30
LEARNING_RATE = 1e-2
MODEL_SEED_OFFSET = 950000
GRAPH_IDS = ("PROTEINS-0282", "PROTEINS-0151")
ACTIVATIONS = ("relu", "leaky_relu")


def _indexed_edges(edges, node_ids):
    indices = {node: index for index, node in enumerate(node_ids)}
    return tuple((indices[source], indices[target]) for source, target in edges)


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _parameter_snapshot(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _update_norm(model, before) -> float:
    return float(
        torch.sqrt(
            sum(
                torch.sum((parameter.detach() - before[name]) ** 2)
                for name, parameter in model.named_parameters()
            )
        )
    )


def _make_model(graph, final_activation: str, model_seed: int):
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    config = PilotConfig(curvature=1.0, tau=0.5)
    args = model_args(graph, config, final_activation=final_activation)
    return StabilityLPModel(args).cpu()


def _graph_view(record):
    return SimpleNamespace(
        graph_id=record["graph_id"],
        dataset_index=record["dataset_index"],
        node_ids=tuple(record["node_ids"]),
        features=torch.tensor(record["features"], dtype=torch.float32),
        edges=tuple(tuple(edge) for edge in record["original_edges"]),
    )


def _inputs(record, graph):
    node_count = len(graph.node_ids)
    original_edges = _indexed_edges(graph.edges, graph.node_ids)
    damaged_edges = _indexed_edges(record["damaged_edges"], graph.node_ids)
    original_adjacency = make_adjacency(node_count, original_edges)
    damaged_adjacency = make_adjacency(node_count, damaged_edges)
    train_edges, validation_edges, train_negatives, validation_negatives = _edge_split(
        graph.edges, graph, graph.dataset_index + MODEL_SEED_OFFSET + 1
    )
    train_adjacency = make_adjacency(node_count, train_edges)
    train_positive = _edge_tensor(train_edges, torch.device("cpu"))
    train_negative = _edge_tensor(train_negatives, torch.device("cpu"))
    all_pairs = tuple(itertools.combinations(range(node_count), 2))
    removed = {frozenset(edge) for edge in record["removed_edges"]}
    original = {frozenset(edge) for edge in graph.edges}
    true_nonedges = {
        frozenset((graph.node_ids[source], graph.node_ids[target]))
        for source, target in all_pairs
        if frozenset((graph.node_ids[source], graph.node_ids[target])) not in original
    }
    candidates = tuple(
        pair
        for pair in all_pairs
        if frozenset((graph.node_ids[pair[0]], graph.node_ids[pair[1]]))
        in removed | true_nonedges
    )
    candidate_keys = tuple(
        frozenset((graph.node_ids[source], graph.node_ids[target]))
        for source, target in candidates
    )
    labels = np.asarray([key in removed for key in candidate_keys], dtype=np.int64)
    return {
        "original_adjacency": original_adjacency,
        "damaged_adjacency": damaged_adjacency,
        "train_adjacency": train_adjacency,
        "train_positive": train_positive,
        "train_negative": train_negative,
        "all_pairs": all_pairs,
        "candidates": candidates,
        "candidate_keys": candidate_keys,
        "removed": removed,
        "original": original,
        "true_nonedges": true_nonedges,
        "labels": labels,
    }


def _scores(model, embeddings, pairs):
    edge_index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    probabilities = model.decode(embeddings, edge_index)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("decoder produced non-finite probabilities")
    return probabilities.detach().cpu().numpy().astype(np.float64)


def _ranking_metrics(scores, inputs):
    labels = inputs["labels"]
    average_precision = float(average_precision_score(labels, scores))
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    top_three = order[:3]
    top_three_recovered = int(sum(labels[index] for index in top_three))
    cutoff_score = scores[top_three[-1]]
    tied_at_cutoff = int(np.sum(scores == cutoff_score))
    return {
        "average_precision": average_precision,
        "removed_edges_in_top_three": top_three_recovered,
        "top_three_candidate_order": [
            sorted(inputs["candidate_keys"][index])
            for index in top_three
        ],
        "tie_count_at_top_three_cutoff": tied_at_cutoff,
        "candidate_count": int(len(scores)),
        "removed_edge_count": int(labels.sum()),
    }


def _preflight_variant(graph, inputs, activation: str):
    model_seed = graph.dataset_index + MODEL_SEED_OFFSET
    model = _make_model(graph, activation, model_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    with torch.no_grad():
        initial_embeddings = model.encode(
            graph.features, inputs["original_adjacency"]
        )[2]
    finite_embeddings = bool(torch.isfinite(initial_embeddings).all())
    distinct_embeddings = int(torch.unique(initial_embeddings, dim=0).shape[0])

    before = _parameter_snapshot(model)
    optimizer.zero_grad()
    embeddings = model.encode(graph.features, inputs["train_adjacency"])[2]
    loss = _loss(
        model, embeddings, inputs["train_positive"], inputs["train_negative"]
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    gradient_norm = float(
        torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients))
    ) if gradients else 0.0
    optimizer.step()
    update_norm = _update_norm(model, before)
    return {
        "initial_state_sha256": _state_hash(_make_model(graph, activation, model_seed)),
        "initial_embeddings_finite": finite_embeddings,
        "initial_embedding_distinct_node_rows": distinct_embeddings,
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": finite_gradients,
        "gradient_l2_norm": gradient_norm,
        "parameter_update_l2_norm": update_norm,
        "parameter_update_nonzero": update_norm > 0.0,
    }


def _train_variant(graph, inputs, activation: str, checkpoint_path: Path):
    model_seed = graph.dataset_index + MODEL_SEED_OFFSET
    model = _make_model(graph, activation, model_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(graph.features, inputs["train_adjacency"])[2]
        loss = _loss(
            model, embeddings, inputs["train_positive"], inputs["train_negative"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{graph.graph_id}/{activation}: non-finite loss at epoch {epoch + 1}"
            )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(
                f"{graph.graph_id}/{activation}: missing or non-finite gradients "
                f"at epoch {epoch + 1}"
            )
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        intact_embeddings = model.encode(
            graph.features, inputs["original_adjacency"]
        )[2]
        damaged_embeddings = model.encode(
            graph.features, inputs["damaged_adjacency"]
        )[2]
        intact_scores = _scores(model, intact_embeddings, inputs["all_pairs"])
        damaged_scores = _scores(model, damaged_embeddings, inputs["candidates"])
    if not torch.isfinite(intact_embeddings).all() or not torch.isfinite(
        damaged_embeddings
    ).all():
        raise RuntimeError(f"{graph.graph_id}/{activation}: non-finite final embeddings")

    intact_keys = tuple(
        frozenset((graph.node_ids[source], graph.node_ids[target]))
        for source, target in inputs["all_pairs"]
    )
    intact_labels = np.asarray(
        [key in inputs["original"] for key in intact_keys], dtype=np.int64
    )
    intact_metrics = {
        "average_precision": float(average_precision_score(intact_labels, intact_scores)),
        "positive_edge_count": int(intact_labels.sum()),
        "true_nonedges_count": int(len(intact_labels) - intact_labels.sum()),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return {
        "activation": activation,
        "model_seed": model_seed,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "training_losses": losses,
        "final_training_loss": losses[-1],
        "initial_state_sha256": _state_hash(
            _make_model(graph, activation, model_seed)
        ),
        "final_state_sha256": _state_hash(model),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "intact_embedding_distinct_node_rows": int(
            torch.unique(intact_embeddings, dim=0).shape[0]
        ),
        "damaged_embedding_distinct_node_rows": int(
            torch.unique(damaged_embeddings, dim=0).shape[0]
        ),
        "intact_graph_reconstruction": intact_metrics,
        "damaged_input_removed_edge_ranking": _ranking_metrics(
            damaged_scores, inputs
        ),
    }


def run_experiment(calibration_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite development experiment output: {output_dir}"
        )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    records = {record["graph_id"]: record for record in calibration["calibration_graphs"]}
    if set(GRAPH_IDS) - records.keys():
        raise ValueError("calibration artifact is missing an approved graph")

    torch.set_num_threads(1)
    started = time.perf_counter()
    graph_inputs = {}
    preflight = {}
    for graph_id in GRAPH_IDS:
        record = records[graph_id]
        graph = _graph_view(record)
        inputs = _inputs(record, graph)
        graph_inputs[graph_id] = (record, graph, inputs)
        preflight[graph_id] = {
            activation: _preflight_variant(graph, inputs, activation)
            for activation in ACTIVATIONS
        }

    # This gate is specific to the approved 0282 LeakyReLU initialization and step.
    leaky = preflight["PROTEINS-0282"]["leaky_relu"]
    if not (
        leaky["initial_embeddings_finite"]
        and leaky["initial_embedding_distinct_node_rows"] > 1
        and leaky["loss_finite"]
        and leaky["gradients_finite"]
        and leaky["parameter_update_nonzero"]
    ):
        return {
            "status": "preflight_gate_failed",
            "preflight": preflight,
            "training_started": False,
            "duration_seconds": time.perf_counter() - started,
        }

    results = {}
    for graph_id, (record, graph, inputs) in graph_inputs.items():
        initial_hashes = {
            preflight[graph_id][activation]["initial_state_sha256"]
            for activation in ACTIVATIONS
        }
        if len(initial_hashes) != 1:
            raise RuntimeError(f"{graph_id}: variants did not share initialization")
        results[graph_id] = {
            activation: _train_variant(
                graph,
                inputs,
                activation,
                output_dir / "models" / graph_id / activation / "autoencoder.pt",
            )
            for activation in ACTIVATIONS
        }

    return {
        "status": "complete",
        "study_scope": "development experiment; known-graph recovery; no diffusion",
        "source_calibration_artifact": str(calibration_path),
        "source_calibration_artifact_sha256": hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
        "method": {
            "graphs": list(GRAPH_IDS),
            "activations": {
                "control": "final-layer ReLU",
                "variant": "final-layer LeakyReLU, negative_slope=0.01",
                "other_layers": "unchanged ReLU",
            },
            "matched_initialization": True,
            "matched_features_graph_inputs_edge_splits_and_negative_samples": True,
            "autoencoder_epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "model_seed_offset": MODEL_SEED_OFFSET,
            "metrics": {
                "intact_graph_reconstruction": "average precision over all node pairs",
                "damaged_input_removed_edge_ranking": (
                    "average precision over removed edges and true nonedges; "
                    "top three by descending score, ties by candidate order"
                ),
            },
            "denoiser_trained": False,
            "thresholding_or_recalibration": False,
        },
        "preflight": preflight,
        "graphs": results,
        "duration_seconds": time.perf_counter() - started,
    }
