from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from stability.hypdiff_models import StabilityLPModel
from stability.edge_perturbation import remove_edges
from stability.pilot_config import PilotConfig
from stability.proteins_dataset import load_proteins_graphs
from stability.wp7_training import _edge_split, _edge_tensor, _loss, make_adjacency, model_args


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tu"
OUTPUT_DIR = ROOT / "artifacts" / "proteins_pilot" / "activation_development_fresh20261106"
GRAPH_SPECS = (
    {
        "graph_id": "PROTEINS-1034",
        "content_sha256": "cfcd6a2118754f2cd925eb2e1eaddd4d3867c857bb8c13f302a6916b80f915aa",
        "perturbation_seed": 1101034,
        "removed_edges": (
            ("PROTEINS-1034:n005", "PROTEINS-1034:n021"),
            ("PROTEINS-1034:n012", "PROTEINS-1034:n013"),
            ("PROTEINS-1034:n015", "PROTEINS-1034:n029"),
        ),
    },
    {
        "graph_id": "PROTEINS-0772",
        "content_sha256": "717b496bec6d28e083cb252dd54724a541b5e95840e008684d02d9e78d485294",
        "perturbation_seed": 1100772,
        "removed_edges": (
            ("PROTEINS-0772:n012", "PROTEINS-0772:n020"),
            ("PROTEINS-0772:n008", "PROTEINS-0772:n017"),
            ("PROTEINS-0772:n001", "PROTEINS-0772:n009"),
        ),
    },
    {
        "graph_id": "PROTEINS-0330",
        "content_sha256": "76114ff787a616c6ba4c0c115abb869c99568e78f9872fd46a537562c4f39d9c",
        "perturbation_seed": 1100330,
        "removed_edges": (
            ("PROTEINS-0330:n002", "PROTEINS-0330:n019"),
            ("PROTEINS-0330:n021", "PROTEINS-0330:n022"),
            ("PROTEINS-0330:n024", "PROTEINS-0330:n030"),
        ),
    },
)
SELECTION_SEED = 20261106
MODEL_SEED_OFFSET = 1200000
EPOCHS = 30
LEARNING_RATE = 1e-2
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


def _make_model(graph, activation: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args = model_args(
        graph,
        PilotConfig(curvature=1.0, tau=0.5),
        final_activation=activation,
    )
    return StabilityLPModel(args).cpu()


def _inputs(graph, perturbation):
    node_count = len(graph.node_ids)
    original_edges = _indexed_edges(graph.edges, graph.node_ids)
    damaged_edges = _indexed_edges(perturbation.edges, graph.node_ids)
    model_seed = graph.dataset_index + MODEL_SEED_OFFSET
    train_edges, validation_edges, train_negatives, validation_negatives = _edge_split(
        graph.edges, graph, model_seed + 1
    )
    train_adjacency = make_adjacency(node_count, train_edges)
    original_adjacency = make_adjacency(node_count, original_edges)
    damaged_adjacency = make_adjacency(node_count, damaged_edges)
    all_pairs = tuple(itertools.combinations(range(node_count), 2))
    original = {frozenset(edge) for edge in graph.edges}
    removed = {frozenset(edge) for edge in perturbation.removed_edges}
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
        "train_positive": _edge_tensor(train_edges, torch.device("cpu")),
        "train_negative": _edge_tensor(train_negatives, torch.device("cpu")),
        "all_pairs": all_pairs,
        "candidates": candidates,
        "candidate_keys": candidate_keys,
        "original": original,
        "removed": removed,
        "true_nonedges": true_nonedges,
        "labels": labels,
        "validation_edges": validation_edges,
        "validation_negatives": validation_negatives,
    }


def _probabilities(model, embeddings, pairs):
    edge_index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    scores = model.decode(embeddings, edge_index)
    if not torch.isfinite(scores).all():
        raise RuntimeError("decoder produced non-finite probabilities")
    return scores.detach().cpu().numpy().astype(np.float64)


def _ranking_metrics(scores, inputs):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(inputs["labels"], dtype=np.int64)
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    top_three = order[:3]
    cutoff = scores[top_three[-1]]
    above = int(np.sum(scores > cutoff))
    tied_positive = int(
        sum(labels[index] for index, score in enumerate(scores) if score == cutoff)
    )
    slots_at_cutoff = max(0, 3 - above)
    guaranteed = int(sum(labels[index] for index in top_three if scores[index] > cutoff))
    possible = guaranteed + min(slots_at_cutoff, tied_positive)
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "removed_edges_guaranteed_in_top_three": guaranteed,
        "removed_edges_possible_in_top_three": possible,
        "top_three_cutoff_tie_count": int(np.sum(scores == cutoff)),
        "candidate_count": int(len(scores)),
        "removed_edge_count": int(labels.sum()),
    }


def _preflight(graph, inputs, activation: str):
    seed = graph.dataset_index + MODEL_SEED_OFFSET
    model = _make_model(graph, activation, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    with torch.no_grad():
        embeddings = model.encode(graph.features, inputs["original_adjacency"])[2]
    initial_finite = bool(torch.isfinite(embeddings).all())
    distinct_rows = int(torch.unique(embeddings, dim=0).shape[0])

    optimizer.zero_grad()
    train_embeddings = model.encode(
        graph.features, inputs["train_adjacency"]
    )[2]
    loss = _loss(
        model, train_embeddings, inputs["train_positive"], inputs["train_negative"]
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    gradient_norm = (
        float(torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)))
        if gradients else 0.0
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.step()
    update_norm = float(
        torch.sqrt(
            sum(
                torch.sum((parameter.detach() - before[name]) ** 2)
                for name, parameter in model.named_parameters()
            )
        )
    )
    return {
        "initial_state_sha256": _state_hash(
            _make_model(graph, activation, seed)
        ),
        "initial_embeddings_finite": initial_finite,
        "initial_distinct_node_rows": distinct_rows,
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": gradients_finite,
        "gradient_l2_norm": gradient_norm,
        "parameter_update_l2_norm": update_norm,
        "parameter_update_nonzero": update_norm > 0.0,
    }


def _train(graph, inputs, activation: str, checkpoint_path: Path):
    seed = graph.dataset_index + MODEL_SEED_OFFSET
    model = _make_model(graph, activation, seed)
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
            parameter.grad for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
            raise RuntimeError(
                f"{graph.graph_id}/{activation}: non-finite gradients "
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
        intact_scores = _probabilities(model, intact_embeddings, inputs["all_pairs"])
        damaged_scores = _probabilities(model, damaged_embeddings, inputs["candidates"])
    if not torch.isfinite(intact_embeddings).all() or not torch.isfinite(
        damaged_embeddings
    ).all():
        raise RuntimeError(f"{graph.graph_id}/{activation}: non-finite embeddings")

    intact_keys = tuple(
        frozenset((graph.node_ids[source], graph.node_ids[target]))
        for source, target in inputs["all_pairs"]
    )
    intact_labels = np.asarray(
        [key in inputs["original"] for key in intact_keys], dtype=np.int64
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return {
        "activation": activation,
        "model_seed": seed,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "training_losses": losses,
        "final_training_loss": losses[-1],
        "initial_state_sha256": _state_hash(_make_model(graph, activation, seed)),
        "final_state_sha256": _state_hash(model),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "intact_embeddings_distinct_node_rows": int(
            torch.unique(intact_embeddings, dim=0).shape[0]
        ),
        "damaged_embeddings_distinct_node_rows": int(
            torch.unique(damaged_embeddings, dim=0).shape[0]
        ),
        "intact_reconstruction_average_precision": float(
            average_precision_score(intact_labels, intact_scores)
        ),
        "damaged_input_ranking": _ranking_metrics(damaged_scores, inputs),
    }


def run_experiment() -> dict:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {OUTPUT_DIR}")
    graphs, raw_hashes = load_proteins_graphs(DATA_ROOT, download=False)
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    specs_by_id = {spec["graph_id"]: spec for spec in GRAPH_SPECS}
    if len(specs_by_id) != len(GRAPH_SPECS):
        raise RuntimeError("experiment graph list contains duplicate IDs")

    selected = []
    for spec in GRAPH_SPECS:
        graph = graph_by_id.get(spec["graph_id"])
        if graph is None:
            raise RuntimeError(f"preselected graph is no longer eligible: {spec['graph_id']}")
        if graph.content_sha256 != spec["content_sha256"]:
            raise RuntimeError(f"content hash changed for {spec['graph_id']}")
        perturbation = remove_edges(
            graph.node_ids,
            graph.features,
            graph.edges,
            num_edges=3,
            seed=spec["perturbation_seed"],
        )
        if not perturbation.is_connected:
            raise RuntimeError(f"fixed perturbation disconnects {spec['graph_id']}")
        if set(map(frozenset, perturbation.removed_edges)) != set(
            map(frozenset, spec["removed_edges"])
        ):
            raise RuntimeError(f"fixed removed edges changed for {spec['graph_id']}")
        selected.append((spec, graph, perturbation))

    torch.set_num_threads(1)
    started = time.perf_counter()
    prepared = {}
    preflight = {}
    for spec, graph, perturbation in selected:
        inputs = _inputs(graph, perturbation)
        prepared[graph.graph_id] = (spec, graph, perturbation, inputs)
        preflight[graph.graph_id] = {
            activation: _preflight(graph, inputs, activation)
            for activation in ACTIVATIONS
        }

    leaky_passed = all(
        row["initial_embeddings_finite"]
        and row["initial_distinct_node_rows"] > 1
        and row["loss_finite"]
        and row["gradients_finite"]
        and row["parameter_update_nonzero"]
        for row in (preflight[graph_id]["leaky_relu"] for graph_id in preflight)
    )
    if not leaky_passed:
        return {
            "status": "preflight_gate_failed",
            "selection_seed": SELECTION_SEED,
            "preflight": preflight,
            "training_started": False,
            "duration_seconds": time.perf_counter() - started,
        }

    results = {}
    for graph_id, (spec, graph, perturbation, inputs) in prepared.items():
        hashes = {
            preflight[graph_id][activation]["initial_state_sha256"]
            for activation in ACTIVATIONS
        }
        if len(hashes) != 1:
            raise RuntimeError(f"{graph_id}: activations did not share initialization")
        results[graph_id] = {
            "dataset_index": graph.dataset_index,
            "content_sha256": graph.content_sha256,
            "node_count": len(graph.node_ids),
            "edge_count": len(graph.edges),
            "feature_dim": int(graph.features.shape[1]),
            "model_seed": graph.dataset_index + MODEL_SEED_OFFSET,
            "perturbation_seed": spec["perturbation_seed"],
            "removed_edges": [list(edge) for edge in perturbation.removed_edges],
            "perturbation_preserves_connectivity": perturbation.is_connected,
            "variants": {
                activation: _train(
                    graph,
                    inputs,
                    activation,
                    OUTPUT_DIR
                    / "models"
                    / graph_id
                    / activation
                    / "autoencoder.pt",
                )
                for activation in ACTIVATIONS
            },
        }

    return {
        "status": "complete",
        "study_scope": "development-only encoder-decoder comparison on known graphs",
        "selection": {
            "seed": SELECTION_SEED,
            "rule": (
                "selected before outcome inspection from distinct eligible graphs "
                "after excluding all official calibration/evaluation indices"
            ),
            "graphs": [
                {
                    "graph_id": spec["graph_id"],
                    "dataset_index": graph.dataset_index,
                    "content_sha256": graph.content_sha256,
                    "node_count": len(graph.node_ids),
                    "edge_count": len(graph.edges),
                    "perturbation_seed": spec["perturbation_seed"],
                    "removed_edges": [list(edge) for edge in perturbation.removed_edges],
                    "perturbation_preserves_connectivity": perturbation.is_connected,
                }
                for spec, graph, perturbation in selected
            ],
        },
        "dataset_raw_file_sha256": raw_hashes,
        "method": {
            "activations": {
                "control": "final-layer ReLU",
                "variant": "final-layer LeakyReLU, negative_slope=0.01",
                "other_layers": "unchanged ReLU",
            },
            "matched_initialization_and_training_inputs": True,
            "model_seed_offset": MODEL_SEED_OFFSET,
            "edge_split_seed": "model_seed + 1",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "epochs_per_variant_per_graph": EPOCHS,
            "total_training_runs": len(selected) * len(ACTIVATIONS),
            "total_autoencoder_epochs": len(selected) * len(ACTIVATIONS) * EPOCHS,
            "metrics": {
                "encoder_collapse": "all node embedding rows exactly identical",
                "damaged_input_average_precision": (
                    "removed edges versus all true nonedges on identical damaged inputs"
                ),
                "top_three": (
                    "guaranteed hits strictly above the third-place score; possible "
                    "hits include up to available slots among ties at that score"
                ),
            },
            "denoiser_trained": False,
            "thresholding": False,
            "recalibration": False,
        },
        "preflight": preflight,
        "graphs": results,
        "duration_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    result = run_experiment()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT_DIR / "experiment.json"
    artifact.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
