"""Tests for Stage 2: data split, HGCN encoder, Fermi-Dirac decoder, autoencoder training.

Upstream HypDiff (layers/, manifolds/) and the old stability/ package are used
ONLY here, as references.
"""

from __future__ import annotations

import copy
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from hypstab.config import ConfigError, config_from_dict
from hypstab.data.graph import GraphData
from hypstab.data.split import canonical_edges, pair_keys, split_edges
from hypstab.geometry import poincare as pb
from hypstab.metrics import roc_auc
from hypstab.models.decoder import FermiDiracDecoder
from hypstab.models.hgcn import HGCNEncoder, normalized_adjacency
from hypstab.training.autoencoder import build_autoencoder, train_autoencoder

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def raw() -> dict:
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


def sbm_graph(seed: int = 0, sizes=(30, 30, 30), p_in: float = 0.25, p_out: float = 0.01) -> GraphData:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    n = len(labels)
    i, j = np.triu_indices(n, 1)
    prob = np.where(labels[i] == labels[j], p_in, p_out)
    keep = rng.random(len(i)) < prob
    edges = canonical_edges(np.stack([i[keep], j[keep]], axis=1), n)
    features = torch.tensor(rng.random((n, 12)), dtype=torch.float32)
    features = features / features.sum(1, keepdim=True)
    return GraphData(name="sbm", features=features, edges=edges, class_labels=labels)


def tiny_config(raw: dict, **encoder):
    raw = copy.deepcopy(raw)
    raw["run"]["device"] = "cpu"
    raw["split"]["control_pairs"] = 60
    raw["encoder"].update({"dim": 16, "max_epochs": 80, "patience": 25, "dropout": 0.2, **encoder})
    return config_from_dict(raw)


def make_split(graph, config):
    s = config.split
    return split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction,
                       s.pool_fraction, s.control_pairs, s.seed)


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("split", "train_fraction", 0.6, "sum to 1"),
        ("split", "val_fraction", 0.0, r"\(0, 1\)"),
        ("encoder", "activation", "gelu", "encoder.activation"),
        ("encoder", "dropout", 1.0, "dropout"),
        ("encoder", "dim", 0, "encoder.dim"),
        ("decoder", "fermi_dirac_t", 0.0, "fermi_dirac_t"),
        ("data", "dataset", "PROTEINS", "data.dataset"),
        ("encoder", "bias", 0, "true/false"),
    ],
)
def test_new_sections_validate(raw, section, key, value, message):
    raw[section][key] = value
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


# --------------------------------------------------------------------------- split / metrics


def test_split_identical_to_old_code():
    old = pytest.importorskip("stability.edge_split")
    graph = sbm_graph(seed=3)
    args = (graph.edges, graph.node_count, 0.5, 0.1, 0.4, 100, 123)
    new_split, old_split = split_edges(*args), old.split_edges(*args)
    for name in ("train", "val", "pool", "val_negatives", "control_pairs"):
        assert np.array_equal(getattr(new_split, name), getattr(old_split, name)), name


def test_roc_auc_matches_sklearn_including_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    pos = np.round(rng.normal(1, 1, 300), 1)  # rounding creates ties
    neg = np.round(rng.normal(0, 1, 500), 1)
    expected = metrics.roc_auc_score(np.r_[np.ones(300), np.zeros(500)], np.r_[pos, neg])
    assert math.isclose(roc_auc(pos, neg), expected, rel_tol=1e-12)
    assert roc_auc([1, 2], [0, 0.5]) == 1.0 and roc_auc([0], [0]) == 0.5
    with pytest.raises(ValueError):
        roc_auc([np.nan], [0])


def test_normalized_adjacency_dense_formula():
    edges = np.array([[0, 1], [1, 2]])
    a = normalized_adjacency(4, edges, "cpu").to_dense()
    adj = np.eye(4)
    adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = 1
    d = adj.sum(1)
    np.testing.assert_allclose(a.numpy(), adj / np.sqrt(np.outer(d, d)), rtol=1e-6)


# --------------------------------------------------------------------------- encoder / decoder


def test_encoder_output_inside_ball_and_shape():
    graph = sbm_graph()
    enc = HGCNEncoder(12, 8, 2, c=1.0, dropout=0.0, bias=True, activation="relu", final_activation="relu")
    out = enc(graph.features, normalized_adjacency(graph.node_count, graph.edges, "cpu"))
    assert out.shape == (graph.node_count, 8)
    assert torch.isfinite(out).all() and (out.norm(dim=-1) < 1).all()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("c", [1.0, 0.7])
def test_encoder_matches_upstream_hgcn(bias, c):
    """Same weights -> same embeddings as upstream HypDiff's HGCN layers (reference only)."""
    hyp_layers = pytest.importorskip("layers.hyp_layers")
    manifolds = pytest.importorskip("manifolds")
    import torch.nn.functional as F

    graph = sbm_graph(seed=1)
    adjacency = normalized_adjacency(graph.node_count, graph.edges, "cpu")
    torch.manual_seed(0)
    ours = HGCNEncoder(12, 8, 2, c=c, dropout=0.0, bias=bias, activation="relu", final_activation="relu")
    if bias:
        with torch.no_grad():
            for layer in ours.layers:
                layer.linear.bias.normal_(0, 0.1)
    manifold = manifolds.PoincareBall()
    curv = torch.tensor([c])
    upstream = [
        hyp_layers.HyperbolicGraphConvolution(manifold, i, o, curv, curv, 0.0, F.relu, int(bias), 0, 0)
        for i, o in ((12, 8), (8, 8))
    ]
    for up, mine in zip(upstream, ours.layers):
        up.linear.weight.data.copy_(mine.linear.weight.data)
        if bias:
            up.linear.bias.data.copy_(mine.linear.bias.data)
        up.eval()
    ours.eval()
    x = manifold.proj(manifold.expmap0(graph.features, c=curv), c=curv)
    for up in upstream:
        x, _ = up((x, adjacency))
    torch.testing.assert_close(ours(graph.features, adjacency), x, rtol=1e-5, atol=1e-6)


def test_fermi_dirac_matches_upstream_and_formula():
    layers = pytest.importorskip("layers.layers")
    x = pb.expmap0(torch.randn(10, 4, dtype=torch.float64) * 0.5, 1.0)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 9], [5, 5]])
    dec = FermiDiracDecoder(r=2.0, t=1.0, c=1.0)
    d2 = pb.sqdistance(x[pairs[:, 0]], x[pairs[:, 1]], 1.0)
    torch.testing.assert_close(dec.probabilities(x, pairs), 1 / (torch.exp(d2 - 2.0) + 1))
    torch.testing.assert_close(dec.probabilities(x, pairs), layers.FermiDiracDecoder(2.0, 1.0)(d2))
    assert dec.probabilities(x, pairs)[3] == torch.sigmoid(torch.tensor(2.0, dtype=torch.float64))


# --------------------------------------------------------------------------- training


def test_training_learns_and_stops_early(raw):
    graph = sbm_graph()
    config = tiny_config(raw)
    split = make_split(graph, config)
    result = train_autoencoder(graph, split, config, torch.device("cpu"))
    h = result.history
    # random features + half of the edges: ~0.68-0.73 on this small graph (old code: same range)
    assert h["best_val_auc"] > 0.64 and h["best_val_auc"] > h["val_auc"][0] + 0.03
    assert h["best_val_auc"] == max(h["val_auc"])
    assert len(h["train_loss"]) == len(h["val_auc"]) == h["epochs_run"] <= 80
    if h["early_stopped"]:
        assert h["epochs_run"] - 1 - h["best_epoch"] == config.encoder.patience
    assert not result.model.training  # returned in eval mode, best weights
    with torch.no_grad():
        emb = result.model.encode(graph.features, result.train_adjacency)
    val_pairs = torch.as_tensor(split.val)
    val_neg = torch.as_tensor(split.val_negatives)
    auc = roc_auc(result.model.edge_logits(emb, val_pairs).numpy(), result.model.edge_logits(emb, val_neg).numpy())
    assert math.isclose(auc, h["best_val_auc"], rel_tol=1e-9)


def test_training_is_reproducible(raw):
    graph = sbm_graph()
    config = tiny_config(raw, max_epochs=15, patience=15)
    split = make_split(graph, config)
    a = train_autoencoder(graph, split, config, torch.device("cpu"))
    b = train_autoencoder(graph, split, config, torch.device("cpu"))
    assert a.history["train_loss"] == b.history["train_loss"]
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key])


def test_no_leakage_into_training(raw):
    graph = sbm_graph()
    config = tiny_config(raw, max_epochs=20, patience=20)
    split = make_split(graph, config)
    result = train_autoencoder(graph, split, config, torch.device("cpu"), audit=True)
    n = graph.node_count
    held_out = {
        "val": pair_keys(split.val, n),
        "pool": pair_keys(split.pool, n),
        "val_negatives": pair_keys(split.val_negatives, n),
        "control_pairs": pair_keys(split.control_pairs, n),
    }
    train_keys = pair_keys(split.train, n)
    assert np.array_equal(np.sort(result.audit["adjacency_keys"]), np.sort(train_keys))
    assert np.array_equal(np.sort(result.audit["positive_keys"]), np.sort(train_keys))
    for name, keys in held_out.items():
        for used in ("adjacency_keys", "positive_keys", "negative_keys"):
            assert not np.isin(keys, result.audit[used]).any(), f"{name} leaked into {used}"
    assert not np.isin(train_keys, result.audit["negative_keys"]).any()


def test_build_autoencoder_follows_config(raw):
    config = tiny_config(raw, num_layers=3, final_activation="identity")
    model = build_autoencoder(12, config)
    assert len(model.encoder.layers) == 3
    assert model.encoder.layers[-1].act_name == "identity"
    assert model.encoder.layers[0].linear.weight.shape == (16, 12)
    assert model.decoder.r == config.decoder.fermi_dirac_r


def test_training_trajectory_matches_old_code_without_dropout(raw):
    """Same seed, no dropout: loss per epoch equals the old stability/ code (up to float noise)."""
    old_pipeline = pytest.importorskip("tests.test_edge_recovery_pipeline")
    old_training = pytest.importorskip("stability.model_training")
    old_data = pytest.importorskip("stability.graph_data")
    graph = sbm_graph()
    config = tiny_config(raw, max_epochs=30, patience=30, dropout=0.0)
    split = make_split(graph, config)
    old_config = old_pipeline.tiny_config(
        model=dict(dim=16, dropout=0.0, seed=config.encoder.seed),
        autoencoder=dict(max_epochs=30, patience=30),
        split=dict(control_pairs=60),
    )
    old_graph = old_data.GraphData(name="sbm", features=graph.features, edges=graph.edges,
                                   class_labels=graph.class_labels)
    _, _, _, old_history, _ = old_training.train_encoder(old_graph, split, old_config, torch.device("cpu"))
    new = train_autoencoder(graph, split, config, torch.device("cpu")).history
    np.testing.assert_allclose(new["train_loss"], old_history["autoencoder_train_loss"], atol=1e-4)
    np.testing.assert_allclose(new["val_auc"], old_history["autoencoder_val_auc"], atol=5e-3)
