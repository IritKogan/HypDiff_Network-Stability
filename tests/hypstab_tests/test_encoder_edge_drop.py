"""Tests for encoder training on damaged train graphs (encoder.edge_drop_*, students 2026-11-27)."""

from __future__ import annotations

import copy
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from hypstab.config import ConfigError, config_from_dict
from hypstab.data.split import pair_keys
from hypstab.training.autoencoder import draw_edge_drop, grouped_link_loss, link_loss, train_autoencoder
from tests.hypstab_tests.test_autoencoder import make_split, sbm_graph, tiny_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CPU = torch.device("cpu")


@pytest.fixture()
def raw() -> dict:
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


def drop_config(raw: dict, mode: str, fraction: float, loss: str = "dropped", **encoder):
    return tiny_config(raw, edge_drop_mode=mode, edge_drop_fraction=fraction, edge_drop_loss=loss, **encoder)


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize("name", ["cora", "cora_structure", "sbm", "ba"])
def test_main_configs_keep_the_original_training(name):
    with (REPO_ROOT / "configs" / "hypstab" / f"{name}.toml").open("rb") as handle:
        enc = config_from_dict(tomllib.load(handle)).encoder
    assert (enc.edge_drop_mode, enc.edge_drop_fraction, enc.edge_drop_loss) == ("none", 0.0, "dropped")


@pytest.mark.parametrize(
    "mode, fraction, loss, message",
    [
        ("sometimes", 0.1, "dropped", "encoder.edge_drop_mode"),
        ("fixed", 0.1, "all", "encoder.edge_drop_loss"),
        ("none", 0.02, "dropped", "must be 0"),
        ("fixed", 0.0, "dropped", r"\(0, 1\)"),
        ("uniform", 1.0, "dropped", r"\(0, 1\)"),
        ("fixed", -0.1, "dropped", r"\(0, 1\)"),
    ],
)
def test_edge_drop_config_is_validated(raw, mode, fraction, loss, message):
    raw["encoder"].update(edge_drop_mode=mode, edge_drop_fraction=fraction, edge_drop_loss=loss)
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


@pytest.mark.parametrize("key", ["edge_drop_mode", "edge_drop_fraction", "edge_drop_loss"])
def test_edge_drop_keys_are_required(raw, key):
    del raw["encoder"][key]
    with pytest.raises(ConfigError, match=key):
        config_from_dict(raw)


# --------------------------------------------------------------------------- drawing / loss


def test_draw_edge_drop_modes():
    rng = np.random.default_rng(0)
    q, mask = draw_edge_drop("none", 0.0, 50, rng)
    assert q == 0.0 and not mask.any()
    assert rng.random() == np.random.default_rng(0).random()  # "none" consumes no random numbers

    q, mask = draw_edge_drop("fixed", 0.3, 20000, np.random.default_rng(1))
    assert q == 0.3 and abs(mask.mean() - 0.3) < 0.02

    rng = np.random.default_rng(2)
    draws = [draw_edge_drop("uniform", 0.4, 10, rng)[0] for _ in range(2000)]
    assert 0.0 <= min(draws) and max(draws) <= 0.4
    assert abs(np.mean(draws) - 0.2) < 0.02


def test_grouped_loss_with_one_group_equals_link_loss(raw):
    graph = sbm_graph()
    config = tiny_config(raw)
    from hypstab.models.hgcn import normalized_adjacency
    from hypstab.training.autoencoder import build_autoencoder

    model = build_autoencoder(graph.features.shape[1], config)
    split = make_split(graph, config)
    emb = model.encode(graph.features, normalized_adjacency(graph.node_count, split.train, CPU))
    pos = torch.as_tensor(split.train[:40])
    neg = torch.as_tensor(split.val_negatives[:30])
    assert torch.allclose(grouped_link_loss(model, emb, [pos], neg), link_loss(model, emb, pos, neg))
    # each group is its own mean: a group of 3 edges weighs as much as a group of 37
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    small, large = pos[:3], pos[3:]
    expected = (bce(model.edge_logits(emb, small), torch.ones(3))
                + bce(model.edge_logits(emb, large), torch.ones(37))
                + bce(model.edge_logits(emb, neg), torch.zeros(30)))
    assert torch.allclose(grouped_link_loss(model, emb, [small, large], neg), expected)


# --------------------------------------------------------------------------- training


def test_mode_none_is_unchanged_by_the_new_keys(raw):
    """'none' gives the same trajectory whatever the (unused) loss choice; no drop records."""
    graph = sbm_graph()
    a = train_autoencoder(graph, make_split(graph, tiny_config(raw, max_epochs=15, patience=15)),
                          tiny_config(raw, max_epochs=15, patience=15), CPU).history
    config_b = drop_config(raw, "none", 0.0, "dropped_and_kept", max_epochs=15, patience=15)
    b = train_autoencoder(graph, make_split(graph, config_b), config_b, CPU).history
    assert a["train_loss"] == b["train_loss"] and a["val_auc"] == b["val_auc"]
    assert b["edge_drop_fractions"] == [] and b["edge_drop_skipped_epochs"] == []


@pytest.mark.parametrize("mode, fraction", [("fixed", 0.2), ("uniform", 0.4)])
@pytest.mark.parametrize("loss", ["dropped", "dropped_and_kept"])
def test_dropped_edges_are_hidden_and_are_the_positives(raw, mode, fraction, loss):
    graph = sbm_graph()
    config = drop_config(raw, mode, fraction, loss, max_epochs=12, patience=12)
    split = make_split(graph, config)
    result = train_autoencoder(graph, split, config, CPU, audit=True)
    n = graph.node_count
    train_keys = pair_keys(split.train, n)
    h = result.history
    assert len(h["edge_drop_fractions"]) == len(h["edge_drop_counts"]) == h["epochs_run"]
    if mode == "fixed":
        assert all(q == fraction for q in h["edge_drop_fractions"])
    else:
        assert all(0.0 <= q <= fraction for q in h["edge_drop_fractions"])
    for epoch, (seen, positives) in enumerate(zip(result.audit["epoch_adjacency_keys"],
                                                   result.audit["epoch_positive_keys"])):
        hidden = np.setdiff1d(train_keys, seen)
        assert np.isin(seen, train_keys).all()
        assert len(hidden) == h["edge_drop_counts"][epoch]
        if loss == "dropped":
            assert np.array_equal(np.sort(positives), np.sort(hidden))  # positives = exactly the hidden edges
        else:
            assert np.array_equal(np.sort(positives), np.sort(train_keys))


@pytest.mark.parametrize("mode, fraction", [("fixed", 0.02), ("uniform", 0.4)])
def test_no_leakage_with_edge_drop(raw, mode, fraction):
    graph = sbm_graph()
    config = drop_config(raw, mode, fraction, max_epochs=15, patience=15)
    split = make_split(graph, config)
    result = train_autoencoder(graph, split, config, CPU, audit=True)
    n = graph.node_count
    train_keys = pair_keys(split.train, n)
    used = {
        "adjacency": np.concatenate(result.audit["epoch_adjacency_keys"] + [result.audit["adjacency_keys"]]),
        "positive": result.audit["positive_keys"],
        "negative": result.audit["negative_keys"],
    }
    assert np.isin(used["adjacency"], train_keys).all() and np.isin(used["positive"], train_keys).all()
    for name, keys in {"val": split.val, "pool": split.pool, "val_negatives": split.val_negatives,
                       "control_pairs": split.control_pairs}.items():
        for kind, seen in used.items():
            assert not np.isin(pair_keys(keys, n), seen).any(), f"{name} leaked into {kind}"


def test_epoch_without_dropped_edges_makes_no_update(raw, monkeypatch):
    import hypstab.training.autoencoder as module

    real = module.draw_edge_drop
    calls = {"n": 0}

    def no_drop_on_second_epoch(mode, fraction, edge_count, rng):
        calls["n"] += 1
        q, mask = real(mode, fraction, edge_count, rng)
        return (0.0, np.zeros_like(mask)) if calls["n"] == 2 else (q, mask)

    monkeypatch.setattr(module, "draw_edge_drop", no_drop_on_second_epoch)
    graph = sbm_graph()
    config = drop_config(raw, "uniform", 0.4, max_epochs=4, patience=4)
    h = train_autoencoder(graph, make_split(graph, config), config, CPU).history
    assert h["edge_drop_skipped_epochs"] == [1]
    assert math.isnan(h["train_loss"][1]) and h["val_auc"][1] == h["val_auc"][0]  # weights unchanged
    assert len(h["train_loss"]) == len(h["val_auc"]) == 4


def test_edge_drop_training_learns_and_is_reproducible(raw):
    graph = sbm_graph()
    config = drop_config(raw, "uniform", 0.4)
    split = make_split(graph, config)
    a = train_autoencoder(graph, split, config, CPU).history
    assert a["best_val_auc"] > 0.64 and a["best_val_auc"] > a["val_auc"][0] + 0.03
    b = train_autoencoder(graph, split, config, CPU).history
    np.testing.assert_array_equal(a["train_loss"], b["train_loss"])  # NaN (skipped epochs) compare equal here
    assert a["edge_drop_fractions"] == b["edge_drop_fractions"]


def test_config_round_trip_keeps_edge_drop(raw):
    from hypstab.config import config_to_toml

    config = drop_config(copy.deepcopy(raw), "fixed", 0.02)
    again = config_from_dict(tomllib.loads(config_to_toml(config)))
    assert again == config
