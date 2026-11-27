"""Tests for the structural node features from TRAIN edges (hypstab/data/features.py, 2026-11-27).

"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from hypstab.config import ConfigError, config_from_dict
from hypstab.data.features import (
    TRAIN_EDGE_FEATURE_MODES,
    identity_degree_features,
    with_train_edge_features,
)
from hypstab.data.graph import GraphData, load_graph
from hypstab.data.split import split_edges
import hypstab.experiment.pipeline as pipeline_module
from hypstab.experiment.probabilities import dataset_label

from .test_autoencoder import sbm_graph
from .test_experiment import experiment_config

REPO_ROOT = Path(__file__).resolve().parents[2]

# triangle 0-1-2 plus a pendant node 3 attached to 0, and an isolated node 4
SMALL = np.array([[0, 1], [0, 2], [1, 2], [0, 3]])


@pytest.fixture()
def raw() -> dict:
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


# --------------------------------------------------------------------------- recipes by hand


def test_identity_degree_by_hand():
    x = identity_degree_features(SMALL, 5)
    assert x.shape == (5, 6)
    np.testing.assert_array_equal(x[:, :5], np.eye(5))
    # degrees 3, 2, 2, 1, 0 -> (degree + 1) / 5 (upstream adds self-loops before summing)
    np.testing.assert_allclose(x[:, 5], [4 / 5, 3 / 5, 3 / 5, 2 / 5, 1 / 5])


# --------------------------------------------------------------------------- leakage


def test_features_use_train_edges_only():
    graph = sbm_graph(seed=5)
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 60, 7)
    built = with_train_edge_features(graph, split.train, "identity_degree").features.numpy()
    np.testing.assert_allclose(built, identity_degree_features(split.train, graph.node_count), rtol=1e-6)
    # removing every val / pool edge from the graph changes nothing
    train_only = GraphData(name="t", features=graph.features, edges=split.train, class_labels=None)
    again = with_train_edge_features(train_only, split.train, "identity_degree").features.numpy()
    np.testing.assert_array_equal(built, again)
    # while the same recipe on ALL edges differs (the held-out edges would matter)
    assert not np.allclose(built, identity_degree_features(graph.edges, graph.node_count))


def test_other_modes_leave_the_graph_unchanged():
    graph = sbm_graph()
    for mode in ("original", "identity"):
        assert with_train_edge_features(graph, graph.edges, mode) is graph


def test_load_graph_drops_original_features_for_structural_modes(raw):
    cfg = config_from_dict(raw)
    graph = load_graph("BA", cfg.data.root, False, cfg.synthetic, "identity_degree")
    assert torch.equal(graph.features, torch.eye(graph.node_count))


# --------------------------------------------------------------------------- config / wiring


def test_feature_mode_accepted(raw):
    raw["data"]["features"] = "identity_degree"
    assert config_from_dict(raw).data.features == "identity_degree"


@pytest.mark.parametrize("mode", ["degree_only", "structural"])   # structural: removed 2026-11-27
def test_unknown_feature_mode_rejected(raw, mode):
    assert TRAIN_EDGE_FEATURE_MODES == ("identity_degree",)
    raw["data"]["features"] = mode
    with pytest.raises(ConfigError, match="data.features"):
        config_from_dict(raw)


def test_dataset_label():
    assert dataset_label("Cora", "original") == "Cora"
    assert dataset_label("Cora", "identity_degree") == "Cora-structure"
    assert dataset_label("SBM", "identity_degree") == "SBM"


class _Stop(Exception):
    pass


def test_pipeline_trains_on_train_edge_features(raw, tmp_path, monkeypatch):
    raw = copy.deepcopy(raw)
    raw["data"]["features"] = "identity_degree"
    config = experiment_config(raw)
    captured = {}

    def stop(graph, split, *args, **kwargs):
        captured["graph"], captured["split"] = graph, split
        raise _Stop

    monkeypatch.setattr(pipeline_module, "get_bundles", stop)
    with pytest.raises(_Stop):
        pipeline_module.run_experiment(config, tmp_path, torch.device("cpu"), graph=sbm_graph())
    graph, split = captured["graph"], captured["split"]
    np.testing.assert_allclose(graph.features.numpy(), identity_degree_features(split.train, graph.node_count),
                               rtol=1e-6)


def test_train_script_applies_the_features():
    text = (REPO_ROOT / "scripts" / "hypstab_train.py").read_text(encoding="utf-8")
    assert "with_train_edge_features(graph, split.train, data.features)" in text
    text = (REPO_ROOT / "scripts" / "hypstab_e2.py").read_text(encoding="utf-8")
    assert "with_train_edge_features(graph, split.train, base.data.features)" in text
