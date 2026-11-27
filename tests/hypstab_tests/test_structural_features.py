"""Tests for the structural node features from TRAIN edges (hypstab/data/features.py, 2026-11-27).

The upstream HypDiff data.py (repo root) is used ONLY here, as a reference: its function
node_festure_creator is taken from the file's source text (data.py itself imports dgl).
"""

from __future__ import annotations

import ast
import copy
import tomllib
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from hypstab.config import ConfigError, config_from_dict
from hypstab.data.features import (
    identity_degree_features,
    structural_features,
    train_adjacency,
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


def upstream_node_feature_creator():
    source = (REPO_ROOT / "data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "node_festure_creator")
    namespace = {"np": np, "numpy": np, "csr_matrix": sp.csr_matrix}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "data.py", "exec"), namespace)
    return namespace["node_festure_creator"]


# --------------------------------------------------------------------------- recipes by hand


def test_identity_degree_by_hand():
    x = identity_degree_features(SMALL, 5)
    assert x.shape == (5, 6)
    np.testing.assert_array_equal(x[:, :5], np.eye(5))
    # degrees 3, 2, 2, 1, 0 -> (degree + 1) / 5 (upstream adds self-loops before summing)
    np.testing.assert_allclose(x[:, 5], [4 / 5, 3 / 5, 3 / 5, 2 / 5, 1 / 5])


def test_structural_by_hand():
    x = structural_features(SMALL, 5)
    n = 5.0
    degree = np.array([3, 2, 2, 1, 0])
    a3 = np.array([2, 2, 2, 0, 0])            # 2 x triangles through the node
    a = np.zeros((5, 5))
    a[SMALL[:, 0], SMALL[:, 1]] = a[SMALL[:, 1], SMALL[:, 0]] = 1
    a4 = np.diag(np.linalg.matrix_power(a, 4))
    np.testing.assert_allclose(x, np.stack([degree / n, degree / n, a3 / n ** 2, a4 / n ** 3], axis=1))
    # triangle alone: 6 closed walks of length 4 per node (trace A^4 = 2^4 + 1 + 1 = 18)
    alone = structural_features(SMALL[:3], 3)
    np.testing.assert_allclose(alone[:, 3], 6 / 27)


def test_structural_equals_upstream_function():
    creator = upstream_node_feature_creator()
    graph = sbm_graph(seed=4)
    adj = train_adjacency(graph.edges, graph.node_count)
    expected = creator(adj, 3)          # steps=3, no random columns, no identity
    np.testing.assert_allclose(structural_features(graph.edges, graph.node_count), expected, rtol=1e-12)


# --------------------------------------------------------------------------- leakage


def test_features_use_train_edges_only():
    graph = sbm_graph(seed=5)
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 60, 7)
    for mode in ("identity_degree", "structural"):
        built = with_train_edge_features(graph, split.train, mode).features.numpy()
        recipe = identity_degree_features if mode == "identity_degree" else structural_features
        np.testing.assert_allclose(built, recipe(split.train, graph.node_count), rtol=1e-6)
        # removing every val / pool edge from the graph changes nothing
        train_only = GraphData(name="t", features=graph.features, edges=split.train, class_labels=None)
        again = with_train_edge_features(train_only, split.train, mode).features.numpy()
        np.testing.assert_array_equal(built, again)
        # while the same recipe on ALL edges differs (the held-out edges would matter)
        assert not np.allclose(built, recipe(graph.edges, graph.node_count))


def test_other_modes_leave_the_graph_unchanged():
    graph = sbm_graph()
    for mode in ("original", "identity"):
        assert with_train_edge_features(graph, graph.edges, mode) is graph


def test_load_graph_drops_original_features_for_structural_modes(raw):
    cfg = config_from_dict(raw)
    for mode in ("identity_degree", "structural"):
        graph = load_graph("BA", cfg.data.root, False, cfg.synthetic, mode)
        assert torch.equal(graph.features, torch.eye(graph.node_count))


# --------------------------------------------------------------------------- config / wiring


@pytest.mark.parametrize("mode", ["identity_degree", "structural"])
def test_feature_modes_accepted(raw, mode):
    raw["data"]["features"] = mode
    assert config_from_dict(raw).data.features == mode


def test_unknown_feature_mode_rejected(raw):
    raw["data"]["features"] = "degree_only"
    with pytest.raises(ConfigError, match="data.features"):
        config_from_dict(raw)


def test_dataset_label():
    assert dataset_label("Cora", "original") == "Cora"
    assert dataset_label("Cora", "structural") == "Cora-structure"
    assert dataset_label("Cora", "identity_degree") == "Cora-structure"
    assert dataset_label("SBM", "structural") == "SBM"


class _Stop(Exception):
    pass


@pytest.mark.parametrize("mode", ["identity_degree", "structural"])
def test_pipeline_trains_on_train_edge_features(raw, mode, tmp_path, monkeypatch):
    raw = copy.deepcopy(raw)
    raw["data"]["features"] = mode
    config = experiment_config(raw)
    captured = {}

    def stop(graph, split, *args, **kwargs):
        captured["graph"], captured["split"] = graph, split
        raise _Stop

    monkeypatch.setattr(pipeline_module, "get_bundles", stop)
    with pytest.raises(_Stop):
        pipeline_module.run_experiment(config, tmp_path, torch.device("cpu"), graph=sbm_graph())
    graph, split = captured["graph"], captured["split"]
    recipe = identity_degree_features if mode == "identity_degree" else structural_features
    np.testing.assert_allclose(graph.features.numpy(), recipe(split.train, graph.node_count), rtol=1e-6)


def test_train_script_applies_the_features():
    text = (REPO_ROOT / "scripts" / "hypstab_train.py").read_text(encoding="utf-8")
    assert "with_train_edge_features(graph, split.train, data.features)" in text
    text = (REPO_ROOT / "scripts" / "hypstab_e2.py").read_text(encoding="utf-8")
    assert "with_train_edge_features(graph, split.train, base.data.features)" in text
