"""Cora with identity features (structure only; advisor 2026-11-26): config and data loading."""

from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from hypstab.config import ConfigError, config_from_dict, load_config
from hypstab.data.graph import GraphData, load_graph, with_identity_features

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "hypstab"


def _toy_graph() -> GraphData:
    return GraphData(name="toy", features=torch.rand(4, 3), edges=np.array([[0, 1], [1, 2], [2, 3]]),
                     class_labels=np.array([0, 0, 1, 1]))


def test_identity_features_keep_edges_and_labels():
    graph = _toy_graph()
    structure = with_identity_features(graph)
    assert torch.equal(structure.features, torch.eye(4))
    assert structure.features.dtype == torch.float32
    assert np.array_equal(structure.edges, graph.edges)
    assert np.array_equal(structure.class_labels, graph.class_labels)
    assert structure.name == graph.name


def test_cora_structure_config_differs_from_cora_only_in_name_and_features():
    with (CONFIGS / "cora.toml").open("rb") as handle:
        cora = tomllib.load(handle)
    with (CONFIGS / "cora_structure.toml").open("rb") as handle:
        other = tomllib.load(handle)
    assert cora["data"]["features"] == "original"
    assert other["data"]["features"] == "identity" and other["run"]["name"] == "cora_structure"
    other["data"]["features"], other["run"]["name"] = "original", "cora"
    assert other == cora
    assert load_config(CONFIGS / "cora_structure.toml").data.features == "identity"


def test_feature_mode_is_required_and_validated():
    with (CONFIGS / "cora.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    raw["data"]["features"] = "degree"
    with pytest.raises(ConfigError, match="data.features"):
        config_from_dict(raw)
    del raw["data"]["features"]
    with pytest.raises(ConfigError):
        config_from_dict(raw)
    with pytest.raises(ValueError, match="feature mode"):
        load_graph("Cora", "data", False, features="degree")


def test_synthetic_graph_identity_mode_equals_original():
    config = load_config(CONFIGS / "sbm.toml")
    original = load_graph("SBM", "data", False, config.synthetic, "original")
    identity = load_graph("SBM", "data", False, config.synthetic, "identity")
    assert torch.equal(original.features, identity.features)
    assert np.array_equal(original.edges, identity.edges)
