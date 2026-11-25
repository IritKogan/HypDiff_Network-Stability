"""Tests for the synthetic graphs (SBM, BA) and the adjusted Rand index."""

from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
import pytest
from scipy.special import comb

from hypstab.config import ConfigError, config_from_dict, load_config
from hypstab.data.graph import load_graph
from hypstab.data.synthetic import ba_graph, sbm_graph
from hypstab.metrics import adjusted_rand_index

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs" / "hypstab"


def test_sbm_graph_structure():
    g = sbm_graph(blocks=4, block_size=50, p_in=0.2, p_out=0.01, seed=3)
    assert g.node_count == 200 and g.features.shape == (200, 200)
    assert np.all(g.edges[:, 0] < g.edges[:, 1]) and len(np.unique(g.edges, axis=0)) == len(g.edges)
    same = g.class_labels[g.edges[:, 0]] == g.class_labels[g.edges[:, 1]]
    expected_in, expected_out = 4 * comb(50, 2) * 0.2, (comb(200, 2) - 4 * comb(50, 2)) * 0.01
    assert abs(same.sum() - expected_in) < 4 * np.sqrt(expected_in)
    assert abs((~same).sum() - expected_out) < 4 * np.sqrt(expected_out)
    again = sbm_graph(blocks=4, block_size=50, p_in=0.2, p_out=0.01, seed=3)
    np.testing.assert_array_equal(g.edges, again.edges)
    assert len(sbm_graph(4, 50, 0.2, 0.01, seed=4).edges) != len(g.edges) or not np.array_equal(
        sbm_graph(4, 50, 0.2, 0.01, seed=4).edges, g.edges)


def test_ba_graph_structure():
    g = ba_graph(nodes=300, m=2, seed=5)
    assert g.node_count == 300 and g.class_labels is None
    assert len(g.edges) == (300 - 2) * 2                      # networkx: m * (n - m) edges
    degrees = np.bincount(g.edges.ravel(), minlength=300)
    assert degrees.min() >= 2 and degrees.max() > 5 * np.median(degrees)   # hubs
    np.testing.assert_array_equal(g.edges, ba_graph(300, 2, 5).edges)


@pytest.mark.parametrize("name, dataset", [("sbm", "SBM"), ("ba", "BA")])
def test_synthetic_configs_differ_from_cora_only_in_name_and_dataset(name, dataset):
    with (CONFIGS / "cora.toml").open("rb") as handle:
        cora = tomllib.load(handle)
    with (CONFIGS / f"{name}.toml").open("rb") as handle:
        other = tomllib.load(handle)
    assert other["data"]["dataset"] == dataset and other["run"]["name"] == name
    other["data"]["dataset"], other["run"]["name"] = cora["data"]["dataset"], cora["run"]["name"]
    assert other == cora
    config = load_config(CONFIGS / f"{name}.toml")
    graph = load_graph(config.data.dataset, config.data.root, config.data.download, config.synthetic)
    assert graph.node_count == 2100


def test_synthetic_needs_section_and_valid_values():
    with pytest.raises(ValueError, match="synthetic"):
        load_graph("SBM", "data", False)
    with (CONFIGS / "cora.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    raw["synthetic"]["sbm_p_in"] = 1.5
    with pytest.raises(ConfigError, match="sbm_p_in"):
        config_from_dict(raw)


def test_adjusted_rand_index():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(a, a) == 1.0
    assert adjusted_rand_index(a, np.array([5, 5, 7, 7, 9, 9])) == 1.0        # relabelling
    # value from sklearn.metrics.adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2]) = 0.5714285714...
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(4 / 7)
    rng = np.random.default_rng(0)
    assert abs(adjusted_rand_index(rng.integers(0, 5, 5000), rng.integers(0, 5, 5000))) < 0.01
