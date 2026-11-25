"""Tests for the hyperbolic-vs-Euclidean check: geometry.manifold = "euclidean"."""

from __future__ import annotations

import copy
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hypstab.config import ConfigError, config_from_dict
from hypstab.evaluation.validation import validation_report
from hypstab.geometry import euclidean, get_geometry, poincare
from hypstab.models.clamp import clamp_to_centers
from hypstab.models.hgcn import HGCNEncoder, normalized_adjacency
from hypstab.models.kmeans import hyperbolic_kmeans
from hypstab.training.autoencoder import build_autoencoder, train_autoencoder
from hypstab.training.communities import fit_communities
from hypstab.training.denoiser import train_denoiser

from .test_autoencoder import make_split, sbm_graph

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def raw() -> dict:
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_get_geometry(raw):
    assert get_geometry("poincare") is poincare
    assert get_geometry("euclidean") is euclidean
    with pytest.raises(ValueError):
        get_geometry("lorentz")
    raw["geometry"]["manifold"] = "lorentz"
    with pytest.raises(ConfigError, match="geometry.manifold"):
        config_from_dict(raw)


def test_euclidean_maps():
    x, y = torch.randn(5, 4), torch.randn(5, 4)
    torch.testing.assert_close(euclidean.expmap(x, euclidean.logmap(x, y)), y)
    torch.testing.assert_close(euclidean.distance(x, y), (x - y).norm(dim=-1))
    torch.testing.assert_close(euclidean.sqdistance(x, y), (x - y).pow(2).sum(-1))
    m = torch.randn(3, 4)
    torch.testing.assert_close(euclidean.mobius_matvec(m, x), x @ m.T)
    assert euclidean.expmap0(x) is x and euclidean.logmap0(x) is x


def test_euclidean_hgcn_is_a_plain_gcn():
    torch.manual_seed(0)
    graph = sbm_graph()
    adjacency = normalized_adjacency(graph.node_count, graph.edges, "cpu")
    encoder = HGCNEncoder(graph.features.shape[1], 8, 2, 1.0, 0.0, True, "relu", "relu", manifold="euclidean")
    encoder.eval()
    x = graph.features
    for layer in encoder.layers:
        x = F.relu(torch.sparse.mm(adjacency, x @ layer.linear.weight.T + layer.linear.bias))
    torch.testing.assert_close(encoder(graph.features, adjacency), x)


def test_euclidean_kmeans_centres_are_means():
    torch.manual_seed(1)
    points = torch.cat([torch.randn(20, 3) * 5, torch.randn(20, 3) * 5 + 40])   # outside any unit ball
    result = hyperbolic_kmeans(points, k=2, c=1.0, seed=0, max_iter=20, manifold="euclidean")
    assert result.converged
    for j in range(2):
        torch.testing.assert_close(result.centers[j].double(), points[result.labels == j].double().mean(0),
                                   rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="inside the ball"):
        hyperbolic_kmeans(points, k=2, c=1.0, seed=0, max_iter=20)


def test_euclidean_clamp_pulls_outliers_to_centres():
    points = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    labels = torch.zeros(3, dtype=torch.long)
    out = clamp_to_centers(points, labels, torch.zeros(1, 2), 1.0, 0.5, manifold="euclidean")
    assert out[2, 0] < 10.0
    torch.testing.assert_close(out[:2], points[:2], atol=0.2, rtol=0)


def test_euclidean_pipeline_trains_end_to_end(raw):
    raw = copy.deepcopy(raw)
    raw["run"]["device"] = "cpu"
    raw["geometry"]["manifold"] = "euclidean"
    raw["split"]["control_pairs"] = 60
    raw["encoder"].update(dim=16, max_epochs=25, patience=25, dropout=0.2)
    raw["community"].update(n_clusters=3)
    raw["denoiser"].update(hidden_dim=16, layers=2, epochs=6)
    raw["recovery"].update(reconstructions=2)
    config = config_from_dict(raw)
    graph = sbm_graph()
    split = make_split(graph, config)
    result = train_autoencoder(graph, split, config, torch.device("cpu"))
    assert result.model.encoder.geo is euclidean and result.model.decoder.geo is euclidean
    context = fit_communities(result.model, graph.features, result.train_adjacency, config)
    diffusion, _ = train_denoiser(graph, split, config, torch.device("cpu"), result.model, context)
    assert diffusion.geo is euclidean
    report = validation_report(result.model, graph.features, result.train_adjacency, split, context, config,
                               diffusion)
    for method in ("no_diffusion", "clamp_only", "diffusion"):
        assert 0.0 <= report[method]["auc"] <= 1.0 and np.isfinite(report[method]["auc"])
    assert build_autoencoder(graph.features.shape[1], config).encoder.geo is euclidean
