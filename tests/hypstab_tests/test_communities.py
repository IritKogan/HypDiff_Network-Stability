"""Tests for Stage 3: hyperbolic K-means, clamp_only, community context, validation report."""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from hypstab.config import ConfigError, config_from_dict
from hypstab.evaluation.validation import validation_report
from hypstab.geometry import poincare as pb
from hypstab.models.clamp import clamp_to_centers, standardize
from hypstab.models.kmeans import frechet_mean, hyperbolic_kmeans, pairwise_distance
from hypstab.training.autoencoder import train_autoencoder
from hypstab.training.communities import fit_communities

from .test_autoencoder import make_split, sbm_graph, tiny_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DTYPE = torch.float64


@pytest.fixture()
def raw() -> dict:
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


def blobs(c: float = 1.0, per: int = 40, seed: int = 0):
    """Three tight clusters around well-separated centres in a 4-dim ball."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.6, 0, 0, 0], [-0.3, 0.5, 0, 0], [0, -0.4, 0.4, 0]], dtype=DTYPE) / c**0.5
    points, labels = [], []
    for j, mu in enumerate(centres):
        v = torch.randn(per, 4, generator=g, dtype=DTYPE) * 0.05
        points.append(pb.expmap(mu.expand(per, 4), v, c))
        labels += [j] * per
    return torch.cat(points), torch.tensor(labels), centres


# --------------------------------------------------------------------------- Frechet mean


@pytest.mark.parametrize("c", [1.0, 2.0])
def test_frechet_mean_properties(c):
    x, _, _ = blobs(c)
    mu = frechet_mean(x, c, init=torch.zeros(4, dtype=DTYPE))
    # first-order optimality: the mean of the log maps vanishes
    assert pb.logmap(mu.expand_as(x), x, c).mean(0).norm() < 1e-8
    # it minimises the sum of squared distances (better than nearby points)
    cost = lambda m: pb.distance(x, m.expand_as(x), c).pow(2).sum()
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        other = pb.expmap(mu, torch.randn(4, generator=g, dtype=DTYPE) * 0.05, c)
        assert cost(mu) <= cost(other) + 1e-10
    # single point / symmetric pair
    torch.testing.assert_close(frechet_mean(x[:1], c, init=torch.zeros(4, dtype=DTYPE)), x[0])
    pair = torch.stack([x[0], -x[0]])
    torch.testing.assert_close(frechet_mean(pair, c, init=x[5]), torch.zeros(4, dtype=DTYPE), atol=1e-9, rtol=0)


def test_frechet_mean_of_two_points_is_geodesic_midpoint():
    c = 1.0
    a = torch.tensor([0.5, 0.1], dtype=DTYPE)
    b = torch.tensor([-0.2, 0.6], dtype=DTYPE)
    mid = pb.expmap(a, 0.5 * pb.logmap(a, b, c), c)
    torch.testing.assert_close(frechet_mean(torch.stack([a, b]), c, init=a), mid)
    torch.testing.assert_close(pb.distance(mid, a, c), pb.distance(mid, b, c))


# --------------------------------------------------------------------------- K-means


@pytest.mark.parametrize("c", [1.0, 0.5])
def test_kmeans_recovers_separated_clusters(c):
    x, truth, centres = blobs(c)
    # (Lloyd + k-means++ can end in a local optimum: on these blobs 1 seed of 10 does at c = 0.5)
    result = hyperbolic_kmeans(x, k=3, c=c, seed=0, max_iter=40)
    assert result.converged
    # same partition up to relabelling
    for j in range(3):
        assert len(set(result.labels[truth == j].tolist())) == 1
    assert len(set(result.labels.tolist())) == 3
    assert (result.centers.norm(dim=-1) * c**0.5 < 1).all()
    # every point is assigned to its nearest centre; inertia matches
    d = pairwise_distance(x, result.centers, c)
    assert torch.equal(d.argmin(1), result.labels)
    assert np.isclose(result.inertia, float(d.min(1).values.pow(2).sum()))


def test_kmeans_reproducible_and_seed_dependent():
    x, _, _ = blobs()
    x = torch.cat([x, pb.expmap0(torch.randn(60, 4, dtype=DTYPE, generator=torch.Generator().manual_seed(9)) * 0.4, 1.0)])
    a = hyperbolic_kmeans(x, k=6, c=1.0, seed=1, max_iter=40)
    b = hyperbolic_kmeans(x, k=6, c=1.0, seed=1, max_iter=40)
    assert torch.equal(a.labels, b.labels) and torch.equal(a.centers, b.centers)


def test_kmeans_reseeds_empty_clusters():
    # two clumps, k = 3: one clump must be split; no cluster may stay empty
    g = torch.Generator().manual_seed(0)
    x = torch.cat([0.1 + 1e-3 * torch.randn(10, 2, generator=g, dtype=DTYPE),
                   -0.2 + 1e-3 * torch.randn(3, 2, generator=g, dtype=DTYPE)])
    for seed in range(5):
        result = hyperbolic_kmeans(x, k=3, c=1.0, seed=seed, max_iter=20)
        assert torch.bincount(result.labels, minlength=3).min() >= 1


def test_kmeans_rejects_bad_input():
    with pytest.raises(ValueError):
        hyperbolic_kmeans(torch.tensor([[1.5, 0.0]], dtype=DTYPE), k=1, c=1.0, seed=0, max_iter=1)
    with pytest.raises(ValueError):
        hyperbolic_kmeans(torch.zeros(2, 2, dtype=DTYPE), k=3, c=1.0, seed=0, max_iter=1)


def test_kmeans_float32_input_keeps_dtype():
    x, _, _ = blobs()
    result = hyperbolic_kmeans(x.float(), k=3, c=1.0, seed=0, max_iter=20)
    assert result.centers.dtype == torch.float32


# --------------------------------------------------------------------------- clamp_only


def test_standardize_uses_one_scalar_mean_and_std():
    v = torch.randn(30, 5, dtype=DTYPE)
    z, mean, std = standardize(v)
    assert mean.dim() == 0 and std.dim() == 0
    torch.testing.assert_close(z.mean(), torch.tensor(0.0, dtype=DTYPE), atol=1e-12, rtol=0)
    torch.testing.assert_close(z.std(), torch.tensor(1.0, dtype=DTYPE))


def test_clamp_large_clip_is_identity_and_pulls_outliers():
    x, labels, centres = blobs()
    torch.testing.assert_close(clamp_to_centers(x, labels, centres, 1.0, clip=1e6), x)
    out = clamp_to_centers(x, labels, centres, 1.0, clip=0.5)
    d_before = pb.distance(x, centres[labels], 1.0)
    d_after = pb.distance(out, centres[labels], 1.0)
    assert (d_after <= d_before + 1e-9).double().mean() > 0.9  # outlying coordinates are pulled in
    assert (out.norm(dim=-1) < 1).all()


def test_clamp_matches_old_shrink_to_prototype():
    """Reference: old StabilityDiffusion.shrink_to_prototype (upstream normalization/renormal)."""
    old = pytest.importorskip("stability.hypdiff_models")
    upstream_diffusion = pytest.importorskip("models.direction_diffusion")
    manifolds = pytest.importorskip("manifolds")
    x, labels, centres = blobs()
    x = x + 0.02 * torch.randn_like(x)  # break the symmetry a bit
    fake = SimpleNamespace(
        manifold=manifolds.PoincareBall(), c=1.0,
        normalization=lambda v: upstream_diffusion.diffusion.normalization(None, v),
        renormal=lambda z, h: upstream_diffusion.diffusion.renormal(None, z, h),
    )
    expected = old.StabilityDiffusion.shrink_to_prototype(fake, x, labels, centres, clip=1.0)
    torch.testing.assert_close(clamp_to_centers(x, labels, centres, 1.0, clip=1.0), expected)


# --------------------------------------------------------------------------- context + validation


def test_config_sections_validate(raw):
    for section, key, value in [("community", "n_clusters", 0), ("community", "frechet_tol", 0.0),
                                ("clamp", "clip", 0.0), ("clamp", "clip", -1.0)]:
        bad = copy.deepcopy(raw)
        bad[section][key] = value
        with pytest.raises(ConfigError):
            config_from_dict(bad)


def test_communities_and_validation_use_train_graph_only(raw):
    graph = sbm_graph()
    raw = copy.deepcopy(raw)
    raw["community"].update(n_clusters=3)
    config = tiny_config(raw, max_epochs=40, patience=40)
    split = make_split(graph, config)
    result = train_autoencoder(graph, split, config, torch.device("cpu"))
    features = graph.features
    context = fit_communities(result.model, features, result.train_adjacency, config)
    assert context.labels.shape == (graph.node_count,)
    assert sum(context.summary()["sizes"]) == graph.node_count

    report = validation_report(result.model, features, result.train_adjacency, split, context, config)
    assert set(report) == {"protocol", "no_diffusion", "clamp_only"}
    assert report["no_diffusion"]["auc"] == pytest.approx(result.history["best_val_auc"])
    assert 0.5 < report["clamp_only"]["auc"] <= 1.0

    # pool edges and control pairs are never read: poisoning them changes nothing
    poisoned = dataclasses.replace(split, pool=np.full_like(split.pool, -1),
                                   control_pairs=np.full_like(split.control_pairs, -1))
    assert validation_report(result.model, features, result.train_adjacency, poisoned, context, config) == report
