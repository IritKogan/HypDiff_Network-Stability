"""Tests for Stage 4a: schedule, angular noise, denoiser network, recovery, training, validation.

The old stability/ code, upstream HypDiff and torch_geometric are used ONLY here, as references.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest
import torch

from hypstab.config import ConfigError, config_from_dict
from hypstab.data.split import pair_keys
from hypstab.diffusion.denoiser import GraphConv, ResidualGCNDenoiser
from hypstab.diffusion.model import HyperbolicDiffusion
from hypstab.diffusion.schedule import DiffusionSchedule
from hypstab.evaluation.validation import validation_report
from hypstab.geometry import poincare as pb
from hypstab.models.hgcn import normalized_adjacency
from hypstab.training import denoiser as denoiser_training
from hypstab.training.autoencoder import train_autoencoder
from hypstab.training.communities import fit_communities
from hypstab.training.denoiser import build_diffusion, train_denoiser

from .test_autoencoder import make_split, sbm_graph, tiny_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def raw() -> dict:
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


def small_config(raw: dict, **sections):
    raw = copy.deepcopy(raw)
    raw["community"].update(n_clusters=3)
    raw["denoiser"].update(hidden_dim=16, layers=2, epochs=6)
    raw["recovery"].update(reconstructions=3)
    for section, values in sections.items():
        raw[section].update(values)
    return tiny_config(raw, max_epochs=25, patience=25)


@pytest.fixture(scope="module")
def trained(request):
    """A small trained autoencoder + communities on the SBM graph (shared by several tests)."""
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        raw_cfg = tomllib.load(handle)
    config = small_config(raw_cfg)
    graph = sbm_graph()
    split = make_split(graph, config)
    result = train_autoencoder(graph, split, config, torch.device("cpu"))
    context = fit_communities(result.model, graph.features, result.train_adjacency, config)
    return SimpleNamespace(raw=raw_cfg, config=config, graph=graph, split=split, ae=result, context=context)


# --------------------------------------------------------------------------- schedule


@pytest.mark.parametrize("T, delta, minimum", [(200, 1.0, 154), (1000, 0.4, 999), (1000, 1.0, 813)])
def test_schedule_matches_old_code_and_minimum(T, delta, minimum):
    old = pytest.importorskip("stability.diffusion_schedule")
    schedule = DiffusionSchedule(T, delta, 1.0)
    np.testing.assert_allclose(schedule.signal, old.signal_coefficients(T, delta, 1.0), rtol=1e-12)
    assert schedule.last_valid_t_star() == minimum == old.last_valid_t_star(T, delta, 1.0)
    schedule.check_t_star(minimum)
    if minimum < T - 1:
        with pytest.raises(ValueError, match="past the minimum"):
            schedule.check_t_star(minimum + 1)
    with pytest.raises(ValueError):
        schedule.check_t_star(T)


def test_schedule_values_used_in_e1():
    # E1 (GPU, 2026-11-24): noise / signal coefficients of the baseline and the paper variant
    old = DiffusionSchedule(200, 1.0, 1.0)
    paper = DiffusionSchedule(1000, 0.4, 1.0)
    assert old.noise[7] == pytest.approx(0.13365652097461683) and old.signal[7] == pytest.approx(1.0050268021361153)
    assert paper.noise[35] == pytest.approx(0.1265866075900335)
    assert paper.signal[35] == pytest.approx(1.0199099137551402)


def test_coefficients_shape_and_range_check():
    schedule = DiffusionSchedule(50, 0.4, 1.0)
    a, s = schedule.coefficients(torch.tensor([0, 10, 49]), torch.float32)
    assert a.shape == s.shape == (3, 1)
    with pytest.raises(ValueError):
        schedule.coefficients(torch.tensor([50]), torch.float32)


# --------------------------------------------------------------------------- network


def test_graph_conv_matches_torch_geometric_gcnconv():
    gcn = pytest.importorskip("torch_geometric.nn")
    graph = sbm_graph(seed=2)
    adjacency = normalized_adjacency(graph.node_count, graph.edges, "cpu")
    ours = GraphConv(12, 5)
    with torch.no_grad():
        ours.bias.normal_()
    reference = gcn.GCNConv(12, 5)
    with torch.no_grad():
        reference.lin.weight.copy_(ours.weight.T)
        reference.bias.copy_(ours.bias)
    edge_index = adjacency.indices()  # includes the self-loops, as in the old code
    torch.testing.assert_close(ours(graph.features, adjacency), reference(graph.features, edge_index))


def test_denoiser_is_identity_at_initialisation():
    net = ResidualGCNDenoiser(6, 8, 2)
    graph = sbm_graph()
    x = torch.randn(graph.node_count, 6)
    adjacency = normalized_adjacency(graph.node_count, graph.edges, "cpu")
    torch.testing.assert_close(net(x, adjacency, torch.full((graph.node_count,), 0.3)), x)


# --------------------------------------------------------------------------- forward process


def make_diffusion(angular="folded", clip=True, T=1000, delta=0.4, dim=4):
    return HyperbolicDiffusion(dim, 8, 2, DiffusionSchedule(T, delta, 1.0), 1.0, angular, clip)


def test_angular_noise_signs():
    centers = torch.tensor([[0.3, -0.2, 0.0, 0.1], [-0.1, 0.4, 0.2, -0.3]])
    labels = torch.tensor([0, 1] * 50)
    like = torch.zeros(100, 4)
    signs = torch.sign(pb.logmap0(centers, 1.0))[labels]
    g = torch.Generator().manual_seed(0)
    folded = make_diffusion("folded").angular_noise_sample(like, labels, centers, g)
    assert torch.all(folded * signs >= 0)                 # restricted to the centre's orthant
    assert torch.all(folded[:, 2][labels == 0] == 0)      # sgn(0) = 0 -> no noise (as in HypDiff)
    g = torch.Generator().manual_seed(0)
    upstream = make_diffusion("upstream").angular_noise_sample(like, labels, centers, g)
    torch.testing.assert_close(upstream.abs(), folded.abs())   # same draws, only the sign rule differs
    assert (upstream * signs < 0).any()


def test_tangent_round_trip_and_q_sample():
    diffusion = make_diffusion()
    x = pb.expmap0(0.3 * torch.randn(40, 4, dtype=torch.float64), 1.0)
    centers = pb.expmap0(0.2 * torch.randn(3, 4, dtype=torch.float64), 1.0)
    labels = torch.arange(40) % 3
    x0, mean, std, node_centers = diffusion.tangent_coordinates(x, labels, centers)
    torch.testing.assert_close(diffusion.to_ball(x0, mean, std, node_centers), x)
    t = torch.full((40,), 35)
    noise = torch.randn_like(x0)
    a, s = diffusion.schedule.signal[35], diffusion.schedule.noise[35]
    torch.testing.assert_close(diffusion.q_sample(x0, t, noise), a * x0 + s * noise)


def test_recover_rejects_bad_t_star():
    diffusion = make_diffusion(T=200, delta=1.0)
    with pytest.raises(ValueError, match="past the minimum"):
        diffusion.recover(torch.zeros(3, 4), None, torch.zeros(3, dtype=torch.long), torch.zeros(1, 4), 160, None)


@pytest.mark.parametrize("angular, T, delta, t_star", [("upstream", 200, 1.0, 7), ("folded", 1000, 0.4, 35)])
def test_recovery_matches_old_code(angular, T, delta, t_star):
    """Same weights, same noise seed: one-step recovery equals the old StabilityDiffusion."""
    old = pytest.importorskip("stability.hypdiff_models")
    torch.manual_seed(0)
    graph = sbm_graph(seed=4)
    adjacency = normalized_adjacency(graph.node_count, graph.edges, "cpu")
    dim = 8
    embeddings = pb.expmap0(0.4 * torch.randn(graph.node_count, dim), 1.0)
    centers = pb.expmap0(0.3 * torch.randn(3, dim), 1.0)
    labels = torch.as_tensor(graph.class_labels)
    ours = HyperbolicDiffusion(dim, 16, 2, DiffusionSchedule(T, delta, 1.0), 1.0, angular, True)
    with torch.no_grad():
        for p in ours.parameters():
            p.normal_(0, 0.1)  # non-trivial network (the output layer starts at zero)
    args = SimpleNamespace(
        r=2.0, t=1.0, radial_delta=delta, diffusion_timesteps=T, angular_noise=angular, clip_x0=True,
        recovery_sampler="one_step", denoiser_architecture="residual_gcn", denoiser_hidden_dim=16,
        denoiser_layers=2, noise_formula="exact",
    )
    reference = old.StabilityDiffusion(args, dim=dim, curvature=1.0, device="cpu")
    net, ref = ours.net, reference.gcn
    with torch.no_grad():
        for name in ("input", "output"):
            getattr(ref, name).load_state_dict(getattr(net, name).state_dict())
        ref.time.load_state_dict(net.time.state_dict())
        ref.norms.load_state_dict(net.norms.state_dict())
        for mine, theirs in zip(net.convs, ref.convs):
            theirs.lin.weight.copy_(mine.weight.T)
            theirs.bias.copy_(mine.bias)
    ours.eval()
    reference.eval()
    with torch.no_grad():
        mine = ours.recover(embeddings, adjacency, labels, centers, t_star, torch.Generator().manual_seed(5))
        theirs = reference.sample_from_tstar(embeddings, adjacency, labels, centers, t_star,
                                             torch.Generator().manual_seed(5))
    torch.testing.assert_close(mine, theirs.to(mine.dtype), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"recovery": {"t_star": 40}}, "must be < denoiser.train_t_max"),
        ({"diffusion": {"timesteps": 200, "radial_delta": 1.0}, "denoiser": {"train_t_max": 200},
          "recovery": {"t_star": 160}}, "past the minimum"),
        ({"denoiser": {"train_t_max": 2000}}, "train_t_max must be <="),
        ({"diffusion": {"angular_noise": "isotropic"}}, "diffusion.angular_noise"),
        ({"recovery": {"sampler": "ancestral"}}, "recovery.sampler"),
        ({"recovery": {"steps": 37}}, "recovery.steps"),
        ({"denoiser": {"training": "unrolled"}}, "use recovery.sampler"),
        ({"denoiser": {"train_max_drop": 1.0}}, "train_max_drop"),
    ],
)
def test_diffusion_config_checks(raw, changes, message):
    bad = copy.deepcopy(raw)
    for section, values in changes.items():
        bad[section].update(values)
    with pytest.raises(ConfigError, match=message):
        config_from_dict(bad)


def test_cora_config_uses_paper_values(raw):
    config = config_from_dict(raw)
    assert (config.diffusion.timesteps, config.diffusion.radial_delta, config.diffusion.angular_noise) == (
        1000, 0.4, "folded")


# --------------------------------------------------------------------------- training + validation


def test_denoiser_training_is_leakage_free_and_reproducible(trained):
    cpu = torch.device("cpu")
    a, history = train_denoiser(trained.graph, trained.split, trained.config, cpu, trained.ae.model,
                                trained.context, audit=True)
    b, history_b = train_denoiser(trained.graph, trained.split, trained.config, cpu, trained.ae.model,
                                  trained.context)
    assert history["loss"] == history_b["loss"] and len(history["loss"]) == trained.config.denoiser.epochs
    assert all(np.isfinite(history["loss"])) and history["skipped_epochs"] == []
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    # the encoder stays frozen and trainable flags are restored
    assert all(p.requires_grad for p in trained.ae.model.parameters())
    n = trained.graph.node_count
    audit = history["audit"]
    train_keys = pair_keys(trained.split.train, n)
    assert np.isin(audit["adjacency"], train_keys).all() and np.isin(audit["positive"], train_keys).all()
    for name in ("val", "pool", "val_negatives", "control_pairs"):
        keys = pair_keys(getattr(trained.split, name), n)
        for used in ("adjacency", "positive", "negative"):
            assert not np.isin(keys, audit[used]).any(), f"{name} leaked into {used}"
    assert not np.isin(train_keys, audit["negative"]).any()


def test_nonfinite_epochs_are_skipped_then_stop(trained, monkeypatch):
    real = denoiser_training.edge_reconstruction_loss
    calls = {"n": 0}

    def sometimes_nan(*args, **kwargs):
        calls["n"] += 1
        loss = real(*args, **kwargs)
        return loss * float("nan") if calls["n"] in (2, 4) else loss

    monkeypatch.setattr(denoiser_training, "edge_reconstruction_loss", sometimes_nan)
    config = dataclasses.replace(trained.config, denoiser=dataclasses.replace(trained.config.denoiser,
                                                                              nonfinite_max_skips=2))
    _, history = train_denoiser(trained.graph, trained.split, config, torch.device("cpu"),
                                trained.ae.model, trained.context)
    assert [s["epoch"] for s in history["skipped_epochs"]] == [1, 3]
    assert np.isnan(history["loss"][1]) and np.isfinite(history["loss"][0])
    calls["n"] = 0
    config = dataclasses.replace(config, denoiser=dataclasses.replace(config.denoiser, nonfinite_max_skips=1))
    with pytest.raises(RuntimeError, match="nonfinite_max_skips"):
        train_denoiser(trained.graph, trained.split, config, torch.device("cpu"), trained.ae.model, trained.context)


def test_validation_report_with_diffusion(trained):
    cpu = torch.device("cpu")
    diffusion, _ = train_denoiser(trained.graph, trained.split, trained.config, cpu, trained.ae.model,
                                  trained.context)
    args = (trained.ae.model, trained.graph.features, trained.ae.train_adjacency, trained.split,
            trained.context, trained.config, diffusion)
    report = validation_report(*args)
    assert set(report) == {"protocol", "no_diffusion", "clamp_only", "diffusion"}
    d = report["diffusion"]
    assert d["K"] == 3 and 0.5 < d["auc"] <= 1 and d["probability_std_across_K_mean"] > 0
    assert validation_report(*args) == report  # fixed recovery seeds -> deterministic
    poisoned = dataclasses.replace(trained.split, pool=np.full_like(trained.split.pool, -1),
                                   control_pairs=np.full_like(trained.split.control_pairs, -1))
    assert validation_report(*args[:3], poisoned, *args[4:]) == report


def test_build_diffusion_follows_config(trained):
    diffusion = build_diffusion(trained.config)
    assert diffusion.schedule.timesteps == trained.config.diffusion.timesteps
    assert diffusion.angular_noise == "folded" and diffusion.clip_x0
    assert len(diffusion.net.convs) == trained.config.denoiser.layers
