"""Tests for Stage 4b: iterative sampler, unrolled (E2) training, informativeness metrics, E2 script."""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hypstab.diffusion.model import HyperbolicDiffusion, sampling_steps
from hypstab.diffusion.schedule import DiffusionSchedule
from hypstab.evaluation import informativeness as inf
from hypstab.geometry import poincare as pb
from hypstab.models.hgcn import normalized_adjacency
from hypstab.training.denoiser import train_denoiser

from .test_autoencoder import sbm_graph
from .test_diffusion import trained  # noqa: F401  (module fixture)


def test_sampling_steps():
    assert sampling_steps(35, 8) == [35, 30, 25, 20, 15, 10, 5, 0]
    assert sampling_steps(7, 8) == list(range(7, -1, -1))     # every step
    assert sampling_steps(150, 8) == [150, 129, 107, 86, 64, 43, 21, 0]
    assert sampling_steps(3, 10) == [3, 2, 1, 0]              # never more than t* + 1
    with pytest.raises(ValueError):
        sampling_steps(10, 1)


def test_reverse_chain_with_oracle_recovers_x0():
    """If the network predicted the true x0, the deterministic chain returns it exactly."""
    diffusion = HyperbolicDiffusion(4, 8, 1, DiffusionSchedule(1000, 0.4, 1.0), 1.0, "folded", clip_x0=False)
    x0 = torch.randn(30, 4, dtype=torch.float64)
    diffusion.predict_x0 = lambda x_t, adjacency, t: x0
    noise = torch.randn_like(x0)
    for steps in ([35, 30, 20, 0], sampling_steps(150, 8)):
        x_t = diffusion.q_sample(x0, torch.full((30,), steps[0]), noise)
        torch.testing.assert_close(diffusion.reverse_chain(x_t, None, steps), x0)
    with pytest.raises(ValueError):
        diffusion.reverse_chain(x0, None, [5, 3])          # must end at 0


@pytest.mark.parametrize("angular, T, delta, t_star", [("upstream", 200, 1.0, 7), ("folded", 1000, 0.4, 12)])
def test_iterative_matches_old_code_every_step(angular, T, delta, t_star):
    """steps = t* + 1 equals the old StabilityDiffusion 'iterative' sampler (exact noise formula)."""
    old = pytest.importorskip("stability.hypdiff_models")
    torch.manual_seed(1)
    graph = sbm_graph(seed=6)
    adjacency = normalized_adjacency(graph.node_count, graph.edges, "cpu")
    dim = 8
    embeddings = pb.expmap0(0.4 * torch.randn(graph.node_count, dim), 1.0)
    centers = pb.expmap0(0.3 * torch.randn(3, dim), 1.0)
    labels = torch.as_tensor(graph.class_labels)
    ours = HyperbolicDiffusion(dim, 16, 2, DiffusionSchedule(T, delta, 1.0), 1.0, angular, True)
    with torch.no_grad():
        for p in ours.parameters():
            p.normal_(0, 0.1)
    args = SimpleNamespace(
        r=2.0, t=1.0, radial_delta=delta, diffusion_timesteps=T, angular_noise=angular, clip_x0=True,
        recovery_sampler="iterative", denoiser_architecture="residual_gcn", denoiser_hidden_dim=16,
        denoiser_layers=2, noise_formula="exact",
    )
    reference = old.StabilityDiffusion(args, dim=dim, curvature=1.0, device="cpu")
    net, ref = ours.net, reference.gcn
    with torch.no_grad():
        ref.input.load_state_dict(net.input.state_dict())
        ref.output.load_state_dict(net.output.state_dict())
        ref.time.load_state_dict(net.time.state_dict())
        ref.norms.load_state_dict(net.norms.state_dict())
        for mine, theirs in zip(net.convs, ref.convs):
            theirs.lin.weight.copy_(mine.weight.T)
            theirs.bias.copy_(mine.bias)
    ours.eval()
    reference.eval()
    with torch.no_grad():
        mine = ours.recover(embeddings, adjacency, labels, centers, t_star, torch.Generator().manual_seed(3),
                            sampler="iterative", steps=t_star + 1)
        theirs = reference.sample_from_tstar(embeddings, adjacency, labels, centers, t_star,
                                             torch.Generator().manual_seed(3))
    torch.testing.assert_close(mine, theirs.to(mine.dtype), rtol=1e-4, atol=1e-5)


def test_unrolled_training_runs_and_backpropagates_through_the_chain(trained):  # noqa: F811
    raw = copy.deepcopy(trained.raw)
    config = trained.config
    config = dataclasses.replace(
        config,
        denoiser=dataclasses.replace(config.denoiser, training="unrolled", epochs=3),
        recovery=dataclasses.replace(config.recovery, sampler="iterative", steps=4),
    )
    config.validate()
    diffusion, history = train_denoiser(trained.graph, trained.split, config, torch.device("cpu"),
                                        trained.ae.model, trained.context)
    assert len(history["loss"]) == 3 and all(np.isfinite(history["loss"]))
    # gradients reach the network through every step of the chain (output layer starts at zero, so
    # after training it must have moved)
    assert diffusion.net.output.weight.abs().sum() > 0
    del raw


# --------------------------------------------------------------------------- informativeness metrics


def test_stratified_auc_matches_old_code():
    old = pytest.importorskip("stability.sample_diversity")
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    score = labels * 0.3 + rng.random(400)
    strata = rng.random(400) + 0.2 * labels
    assert inf.stratified_auc(score, labels, strata) == pytest.approx(old.stratified_auc(score, labels, strata))


def test_logistic_and_ll_gain_match_sklearn():
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    old = pytest.importorskip("stability.sample_diversity")
    rng = np.random.default_rng(1)
    n = 600
    labels = rng.integers(0, 2, n)
    base = np.clip(0.3 + 0.4 * labels + 0.2 * rng.standard_normal(n), 0.01, 0.99)
    extra = labels * 0.5 + rng.standard_normal(n)
    x = np.c_[np.log(base / (1 - base)), extra]
    theta = inf.fit_logistic(x, labels)
    reference = sklearn_lm.LogisticRegression().fit(x, labels)
    # sklearn's lbfgs stops at tol 1e-4; ours converges tighter -> agreement to ~1e-3 relative
    np.testing.assert_allclose(theta, np.r_[reference.intercept_, reference.coef_[0]], rtol=2e-3, atol=1e-4)
    # informative extra -> positive gain; same order of magnitude as the old (sklearn folds) value
    ours = inf.incremental_log_likelihood(base, extra, labels)["gain_per_pair"]
    theirs = old.incremental_log_likelihood(base, extra, labels)["gain_per_pair"]
    assert ours > 0.005 and theirs > 0.005 and abs(ours - theirs) < 0.005
    noise_gain = inf.incremental_log_likelihood(base, rng.standard_normal(n), labels)["gain_per_pair"]
    assert noise_gain < 0.005


def test_stratified_folds_are_balanced_partition():
    labels = np.r_[np.zeros(53), np.ones(47)].astype(int)
    folds = inf.stratified_folds(labels, 5, seed=0)
    assert sorted(np.concatenate(folds).tolist()) == list(range(100))
    for test in folds:
        assert 9 <= labels[test].sum() <= 10


def test_variant_metrics_matches_old_e1_metrics():
    old = pytest.importorskip("stability.sample_diversity")
    rng = np.random.default_rng(2)
    K, n = 6, 300
    prob_pos = np.clip(0.6 + 0.2 * rng.standard_normal((1, n)) + 0.05 * rng.standard_normal((K, n)), 0.001, 0.999)
    prob_neg = np.clip(0.4 + 0.2 * rng.standard_normal((1, n)) + 0.05 * rng.standard_normal((K, n)), 0.001, 0.999)
    clamp_pos, clamp_neg = prob_pos.mean(0), prob_neg.mean(0)
    ours = inf.variant_metrics(prob_pos, prob_neg, clamp_pos, clamp_neg, target_fpr=0.05, quality_tolerance=0.0,
                               informative_auc=0.55, min_ll_gain=0.001)
    theirs = old.schedule_variant_metrics(prob_pos, prob_neg, clamp_pos, clamp_neg, 0.05)
    for key in ("tau", "auc_mean_probability", "auc_clamp_only", "probability_std_positive",
                "expected_share_std_across_K"):
        assert ours[key] == pytest.approx(theirs[key]), key
    assert ours["p_e"]["stratified_auc_given_own_mean"] == pytest.approx(theirs["p_e"]["stratified_auc_given_own_mean"])
    assert ours["p_e"]["share_intermediate"] == pytest.approx(theirs["p_e"]["share_intermediate"])
    assert ours["spread"]["stratified_auc_given_own_mean"] == pytest.approx(
        theirs["spread"]["stratified_auc_given_own_mean"])
    # pure per-reconstruction noise around the mean -> not informative
    assert not ours["informative"]


# --------------------------------------------------------------------------- E2 script (tiny graph)


def test_e2_plan_file_is_consistent():
    import tomllib
    from pathlib import Path

    from hypstab.config import config_from_dict

    root = Path(__file__).resolve().parents[2]
    plan = tomllib.loads((root / "configs" / "hypstab" / "e2.toml").read_text(encoding="utf-8"))
    raw = tomllib.loads((root / "configs" / "hypstab" / "cora.toml").read_text(encoding="utf-8"))
    import sys

    sys.path.insert(0, str(root / "scripts"))
    from hypstab_e2 import variant_config

    names = []
    for v in plan["variant"]:
        config = variant_config(raw, v["overrides"], smoke=False)   # every variant is a valid config
        if "denoiser_from" in v:
            assert v["denoiser_from"] in names
        if config.denoiser.training == "unrolled":
            assert config.recovery.sampler == "iterative"
        names.append(v["name"])
    assert set(plan["criteria"]) == {"target_fpr", "quality_tolerance", "informative_auc", "min_ll_gain"}
    config_from_dict(raw)
