"""denoiser.timesteps (diffusion length T, 2026-11-24, experiment E1) and the t* guard.

* timesteps = 200 (cora.toml default) is bit-identical to the hard-coded upstream value: same
  schedule tensors, same trained denoiser, same recovered embeddings.
* stability/diffusion_schedule.py reproduces the coefficients StabilityDiffusion actually uses.
* t* past the minimum of a_t is rejected by the config and by the model.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.direction_diffusion import get_alphas
from stability.diffusion_schedule import (
    check_t_star, last_valid_t_star, noise_coefficients, signal_coefficients,
)
from stability.edge_split import split_edges
from stability.hypdiff_models import StabilityDiffusion
from stability.model_training import build_model_args, train_models
from tests.test_edge_recovery_pipeline import graph, raw_config, tiny_config  # noqa: F401  (fixture)

SCHEDULE = ("alphas", "alphas_cumprod", "sqrt_alphas_cumprod", "minus_alphas_cumprod",
            "posterior_variance", "posterior_mean_coef1", "posterior_mean_coef2")


def _denoiser(graph, **denoiser):
    args, _ = build_model_args(graph, tiny_config(denoiser=denoiser), torch.device("cpu"))
    return args, StabilityDiffusion(args, dim=args.dim, device="cpu")


def test_cora_config_keeps_the_previous_length():
    assert raw_config()["denoiser"]["timesteps"] == 200


def test_timesteps_200_equals_the_previous_hard_coded_schedule(graph):
    args, explicit = _denoiser(graph, timesteps=200)
    legacy_args = SimpleNamespace(**{k: v for k, v in vars(args).items() if k != "diffusion_timesteps"})
    torch.manual_seed(0)
    legacy = StabilityDiffusion(legacy_args, dim=args.dim, device="cpu")   # attribute absent -> 200
    torch.manual_seed(0)
    explicit = StabilityDiffusion(args, dim=args.dim, device="cpu")
    assert legacy.num_timesteps == explicit.num_timesteps == 200
    for name in SCHEDULE:
        assert torch.equal(getattr(legacy, name), getattr(explicit, name)), name
    for a, b in zip(legacy.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(a, b)
    alphas, _ = get_alphas(200)
    assert torch.equal(explicit.alphas, alphas)


def test_timesteps_200_training_and_recovery_are_bit_identical(graph):
    """Full train + recovery with the key set to 200 vs the attribute missing (old code path)."""
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    config = tiny_config()
    device = torch.device("cpu")
    import stability.model_training as mt

    original = mt.build_model_args

    def without_timesteps(*a, **k):
        args, pilot = original(*a, **k)
        delattr(args, "diffusion_timesteps")
        return args, pilot

    new = train_models(graph, split, config, device)
    mt.build_model_args = without_timesteps
    try:
        old = train_models(graph, split, config, device)
    finally:
        mt.build_model_args = original
    assert old.denoiser.num_timesteps == new.denoiser.num_timesteps == 200
    for a, b in zip(old.denoiser.state_dict().values(), new.denoiser.state_dict().values()):
        assert torch.equal(a, b)
    with torch.no_grad():
        emb = new.encoder.encode(graph.features, new.train_adjacency)[2]
        outs = []
        for models in (old, new):
            g = torch.Generator().manual_seed(11)
            outs.append(models.denoiser.sample_from_tstar(emb, new.train_adjacency, new.context.labels,
                                                          new.context.centers, config.recovery.t_star, g))
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("timesteps,delta", [(200, 1.0), (1000, 0.4), (1000, 1.0), (50, 0.0)])
def test_closed_form_matches_the_model_coefficients(graph, timesteps, delta):
    _, den = _denoiser(graph, timesteps=timesteps, radial_delta=delta)
    assert den.num_timesteps == timesteps
    a = signal_coefficients(timesteps, delta)
    s = noise_coefficients(timesteps)
    for t in (0, 1, timesteps // 3, timesteps - 1):
        assert a[t] == pytest.approx(den.signal_coefficient(t), abs=1e-6)
        assert s[t] == pytest.approx(den.noise_coefficient(t), abs=1e-12)


def test_last_valid_t_star_values():
    assert last_valid_t_star(200, 1.0) == 154          # upstream code values
    assert last_valid_t_star(1000, 0.4) == 999         # paper values: every t* valid
    assert last_valid_t_star(1000, 1.0) == 813
    assert last_valid_t_star(1000, 0.0) == 999         # no radial prior: a_t decreasing
    a = signal_coefficients(1000, 0.4)
    assert np.all(np.diff(a[76:]) < 0)                 # after the initial radial rise: decreasing


def test_noise_level_increases_with_t_in_the_e1_settings():
    for timesteps, delta in ((200, 1.0), (1000, 0.4)):
        snr = signal_coefficients(timesteps, delta)[1:] / noise_coefficients(timesteps)[1:]
        assert np.all(np.diff(snr[: last_valid_t_star(timesteps, delta)]) < 0)


def test_check_t_star():
    check_t_star(154, 200, 1.0)
    check_t_star(500, 1000, 0.4)
    with pytest.raises(ValueError, match="past the minimum"):
        check_t_star(155, 200, 1.0)
    with pytest.raises(ValueError, match="in \\[0, 199\\]"):
        check_t_star(200, 200, 0.0)


def test_config_validation_uses_timesteps_and_the_guard():
    config = tiny_config(denoiser={"timesteps": 1000, "radial_delta": 0.4, "train_t_max": 571},
                         recovery={"t_star": 500})
    assert config.denoiser.timesteps == 1000
    with pytest.raises(ValueError, match="past the minimum"):
        tiny_config(recovery={"t_star": 160})                       # T=200, delta=1
    with pytest.raises(ValueError, match="train_t_max"):
        tiny_config(denoiser={"train_t_max": 201})
    with pytest.raises(ValueError, match="timesteps"):
        tiny_config(denoiser={"timesteps": 1})
    with pytest.raises(ValueError, match="missing keys"):
        raw = raw_config()
        raw["denoiser"].pop("timesteps")
        from stability.experiment_config import config_from_dict
        config_from_dict(raw)


def test_model_rejects_t_star_past_the_minimum(graph):
    _, den = _denoiser(graph)
    x = torch.zeros(3, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="past the minimum"):
        den._denoise_from_tstar(x, None, torch.ones(1, 4), torch.zeros(3, dtype=torch.long), 170,
                                torch.Generator())


def test_long_schedule_trains_and_recovers(graph):
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    config = tiny_config(denoiser={"timesteps": 1000, "radial_delta": 0.4, "angular_noise": "folded",
                                   "train_t_max": 286}, recovery={"t_star": 250})
    models = train_models(graph, split, config, torch.device("cpu"))
    assert models.denoiser.num_timesteps == 1000
    with torch.no_grad():
        emb = models.encoder.encode(graph.features, models.train_adjacency)[2]
        out = models.denoiser.sample_from_tstar(emb, models.train_adjacency, models.context.labels,
                                                models.context.centers, 250, torch.Generator().manual_seed(1))
    assert torch.isfinite(out).all() and out.shape == emb.shape
