"""Unit tests for ``diffusion.predict_noise_from_start``.

The forward process implemented by ``diffusion.q_sample`` is

    x_t = (sqrt(abar_t) + prior(t)) * x0 + sqrt(1 - abar_t) * eps

so the exact inverse is

    eps = (x_t - (sqrt(abar_t) + prior(t)) * x0) / sqrt(1 - abar_t).

Upstream HypDiff (RingBDStack/HypDiff @ 1620e3c) instead uses

    eps_up = ((1/sqrt(abar_t) + prior(t)) * x_t - x0) / sqrt(1/abar_t - 1)

Substituting x_t gives the closed-form error

    eps_up - eps = a(t) * x0 + b(t) * eps
    a(t) = prior * (1 + abar + prior * sqrt(abar)) / sqrt(1 - abar)
    b(t) = prior * sqrt(abar)

These tests build x_t from a *known* noise tensor and check both formulas at
t = 7 (our partial-noise recovery t*) and t = 199 (= T - 1). Both are reached
through ``StabilityDiffusion(noise_formula=...)``; "upstream" delegates to the
unmodified upstream method in models/direction_diffusion.py.
"""

from types import SimpleNamespace

import pytest
import torch

from stability.hypdiff_models import StabilityDiffusion

NUM_NODES = 64
DIM = 16


def reference_upstream_formula(model, x_t, t, x0):
    """Verbatim copy of upstream HypDiff 1620e3c predict_noise_from_start."""
    prio = model.getpri(t)
    return (
        ((model.extract(model.sqrt_recip_alphas_cumprod, t, x_t.shape) + prio) * x_t - x0)
        / model.extract(model.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    )


def _make(noise_formula):
    torch.manual_seed(0)
    return StabilityDiffusion(
        SimpleNamespace(r=2.0, t=1.0), dim=DIM, noise_formula=noise_formula
    ).to("cpu")


@pytest.fixture(scope="module")
def model():
    return _make("exact")


@pytest.fixture(scope="module")
def upstream_model():
    return _make("upstream")


def upstream_predict_noise_from_start(model, x_t, t, x0):
    return _make("upstream").predict_noise_from_start(x_t, t, x0)


@pytest.fixture(scope="module")
def known_tensors():
    generator = torch.Generator().manual_seed(1234)
    x0 = torch.randn(NUM_NODES, DIM, generator=generator, dtype=torch.float64).clamp(-1.0, 1.0)
    eps = torch.randn(NUM_NODES, DIM, generator=generator, dtype=torch.float64)
    return x0, eps


def _batch_t(timestep):
    return torch.full((NUM_NODES,), timestep, dtype=torch.long)


def _forward(model, x0, eps, timestep):
    return model.q_sample(x0, _batch_t(timestep), None, eps, None, None, restrict=False)


@pytest.mark.parametrize("timestep", [7, 199])
def test_new_formula_recovers_known_noise_exactly(model, known_tensors, timestep):
    x0, eps = known_tensors
    x_t = _forward(model, x0, eps, timestep)

    recovered = model.predict_noise_from_start(x_t, _batch_t(timestep), x0)

    torch.testing.assert_close(recovered, eps, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("timestep", [7, 199])
def test_upstream_formula_error_matches_closed_form(model, known_tensors, timestep):
    x0, eps = known_tensors
    x_t = _forward(model, x0, eps, timestep)

    recovered = upstream_predict_noise_from_start(model, x_t, _batch_t(timestep), x0)

    sqrt_abar = model.sqrt_alphas_cumprod[timestep].item()
    sqrt_one_minus_abar = model.minus_alphas_cumprod[timestep].item()
    prior = model.getpri(_batch_t(timestep))[0, 0].item()
    a = prior * (1.0 + sqrt_abar**2 + prior * sqrt_abar) / sqrt_one_minus_abar
    b = prior * sqrt_abar

    torch.testing.assert_close(recovered - eps, a * x0 + b * eps, atol=1e-10, rtol=0.0)
    # The upstream formula is NOT an exact inverse: the x0 leak is material.
    assert a > 0.2


def test_upstream_x0_leak_values_at_t7_and_t199(model):
    """Pin the measured size of the upstream error (regression guard)."""
    def leak(timestep):
        sqrt_abar = model.sqrt_alphas_cumprod[timestep].item()
        s1 = model.minus_alphas_cumprod[timestep].item()
        prior = model.getpri(_batch_t(timestep))[0, 0].item()
        return prior * (1.0 + sqrt_abar**2 + prior * sqrt_abar) / s1

    assert leak(7) == pytest.approx(0.209, abs=1e-3)
    assert leak(199) == pytest.approx(0.379, abs=1e-3)


def test_formulas_agree_at_t0_where_prior_vanishes(model, known_tensors):
    x0, eps = known_tensors
    x_t = _forward(model, x0, eps, 0)

    new = model.predict_noise_from_start(x_t, _batch_t(0), x0)
    old = upstream_predict_noise_from_start(model, x_t, _batch_t(0), x0)

    torch.testing.assert_close(new, old, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(new, eps, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("timestep", [7, 199])
def test_new_formula_recovers_direction_restricted_noise(model, known_tensors, timestep):
    """With restrict=True q_sample flips noise signs per label; the inverse must return that noise."""
    x0, eps = known_tensors
    labels = torch.arange(NUM_NODES) % 2
    direction = [torch.ones(DIM, dtype=torch.float64), -torch.ones(DIM, dtype=torch.float64)]
    expected = eps.clone()
    expected[labels == 1] *= -1.0

    x_t = model.q_sample(x0, _batch_t(timestep), direction, eps.clone(), labels, None, restrict=True)
    recovered = model.predict_noise_from_start(x_t, _batch_t(timestep), x0)

    torch.testing.assert_close(recovered, expected, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("timestep", [7, 199])
def test_upstream_option_calls_unmodified_upstream_method(upstream_model, known_tensors, timestep):
    x0, eps = known_tensors
    x_t = _forward(upstream_model, x0, eps, timestep)
    t = _batch_t(timestep)

    torch.testing.assert_close(
        upstream_model.predict_noise_from_start(x_t, t, x0),
        reference_upstream_formula(upstream_model, x_t, t, x0),
        atol=0.0,
        rtol=0.0,
    )


def test_noise_formula_comes_from_args_and_rejects_unknown_values():
    args = SimpleNamespace(r=2.0, t=1.0, noise_formula="upstream")
    assert StabilityDiffusion(args, dim=DIM).noise_formula == "upstream"
    assert StabilityDiffusion(SimpleNamespace(r=2.0, t=1.0), dim=DIM).noise_formula == "exact"
    with pytest.raises(ValueError, match="noise_formula"):
        StabilityDiffusion(SimpleNamespace(r=2.0, t=1.0), dim=DIM, noise_formula="ddpm")
