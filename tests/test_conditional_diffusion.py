"""Decision 9: conditional geometric diffusion and the stochastic (generalised DDIM) sampler."""

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from stability.edge_split import split_edges
from stability.experiment_config import config_from_dict
from stability.hypdiff_models import RESTRICTED_NOISE_MEAN, StabilityDiffusion
from stability.model_training import build_model_args, train_models

from tests.test_edge_recovery_pipeline import graph, raw_config, tiny_config  # noqa: F401  (fixture)


def make_denoiser(graph, **sections):
    args, _ = build_model_args(graph, tiny_config(**sections), torch.device("cpu"))
    return StabilityDiffusion(args, dim=args.dim, device="cpu")


def geometry(nodes, dim, communities=3, seed=0):
    rng = torch.Generator().manual_seed(seed)
    direction = torch.randn(communities, dim, generator=rng, dtype=torch.float64)
    labels = torch.randint(0, communities, (nodes,), generator=rng)
    return direction, labels


# ---------------------------------------------------------------- sampler maths


@pytest.mark.parametrize("angular_noise, centred_std", [
    ("upstream", 1.0), ("folded", math.sqrt(1 - RESTRICTED_NOISE_MEAN ** 2))])
@pytest.mark.parametrize("eta", [0.5, 1.0])
@pytest.mark.parametrize("t, t_prev", [(7, 6), (50, 40), (150, 100), (100, 0)])
def test_stochastic_step_preserves_forward_marginals_with_an_oracle(graph, eta, t, t_prev,
                                                                    angular_noise, centred_std):
    """If x0_hat is the true x0 and x_t ~ q(x_t | x0), then x_prev ~ q(x_prev | x0) in
    mean and variance, for both angular-noise modes (folded noise is not zero-mean)."""
    denoiser = make_denoiser(graph, denoiser={"angular_noise": angular_noise},
                             recovery={"sampler": "stochastic", "eta": eta})
    nodes, dim = 40000, 4
    direction, labels = geometry(nodes, dim)
    gen = torch.Generator().manual_seed(1)
    x0 = torch.randn(nodes, dim, generator=gen, dtype=torch.float64)
    like = torch.zeros(nodes, dim, dtype=torch.float64)
    eps = denoiser.restricted_noise(like, direction, labels, gen)
    x_t = denoiser.signal_coefficient(t) * x0 + denoiser.noise_coefficient(t) * eps
    mean = denoiser.restricted_noise_mean(like, direction, labels)
    fresh = denoiser.restricted_noise(like, direction, labels, gen)
    x_prev = denoiser.stochastic_step(x_t, x0, t, t_prev, fresh, mean)
    residual = (x_prev - denoiser.signal_coefficient(t_prev) * x0) / denoiser.noise_coefficient(t_prev)
    centred = residual - mean
    assert centred.mean().item() == pytest.approx(0.0, abs=0.01)
    assert centred.std().item() == pytest.approx(centred_std, rel=0.02)
    # and the noise actually changed: not a deterministic function of x_t
    assert not torch.allclose(x_prev, denoiser.stochastic_step(
        x_t, x0, t, t_prev, denoiser.restricted_noise(like, direction, labels, gen), mean))


def test_upstream_angular_noise_is_statistically_isotropic_and_folded_is_not(graph):
    """Paper Eq. 5 / upstream code: sgn(direction) * eps is still N(0, I) (sign flip of a
    symmetric Gaussian). Folded (DDM, Eq. C.1): sgn(direction) * |eps| lies in the
    community direction's orthant, mean sqrt(2/pi) * sign."""
    direction, labels = geometry(20000, 4)
    like = torch.zeros(20000, 4, dtype=torch.float64)
    signs = torch.sign(direction[labels])
    upstream = make_denoiser(graph)
    noise = upstream.restricted_noise(like, direction, labels, torch.Generator().manual_seed(2))
    assert (noise * signs).mean().item() == pytest.approx(0.0, abs=0.02)
    assert ((noise * signs) >= 0).double().mean().item() == pytest.approx(0.5, abs=0.02)
    assert torch.equal(upstream.restricted_noise_mean(like, direction, labels), torch.zeros_like(like))
    folded = make_denoiser(graph, denoiser={"angular_noise": "folded"})
    noise = folded.restricted_noise(like, direction, labels, torch.Generator().manual_seed(2))
    assert torch.all(noise * signs >= 0)
    assert torch.allclose(folded.restricted_noise_mean(like, direction, labels), RESTRICTED_NOISE_MEAN * signs)


def test_signal_coefficient_is_not_monotone_and_stochastic_range_is_checked(graph):
    denoiser = make_denoiser(graph, recovery={"sampler": "stochastic", "eta": 1.0, "steps": 200})
    assert denoiser.last_valid_stochastic_t() == 154           # radial_delta = 1 (upstream)
    assert denoiser.signal_coefficient(199) > denoiser.signal_coefficient(154)
    denoiser.check_stochastic_range(150)
    with pytest.raises(ValueError, match="radial prior"):
        denoiser.check_stochastic_range(170)
    deterministic = make_denoiser(graph, recovery={"sampler": "stochastic", "eta": 0.0})
    deterministic.check_stochastic_range(199)                   # eta = 0 is always valid
    flat = make_denoiser(graph, denoiser={"radial_delta": 0.0},
                         recovery={"sampler": "stochastic", "eta": 1.0})
    assert flat.last_valid_stochastic_t() == 199


def test_training_fails_fast_when_the_stochastic_sampler_cannot_run(graph):
    config = tiny_config(recovery={"sampler": "stochastic", "eta": 1.0, "t_star": 180, "steps": 200})
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    with pytest.raises(ValueError, match="stochastic sampler invalid"):
        train_models(graph, split, config, torch.device("cpu"))


def test_schedule_is_strided_and_ends_at_zero(graph):
    denoiser = make_denoiser(graph, recovery={"sampler": "stochastic", "steps": 20})
    schedule = denoiser.stochastic_schedule(100)
    assert schedule[0] == 100 and schedule[-1] == 0 and len(schedule) == 20
    assert schedule == sorted(schedule, reverse=True) and len(set(schedule)) == len(schedule)
    assert denoiser.stochastic_schedule(7) == list(range(7, -1, -1))
    assert denoiser.stochastic_schedule(0) == [0]


# ---------------------------------------------------------------- sampler behaviour


def _run(denoiser, graph, seed, t_star=30, condition=None):
    """x_start (= condition) is a fixed standardised embedding; tiny_config uses dim = 4."""
    x = condition if condition is not None else torch.randn(
        graph.node_count, 4, generator=torch.Generator().manual_seed(99), dtype=torch.float64)
    edges = torch.as_tensor(graph.edges.T)
    direction, labels = geometry(graph.node_count, x.shape[1], seed=4)
    return denoiser._denoise_from_tstar(x, edges, direction, labels, t_star,
                                        torch.Generator().manual_seed(seed))


def test_eta_zero_full_stride_equals_the_iterative_sampler(graph):
    torch.manual_seed(0)
    iterative = make_denoiser(graph, denoiser={"architecture": "upstream_gcn"},
                              recovery={"sampler": "iterative"})
    stochastic = make_denoiser(graph, denoiser={"architecture": "upstream_gcn"},
                               recovery={"sampler": "stochastic", "eta": 0.0, "steps": 200})
    stochastic.load_state_dict(iterative.state_dict())
    with torch.no_grad():
        a = _run(iterative, graph, seed=5)
        b = _run(stochastic, graph, seed=5)
    assert torch.allclose(a, b, atol=1e-6)


def test_eta_positive_samples_differ_across_seeds_and_repeat_per_seed(graph):
    torch.manual_seed(0)
    denoiser = make_denoiser(graph, denoiser={"architecture": "upstream_gcn"},
                             recovery={"sampler": "stochastic", "eta": 1.0, "steps": 10})
    with torch.no_grad():
        first, again, other = _run(denoiser, graph, 5), _run(denoiser, graph, 5), _run(denoiser, graph, 6)
    assert torch.equal(first, again)
    assert not torch.allclose(first, other)


def test_stochastic_sampler_calls_the_network_steps_times(graph, monkeypatch):
    denoiser = make_denoiser(graph, recovery={"sampler": "stochastic", "eta": 1.0, "steps": 6})
    calls = []
    original = denoiser.gcn.forward
    monkeypatch.setattr(denoiser.gcn, "forward", lambda *a, **k: calls.append(1) or original(*a, **k))
    with torch.no_grad():
        _run(denoiser, graph, seed=1, t_star=60)
    assert len(calls) == 6


# ---------------------------------------------------------------- conditional networks


def test_conditional_residual_network_returns_its_condition_at_initialisation():
    from stability.denoisers import ConditionalResidualGCNDenoiser

    network = ConditionalResidualGCNDenoiser(dim=4, hidden_dim=8, layers=2)
    x, condition = torch.randn(5, 4), torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert torch.equal(network(x, edge_index, torch.full((5,), 0.3), condition), condition)


def test_conditional_gcn_copies_upstream_gcndec_widths():
    from models.direction_diffusion import gcndec
    from stability.denoisers import ConditionalGCNDenoiser

    upstream, conditional = gcndec(4), ConditionalGCNDenoiser(4)
    assert conditional.conv1.in_channels == 2 * 4 + 1 and upstream.conv1.in_channels == 4 + 1
    for name in ("conv2", "conv3", "conv4"):
        assert getattr(conditional, name).out_channels == getattr(upstream, name).out_channels
    for name in ("fc1", "fc2", "fc3"):
        assert getattr(conditional, name).weight.shape == getattr(upstream, name).weight.shape
    out = conditional(torch.randn(6, 4), torch.tensor([[0, 1], [1, 2]]), torch.zeros(6), torch.randn(6, 4))
    assert out.shape == (6, 4)


@pytest.mark.parametrize("architecture", ["conditional_gcn", "conditional_residual_gcn"])
def test_samplers_pass_the_damaged_embedding_as_condition(graph, architecture, monkeypatch):
    denoiser = make_denoiser(graph, denoiser={"architecture": architecture, "training": "conditional"},
                             recovery={"sampler": "stochastic", "eta": 1.0, "steps": 4})
    seen = []
    original = denoiser.gcn.forward
    monkeypatch.setattr(denoiser.gcn, "forward",
                        lambda x, e, t, condition: seen.append(condition.clone()) or original(x, e, t, condition))
    condition = torch.randn(graph.node_count, 4, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    with torch.no_grad():
        _run(denoiser, graph, seed=1, t_star=20, condition=condition)
    assert len(seen) == 4 and all(torch.allclose(c.double(), condition) for c in seen)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize(
    "section, values, message",
    [
        ("denoiser", {"architecture": "conditional_gcn", "training": "generative"}, "condition"),
        ("denoiser", {"t_sampling": "per_batch"}, "t_sampling"),
        ("denoiser", {"edge_loss_weight": -1.0}, "edge_loss_weight"),
        ("denoiser", {"angular_noise": "half"}, "angular_noise"),
        ("recovery", {"sampler": "ddpm"}, "sampler"),
        ("recovery", {"eta": 1.5}, "eta"),
        ("recovery", {"steps": 0}, "steps"),
    ],
)
def test_config_rejects_invalid_conditional_settings(section, values, message):
    raw = raw_config()
    raw[section].update(values)
    with pytest.raises(ValueError, match=message):
        config_from_dict(raw)


# ---------------------------------------------------------------- conditional training


def conditional_config(**extra):
    denoiser = {"training": "conditional", "architecture": "conditional_residual_gcn",
                "train_t_max": 200, "epochs": 30, "edge_loss_weight": 1.0}
    denoiser.update(extra)
    return tiny_config(denoiser=denoiser, recovery={"sampler": "stochastic", "eta": 1.0, "t_star": 30,
                                                    "steps": 5})


def test_conditional_training_does_not_read_pool_edges(graph, monkeypatch):
    from stability.edge_split import EdgeSplit

    config = conditional_config(epochs=5)
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    forbidden = split.forbidden_negative_keys()
    first = train_models(graph, split, config, torch.device("cpu"))
    monkeypatch.setattr(EdgeSplit, "forbidden_negative_keys", lambda self: forbidden)
    second = train_models(graph, replace(split, pool=split.pool[:0]), config, torch.device("cpu"))
    for name, tensor in first.denoiser.state_dict().items():
        assert torch.equal(tensor, second.denoiser.state_dict()[name]), name
    assert len(first.history["denoiser_mse"]) == config.denoiser.epochs


def test_conditional_training_reduces_loss_and_keeps_encoder_frozen(graph):
    from stability.model_training import build_context, train_denoiser, train_encoder

    config = conditional_config(epochs=60)
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    encoder, args, adjacency, _, _ = train_encoder(graph, split, config, torch.device("cpu"))
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    context = build_context(graph, config, encoder, adjacency, torch.device("cpu"))
    _, history = train_denoiser(graph, split, config, torch.device("cpu"), encoder, args, adjacency, context)
    losses = history["denoiser_loss"]
    assert np.mean(losses[-15:]) < np.mean(losses[:15])
    for name, tensor in encoder.state_dict().items():
        assert torch.equal(tensor, before[name]), name


@pytest.mark.parametrize("t_sampling, distinct", [("per_graph", 1), ("per_node", None)])
def test_conditional_loss_t_sampling(graph, monkeypatch, t_sampling, distinct):
    denoiser = make_denoiser(graph, denoiser={"architecture": "conditional_residual_gcn",
                                              "training": "conditional"})
    seen = []
    original = denoiser.q_sample
    monkeypatch.setattr(denoiser, "q_sample", lambda x, t, *a, **k: seen.append(t.clone()) or original(x, t, *a, **k))
    n = graph.node_count
    prototypes = torch.zeros(1, 4, dtype=torch.float64)
    labels = torch.zeros(n, dtype=torch.long)
    clean = 0.1 * torch.randn(n, 4, dtype=torch.float64)
    damaged = clean + 0.01 * torch.randn(n, 4, dtype=torch.float64)
    from stability.model_training import normalized_adjacency

    adjacency = normalized_adjacency(n, graph.edges, torch.device("cpu"))
    torch.manual_seed(0)
    mse, recovered = denoiser.conditional_loss(clean, damaged, adjacency, labels, prototypes, 200, t_sampling)
    assert torch.isfinite(mse) and recovered.shape == clean.shape
    unique = torch.unique(seen[0]).numel()
    assert unique == 1 if distinct == 1 else unique > 1
