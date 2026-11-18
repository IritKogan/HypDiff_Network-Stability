"""denoiser.nonfinite_max_skips: epochs with a non-finite loss or non-finite gradients are skipped
(no parameter update) and recorded; training stops once more than the limit were skipped."""

import numpy as np
import pytest
import torch

import stability.model_training as model_training
from stability.edge_split import split_edges
from stability.model_training import train_models

from test_edge_recovery_pipeline import graph, tiny_config  # noqa: F401  (fixture)

DEVICE = torch.device("cpu")


def _split(graph, config):  # noqa: F811
    s = config.split
    return split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction, s.pool_fraction,
                       s.control_pairs, s.seed)


def _poison(monkeypatch, bad_epochs, kind):
    """Make the edge-reconstruction loss non-finite (kind 'loss') or give it an infinite gradient
    with a finite value (kind 'gradients') at the given call numbers (= denoiser epochs)."""
    real = model_training._edge_reconstruction_loss
    calls = {"n": 0}

    def wrapped(encoder, recovered, **kwargs):
        loss = real(encoder, recovered, **kwargs)
        epoch = calls["n"]
        calls["n"] += 1
        if epoch in bad_epochs:
            if kind == "loss":
                return loss * float("nan")
            return loss + 0.0 * (recovered[:1, :1] * 0.0).sqrt().sum()   # value finite, d/dx sqrt(0) = inf -> nan grad
        return loss

    monkeypatch.setattr(model_training, "_edge_reconstruction_loss", wrapped)


@pytest.mark.parametrize("kind", ["loss", "gradients"])
def test_bad_epochs_are_skipped_and_recorded(graph, monkeypatch, kind):  # noqa: F811
    config = tiny_config(denoiser={"nonfinite_max_skips": 3, "epochs": 8})
    _poison(monkeypatch, {2, 5}, kind)
    models = train_models(graph, _split(graph, config), config, DEVICE)
    assert models.history["denoiser_skipped_epochs"] == [{"epoch": 2, "reason": kind}, {"epoch": 5, "reason": kind}]
    losses = models.history["denoiser_loss"]
    assert len(losses) == config.denoiser.epochs
    assert np.isnan(losses[2]) and np.isnan(losses[5]) and np.isfinite(np.delete(losses, [2, 5])).all()
    assert all(torch.isfinite(p).all() for p in models.denoiser.parameters())


def test_too_many_bad_epochs_stop_training(graph, monkeypatch):  # noqa: F811
    config = tiny_config(denoiser={"nonfinite_max_skips": 1})
    _poison(monkeypatch, {1, 3}, "loss")
    with pytest.raises(RuntimeError, match="nonfinite_max_skips = 1"):
        train_models(graph, _split(graph, config), config, DEVICE)


def test_zero_keeps_the_old_behaviour(graph, monkeypatch):  # noqa: F811
    config = tiny_config(denoiser={"nonfinite_max_skips": 0})
    _poison(monkeypatch, {0}, "loss")
    with pytest.raises(RuntimeError, match="non-finite denoiser loss at epoch 0"):
        train_models(graph, _split(graph, config), config, DEVICE)


def test_without_bad_epochs_training_is_unchanged(graph):  # noqa: F811
    """The guard must not change a clean run: same losses for any limit."""
    split = _split(graph, tiny_config())
    a = train_models(graph, split, tiny_config(denoiser={"nonfinite_max_skips": 0}), DEVICE)
    b = train_models(graph, split, tiny_config(denoiser={"nonfinite_max_skips": 7}), DEVICE)
    assert a.history["denoiser_loss"] == b.history["denoiser_loss"]
    assert a.history["denoiser_skipped_epochs"] == []
    for p, q in zip(a.denoiser.parameters(), b.denoiser.parameters()):
        assert torch.equal(p, q)
