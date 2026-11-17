"""Decision 10 (chosen definition): relative recoverability. A removed edge reappears in a
reconstruction if it scores above a random control pair (true non-edge) of that reconstruction."""

import json

import numpy as np
import pytest
import torch

import stability.edge_recovery_experiment as experiment
from stability.edge_recovery_experiment import run_experiment, run_recoveries
from stability.edge_split import split_edges
from stability.model_training import train_models
from stability.perturbation_masks import make_damage_series

from tests.test_edge_recovery_pipeline import graph, tiny_config  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def trained(graph):
    config = tiny_config(recovery={"edge_decision": "pairwise", "reconstructions": 4})
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    models = train_models(graph, split, config, torch.device("cpu"))
    series = make_damage_series(split.pool, split.all_edges, graph.node_count,
                                config.perturbation.damage_fractions, config.perturbation.attacks,
                                config.perturbation.series, config.perturbation.seed)
    return config, split, models, series


def fake_decoder(monkeypatch, split, removed_score, control_scores):
    """Removed edges get a fixed score; control pairs get the given scores (by control index)."""
    control = torch.as_tensor(split.control_pairs)

    def decode(encoder, embeddings, pairs):
        if pairs.shape == control.shape and torch.equal(pairs, control):
            return np.asarray(control_scores, dtype=np.float64)
        return np.full(pairs.shape[0], removed_score, dtype=np.float64)

    monkeypatch.setattr(experiment, "_decode", decode)


def test_frequency_estimates_the_share_of_control_pairs_scored_lower(graph, trained, monkeypatch):
    config, split, models, series = trained
    C = len(split.control_pairs)
    control_scores = np.linspace(0.0, 1.0, C)
    fake_decoder(monkeypatch, split, 0.7, control_scores)        # beats ~70% of control pairs
    results = run_recoveries(graph, split, series, models, config, torch.device("cpu"), taus={})
    K = config.recovery.reconstructions
    share = np.mean(control_scores < 0.7)
    for arrays in results.values():
        removed = arrays["removed"]
        assert int(arrays["deterministic_draws"]) == K
        for key in ("recovered", "no_diffusion", "clamp_only"):
            counts = arrays[key][removed]
            draws = counts.size * K
            assert abs(counts.sum() / draws - share) < 4 * np.sqrt(share * (1 - share) / draws), key
            assert np.all(arrays[key][~removed] == 0), key
        assert np.allclose(arrays["diffusion_expected_sum"][removed], share * K, atol=1e-5)


def test_control_pairs_are_never_compared_with_themselves_and_centre_on_one_half(graph, trained, monkeypatch):
    config, split, models, series = trained
    C = len(split.control_pairs)
    scores = np.zeros(C)
    scores[0] = 1.0                                   # the top control pair beats every other one
    fake_decoder(monkeypatch, split, 0.5, scores)
    results = run_recoveries(graph, split, series, models, config, torch.device("cpu"), taus={})
    K = config.recovery.reconstructions
    for arrays in results.values():
        hits = arrays["control_hits"]
        assert np.all(hits[:, :, 0] == K)            # would sometimes lose a tie with itself otherwise
        assert np.all(hits[:, :, 1:] == 0)            # ties (0 vs 0) and losses never count
    fake_decoder(monkeypatch, split, 0.5, np.random.default_rng(0).random(C))
    results = run_recoveries(graph, split, series, models, config, torch.device("cpu"), taus={})
    hits = next(iter(results.values()))["control_hits"]
    assert abs(hits.mean() / K - 0.5) < 0.06


def test_pairwise_runs_are_reproducible(graph, trained):
    config, split, models, series = trained
    first = run_recoveries(graph, split, series, models, config, torch.device("cpu"), taus={})
    again = run_recoveries(graph, split, series, models, config, torch.device("cpu"), taus={})
    for attack in first:
        for key, value in first[attack].items():
            assert np.array_equal(value, again[attack][key]), (attack, key)


def test_end_to_end_pairwise(graph, tmp_path):
    config = tiny_config(recovery={"edge_decision": "pairwise"})
    run_dir = run_experiment(config, output_dir=tmp_path / "pairwise", graph=graph, make_plots=True)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["edge_decision"] == "pairwise"
    for levels in summary["attacks"].values():
        for entry in levels:
            assert 0.0 <= entry["p_e"]["mean"] <= 1.0
            assert abs(entry["control_false_positive"]["mean"] - 0.5) < 0.15
            assert "expected_p_e" in entry
