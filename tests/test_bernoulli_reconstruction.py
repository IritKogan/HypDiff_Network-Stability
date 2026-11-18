"""Decision 10: reconstructions draw every pair from its calibrated probability."""

import json

import numpy as np
import pytest
import torch

from stability.edge_recovery_experiment import run_experiment, run_recoveries
from stability.edge_split import split_edges
from stability.model_training import train_models
from stability.perturbation_masks import make_damage_series
from stability.probability_calibration import PlattCalibrator
from stability.validation_recovery import validation_recovery

from tests.test_edge_recovery_pipeline import graph, tiny_config  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def trained(graph):
    config = tiny_config(recovery={"edge_decision": "bernoulli", "reconstruction_sources": ["noise"]})
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    models = train_models(graph, split, config, torch.device("cpu"))
    series = make_damage_series(split.pool, split.all_edges, graph.node_count,
                                config.perturbation.damage_fractions, config.perturbation.attacks,
                                config.perturbation.series, config.perturbation.seed)
    return config, split, models, series


def constant(q):
    """Calibrator with slope 0: q at its fit prevalence (prior shift switched off by using it)."""
    return PlattCalibrator(0.0, float(np.log(q / (1 - q))), 0.5)


def test_bernoulli_frequency_follows_the_probability(graph, trained, monkeypatch):
    config, split, models, series = trained
    calibrators = {m: constant(0.3) for m in ("diffusion", "no_diffusion", "clamp_only")}
    # no prior shift: evaluate the constant probability as given
    monkeypatch.setattr(PlattCalibrator, "apply", lambda self, p, prevalence=None:
                        np.full(np.shape(p), 1 / (1 + np.exp(-self.intercept))))
    results = run_recoveries(graph, split, series, models, config, torch.device("cpu"), taus={},
                             calibrators=calibrators)
    K = config.recovery.reconstructions
    for arrays in results.values():
        removed = arrays["removed"]
        assert int(arrays["deterministic_draws"]) == K
        def close(counts):   # within 4 binomial standard deviations of q = 0.3
            draws = counts.size * K
            return abs(counts.sum() / draws - 0.3) < 4 * np.sqrt(0.21 / draws)

        for key in ("recovered", "no_diffusion", "clamp_only"):
            counts = arrays[key][removed]
            assert counts.max() <= K and close(counts), key
            assert np.all(arrays[key][~removed] == 0), key
        assert close(arrays["control_hits"])
        assert np.allclose(arrays["diffusion_expected_sum"][removed], 0.3 * K, atol=1e-4)


def test_bernoulli_draws_are_shared_across_attacks(graph, trained, monkeypatch):
    """Common random numbers: same (series, level, k, method) -> same uniforms for a pair."""
    config, split, models, series = trained
    monkeypatch.setattr(PlattCalibrator, "apply", lambda self, p, prevalence=None: np.full(np.shape(p), 0.5))
    results = run_recoveries(graph, split, series, models, config, torch.device("cpu"), taus={},
                             calibrators={m: constant(0.5) for m in ("diffusion", "no_diffusion", "clamp_only")})
    first, second = (results[a] for a in list(results)[:2])
    both = first["removed"] & second["removed"]
    assert both.any()
    assert np.array_equal(first["no_diffusion"][both], second["no_diffusion"][both])
    assert np.array_equal(first["control_hits"], second["control_hits"])


def test_bernoulli_needs_calibrators(graph, trained):
    config, split, models, series = trained
    with pytest.raises(ValueError, match="calibrators"):
        run_recoveries(graph, split, series, models, config, torch.device("cpu"), taus={})


def test_validation_fits_base_rate_calibrators(graph, trained):
    config, split, models, _ = trained
    result = validation_recovery(graph, split, models, config, torch.device("cpu"))
    n = graph.node_count
    assert result["base_rate"] == {"missing_edges": len(split.val) + len(split.pool),
                                   "non_edges": n * (n - 1) // 2 - len(split.all_edges)}
    for method in ("diffusion", "no_diffusion", "clamp_only"):
        calibration = result[method]["calibration"]
        calibrator = PlattCalibrator(**calibration["calibrator"])
        expected = result["base_rate"]["missing_edges"] / sum(result["base_rate"].values())
        assert calibrator.fit_prevalence == pytest.approx(expected)
        assert np.isfinite(calibration["cross_fitted"]["log_loss"])


@pytest.mark.parametrize("decision", ["bernoulli", "threshold"])
def test_end_to_end_both_edge_decisions(graph, tmp_path, decision):
    config = tiny_config(recovery={"edge_decision": decision, "reconstruction_sources": ["noise"]})
    run_dir = run_experiment(config, output_dir=tmp_path / decision, graph=graph, make_plots=False)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["edge_decision"] == decision
    for levels in summary["attacks"].values():
        for entry in levels:
            assert ("expected_p_e" in entry) == (decision == "bernoulli")
            assert entry["p_e"]["count"] > 0
