import numpy as np
import pytest

from stability.probability_calibration import (
    PlattCalibrator,
    base_rate_weights,
    cross_fitted_quality,
    fit_platt,
    logit,
    sigmoid,
)


def synthetic(prevalence_weight, n=4000, seed=0):
    """Scores whose true log-odds are 2 * logit(score) - 1 at a 50/50 base rate."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, size=2 * n)
    truth = sigmoid(2 * logit(p) - 1)
    y = rng.random(2 * n) < truth
    return p[y], p[~y]


def test_fit_platt_recovers_known_parameters_unweighted():
    positive, negative = synthetic(1.0, n=40000)
    calibrator = fit_platt(positive, negative, 1.0, 1.0)
    assert calibrator.slope == pytest.approx(2.0, abs=0.06)
    assert calibrator.intercept == pytest.approx(-1.0, abs=0.06)


def test_weights_shift_only_the_intercept_by_the_log_odds_ratio():
    positive, negative = synthetic(1.0, n=40000, seed=1)
    plain = fit_platt(positive, negative, 1.0, 1.0)
    weighted = fit_platt(positive, negative, 1.0, 9.0)
    # for a correctly specified model the weights move the intercept by log(1/9); finite-sample
    # estimates of the slope differ slightly
    assert weighted.slope == pytest.approx(plain.slope, abs=0.05)
    assert weighted.intercept - plain.intercept == pytest.approx(np.log(1 / 9), abs=0.05)


def test_prior_shift_matches_refitting_at_the_new_prevalence():
    positive, negative = synthetic(1.0, n=40000, seed=2)
    fitted = fit_platt(positive, negative, 1.0, 9.0)                   # fit prevalence 0.1
    target = fit_platt(positive, negative, 1.0, 99.0)                  # prevalence ~0.01
    p = np.linspace(0.05, 0.95, 7)
    shifted = fitted.apply(p, prevalence=target.fit_prevalence)
    assert np.allclose(shifted, target.apply(p), rtol=0.15)
    assert np.all(np.diff(fitted.apply(p)) > 0)                        # monotone


def test_base_rate_weights():
    assert base_rate_weights(2600, 3_600_000, 520, 520) == (5.0, 3_600_000 / 520)
    with pytest.raises(ValueError):
        base_rate_weights(0, 10, 1, 1)


def test_cross_fitted_quality_beats_the_base_rate_for_an_informative_score():
    positive, negative = synthetic(1.0, n=3000, seed=3)
    scores = cross_fitted_quality(positive, negative, 1.0, 50.0, folds=2, seed=0)
    assert scores["log_loss"] < scores["log_loss_base_rate_only"]
    assert scores["prevalence"] == pytest.approx(len(positive) / (len(positive) + 50 * len(negative)))
    assert scores["ece"] < 0.02
    # uninformative score: no better than the base rate
    rng = np.random.default_rng(4)
    noise = cross_fitted_quality(rng.random(3000), rng.random(3000), 1.0, 50.0, folds=2, seed=0)
    assert noise["log_loss"] >= noise["log_loss_base_rate_only"] - 1e-3


def test_cross_fitting_keeps_samples_of_one_pair_in_one_fold():
    positive = np.tile(np.linspace(0.1, 0.9, 50), (5, 1))          # 5 samples x 50 pairs
    negative = np.tile(np.linspace(0.05, 0.6, 80), (5, 1))
    scores = cross_fitted_quality(positive, negative, 1.0, 1.0, folds=2, seed=1)
    assert np.isfinite(scores["log_loss"])


def test_calibrator_round_trip():
    calibrator = PlattCalibrator(1.5, -2.0, 0.01)
    assert PlattCalibrator(**calibrator.to_dict()) == calibrator
