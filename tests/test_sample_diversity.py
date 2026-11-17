import numpy as np
import pytest

from stability.sample_diversity import (
    frequency_profile,
    probability_diversity,
    recovery_frequency,
    tangent_spread,
)


def test_tangent_spread_zero_for_identical_samples():
    sample = np.random.default_rng(0).normal(size=(5, 3))
    assert tangent_spread(np.stack([sample] * 4)) == 0.0


def test_tangent_spread_matches_known_std():
    samples = np.zeros((2, 4, 3))
    samples[1] += 2.0  # two samples differing by 2 -> population std 1
    assert tangent_spread(samples) == pytest.approx(1.0)


def test_tangent_spread_rejects_single_sample():
    with pytest.raises(ValueError):
        tangent_spread(np.zeros((1, 2, 2)))


def test_recovery_frequency_and_profile():
    probabilities = np.array([[0.9, 0.1, 0.6], [0.8, 0.2, 0.4]])
    frequency = recovery_frequency(probabilities, tau=0.5)
    np.testing.assert_allclose(frequency, [1.0, 0.0, 0.5])
    profile = frequency_profile(frequency)
    assert profile["share_one"] == pytest.approx(1 / 3)
    assert profile["share_zero"] == pytest.approx(1 / 3)
    assert profile["share_intermediate"] == pytest.approx(1 / 3)
    assert profile["mean_binomial_variance"] == pytest.approx(0.25 / 3)


def test_probability_diversity_keys():
    rng = np.random.default_rng(1)
    result = probability_diversity(rng.random((3, 10)), rng.random((3, 20)), tau=0.5)
    assert set(result) == {"probability_std_positive", "probability_std_negative", "positive", "negative"}


def test_stratified_auc_is_half_when_score_is_a_function_of_strata():
    from stability.sample_diversity import stratified_auc

    rng = np.random.default_rng(2)
    base = rng.random(2000)
    labels = (rng.random(2000) < base).astype(int)
    # score adds nothing beyond base inside narrow strata -> close to 0.5
    noise_score = rng.random(2000)
    assert abs(stratified_auc(noise_score, labels, base, n_strata=10) - 0.5) < 0.05
    # score that knows the label -> high even within strata
    assert stratified_auc(labels + 0.01 * rng.random(2000), labels, base) > 0.95


def test_incremental_log_likelihood_detects_extra_information():
    from stability.sample_diversity import incremental_log_likelihood

    rng = np.random.default_rng(3)
    n = 3000
    latent = rng.normal(size=n)
    hidden = rng.normal(size=n)
    labels = (latent + hidden + 0.3 * rng.normal(size=n) > 0).astype(int)
    base = 1 / (1 + np.exp(-latent))
    useful = incremental_log_likelihood(base, hidden, labels)
    useless = incremental_log_likelihood(base, rng.normal(size=n), labels)
    assert useful["gain_per_pair"] > 0.1
    assert abs(useless["gain_per_pair"]) < 0.01


def test_calibration_table_is_zero_for_a_calibrated_score_and_uses_weights():
    from stability.sample_diversity import calibration_table

    rng = np.random.default_rng(4)
    score = rng.random(200000)
    labels = (rng.random(200000) < score).astype(int)
    assert calibration_table(score, labels)["ece"] < 0.01
    # constant score 0.5 on a balanced set is calibrated; with negatives weighted x9 it is not
    labels = np.r_[np.ones(100), np.zeros(100)]
    score = np.full(200, 0.5)
    assert calibration_table(score, labels)["ece"] == pytest.approx(0.0)
    weighted = calibration_table(score, labels, np.r_[np.ones(100), 9 * np.ones(100)])
    assert weighted["ece"] == pytest.approx(0.4)
    assert weighted["bins"][0]["observed"] == pytest.approx(0.1)
