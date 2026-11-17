import numpy as np
import pytest

from stability.iterative_reconstruction import candidate_pairs, gumbel_top_k


def test_candidate_pairs_are_all_non_edges():
    edges = np.array([[0, 1], [2, 1], [3, 0]])
    pairs = candidate_pairs(4, edges)
    assert len(pairs) == 4 * 3 // 2 - 3
    present = {tuple(sorted(e)) for e in edges.tolist()}
    assert all(tuple(p) not in present and p[0] < p[1] for p in pairs.tolist())


def test_gumbel_top_k_zero_temperature_is_the_top_k():
    p = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    chosen = gumbel_top_k(p, 2, 0.0, np.random.default_rng(0))
    assert set(chosen.tolist()) == {1, 3}


def test_gumbel_top_k_samples_proportionally_to_odds():
    # one draw (count 1) picks item i with probability odds_i^(1/T) / sum_j odds_j^(1/T)
    p = np.array([0.5, 0.8, 0.2])
    odds = p / (1 - p)
    expected = odds / odds.sum()
    rng = np.random.default_rng(1)
    picks = np.array([gumbel_top_k(p, 1, 1.0, rng)[0] for _ in range(20000)])
    observed = np.bincount(picks, minlength=3) / len(picks)
    assert np.allclose(observed, expected, atol=0.015)
    chosen = gumbel_top_k(p, 3, 1.0, rng)
    assert sorted(chosen.tolist()) == [0, 1, 2]           # without replacement


def test_gumbel_top_k_validates_arguments():
    with pytest.raises(ValueError):
        gumbel_top_k(np.array([0.5]), 2, 1.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        gumbel_top_k(np.array([0.5]), 1, -1.0, np.random.default_rng(0))
    assert gumbel_top_k(np.array([0.5]), 0, 1.0, np.random.default_rng(0)).size == 0
