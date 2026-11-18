import numpy as np
import pytest

from stability.bootstrap_ensemble import (
    MemberSplit, member_seed, member_split, pairwise_outcomes, rank_against,
)
from stability.edge_split import pair_keys, split_edges


def _split():
    rng = np.random.default_rng(0)
    n = 60
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    edges = pairs[rng.choice(len(pairs), size=300, replace=False)]
    return split_edges(edges, n, 0.5, 0.1, 0.4, 100, seed=1)


def test_seed_only_member0_keeps_config_seed_and_all_train_edges():
    split = _split()
    assert member_seed(123, "seed_only", 0) == 123
    assert member_seed(123, "seed_only", 1) != 123
    assert member_seed(123, "subsample", 0) != 123
    member = member_split(split, "seed_only", 3, 0.8, 7)
    np.testing.assert_array_equal(member.train, split.train)
    assert len(member.held_out_train) == 0


@pytest.mark.parametrize("mode", ["subsample", "bootstrap"])
def test_resampled_train_is_subset_and_partition(mode):
    split = _split()
    member = member_split(split, mode, 2, 0.8, 7)
    train_keys = set(pair_keys(split.train, split.node_count).tolist())
    used = set(pair_keys(member.train, split.node_count).tolist())
    held = set(pair_keys(member.held_out_train, split.node_count).tolist())
    assert used | held == train_keys and not (used & held)
    if mode == "subsample":
        assert len(used) == round(0.8 * len(split.train))
    else:
        assert 0.5 * len(split.train) < len(used) < 0.75 * len(split.train)
    # validation / pool / control untouched
    np.testing.assert_array_equal(member.val, split.val)
    np.testing.assert_array_equal(member.pool, split.pool)
    np.testing.assert_array_equal(member.control_pairs, split.control_pairs)
    # members differ, same member is reproducible
    again = member_split(split, mode, 2, 0.8, 7)
    other = member_split(split, mode, 3, 0.8, 7)
    np.testing.assert_array_equal(again.train, member.train)
    assert set(pair_keys(other.train, split.node_count).tolist()) != used


def test_held_out_train_edges_stay_forbidden_as_negatives():
    split = _split()
    member = member_split(split, "subsample", 0, 0.5, 7)
    assert isinstance(member, MemberSplit)
    forbidden = set(member.forbidden_negative_keys().tolist())
    for group in (split.train, split.val, split.pool, split.val_negatives, split.control_pairs):
        assert set(pair_keys(group, split.node_count).tolist()) <= forbidden


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        member_split(_split(), "jackknife", 0, 0.8, 7)
    with pytest.raises(ValueError):
        member_seed(1, "jackknife", 0)


def test_rank_against_counts_strictly_lower():
    np.testing.assert_allclose(rank_against(np.array([0.0, 0.25, 0.5, 2.0]), np.array([0.1, 0.25, 0.3, 0.9])),
                               [0.0, 0.25, 0.75, 1.0])


def test_pairwise_outcomes_never_uses_self_and_control_near_half():
    rng = np.random.default_rng(0)
    labels = np.r_[np.ones(50), np.zeros(400)].astype(int)
    scores = np.c_[rng.normal(2, 1, size=(20, 50)), rng.normal(0, 1, size=(20, 400))]
    outcomes = pairwise_outcomes(scores, labels, np.random.default_rng(1))
    assert outcomes.shape == scores.shape
    assert abs(outcomes[:, labels == 0].mean() - 0.5) < 0.03
    assert outcomes[:, labels == 1].mean() > 0.85
    # two negatives only: each must be compared with the other one
    out = pairwise_outcomes(np.array([[1.0, 0.0, 0.5]]), np.array([1, 0, 0]), np.random.default_rng(2))
    np.testing.assert_array_equal(out[0, 1:], [0, 1])
