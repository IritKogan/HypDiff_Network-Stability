"""Tests for Stage 5: damage series, pairwise decisions (ways A and B), p_e, ensemble members, pipeline.

The old stability/ package is used ONLY here, as a reference.
"""

from __future__ import annotations

import copy
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from hypstab.config import ConfigError, config_from_dict
from hypstab.data.split import pair_keys
from hypstab.experiment import recovery as recovery_module
from hypstab.experiment.damage import make_damage_series, series_from_config
from hypstab.experiment.models import member_config, member_seed
from hypstab.experiment.pe import per_edge
from hypstab.experiment.pipeline import make_split, run_experiment
from hypstab.experiment.recovery import (
    METHODS,
    lower_counts,
    pairwise_decision,
    reference_draws,
    way_b_draws,
)

from .test_autoencoder import sbm_graph

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def raw() -> dict:
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


def experiment_config(raw: dict, **sections):
    raw = copy.deepcopy(raw)
    raw["run"]["device"] = "cpu"
    raw["split"]["control_pairs"] = 60
    raw["encoder"].update(dim=16, max_epochs=25, patience=25, dropout=0.2)
    raw["community"].update(n_clusters=3)
    raw["denoiser"].update(hidden_dim=16, layers=2, epochs=6)
    raw["recovery"].update(reconstructions=3)
    raw["perturbation"].update(damage_fractions=[0.05, 0.1, 0.2], series=2)
    raw["experiment"].update(way_b_references=4, output="pairwise", sources=["noise", "ensemble"])
    for section, values in sections.items():
        raw[section].update(values)
    return config_from_dict(raw)


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("perturbation", "damage_fractions", [0.2, 0.1], "increasing"),
        ("perturbation", "damage_fractions", [0.1, 0.4], "pool_fraction"),
        ("perturbation", "attacks", ["random", "cascade"], "perturbation.attacks"),
        ("perturbation", "attacks", ["random", "random"], "repeats"),
        ("perturbation", "series", 0, "series"),
        ("experiment", "sources", ["noise", "bootstrap"], "experiment.sources"),
        ("experiment", "sources", [], "non-empty"),
        ("experiment", "way_b_references", 0, "way_b_references"),
    ],
)
def test_bad_experiment_values_rejected(raw, section, key, value, message):
    raw[section][key] = value
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


def test_cora_experiment_values_match_old_config(raw):
    with (REPO_ROOT / "configs" / "cora.toml").open("rb") as handle:
        old = tomllib.load(handle)["perturbation"]
    new = raw["perturbation"]
    for key in ("seed", "damage_fractions", "series", "attacks"):
        assert new[key] == old[key]


# --------------------------------------------------------------------------- damage series


def test_damage_series_identical_to_old_code():
    from stability.perturbation_masks import make_damage_series as old_series

    graph = sbm_graph(seed=3, sizes=(40, 40, 40), p_in=0.2)
    rng = np.random.default_rng(0)
    pool = graph.edges[rng.permutation(len(graph.edges))[: len(graph.edges) // 2]]
    args = (pool, graph.edges, graph.node_count, (0.1, 0.2, 0.3),
            ("random", "targeted_weighted", "targeted_deterministic"), 4, 20261119)
    new, old = make_damage_series(*args), old_series(*args)
    assert new.keys() == old.keys()
    for attack in new:
        assert len(new[attack]) == len(old[attack])
        for a, b in zip(new[attack], old[attack]):
            np.testing.assert_array_equal(a.order, b.order)
            assert a.removal_counts == b.removal_counts
            for level in range(3):
                np.testing.assert_array_equal(a.removed_pool_indices(level), b.removed_pool_indices(level))


def test_series_are_nested_and_pool_only(raw):
    config = experiment_config(raw)
    graph = sbm_graph()
    split = make_split(graph, config)
    series = series_from_config(split, split.all_edges, graph.node_count, config)
    assert len(series["targeted_deterministic"]) == 1
    assert len(series["random"]) == config.perturbation.series
    for items in series.values():
        for item in items:
            sets = [set(item.removed_pool_indices(l).tolist()) for l in range(3)]
            assert sets[0] <= sets[1] <= sets[2]
            assert all(0 <= i < len(split.pool) for i in sets[2])


# --------------------------------------------------------------------------- decisions


def test_reference_draws_match_old_code():
    """Same SeedSequence and the same 'skip itself' rule as stability/edge_recovery_experiment.py."""
    seed, method, s, level, k, P, C = 20261120, "clamp_only", 3, 1, 7, 50, 40
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11, 2, s, level, k]))
    old_pool = rng.integers(0, C, size=P)
    old_control = rng.integers(0, C - 1, size=C)
    old_control = old_control + (old_control >= np.arange(C))
    new_pool, new_control = reference_draws(seed, method, s, level, k, P, C)
    np.testing.assert_array_equal(new_pool, old_pool)
    np.testing.assert_array_equal(new_control, old_control)
    assert np.all(new_control != np.arange(C))
    _, way_b_control = way_b_draws(seed, method, s, level, 5, P, C)
    assert np.all(way_b_control != np.arange(C)[None, :])


def test_pairwise_decision_and_shares():
    control = np.array([0.0, 1.0, 2.0, 3.0])
    removed = np.array([-1.0, 1.5, 9.0])
    hit, control_hit, share, control_share = pairwise_decision(
        removed, control, np.array([0, 0, 3]), np.array([1, 0, 3, 2]))
    np.testing.assert_array_equal(hit, [False, True, True])
    np.testing.assert_array_equal(control_hit, [False, True, False, True])
    np.testing.assert_allclose(share, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(control_share, [0.0, 1 / 3, 2 / 3, 1.0])
    assert lower_counts(np.sort(control), np.array([1.0])) == 1   # strictly lower


def test_way_b_with_one_reconstruction_orders_like_logits():
    """For K = 1 the averaged score F = (controls strictly lower) / C orders pairs like the logits."""
    rng = np.random.default_rng(1)
    control = rng.normal(size=200)
    removed = rng.normal(0.5, 1, size=80)
    ordered = np.sort(control)
    f_removed = lower_counts(ordered, removed) / len(control)
    f_control = lower_counts(ordered, control) / len(control)
    refs = rng.integers(0, len(control), size=80)
    np.testing.assert_array_equal(f_removed > f_control[refs], removed > control[refs])


def test_per_edge_counts_by_hand():
    L, S, K, P, C = 1, 2, 2, 3, 4
    removed = np.zeros((L, S, P), dtype=bool)
    removed[0, 0, [0, 1]] = True
    removed[0, 1, [1]] = True
    hit = np.zeros((L, S, K, P), dtype=bool)
    hit[0, 0, :, 0] = [True, False]      # edge 0: 1 of 2
    hit[0, 0, :, 1] = [True, True]       # edge 1: 2 + 1 of 4
    hit[0, 1, :, 1] = [False, True]
    nan = np.full((L, S, K, P), np.nan, dtype=np.float32)
    share = nan.copy()
    share[removed[:, :, None, :].repeat(K, 2)] = 0.5
    arrays = {"removed": removed, "hit_diffusion": hit, "share_diffusion": share, "prob_diffusion": share,
              "wayb_hits_diffusion": np.array([[[3, 2, 0], [0, 4, 0]]], dtype=np.int16),
              "wayb_share_diffusion": np.where(removed, 0.25, np.nan),
              "control_hits_diffusion": np.full((L, S, C), 1, dtype=np.int16),
              "control_share_sum_diffusion": np.ones((L, S, C), dtype=np.float32),
              "control_wayb_hits_diffusion": np.full((L, S, C), 2, dtype=np.int16),
              "control_prob_k_sum_diffusion": np.ones((L, K, C), dtype=np.float32)}
    q = per_edge(arrays, "diffusion", references=4)
    np.testing.assert_array_equal(q["n_e"][0], [1, 2, 0])
    np.testing.assert_allclose(q["p_e"][0][:2], [0.5, 0.75])
    assert np.isnan(q["p_e"][0, 2])
    np.testing.assert_allclose(q["p_e_way_b"][0][:2], [3 / 4, 6 / 8])
    np.testing.assert_allclose(q["expected_p_e"][0][:2], [0.5, 0.5])
    np.testing.assert_allclose(q["control"][0], 0.5)
    np.testing.assert_allclose(q["control_way_b"][0], 0.5)
    hit[0, 1, 0, 0] = True               # reappears where it was not removed -> error
    with pytest.raises(ValueError, match="not removed"):
        per_edge(arrays, "diffusion", references=4)


# --------------------------------------------------------------------------- ensemble members


def test_member_config_changes_only_model_seeds(raw):
    config = experiment_config(raw)
    assert member_config(config, 0) is config
    one, two = member_config(config, 1), member_config(config, 2)
    for cfg in (one, two):
        assert cfg.encoder.seed != config.encoder.seed
        assert cfg.community.seed != config.community.seed
        assert cfg.denoiser.seed != config.denoiser.seed
        for section in ("split", "perturbation", "recovery", "diffusion", "experiment"):
            assert getattr(cfg, section) == getattr(config, section)
    assert one.encoder.seed != two.encoder.seed
    assert member_seed(5, 3) == member_seed(5, 3)


# --------------------------------------------------------------------------- pipeline


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """One tiny run (both sources), with the edges of every damaged graph recorded."""
    with (REPO_ROOT / "configs" / "hypstab" / "cora.toml").open("rb") as handle:
        config = experiment_config(tomllib.load(handle))
    graph = sbm_graph()
    run_dir = tmp_path_factory.mktemp("run")
    seen = []
    original = recovery_module.normalized_adjacency

    def recording(node_count, edges, device):
        seen.append(np.asarray(edges).copy())
        return original(node_count, edges, device)

    recovery_module.normalized_adjacency = recording
    try:
        summary = run_experiment(config, run_dir, torch.device("cpu"), graph=graph)
    finally:
        recovery_module.normalized_adjacency = original
    return config, graph, run_dir, summary, seen


def _load(run_dir, source, attack):
    return dict(np.load(run_dir / "results" / f"{source}__{attack}.npz"))


def test_pipeline_outputs(pipeline_run):
    config, graph, run_dir, summary, _ = pipeline_run
    for name in ("split.npz", "series.npz", "models.json", "summary.json", "edges.csv"):
        assert (run_dir / name).is_file()
    for member in range(config.recovery.reconstructions):
        assert (run_dir / "models" / f"member_{member:02d}" / "done").is_file()
    assert len(list((run_dir / "figures").glob("ecdf_*.png"))) == 6
    rows = summary["headline"]
    assert len(rows) == 2 * 3 * 3 * 3            # sources x attacks x levels x methods
    for row in rows:
        assert 0 <= row["mean_p_e_way_a"] <= 1 and 0 <= row["mean_p_e_way_b"] <= 1
        assert abs(row["control_way_a"] - 0.5) < 0.2 and abs(row["control_way_b"] - 0.5) < 0.2


def test_no_removed_edge_in_any_damaged_graph(pipeline_run):
    """Leakage: no damaged graph contains a removed pool edge; removed edges are pool edges only."""
    config, graph, run_dir, _, seen = pipeline_run
    split = np.load(run_dir / "split.npz")
    n = graph.node_count
    pool_keys = pair_keys(split["pool"], n)
    checked = 0
    for source in config.experiment.sources:
        for attack in config.perturbation.attacks:
            removed = _load(run_dir, source, attack)["removed"]
            assert removed.shape[-1] == len(split["pool"])
            for level in range(removed.shape[0]):
                for s in range(removed.shape[1]):
                    removed_keys = pool_keys[removed[level, s]]
                    damaged = [e for e in seen if len(e) == len(graph.edges) - len(removed_keys)]
                    assert damaged
                    match = [e for e in damaged if not np.isin(removed_keys, pair_keys(e, n)).any()]
                    assert match, f"{source}/{attack} level {level} series {s}"
                    checked += 1
    assert checked > 0
    for edges in seen:     # every damaged graph keeps all train + validation edges
        assert np.isin(pair_keys(split["train"], n), pair_keys(edges, n)).all()


def test_sources_are_paired(pipeline_run):
    """Reconstruction 0 of the ensemble = member 0 with noise draw 0 = reconstruction 0 of the noise source."""
    config, _, run_dir, _, _ = pipeline_run
    for attack in config.perturbation.attacks:
        noise, ensemble = _load(run_dir, "noise", attack), _load(run_dir, "ensemble", attack)
        np.testing.assert_array_equal(noise["removed"], ensemble["removed"])
        for m in METHODS:
            np.testing.assert_array_equal(noise[f"hit_{m}"][:, :, 0], ensemble[f"hit_{m}"][:, :, 0])
            np.testing.assert_allclose(noise[f"prob_{m}"][:, :, 0], ensemble[f"prob_{m}"][:, :, 0])


def test_deterministic_methods_one_value_for_one_model(pipeline_run):
    """noise source: clamp_only / no_diffusion give the same scores in every reconstruction (only the
    reference draws differ); the diffusion differs between noise draws."""
    _, _, run_dir, _, _ = pipeline_run
    arrays = _load(run_dir, "noise", "random")
    for m in ("clamp_only", "no_diffusion"):
        share = arrays[f"share_{m}"]
        np.testing.assert_array_equal(np.isnan(share[:, :, 0]), np.isnan(share[:, :, 1]))
        np.testing.assert_allclose(share[:, :, 0], share[:, :, 1])
    prob = arrays["prob_diffusion"]
    assert not np.allclose(np.nan_to_num(prob[:, :, 0]), np.nan_to_num(prob[:, :, 1]))


def test_resume_loads_and_reproduces(pipeline_run, capsys):
    config, graph, run_dir, summary, _ = pipeline_run
    again = run_experiment(config, run_dir, torch.device("cpu"), graph=graph)
    out = capsys.readouterr().out
    assert "member 2: loaded" in out and "ensemble / random: loaded" in out and "training" not in out
    assert again["headline"] == summary["headline"]


def test_resume_refuses_a_different_split(pipeline_run, raw):
    config, graph, run_dir, _, _ = pipeline_run
    other = experiment_config(raw, split={"seed": 1})
    with pytest.raises(RuntimeError, match="differs"):
        run_experiment(other, run_dir, torch.device("cpu"), graph=graph)
