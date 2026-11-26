"""experiment.output = "probabilities" (advisor meeting 2026-11-26): recovery, analysis, pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from hypstab.config import ConfigError, config_from_dict
from hypstab.data.split import pair_keys
from hypstab.experiment import probabilities as pa
from hypstab.experiment import recovery as recovery_module
from hypstab.experiment.damage import series_from_config
from hypstab.experiment.models import get_bundles
from hypstab.experiment.pipeline import make_split, run_experiment
from hypstab.experiment.recovery import run_attack

from .test_autoencoder import sbm_graph

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs" / "hypstab"


@pytest.fixture()
def raw() -> dict:
    with (CONFIGS / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


def small_config(raw: dict, **sections):
    raw = copy.deepcopy(raw)
    raw["run"]["device"] = "cpu"
    raw["split"]["control_pairs"] = 60
    raw["encoder"].update(dim=16, max_epochs=25, patience=25, dropout=0.2)
    raw["community"].update(n_clusters=3)
    raw["denoiser"].update(hidden_dim=16, layers=2, epochs=6)
    raw["recovery"].update(reconstructions=4)
    raw["perturbation"].update(damage_fractions=[0.05, 0.1, 0.2], series=3)
    for section, values in sections.items():
        raw[section].update(values)
    return config_from_dict(raw)


# --------------------------------------------------------------------------- config


def test_main_configs_use_probabilities_one_model_k100():
    for name in ("cora", "cora_structure", "sbm", "ba"):
        with (CONFIGS / f"{name}.toml").open("rb") as handle:
            raw = tomllib.load(handle)
        config = config_from_dict(raw)
        assert config.experiment.output == "probabilities"
        assert tuple(config.experiment.sources) == ("noise",)
        assert config.recovery.reconstructions == 100


def test_output_validated_and_needs_one_model(raw):
    bad = copy.deepcopy(raw)
    bad["experiment"]["output"] = "votes"
    with pytest.raises(ConfigError, match="experiment.output"):
        config_from_dict(bad)
    bad = copy.deepcopy(raw)
    bad["experiment"]["sources"] = ["noise", "ensemble"]
    with pytest.raises(ConfigError, match="one model"):
        config_from_dict(bad)
    bad = copy.deepcopy(raw)
    del bad["experiment"]["output"]
    with pytest.raises(ConfigError):
        config_from_dict(bad)


def test_probability_ceiling():
    assert pa.probability_ceiling(2.0, 1.0) == pytest.approx(0.8807970779778823)


# --------------------------------------------------------------------------- analysis by hand


def _toy_arrays():
    """L=2 levels, S=2 series, K=2, P=3 edges. Edge 0 removed in both series from level 0; edge 1 only in
    series 0 from level 1; edge 2 never."""
    L, S, K, P = 2, 2, 2, 3
    removed = np.zeros((L, S, P), dtype=bool)
    removed[:, :, 0] = True
    removed[1, 0, 1] = True
    prob = np.full((L, S, K, P), np.nan, dtype=np.float32)
    prob[0, :, :, 0] = [[0.8, 0.7], [0.6, 0.5]]
    prob[1, :, :, 0] = [[0.4, 0.3], [0.6, 0.5]]
    prob[1, 0, :, 1] = [0.2, 0.1]
    no_diff = np.full((L, S, P), np.nan, dtype=np.float32)
    no_diff[:, :, 0] = 0.5
    no_diff[1, 0, 1] = 0.3
    return {"removed": removed, "prob_diffusion": prob, "prob_no_diffusion": no_diff}


def test_equal_weight_per_edge():
    arrays = _toy_arrays()
    values, weights = pa.level_probabilities(arrays, 1)
    assert len(values) == 4 + 2                     # edge 0: 2 series x 2 recon; edge 1: 1 x 2
    assert weights.sum() == pytest.approx(2.0)       # two edges, weight 1 each
    edge1 = np.isin(values, np.float32([0.2, 0.1]).astype(np.float64))
    assert weights[edge1].sum() == pytest.approx(1.0)
    share = pa.weighted_histogram(values, weights, np.array([0.0, 0.5, 1.0]))
    # edge 1 (0.2, 0.1) -> low bin with weight 1; edge 0: 0.4, 0.3 low, 0.6, 0.5 high -> 0.5 / 0.5
    np.testing.assert_allclose(share, [1.5 / 2, 0.5 / 2])


def test_lift_by_hand():
    values, weights = pa.level_lift(_toy_arrays(), 1)
    np.testing.assert_allclose(np.sort(values), np.sort([-0.1, -0.2, 0.1, 0.0, -0.1, -0.2]), atol=1e-6)
    assert weights.sum() == pytest.approx(2.0)


def test_trajectories_by_hand():
    t = pa.trajectories(_toy_arrays())
    # only (series, edge) rows whose first removal is before the last level: edge 0 in series 0 and 1
    assert sorted(zip(t["series"].tolist(), t["edge"].tolist())) == [(0, 0), (1, 0)]
    row = int(np.flatnonzero(t["series"] == 0)[0])
    np.testing.assert_allclose(t["paths"][row], [[0.8, 0.4], [0.7, 0.3]], atol=1e-6)
    np.testing.assert_allclose(t["change"][row], [-0.4, -0.4], atol=1e-6)
    assert t["weight"].sum() == pytest.approx(1.0)  # one edge


def test_trajectories_nan_before_first_removal():
    arrays = _toy_arrays()
    arrays["removed"] = np.concatenate([arrays["removed"], arrays["removed"][-1:]])
    arrays["prob_diffusion"] = np.concatenate([arrays["prob_diffusion"], arrays["prob_diffusion"][-1:]])
    t = pa.trajectories(arrays)
    row = int(np.flatnonzero(t["edge"] == 1)[0])
    assert t["first"][row] == 1
    assert np.isnan(t["paths"][row, :, 0]).all() and np.isfinite(t["paths"][row, :, 1:]).all()


def test_trajectories_need_nested_levels():
    arrays = _toy_arrays()
    arrays["removed"][1, 1, 0] = False
    with pytest.raises(ValueError, match="nested"):
        pa.trajectories(arrays)


def test_weighted_quantiles():
    values = np.array([1.0, 2.0, 3.0])
    assert pa.weighted_quantiles(values, np.array([1.0, 1.0, 2.0]), [0.0, 0.5, 1.0]) == [1.0, 2.0, 3.0]
    assert pa.weighted_quantiles(values, np.array([1.0, 0.0, 3.0]), [0.5]) == [3.0]


# --------------------------------------------------------------------------- pipeline


@pytest.fixture(scope="module")
def prob_run(tmp_path_factory):
    with (CONFIGS / "cora.toml").open("rb") as handle:
        config = small_config(tomllib.load(handle))
    graph = sbm_graph()
    run_dir = tmp_path_factory.mktemp("prob_run")
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


def _load(run_dir, attack):
    return dict(np.load(run_dir / "results" / f"noise__{attack}.npz"))


def test_probability_run_outputs(prob_run):
    config, graph, run_dir, summary, _ = prob_run
    assert sorted(p.name for p in (run_dir / "results").iterdir()) == sorted(
        f"noise__{a}.npz" for a in config.perturbation.attacks)
    assert (run_dir / "models" / "member_00" / "done").is_file()
    assert not (run_dir / "models" / "member_01").exists()             # one model only
    K, L = config.recovery.reconstructions, len(config.perturbation.damage_fractions)
    for attack in config.perturbation.attacks:
        arrays = _load(run_dir, attack)
        assert set(arrays) == {"removed", "prob_diffusion", "prob_no_diffusion"}
        removed, prob, no_diff = arrays["removed"], arrays["prob_diffusion"], arrays["prob_no_diffusion"]
        assert prob.shape[:3] == (L, removed.shape[1], K) and prob.dtype == np.float32
        mask = np.broadcast_to(removed[:, :, None, :], prob.shape)
        assert np.isfinite(prob[mask]).all() and np.isnan(prob[~mask]).all()
        assert np.isfinite(no_diff[removed]).all() and np.isnan(no_diff[~removed]).all()
        assert prob[mask].max() <= summary["probability_ceiling"] + 1e-6
        assert prob[mask].min() >= 0.0
    loaded = json.loads((run_dir / "summary.json").read_text())
    assert loaded["output"] == "probabilities" and loaded["K"] == K
    level = loaded["attacks"]["random"]["levels"][1]
    assert sum(level["probabilities"]["histogram_share"]) == pytest.approx(1.0)
    assert "change_first_to_last_level" in loaded["attacks"]["random"]
    figures = {p.name for p in (run_dir / "figures").iterdir()}
    for attack in config.perturbation.attacks:
        assert f"probabilities_{attack}.png" in figures and f"lift_{attack}.png" in figures
    assert "trajectories_random.png" in figures and "per_edge_random.png" in figures
    header = (run_dir / "edges.csv").read_text().splitlines()[0]
    assert "prob_q50" in header and "lift_median" in header and "same_class" in header


def test_probabilities_equal_pairwise_mode_for_the_same_model(prob_run):
    """Same damaged graphs and noise seeds: the probability mode reproduces the pairwise mode's
    diffusion and no_diffusion probabilities exactly (common random numbers)."""
    config, graph, run_dir, _, _ = prob_run
    split = make_split(graph, config)
    series = series_from_config(split, split.all_edges, graph.node_count, config)
    bundle = get_bundles(graph, split, config, 1, run_dir, torch.device("cpu"))[0]    # loads member 0
    K = config.recovery.reconstructions
    pairwise = run_attack(graph, split, series["random"], [bundle] * K, config, torch.device("cpu"))
    arrays = _load(run_dir, "random")
    np.testing.assert_array_equal(pairwise["removed"], arrays["removed"])
    np.testing.assert_allclose(pairwise["prob_diffusion"], arrays["prob_diffusion"], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(pairwise["prob_no_diffusion"][:, :, 0], arrays["prob_no_diffusion"],
                               rtol=1e-6, equal_nan=True)


def test_diffusion_draws_differ_between_reconstructions(prob_run):
    config, _, run_dir, _, _ = prob_run
    arrays = _load(run_dir, "random")
    prob = arrays["prob_diffusion"]
    removed = np.broadcast_to(arrays["removed"][:, :, None, :], prob.shape)
    spread = np.nanmax(prob, axis=2) - np.nanmin(prob, axis=2)
    assert np.nanmax(spread[arrays["removed"]]) > 0


def test_no_removed_edge_in_any_damaged_graph_probabilities(prob_run):
    """Leakage: no damaged graph contains a removed pool edge; train + validation edges always kept."""
    config, graph, run_dir, _, seen = prob_run
    split = np.load(run_dir / "split.npz")
    n = graph.node_count
    pool_keys = pair_keys(split["pool"], n)
    checked = 0
    for attack in config.perturbation.attacks:
        removed = _load(run_dir, attack)["removed"]
        for level in range(removed.shape[0]):
            for s in range(removed.shape[1]):
                removed_keys = pool_keys[removed[level, s]]
                damaged = [e for e in seen if len(e) == len(graph.edges) - len(removed_keys)]
                match = [e for e in damaged if not np.isin(removed_keys, pair_keys(e, n)).any()]
                assert match, f"{attack} level {level} series {s}"
                checked += 1
    assert checked > 0
    for edges in seen:
        assert np.isin(pair_keys(split["train"], n), pair_keys(edges, n)).all()
        assert np.isin(pair_keys(split["val"], n), pair_keys(edges, n)).all()


def test_resume_loads_results(prob_run, capsys):
    config, graph, run_dir, summary, _ = prob_run
    again = run_experiment(config, run_dir, torch.device("cpu"), graph=graph)
    assert "loaded" in capsys.readouterr().out
    assert again["attacks"] == json.loads(json.dumps(summary["attacks"]))
