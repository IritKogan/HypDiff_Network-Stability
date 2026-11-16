"""Tests for the leakage-free, config-driven edge-recovery pipeline."""

import json
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from stability.edge_recovery_distribution import control_probability, per_edge_probability
from stability.edge_recovery_experiment import recovery_seed, run_experiment
from stability.edge_split import pair_keys, sample_non_edges, split_edges
from stability.experiment_config import config_from_dict, load_config
from stability.graph_data import GraphData
from stability.model_training import normalized_adjacency, train_models
from stability.perturbation_masks import degree_products, make_damage_series, removal_count
from stability.wp7_training import make_adjacency

ROOT = Path(__file__).resolve().parents[1]
CORA_CONFIG = ROOT / "configs" / "cora.toml"


def raw_config():
    with CORA_CONFIG.open("rb") as handle:
        return tomllib.load(handle)


def tiny_config(**changes):
    raw = raw_config()
    raw["split"]["control_pairs"] = 40
    raw["model"].update(dim=4, dropout=0.0)
    raw["autoencoder"].update(max_epochs=15, patience=5)
    raw["denoiser"].update(epochs=5)
    raw["community"].update(n_clusters=3, max_epochs=3)
    raw["perturbation"].update(series=3)
    raw["recovery"].update(reconstructions=2)
    raw["run"]["device"] = "cpu"
    for section, values in changes.items():
        raw[section].update(values)
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def graph():
    """Three-block SBM with class labels = blocks."""
    rng = np.random.default_rng(5)
    blocks = np.repeat(np.arange(3), 20)
    edges = []
    for i in range(60):
        for j in range(i + 1, 60):
            p = 0.3 if blocks[i] == blocks[j] else 0.03
            if rng.random() < p:
                edges.append((i, j))
    features = np.eye(3)[blocks] + 0.1 * rng.standard_normal((60, 3))
    features = np.hstack([features, rng.random((60, 5))]).astype(np.float32)
    return GraphData(
        name="SBM-test",
        features=torch.tensor(features),
        edges=np.asarray(edges, dtype=np.int64),
        class_labels=blocks,
    )


# ---------------------------------------------------------------- config


def test_cora_config_loads_with_approved_defaults():
    config = load_config(CORA_CONFIG)
    assert (config.split.train_fraction, config.split.val_fraction, config.split.pool_fraction) == (0.5, 0.1, 0.4)
    assert config.perturbation.damage_fractions == (0.1, 0.2, 0.3)
    assert config.perturbation.series == 20
    assert config.recovery.reconstructions == 10
    assert config.recovery.t_star == 7
    assert config.community.n_clusters == 7
    assert config.model.dim == 128 and config.model.dropout == 0.7
    assert config.denoiser.epochs == 5000
    assert config.autoencoder.max_epochs == 2000
    assert json.loads(json.dumps(config.to_dict()))["perturbation"]["series"] == 20


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda raw: raw["recovery"].pop("tau"), "missing keys"),
        (lambda raw: raw["recovery"].update(taux=0.5), "unknown keys"),
        (lambda raw: raw.update(extra={}), "unknown"),
        (lambda raw: raw["recovery"].update(reconstructions="10"), "expected int"),
        (lambda raw: raw["perturbation"].update(damage_fractions=[0.1, 0.5]), "pool_fraction"),
        (lambda raw: raw["perturbation"].update(damage_fractions=[0.2, 0.1]), "increasing"),
        (lambda raw: raw["split"].update(pool_fraction=0.5), "sum to 1"),
        (lambda raw: raw["perturbation"].update(attacks=["degree"]), "attacks"),
        (lambda raw: raw["recovery"].update(t_star=200), "t_star"),
        (lambda raw: raw["denoiser"].update(noise_formula="ddpm"), "noise_formula"),
        (lambda raw: raw["denoiser"].update(architecture="unet"), "architecture"),
        (lambda raw: raw["denoiser"].update(training="supervised"), "training"),
        (lambda raw: raw["denoiser"].update(train_t_max=0), "train_t_max"),
        (lambda raw: raw["recovery"].update(tau_mode="auto"), "tau_mode"),
        (lambda raw: raw["recovery"].update(target_fpr=0.0), "target_fpr"),
    ],
)
def test_config_rejects_invalid_files(mutate, message):
    raw = raw_config()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        config_from_dict(raw)


def test_config_overrides_are_validated():
    config = load_config(CORA_CONFIG).with_overrides(run={"device": "cpu", "output_root": "/tmp/x"})
    assert config.run.device == "cpu" and config.run.output_root == "/tmp/x"
    with pytest.raises(ValueError, match="unknown keys"):
        config.with_overrides(run={"devise": "cpu"})


# ---------------------------------------------------------------- split


def test_split_partitions_edges_and_keeps_negatives_disjoint(graph):
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    n = graph.node_count
    groups = [pair_keys(g, n) for g in (split.train, split.val, split.pool)]
    assert sum(len(g) for g in groups) == len(graph.edges)
    assert len(np.unique(np.concatenate(groups))) == len(graph.edges)
    assert set(np.concatenate(groups)) == set(pair_keys(graph.edges, n))
    assert len(split.pool) == round(0.4 * len(graph.edges))

    edge_keys = set(pair_keys(graph.edges, n))
    val_neg = set(pair_keys(split.val_negatives, n))
    control = set(pair_keys(split.control_pairs, n))
    assert not (val_neg & edge_keys) and not (control & edge_keys) and not (val_neg & control)
    assert len(control) == 40 and len(val_neg) == len(split.val)

    again = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    assert np.array_equal(again.pool, split.pool) and np.array_equal(again.control_pairs, split.control_pairs)


def test_sample_non_edges_never_returns_forbidden_pairs():
    rng = np.random.default_rng(0)
    forbidden = np.arange(0, 40 * 40, 3)
    pairs = sample_non_edges(200, 40, forbidden, rng)
    keys = pair_keys(pairs, 40)
    assert len(np.unique(keys)) == 200
    assert not np.isin(keys, forbidden).any()
    assert np.all(pairs[:, 0] < pairs[:, 1])


def test_normalized_adjacency_matches_wp7_make_adjacency(graph):
    expected = make_adjacency(graph.node_count, tuple(map(tuple, graph.edges.tolist())))
    actual = normalized_adjacency(graph.node_count, graph.edges, "cpu")
    assert torch.equal(actual.indices(), expected.indices())
    torch.testing.assert_close(actual.values(), expected.values(), rtol=0, atol=0)


# ---------------------------------------------------------------- damage series


def test_damage_series_are_cumulative_and_drawn_from_pool(graph):
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    fractions = (0.1, 0.2, 0.3)
    series = make_damage_series(
        split.pool, split.all_edges, graph.node_count, fractions,
        ("random", "targeted_weighted", "targeted_deterministic"), series=4, seed=9,
    )
    counts = [removal_count(len(graph.edges), f) for f in fractions]
    assert len(series["targeted_deterministic"]) == 1
    for attack, items in series.items():
        for item in items:
            assert sorted(item.order.tolist()) == list(range(len(split.pool)))
            removed = [set(item.removed_pool_indices(level).tolist()) for level in range(3)]
            assert [len(r) for r in removed] == counts
            assert removed[0] <= removed[1] <= removed[2]
    random_orders = {tuple(item.order[: counts[0]]) for item in series["random"]}
    assert len(random_orders) == 4


def test_targeted_attacks_prefer_high_degree_products(graph):
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    weights = degree_products(split.all_edges, graph.node_count, split.pool)
    series = make_damage_series(
        split.pool, split.all_edges, graph.node_count, (0.1,),
        ("random", "targeted_weighted", "targeted_deterministic"), series=50, seed=9,
    )
    count = series["random"][0].removal_counts[0]

    def mean_weight(attack):
        return np.mean([weights[item.order[:count]].mean() for item in series[attack]])

    deterministic = weights[series["targeted_deterministic"][0].order[:count]]
    assert np.all(deterministic >= np.sort(weights)[-count])
    assert mean_weight("targeted_weighted") > mean_weight("random")


# ---------------------------------------------------------------- leakage


def test_training_never_sees_pool_edges_or_control_pairs(graph):
    config = tiny_config()
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    models = train_models(graph, split, config, torch.device("cpu"), audit=True)
    n = graph.node_count
    train_keys = np.unique(pair_keys(split.train, n))
    held_out = np.concatenate([
        pair_keys(split.pool, n), pair_keys(split.val, n),
        pair_keys(split.control_pairs, n), pair_keys(split.val_negatives, n),
    ])
    assert np.array_equal(models.audit["adjacency_keys"], train_keys)
    assert np.array_equal(models.audit["positive_keys"], train_keys)
    assert not np.isin(models.audit["negative_keys"], held_out).any()
    assert models.context.labels.shape[0] == n
    assert models.history["autoencoder_epochs_run"] <= config.autoencoder.max_epochs


def test_training_does_not_read_pool_edges(graph, monkeypatch):
    """Train once normally, once with the pool emptied out of the split. The only
    remaining use of pool edges is "do not sample them as negatives" (they are
    true edges); with that forbidden set held fixed, the models must be identical,
    i.e. no pool edge reaches training in any other way."""
    from dataclasses import replace

    from stability.edge_split import EdgeSplit

    config = tiny_config()
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    forbidden = split.forbidden_negative_keys()
    with_pool = train_models(graph, split, config, torch.device("cpu"))

    monkeypatch.setattr(EdgeSplit, "forbidden_negative_keys", lambda self: forbidden)
    without_pool = train_models(graph, replace(split, pool=split.pool[:0]), config, torch.device("cpu"))

    for model in ("encoder", "denoiser"):
        reference = getattr(with_pool, model).state_dict()
        for name, tensor in getattr(without_pool, model).state_dict().items():
            assert torch.equal(tensor, reference[name]), (model, name)
    assert torch.equal(with_pool.context.labels, without_pool.context.labels)
    assert torch.equal(with_pool.context.centers, without_pool.context.centers)
    assert with_pool.history["autoencoder_val_auc"] == without_pool.history["autoencoder_val_auc"]


# ---------------------------------------------------------------- p_e


def test_per_edge_probability_counts_only_series_where_edge_was_removed():
    removed = np.array([[[True, False, True], [True, True, False]]])     # L=1, S=2, P=3
    recovered = np.array([[[2, 0, 0], [1, 2, 0]]], dtype=np.int16)        # K=2
    n_e, r_e, p_e = per_edge_probability(removed, recovered, reconstructions=2)
    assert n_e.tolist() == [[2, 1, 1]]
    assert r_e.tolist() == [[3, 2, 0]]
    assert p_e.tolist() == [[0.75, 1.0, 0.0]]
    with pytest.raises(ValueError, match="not removed"):
        per_edge_probability(removed, np.array([[[0, 1, 0], [0, 0, 0]]]), 2)
    hits = np.array([[[1, 0], [2, 0]]])
    assert control_probability(hits, 2).tolist() == [[0.75, 0.0]]


def test_recovery_seeds_are_distinct_and_level_independent():
    seeds = {recovery_seed(1, s, k) for s in range(20) for k in range(10)}
    assert len(seeds) == 200
    assert recovery_seed(1, 3, 4) == recovery_seed(1, 3, 4)


# ---------------------------------------------------------------- end to end


def test_end_to_end_run_writes_reproducible_results(graph, tmp_path):
    config = tiny_config()
    first = run_experiment(config, config_text="# test", output_dir=tmp_path / "a", graph=graph, make_plots=True)
    second = run_experiment(config, output_dir=tmp_path / "b", graph=graph, make_plots=False)

    for name in ("config.json", "config.toml", "environment.json", "split.npz", "series.npz",
                 "training_history.json", "recovery_counts.npz", "edge_recovery.csv",
                 "control_recovery.csv", "summary.json", "figures/p_e_distributions.png",
                 "checkpoints/encoder.pt", "checkpoints/denoiser.pt"):
        assert (first / name).is_file(), name

    saved = json.loads((first / "config.json").read_text())
    assert saved == config.to_dict()
    environment = json.loads((first / "environment.json").read_text())
    assert environment["packages"]["torch"] == torch.__version__

    counts_a = np.load(first / "recovery_counts.npz")
    counts_b = np.load(second / "recovery_counts.npz")
    assert sorted(counts_a.files) == sorted(counts_b.files)
    for key in counts_a.files:
        assert np.array_equal(counts_a[key], counts_b[key]), key

    summary = json.loads((first / "summary.json").read_text())
    pool_size = summary["split_sizes"]["pool"]
    for attack in config.perturbation.attacks:
        removed = counts_a[f"{attack}__removed"]
        recovered = counts_a[f"{attack}__recovered"]
        assert removed.shape[2] == pool_size
        assert np.all(recovered <= config.recovery.reconstructions)
        assert np.all(removed[:-1] <= removed[1:])          # cumulative within series
        levels = summary["attacks"][attack]
        assert [entry["damage_fraction"] for entry in levels] == list(config.perturbation.damage_fractions)
        for level, entry in enumerate(levels):
            assert entry["edges_removed_per_series"] == removed[level, 0].sum()
            assert entry["pool_edges_evaluated"] == int((removed[level].sum(axis=0) > 0).sum())
            assert "p_e_same_class" in entry and "control_false_positive" in entry
            assert 0.0 <= entry["auc_removed_vs_control"] <= 1.0
    assert set(summary["thresholds"]) == {"diffusion", "no_diffusion", "clamp_only"}
    assert (first / "validation_recovery.json").is_file()


# ---------------------------------------------------------------- denoiser options


def test_residual_denoiser_is_identity_at_initialisation():
    from stability.denoisers import ResidualGCNDenoiser

    torch.manual_seed(0)
    network = ResidualGCNDenoiser(dim=4, hidden_dim=8, layers=2)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert torch.equal(network(x, edge_index, torch.full((5,), 0.1)), x)


def test_default_denoiser_options_reproduce_upstream_architecture(graph):
    from models.direction_diffusion import gcndec
    from stability.model_training import build_model_args

    config = tiny_config()
    args, _ = build_model_args(graph, config, torch.device("cpu"))
    from stability.hypdiff_models import StabilityDiffusion

    denoiser = StabilityDiffusion(args, dim=args.dim, device="cpu")
    assert isinstance(denoiser.gcn, gcndec) and denoiser.clip_x0 is True
    residual = tiny_config(denoiser={"architecture": "residual_gcn", "clip_x0": False})
    args, _ = build_model_args(graph, residual, torch.device("cpu"))
    denoiser = StabilityDiffusion(args, dim=args.dim, device="cpu")
    assert type(denoiser.gcn).__name__ == "ResidualGCNDenoiser" and denoiser.clip_x0 is False


@pytest.mark.parametrize("training", ["generative", "damage_aware"])
def test_denoiser_training_does_not_read_pool_edges(graph, monkeypatch, training):
    from dataclasses import replace

    from stability.edge_split import EdgeSplit

    config = tiny_config(denoiser={"training": training, "architecture": "residual_gcn"})
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    forbidden = split.forbidden_negative_keys()
    first = train_models(graph, split, config, torch.device("cpu"))
    monkeypatch.setattr(EdgeSplit, "forbidden_negative_keys", lambda self: forbidden)
    second = train_models(graph, replace(split, pool=split.pool[:0]), config, torch.device("cpu"))
    for name, tensor in first.denoiser.state_dict().items():
        assert torch.equal(tensor, second.denoiser.state_dict()[name]), name
    if training == "damage_aware":
        drops = first.history["denoiser_drop_fractions"]
        assert len(drops) == config.denoiser.epochs
        assert all(0.0 <= d < config.denoiser.train_max_drop for d in drops)


def test_validation_recovery_uses_train_edges_only_and_calibrates_tau(graph, monkeypatch):
    import stability.validation_recovery as validation_module

    config = tiny_config()
    split = split_edges(graph.edges, graph.node_count, 0.5, 0.1, 0.4, 40, seed=3)
    models = train_models(graph, split, config, torch.device("cpu"))
    seen = []
    real = validation_module.normalized_adjacency

    def spy(node_count, edges, device):
        seen.append(np.asarray(edges).copy())
        return real(node_count, edges, device)

    monkeypatch.setattr(validation_module, "normalized_adjacency", spy)
    result = validation_module.validation_recovery(graph, split, models, config, torch.device("cpu"))
    assert len(seen) == 1 and np.array_equal(seen[0], split.train)
    for method in ("diffusion", "no_diffusion", "clamp_only"):
        summary = result[method]
        assert 0.0 <= summary["auc_mean_probability"] <= 1.0
        # calibrated so that the validation false-positive rate is about target_fpr
        assert summary["false_positive_rate"] <= config.recovery.target_fpr + 1.0 / len(split.val_negatives)

    fixed = tiny_config(recovery={"tau_mode": "fixed", "tau": 0.5})
    result = validation_module.validation_recovery(graph, split, models, fixed, torch.device("cpu"))
    assert result["diffusion"]["tau"] == 0.5 and result["no_diffusion"]["tau"] == 0.5


def test_calibrate_tau_respects_target_with_ties():
    from stability.validation_recovery import calibrate_tau

    negatives = np.array([0.9] * 30 + [0.2] * 70)
    tau = calibrate_tau(negatives, 0.05)
    assert np.mean(negatives >= tau) <= 0.05
    spread = np.linspace(0, 1, 1001)
    tau = calibrate_tau(spread, 0.05)
    assert np.mean(spread >= tau) == pytest.approx(0.05, abs=0.002)
