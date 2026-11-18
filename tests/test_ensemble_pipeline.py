"""Decision 11: ensemble reconstructions in the pipeline (reconstruction k = model k), paired with the
single-model ("noise") source."""

import json
import shutil

import numpy as np
import pytest
import torch

import stability.ensemble_training as ensemble_training
from stability import bootstrap_ensemble
from stability.edge_recovery_experiment import run_experiment, run_recoveries
from stability.edge_split import pair_keys, split_edges
from stability.ensemble_training import load_models, member_config, save_models, train_ensemble
from stability.experiment_config import RESAMPLE_MODES, load_config
from stability.model_training import train_models
from stability.perturbation_masks import make_damage_series

from test_edge_recovery_pipeline import CORA_CONFIG, graph, tiny_config  # noqa: F401  (fixture)

DEVICE = torch.device("cpu")


def _setup(graph, **changes):
    config = tiny_config(**changes)
    s, p = config.split, config.perturbation
    split = split_edges(graph.edges, graph.node_count, s.train_fraction, s.val_fraction, s.pool_fraction,
                        s.control_pairs, s.seed)
    series = make_damage_series(split.pool, split.all_edges, graph.node_count, p.damage_fractions,
                                ("random",), 2, p.seed)
    return config, split, series


@pytest.fixture(scope="module")
def trained(graph):  # noqa: F811
    config, split, series = _setup(graph, recovery={"reconstructions": 3})
    base = train_models(graph, split, config, DEVICE)
    members = [base] + [train_models(graph, split, member_config(config, m), DEVICE) for m in (1, 2)]
    return config, split, series, members


# ---------------------------------------------------------------- config


def test_cora_config_runs_both_sources_with_seed_only():
    config = load_config(CORA_CONFIG)
    assert config.recovery.reconstruction_sources == ("noise", "ensemble")
    assert config.ensemble.resample_mode == "seed_only"
    assert config.recovery.reconstructions == 10           # K = 10 models
    assert RESAMPLE_MODES == bootstrap_ensemble.RESAMPLE_MODES


@pytest.mark.parametrize("changes, message", [
    ({"recovery": {"edge_decision": "bernoulli"}}, "pairwise"),
    ({"recovery": {"reconstruction_sources": ["noise", "dropout"]}}, "reconstruction_sources"),
    ({"recovery": {"reconstruction_sources": []}}, "reconstruction_sources"),
    ({"recovery": {"reconstructions": 1}}, "reconstructions >= 2"),
    ({"ensemble": {"resample_mode": "jackknife"}}, "resample_mode"),
    ({"ensemble": {"fraction": 0.0}}, "fraction"),
])
def test_invalid_ensemble_settings_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**changes)


def test_member_config_changes_only_seeds():
    config = tiny_config()
    first, other = member_config(config, 0), member_config(config, 3)
    assert first == config                                  # seed_only member 0 = the single model
    assert other.model.seed != config.model.seed and other.community.seed != config.community.seed
    assert other.with_overrides(model={"seed": config.model.seed},
                                community={"seed": config.community.seed}) == config


# ---------------------------------------------------------------- recoveries


def _counts(results):
    return {k: v for k, v in results["random"].items()}


def test_same_model_for_every_member_equals_the_noise_source(graph, trained):  # noqa: F811
    """Pairing: noise seeds and reference draws do not depend on the source."""
    config, split, series, members = trained
    base = members[0]
    single = run_recoveries(graph, split, series, base, config, DEVICE, {}, keep_reconstructions=True)
    same = run_recoveries(graph, split, series, base, config, DEVICE, {}, keep_reconstructions=True,
                          member_models=[base] * 3)
    for key, value in _counts(single).items():
        assert np.array_equal(value, same["random"][key]), key


def test_reconstruction_k_goes_through_model_k(graph, trained):  # noqa: F811
    config, split, series, members = trained
    ensemble = run_recoveries(graph, split, series, members[0], config, DEVICE, {}, keep_reconstructions=True,
                              member_models=members)["random"]
    for k, member in enumerate(members):
        alone = run_recoveries(graph, split, series, member, config, DEVICE, {}, keep_reconstructions=True)["random"]
        # diffusion: same model, noise draw k and reference draw k -> identical outcome / probabilities
        assert np.array_equal(ensemble["recovered_by_reconstruction"][:, :, k],
                              alone["recovered_by_reconstruction"][:, :, k])
        np.testing.assert_array_equal(ensemble["diffusion_probability_by_reconstruction_sum"][:, k],
                                      alone["diffusion_probability_by_reconstruction_sum"][:, k])
        # deterministic methods also use model k in the ensemble
        for method in ("no_diffusion", "clamp_only"):
            np.testing.assert_array_equal(ensemble[f"{method}_probability_by_reconstruction_sum"][:, k],
                                          alone[f"{method}_probability_by_reconstruction_sum"][:, k])
            assert np.array_equal(ensemble[f"{method}_by_reconstruction"][:, :, k],
                                  alone[f"{method}_by_reconstruction"][:, :, k])
    # models differ -> deterministic scores vary across reconstructions in the ensemble only
    nd = ensemble["no_diffusion_probability_by_reconstruction_sum"]
    assert not np.allclose(nd[:, 0], nd[:, 1])
    for name in ("recovered", "no_diffusion", "clamp_only"):
        key = "recovered_by_reconstruction" if name == "recovered" else f"{name}_by_reconstruction"
        assert np.array_equal(ensemble[key].sum(axis=2), ensemble[name])


def test_member_models_are_checked(graph, trained):  # noqa: F811
    config, split, series, members = trained
    with pytest.raises(ValueError, match="one model per reconstruction"):
        run_recoveries(graph, split, series, members[0], config, DEVICE, {}, member_models=members[:2])
    threshold = config.with_overrides(recovery={"edge_decision": "threshold", "reconstruction_sources": ("noise",)})
    taus = {m: 0.5 for m in ("diffusion", "no_diffusion", "clamp_only")}
    with pytest.raises(ValueError, match="pairwise"):
        run_recoveries(graph, split, series, members[0], threshold, DEVICE, taus, member_models=members)


# ---------------------------------------------------------------- training, saving, loading


def test_save_and_load_reproduce_the_model(graph, trained, tmp_path):  # noqa: F811
    config, split, _, members = trained
    model = members[1]
    save_models(model, tmp_path / "m", {"member": 1})
    assert (tmp_path / "m" / "done").is_file()
    loaded = load_models(graph, split.train, member_config(config, 1), DEVICE, tmp_path / "m")
    pairs = torch.as_tensor(split.control_pairs[:30])
    adjacency = model.train_adjacency
    with torch.no_grad():
        a = model.encoder.decode(model.encoder.encode(graph.features, adjacency)[2], pairs)
        b = loaded.encoder.decode(loaded.encoder.encode(graph.features, loaded.train_adjacency)[2], pairs)
    assert torch.equal(a, b)
    assert torch.equal(model.context.labels, loaded.context.labels)


def test_ensemble_training_uses_train_edges_only_and_resumes(graph, monkeypatch, tmp_path):  # noqa: F811
    config, split, _ = _setup(graph, recovery={"reconstructions": 3},
                              ensemble={"resample_mode": "subsample", "fraction": 0.7})
    base = train_models(graph, split, config, DEVICE)
    seen = []
    real = ensemble_training.train_models

    def spy(graph_, member_split_, member_config_, device, log_every=0):
        seen.append(member_split_)
        return real(graph_, member_split_, member_config_, device, log_every=log_every)

    monkeypatch.setattr(ensemble_training, "train_models", spy)
    members, records = train_ensemble(graph, split, config, DEVICE, tmp_path / "checkpoints", base)
    assert len(members) == 3 and len(seen) == 3            # subsample: member 0 is trained too
    train_keys = set(pair_keys(split.train, split.node_count).tolist())
    for msplit in seen:
        used = set(pair_keys(msplit.train, split.node_count).tolist())
        assert used <= train_keys and len(used) == round(0.7 * len(split.train))
        forbidden = set(msplit.forbidden_negative_keys().tolist())
        assert set(pair_keys(split.pool, split.node_count).tolist()) <= forbidden
        assert set(pair_keys(split.control_pairs, split.node_count).tolist()) <= forbidden
    assert [r["held_out_train_edges"] for r in records] == [len(split.train) - len(s.train) for s in seen]

    # resume: finished members are loaded, never retrained
    def fail(*args, **kwargs):
        raise AssertionError("a finished member was retrained")

    monkeypatch.setattr(ensemble_training, "train_models", fail)
    again, again_records = train_ensemble(graph, split, config, DEVICE, tmp_path / "checkpoints", base)
    assert again_records == records
    # an unfinished member (no "done" marker) is trained again
    (tmp_path / "checkpoints" / "member_02" / "done").unlink()
    monkeypatch.setattr(ensemble_training, "train_models", spy)
    train_ensemble(graph, split, config, DEVICE, tmp_path / "checkpoints", base)
    assert len(seen) == 4


def test_seed_only_member_zero_is_the_shared_single_model(graph, tmp_path):  # noqa: F811
    config, split, _ = _setup(graph, recovery={"reconstructions": 2})
    base = train_models(graph, split, config, DEVICE)
    members, records = train_ensemble(graph, split, config, DEVICE, tmp_path / "checkpoints", base)
    assert members[0] is base and records[0]["shared_with_noise_source"]
    assert not (tmp_path / "checkpoints" / "member_00").exists()
    assert (tmp_path / "checkpoints" / "member_01" / "done").is_file()


# ---------------------------------------------------------------- end to end


def test_end_to_end_paired_run_and_resume(graph, tmp_path, monkeypatch):  # noqa: F811
    config = tiny_config(perturbation={"series": 2, "attacks": ["random"]})
    run = run_experiment(config, config_text="# test", output_dir=tmp_path / "run", graph=graph, make_plots=True)
    for name in ("recovery_counts.npz", "recovery_counts_ensemble.npz", "edge_recovery_ensemble.csv",
                 "control_recovery_ensemble.csv", "ensemble_members.json", "figures/p_e_distributions_ensemble.png",
                 "checkpoints/done", "checkpoints/member_01/done", "checkpoints/member_01/validation_recovery.json",
                 "checkpoints/member_01/training_history.json"):
        assert (run / name).is_file(), name
    summary = json.loads((run / "summary.json").read_text())
    assert summary["reconstruction_sources"] == ["noise", "ensemble"]
    assert summary["ensemble"]["members"] == 2 and len(summary["ensemble"]["member_validation_auc_mean_probability"]) == 2
    assert set(summary["recovery_seconds"]) == {"noise", "ensemble"}
    noise = np.load(run / "recovery_counts.npz")
    ensemble = np.load(run / "recovery_counts_ensemble.npz")
    assert sorted(noise.files) == sorted(ensemble.files)
    assert np.array_equal(noise["random__removed"], ensemble["random__removed"])    # same damage
    # member 0 is shared: reconstruction 0 is identical in both sources
    assert np.array_equal(noise["random__recovered_by_reconstruction"][:, :, 0],
                          ensemble["random__recovered_by_reconstruction"][:, :, 0])

    with pytest.raises(FileExistsError):
        run_experiment(config, output_dir=run, graph=graph, make_plots=False)
    with pytest.raises(ValueError, match="differs"):
        run_experiment(config.with_overrides(recovery={"t_star": 5}), output_dir=run, graph=graph, resume=True)

    # interrupted after training: resume loads both models and reproduces the recoveries exactly
    copy = tmp_path / "copy"
    shutil.copytree(run, copy)
    for name in ("summary.json", "recovery_counts.npz", "recovery_counts_ensemble.npz"):
        (copy / name).unlink()

    def fail(*args, **kwargs):
        raise AssertionError("resume retrained a finished model")

    monkeypatch.setattr("stability.edge_recovery_experiment.train_models", fail)
    monkeypatch.setattr(ensemble_training, "train_models", fail)
    run_experiment(config, output_dir=copy, graph=graph, make_plots=False, resume=True)
    for name in ("recovery_counts.npz", "recovery_counts_ensemble.npz"):
        first, second = np.load(run / name), np.load(copy / name)
        for key in first.files:
            assert np.array_equal(first[key], second[key]), (name, key)
    assert json.loads((copy / "summary.json").read_text())["resumed"] is True
