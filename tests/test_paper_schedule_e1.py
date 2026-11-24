"""Experiment E1 (2026-11-24): plan validation, metrics and the GPU-runner stage."""

from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_paper_schedule import criteria_flags, load_plan
from scripts.run_gpu_checks import ROOT, load_plan as load_gpu_plan, stage_commands
from stability.experiment_config import config_from_dict
from stability.sample_diversity import schedule_variant_metrics

E1_PLAN = ROOT / "configs" / "e1_paper_schedule.toml"
GPU_E1 = ROOT / "configs" / "gpu_e1.toml"


def test_plan_variants_are_valid_configs_and_share_the_encoder():
    loaded = load_plan(E1_PLAN)
    variants = loaded["variants"]
    assert list(variants)[0] == "baseline_T200_t7"
    base = loaded["base"]
    for name, variant in variants.items():
        config = config_from_dict(variant["raw"])
        assert config.recovery.reconstructions == 10
        for section in ("split", "model", "autoencoder", "community", "data"):
            assert variant["raw"][section] == base[section], (name, section)
    baseline = config_from_dict(variants["baseline_T200_t7"]["raw"])
    assert (baseline.denoiser.timesteps, baseline.denoiser.radial_delta, baseline.recovery.t_star,
            baseline.denoiser.train_t_max, baseline.denoiser.angular_noise) == (200, 1.0, 7, 8, "upstream")
    paper = [config_from_dict(v["raw"]) for n, v in variants.items() if n.startswith("paper_")]
    assert [c.recovery.t_star for c in paper] == [35, 75, 150, 250, 500]
    for c in paper:
        assert (c.denoiser.timesteps, c.denoiser.radial_delta, c.denoiser.angular_noise) == (1000, 0.4, "folded")
        assert c.denoiser.train_t_max == round(c.recovery.t_star * 8 / 7)


def test_smoke_plan_is_short():
    loaded = load_plan(E1_PLAN, smoke=True)
    raw = next(iter(loaded["variants"].values()))["raw"]
    assert raw["denoiser"]["epochs"] <= 20 and loaded["reconstructions"] == 3


def _write(tmp_path, text):
    path = tmp_path / "plan.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_plan_rejects_encoder_changes_and_invalid_t_star(tmp_path):
    text = E1_PLAN.read_text(encoding="utf-8")
    bad = text.replace('"recovery.t_star" = 35', '"recovery.t_star" = 35\n"model.dim" = 64')
    with pytest.raises(ValueError, match="shared"):
        load_plan(_write(tmp_path, bad))
    bad = text.replace('[variant.overrides]\n\n[[variant]]', '[variant.overrides]\n"recovery.t_star" = 170\n\n[[variant]]', 1)
    with pytest.raises(ValueError, match="past the minimum"):
        load_plan(_write(tmp_path, bad))
    bad = text.replace('name = "paper_T1000_t75"', 'name = "paper_T1000_t35"')
    with pytest.raises(ValueError, match="duplicate"):
        load_plan(_write(tmp_path, bad))


def _synthetic(informative: bool, K=10, n=400, seed=0):
    """Mean logit separates edges a bit; if informative, edges also have a larger spread across K."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n), np.zeros(n)].astype(bool)
    mean = rng.normal(0, 1, 2 * n) + 0.8 * y
    scale = np.where(y, 1.5, 0.2) if informative else np.full(2 * n, 0.8)
    logits = mean[None] + rng.normal(0, 1, (K, 2 * n)) * scale[None]
    probs = 1 / (1 + np.exp(-logits))
    clamp = 1 / (1 + np.exp(-(mean + rng.normal(0, 0.3, 2 * n))))
    return probs[:, y], probs[:, ~y], clamp[y], clamp[~y]


def test_metrics_detect_an_informative_spread():
    informative = schedule_variant_metrics(*_synthetic(True), target_fpr=0.05)
    plain = schedule_variant_metrics(*_synthetic(False), target_fpr=0.05)
    assert informative["spread"]["stratified_auc_given_own_mean"] > 0.75
    assert informative["spread"]["ll_gain_over_own_mean"] > 0.05
    assert abs(plain["spread"]["stratified_auc_given_own_mean"] - 0.5) < 0.08
    assert abs(plain["spread"]["ll_gain_over_own_mean"]) < 0.01
    for m in (informative, plain):
        assert 0.5 < m["auc_mean_probability"] <= 1 and m["K"] == 10
        assert 0 <= m["expected_share_mean"] <= 1
        assert m["auc_gain_over_clamp_only"] == pytest.approx(m["auc_mean_probability"] - m["auc_clamp_only"])


def test_metrics_of_identical_samples_have_no_spread():
    pos, neg, cp, cn = _synthetic(False, K=1)
    m = schedule_variant_metrics(np.repeat(pos, 4, 0), np.repeat(neg, 4, 0), cp, cn, target_fpr=0.05)
    assert m["probability_std_positive"] == 0 and m["expected_share_std_across_K"] == 0
    assert m["p_e"]["share_intermediate"] == pytest.approx(0, abs=1e-12)
    assert m["spread"]["ll_gain_over_own_mean"] == 0.0
    with pytest.raises(ValueError):
        schedule_variant_metrics(pos[0], neg, cp, cn, target_fpr=0.05)


def test_criteria_flags():
    criteria = {"quality_tolerance": 0.0, "informative_auc": 0.55, "min_ll_gain": 0.001}
    m = {"auc_mean_probability": 0.86, "auc_clamp_only": 0.85,
         "p_e": {"stratified_auc_given_own_mean": 0.60, "ll_gain_over_own_mean": 0.01}}
    assert criteria_flags(m, criteria) == {"quality_pass": True, "informative": True}
    m["p_e"]["ll_gain_over_own_mean"] = 0.0
    m["auc_mean_probability"] = 0.84
    assert criteria_flags(m, criteria) == {"quality_pass": False, "informative": False}


def test_gpu_e1_plan_runs_env_tests_e1_pack(tmp_path):
    plan = load_gpu_plan(GPU_E1)
    assert plan["run"]["stages"] == ["env", "tests", "e1", "pack"]
    commands = stage_commands(plan, tmp_path, python="py")
    (command,) = commands["e1"]
    assert command[:2] == ["py", "scripts/diagnose_paper_schedule.py"]
    assert command[command.index("--device") + 1] == "cuda"
    assert Path(command[command.index("--plan") + 1]) == E1_PLAN
    assert Path(command[command.index("--out-dir") + 1]) == tmp_path / "e1"
    assert "ensemble" not in commands


def test_gpu_plan_requires_e1_section(tmp_path):
    text = GPU_E1.read_text(encoding="utf-8").replace('plan = "configs/e1_paper_schedule.toml"', "")
    with pytest.raises(ValueError, match=r"\[e1\]\.plan"):
        load_gpu_plan(_write(tmp_path, text))
