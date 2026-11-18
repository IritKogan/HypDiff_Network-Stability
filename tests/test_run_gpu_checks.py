from pathlib import Path
import tomllib

import pytest

from scripts.run_gpu_checks import (
    ROOT, STAGES, apply_overrides, dump_toml, load_plan, smoke_config, stage_commands, unsupported_gpus,
)
from stability.experiment_config import config_from_dict

PLAN = ROOT / "configs" / "gpu_checks.toml"
BASE = ROOT / "configs" / "cora.toml"


def _base():
    return tomllib.loads(BASE.read_text(encoding="utf-8"))


def test_plan_is_valid_and_the_experiment_runs_both_sources():
    plan = load_plan(PLAN)
    assert set(plan["run"]["stages"]) <= set(STAGES)
    assert "experiment" in plan["run"]["stages"]
    base = config_from_dict(_base())
    assert base.recovery.reconstruction_sources == ("noise", "ensemble")


def test_plan_rejects_unknown_stage(tmp_path):
    text = PLAN.read_text(encoding="utf-8").replace('"pack"]', '"pack", "deploy"]')
    bad = tmp_path / "plan.toml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(bad)


def test_dump_toml_round_trips_the_cora_config():
    base = _base()
    assert tomllib.loads(dump_toml(base)) == base


def test_overrides_must_name_existing_keys():
    base = _base()
    changed = apply_overrides(base, {"denoiser.epochs": 3})
    assert changed["denoiser"]["epochs"] == 3 and base["denoiser"]["epochs"] != 3
    for bad in ("denoiser.epoch", "nosection.epochs", "denoiser"):
        with pytest.raises(KeyError):
            apply_overrides(base, {bad: 1})


def test_smoke_config_is_a_valid_short_experiment():
    plan = load_plan(PLAN)
    raw = smoke_config(_base(), plan)
    config = config_from_dict(tomllib.loads(dump_toml(raw)))
    assert config.denoiser.epochs == plan["smoke"]["denoiser.epochs"]
    assert config.recovery.reconstructions == plan["smoke"]["recovery.reconstructions"]
    assert config.run.name.endswith("_smoke")
    # everything else unchanged (same split / seeds -> no leakage differences)
    assert raw["split"] == _base()["split"] and raw["recovery"]["t_star"] == _base()["recovery"]["t_star"]


def test_stage_commands_pass_device_and_ensemble_parameters(tmp_path):
    plan = load_plan(PLAN)
    commands = stage_commands(plan, tmp_path, python="py")
    (experiment,) = commands["experiment"]
    assert experiment[:2] == ["py", "scripts/run_experiment.py"]
    assert experiment[experiment.index("--device") + 1] == plan["run"]["device"]
    assert Path(experiment[experiment.index("--run-dir") + 1]) == tmp_path / "runs" / plan["run"]["run_name"]
    assert "--resume" in experiment
    (ensemble,) = commands["ensemble"]
    assert ensemble[ensemble.index("--members") + 1] == "10"
    assert ensemble[ensemble.index("--device") + 1] == plan["run"]["device"]
    assert Path(ensemble[ensemble.index("--out") + 1]) == tmp_path / "ensemble"
    (smoke_run,) = commands["smoke"]
    assert smoke_run[smoke_run.index("--config") + 1].endswith("smoke_config.toml")
    for stage in ("smoke", "experiment", "ensemble"):
        for command in commands[stage]:
            assert (ROOT / command[1]).exists()


def test_unsupported_gpu_architecture_is_detected():
    cu126 = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    assert unsupported_gpus([(12, 0)], cu126) == ["sm_120"]        # Blackwell on a cu126 build
    assert unsupported_gpus([(8, 6)], cu126) == []
    assert unsupported_gpus([(12, 0)], cu126 + ["sm_100", "sm_120"]) == []
