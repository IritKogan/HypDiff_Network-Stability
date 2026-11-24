"""Tests for hypstab/config.py and hypstab/runinfo.py (Stage 0)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tomllib

import pytest
import torch

from hypstab.config import ConfigError, config_from_dict, config_to_dict, config_to_toml, load_config
from hypstab.runinfo import resolve_device, seed_everything, start_run

REPO_ROOT = Path(__file__).resolve().parents[2]
CORA_CONFIG = REPO_ROOT / "configs" / "hypstab" / "cora.toml"


@pytest.fixture()
def raw() -> dict:
    with CORA_CONFIG.open("rb") as handle:
        return tomllib.load(handle)


def test_cora_config_loads():
    config = load_config(CORA_CONFIG)
    assert config.geometry.manifold == "poincare"
    assert config.geometry.curvature == 1.0
    assert config.run.device in ("auto", "cpu", "cuda")


def test_toml_round_trip(raw):
    config = config_from_dict(raw)
    again = config_from_dict(tomllib.loads(config_to_toml(config)))
    assert again == config
    assert config_to_dict(again) == config_to_dict(config)


def test_integer_accepted_for_float(raw):
    raw["geometry"]["curvature"] = 2
    config = config_from_dict(raw)
    assert config.geometry.curvature == 2.0 and isinstance(config.geometry.curvature, float)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["run"].pop("seed"), "missing key"),
        (lambda r: r["run"].__setitem__("sed", 1), "unknown key"),
        (lambda r: r.__setitem__("extra", {}), "unknown key"),
        (lambda r: r.pop("geometry"), "missing key"),
        (lambda r: r["run"].__setitem__("seed", "1"), "must be an integer"),
        (lambda r: r["run"].__setitem__("seed", True), "must be an integer"),
        (lambda r: r["run"].__setitem__("seed", -1), ">= 0"),
        (lambda r: r["run"].__setitem__("device", "gpu"), "run.device must be one of"),
        (lambda r: r["geometry"].__setitem__("curvature", 0.0), "> 0"),
        (lambda r: r["geometry"].__setitem__("curvature", float("inf")), "> 0"),
        (lambda r: r["geometry"].__setitem__("curvature", "1"), "must be a number"),
        (lambda r: r["geometry"].__setitem__("manifold", "lorentz"), "geometry.manifold"),
        (lambda r: r.__setitem__("run", 3), "must be a table"),
    ],
)
def test_bad_values_rejected(raw, mutate, message):
    bad = copy.deepcopy(raw)
    mutate(bad)
    with pytest.raises(ConfigError, match=message):
        config_from_dict(bad)


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_seed_everything_is_reproducible():
    seed_everything(7)
    a = torch.rand(3)
    seed_everything(7)
    assert torch.equal(a, torch.rand(3))


def test_start_run_saves_config_and_info(tmp_path, raw):
    raw["run"]["device"] = "cpu"
    config = config_from_dict(raw)
    run_dir, device = start_run(config, tmp_path / "run")
    assert device == torch.device("cpu")
    assert load_config(run_dir / "config.toml") == config
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert {"started_utc", "git", "versions", "device"} <= set(info)
    assert info["versions"]["torch"] == torch.__version__
    with pytest.raises(FileExistsError):
        start_run(config, run_dir)
