"""Tests for Decision 14 round 1 (scripts/hypstab_round1.py, configs/hypstab/round1.toml)."""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from hypstab.config import config_from_dict
from hypstab.experiment.models import member_seed

from .test_autoencoder import make_split, sbm_graph, tiny_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs" / "hypstab"
PLAN = CONFIGS / "round1.toml"


def load_script():
    spec = importlib.util.spec_from_file_location("hypstab_round1", REPO_ROOT / "scripts" / "hypstab_round1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r1():
    return load_script()


@pytest.fixture(scope="module")
def variants(r1):
    return r1.build_variants(r1.load_plan(PLAN), CONFIGS)


def test_plan_values(r1):
    plan = r1.load_plan(PLAN)
    assert plan["round"]["seeds"] == 3
    assert plan["round"]["k_values"] == [3, 7, 20, 50, 100, 200, 500]
    paper = next(p for p in plan["package"] if p["name"] == "paper")["overrides"]
    assert paper == {"encoder.dim": 256, "encoder.learning_rate": 0.001, "encoder.weight_decay": 1e-5,
                     "encoder.dropout": 0.2, "encoder.adjacency_norm": "row", "encoder.grad_clip_norm": 10.0}


def test_variant_count_and_ids(variants):
    # Cora: 2 packages x 3 edge drops; three featureless networks: 2 features x 2 x 3; x 3 seeds
    assert len(variants) == (6 + 3 * 12) * 3 == 126
    assert len({v["id"] for v in variants}) == 126
    assert {v["network"] for v in variants} == {"cora", "cora_structure", "sbm", "ba"}
    assert {v["features"] for v in variants if v["network"] == "cora"} == {"original"}
    assert {v["features"] for v in variants if v["network"] == "ba"} == {"identity", "identity_degree"}


def test_each_variant_changes_only_its_keys(variants):
    allowed = {("data", "features"), ("encoder", "seed"), ("community", "seed"), ("encoder", "dim"),
               ("encoder", "learning_rate"), ("encoder", "weight_decay"), ("encoder", "dropout"),
               ("encoder", "adjacency_norm"), ("encoder", "grad_clip_norm"), ("encoder", "edge_drop_mode"),
               ("encoder", "edge_drop_fraction"), ("encoder", "edge_drop_loss")}
    for v in variants:
        base = tomllib.loads((CONFIGS / f"{v['network']}.toml").read_text(encoding="utf-8"))
        for section, values in v["raw"].items():
            for key, value in values.items():
                if base[section][key] != value:
                    assert (section, key) in allowed, (v["id"], section, key)
        enc = config_from_dict(v["raw"]).encoder
        if v["package"] == "paper":
            assert (enc.dim, enc.learning_rate, enc.weight_decay, enc.dropout, enc.adjacency_norm,
                    enc.grad_clip_norm) == (256, 0.001, 1e-5, 0.2, "row", 10.0)
        else:
            assert (enc.dim, enc.adjacency_norm, enc.grad_clip_norm) == (base["encoder"]["dim"], "symmetric", 0.0)
        expected = {"none": ("none", 0.0), "fixed02": ("fixed", 0.02), "uniform04": ("uniform", 0.4)}
        assert (enc.edge_drop_mode, enc.edge_drop_fraction) == expected[v["edge_drop"]]
        assert enc.seed == member_seed(base["encoder"]["seed"], v["seed_index"])
        assert v["raw"]["community"]["seed"] == member_seed(base["community"]["seed"], v["seed_index"])


def test_seed_zero_keeps_config_seeds(variants):
    v = next(v for v in variants if v["seed_index"] == 0)
    base = tomllib.loads((CONFIGS / f"{v['network']}.toml").read_text(encoding="utf-8"))
    assert v["raw"]["encoder"]["seed"] == base["encoder"]["seed"]


def test_variant_metrics_measure_validation_pairs_only(r1, raw_cora):
    """Measurements use split.val / split.val_negatives only (training leakage: test_autoencoder /
    test_encoder_edge_drop; pool and control pairs enter training only as excluded negatives)."""
    import inspect

    source = inspect.getsource(r1.variant_metrics)
    assert "split.pool" not in source and "control_pairs" not in source
    graph = sbm_graph()
    config = tiny_config(raw_cora, max_epochs=5, patience=5)
    split = make_split(graph, config)
    metrics = r1.variant_metrics(graph, split, config, torch.device("cpu"), [2, 3], [0.05, 0.5, 0.95])
    assert set(metrics["clamp_only_auc"]) == {"clamp_k2", "clamp_k3"}
    assert metrics["no_diffusion_auc"] == pytest.approx(metrics["best_val_auc_training"], abs=1e-9)
    check = metrics["ba_check"]
    assert set(check["probability_val_edges"]) == {"q05", "q50", "q95"}
    assert check["probability_val_edges"]["q95"] <= 0.881 + 1e-6   # Fermi-Dirac ceiling
    assert -1 <= check["spearman_radius_vs_train_degree"] <= 1


@pytest.fixture()
def raw_cora() -> dict:
    with (CONFIGS / "cora.toml").open("rb") as handle:
        return tomllib.load(handle)


def tiny_plan(tmp_path: Path) -> Path:
    plan = PLAN.read_text(encoding="utf-8")
    plan = plan.replace("seeds = 3", "seeds = 1").replace("k_values = [3, 7, 20, 50, 100, 200, 500]",
                                                          "k_values = [2, 3]")
    path = tmp_path / "plan.toml"
    path.write_text(plan, encoding="utf-8")
    return path


def test_end_to_end_resume_and_zip(r1, tmp_path, capsys):
    plan = tiny_plan(tmp_path)
    run_dir = tmp_path / "round1"
    args = ["--plan", str(plan), "--run-dir", str(run_dir), "--smoke", "--only", "ba", "--device", "cpu"]
    rows = r1.main(args + ["--zip"])
    assert len(rows) == 2 * 2 * 3                      # 2 features x 2 packages x 3 edge drops, 1 seed
    with (run_dir / "round1_summary.csv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 12 and {"no_diffusion_auc", "clamp_k2", "clamp_k3", "spearman_radius_vs_train_degree",
                                 "probability_val_non_edges_q50", "distance_val_edges_q05"} <= set(table[0])
    assert (tmp_path / "round1.zip").exists()
    saved = json.loads((run_dir / "variants" / table[0]["id"] / "metrics.json").read_text(encoding="utf-8"))
    assert "history" not in saved
    assert not list(run_dir.rglob("*.pt"))            # no model weights in the run dir / zip
    with pytest.raises(SystemExit, match="--resume"):
        r1.main(args)
    capsys.readouterr()
    again = r1.main(args + ["--resume"])
    assert "loaded" in capsys.readouterr().out
    assert [r["no_diffusion_auc"] for r in again] == [r["no_diffusion_auc"] for r in rows]
    plan.write_text(plan.read_text(encoding="utf-8").replace("k_values = [2, 3]", "k_values = [2]"),
                    encoding="utf-8")
    with pytest.raises(SystemExit, match="plan differs"):
        r1.main(args + ["--resume"])


def test_clamp_equals_fit_communities_and_validation_report(r1, raw_cora):
    """The script's clamp on the shared embeddings = the pipeline's own route (deterministic on CPU)."""
    from hypstab.config import config_to_toml
    from hypstab.evaluation.validation import validation_report
    from hypstab.training.autoencoder import train_autoencoder
    from hypstab.training.communities import fit_communities

    graph = sbm_graph()
    config = tiny_config(raw_cora, max_epochs=5, patience=5)
    split = make_split(graph, config)
    torch.manual_seed(0)
    metrics = r1.variant_metrics(graph, split, config, torch.device("cpu"), [2, 3], [0.5])
    torch.manual_seed(0)
    result = train_autoencoder(graph, split, config, torch.device("cpu"))
    for k in (2, 3):
        raw = tomllib.loads(config_to_toml(config))
        raw["community"]["n_clusters"] = k
        config_k = config_from_dict(raw)
        context = fit_communities(result.model, graph.features, result.train_adjacency, config_k)
        report = validation_report(result.model, graph.features, result.train_adjacency, split, context, config_k)
        assert metrics["clamp_only_auc"][f"clamp_k{k}"] == pytest.approx(report["clamp_only"]["auc"], abs=1e-12)
        assert metrics["no_diffusion_auc"] == pytest.approx(report["no_diffusion"]["auc"], abs=1e-12)
