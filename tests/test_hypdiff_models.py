"""Tests for stability/hypdiff_models.py and the upstream-untouched rule."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.direction_diffusion import gcndec, mlp
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.pilot_config import PilotConfig
from stability.synthetic_instances import generate_synthetic_graph
from stability.wp7_training import model_args

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_MANIFEST = Path(__file__).parent / "data" / "upstream_hypdiff_1620e3c.sha256"


def _normalized_sha256(path):
    # CRLF -> LF so Windows checkouts (core.autocrlf) hash the same.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_upstream_hypdiff_files_are_unmodified():
    """Decision 3: upstream files must stay identical to RingBDStack/HypDiff @ 1620e3c."""
    mismatched = []
    for line in UPSTREAM_MANIFEST.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = REPO_ROOT / relative
        if not path.exists() or _normalized_sha256(path) != expected:
            mismatched.append(relative)
    assert not mismatched, (
        "upstream HypDiff files changed; put project changes in stability/ instead: "
        f"{mismatched}"
    )


def test_stability_diffusion_builds_on_cpu_with_upstream_submodules():
    torch.manual_seed(0)
    model = StabilityDiffusion(SimpleNamespace(r=2.0, t=1.0), dim=8)
    torch.manual_seed(0)
    reference_gcn, reference_mlp = gcndec(8), mlp(8)

    assert model.alphas_cumprod.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert model.num_timesteps == 200
    # Same construction order as upstream -> identical initial weights and keys.
    for name, tensor in reference_gcn.state_dict().items():
        assert torch.equal(model.gcn.state_dict()[name].cpu(), tensor)
    for name, tensor in reference_mlp.state_dict().items():
        assert torch.equal(model.mlp.state_dict()[name].cpu(), tensor)
    assert {key.split(".")[0] for key in model.state_dict()} == {"gcn", "mlp"}


def test_stability_diffusion_rejects_non_positive_curvature():
    with pytest.raises(ValueError, match="curvature must be positive"):
        StabilityDiffusion(SimpleNamespace(r=2.0, t=1.0), dim=8, curvature=0.0)


def test_pilot_config_propagates_noise_formula_to_denoiser():
    graph = generate_synthetic_graph(71001, "calibration")
    args = model_args(graph, PilotConfig(noise_formula="upstream"))
    assert StabilityDiffusion(args, dim=args.dim).noise_formula == "upstream"
    with pytest.raises(ValueError, match="noise_formula"):
        PilotConfig(noise_formula="ddpm")


def test_stability_lp_model_rejects_unknown_final_activation():
    graph = generate_synthetic_graph(71001, "calibration")
    args = model_args(graph, PilotConfig(), final_activation="relu")
    args.final_activation = "tanh"
    with pytest.raises(ValueError, match="final_activation"):
        StabilityLPModel(args)


def test_stability_lp_model_leaky_relu_keeps_initial_weights():
    graph = generate_synthetic_graph(71001, "calibration")
    torch.manual_seed(3)
    relu = StabilityLPModel(model_args(graph, PilotConfig(), final_activation="relu"))
    torch.manual_seed(3)
    leaky = StabilityLPModel(model_args(graph, PilotConfig(), final_activation="leaky_relu"))

    assert relu.state_dict().keys() == leaky.state_dict().keys()
    for name, tensor in relu.state_dict().items():
        assert torch.equal(tensor, leaky.state_dict()[name])
