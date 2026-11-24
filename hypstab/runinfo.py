"""Run bookkeeping: device choice, seeding, and what is saved with every run.

Every run directory gets ``config.toml`` (the resolved config) and
``run_info.json`` (git commit, library versions, device, start time), as required
by Decision 6 / the project rules.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
from typing import Any

import numpy as np
import torch

from hypstab.config import HypstabConfig, config_to_toml

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_device(device: str) -> torch.device:
    """'auto' -> cuda if available else cpu; 'cuda' fails loudly when no GPU is visible."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device = 'cuda' but no CUDA GPU is visible")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {device!r}")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def git_info(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit hash and whether the working tree has uncommitted changes (None if not a git checkout)."""

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
    }
    for name in ("scipy", "networkx", "torch_geometric", "pandas", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def run_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "versions": library_versions(),
        "platform": platform.platform(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    return info


def start_run(config: HypstabConfig, run_dir: str | Path | None = None) -> tuple[Path, torch.device]:
    """Create the run directory, save config + run info, seed, and return (run_dir, device).

    Default directory: <run.output_root>/<run.name>_<UTC timestamp>. An existing
    directory is refused if it already holds a config.toml (no silent overwrite).
    """
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(config.run.output_root) / f"{config.run.name}_{stamp}"
    run_dir = Path(run_dir)
    if (run_dir / "config.toml").exists():
        raise FileExistsError(f"{run_dir} already contains a run (config.toml)")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.run.device)
    seed_everything(config.run.seed)
    (run_dir / "config.toml").write_text(config_to_toml(config), encoding="utf-8")
    (run_dir / "run_info.json").write_text(json.dumps(run_info(device), indent=2), encoding="utf-8")
    return run_dir, device
