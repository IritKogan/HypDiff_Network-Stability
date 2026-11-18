"""Consolidated run for a local machine with an NVIDIA GPU (Windows or Linux).

Runs, in order (configs/gpu_checks.toml, [run].stages):
  env        versions, GPU name, CUDA check                          -> env.json
  tests      full pytest suite                                       -> logs/tests.log
  smoke      short GPU run of the experiment (a few minutes)          -> smoke/
  experiment full Cora run, scripts/run_experiment.py: single model AND
             the K-model ensemble, paired (Decision 11)              -> runs/<run name>/
  ensemble   (optional, not in the default stages) validation-only
             ensemble diagnostic                                     -> ensemble/bootstrap_ensemble.json
  pack       zip of the whole output directory, to send back         -> <out>.zip

Every stage logs to <out>/logs/<stage>.log and leaves <out>/stage_<name>.done when it finishes.
Re-running with the same --out skips finished stages; the experiment resumes inside its run directory
(trained models are loaded, not retrained), so an interrupted run can simply be started again.

Setup on the GPU machine (once):
    git clone --branch phase-b-work https://github.com/IritKogan/HypDiff_Network-Stability
    cd HypDiff_Network-Stability
    conda create -n hypdiff-gpu python=3.11 -y && conda activate hypdiff-gpu
    # torch WITH CUDA (on Windows the plain "pip install torch" is CPU-only). torch 2.12 wheels exist
    # for cu126 / cu130 / cu132. Blackwell GPUs (RTX 50xx, RTX PRO ... Blackwell; sm_120) need cu130 or
    # newer; older GPUs with a driver below CUDA 13 (nvidia-smi "CUDA Version") need cu126.
    pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
    pip install -r requirements-phaseb.txt pytest
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
Then:
    python scripts/run_gpu_checks.py                       # all stages, new output directory
    python scripts/run_gpu_checks.py --out artifacts/gpu_checks/<dir>   # resume an interrupted run
    python scripts/run_gpu_checks.py --stages env smoke    # only some stages
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("env", "tests", "smoke", "experiment", "ensemble", "pack")


# ----------------------------------------------------------------------------- plan / config
def load_plan(path: Path) -> dict[str, Any]:
    """Read and validate configs/gpu_checks.toml."""
    plan = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    run = plan.get("run", {})
    for key in ("base_config", "output_root", "device", "stages", "run_name"):
        if key not in run:
            raise ValueError(f"[run].{key} missing in {path}")
    unknown = [s for s in run["stages"] if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; expected a subset of {STAGES}")
    if run["device"] not in ("cuda", "auto", "cpu"):
        raise ValueError("[run].device must be 'cuda', 'auto' or 'cpu'")
    ensemble = plan.get("ensemble", {})
    for key in ("modes", "members", "fraction", "jobs", "seed"):
        if key not in ensemble:
            raise ValueError(f"[ensemble].{key} missing in {path}")
    if int(ensemble["members"]) < 2:
        raise ValueError("[ensemble].members must be >= 2")
    return plan


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Copy of a config dict with "section.key" = value overrides; keys must already exist."""
    result = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if not key or section not in result or key not in result[section]:
            raise KeyError(f"override {dotted!r} does not name an existing config key")
        result[section][key] = value
    return result


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # TOML basic strings use JSON-compatible escapes
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def dump_toml(raw: dict[str, Any]) -> str:
    """Minimal writer for flat ``[section] key = scalar | list`` configs (like configs/cora.toml)."""
    lines = []
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise TypeError("top-level entries must be tables")
        lines.append(f"[{section}]")
        lines += [f"{key} = {_toml_value(value)}" for key, value in values.items()]
        lines.append("")
    return "\n".join(lines)


def smoke_config(base: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    raw = apply_overrides(base, plan["smoke"])
    raw["run"]["name"] = f"{base['run']['name']}_smoke"
    return raw


def stage_commands(plan: dict[str, Any], out: Path, python: str = sys.executable) -> dict[str, list[list[str]]]:
    """Subprocess commands of the stages that run programs (run with cwd = repository root)."""
    run, ens = plan["run"], plan["ensemble"]
    device = run["device"]
    base = str(ROOT / run["base_config"])
    smoke_toml = str(out / "smoke" / "smoke_config.toml")

    def ensemble(config: str, members: int, target: Path) -> list[str]:
        return [python, "scripts/diagnose_bootstrap_ensemble.py", "--config", config,
                "--device", device, "--modes", *ens["modes"], "--members", str(members),
                "--fraction", str(ens["fraction"]), "--jobs", str(ens["jobs"]),
                "--seed", str(ens["seed"]), "--out", str(target)]

    return {
        "tests": [[python, "-m", "pytest", "-q", "-p", "no:warnings"]],
        "smoke": [
            [python, "scripts/run_experiment.py", "--config", smoke_toml, "--device", device,
             "--run-dir", str(out / "smoke" / "run"), "--resume", "--log-every", "10"],
        ],
        "experiment": [[python, "scripts/run_experiment.py", "--config", base, "--device", device,
                        "--run-dir", str(out / "runs" / run["run_name"]), "--resume"]],
        "ensemble": [ensemble(base, int(ens["members"]), out / "ensemble")],
    }


# ----------------------------------------------------------------------------- execution
def _run_logged(command: list[str], log_path: Path) -> None:
    """Run a command from the repository root, echo its output and append it to log_path."""
    print("$ " + " ".join(command), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise RuntimeError(f"command failed (exit {process.returncode}); see {log_path}")


def _required_versions() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements-phaseb.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.strip().startswith("#"):
            name, version = line.strip().split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def unsupported_gpus(capabilities: list[tuple[int, int]], arch_list: list[str]) -> list[str]:
    """GPU architectures (sm_XY) that the installed torch build has no kernels for."""
    built = {a for a in arch_list if a.startswith("sm_")}
    return [f"sm_{major}{minor}" for major, minor in capabilities if f"sm_{major}{minor}" not in built]


def environment_check(device: str) -> dict[str, Any]:
    from importlib import metadata

    import torch

    pins = _required_versions()
    installed = {}
    for name in pins:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [{"name": torch.cuda.get_device_name(i),
                  "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                 for i in range(torch.cuda.device_count())],
        "torch_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "pinned": pins,
        "installed": installed,
        "version_mismatches": {n: {"pinned": pins[n], "installed": installed[n]}
                               for n in pins if (installed[n] or "").split("+")[0] != pins[n]},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    print(json.dumps(record, indent=2))
    if sys.version_info < (3, 11):
        raise RuntimeError("Python >= 3.11 is required (tomllib)")
    if device == "cuda" and not record["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available to torch (torch build: %s). Install a CUDA build of torch, see the "
            "setup notes at the top of this script." % torch.__version__)
    if record["cuda_available"]:
        missing = unsupported_gpus([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())],
                                   record["torch_arch_list"])
        if missing and device != "cpu":
            raise RuntimeError(
                f"this torch build ({torch.__version__}) has no kernels for the GPU architecture {missing} "
                f"(built for {record['torch_arch_list']}). Newer GPUs (e.g. RTX 50xx / Blackwell, sm_120) need "
                "a CUDA 13 build: pip install --force-reinstall torch==2.12.0 "
                "--index-url https://download.pytorch.org/whl/cu130")
    if installed.get("networkx") != pins.get("networkx"):
        raise RuntimeError("networkx must be exactly %s (pip install networkx==%s)"
                           % (pins.get("networkx"), pins.get("networkx")))
    if record["version_mismatches"]:
        print("WARNING: versions differ from requirements-phaseb.txt:", record["version_mismatches"])
    return record


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - git may be missing (zip download)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs" / "gpu_checks.toml")
    parser.add_argument("--out", type=Path, help="output directory (reuse it to resume)")
    parser.add_argument("--stages", nargs="+", choices=STAGES, help="override [run].stages")
    parser.add_argument("--force", action="store_true", help="re-run stages that already finished")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    stages = args.stages or plan["run"]["stages"]
    out = args.out
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(plan["run"]["output_root"]) / f"gpu_checks_{stamp}"
    out = out if out.is_absolute() else ROOT / out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.plan, out / "gpu_checks_plan.toml")
    base = tomllib.loads((ROOT / plan["run"]["base_config"]).read_text(encoding="utf-8"))
    (out / "smoke").mkdir(exist_ok=True)
    (out / "smoke" / "smoke_config.toml").write_text(dump_toml(smoke_config(base, plan)), encoding="utf-8")
    commands = stage_commands(plan, out)
    print(f"output directory: {out}\nstages: {stages}", flush=True)

    timings = {}
    timing_file = out / "timings.json"
    if timing_file.exists():
        timings = json.loads(timing_file.read_text(encoding="utf-8"))
    for stage in stages:
        marker = out / f"stage_{stage}.done"
        if marker.exists() and not args.force and stage != "pack":
            print(f"\n=== {stage}: already done, skipped ===", flush=True)
            continue
        print(f"\n=== {stage} ===", flush=True)
        started = time.time()
        if stage == "env":
            record = environment_check(plan["run"]["device"])
            (out / "env.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        elif stage == "pack":
            archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
            print(f"archive: {archive}")
        else:
            for command in commands[stage]:
                _run_logged(command, out / "logs" / f"{stage}.log")
        timings[stage] = round(time.time() - started, 1)
        timing_file.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        print(f"=== {stage}: done in {timings[stage]:.0f}s ===", flush=True)
    print(f"\nAll requested stages finished. Output: {out}")


if __name__ == "__main__":
    main()
