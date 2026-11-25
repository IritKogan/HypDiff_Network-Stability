"""Run the full stability experiment (Stage 5): models, damage series, recovery, p_e ways A and B.

Usage (from the repository root):
    python scripts/hypstab_run.py --config configs/hypstab/cora.toml [--run-dir DIR]
        [--resume]                     continue an interrupted run in --run-dir (same config required)
        [--set section.key=value ...]  override config values (TOML syntax); saved in the run's config.toml
        [--smoke]                      tiny settings (few epochs / series) to check that everything runs
        [--zip]                        at the end write <run_dir>.zip (everything except models/) to send back

See hypstab/experiment/pipeline.py for the files written to the run directory.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import tomllib
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hypstab.config import config_from_dict, config_to_toml  # noqa: E402
from hypstab.experiment.pipeline import run_experiment  # noqa: E402
from hypstab.runinfo import resolve_device, run_info, seed_everything, start_run  # noqa: E402
from hypstab_train import apply_overrides  # noqa: E402

SMOKE = [
    "encoder.max_epochs=30", "encoder.patience=30", "denoiser.epochs=20",
    "perturbation.series=2", "recovery.reconstructions=3", "experiment.way_b_references=3",
]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "hypstab" / "cora.toml"))
    parser.add_argument("--run-dir", default=None, help="default: <run.output_root>/<run.name>_<UTC>")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)

    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    overrides = (SMOKE if args.smoke else []) + args.set
    if args.smoke:
        raw["run"]["name"] = raw["run"]["name"] + "_smoke"
    config = config_from_dict(apply_overrides(raw, overrides))

    if args.resume:
        if not args.run_dir:
            raise SystemExit("--resume needs --run-dir")
        run_dir = Path(args.run_dir)
        saved = (run_dir / "config.toml").read_text(encoding="utf-8")
        if saved != config_to_toml(config):
            raise SystemExit("--resume: the config differs from the run's saved config.toml")
        device = resolve_device(config.run.device)
        seed_everything(config.run.seed)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (run_dir / f"run_info_resume_{stamp}.json").write_text(json.dumps(run_info(device), indent=2),
                                                               encoding="utf-8")
    else:
        run_dir, device = start_run(config, args.run_dir)
    print(f"run dir: {run_dir} | device: {device}", flush=True)

    started = time.perf_counter()
    summary = run_experiment(config, run_dir, device, log_every=args.log_every)
    print(f"done in {time.perf_counter() - started:.0f} s", flush=True)
    for row in summary["headline"]:
        print(f"{row['source']:8s} {row['attack']:22s} {row['damage_fraction']:.2f} {row['method']:12s} "
              f"p_e A {row['mean_p_e_way_a']:.3f} | B {row['mean_p_e_way_b']:.3f} | "
              f"AUC {row['auc_mean_probability']:.3f}", flush=True)
    if args.zip:
        archive = write_zip(Path(run_dir))
        print(f"zip: {archive} ({archive.stat().st_size / 1e6:.1f} MB)", flush=True)
    return summary


def write_zip(run_dir: Path) -> Path:
    """<run_dir>.zip with every file of the run except the trained models (large, not needed for analysis)."""
    archive = run_dir.with_name(run_dir.name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(run_dir.rglob("*")):
            relative = path.relative_to(run_dir)
            if path.is_file() and relative.parts[0] != "models":
                handle.write(path, Path(run_dir.name) / relative)
        for path in sorted((run_dir / "models").glob("member_*/record.json")):
            handle.write(path, Path(run_dir.name) / path.relative_to(run_dir))   # histories, seeds, times
    return archive


if __name__ == "__main__":
    main()
