"""Run a config-driven edge-recovery stability experiment.

Example:
    python scripts/run_experiment.py --config configs/cora.toml
    python scripts/run_experiment.py --config configs/cora.toml \
        --output-root /content/drive/MyDrive/hypdiff_runs --device cuda
    # fixed run directory, continued after an interruption (finished models are loaded):
    python scripts/run_experiment.py --config configs/cora.toml --run-dir artifacts/runs/cora_main --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stability.experiment_config import config_from_dict  # noqa: E402
from stability.edge_recovery_experiment import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", help="override run.output_root (recorded in config.json)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="override run.device")
    parser.add_argument("--log-every", type=int, default=100, help="progress print interval (epochs)")
    parser.add_argument("--run-dir", type=Path, help="exact run directory (default: <output_root>/<name>_<time>)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing --run-dir with the same config (finished models are loaded)")
    args = parser.parse_args()

    config_text = args.config.read_text(encoding="utf-8")
    config = config_from_dict(tomllib.loads(config_text))
    overrides = {}
    if args.output_root:
        overrides["output_root"] = args.output_root
    if args.device:
        overrides["device"] = args.device
    if overrides:
        config = config.with_overrides(run=overrides)
    if args.resume and args.run_dir is None:
        parser.error("--resume needs --run-dir")
    run_dir = run_experiment(config, config_text=config_text, log_every=args.log_every,
                             output_dir=args.run_dir, resume=args.resume)
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
