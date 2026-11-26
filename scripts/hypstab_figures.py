"""Re-draw the figures of a finished "probabilities" run (no model, no GPU needed).

Reads <run_dir>/config.toml (the config saved with the run) and <run_dir>/results/noise__<attack>.npz,
writes <run_dir>/figures/*.png (hypstab/experiment/probabilities.py: plot_run).

  python scripts/hypstab_figures.py --run-dir artifacts/hypstab_runs/cora_k100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hypstab.config import load_config  # noqa: E402
from hypstab.experiment.probabilities import plot_run  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.toml")
    if config.experiment.output != "probabilities":
        raise SystemExit(f"{run_dir}: experiment.output = {config.experiment.output!r}, expected 'probabilities'")
    for path in plot_run(run_dir, config):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
