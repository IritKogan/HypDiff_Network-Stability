from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stability.cora_edge_recovery import RECONSTRUCTIONS, run_cora_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Cora cumulative random and targeted edge recovery."
    )
    parser.add_argument("--k", type=int, default=RECONSTRUCTIONS)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the Planetoid Cora dataset if it is not cached",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new run directory (defaults to a timestamped Cora artifact directory)",
    )
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        ROOT / "artifacts" / "cora_edge_recovery" / f"run_{stamp}"
    )
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside the repository root")
    summary = run_cora_experiment(
        output_dir=output_dir,
        reconstruction_count=args.k,
        tau=args.tau,
        download=args.download,
    )
    print(f"Saved Cora edge-level results: {summary}")


if __name__ == "__main__":
    main()
