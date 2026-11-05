from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stability.sbm_ba_recovery_pilot import run_pilot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new, unused artifact directory (defaults to the fixed pilot directory)",
    )
    parser.add_argument(
        "--reuse-checkpoint-root",
        type=Path,
        help="optional prior pilot models directory with matching graph checkpoints",
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        if not output_dir.is_relative_to(ROOT):
            parser.error("--output-dir must be inside the repository root")
        from stability import sbm_ba_recovery_pilot

        sbm_ba_recovery_pilot.OUTPUT_DIR = output_dir
    if args.reuse_checkpoint_root is not None:
        from stability import sbm_ba_recovery_pilot

        sbm_ba_recovery_pilot.REUSE_CHECKPOINT_ROOT = (
            args.reuse_checkpoint_root.resolve()
        )
    print(json.dumps({"status": run_pilot()}, indent=2))
