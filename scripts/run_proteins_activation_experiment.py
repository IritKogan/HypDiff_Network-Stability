from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stability.proteins_activation_experiment import run_experiment


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = (
    ROOT
    / "artifacts"
    / "proteins_pilot"
    / "calibration_stable_clustering"
    / "calibration.json"
)
OUTPUT_DIR = ROOT / "artifacts" / "proteins_pilot" / "activation_development"


def main():
    if not CALIBRATION.is_file():
        raise FileNotFoundError(f"saved calibration artifact not found: {CALIBRATION}")
    result = run_experiment(CALIBRATION, OUTPUT_DIR)
    if result["status"] == "complete":
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destination = OUTPUT_DIR / "experiment.json"
        destination.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
