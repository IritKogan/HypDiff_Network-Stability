from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class PilotConfig:
    curvature: float = 1.0
    tau: float = 0.5
    # "exact" inverts q_sample; "upstream" = original HypDiff formula.
    noise_formula: str = "exact"

    def __post_init__(self) -> None:
        if not math.isclose(self.curvature, 1.0):
            raise ValueError("the partial-recovery pilot currently requires curvature=1.0")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        if self.noise_formula not in ("exact", "upstream"):
            raise ValueError("noise_formula must be 'exact' or 'upstream'")

    def apply_to_args(self, args: Any) -> Any:
        args.c = self.curvature
        args.pilot_curvature = self.curvature
        args.noise_formula = self.noise_formula
        return args
