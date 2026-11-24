"""Noise schedule of HypDiff's forward process (paper Eq. 6, book section 2).

In the standardised tangent space at each node's community centre:

    x_t = a_t * x_0 + s_t * eps,   a_t = sqrt(abar_t) + delta * tanh(sqrt(c) * t / 500),
                                   s_t = sqrt(1 - abar_t)

with a linear beta schedule from 1e-4 to 0.02 scaled by 1000 / T (as in the HypDiff
code), abar_t = prod_{s <= t} (1 - beta_s), t = 0 .. T-1. The radial term
delta * tanh(t / 500) is the paper's lambda_o * t / T0 with lambda_o = 2, T0 = 1000.

a_t is not monotone: it rises slightly first, then falls to a minimum and (for a
large delta) rises again. Past the minimum the reverse process is ill-posed, so
t* must not lie beyond it (``last_valid_t_star``; project_context Decision 4).
T = 200 / delta = 1 (old code): minimum at t = 154. T = 1000 / delta = 0.4 (paper):
at t = 999 (every t* valid).
"""

from __future__ import annotations

import math

import numpy as np
import torch

RADIAL_SCALE = 500.0


class DiffusionSchedule:
    def __init__(self, timesteps: int, radial_delta: float, curvature: float):
        if timesteps < 2:
            raise ValueError("timesteps must be >= 2")
        if radial_delta < 0:
            raise ValueError("radial_delta must be >= 0")
        self.timesteps = int(timesteps)
        self.radial_delta = float(radial_delta)
        self.curvature = float(curvature)
        scale = 1000.0 / timesteps
        self.betas = np.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=np.float64)
        if self.betas[-1] >= 1:
            raise ValueError(f"timesteps = {timesteps} gives beta >= 1")
        abar = np.cumprod(1.0 - self.betas)
        t = np.arange(timesteps, dtype=np.float64)
        self.radial = radial_delta * np.tanh(math.sqrt(curvature) * t / RADIAL_SCALE)
        self.signal = np.sqrt(abar) + self.radial          # a_t
        self.noise = np.sqrt(1.0 - abar)                     # s_t

    def last_valid_t_star(self) -> int:
        return int(np.argmin(self.signal))

    def check_t_star(self, t_star: int) -> None:
        if not 0 <= t_star < self.timesteps:
            raise ValueError(f"t_star must be in [0, {self.timesteps - 1}], got {t_star}")
        limit = self.last_valid_t_star()
        if t_star > limit:
            raise ValueError(
                f"t_star = {t_star} is past the minimum of a_t (t = {limit} for T = {self.timesteps}, "
                f"delta = {self.radial_delta}); the reverse process is ill-posed there"
            )

    def coefficients(self, t: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """(a_t, s_t) for integer steps t of shape (n,), returned with shape (n, 1)."""
        index = t.detach().cpu().numpy().astype(np.int64)
        if index.min() < 0 or index.max() >= self.timesteps:
            raise ValueError("diffusion step outside [0, T)")
        a = torch.as_tensor(self.signal[index], dtype=dtype, device=t.device).reshape(-1, 1)
        s = torch.as_tensor(self.noise[index], dtype=dtype, device=t.device).reshape(-1, 1)
        return a, s
