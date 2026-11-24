"""Closed-form HypDiff forward-process coefficients (no torch, no upstream import).

HypDiff's forward process (upstream ``models/direction_diffusion.py``, used unchanged by
``StabilityDiffusion.q_sample``) is

    x_t = a_t * x0 + s_t * eps,   a_t = sqrt(abar_t) + delta * tanh(sqrt(c) * t / 500),
                                  s_t = sqrt(1 - abar_t),

with the linear beta schedule of upstream ``get_alphas(T)``: betas = linspace(1e-4, 0.02, T)
scaled by 1000 / T. The radial prior delta * tanh(.) uses the RAW step t (t / 500 = the paper's
lambda * t / T0 with lambda = 2, T0 = 1000), independent of T.

a_t is not monotone: the radial prior first makes it RISE slightly (by design: popularity pushes
nodes outward), then the sqrt(abar_t) decay wins and a_t falls to a minimum; with a large delta it
rises again after the minimum. Past that minimum the signal coefficient grows while the noise also
grows, so a reverse process started there is ill-posed (Findings about HypDiff in project_context.md).

``last_valid_t_star`` = position of the minimum of a_t: the largest t* the recovery may start from.
  T=200,  delta=1   -> 154 (upstream code values; previous runs used t*=7)
  T=1000, delta=0.4 -> 999 (paper values: every t* is valid)
  T=1000, delta=1   -> 813
"""

from __future__ import annotations

import math

import numpy as np

RADIAL_SCALE = 500.0  # upstream getpri(): tanh(sqrt(c) * t / 500)


def betas(timesteps: int) -> np.ndarray:
    """Upstream get_alphas(T) linear schedule (float64)."""
    if timesteps < 2:
        raise ValueError("timesteps must be >= 2")
    scale = 1000.0 / timesteps
    return np.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=np.float64)


def noise_coefficients(timesteps: int) -> np.ndarray:
    """s_t = sqrt(1 - abar_t), t = 0 .. T-1."""
    return np.sqrt(1.0 - np.cumprod(1.0 - betas(timesteps)))


def signal_coefficients(timesteps: int, radial_delta: float, curvature: float = 1.0) -> np.ndarray:
    """a_t = sqrt(abar_t) + delta * tanh(sqrt(c) * t / 500), t = 0 .. T-1."""
    t = np.arange(timesteps, dtype=np.float64)
    radial = radial_delta * np.tanh(math.sqrt(curvature) * t / RADIAL_SCALE)
    return np.sqrt(np.cumprod(1.0 - betas(timesteps))) + radial


def last_valid_t_star(timesteps: int, radial_delta: float, curvature: float = 1.0) -> int:
    """Position of the minimum of a_t on [0, T): t* above it is rejected (see module docstring)."""
    return int(np.argmin(signal_coefficients(timesteps, radial_delta, curvature)))


def check_t_star(t_star: int, timesteps: int, radial_delta: float, curvature: float = 1.0) -> None:
    """Raise ValueError if t* is outside [0, T) or past the minimum of a_t."""
    if not 0 <= t_star < timesteps:
        raise ValueError(f"t_star must be in [0, {timesteps - 1}] (timesteps = {timesteps}), got {t_star}")
    limit = last_valid_t_star(timesteps, radial_delta, curvature)
    if t_star > limit:
        raise ValueError(
            f"t_star = {t_star} is past the minimum of the signal coefficient a_t (t = {limit} for "
            f"timesteps = {timesteps}, radial_delta = {radial_delta}): a_t rises again there, so the "
            f"reverse process is ill-posed. Use t_star <= {limit}, more timesteps or a smaller radial_delta."
        )
