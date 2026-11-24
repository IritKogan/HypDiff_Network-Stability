"""Hyperbolic anisotropic diffusion used for edge recovery (book section 2 + Decision 8).

Recovery of a damaged graph (embeddings h of the damaged graph, its adjacency A):

1. Tangent coordinates at the community centre mu_i of every node (book Eq. 3):
       v_i = logmap_{mu_i}(h_i),   x_0 = (v - mean(v)) / std(v)     (one scalar mean / std)
2. Anisotropic forward noise up to step t (book / paper Eq. 5-6):
       eps_i = sgn(logmap_0(mu_i)) * e_i          angular_noise = "upstream" (paper Eq. 5, code)
       eps_i = sgn(logmap_0(mu_i)) * |e_i|        angular_noise = "folded"   (paper App. Eq. C.1)
       x_t   = a_t x_0 + s_t eps                  (schedule.py; e ~ N(0, I))
   Note: "upstream" is a sign flip of a symmetric Gaussian, i.e. still isotropic N(0, I);
   only "folded" restricts the noise to the community direction's orthant.
3. The denoiser predicts x_0 from (x_t, A, t / T); optionally clipped to [-1, 1] (clip_x0).
4. Back to the ball: h' = expmap_{mu_i}( x_0 * std(v) + mean(v) ).

Samplers:
- "one_step" (Decision 8): x_0 is predicted directly from x_{t*}, the same single step the
  network is trained on.
- "iterative" (book: iterative reverse process; Stage 4b): ``steps`` network calls on the
  strided steps t* = t_1 > t_2 > ... > t_n = 0 (``sampling_steps``). At each step
      x0_hat  = f(x_t, A, t / T)                         (clipped if clip_x0)
      eps_hat = (x_t - a_t x0_hat) / s_t                 (exact inverse of q_sample, Decision 5)
      x_prev  = a_prev x0_hat + s_prev eps_hat           (deterministic, DDIM eta = 0)
  and the prediction at t = 0 is the result. With steps = t* + 1 this is the old code's
  "iterative" sampler (every step). HypDiff's original noise formula ignores the radial
  term of a_t and is not offered (found wrong, Decision 5).
- Training "unrolled" (experiment E2): the same chain from t*, differentiable, loss on the
  final output (hypstab/training/denoiser.py).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from hypstab.diffusion.denoiser import ResidualGCNDenoiser
from hypstab.diffusion.schedule import DiffusionSchedule
from hypstab.geometry import poincare as pb
from hypstab.models.clamp import standardize

ANGULAR_NOISE = ("upstream", "folded")
SAMPLERS = ("one_step", "iterative")


def sampling_steps(t_star: int, steps: int) -> list[int]:
    """Strided steps from t* down to 0 (both included), at most ``steps`` of them, distinct.

    E.g. t* = 35, steps = 8 -> [35, 30, 25, 20, 15, 10, 5, 0]; steps = t* + 1 -> every step."""
    if steps < 2:
        raise ValueError("an iterative chain needs steps >= 2")
    grid = np.unique(np.round(np.linspace(t_star, 0, steps)).astype(int))[::-1]
    return [int(t) for t in grid]


class HyperbolicDiffusion(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        layers: int,
        schedule: DiffusionSchedule,
        curvature: float,
        angular_noise: str,
        clip_x0: bool,
    ):
        super().__init__()
        if angular_noise not in ANGULAR_NOISE:
            raise ValueError(f"angular_noise must be one of {ANGULAR_NOISE}")
        self.net = ResidualGCNDenoiser(dim, hidden_dim, layers)
        self.schedule = schedule
        self.c = curvature
        self.angular_noise = angular_noise
        self.clip_x0 = clip_x0

    # -- pieces -----------------------------------------------------------------------

    def tangent_coordinates(self, embeddings: Tensor, labels: Tensor, centers: Tensor):
        """Standardised tangent coordinates at the community centres + what is needed to map back."""
        node_centers = centers.to(embeddings)[labels]
        tangent = pb.logmap(node_centers, embeddings, self.c)
        x0, mean, std = standardize(tangent)
        if not torch.isfinite(x0).all():
            raise ValueError("non-finite standardised tangent coordinates")
        return x0, mean, std, node_centers

    def to_ball(self, x0: Tensor, mean: Tensor, std: Tensor, node_centers: Tensor) -> Tensor:
        return pb.expmap(node_centers, x0 * std + mean, self.c)

    def angular_noise_sample(self, like: Tensor, labels: Tensor, centers: Tensor,
                             generator: torch.Generator | None) -> Tensor:
        e = torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=generator)
        signs = torch.sign(pb.logmap0(centers.to(like), self.c))[labels]
        return (e.abs() if self.angular_noise == "folded" else e) * signs

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        a, s = self.schedule.coefficients(t, x0.dtype)
        return a * x0 + s * noise

    def predict_x0(self, x_t: Tensor, adjacency: Tensor, t: Tensor) -> Tensor:
        x0 = self.net(x_t, adjacency, t.to(x_t.dtype) / self.schedule.timesteps)
        return x0.clamp(-1.0, 1.0) if self.clip_x0 else x0

    # -- training and recovery --------------------------------------------------------

    def denoise_once(self, embeddings: Tensor, adjacency: Tensor, labels: Tensor, centers: Tensor,
                     t: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """One differentiable noise + denoise step at per-node steps t (n,); returns ball points."""
        x0, mean, std, node_centers = self.tangent_coordinates(embeddings, labels, centers)
        x_t = self.q_sample(x0, t, self.angular_noise_sample(x0, labels, centers, generator))
        return self.to_ball(self.predict_x0(x_t, adjacency, t), mean, std, node_centers)

    def reverse_chain(self, x_t: Tensor, adjacency: Tensor, steps: list[int]) -> Tensor:
        """Deterministic strided reverse chain over ``steps`` (descending, ending at 0); returns x0."""
        if steps[-1] != 0 or any(a <= b for a, b in zip(steps, steps[1:])):
            raise ValueError("steps must be strictly decreasing and end at 0")
        n = x_t.shape[0]
        for i, t in enumerate(steps):
            t_vec = torch.full((n,), t, dtype=torch.long, device=x_t.device)
            x0 = self.predict_x0(x_t, adjacency, t_vec)
            if t == 0:
                return x0
            a_t, s_t = self.schedule.coefficients(t_vec, x_t.dtype)
            prev = torch.full((n,), steps[i + 1], dtype=torch.long, device=x_t.device)
            a_p, s_p = self.schedule.coefficients(prev, x_t.dtype)
            eps_hat = (x_t - a_t * x0) / s_t
            x_t = a_p * x0 + s_p * eps_hat
        raise AssertionError("unreachable")

    def denoise_chain(self, embeddings: Tensor, adjacency: Tensor, labels: Tensor, centers: Tensor,
                      steps: list[int], generator: torch.Generator | None = None) -> Tensor:
        """Noise to steps[0] = t*, run the reverse chain, return ball points (differentiable)."""
        x0, mean, std, node_centers = self.tangent_coordinates(embeddings, labels, centers)
        t = torch.full((x0.shape[0],), steps[0], dtype=torch.long, device=x0.device)
        x_t = self.q_sample(x0, t, self.angular_noise_sample(x0, labels, centers, generator))
        return self.to_ball(self.reverse_chain(x_t, adjacency, steps), mean, std, node_centers)

    def recover(self, embeddings: Tensor, adjacency: Tensor, labels: Tensor, centers: Tensor,
                t_star: int, generator: torch.Generator | None, sampler: str = "one_step",
                steps: int = 2) -> Tensor:
        """Recovery from t* with the given sampler ("one_step" | "iterative" with ``steps`` calls)."""
        self.schedule.check_t_star(t_star)
        if sampler == "one_step":
            t = torch.full((embeddings.shape[0],), int(t_star), dtype=torch.long, device=embeddings.device)
            return self.denoise_once(embeddings, adjacency, labels, centers, t, generator)
        if sampler == "iterative":
            return self.denoise_chain(embeddings, adjacency, labels, centers,
                                      sampling_steps(int(t_star), steps), generator)
        raise ValueError(f"sampler must be one of {SAMPLERS}")
