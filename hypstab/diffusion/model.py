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

Sampler "one_step" (Decision 8): x_0 is predicted directly from x_{t*}, the same single
step the network is trained on. The iterative reverse chain (book; experiment E2) is
added in Stage 4b.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hypstab.diffusion.denoiser import ResidualGCNDenoiser
from hypstab.diffusion.schedule import DiffusionSchedule
from hypstab.geometry import poincare as pb
from hypstab.models.clamp import standardize

ANGULAR_NOISE = ("upstream", "folded")


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

    def recover(self, embeddings: Tensor, adjacency: Tensor, labels: Tensor, centers: Tensor,
                t_star: int, generator: torch.Generator | None) -> Tensor:
        """Recovery from t* (sampler "one_step")."""
        self.schedule.check_t_star(t_star)
        t = torch.full((embeddings.shape[0],), int(t_star), dtype=torch.long, device=embeddings.device)
        return self.denoise_once(embeddings, adjacency, labels, centers, t, generator)
