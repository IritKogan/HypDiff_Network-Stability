"""Project subclasses of upstream HypDiff models.

Upstream HypDiff (RingBDStack/HypDiff @ 1620e3cc6ac7a558055e9888257568986765f082)
is kept byte-identical in ``models/``. Every project change to model code lives
here, as a subclass, so that the diff against upstream is explicit:

``StabilityDiffusion(models.direction_diffusion.diffusion)``
    * ``__init__``   documented copy of upstream ``__init__`` with ``.cuda()``
      replaced by ``.to(device)`` (upstream crashes on CPU-only machines) and
      curvature taken as a parameter instead of the hard-coded ``self.c = 1.0``.
    * ``q_sample``   documented copy, ``.cuda()`` -> ``.to(x_start.device)``.
    * ``cal_u0`` / ``forward``  documented copies using ``self.c`` instead of the
      hard-coded ``c=1.0`` (identical behaviour at c = 1).
    * ``predict_noise_from_start``  selectable via ``noise_formula``:
      ``"exact"`` inverts ``q_sample`` exactly; ``"upstream"`` calls the original
      upstream method unchanged. See tests/test_predict_noise_from_start.py.
    * ``tran_direction``  vectorised copy of the upstream per-node loop
      (same result; upstream loops over nodes in Python). Option ``angular_noise``:
      "upstream" = sgn(direction) * eps (paper Eq. 5 and upstream code; NOTE: a sign
      flip of a symmetric Gaussian, so the noise is still isotropic N(0, I)), or
      "folded" = sgn(direction) * |eps| (DDM / paper Appendix Eq. C.1, the form under
      which Theorem 3.2's folded-normal limit holds: noise confined to the community
      direction's orthant).
    * ``draw_noise`` / ``_denoise_from_tstar`` / ``sample_from_tstar``  new
      project methods for partial-noise recovery from t* < T.
    * ``radial_delta`` (paper's delta, upstream hard-codes 1),
      ``denoiser_architecture`` ("upstream_gcn" | "residual_gcn"), ``clip_x0``
      ``restoration_loss`` (damage-aware training), ``denoise_once``
      (edge-reconstruction training) and ``recovery_sampler`` ("iterative" =
      t* .. 0 reverse loop, "one_step" = x0 predicted directly from x_{t*}) — options read
      from ``args``; defaults reproduce the upstream behaviour.
    * Decision 9 (conditional geometric diffusion): conditional architectures
      ("conditional_gcn" | "conditional_residual_gcn") receive the standardised damaged
      embedding as an extra input (``_predict_x0``); ``conditional_loss`` trains on the
      noised CLEAN train-graph embedding; sampler "stochastic" is a generalised DDIM over
      HypDiff's marginals with stochasticity ``recovery_eta`` and ``recovery_steps``
      strided steps, adding FRESH direction-restricted noise at every step
      (``stochastic_step``). eta = 0 reproduces the "iterative" sampler exactly.

    Upstream methods not used by the project (``p_sample``, ``p_sample_loop``,
    ``x_tran``) are inherited unchanged and still assume c = 1.

``StabilityLPModel(models.base_models.LPModel)``
    Adds the optional ``args.final_activation`` ("relu" = upstream default,
    or "leaky_relu" for the last HGCN layer). Activations carry no parameters,
    so swapping after construction yields the same initial weights and RNG
    stream as upstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
from manifolds.poincare import PoincareBall
from models.base_models import LPModel
from models.direction_diffusion import diffusion, gcndec, get_alphas, mlp
import math

from stability.denoisers import (
    ConditionalGCNDenoiser,
    ConditionalResidualGCNDenoiser,
    ResidualGCNDenoiser,
)

NOISE_FORMULAS = ("exact", "upstream")
DENOISER_ARCHITECTURES = ("upstream_gcn", "residual_gcn", "conditional_gcn", "conditional_residual_gcn")
SAMPLERS = ("iterative", "one_step", "stochastic")
ANGULAR_NOISE = ("upstream", "folded")
# E|z| for z ~ N(0, 1): mean of HypDiff's direction-restricted noise |z| * sign(direction).
RESTRICTED_NOISE_MEAN = math.sqrt(2.0 / math.pi)
FINAL_ACTIVATIONS = ("relu", "leaky_relu")


class StabilityDiffusion(diffusion):
    def __init__(self, args, dim, curvature=1.0, noise_formula=None, device=None):
        # Copy of upstream diffusion.__init__ (see module docstring). The
        # construction order of submodules is unchanged, so parameter
        # initialisation and state_dict keys match upstream checkpoints.
        nn.Module.__init__(self)
        # Schedule tensors are plain attributes (upstream), so Module.to() does not
        # move them: the device must be chosen at construction.
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        self.device = device
        # Radial popularity coefficient delta of the paper's Eq. (6); upstream hard-codes 1
        # (paper experiments: 0.4). getpri() = delta * tanh(sqrt(c) * t / 500), and
        # t / 500 equals the paper's lambda_o * t / T0 with lambda_o = 2, T0 = 1000.
        self.prior = float(getattr(args, "radial_delta", 1.0))
        if self.prior < 0:
            raise ValueError("radial_delta must be non-negative")
        self.c = float(curvature)
        if self.c <= 0:
            raise ValueError("curvature must be positive")
        if noise_formula is None:
            noise_formula = getattr(args, "noise_formula", "exact")
        if noise_formula not in NOISE_FORMULAS:
            raise ValueError(f"noise_formula must be one of {NOISE_FORMULAS}")
        self.noise_formula = noise_formula
        self.gcn = gcndec(dim)
        self.gcn.to(device)
        self.mlp = mlp(dim)
        self.mlp = self.mlp.to(device)
        self.num_timesteps = 200
        self.manifold = PoincareBall()
        self.alphas, betas = get_alphas(self.num_timesteps)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
            1. - self.alphas_cumprod)
        for name in (
            "alphas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
            "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
            "posterior_variance", "minus_alphas_cumprod",
            "posterior_log_variance_clipped", "posterior_mean_coef1",
            "posterior_mean_coef2",
        ):
            setattr(self, name, getattr(self, name).to(device))
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        # Project options (defaults = upstream behaviour).
        self.clip_x0 = bool(getattr(args, "clip_x0", True))
        self.sampler = getattr(args, "recovery_sampler", "iterative")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"recovery_sampler must be one of {SAMPLERS}")
        self.angular_noise = getattr(args, "angular_noise", "upstream")
        if self.angular_noise not in ANGULAR_NOISE:
            raise ValueError(f"angular_noise must be one of {ANGULAR_NOISE}")
        self.eta = float(getattr(args, "recovery_eta", 0.0))
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("recovery_eta must be in [0, 1]")
        steps = getattr(args, "recovery_steps", None)
        self.sampler_steps = None if steps is None else int(steps)
        if self.sampler_steps is not None and self.sampler_steps < 1:
            raise ValueError("recovery_steps must be positive")
        self.architecture = getattr(args, "denoiser_architecture", "upstream_gcn")
        if self.architecture not in DENOISER_ARCHITECTURES:
            raise ValueError(f"denoiser_architecture must be one of {DENOISER_ARCHITECTURES}")
        if self.architecture == "residual_gcn":
            self.gcn = ResidualGCNDenoiser(
                dim, int(args.denoiser_hidden_dim), int(args.denoiser_layers)
            ).to(device)
        elif self.architecture == "conditional_gcn":
            self.gcn = ConditionalGCNDenoiser(dim).to(device)
        elif self.architecture == "conditional_residual_gcn":
            self.gcn = ConditionalResidualGCNDenoiser(
                dim, int(args.denoiser_hidden_dim), int(args.denoiser_layers)
            ).to(device)
        self.conditional = bool(getattr(self.gcn, "conditional", False))

    # -- documented copies of upstream methods (self.c / device fixes) ------

    @staticmethod
    def _stack(vectors, like):
        if isinstance(vectors, torch.Tensor):
            return vectors.to(device=like.device)
        return torch.stack([torch.as_tensor(v) for v in vectors]).to(device=like.device)

    @staticmethod
    def _labels(labels, like):
        return torch.as_tensor(labels, dtype=torch.long).to(device=like.device)

    def tran_direction(self, direction_vector, gaussian_point, labels):
        # Vectorised upstream loop: gaussian_point[i] *= sign(direction[labels[i]]),
        # updated in place like upstream. "folded": |gaussian_point| * sign (DDM, Eq. C.1).
        signs = torch.sign(self._stack(direction_vector, gaussian_point)[
            self._labels(labels, gaussian_point)
        ])
        magnitude = gaussian_point.abs() if self.angular_noise == "folded" else gaussian_point
        gaussian_point.copy_(magnitude * signs)
        return gaussian_point

    def cal_u0(self, data):
        u = []
        for x in data:
            x = self.manifold.logmap0(x, c=self.c)
            u0 = torch.mean(x, dim=0)
            u0 = self.manifold.expmap0(u0, c=self.c)
            u.append(u0)
        return u

    def q_sample(self, x_start, t, direction, noise, labels, u0, restrict=False):
        noise = self.default(noise, lambda: torch.randn_like(x_start))
        if restrict:
            noise = self.tran_direction(direction, noise, labels)
        alphas_cumprod, minus_alphas_cumprod = self.get_alphas2(self.num_timesteps)
        alphas_cumprod = alphas_cumprod.to(x_start.device)
        minus_alphas_cumprod = minus_alphas_cumprod.to(x_start.device)
        pri_cumprod = self.getpri(t)
        return (
            self.extract(alphas_cumprod, t, x_start.shape) * x_start
            + self.extract(minus_alphas_cumprod, t, x_start.shape) * noise
            + pri_cumprod * x_start
        )

    def forward(self, x, adj, labels, u0, restrict=True, graphset=False):
        if self.conditional:
            raise ValueError("generative (upstream) training has no condition; use training='conditional'")
        x = x.to(float)
        h = x
        b, n = h.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=h.device).long()
        noise = torch.randn_like(h)
        direction = []
        for u in u0:
            direction.append(self.manifold.logmap0(u, c=self.c))
        if not restrict:
            x_start = self.manifold.logmap0(x, c=self.c)
        else:
            # Vectorised upstream loop x_start[i] = logmap(u0[labels[i]], x[i]).
            centers = self._stack(u0, x).to(dtype=x.dtype)[self._labels(labels, x)]
            x_start = self.manifold.logmap(centers, x, c=self.c)
        x_start = self.normalization(x_start)
        xt = self.q_sample(x_start=x_start, t=t, direction=direction, noise=noise,
                           labels=labels, u0=u0, restrict=True)
        t = t / self.num_timesteps
        if graphset:
            out = self.mlp(xt, t)
        else:
            edge_index = self.edge(adj)
            out = self.gcn(xt, edge_index, t)
        x_start = x_start.to(torch.float32)
        return F.mse_loss(out, x_start)

    # -- noise prediction ---------------------------------------------------

    def predict_noise_from_start(self, x_t, t, x0):
        if self.noise_formula == "upstream":
            return super().predict_noise_from_start(x_t, t, x0)
        # Exact inverse of q_sample:
        # x_t = (sqrt(abar_t) + prior(t)) * x0 + sqrt(1 - abar_t) * eps
        prior = self.getpri(t)
        sqrt_abar = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_abar = self.extract(self.minus_alphas_cumprod, t, x_t.shape)
        return (x_t - (sqrt_abar + prior) * x0) / sqrt_one_minus_abar

    # -- project methods: partial-noise recovery from t* --------------------

    def draw_noise(self, x_start, generator):
        return torch.randn(
            x_start.shape,
            dtype=x_start.dtype,
            device=x_start.device,
            generator=generator,
        )

    def _denoise_from_tstar(
        self, x_start, edge_index, direction, labels, t_star, generator
    ):
        if not 0 <= t_star < self.num_timesteps:
            raise ValueError(
                f"t_star must be in [0, {self.num_timesteps - 1}]"
            )
        noise = self.draw_noise(x_start, generator)
        x_t = self.q_sample(
            x_start=x_start,
            t=torch.full(
                (x_start.shape[0],), t_star, device=x_start.device, dtype=torch.long
            ),
            direction=direction,
            noise=noise,
            labels=labels,
            u0=None,
            restrict=True,
        )

        if self.sampler == "one_step":
            # Predict x0 directly from x_{t*} (the network is an x0-predictor); matches
            # the single-step edge-reconstruction training objective.
            batch_t = torch.full((x_t.shape[0],), t_star, device=x_t.device, dtype=torch.long)
            x0 = self._predict_x0(x_t, edge_index,
                                  batch_t.to(dtype=torch.float32) / self.num_timesteps, x_start)
            return x0.clamp(-1.0, 1.0) if self.clip_x0 else x0

        if self.sampler == "stochastic":
            return self._stochastic_reverse(x_t, x_start, edge_index, direction, labels,
                                            t_star, generator)

        for timestep in range(t_star, -1, -1):
            batch_t = torch.full(
                (x_t.shape[0],), timestep, device=x_t.device, dtype=torch.long
            )
            # Network was trained on t / num_timesteps (see forward).
            normalized_t = batch_t.to(dtype=torch.float32) / self.num_timesteps
            x0 = self._predict_x0(x_t, edge_index, normalized_t, x_start)
            if self.clip_x0:  # upstream clip_denoised; x0 is standardised (std 1)
                x0 = x0.clamp(-1.0, 1.0)
            if timestep == 0:
                return x0

            noise = self.predict_noise_from_start(x_t, batch_t, x0)
            previous_t = torch.full_like(batch_t, timestep - 1)
            x_t = self.q_sample(
                x_start=x0,
                t=previous_t,
                direction=direction,
                noise=noise,
                labels=labels,
                u0=None,
                restrict=False,
            )
        raise RuntimeError("reverse diffusion loop did not produce a clean endpoint")

    def shrink_to_prototype(self, damaged_embeddings, labels, prototypes, clip=1.0):
        """Ablation baseline (no network, no noise): the clamp step of the recovery
        alone — standardised tangent coordinates at the community prototype are
        clipped to [-clip, clip] and mapped back."""
        labels = labels.to(device=damaged_embeddings.device, dtype=torch.long)
        prototypes = prototypes.to(device=damaged_embeddings.device, dtype=damaged_embeddings.dtype)
        node_prototypes = prototypes[labels]
        tangent = self.manifold.logmap(node_prototypes, damaged_embeddings, c=self.c)
        clipped = self.normalization(tangent).clamp(-clip, clip)
        return self.manifold.expmap(self.renormal(clipped, tangent), node_prototypes, c=self.c)

    def denoise_once(self, damaged_embeddings, damaged_adjacency, labels, prototypes, t):
        """Differentiable single recovery step used for edge-reconstruction training.

        Same geometry as sample_from_tstar (tangent space at the community prototype,
        standardised with the damaged statistics, anisotropic noise to step t), but
        only one network call: x0 = f(x_t, A, t/T), optionally clamped, mapped back
        to the Poincare ball. ``t`` is a LongTensor with one step per node.
        """
        labels = labels.to(device=damaged_embeddings.device, dtype=torch.long)
        prototypes = prototypes.to(device=damaged_embeddings.device, dtype=damaged_embeddings.dtype)
        node_prototypes = prototypes[labels]
        tangent = self.manifold.logmap(node_prototypes, damaged_embeddings, c=self.c)
        normalized = self.normalization(tangent)
        direction = torch.stack([self.manifold.logmap0(p, c=self.c) for p in prototypes])
        noise = torch.randn_like(normalized)
        x_t = self.q_sample(normalized, t, direction, noise, labels, None, restrict=True)
        x0 = self._predict_x0(x_t, self.edge(damaged_adjacency.coalesce()),
                              t / self.num_timesteps, normalized)
        if self.clip_x0:
            x0 = x0.clamp(-1.0, 1.0)
        recovered_tangent = self.renormal(x0.to(tangent.dtype), tangent)
        return self.manifold.expmap(recovered_tangent, node_prototypes, c=self.c)

    def restoration_loss(self, clean_embeddings, damaged_embeddings, damaged_adjacency,
                         labels, prototypes, t_max):
        """Damage-aware training objective (project).

        Input: embeddings of an edge-dropped training graph, noised to a random
        t < t_max exactly as in sample_from_tstar. Target: the embeddings of the
        intact training graph. Both are expressed in the tangent space of each
        node's community prototype and standardised with the DAMAGED statistics,
        the only ones available at recovery time (sample_from_tstar renormalises
        with them), so the prediction lands directly on the clean embedding.
        """
        labels = labels.to(device=damaged_embeddings.device, dtype=torch.long)
        prototypes = prototypes.to(device=damaged_embeddings.device, dtype=damaged_embeddings.dtype)
        node_prototypes = prototypes[labels]
        tangent_damaged = self.manifold.logmap(node_prototypes, damaged_embeddings, c=self.c)
        tangent_clean = self.manifold.logmap(node_prototypes, clean_embeddings, c=self.c)
        mean, std = torch.mean(tangent_damaged), torch.std(tangent_damaged)
        x_damaged = (tangent_damaged - mean) / std
        target = (tangent_clean - mean) / std
        direction = torch.stack([self.manifold.logmap0(p, c=self.c) for p in prototypes])
        t = torch.randint(0, t_max, (x_damaged.shape[0],), device=x_damaged.device).long()
        noise = torch.randn_like(x_damaged)
        x_t = self.q_sample(x_damaged, t, direction, noise, labels, None, restrict=True)
        prediction = self._predict_x0(x_t, self.edge(damaged_adjacency.coalesce()),
                                      t / self.num_timesteps, x_damaged)
        return F.mse_loss(prediction, target.to(torch.float32))

    def conditional_loss(self, clean_embeddings, damaged_embeddings, damaged_adjacency,
                         labels, prototypes, t_max, t_sampling="per_graph"):
        """Conditional diffusion objective (Decision 9).

        HypDiff's generative objective (noise the CLEAN embedding with the geometric
        forward process at a random t, predict x0) conditioned on the damaged graph:
        the network sees the damaged edges (message passing) and, for conditional
        architectures, the standardised damaged embedding. Both embeddings are taken in
        the tangent space of each node's community prototype and standardised with the
        DAMAGED statistics (the only ones available at recovery time).

        Returns (mse, recovered_embeddings): the x0 prediction mapped back to the
        Poincare ball (clipped like the sampler) for an optional decoder loss.
        """
        if t_sampling not in ("per_graph", "per_node"):
            raise ValueError("t_sampling must be 'per_graph' or 'per_node'")
        labels = labels.to(device=damaged_embeddings.device, dtype=torch.long)
        prototypes = prototypes.to(device=damaged_embeddings.device, dtype=damaged_embeddings.dtype)
        node_prototypes = prototypes[labels]
        tangent_damaged = self.manifold.logmap(node_prototypes, damaged_embeddings, c=self.c)
        tangent_clean = self.manifold.logmap(node_prototypes, clean_embeddings, c=self.c)
        mean, std = torch.mean(tangent_damaged), torch.std(tangent_damaged)
        condition = (tangent_damaged - mean) / std
        target = (tangent_clean - mean) / std
        direction = torch.stack([self.manifold.logmap0(p, c=self.c) for p in prototypes])
        n = condition.shape[0]
        if t_sampling == "per_graph":
            t = torch.randint(0, t_max, (1,), device=condition.device).long().expand(n).clone()
        else:
            t = torch.randint(0, t_max, (n,), device=condition.device).long()
        noise = torch.randn_like(target)
        x_t = self.q_sample(target, t, direction, noise, labels, None, restrict=True)
        prediction = self._predict_x0(x_t, self.edge(damaged_adjacency.coalesce()),
                                      t / self.num_timesteps, condition)
        mse = F.mse_loss(prediction, target.to(torch.float32))
        x0 = prediction.clamp(-1.0, 1.0) if self.clip_x0 else prediction
        recovered = self.manifold.expmap(x0.to(tangent_damaged.dtype) * std + mean, node_prototypes, c=self.c)
        return mse, recovered

    # -- Decision 9: conditioning and stochastic (generalised DDIM) sampling --

    def _predict_x0(self, x_t, edge_index, normalized_t, condition):
        if self.conditional:
            return self.gcn(x_t, edge_index, normalized_t, condition)
        return self.gcn(x_t, edge_index, normalized_t)

    def signal_coefficient(self, t: int) -> float:
        """a_t in x_t = a_t x0 + s_t eps (sqrt(abar_t) + radial prior; not monotone in t)."""
        return float(self.sqrt_alphas_cumprod[t]) + float(self.getpri(torch.tensor([t]))[0, 0])

    def noise_coefficient(self, t: int) -> float:
        """s_t in x_t = a_t x0 + s_t eps."""
        return float(self.minus_alphas_cumprod[t])

    def stochastic_schedule(self, t_star: int) -> list[int]:
        """Descending timesteps t* .. 0 (strided to ``sampler_steps`` network calls)."""
        if not 0 <= t_star < self.num_timesteps:
            raise ValueError(f"t_star must be in [0, {self.num_timesteps - 1}]")
        if self.sampler_steps is None or self.sampler_steps >= t_star + 1:
            return list(range(t_star, -1, -1))
        points = torch.linspace(t_star, 0, self.sampler_steps).round().long().tolist()
        schedule = sorted(set(points), reverse=True)
        return schedule if schedule[-1] == 0 else schedule + [0]

    def step_variance(self, t: int, t_prev: int) -> float:
        """s_t^2 - (a_t / a_prev)^2 s_prev^2: must be >= 0 for a valid stochastic step.
        HypDiff's radial prior makes a_t rise again at large t, where this turns negative."""
        a_t, a_p = self.signal_coefficient(t), self.signal_coefficient(t_prev)
        s_t, s_p = self.noise_coefficient(t), self.noise_coefficient(t_prev)
        return s_t ** 2 - (a_t / a_p) ** 2 * s_p ** 2

    def check_stochastic_range(self, t_star: int) -> None:
        if self.sampler != "stochastic" or self.eta == 0.0:
            return
        schedule = self.stochastic_schedule(t_star)
        for t, t_prev in zip(schedule[:-1], schedule[1:]):
            if self.step_variance(t, t_prev) < 0:
                raise ValueError(
                    f"stochastic sampler invalid at t={t}->{t_prev} (radial prior makes the signal "
                    f"coefficient rise): use a smaller t_star (last valid single step: "
                    f"{self.last_valid_stochastic_t()}) or a smaller radial_delta"
                )

    def last_valid_stochastic_t(self) -> int:
        last = 0
        for t in range(1, self.num_timesteps):
            if self.step_variance(t, t - 1) < 0:
                break
            last = t
        return last

    def restricted_noise(self, like, direction, labels, generator):
        """Fresh HypDiff angular noise (``tran_direction``, mode ``angular_noise``)."""
        return self.tran_direction(direction, self.draw_noise(like, generator), labels)

    def restricted_noise_mean(self, like, direction, labels):
        """E[noise]: 0 for "upstream" (symmetric), sqrt(2/pi) * sign for "folded"."""
        if self.angular_noise == "upstream":
            return torch.zeros_like(like)
        signs = torch.sign(self._stack(direction, like)[self._labels(labels, like)])
        return (RESTRICTED_NOISE_MEAN * signs).to(dtype=like.dtype)

    def stochastic_step(self, x_t, x0, t: int, t_prev: int, fresh_noise, noise_mean):
        """One generalised-DDIM step on HypDiff's marginals x_t = a_t x0 + s_t eps.

        HypDiff's eps is direction-restricted and not zero-mean, so the step works on
        the centred part u = eps - E[eps]: x_prev = a_prev x0 + s_prev E[eps]
        + sqrt(s_prev^2 - sigma^2) u_hat + sigma (z - E[eps]), with z fresh restricted
        noise and sigma = eta (s_prev / s_t) sqrt(s_t^2 - (a_t / a_prev)^2 s_prev^2).
        This keeps the forward marginal's mean and variance at t_prev (oracle test).
        eta = 0 gives x_prev = a_prev x0 + s_prev eps_hat (the "iterative" sampler).
        """
        a_t, a_p = self.signal_coefficient(t), self.signal_coefficient(t_prev)
        s_t, s_p = self.noise_coefficient(t), self.noise_coefficient(t_prev)
        eps_hat = (x_t - a_t * x0) / s_t
        if self.eta == 0.0:
            return a_p * x0 + s_p * eps_hat
        variance = self.step_variance(t, t_prev)
        if variance < 0:
            raise ValueError(f"stochastic step {t}->{t_prev} has negative variance")
        sigma = self.eta * (s_p / s_t) * math.sqrt(variance)
        u_hat = eps_hat - noise_mean
        return (a_p * x0 + s_p * noise_mean
                + math.sqrt(max(s_p ** 2 - sigma ** 2, 0.0)) * u_hat
                + sigma * (fresh_noise - noise_mean))

    def _stochastic_reverse(self, x_t, condition, edge_index, direction, labels, t_star, generator):
        self.check_stochastic_range(t_star)
        schedule = self.stochastic_schedule(t_star)
        noise_mean = self.restricted_noise_mean(x_t, direction, labels)
        for index, timestep in enumerate(schedule):
            batch_t = torch.full((x_t.shape[0],), timestep, device=x_t.device, dtype=torch.long)
            x0 = self._predict_x0(x_t, edge_index, batch_t.to(torch.float32) / self.num_timesteps,
                                  condition)
            if self.clip_x0:
                x0 = x0.clamp(-1.0, 1.0)
            if timestep == 0:
                return x0
            x0 = x0.to(x_t.dtype)
            fresh = (self.restricted_noise(x_t, direction, labels, generator)
                     if self.eta > 0 else None)
            x_t = self.stochastic_step(x_t, x0, timestep, schedule[index + 1], fresh, noise_mean)
        raise RuntimeError("stochastic reverse process did not reach t = 0")

    def sample_from_tstar(
        self,
        damaged_embeddings,
        damaged_adjacency,
        labels,
        prototypes,
        t_star,
        generator,
    ):
        """Recover embeddings using only damaged graph inputs and frozen context."""
        if labels.ndim != 1 or labels.shape[0] != damaged_embeddings.shape[0]:
            raise ValueError("labels must contain one community index per node")
        if prototypes.ndim != 2 or prototypes.shape[1] != damaged_embeddings.shape[1]:
            raise ValueError("prototypes must have shape (communities, embedding_dim)")
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= prototypes.shape[0]
        ):
            raise ValueError("labels contain an invalid community index")
        if damaged_adjacency.shape[0] != damaged_embeddings.shape[0]:
            raise ValueError("damaged adjacency and embeddings must have matching nodes")

        labels = labels.to(device=damaged_embeddings.device, dtype=torch.long)
        prototypes = prototypes.to(
            device=damaged_embeddings.device, dtype=damaged_embeddings.dtype
        )
        node_prototypes = prototypes[labels]
        tangent = self.manifold.logmap(node_prototypes, damaged_embeddings, c=self.c)
        tangent_normalized = self.normalization(tangent)
        if not torch.isfinite(tangent_normalized).all():
            raise ValueError("damaged embeddings produced non-finite tangent coordinates")
        direction = torch.stack(
            [self.manifold.logmap0(prototype, c=self.c) for prototype in prototypes]
        )
        recovered_tangent = self._denoise_from_tstar(
            tangent_normalized,
            self.edge(damaged_adjacency.coalesce()),
            direction,
            labels,
            t_star,
            generator,
        )
        recovered_tangent = self.renormal(recovered_tangent, tangent)
        return self.manifold.expmap(recovered_tangent, node_prototypes, c=self.c)


def _leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.01)


class StabilityLPModel(LPModel):
    def __init__(self, args):
        final_activation = getattr(args, "final_activation", "relu")
        if final_activation not in FINAL_ACTIVATIONS:
            raise ValueError("final_activation must be 'relu' or 'leaky_relu'")
        super().__init__(args)
        self.final_activation = final_activation
        if final_activation == "leaky_relu":
            if args.model != "HGCN":
                raise ValueError("final_activation='leaky_relu' is only supported for HGCN")
            self.encoder.layers[-1].hyp_act.act = _leaky_relu
