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
      (same result; upstream loops over nodes in Python).
    * ``draw_noise`` / ``_denoise_from_tstar`` / ``sample_from_tstar``  new
      project methods for partial-noise recovery from t* < T.
    * ``radial_delta`` (paper's delta, upstream hard-codes 1),
      ``denoiser_architecture`` ("upstream_gcn" | "residual_gcn"), ``clip_x0``
      ``restoration_loss`` (damage-aware training), ``denoise_once``
      (edge-reconstruction training) and ``recovery_sampler`` ("iterative" =
      t* .. 0 reverse loop, "one_step" = x0 predicted directly from x_{t*}) — options read
      from ``args``; defaults reproduce the upstream behaviour.

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
from stability.denoisers import ResidualGCNDenoiser

NOISE_FORMULAS = ("exact", "upstream")
DENOISER_ARCHITECTURES = ("upstream_gcn", "residual_gcn")
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
        if self.sampler not in ("iterative", "one_step"):
            raise ValueError("recovery_sampler must be 'iterative' or 'one_step'")
        self.architecture = getattr(args, "denoiser_architecture", "upstream_gcn")
        if self.architecture not in DENOISER_ARCHITECTURES:
            raise ValueError(f"denoiser_architecture must be one of {DENOISER_ARCHITECTURES}")
        if self.architecture == "residual_gcn":
            self.gcn = ResidualGCNDenoiser(
                dim, int(args.denoiser_hidden_dim), int(args.denoiser_layers)
            ).to(device)

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
        # updated in place like upstream.
        signs = torch.sign(self._stack(direction_vector, gaussian_point)[
            self._labels(labels, gaussian_point)
        ])
        gaussian_point.copy_(gaussian_point * signs)
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
            x0 = self.gcn(x_t, edge_index, batch_t.to(dtype=torch.float32) / self.num_timesteps)
            return x0.clamp(-1.0, 1.0) if self.clip_x0 else x0

        for timestep in range(t_star, -1, -1):
            batch_t = torch.full(
                (x_t.shape[0],), timestep, device=x_t.device, dtype=torch.long
            )
            # Network was trained on t / num_timesteps (see forward).
            normalized_t = batch_t.to(dtype=torch.float32) / self.num_timesteps
            x0 = self.gcn(x_t, edge_index, normalized_t)
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
        x0 = self.gcn(x_t, self.edge(damaged_adjacency.coalesce()), t / self.num_timesteps)
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
        prediction = self.gcn(x_t, self.edge(damaged_adjacency.coalesce()), t / self.num_timesteps)
        return F.mse_loss(prediction, target.to(torch.float32))

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
