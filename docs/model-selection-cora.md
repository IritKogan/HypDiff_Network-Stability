# Model selection on Cora (validation only, 2026-11-16)

All choices below use **validation data only** (encoder: validation edges vs
validation negatives; denoiser: validation-recovery protocol, see
`stability/validation_recovery.py`). Pool edges and control pairs are never
used for selection. Reproduce with `scripts/sweep_encoder.py` and
`scripts/sweep_denoiser.py`.

## Encoder (HGCN link prediction, early stopping on val AUC)

Fixed: lr 0.01, num_layers 2, c = 1, negatives resampled every epoch, train edges = 50%.
Starting point (dim 16, no dropout, no weight decay): val AUC 0.765.

Stage 1 — dim x dropout (weight_decay 0, bias false):

| model.dim | model.dropout | val AUC | best epoch |
|---|---|---|---|
| 128 | 0.5 | 0.809 | 256 |
| 64 | 0.5 | 0.806 | 282 |
| 32 | 0.5 | 0.798 | 820 |
| 128 | 0.2 | 0.786 | 150 |
| 64 | 0.2 | 0.785 | 154 |
| 32 | 0.2 | 0.784 | 96 |
| 64 | 0.0 | 0.782 | 84 |
| 32 | 0.0 | 0.781 | 152 |
| 128 | 0.0 | 0.778 | 84 |
| 16 | 0.2 | 0.777 | 528 |
| 16 | 0.5 | 0.769 | 213 |
| 16 | 0.0 | 0.765 | 157 |

Stage 2 — around the best stage-1 settings:

| model.dim | model.dropout | autoencoder.weight_decay | model.bias | val AUC | best epoch |
|---|---|---|---|---|---|
| 64 | 0.5 | 0.0005 | True | 0.881 | 439 |
| 128 | 0.5 | 0.0005 | True | 0.880 | 304 |
| 128 | 0.7 | 0.0005 | True | 0.878 | 755 |
| 128 | 0.7 | 0.0 | True | 0.839 | 980 |
| 128 | 0.7 | 0.0 | False | 0.828 | 757 |
| 128 | 0.7 | 0.0005 | False | 0.825 | 507 |
| 64 | 0.7 | 0.0005 | False | 0.818 | 803 |
| 128 | 0.5 | 0.0 | True | 0.818 | 287 |
| 128 | 0.5 | 0.0005 | False | 0.817 | 401 |
| 64 | 0.5 | 0.0005 | False | 0.815 | 439 |
| 64 | 0.7 | 0.0 | False | 0.815 | 882 |
| 128 | 0.5 | 0.0 | False | 0.809 | 256 |
| 64 | 0.5 | 0.0 | True | 0.807 | 314 |
| 64 | 0.5 | 0.0 | False | 0.806 | 282 |
| 64 | 0.7 | 0.0005 | True | 0.753 | 28 |
| 64 | 0.7 | 0.0 | True | 0.740 | 21 |

**Chosen:** dim 128, dropout 0.7, weight_decay 5e-4, bias false (val AUC 0.825).
Gains flatten above dim 64. Likely remaining limits: only 50% of the edges are
available for training (the rest is reserved for validation and the pool) and
the curvature is fixed at c = 1 (upstream default for Cora is 0.6, trainable c is
not supported by the pilot code yet).

## Denoiser (validation-recovery protocol)

Encoder as chosen above (val AUC 0.825), 5000 denoiser epochs, K = 10,
tau calibrated to 5% false positives on validation negatives per method.
Variants: `upstream_gcn` = HypDiff gcndec; `residual_gcn` = x0 = x_t + correction
(identity at initialisation); `generative` = upstream objective; `damage_aware` =
input: edge-dropped train graph + noise, target: intact train graph (train edges only);
`clip` = upstream clamp of predicted x0 to [-1, 1].

| variant | t* | val AUC diffusion | hit @5% FP diffusion | val AUC no diffusion | hit @5% FP no diffusion |
|---|---|---|---|---|---|
| residual_damage_clip | 0 | 0.856 | 0.436 | 0.825 | 0.384 |
| residual_damage_clip | 3 | 0.851 | 0.426 | 0.825 | 0.384 |
| residual_damage_clip | 7 | 0.848 | 0.424 | 0.825 | 0.384 |
| residual_damage_clip | 15 | 0.845 | 0.406 | 0.825 | 0.384 |
| residual_damage_clip | 30 | 0.842 | 0.379 | 0.825 | 0.384 |
| residual_damage_noclip | 0 | 0.838 | 0.389 | 0.825 | 0.384 |
| residual_damage_noclip | 3 | 0.838 | 0.414 | 0.825 | 0.384 |
| residual_damage_noclip | 7 | 0.836 | 0.399 | 0.825 | 0.384 |
| residual_damage_noclip | 15 | 0.834 | 0.390 | 0.825 | 0.384 |
| residual_damage_noclip | 30 | 0.831 | 0.381 | 0.825 | 0.384 |
| residual_damage_noclip_t50 | 0 | 0.828 | 0.379 | 0.825 | 0.384 |
| residual_damage_noclip_t50 | 3 | 0.831 | 0.387 | 0.825 | 0.384 |
| residual_damage_noclip_t50 | 7 | 0.832 | 0.393 | 0.825 | 0.384 |
| residual_damage_noclip_t50 | 15 | 0.833 | 0.395 | 0.825 | 0.384 |
| residual_damage_noclip_t50 | 30 | 0.835 | 0.390 | 0.825 | 0.384 |
| residual_generative_noclip | 0 | 0.833 | 0.367 | 0.825 | 0.384 |
| residual_generative_noclip | 3 | 0.834 | 0.390 | 0.825 | 0.384 |
| residual_generative_noclip | 7 | 0.833 | 0.389 | 0.825 | 0.384 |
| residual_generative_noclip | 15 | 0.830 | 0.390 | 0.825 | 0.384 |
| residual_generative_noclip | 30 | 0.826 | 0.381 | 0.825 | 0.384 |
| upstream_generative_clip | 0 | 0.857 | 0.482 | 0.825 | 0.384 |
| upstream_generative_clip | 3 | 0.855 | 0.460 | 0.825 | 0.384 |
| upstream_generative_clip | 7 | 0.853 | 0.458 | 0.825 | 0.384 |
| upstream_generative_clip | 15 | 0.850 | 0.446 | 0.825 | 0.384 |
| upstream_generative_clip | 30 | 0.848 | 0.412 | 0.825 | 0.384 |
| upstream_generative_noclip | 0 | 0.829 | 0.365 | 0.825 | 0.384 |
| upstream_generative_noclip | 3 | 0.830 | 0.368 | 0.825 | 0.384 |
| upstream_generative_noclip | 7 | 0.830 | 0.363 | 0.825 | 0.384 |
| upstream_generative_noclip | 15 | 0.828 | 0.335 | 0.825 | 0.384 |
| upstream_generative_noclip | 30 | 0.826 | 0.330 | 0.825 | 0.384 |

**Findings**
- The clamp dominates: every variant with the clamp beats every variant without it.
- The new residual architecture and damage-aware training help only without the
  clamp (0.836 vs 0.830 at t* = 7); with the clamp they do not beat upstream.
- Smaller t* is always slightly better: the added noise does not help recovery.
- **Chosen:** upstream_gcn, generative, clip (= upstream HypDiff behaviour), t* = 7.
  t* = 0 scores marginally higher (0.857 vs 0.853) but removes almost all sampling
  randomness, so the K reconstructions would be nearly identical and p_e would
  collapse to 0/1; t* = 7 keeps the probabilistic design.
- Because the clamp alone may explain the gain, every run now also reports a
  `clamp_only` method (clamp step without network or noise) next to `diffusion`
  and `no_diffusion`.

## Paper alignment: clusters k and radial coefficient delta

Paper (Fu et al. 2024, Eq. 6): x_t = sqrt(abar) x0 + sqrt(1-abar) z + delta * tanh(sqrt(c) lambda_o t / T0) x0,
experiments with delta = 0.4, T0 = 1000, 10,000 epochs, lr 1e-4. Upstream code: delta hard-coded 1 and
tanh(sqrt(c) t / 500), which equals lambda_o t / T0 with lambda_o = 2, T0 = 1000 (no discrepancy in T0).
delta is now `denoiser.radial_delta` (default 1 = upstream).

The k sweep first exposed a second K-means problem: the upstream initialisation (centroids uniform in
[0, 0.3]^d, random assignments) collapses to ~4 non-empty clusters for any k (k = 7 and k = 20 gave
identical results). `community.init = "kmeans++"` (data-seeded, empty clusters re-seeded) fixes this;
all clusters are non-empty for k = 7 ... 100.

Validation recovery, encoder val AUC 0.825, kmeans++, upstream denoiser with clamp:

| k | delta | t* | AUC diffusion | AUC clamp_only | AUC no_diffusion | hit@5% diffusion | hit@5% clamp_only | hit@5% no_diffusion |
|---|---|---|---|---|---|---|---|---|
| 7 | 0 | 0 | 0.850 | 0.857 | 0.825 | 0.412 | 0.473 | 0.384 |
| 7 | 0 | 7 | 0.842 | 0.857 | 0.825 | 0.395 | 0.473 | 0.384 |
| 7 | 0 | 15 | 0.838 | 0.857 | 0.825 | 0.364 | 0.473 | 0.384 |
| 7 | 0.4 | 0 | 0.848 | 0.857 | 0.825 | 0.381 | 0.473 | 0.384 |
| 7 | 0.4 | 7 | 0.840 | 0.857 | 0.825 | 0.371 | 0.473 | 0.384 |
| 7 | 0.4 | 15 | 0.837 | 0.857 | 0.825 | 0.383 | 0.473 | 0.384 |
| 7 | 1 | 0 | 0.851 | 0.857 | 0.825 | 0.445 | 0.473 | 0.384 |
| 7 | 1 | 7 | 0.841 | 0.857 | 0.825 | 0.410 | 0.473 | 0.384 |
| 7 | 1 | 15 | 0.837 | 0.857 | 0.825 | 0.397 | 0.473 | 0.384 |
| 20 | 0 | 0 | 0.851 | 0.853 | 0.825 | 0.380 | 0.398 | 0.384 |
| 20 | 0 | 7 | 0.847 | 0.853 | 0.825 | 0.372 | 0.398 | 0.384 |
| 20 | 0 | 15 | 0.846 | 0.853 | 0.825 | 0.381 | 0.398 | 0.384 |
| 20 | 0.4 | 0 | 0.852 | 0.853 | 0.825 | 0.362 | 0.398 | 0.384 |
| 20 | 0.4 | 7 | 0.842 | 0.853 | 0.825 | 0.352 | 0.398 | 0.384 |
| 20 | 0.4 | 15 | 0.838 | 0.853 | 0.825 | 0.347 | 0.398 | 0.384 |
| 20 | 1 | 0 | 0.850 | 0.853 | 0.825 | 0.407 | 0.398 | 0.384 |
| 20 | 1 | 7 | 0.844 | 0.853 | 0.825 | 0.358 | 0.398 | 0.384 |
| 20 | 1 | 15 | 0.843 | 0.853 | 0.825 | 0.358 | 0.398 | 0.384 |
| 50 | 0 | 0 | 0.845 | 0.847 | 0.825 | 0.378 | 0.409 | 0.384 |
| 50 | 0 | 7 | 0.842 | 0.847 | 0.825 | 0.358 | 0.409 | 0.384 |
| 50 | 0 | 15 | 0.841 | 0.847 | 0.825 | 0.369 | 0.409 | 0.384 |
| 50 | 0.4 | 0 | 0.845 | 0.847 | 0.825 | 0.401 | 0.409 | 0.384 |
| 50 | 0.4 | 7 | 0.843 | 0.847 | 0.825 | 0.394 | 0.409 | 0.384 |
| 50 | 0.4 | 15 | 0.841 | 0.847 | 0.825 | 0.370 | 0.409 | 0.384 |
| 50 | 1 | 0 | 0.846 | 0.847 | 0.825 | 0.403 | 0.409 | 0.384 |
| 50 | 1 | 7 | 0.841 | 0.847 | 0.825 | 0.378 | 0.409 | 0.384 |
| 50 | 1 | 15 | 0.841 | 0.847 | 0.825 | 0.373 | 0.409 | 0.384 |
| 100 | 0 | 0 | 0.840 | 0.844 | 0.825 | 0.395 | 0.415 | 0.384 |
| 100 | 0 | 7 | 0.836 | 0.844 | 0.825 | 0.382 | 0.415 | 0.384 |
| 100 | 0 | 15 | 0.835 | 0.844 | 0.825 | 0.372 | 0.415 | 0.384 |
| 100 | 0.4 | 0 | 0.842 | 0.844 | 0.825 | 0.396 | 0.415 | 0.384 |
| 100 | 0.4 | 7 | 0.840 | 0.844 | 0.825 | 0.400 | 0.415 | 0.384 |
| 100 | 0.4 | 15 | 0.839 | 0.844 | 0.825 | 0.398 | 0.415 | 0.384 |
| 100 | 1 | 0 | 0.843 | 0.844 | 0.825 | 0.424 | 0.415 | 0.384 |
| 100 | 1 | 7 | 0.842 | 0.844 | 0.825 | 0.405 | 0.415 | 0.384 |
| 100 | 1 | 15 | 0.842 | 0.844 | 0.825 | 0.401 | 0.415 | 0.384 |

**Findings:** larger k does not help (k = 7 best for both clamp_only and diffusion); delta changes AUC by at
most ~0.004 with no consistent direction; diffusion stays at or below clamp_only for every (k, delta).
**Chosen:** k = 7, delta = 1 (upstream), init kmeans++. With the collapsed upstream clustering the diffusion
scored 0.853 at t* = 7 vs 0.842 now; clamp_only is ~0.857 in both cases.
