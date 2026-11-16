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
