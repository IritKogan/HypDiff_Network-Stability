# Leakage-free edge-recovery pipeline (2026-11-16)

Entry point: `python scripts/run_experiment.py --config configs/cora.toml`
(Colab: `notebooks/run_cora_colab.ipynb`). Replaces `stability/cora_edge_recovery.py`.

## Protocol

1. **Edge split** (`stability/edge_split.py`, `[split]`), done once:
   `train` 50% / `val` 10% / `pool` 40% of the undirected edges, plus validation
   negatives and `control_pairs` (true non-edges).
2. **Training** (`stability/model_training.py`) uses only `train` edges:
   encoder message passing and positives, community centres (hyperbolic K-means,
   `n_clusters = 7`) and the denoiser input graph. Encoder early stopping uses
   validation AUC (`val` vs validation negatives). Training negatives are
   resampled every epoch from non-edges that are neither validation negatives
   nor control pairs; pool edges are true edges, so they are never negatives.
   Pool edges and control pairs are therefore symmetric: neither ever receives
   a training signal.
3. **Cumulative damage series** (`stability/perturbation_masks.py`): each
   series is one ordering of the pool; level L removes the first
   `round(L * |E|)` edges, so 10% ⊂ 20% ⊂ 30% inside a series. N independent
   series per attack: `random`, `targeted_weighted` (sampled without
   replacement with P ~ k_i·k_j) and one `targeted_deterministic` reference
   series (k_i·k_j descending). Degrees are taken on the full graph.
4. **Recovery**: damaged graph = all edges minus the removed ones; encode,
   K partial-noise recoveries from t*, decode removed edges and control pairs
   with threshold tau. A no-diffusion baseline (decode the damaged embeddings
   directly) is recorded for the same pairs. Noise seeds depend only on
   (series, k), so they are shared across attacks and levels.
4b. **Validation recovery and thresholds** (`stability/validation_recovery.py`):
   input graph = train edges only; validation edges vs validation negatives.
   Per method (`diffusion`, `no_diffusion`, `clamp_only`) the threshold tau is
   calibrated so that the false-positive rate on validation negatives equals
   `recovery.target_fpr` (`tau_mode = "val_fpr"`), so methods are compared at the
   same false-positive rate. Used also for model selection
   (`scripts/sweep_encoder.py`, `scripts/sweep_denoiser.py`,
   docs/model-selection-cora.md). Pool and control pairs are never involved.
5. **Scoring** (`stability/edge_recovery_distribution.py`):
   n_e = #series in which e was removed, r_e = #reconstructions in which it
   reappeared, p_e = r_e / (n_e·K). Control pairs: p_c = hits / (N·K).
   Reported per attack and level: distribution summaries, histograms,
   same-class vs different-class edges (Cora labels used only here), KS /
   Wasserstein distances to the control group, threshold-free AUC (mean
   probability, removed edges vs control pairs) for all three methods, to the no-diffusion baseline
   and between attacks. Levels are nested within a series, so between-level
   comparisons are descriptive only.

## Tests

`tests/test_edge_recovery_pipeline.py`: config validation, split disjointness,
cumulative series, targeted preference, leakage audit (actual adjacency,
positives and every sampled negative), invariance of the trained models to the
pool edges, p_e arithmetic, and a reproducible end-to-end run.

## Run directory contents

`config.toml`, `config.json` (resolved, incl. CLI overrides), `environment.json`
(git commit/dirty, versions, device), `split.npz`, `series.npz`,
`checkpoints/`, `training_history.json`, `recovery_counts.npz` (raw
removed/recovered arrays per attack, level, series, edge), `edge_recovery.csv`,
`control_recovery.csv`, `summary.json`, `figures/`.

## Ensemble reconstructions (Decision 11, 2026-11-18)

`recovery.reconstruction_sources = ["noise", "ensemble"]` runs two reconstruction sources in one run,
paired: same split, damage series, noise seeds (`recovery_seed(seed, series, k)`) and pairwise reference
draws (`SeedSequence([seed, 11, method, series, level, k])`).

- `noise`: one model (config seeds), K diffusion noise draws (as before). Files without suffix.
- `ensemble`: K = `recovery.reconstructions` models; reconstruction k of every damaged graph goes through
  model k (its encoder, community context and denoiser) with noise draw k, for diffusion, no_diffusion and
  clamp_only. Files with suffix `_ensemble`; `summary["ensemble"]`; `ensemble_members.json`.
- Models (`stability/ensemble_training.py`): `[ensemble] resample_mode = "seed_only"` -> all train edges,
  seeds from `member_seed` (stability/bootstrap_ensemble.py); model 1 = the noise model (shared, not
  retrained). Every model is trained on TRAIN edges only; with subsample / bootstrap the held-out train
  edges stay forbidden negatives. Pool, control and validation pairs are never used.
- Checkpoints: model 1 in `checkpoints/`, model k+1 in `checkpoints/member_XX/` (encoder, denoiser,
  community context, training_history.json with the curves, member.json with seeds / edges / training
  seconds, validation_recovery.json, `done` marker written last).
- Resume: `run_experiment.py --run-dir DIR --resume` continues DIR if its config.json equals the given
  config; finished models are loaded, unfinished ones retrained; recoveries are recomputed (identical,
  tested). A finished run (summary.json present) is returned unchanged.
- Ensemble requires `edge_decision = "pairwise"` (bernoulli / threshold calibrate one model).

Per-reconstruction outputs in `recovery_counts*.npz` (keys `<attack>__<name>`), used by the method
comparison and the Streamlit GUI:

| name | shape | meaning |
|---|---|---|
| `recovered_by_reconstruction`, `no_diffusion_by_reconstruction`, `clamp_only_by_reconstruction` | (levels, series, K, pool) bool | edge reappeared in reconstruction k (sums over k to the counts) |
| `<method>_expected_by_reconstruction` | (levels, series, K, pool) float16 | share of control pairs scored lower (no reference-draw noise) |
| `<method>_probability_by_reconstruction_sum` | (levels, K, pool) float32 | decoded probability in reconstruction k, summed over series |
| `control_<method>_probability_by_reconstruction_sum` | (levels, K, control) float32 | same for control pairs |

For the noise source the deterministic methods have identical values for every k (one model).
