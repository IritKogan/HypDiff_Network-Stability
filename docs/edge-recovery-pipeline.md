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
5. **Scoring** (`stability/edge_recovery_distribution.py`):
   n_e = #series in which e was removed, r_e = #reconstructions in which it
   reappeared, p_e = r_e / (n_e·K). Control pairs: p_c = hits / (N·K).
   Reported per attack and level: distribution summaries, histograms,
   same-class vs different-class edges (Cora labels used only here), KS /
   Wasserstein distances to the control group, to the no-diffusion baseline
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
