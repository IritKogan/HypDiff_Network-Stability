# Hyperbolic vs Euclidean check (2026-11-25, new code, sandbox CPU)

Question (likely advisor / defence question): does the hyperbolic geometry help, or would the same
pipeline in flat space do as well?

Set-up: the whole hypstab pipeline with `geometry.manifold = "euclidean"` (hypstab/geometry/euclidean.py):
the HGCN becomes a plain GCN (same layers, dropout, activations), the Fermi-Dirac decoder uses the squared
Euclidean distance, K-means uses ordinary means, the clamp and the diffusion work in "point minus centre"
coordinates. Same split, train edges only, same denoiser training (edge_reconstruction, 5000 epochs),
same recovery (one_step, t* 35, K 10). Validation protocol (Decision 7): train graph as input,
validation edges vs validation negatives. One run per row.

| model | no_diffusion | clamp_only | diffusion |
|---|---|---|---|
| hyperbolic (cora.toml defaults) | 0.827 | 0.852 | 0.863 |
| Euclidean, same settings (dropout 0.7, weight decay 5e-4) | 0.808 | 0.810 | 0.828 |
| Euclidean, dropout 0.5, weight decay 0 (best Euclidean) | 0.812 | 0.841 | 0.847 |
| Euclidean, dropout 0.5, weight decay 5e-4 | 0.815 | 0.813 | 0.831 |

The encoder settings were tuned for the hyperbolic model (Decision 7), so the Euclidean encoder got a small
sweep of its own (dropout 0.2 / 0.5 x weight decay 0 / 5e-4, autoencoder only; best encoder AUC 0.815).

Result: the hyperbolic model is ahead in all three methods, also against the best Euclidean setting:
+0.012 (no_diffusion), +0.011 (clamp_only), +0.016 (diffusion). The gain is real but modest, and each row
is a single run (the hyperbolic diffusion value varied 0.855-0.863 between runs), so the honest statement
is "consistently better by about 0.01-0.02 AUC on Cora", not a large effect.

Runs (sandbox only): artifacts/hypstab_runs/geom_poincare, geom_euclidean, euc_sweep_*,
geom_euclidean_d0.5_w0.0, geom_euclidean_d0.5_w0.0005. Reproduce with
`python scripts/hypstab_train.py --set geometry.manifold='"euclidean"' [--set encoder.dropout=0.5 ...]`.
