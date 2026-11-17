# Stage 1 — Does the diffusion produce a meaningful distribution? (2026-11-17)

Context: the students decided (2026-11-17) that the diffusion must make a real contribution:
its role is to produce a *distribution* of plausible repaired graphs, so that p_e measures how
strongly an edge is supported by all plausible graphs consistent with the damaged graph, and
HypDiff's geometric (angular + radial) noise must remain part of the model. Proposed direction:
a hyperbolic diffusion model **conditioned on the damaged graph**, trained over the full noise
range and sampled ancestrally (fresh geometric noise at every step). Stage 1 = diagnosis of the
current models before writing new code. Stage 2 (implementation) is NOT approved yet.

All numbers: Cora, validation-recovery protocol only (input = train edges; validation edges
vs validation negatives, 528 + 528 pairs; pool and control never used), configs/cora.toml at
998c0e5, K = 10, tau at 5% FP on validation negatives. Encoder val AUC 0.825.

Reproduce:
```
python scripts/diagnose_sample_diversity.py --config configs/cora.toml --out-dir artifacts/diagnostics/sample_diversity
python scripts/analyze_pe_informativeness.py --config configs/cora.toml --checkpoints artifacts/diagnostics/sample_diversity
```
(~16 min + 1 min on the sandbox CPU; outputs sample_diversity.json, pe_informativeness.json.)

Variants: `current_residual_edgerec` = current default (Decision 8: residual_gcn,
edge_reconstruction, train_t_max 8); `upstream_generative` = original HypDiff denoiser and
objective (upstream_gcn, generative, train_t_max 200, clip).

## 1. How much of the injected noise survives the denoiser

`sensitivity` = std across K of predicted x0 / std across K of x_{t*} (standardised tangent
space at the community prototype). 0 = noise fully removed, 1 = passed through.

| variant | sampler | t* | noise coef | radial coef | sensitivity | hit@5%FP | AUC | p_e in (0,1), val edges |
|---|---|---|---|---|---|---|---|---|
| current | one_step | 3 | 0.071 | 0.006 | 0.41 | 0.524 | 0.854 | 0.03 |
| current | one_step | 7 | 0.134 | 0.014 | 0.47 | 0.517 | 0.856 | 0.07 |
| current | one_step | 15 | 0.257 | 0.030 | 0.61 | 0.503 | 0.860 | 0.15 |
| current | one_step | 30 | 0.470 | 0.060 | 0.80 | 0.460 | 0.857 | 0.33 |
| current | one_step | 50 | 0.698 | 0.100 | 0.85 | 0.347 | 0.840 | 0.45 |
| current | one_step | 100 | 0.963 | 0.197 | 0.80 | 0.260 | 0.808 | 0.44 |
| current | one_step | 150 | 0.999 | 0.291 | 0.78 | 0.221 | 0.797 | 0.42 |
| current | iterative | 7 | 0.134 | 0.014 | 1.77 | 0.324 | 0.815 | 0.21 |
| current | iterative | 30 | 0.470 | 0.060 | 1.19 | 0.282 | 0.800 | 0.51 |
| current | iterative | 150 | 0.999 | 0.291 | 0.74 | 0.202 | 0.780 | 0.53 |
| upstream | one_step | 7 | 0.134 | 0.014 | 0.08 | 0.443 | 0.851 | 0.01 |
| upstream | one_step | 50 | 0.698 | 0.100 | 0.09 | 0.443 | 0.852 | 0.09 |
| upstream | one_step | 150 | 0.999 | 0.291 | 0.17 | 0.426 | 0.848 | 0.19 |
| upstream | iterative | 7 | 0.134 | 0.014 | 0.07 | 0.410 | 0.841 | 0.02 |
| upstream | iterative | 50 | 0.698 | 0.100 | 0.10 | 0.341 | 0.832 | 0.12 |
| upstream | iterative | 150 | 0.999 | 0.291 | 0.19 | 0.346 | 0.827 | 0.29 |

(Full grid incl. t* 3/15/30/100 in the JSON.) Clamp_only (deterministic): hit 0.473, AUC 0.857.

Findings:
- **At t* = 7 the noise is tiny** (std 0.13 vs signal 0.99) and the radial (popularity) term of
  HypDiff is negligible (0.014; 0.29 even at t* = 150). The geometry is barely active.
- **The original HypDiff denoiser removes ~90% of the noise at every t***, even at t* = 150
  where x_{t*} is almost pure noise, and still reaches AUC 0.85. Its output is therefore
  determined by the graph structure and the community-prototype anchor, not by the sample:
  the K reconstructions collapse.
- **The residual (current) denoiser passes the noise through** (identity at init). Larger t*
  gives more intermediate p_e but quality drops (hit 0.52 -> 0.35 at t* = 50).
- The iterative loop with the edge-reconstruction model amplifies the noise (sensitivity > 1)
  and loses quality, consistent with the drift reported in Decision 8.
- **Most of the AUC comes from the community-prototype anchor**: clamp_only alone gives 0.857,
  and even near-pure-noise inputs (t* = 150) keep AUC ~0.80-0.85.

## 2. Is the spread informative? (pe_informativeness.json)

If p_e is a meaningful probability, it should (a) carry information beyond a deterministic
score and (b) beyond the model's own mean probability (otherwise the "distribution" is just a
monotone transform of a point estimate).

| variant | sampler | t* | p_e in (0,1), all pairs | AUC p_e | AUC mean prob | strat. AUC p_e within deciles of own mean | LL gain p_e over own mean (nats/pair) | LL gain mean over clamp |
|---|---|---|---|---|---|---|---|---|
| current | one_step | 7 | 0.05 | 0.748 | 0.856 | 0.45 | 0.011 | 0.007 |
| current | one_step | 15 | 0.10 | 0.757 | 0.860 | 0.51 | 0.009 | 0.008 |
| current | one_step | 30 | 0.20 | 0.769 | 0.857 | 0.54 | 0.008 | 0.005 |
| current | one_step | 50 | 0.28 | 0.746 | 0.840 | 0.51 | 0.001 | 0.000 |
| current | iterative | 7 | 0.13 | 0.685 | 0.815 | 0.51 | 0.004 | -0.001 |
| current | iterative | 30 | 0.31 | 0.731 | 0.800 | 0.58 | 0.002 | 0.000 |
| upstream | one_step | 50 | 0.05 | 0.716 | 0.852 | 0.74 | 0.006 | 0.000 |
| upstream | one_step | 150 | 0.11 | 0.728 | 0.848 | 0.58 | 0.005 | -0.001 |
| upstream | iterative | 100 | 0.14 | 0.695 | 0.830 | 0.52 | 0.000 | -0.001 |
| upstream | iterative | 150 | 0.17 | 0.716 | 0.827 | 0.55 | 0.002 | -0.001 |

Reliability (current, one_step, t* = 7; balanced pairs): p_e = 0 -> 32% true edges
(730 pairs); p_e = 0.1 ... 0.9 -> 64-100% true edges (47 pairs); p_e = 1 -> 91% (279 pairs).
Same pattern at t* = 30/50: any p_e > 0 means ~75-95% true edges.

Findings:
- **In no configuration does the K-sample spread add information**: within strata of the
  model's own mean probability p_e is at chance (0.45-0.58; the one 0.74 is on 5% of pairs),
  and the log-likelihood gain is <= 0.011 nats per pair.
- **The diffusion's mean output adds ~nothing over clamp_only** (<= 0.008 nats/pair).
- **p_e is not calibrated**: it behaves as "crossed the 5%-FP threshold at least once", not as
  a probability. Thresholding also loses ranking information (AUC p_e 0.69-0.77 vs mean
  probability 0.80-0.86).

## 3. Conclusion of Stage 1

The near-binary p_e is not a tuning problem. The current denoisers either delete the sampled
noise (upstream) or pass raw noise through (residual); in both cases the spread of the K
reconstructions is uninformative, the geometric noise is negligible at the t* in use, and the
signal comes from the encoder + community anchor. Any "probabilistic" claim requires a model
whose samples come from a learned conditional distribution (Stage 2), and success must be
measured by informativeness and calibration of p_e, not only by hit rate:
1. quality: hit@5%FP >= clamp_only (0.47);
2. informativeness: stratified AUC of p_e within deciles of the model's own mean clearly > 0.5
   and positive LL gain;
3. calibration: reliability of p_e (or of the per-pair sample mean) close to the diagonal.

Side observation for the advisor (Decision 1): thresholding each sample with tau and counting
discards information; the per-pair mean decoded probability over the K samples is a natural
alternative / complement to p_e. To be discussed, no change made.

## 4. Literature check on originality (2026-11-17, web search; not exhaustive)

Closest prior work, by theme:
- **Probabilistic network reconstruction with per-edge reliability.** Guimerà & Sales-Pardo
  (PNAS 2009) average over an ensemble of stochastic block models to score how reliable each
  observed or missing link is (missing and spurious interactions). Peixoto (PRX 2018; arXiv
  2503.07736, 2025) samples posterior distributions over whole networks (SBM-based, MCMC) and
  derives marginal edge probabilities and uncertainties. -> The idea "edge probability =
  frequency over plausible reconstructions" is established. Our version differs in the
  generative model (learned hyperbolic latent diffusion vs SBM inference) and in the question
  (stability under controlled damage, not data errors). These must be cited and compared.
- **Diffusion / flow models for graph completion (inpainting).** SGDM (Trivedi et al., subgraph
  diffusion, 2023/2024) recovers missing edges conditioned on the observed graph; Prior-Informed
  Flow Matching (arXiv 2601.22107, 2026) reconstructs missing links on Cora, PROTEINS and IMDB-B
  with a conditional flow from a link-predictor prior. It works in adjacency space without
  hyperbolic geometry, does not study attacks/robustness, and reports that a single step gives
  the best AUC while more steps improve distributional quality (same trade-off we see). Other
  diffusion link predictors exist (sub-graph diffusion for link prediction, arXiv 2409.08487;
  latent graph diffusion framing prediction as conditional generation). -> "Conditional
  diffusion for missing links" is not novel by itself.
- **Hyperbolic generative models.** HypDiff (ICML 2024), HGDM (AAAI 2024), GGBall (ICLR 2026,
  Poincaré-ball flow matching). All target unconditional generation; none found for
  conditional completion or stability.
- **Hyperbolic link prediction.** Kitsak, Voitalov & Krioukov (PRR 2020): hyperbolic geometry
  helps most for hard missing links (large missing fractions, dissimilar nodes) — relevant to
  the targeted-attack setting and a classical baseline candidate.
- **Network robustness.** Classical measures (LCC / percolation, efficiency, edge betweenness,
  curvature, edge instability) are deterministic structural scores; recovery/repair studies
  after attacks exist (e.g. causal reconstruction after adversarial node removal, Nat. Commun.
  2019) but do not use generative latent models. Diffusion-based uncertainty quantification is
  an active topic in other domains (calibrated posterior sampling).

Assessment: none of the works found combines (i) a hyperbolic geometric latent diffusion model
conditioned on a damaged graph, (ii) controlled cumulative random/targeted damage, and (iii)
stability measured as the distribution of per-edge recovery frequencies across damage levels,
with calibration as the validity check. The defensible novelty is this combination and the
geometric decomposition (angular vs radial), not conditional diffusion or edge reliability as
such. Guimerà & Sales-Pardo, Peixoto, SGDM and PIFM should be in the related-work section; an
SBM-ensemble reliability score is a natural baseline.
