# IMDB-BINARY expanded known-graph recovery evaluation

## Design and pre-outcome selection

This is **known-graph recovery**, not unseen-graph prediction: each graph's
model is trained on that intact graph before its edges are perturbed. The
primary evaluation set is the 20 newly selected sources below. The five
previous IMDB-BINARY evaluation graphs are reported separately for corrected
F1 compatibility and are included only in a clearly labeled supplementary
25-source summary.

The pre-outcome selection manifest was written before model training and
contains the IDs, content hashes, training/split seeds, perturbation masks,
and reconstruction seeds:
`artifacts/imdb_binary_pilot_20261110/expanded_evaluation_20261110_01/selection_manifest.json`.
Its SHA-256 is
`8f7ddcebe911a93ac26cf65883b64c62e09da2e73d4336f458e5ca20d5e9f9bb`.
Selection used seed `20261111`; the eligibility rule was 12–50 nodes, at
most 200 unique undirected edges, connected simple graph, and a feasible
validation-first split. The seeded order was over distinct eligible graph
contents. All previously used development, calibration, and evaluation
graphs were excluded. The raw dataset had 915 eligible graphs, 771 distinct
eligible contents, and 760 nonexcluded candidates. `IMDB-BINARY-0012` was
skipped before selection because its validation-first split would have no
true nonedges.

| Rank | Graph | Model seed | Validation-first split seed |
|---:|---|---:|---:|
| 1 | IMDB-BINARY-0739 | 3000739 | 3000740 |
| 2 | IMDB-BINARY-0276 | 3000276 | 3000277 |
| 3 | IMDB-BINARY-0433 | 3000433 | 3000434 |
| 4 | IMDB-BINARY-0144 | 3000144 | 3000145 |
| 5 | IMDB-BINARY-0578 | 3000578 | 3000579 |
| 6 | IMDB-BINARY-0192 | 3000192 | 3000193 |
| 7 | IMDB-BINARY-0606 | 3000606 | 3000607 |
| 8 | IMDB-BINARY-0656 | 3000656 | 3000657 |
| 9 | IMDB-BINARY-0609 | 3000609 | 3000610 |
| 10 | IMDB-BINARY-0762 | 3000762 | 3000763 |
| 11 | IMDB-BINARY-0177 | 3000177 | 3000178 |
| 12 | IMDB-BINARY-0731 | 3000731 | 3000732 |
| 13 | IMDB-BINARY-0732 | 3000732 | 3000733 |
| 14 | IMDB-BINARY-0761 | 3000761 | 3000762 |
| 15 | IMDB-BINARY-0215 | 3000215 | 3000216 |
| 16 | IMDB-BINARY-0375 | 3000375 | 3000376 |
| 17 | IMDB-BINARY-0162 | 3000162 | 3000163 |
| 18 | IMDB-BINARY-0601 | 3000601 | 3000602 |
| 19 | IMDB-BINARY-0062 | 3000062 | 3000063 |
| 20 | IMDB-BINARY-0445 | 3000445 | 3000446 |

The raw files contain no node attributes or node labels. Inputs use the
approved constant all-ones feature plus one-hot local node identity, with
ReLU. Graph class labels are not used. Each graph-specific model used the
approved validation-first edge/nonedge split, 30 encoder epochs, 50 denoiser
epochs, CPU execution, curvature 1, and the 200-step partial-recovery
schedule. There are five damage levels (10%, 20%, 30%, 40%, 50%), with one
deterministic intact-degree-product mask and three seeded uniform-random
masks per graph/level. The same masks, damaged inputs, node order, and
candidate pairs are used for both methods; partial recovery has eight draws
per mask and no diffusion has one deterministic output.

The random-mask seed is
`20261110 + dataset_index*10000 + level_index*100 + mask_index`; collision
retry increments it by 3. Reconstruction seed bases are
`7000000 + 3000000 + zero_based_selection_rank*10000 + level_index*100`;
random mask index adds `mask_index*10`. All exact seeds and saved masks are
in the selection manifest.

The method thresholds remain frozen at partial recovery `0.8807944655418396`
and no diffusion `0.8806627988815308`, exactly as selected on the original
calibration set. No expanded evaluation result was used to change a
threshold, model setting, or metric definition.

## Primary result: exact edge recovery and false positives

Book stability is the fraction of removed intact edges recovered. False
positives are proposed candidate edges absent from the intact graph. The
20-source macro averages first average within each graph/attack/damage cell,
then weight all sources equally; attacks and damage levels are equally
weighted in the overall summary.

| Method | Book stability / removed-edge recovery | Mean false positives per graph/attack/damage group | Mean true-nonedge FPR |
|---|---:|---:|---:|
| Partial recovery | 24.43% | 10.70 | 6.84% |
| No-diffusion decoding | 5.55% | 3.88 | 2.20% |
| Paired difference (partial − no diffusion) | +18.87 percentage points | +6.81 | +4.64 percentage points |

The 95% intervals below are paired bootstrap intervals over the 20 source
graphs (10,000 resamples); all masks and partial draws stay nested within
their source graph.

| Metric | Partial recovery | No diffusion | Paired difference |
|---|---:|---:|---:|
| Book stability | 24.43% [18.14%, 30.96%] | 5.55% [3.88%, 7.37%] | +18.87 pp [13.11, 24.73] |
| False positives per group | 10.70 [3.69, 19.60] | 3.88 [0.59, 8.55] | +6.81 [2.79, 11.50] |
| True-nonedge FPR | 6.84% [2.91%, 11.48%] | 2.20% [0.50%, 4.40%] | +4.64 pp [2.09, 7.60] |

Partial recovery recovered more removed edges on average, while proposing
more false positives. Its overall evaluation FPR exceeded 5%; this does not
alter the calibration-selected threshold or the original evaluation results.
The evaluation labels were used only after predictions were produced to
score recovered removed edges and true-nonedge false positives; they were
never used to filter predictions.

Per-attack/damage macro means (`score`, false-positive count, FPR) are:

| Attack | Damage | Partial recovery | No diffusion |
|---|---:|---:|---:|
| Targeted | 10% | 0.045, 10.61, 6.6% | 0.013, 6.10, 3.1% |
| Targeted | 20% | 0.058, 9.49, 5.9% | 0.017, 4.70, 2.9% |
| Targeted | 30% | 0.099, 12.26, 7.3% | 0.023, 4.60, 2.7% |
| Targeted | 40% | 0.091, 15.16, 9.5% | 0.030, 3.70, 1.9% |
| Targeted | 50% | 0.105, 14.74, 9.3% | 0.037, 4.50, 2.8% |
| Random | 10% | 0.502, 10.84, 7.1% | 0.160, 4.35, 2.5% |
| Random | 20% | 0.492, 8.70, 5.9% | 0.110, 3.95, 2.0% |
| Random | 30% | 0.389, 8.04, 5.5% | 0.061, 2.82, 1.6% |
| Random | 40% | 0.364, 8.49, 5.5% | 0.057, 2.40, 1.5% |
| Random | 50% | 0.298, 8.65, 5.7% | 0.047, 1.68, 1.0% |

## Distribution metrics and the F1dc correction

The project book requests precision-recall `F1pr` and density-coverage
`F1dc`, but leaves their operational details unspecified. This analysis
retains the approved graph feature space: normalized degree histograms,
100-bin clustering coefficients on `[0,1]`, and 200-bin normalized-Laplacian
eigenvalues on `[-1e-5,2]`. Graph distance is the equal-weight mean of the
three total-variation distances. The neighborhood boundary is inclusive.
The main neighborhood size is `k=2`; `k=1` and `k=3` are sensitivity results.
Each of 1,000 point replicates selects one mask and one partial draw per
source, equally weights each source, and uses the same mask for both methods.

During this expanded work, review found that the earlier five-source F1
implementation had set coverage equal to recall. The corrected standard
PRDC coverage is the fraction of real samples whose nearest generated sample
falls within that real sample's own real-set `k`-nearest-neighbor radius.
The previous report and artifacts remain unchanged; their `F1dc` values
should be treated as the earlier custom density/recall harmonic score, not
standard density-coverage F1. The five saved evaluation graphs were
recomputed offline using the corrected definition for consistent
supplementary pooling. `F1pr` remains the precision/recall harmonic mean.
Density is not clipped and may exceed one; `F1dc` is not clipped either.

Macro means across the 10 attack/damage strata are:

| Source set | `k` | Partial `F1pr` | No-diffusion `F1pr` | Partial `F1dc` | No-diffusion `F1dc` |
|---|---:|---:|---:|---:|---:|
| New 20 (primary) | 1 | 0.374 | 0.222 | 0.383 | 0.208 |
| New 20 (primary) | 2 | 0.596 | 0.380 | 0.502 | 0.309 |
| New 20 (primary) | 3 | 0.708 | 0.472 | 0.570 | 0.393 |
| Previous 5 (corrected, secondary) | 1 | 0.661 | 0.529 | 0.589 | 0.506 |
| Previous 5 (corrected, secondary) | 2 | 0.803 | 0.671 | 0.701 | 0.583 |
| Previous 5 (corrected, secondary) | 3 | 0.938 | 0.836 | 0.768 | 0.662 |
| Pooled 25 (supplementary) | 1 | 0.384 | 0.253 | 0.388 | 0.251 |
| Pooled 25 (supplementary) | 2 | 0.577 | 0.390 | 0.461 | 0.310 |
| Pooled 25 (supplementary) | 3 | 0.676 | 0.452 | 0.491 | 0.339 |

For the new 20-source primary set, no F1pr or F1dc value equaled 1.0 in the
1,000 point replicates at `k=1,2,3`; F1dc did not exceed 1.0. This contrasts
with saturation on the previous five-source set and illustrates why the
expanded independent source set is more informative. The k sensitivity
remains substantial, so F1 scores are structural-distribution diagnostics,
not replacements for exact edge recovery or the previously completed MMD
analysis.

Duplicate counts are summed over the 10 strata and count repeated outputs
within a source and stratum:

| Source set | Method | Outputs | Repeated exact edge sets | Repeated feature vectors |
|---|---|---:|---:|---:|
| New 20 | Partial recovery | 3,200 | 2,378 (74.3%) | 2,396 (74.9%) |
| New 20 | No diffusion | 400 | 0 | 1 |
| Previous 5 (corrected, secondary) | Partial recovery | 800 | 511 (63.9%) | 511 (63.9%) |
| Previous 5 (corrected, secondary) | No diffusion | 100 | 0 | 0 |

Repeated draws are preserved in the saved outputs but are not counted as
independent source graphs. Source-bootstrap intervals are paired by source;
the F1 report also includes per-stratum source-bootstrap intervals and
conditional output-resampling quantiles.

Pre-threshold candidate scores also contain ties when rounded to `1e-7`.
Across the 400 no-diffusion outputs, 71 had no tied score pairs; the median
tie excess was 28 candidate pairs (mean 48.2). Across 3,200 partial-recovery
draws, 127 had no tied score pairs; the median tie excess was 38 (mean 59.9).
These are score ties across the complete candidate ranking, not duplicate
graphs; exact edge-set and feature-vector duplicates are reported separately.

## Runtime, validation, and artifacts

Training and scoring the 20 graph-specific models took 105.6 CPU seconds.
The corrected offline F1 analysis, including 1,000 point selections and
2,000 source-bootstrap resamples per stratum for the 20-source primary,
corrected previous-five, and pooled-25 sets, took 48.1 seconds. Both were
within their approved 10-minute and 5-minute caps.

Validation confirmed all 20 selected graphs passed the validation-first
split, were content-distinct, and were absent from the prior development,
calibration, and evaluation sets. All 400 masks/cases retained candidate
ordering and node IDs; 3,200 partial draws and 400 no-diffusion outputs had
finite scores. Frozen thresholds matched the original calibration summary.
The full test suite passed (60 tests), including corrected
coverage-versus-recall, inclusive boundary ties, density above one, and
edgeless-node preservation.

Primary and supplementary outputs are under
`artifacts/imdb_binary_pilot_20261110/expanded_evaluation_20261110_01/`:

- `selection_manifest.json` and `selection_manifest.sha256`
- `evaluation/graphs/` (per-source checkpoints, masks, candidates, and scores)
- `evaluation/summary.json` (exact-edge results and source-level bootstrap)
- `distribution/f1_metrics.json` and `distribution/manifest.json`
- Per-stratum distribution metric files and `distribution/work_in_progress.json`

The prior five-source F1 report, MMD outputs, exact-edge results, thresholds,
and other experiment artifacts were not overwritten. The GPU efficiency
benchmark remains a separate work package.
