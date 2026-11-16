# IMDB-BINARY recovery and four-graph-set follow-up

## Primary question and recovery scores

The primary question is: given a damaged graph, how often does HypDiff
recover the exact removed edges, and how many incorrect edges does it
propose? This analysis reuses the saved evaluation outputs and the
method-specific thresholds selected on calibration graphs. It does not
reconstruct, retrain, or recalibrate. Removed-edge and intact-true-nonedge
labels are used only for scoring after predictions have been made.

The book stability score is the mean fraction of removed edges recovered.
False positives are predicted candidate edges that were absent from the
intact graph. The saved evaluation summary averages over 50
graph/attack/damage groups (800 partial-recovery draws and 100 no-diffusion
outputs):

| Method | Frozen threshold | Book stability score | Mean false positives per group | Mean true-nonedge FPR |
|---|---:|---:|---:|---:|
| Partial recovery | 0.8807944655 | 23.15% | 4.21 | 3.509% |
| No-diffusion decoding | 0.8806627989 | 7.82% | 0.56 | 0.278% |

These are averages on the five fixed evaluation graphs and perturbations,
not population estimates.

## Four graph sets and structural comparison

The offline comparison includes four sets for each attack and damage level:

1. The five intact evaluation graphs, used as the reference distribution.
2. Damaged graphs before additions, one graph per saved mask.
3. Damaged graphs plus saved partial-recovery additions.
4. Damaged graphs plus saved no-diffusion additions.

Every graph retains the full original node set, including isolates. The
completed-graph predictions are reused from the prior offline analysis and
their MMD values were verified to match that analysis within `1e-12`.
Damaged-only graphs have no stochastic variation across draws, so each
saved mask is represented once; its source graph's total weight is split
equally across masks. For generated outputs, each source graph has total
weight `1/5`, split equally across its masks and draws. Each intact reference
also has weight `1/5`.

The same source-weighted biased `MMD^2` and Gaussian total-variation kernel
are used as in the earlier metrics note. Histograms are normalized per graph:
degree (NetworkX degree histogram, `sigma=1.0`), clustering (100 bins on
`[0,1]`, `sigma=0.1`), and normalized-Laplacian spectrum (200 bins on
`[-1e-5,2]`, `sigma=1.0`). The intact-reference self-comparison is zero.
Lower MMD describes closer similarity in these structural histograms; it
does not measure exact edge recovery or false positives.

Each cell below gives `damaged-only MMD² → completed MMD² (change)`. A
negative change means additions moved that metric closer to the intact
reference distribution; a positive change means it moved farther away.
The partial and no-diffusion results are shown separately.

### Degree MMD²

| Attack | Damage | Damaged | Partial recovery | No diffusion |
|---|---:|---:|---:|---:|
| Targeted | 10% | 0.040796 | 0.081007 (+0.040210) | 0.040796 (+0.000000) |
| Targeted | 20% | 0.085135 | 0.062742 (-0.022393) | 0.089540 (+0.004406) |
| Targeted | 30% | 0.070608 | 0.066786 (-0.003822) | 0.088246 (+0.017638) |
| Targeted | 40% | 0.055971 | 0.055015 (-0.000955) | 0.048258 (-0.007713) |
| Targeted | 50% | 0.140922 | 0.123575 (-0.017347) | 0.100183 (-0.040738) |
| Random | 10% | 0.054145 | 0.019447 (-0.034698) | 0.034324 (-0.019821) |
| Random | 20% | 0.085257 | 0.031561 (-0.053697) | 0.058791 (-0.026466) |
| Random | 30% | 0.085398 | 0.040959 (-0.044439) | 0.085823 (+0.000425) |
| Random | 40% | 0.122683 | 0.079098 (-0.043585) | 0.099582 (-0.023101) |
| Random | 50% | 0.182361 | 0.101065 (-0.081295) | 0.171012 (-0.011348) |

### Clustering MMD²

| Attack | Damage | Damaged | Partial recovery | No diffusion |
|---|---:|---:|---:|---:|
| Targeted | 10% | 0.382188 | 0.359152 (-0.023036) | 0.382188 (+0.000000) |
| Targeted | 20% | 0.540539 | 0.212316 (-0.328223) | 0.359164 (-0.181375) |
| Targeted | 30% | 0.585399 | 0.363926 (-0.221473) | 0.320027 (-0.265372) |
| Targeted | 40% | 0.513606 | 0.417920 (-0.095686) | 0.498347 (-0.015259) |
| Targeted | 50% | 0.584995 | 0.518227 (-0.066768) | 0.641398 (+0.056403) |
| Random | 10% | 0.464597 | 0.370472 (-0.094125) | 0.457201 (-0.007395) |
| Random | 20% | 0.455196 | 0.415081 (-0.040115) | 0.430171 (-0.025024) |
| Random | 30% | 0.458683 | 0.431498 (-0.027185) | 0.457680 (-0.001003) |
| Random | 40% | 0.454597 | 0.432291 (-0.022306) | 0.454193 (-0.000404) |
| Random | 50% | 0.457144 | 0.433210 (-0.023934) | 0.456122 (-0.001022) |

### Spectral MMD²

| Attack | Damage | Damaged | Partial recovery | No diffusion |
|---|---:|---:|---:|---:|
| Targeted | 10% | 0.038553 | 0.061448 (+0.022895) | 0.038553 (+0.000000) |
| Targeted | 20% | 0.102849 | 0.069490 (-0.033359) | 0.105843 (+0.002994) |
| Targeted | 30% | 0.097010 | 0.080137 (-0.016873) | 0.099386 (+0.002376) |
| Targeted | 40% | 0.075241 | 0.063474 (-0.011767) | 0.059969 (-0.015272) |
| Targeted | 50% | 0.157348 | 0.127560 (-0.029788) | 0.098790 (-0.058558) |
| Random | 10% | 0.068049 | 0.026904 (-0.041145) | 0.040955 (-0.027094) |
| Random | 20% | 0.101977 | 0.043666 (-0.058311) | 0.080006 (-0.021971) |
| Random | 30% | 0.119750 | 0.058255 (-0.061494) | 0.125669 (+0.005919) |
| Random | 40% | 0.125352 | 0.073054 (-0.052298) | 0.117485 (-0.007868) |
| Random | 50% | 0.159511 | 0.071577 (-0.087934) | 0.130877 (-0.028634) |

Across the ten attack/damage strata, partial-recovery additions lowered
degree MMD in 9, clustering MMD in 10, and spectral MMD in 9. No-diffusion
additions lowered degree MMD in 6, clustering MMD in 8, and spectral MMD in
6. This is descriptive evidence on five evaluation graphs; it does not
establish that structural similarity implies correct edge recovery.

## Reproducibility and boundaries

The score summary and all four-set MMD values, source weights, input hashes,
and per-stratum comparisons are in
`artifacts/imdb_binary_pilot_20261110/four_graph_set_followup_20261110_01/`.
The original pilot, thresholds, masks, and previous distribution-metrics
outputs were not modified.

F1pr, F1dc, and the 1,000-step/GPU efficiency benchmark remain separate
work packages and were not run.
