# Fresh disjoint PROTEINS recovery comparison

## Scope and separation from official results

This is a development experiment for recovery of known graphs. Five fresh
PROTEINS graphs were used for calibration and five other fresh graphs for
evaluation. Each graph had its own autoencoder and denoiser trained on that
graph's intact topology. The calibration and evaluation sets were chosen before
training and have disjoint content hashes. No dataset download was performed.

The previous official PROTEINS result at frozen `tau = 1.0` remains a separate,
unchanged outcome. The configured Fermi-Dirac decoder has `r = 2`, `t = 1`, and
nonnegative squared distance, so its maximum is
`sigmoid(2) = 0.8807970779778823` (float32 maximum observed:
`0.8807970285415649`). Thus `tau = 1.0` necessarily decodes no edges for any
input. This new experiment uses only its separately calibrated
`tau = 0.8807922601699829`; it does not revise the official threshold or
artifacts.

## Selection, attacks, and training

The graph selection seed was `20261112`. Eligibility required a connected,
simple, undirected graph with 12–50 nodes, at most 200 unique edges, and finite
node features. All PROTEINS graph IDs used in earlier pilot or activation
development work were excluded. Graph identities, content hashes, attack masks,
and seeds were saved before training:

| Split | Fresh graphs |
|---|---|
| Calibration | PROTEINS-0524, PROTEINS-0886, PROTEINS-0554, PROTEINS-0704, PROTEINS-0407 |
| Evaluation | PROTEINS-0739, PROTEINS-0089, PROTEINS-0080, PROTEINS-0642, PROTEINS-0413 |

At 10%, 20%, 30%, 40%, and 50% damage, each graph-level used one deterministic
targeted mask and eight distinct uniform random masks. Targeted edges were
ranked by descending intact degree product, with canonical endpoint order as
the tie-break. Removal counts used `floor(p * E + 0.5)`. The paired methods
used the same damaged inputs and candidate pairs: the book's partial-recovery
method with eight seeded reconstructions per mask, and direct no-diffusion
decoding. This yielded 225 masks and 1,800 partial-recovery reconstructions per
split.

Each graph used the established CPU settings: 30 autoencoder epochs, 50
denoiser epochs, curvature 1.0, `t_star = 7`, and a 200-step diffusion
schedule. Training, calibration, evaluation, and checkpoints are separate
from the official pilot.

## Calibration rule and result

Threshold candidates were the observed decoder probabilities within the
attainable range. The selected shared threshold maximized the partial-recovery
book stability score, required nonzero removed-edge recovery, and had to keep
the macro mean FPR at or below 5% for both methods. For each mask,
reconstruction metrics were averaged first; masks were averaged within each
graph × attack × damage-level cell; graph/damage cells were averaged within
each attack type; and the targeted and random attack types received equal
weight. Ties preferred the lower mean FPR across both methods, then the higher
threshold.

The gate passed at `tau = 0.8807922601699829`:

| Calibration metric at selected threshold | Partial recovery | No diffusion |
|---|---:|---:|
| Book stability score | 0.104729 | 0.044544 |
| Mean FPR | 0.049700 | 0.007023 |
| Mean absolute false-positive edges per mask | 28.208 | 4.653 |
| Mean added-edge precision | 0.173386 | 0.313605 |

The first nonzero-recovery threshold was `0.8807970285415649`, with partial
recovery score 0.063404, partial FPR 0.007305, and no-diffusion FPR 0.005677.
Lowering the threshold to `0.8807728290557861` raised the partial score to
0.161097 but exceeded the cap at 9.983% partial FPR. Calibration therefore
shows a genuine recovery/false-positive trade-off; the selected point is the
highest recovery found under the predefined joint cap.

The selected threshold was written to the separate calibration artifact and
frozen before training/evaluating the five evaluation graphs.

## Evaluation at the frozen development threshold

Metrics below average masks within each graph × attack × damage-level cell and
then give the two attacks equal weight. False-positive counts are absolute
mean counts per mask; precision is the mean over reconstructions with at least
one predicted addition. The partial-recovery diversity is the fraction of
unique outputs among eight samples and mean pairwise Jaccard distance,
aggregated with the same equal attack weighting.

| Metric | Partial recovery | No diffusion |
|---|---:|---:|
| Book stability score | 0.101883 | 0.012553 |
| Added-edge precision | 0.203411 | 0.180645 |
| Mean FPR | 0.037786 | 0.004367 |
| Mean absolute false-positive edges per mask | 9.475 | 1.458 |
| Surviving-edge retention | 0.204728 | 0.091120 |
| Predicted edge count | 17.137 | 4.483 |

Partial recovery averaged 5.995 distinct decoded edge sets per eight samples
(unique-output fraction 0.749) and mean pairwise Jaccard distance 0.306. The
no-diffusion baseline is deterministic; its pairwise diversity is not
applicable.

Per-graph results averaged across the five damage levels (and across the eight
random masks within each random-attack level):

| Graph | Attack | Book score: partial / no diffusion | Precision: partial / no diffusion | False positives: partial / no diffusion | Partial unique-output fraction |
|---|---|---:|---:|---:|---:|
| 0739 | Targeted | 0.007 / 0.012 | 0.015 / 0.050 | 2.325 / 4.000 | 0.925 |
| 0739 | Random | 0.006 / 0.002 | 0.047 / 0.005 | 1.988 / 3.550 | 0.700 |
| 0089 | Targeted | 0.097 / 0.018 | 0.099 / 0.167 | 26.475 / 1.000 | 1.000 |
| 0089 | Random | 0.108 / 0.000 | 0.112 / 0.000 | 16.384 / 0.100 | 0.956 |
| 0080 | Targeted | 0.291 / 0.000 | 0.282 / 0.000 | 7.050 / 2.600 | 0.575 |
| 0080 | Random | 0.154 / 0.015 | 0.206 / 0.184 | 7.319 / 1.200 | 0.725 |
| 0642 | Targeted | 0.157 / 0.066 | 0.173 / 0.800 | 19.450 / 0.400 | 0.975 |
| 0642 | Random | 0.089 / 0.011 | 0.100 / 0.078 | 13.294 / 1.675 | 0.978 |
| 0413 | Targeted | 0.027 / 0.000 | 0.479 / — | 0.075 / 0.000 | 0.325 |
| 0413 | Random | 0.083 / 0.001 | 0.639 / 0.250 | 0.388 / 0.050 | 0.334 |

The table shows substantial per-graph variation. Partial recovery scored higher
on most graph/attack combinations and had higher mean removed-edge recovery and
surviving-edge retention overall, but it predicted more false positives and
used a larger edge budget. Its added-edge precision was only modestly higher
overall. These five evaluation graphs provide a bounded paired comparison, not
evidence of broad dataset-level generalization.

## Runtime, artifacts, and preservation

Total recorded CPU time was 223.4 seconds: 123.4 seconds for calibration and
99.9 seconds for evaluation. This was within the approved 30-minute cap.

- Pre-outcome graph/mask/seed manifest:
  `artifacts/proteins_pilot/fresh_disjoint_random_targeted_20261105/selection.json`
- Calibration and frozen threshold:
  `artifacts/proteins_pilot/fresh_disjoint_random_targeted_20261105/calibration.json`
- Evaluation and per-graph/per-mask metrics:
  `artifacts/proteins_pilot/fresh_disjoint_random_targeted_20261105/evaluation.json`
- Separate checkpoints:
  `artifacts/proteins_pilot/fresh_disjoint_random_targeted_20261105/models/`
- Official calibration SHA-256 remained
  `9e31bb2aac1de2be11cc7518a3a63be90e4140a4238dccc279cd568cc8e181c8`.
- Official evaluation SHA-256 remained
  `7ebae213f90f5c11541941114253732a9b39cba7698333570c60d9048a41418e`.
- No dataset download, official threshold change, or official artifact change
  occurred.
