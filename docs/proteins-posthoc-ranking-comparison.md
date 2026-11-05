# PROTEINS post-hoc ranking comparison

## Scope and interpretation

This is a post-hoc ranking comparison on the five existing PROTEINS evaluation
graphs and their saved perturbations. It does not change the official threshold
or evaluation, and it is not an independent test set. No model was trained and
no threshold was applied.

The final project book distinguishes the requested comparison from Common
Neighbors:

- Section 2.1.2, **Classical Stability Metrics and their Limitations**, discusses
  algebraic connectivity and betweenness centrality as deterministic
  network-level stability measures.
- Section 4.3, **Planned Experiments and Expected Results**, calls for comparing
  the probabilistic stability score against algebraic connectivity and
  betweenness under targeted attacks.
- Common Neighbors is not identified there as one of those classical stability
  comparisons. It is the supplementary, simple missing-edge recovery baseline
  proposed in the implementation requirements. It ranks candidate edges and is
  not itself a probabilistic stability score.

For each graph, all methods use the same candidate additions: removed edges plus
all true nonedges, excluding surviving edges. Common Neighbors is computed only
from the saved damaged adjacency. The no-diffusion method uses saved decoder
probabilities. Partial recovery uses the mean saved decoder probability across
the eight existing reconstructions. Average precision is computed over candidate
additions with removed edges as positives.

Ranks are competition ranks: one plus the number of candidates with a strictly
higher score. Guaranteed top-three recovery counts removed edges strictly above
the third-place score; possible recovery also includes positives tied at that
score that could occupy remaining positions. No candidate-order tie break is
used to inflate recovery.

## Per-graph results

| Graph | Common Neighbors AP | No-diffusion AP | Partial-recovery mean AP | CN top 3 (guaranteed–possible) | No-diffusion top 3 | Partial top 3 |
|---|---:|---:|---:|---:|---:|---:|
| PROTEINS-0555 | 0.106474 | 0.005603 | 0.013118 | 0–2 | 0–0 | 0–0 |
| PROTEINS-0395 | 0.055752 | 0.057608 | 0.021581 | 0–0 | 0–0 | 0–0 |
| PROTEINS-0215 | 0.058586 | 0.036356 | 0.341601 | 0–1 | 0–0 | 1–1 |
| PROTEINS-0925 | 0.057692 | 0.008362 | 0.006966 | 0–0 | 0–0 | 0–0 |
| PROTEINS-1086 | 0.067374 | 0.094444 | 0.026942 | 0–1 | 0–0 | 0–0 |
| Macro mean | 0.069176 | 0.040475 | 0.082042 | 0–4 | 0–0 | 1–1 |

Per-removed-edge competition ranks (rank/tie-count):

| Graph | Removed edge | Common Neighbors | No diffusion | Partial mean |
|---|---|---:|---:|---:|
| 0555 | n025–n026 | 1/14 | 476/1 | 442/1 |
| 0555 | n014–n038 | 1/14 | 680/1 | 663/1 |
| 0555 | n003–n020 | 39/51 | 122/1 | 33/1 |
| 0395 | n005–n019 | 10/18 | 94/1 | 69/1 |
| 0395 | n006–n019 | 28/130 | 113/1 | 144/1 |
| 0395 | n017–n018 | 10/18 | 8/1 | 67/1 |
| 0215 | n011–n035 | 1/15 | 13/1 | 1/1 |
| 0215 | n008–n030 | 16/40 | 152/1 | 158/1 |
| 0215 | n012–n013 | 16/40 | 158/1 | 247/1 |
| 0925 | n003–n021 | 20/33 | 217/1 | 157/1 |
| 0925 | n010–n012 | 20/33 | 292/2 | 412/2 |
| 0925 | n010–n011 | 20/33 | 292/2 | 412/2 |
| 1086 | n022–n023 | 9/23 | 30/1 | 69/1 |
| 1086 | n008–n009 | 1/8 | 6/1 | 52/1 |
| 1086 | n021–n022 | 32/207 | 60/1 | 92/1 |

## Finding

Common Neighbors has the highest AP on PROTEINS-0555, while no diffusion is
highest on PROTEINS-0395 and PROTEINS-1086. Partial recovery has the highest AP
on PROTEINS-0215, largely due to its strong ranking of one removed edge; it is
not consistently superior across graphs. Across five graphs, macro AP is
0.082042 for partial recovery, 0.069176 for Common Neighbors, and 0.040475 for
no diffusion. These descriptive macro means are based on only five already-used
evaluation graphs and do not support a general superiority claim.

Common Neighbors’ apparent rank-1 edges often tie with many candidates; its
possible top-three total is four, but guaranteed top-three recovery is zero.
Partial recovery guarantees one top-three hit overall (on PROTEINS-0215).
No-diffusion guarantees none. This is a ranking diagnostic, not a new
thresholded evaluation or a revision of the official frozen-threshold result.

## LeakyReLU development finding

The separate three-graph encoder comparison found that final-layer LeakyReLU
with slope 0.01 prevented the observed PROTEINS-0282 initialization collapse,
but did not improve removed-edge recovery in the follow-up. Its mean damaged
input AP was 0.035966 versus 0.036444 for ReLU; neither variant recovered a
removed edge in the guaranteed or possible top three on those graphs. The
predefined adoption gate was not met. ReLU remains the default and LeakyReLU
remains an opt-in development variant.

## Provenance and preservation

- Results: `artifacts/proteins_pilot/posthoc_common_neighbors/comparison.json`
- Source evaluation SHA-256:
  `7ebae213f90f5c11541941114253732a9b39cba7698333570c60d9048a41418e`
- Official evaluation SHA-256 after comparison:
  `7ebae213f90f5c11541941114253732a9b39cba7698333570c60d9048a41418e`
- Frozen threshold remains `tau = 1.0`.
- No official calibration/evaluation artifact, checkpoint, or model method was
  changed.
