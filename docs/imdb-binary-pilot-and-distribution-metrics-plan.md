# IMDB-BINARY recovery pilot and distribution-metrics proposal

## Completed recovery pilot

The IMDB-BINARY recovery pilot is a development experiment on known graphs. It
uses constant all-ones node features concatenated with one-hot local node
identity, with the existing ReLU activation. Graph labels are not model inputs.
IMDB-BINARY-0255 remains a separate development case and is excluded from
calibration and evaluation.

The fixed selection seed was `20261110`. Calibration used
IMDB-BINARY-0961, -0502, -0493, -0733, and -0737. Evaluation used
IMDB-BINARY-0377, -0419, -0027, -0773, and -0369. Five damage levels (10% to
50%) and two attack types were tested. Each graph/level had one deterministic
degree-product mask and three seeded uniform-random masks. Partial recovery
used eight draws per mask. Both methods received the same damaged inputs and
candidate-pair ordering.

For each method, calibration selected a separate threshold under the same
5% macro mean true-nonedge FPR cap; both thresholds met that cap and were
frozen before evaluation:

| Method | Frozen threshold | Calibration recovery | Calibration mean FPR |
|---|---:|---:|---:|
| Partial recovery | 0.88079447 | 15.91% | 4.955% |
| No-diffusion baseline | 0.88066280 | 9.43% | 4.968% |

Evaluation recovery, absolute false positives, and realized FPR were:

| Method | Removed-edge recovery | Mean absolute false positives | Realized mean FPR |
|---|---:|---:|---:|
| Partial recovery | 23.15% | 4.21 | 3.509% |
| No-diffusion baseline | 7.82% | 0.56 | 0.278% |

The methods thus met the same calibration cap but had different realized
evaluation FPRs. The macro mean evaluation FPR for partial recovery at 10%
damage was 7.105%; the overall evaluation mean was 3.509%. Evaluation results
were not used to retune thresholds.

Repeated embeddings remain an important limitation. On intact evaluation
graphs, the encoder produced only 3 to 22 distinct node rows per graph at
1e-7 precision, with a zero minimum pairwise distance in each. Score ties
were also common across candidate-pair scores. The detailed artifact reports
per-mask embedding and tie diagnostics; these should accompany any
distribution-level metric rather than being hidden by aggregate scores.

The original run completed calibration and evaluation scoring in about
55 seconds on CPU, inside the approved 20-minute cap. A JSON-list aggregation
error occurred after scoring. It is recorded in `initial_failure.json`;
final summaries were completed from the saved evaluation scores without
retraining or replaying masks. All outputs are local-only under
`artifacts/imdb_binary_pilot_20261110/recovery_20261110_01/`.

## Completed offline distribution-metrics analysis

The approved analysis used only saved probabilities, masks, and the frozen
calibration thresholds. No training, recovery replay, threshold selection,
or source-code change was performed. The exact metric outputs and inputs'
SHA-256 hashes are in
`artifacts/imdb_binary_pilot_20261110/distribution_metrics_20261110_01/`.

### Reconstruction and validation

For each mask and draw, the **decoder edge set** contains only candidate pairs
whose saved probability is at least the method's frozen threshold; it does
not include surviving input edges. The **completed-graph companion** is the
union of those predicted edges with the damaged graph's surviving edges. Both
variants retain every original node, including isolates.

All 900 thresholded draws were checked against the saved per-draw evaluation
metrics before MMD calculation. Recovered-edge and false-positive counts
matched exactly, and FPR matched within `1e-12`. In every draw, decoder edge
count equaled recovered edges plus false positives. All surviving edges were
present in the completed graphs (100% retention). The decoder-only outputs
included 78 empty graphs; completed graphs ranged from 16 to 160 edges.
These are different graph representations and should not be conflated.

### MMD definition and weighting

Each stratum compares the same five intact evaluation graphs with the
corresponding method's generated outputs; development and calibration graphs
are excluded. Targeted degree-product and uniform-random attacks are reported
separately at each damage fraction from 10% to 50%. The results are
descriptive for these five sources, not population-level estimates.

For a graph feature histogram `h`, normalize counts to a probability vector.
The kernel is `k(x,y) = exp(-TV(x,y)^2 / (2 sigma^2))`, where
`TV(x,y) = 0.5 * sum(abs(hx - hy))`; degree histograms with different support
are zero-padded. The reported value is source-weighted biased (V-statistic)
`MMD^2 = sum(a_i a_j k(r_i,r_j)) + sum(b_i b_j k(g_i,g_j)) -
2 sum(a_i b_j k(r_i,g_j))`, with no square root or clipping. Each intact
reference has weight `1/5`. Each generated source has total weight `1/5`,
divided equally across its masks and reconstruction draws (partial recovery:
8 outputs per targeted mask or 24 per random-attack source; baseline: 1 or 3).
Repeated outputs therefore do not increase the effective number of source
graphs.

The histogram features and kernel bandwidths are:

| Metric | Per-graph feature | `sigma` |
|---|---|---:|
| Degree | NetworkX degree histogram, normalized | 1.0 |
| Clustering | All-node clustering coefficients, 100 bins on `[0,1]` | 0.1 |
| Spectral | Normalized-Laplacian eigenvalues, 200 bins on `[-1e-5,2]` | 1.0 |

Spectra were computed serially with a strict eigensolver rather than through
the existing parallel helper, whose reference-sized parameter iterator can
truncate predictions and whose worker silently substitutes zero histograms
on errors. Eigenvalues within `1e-12` of a bin endpoint were clipped to that
endpoint to absorb floating-point boundary roundoff (including a computed
`2.000000000000001`); larger excursions or dropped eigenvalues raise errors.

### Descriptive results

Across the ten attack/damage strata, partial recovery had lower decoder-only
MMD than the no-diffusion baseline in all 10 degree comparisons, all 10
clustering comparisons, and 9 of 10 spectral comparisons. For the completed
graphs, partial recovery had lower MMD in 7 of 10 degree comparisons, 9 of
10 clustering comparisons, and 7 of 10 spectral comparisons. The ranges
below show the individual stratum values across both attack types and all
damage levels; lower MMD indicates closer histograms for that metric and
kernel, not universal graph similarity.

| Graph representation | Metric | Partial recovery MMD² range | No-diffusion MMD² range | Strata where partial is lower |
|---|---|---:|---:|---:|
| Decoder edge set | Degree | 0.3180–0.4964 | 0.4773–0.5408 | 10/10 |
| Decoder edge set | Clustering | 0.5565–1.3617 | 1.0023–1.3872 | 10/10 |
| Decoder edge set | Spectral | 0.3216–0.4513 | 0.4323–0.4721 | 9/10 |
| Completed graph | Degree | 0.0194–0.1236 | 0.0343–0.1710 | 7/10 |
| Completed graph | Clustering | 0.2123–0.5182 | 0.3200–0.6414 | 9/10 |
| Completed graph | Spectral | 0.0269–0.1276 | 0.0386–0.1309 | 7/10 |

The full 120 stratum/method/representation/metric values and the 900-row
reconstruction validation are saved in `mmd_results.json` and
`validation_and_reconstructed_graphs.json`, respectively. These comparisons
are descriptive, based on only five independent evaluation graphs, and do
not establish statistical significance or population-level performance.

F1pr, F1dc, and the 1,000-step/GPU efficiency benchmark remain separate and
were not run. F1pr/F1dc still require verification of the book's exact
definitions and parameters. More independent source graphs—not more draws
from the same five—would be needed for stronger population-level conclusions.

## Separate efficiency benchmark

The IMDB-BINARY efficiency benchmark remains outside this work package. The
completed recovery pilot used the active Phase B path's 200-step diffusion
schedule and does not establish the book's 1,000-step benchmark result. Before
benchmarking, verify the exact book-referenced IMDB-BINARY code path and its
required inputs, what constitutes one timed run, whether initialization and
data movement are included, and the specified warm-up/repetition protocol.
The benchmark targets to verify remain runtime below 15 seconds and GPU memory
below 3 GB. The available environment is CPU-only, so it cannot verify GPU
runtime or VRAM; a suitable GPU environment is required for those claims.
