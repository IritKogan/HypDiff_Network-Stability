# IMDB-BINARY F1pr/F1dc exploratory analysis

## What the project book specifies

The supplied final project book PDF, §4.1 (“Fidelity and Diversity Metrics”),
requests precision-recall (`F1pr`) and density-coverage (`F1dc`) to assess
generative consistency, structural uncertainty, and mode collapse. It does
not specify the operational formulas, graph representation, feature-space
distance, neighborhood parameter, or tie policy. The analysis below is
therefore an explicitly approved exploratory operationalization, not a claim
that these details are prescribed by the book.

## Scope and operational definitions

The analysis reuses the five intact IMDB-BINARY evaluation graphs, saved masks
and completed outputs, and the original frozen thresholds (partial recovery
`0.8807944655`; no diffusion `0.8806627989`). No training, reconstruction,
threshold selection, or recalibration was performed.

Each graph is represented by its normalized degree histogram, 100-bin
clustering-coefficient histogram on `[0,1]`, and 200-bin
normalized-Laplacian eigenvalue histogram on `[-1e-5,2]`. The graph distance
is the equal-weight arithmetic mean of the three total-variation distances;
degree histograms are zero-padded when supports differ. This is a graph-level
structural feature space derived from the previously used MMD features, not
the HypDiff latent embedding.

For each real or generated sample, the neighborhood radius is the distance
to its `k`th nearest *other* sample in the same set. A point on the radius
boundary is included, including ties. The reported components are:

- Precision: fraction of generated samples within any real-sample radius.
- Recall: fraction of real samples within any generated-sample radius.
- Density: generated/real neighborhood memberships divided by `k` times the
  number of generated samples.
- Coverage: fraction of real samples within any generated-sample radius.
- `F1pr = 2 * precision * recall / (precision + recall)`.
- `F1dc = 2 * density * coverage / (density + coverage)`.

An F1 value is set to zero when its denominator is zero. Density can exceed
one when neighborhoods overlap or ties include more than `k` points, so
`F1dc` can exceed one under this formula; values were not clipped.

The generated set consists of **completed graphs** (surviving damaged edges
plus saved additions), with every original node retained. Empty decoder-only
addition sets are not treated as empty completed graphs: the prior saved
outputs contained 78 empty decoder-only sets, while completed graphs had at
least 16 edges. Edgeless graphs would still have defined histogram features
with their node sets retained.

For each attack/damage stratum, 1,000 deterministic paired selection
replicates chose one saved mask per source uniformly and one partial-recovery
draw uniformly; the same mask was used for both methods. Each replicate
therefore contains exactly one generated graph per source, avoiding treating
the 8 draws or 3 masks as independent source graphs. The five intact graphs
are the reference set. The `k=1,2,3` results and paired leave-one-source-out
checks (four references and four generated sources) are reported. Resampling
quantiles describe conditional output variation only, not population
uncertainty.

Exact duplicate graphs and duplicate feature vectors were retained in each
source's saved-output pool and can be selected in the resampling; they were
not deduplicated. Across the 10 strata, partial recovery had 511 repeated
exact graph outputs among 800 saved draws (and the same number of repeated
feature vectors). No-diffusion had zero repeated outputs among its 100
source/mask outputs. These repeated partial draws do not increase the count
of independent source graphs.

## Descriptive results

The table gives the macro mean across the 10 attack/damage strata. The strong
increase with `k`, including saturated values, is a warning that five
reference samples provide unstable neighborhood radii.

| `k` | Partial `F1pr` | No-diffusion `F1pr` | Partial `F1dc` | No-diffusion `F1dc` |
|---:|---:|---:|---:|---:|
| 1 | 0.660 | 0.529 | 0.697 | 0.580 |
| 2 | 0.802 | 0.672 | 0.707 | 0.609 |
| 3 | 0.939 | 0.836 | 0.790 | 0.691 |

At `k=3`, partial recovery's `F1pr` stratum mean ranged from 0.631 to 1.000;
no-diffusion ranged from 0.263 to 1.000. The five targeted strata saturated
at `F1pr=1.0` for both methods. Partial recovery had higher `F1pr` in 5 of
10 strata and tied in the other 5. For `F1dc`, partial recovery was higher in
9 of 10 strata. These small-sample comparisons do not establish that one
method has better generative fidelity or diversity.

Leave-one-source-out macro means at `k=3` ranged as follows across the five
omitted graphs:

| Method | `F1pr` range | `F1dc` range |
|---|---:|---:|
| Partial recovery | 0.903–0.948 | 0.710–0.788 |
| No-diffusion | 0.758–0.872 | 0.607–0.721 |

Although the relative ordering remained partial-above-baseline for these
leave-one-out macro means, the absolute scores changed substantially with
`k`, and the full-reference `F1pr` was saturated on the targeted strata.
This sensitivity means the five-source result is suitable only as a
descriptive diagnostic, not as strong evidence for the book's mode-collapse
or population-level claims.

## Interpretation alongside prior findings

These metrics measure overlap and density in the selected graph-feature
space. They do not measure whether exact removed edges were recovered, nor
how many false edges were added. The existing exact-edge results remain
separate: book stability was 23.15% for partial recovery and 7.82% for
no-diffusion, with mean false positives 4.21 versus 0.56 per
graph/attack/damage group and mean true-nonedge FPR 3.509% versus 0.278%.
The completed degree, clustering, and spectral MMD findings are also
separate structural comparisons; neither MMD nor F1 substitutes for exact
edge recovery and false-positive scoring.

## Reproducibility and next evidence needed

The metric summaries, 1,000-replicate seeds and previews, duplicate counts,
input hashes, and leave-one-source-out results are in
`artifacts/imdb_binary_pilot_20261110/f1_metrics_20261110_01/`. The run
validated saved thresholds against calibration, reused the previously
validated completed outputs, preserved all node IDs, checked finite
histograms and spectral bounds, and ran a hand-checked toy formula case.
Offline calculation took about 56 seconds.

For a stronger distribution-level claim, add at least 20 independent
held-out source graphs (25 total) and repeat evaluation with the thresholds
still frozen. This is a practical target, not a formal power calculation;
the additional graphs matter because neighborhood radii and graph-sample
uncertainty depend on independent source graphs, not more reconstructions
from the same five.

The efficiency benchmark remains outside this analysis.
