# PROTEINS score-scale diagnostic

## Scope and reproducibility

This diagnostic replay used the five saved fresh-PROTEINS evaluation graphs,
the existing checkpoints, perturbation masks, candidate pairs, reconstruction
seeds, and frozen threshold `tau = 0.8807922601699829`. It covered 10%, 30%,
and 50% damage for targeted degree-product and uniform-random attacks: 135
masks and 1,080 reconstructions. No training, recalibration, or code change
was performed.

Before replay, the PROTEINS-0080 targeted 10% reference case was checked
against its saved graph and checkpoint hashes. Its score, mean false-positive
count, and FPR reproduced exactly: 0.75, 4.875, and 0.02834, respectively.
All replayed active-path scores were finite. Runtime was 36.9 seconds on CPU.

All candidate-pair decoder probabilities were saved before thresholding.
Removed-edge and true-nonedge labels are stored separately and were used only
for evaluation, not to filter inference candidates.

## Ranking and prevalence

For each mask, candidate probabilities were averaged across its eight
reconstructions before ranking. ROC-AUC and average precision (AP) were then
computed per mask and averaged across the five graphs and two attack types.
ROC-AUC uses average ranks for ties; AP groups tied scores at the same
threshold, as implemented by scikit-learn.

| Damage | Removed-edge observations | True-nonedge observations | Positive prevalence | Mean ROC-AUC | Mean AP |
|---|---:|---:|---:|---:|---:|
| 10% | 216 | 15,579 | 1.37% | 0.647 | 0.128 |
| 30% | 648 | 15,579 | 3.99% | 0.632 | 0.147 |
| 50% | 1,089 | 15,579 | 6.53% | 0.602 | 0.180 |

Mean ROC-AUC declines as damage increases, indicating weaker positive-versus-
negative ordering on average. AP rises, but the positive prevalence also rises
substantially: more edges have been removed and enter the positive class,
while the true-nonedge candidate population stays fixed. Since AP depends on
prevalence, its increase is not by itself evidence that ranking improved.
The AP values remain above their corresponding prevalence baselines, but the
paired graph/attack cells vary and should not be reduced to the aggregate AP
trend.

## Score distributions and fixed-threshold outcomes

The medians below average the per-mask median probabilities across graphs and
masks within each attack/damage cell. Quantiles were calculated separately
for removed edges and true nonedges, pooling the eight reconstruction scores
within each mask.

| Attack | Damage | Removed-edge median | True-nonedge median | Recovery | Realized FPR | Mean false positives |
|---|---:|---:|---:|---:|---:|---:|
| Targeted | 10% | 0.8793 | 0.7956 | 0.1571 | 4.93% | 12.575 |
| Targeted | 30% | 0.8703 | 0.7964 | 0.0957 | 4.82% | 12.275 |
| Targeted | 50% | 0.7336 | 0.6968 | 0.1050 | 2.60% | 6.700 |
| Random | 10% | 0.8579 | 0.7959 | 0.1114 | 4.02% | 10.250 |
| Random | 30% | 0.8707 | 0.8008 | 0.0838 | 2.85% | 7.272 |
| Random | 50% | 0.8533 | 0.7561 | 0.0717 | 3.24% | 7.584 |

At higher damage, probability distributions shift relative to the frozen
threshold, though not uniformly by attack or graph. The aggregate ROC-AUC
decline indicates that this is not only a score-scale shift: ranking also
worsens on average. Individual graph trajectories are mixed. For example,
PROTEINS-0080 loses ROC-AUC at both attack types as damage increases, whereas
PROTEINS-0642's ROC-AUC improves over these sampled levels. These are
descriptive results for the tested graphs and fixed masks.

## Community occupancy

Two clusters were requested. PROTEINS-0642 assigned all 26 nodes to one
cluster (sizes 26 and 0); the other four graphs occupied both clusters.
Centers were not coincident in any graph. The collapsed result is one graph
and is descriptive evidence only; it cannot establish that collapse caused
any difference in recovery, ranking, or false positives.

## Artifacts

The separate replay outputs are in
`artifacts/proteins_pilot/score_scale_diagnostic_20261110/`:

- `decoder_scores.npz`: all pre-threshold scores, by graph, case, draw, and
  candidate.
- `candidate_pairs.json`: candidate-pair order for each graph.
- `evaluation_labels.json`: removed-edge indices and true-nonedge indices,
  kept separate from score arrays.
- `manifest.json`: graph, checkpoint, seed, mask, and threshold provenance.
- `summary.json`: per-mask and graph/attack/damage ranking, quantile,
  recovery, FPR, false-positive, and cluster-occupancy summaries.

The diagnostic used the existing recovery implementation and projection.
The optional standalone `artanh` hardening discussed during the audit was not
applied.
