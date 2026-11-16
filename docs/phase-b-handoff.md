# Phase B handoff

## Status

Phase B has a reproducible CPU development path for graph perturbation, latent
diffusion recovery, and probabilistic removed-edge scoring. Completed
PROTEINS, synthetic, and IMDB-BINARY experiments are descriptive evaluations;
they do not establish broad generalization or superiority. The IMDB-BINARY
distribution analysis is complete. The GPU efficiency benchmark remains open.

The current CPU environment is Windows, Python 3.11.9, PyTorch 2.12.0+cpu,
PyTorch Geometric 2.8.0.post1, NumPy 2.4.4, SciPy 1.17.1, NetworkX 3.6.1,
scikit-learn 1.8.0, with 16 GB RAM and eight logical processors. CUDA and
`nvidia-smi` are unavailable here. `requirements-phaseb.txt` records the
tested Python dependencies. `docs/source-and-environment.md` records source
provenance and the original CPU smoke-check details.

## Implemented

- `stability/edge_perturbation.py` creates deterministic random and
  degree-product targeted edge-removal masks. Targeted ties use canonical
  endpoint order; removal counts use half-up rounding.
- `stability/scoring.py` scores removed-edge reappearance as a mean recovery
  frequency across reconstructions.
- `stability/partial_recovery.py` supports encode/decode without diffusion and
  repeated partial-noise denoising with fixed reconstruction seeds and a
  community context.
- `stability/proteins_dataset.py` adapts, filters, hashes, and selects cached
  PROTEINS graphs without downloading data.
- Calibration, evaluation, threshold replay, classical metrics, post-hoc
  baselines, diagnostics, and experiment drivers live under `stability/` and
  `scripts/`.
- The final activation option keeps ReLU as the default and exposes
  LeakyReLU(0.01) as an opt-in development setting.
- Numerical and device handling was hardened in the hyperbolic clustering and
  diffusion code, including finite-value checks and device-aware tensors. CPU
  sampling now uses the configured available device instead of requiring
  CUDA unconditionally.
- `.gitignore` excludes local data caches and all synthetic/PROTEINS experiment
  artifacts so datasets, checkpoints, and large result files are not committed.

## Verification

The Poincare expmap/logmap round trip, HGCN CPU encode/decode, and partial
recovery CPU smoke checks passed. At handoff, the full test suite passed:

```powershell
python -m pytest -q
git diff --check
```

Result: **60 passed**, with four existing PyTorch deprecation/sparse-tensor
warnings; `git diff --check` passed.

The SBM/BA JSON aggregation regression test exercises one JSON-list
no-diffusion output and eight partial-recovery samples. Environment validation
and CPU smoke commands are:

```powershell
python scripts\validate_environment.py
python scripts\smoke_lptask_cpu.py
python scripts\smoke_partial_recovery_cpu.py
```

The partial-recovery denoiser configuration is curvature 1.0, a 200-step
diffusion schedule, `t*=7`, 30 autoencoder epochs, 50 denoiser epochs, and
eight reconstructions per damaged mask. The book's proposed 1,000-step
IMDB-B performance target is not currently an end-to-end implementation:
the node-level `lptask` recovery path does not load IMDB-B, while the
graph-collection `graphtask` path has a different graph-pooled architecture.

## Experimental findings

### Original PROTEINS pilot at `tau=1.0`

The Fermi-Dirac decoder is
`P = sigmoid((r - d) / t)`. With configured `r=2`, `t=1`, and squared
distance `d >= 0`, its maximum probability is `sigmoid(2) =
0.8807970779778823`. Thus `tau=1.0` returns an empty decoded graph for any
input in this configuration. The original five-graph targeted comparison
therefore had a zero book score in all 25 graph/damage cases. This is the
official frozen-threshold outcome, not evidence that recovery works.

### Later calibrated PROTEINS comparisons

Later experiments used separate, explicitly documented calibration protocols
and must not be conflated with the original `tau=1.0` pilot:

- The fresh disjoint PROTEINS run used five calibration and five evaluation
  graphs, random and targeted attacks at 10% through 50%, eight random masks
  per cell, and eight partial-recovery samples per mask. At the shared frozen
  `tau=0.8807922601699829`, evaluation partial recovery scored 0.101883
  (FPR 3.779%, mean 9.475 false positives); no diffusion scored 0.012553
  (FPR 0.437%, mean 1.458 false positives).
- In a post-hoc separate-threshold comparison, calibration selected
  `tau=0.8807922601699829` for partial recovery and
  `tau=0.8784035444259644` for no diffusion. Evaluation scores were 0.101883
  and 0.088360; realized FPRs were 3.779% and 6.749%, and mean false-positive
  counts were 9.475 and 17.105, respectively. Both calibration thresholds met
  the 5% cap, but evaluation FPRs differed; this is a common calibration-cap
  comparison, not a matched evaluation-FPR result.
- On the same fresh masks, an inference-only community-context replay first
  reproduced the intact-context results in all 225 cases with no discrepancy
  at absolute tolerance `1e-12`. The damaged-graph context reduced mean score
  from 0.101883 to 0.094050 and mean FPR from 3.779% to 3.172%. The paired
  score difference was -0.007817 (sampling interval -0.010257 to -0.005511).
  Community assignments changed (mean adjusted Rand index 0.481). This
  changes context only at inference; it does not test retraining with damaged
  context.
- The LCC fraction before recovery and removed-edge recovery did not track
  consistently. For PROTEINS-0080 targeted damage, both fell from 10% to 50%
  damage; for PROTEINS-0642 targeted damage, LCC fell from 1.0 to 0.5 while
  recovery rose from 0 to about 0.279. The book score is an edge-recovery
  frequency, not a direct connectivity/resilience measure.
- The supplementary Common Neighbors ranking comparison had mixed per-graph
  results. It is an edge-recovery baseline and remains separate from the
  project's classical stability comparison.
- Algebraic connectivity and mean normalized node betweenness changed under
  targeted attacks while the book score remained zero at the original frozen
  threshold.
- The LeakyReLU(0.01) development variant prevented the observed final-layer
  encoder collapse on PROTEINS-0282, but did not improve top-three recovery on
  the three-graph follow-up. ReLU remains the default; LeakyReLU is opt-in.

### Completed SBM and BA recovery pilot

The completed run is in
`artifacts/synthetic_pilot/sbm_ba_recovery_20261105_retry03/`. SBM used 24
nodes in two blocks of 12, within-block probability 0.35 and between-block
probability 0.05; disconnected graphs were retained. BA used 24 nodes and
`m=2`. Each family had three calibration graphs and five disjoint evaluation
graphs. Calibration/evaluation graph seeds were SBM `51001–51003` /
`52001–52005` and BA `61001–61003` / `62001–62005`. Model seeds were graph
seed plus 3,000,000. Both methods used the same graph-specific checkpoints,
damaged inputs, candidate pairs, degree-product and random masks, and
reconstruction seeds. Damage levels were 10%, 20%, 30%, 40%, and 50%; each
level used one targeted mask and eight distinct random masks. Thresholds were
selected separately for each method on calibration graphs only under the
attack-balanced 5% mean FPR cap, then frozen before evaluation.

| Family | Method | Frozen threshold | Calibration recovery / FPR | Evaluation recovery | Evaluation FPR | Mean absolute false positives |
|---|---|---:|---:|---:|---:|---:|
| SBM | Partial recovery | 0.8807842135 | 0.09389 / 4.992% | 0.05152 | 2.353% | 5.203 |
| SBM | No diffusion | 0.8767504692 | 0.08896 / 4.947% | 0.04069 | 1.982% | 4.483 |
| BA | Partial recovery | 0.8807610869 | 0.04136 / 4.985% | 0.06023 | 5.555% | 12.889 |
| BA | No diffusion | 0.8780790567 | 0.02185 / 4.801% | 0.01180 | 2.012% | 4.668 |

Partial recovery had higher aggregate evaluation recovery and higher realized
FPR than no diffusion in both families. In BA, partial recovery exceeded the
5% FPR cap on evaluation even though it met the cap on calibration. Intervals
in the result files represent reconstruction sampling uncertainty conditional
on the tested graphs and fixed masks; they are not population-level intervals.
The run completed in about 184 seconds, within the approved 20-minute CPU
limit.

The seed/mask plan, thresholds, per-graph evaluation breakdowns, and run
summary are saved in the result directory. `scripts/run_sbm_ba_recovery_pilot.py`
supports a new output directory and optional checkpoint reuse:

```powershell
python scripts\run_sbm_ba_recovery_pilot.py `
  --output-dir artifacts\synthetic_pilot\sbm_ba_recovery_20261105_retry04

python scripts\run_sbm_ba_recovery_pilot.py `
  --output-dir artifacts\synthetic_pilot\sbm_ba_recovery_20261105_retry04 `
  --reuse-checkpoint-root artifacts\synthetic_pilot\sbm_ba_recovery_20261105_retry03\models
```

An output directory must be new and unused. Do not overwrite any prior attempt.
For detailed progress, calibration thresholds, model checkpoints, and per-case
metrics, see the local artifacts listed below.

### Completed IMDB-BINARY expanded known-graph evaluation

The expanded follow-up used 20 newly selected independent sources as the
primary evaluation set; models were trained on each intact graph before
perturbation, so this is known-graph recovery. On the same perturbations and
frozen thresholds, partial recovery had higher exact removed-edge recovery
than no-diffusion decoding (24.43% versus 5.55%), alongside higher mean false
positives (10.70 versus 3.88 per graph/attack/damage group) and higher
realized evaluation FPR (6.84% versus 2.20%).

The 5% value was the calibration-set mean-FPR cap used to select and freeze
the thresholds; it was not an evaluation guarantee. Partial recovery's
6.84% evaluation FPR is the observed result and was not recalibrated. The
corrected standard density-coverage F1dc at k=2 was 0.502 for partial
recovery and 0.309 for no diffusion; corresponding F1pr was 0.596 and 0.380.
The pooled 25-source summary is supplementary only. Sensitivity, uncertainty,
ties, duplicate-output diagnostics, and the correction details are in
`docs/imdb-binary-expanded-evaluation-2026-11-10.md`. The previous five-source
report and artifacts were preserved.

## Still open

- The book's IMDB-B efficiency target is recorded in prior review notes as
  under 15 seconds and below 3 GB GPU memory. The interpretation of the
  project's 1,000-step requirement remains pending supervisor clarification.
  Appendix K, Table C.5 reports 1,000-step denoising for graph generation
  and training GPU memory separately; it does not establish that the current
  fixed-node recovery pipeline must use 1,000 reverse steps.
- Proposed separate benchmark for the current fixed-node recovery pipeline:
  retain its 200-step schedule and `t_star=7`, which executes eight reverse
  denoiser calls per reconstruction (timesteps 7 through 0, inclusive).
  Measure five preselected fixed-node graphs with eight serial
  reconstructions per damaged input (40 reconstructions total), after
  warm-up. Time damaged-graph encoding, the reverse denoising calls, and
  candidate-pair decoding; exclude training, data/checkpoint loading, and
  precomputed intact-graph community context. Synchronize CUDA for wall-clock
  timing and report per-reconstruction median, p95, and range, as well as
  end-to-end time for each eight-reconstruction set. Reset peak-memory
  statistics after warm-up and report peak allocated and reserved GPU memory
  over inference, including resident models, graph inputs/context, activations,
  diffusion intermediates, and decoder candidates. Record GPU, driver,
  CUDA/PyTorch versions, precision, and checkpoint/config hashes.
- This is a proposal only. No benchmark-specific code changes, schedule
  changes, training, or benchmark runs are authorized. Wait for explicit
  approval and a confirmed compatible GPU environment before proceeding.
  The current CPU-only environment cannot verify GPU runtime or memory.
- Broader graph-level uncertainty/generalization beyond the small fixed graph
  samples, and comparison against a trained Euclidean generative baseline.

No GPU efficiency benchmark or threshold recalibration is included in this
handoff.

## Local-only outputs and data

The following paths are intentionally excluded from Git and will be absent
from a fresh clone. Preserve these directories for continuation:

- `artifacts/encoder_damage_diagnostic/`
- `artifacts/proteins_pilot/activation_development/`
- `artifacts/proteins_pilot/activation_development_fresh20261106/`
- `artifacts/proteins_pilot/calibration/`
- `artifacts/proteins_pilot/calibration_stable_clustering/`
- `artifacts/proteins_pilot/common_fpr_replay_20261105/`
- `artifacts/proteins_pilot/community_context_lcc_20261105/`
- `artifacts/proteins_pilot/evaluation_stable_clustering/`
- `artifacts/proteins_pilot/fresh_disjoint_random_targeted_20261105/`
- `artifacts/proteins_pilot/posthoc_common_neighbors/`
- `artifacts/proteins_pilot/random_vs_targeted_classical_comparison/`
- `artifacts/proteins_pilot/targeted_attack_classical_comparison/`
- `artifacts/synthetic_pilot/sbm_ba_recovery_20261105/` (empty initial attempt)
- `artifacts/synthetic_pilot/sbm_ba_recovery_20261105_retry01/`
- `artifacts/synthetic_pilot/sbm_ba_recovery_20261105_retry02/`
- `artifacts/synthetic_pilot/sbm_ba_recovery_20261105_retry03/` (completed results
  and checkpoints)
- `artifacts/wp5_synthetic_pilot/`
- `artifacts/wp7_noise_depth_diagnostic/`
- `artifacts/wp7_prethreshold_diagnostic/`
- `artifacts/wp7_synthetic_calibration/`
- `artifacts/wp7_synthetic_evaluation/`
- `data/tu/PROTEINS/` (cached dataset files)

The configured Git remote at handoff is
`https://github.com/IritKogan/HypDiff_Network-Stability.git`. The requested
destination was
`https://github.com/IritKogan/Network-stability-using-Hyperbolic-Geometric-Latent-Diffusion-Model`.
These names do not match. Do not push until the repository owner confirms or
corrects the remote URL.
