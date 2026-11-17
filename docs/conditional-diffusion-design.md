# Stage 2 design — Conditional geometric hyperbolic diffusion (PROPOSAL, 2026-11-17)

Status: proposal for student approval (Decision 9). No code written yet.
Goal: a denoiser whose K samples are draws from a learned distribution of plausible repaired
graphs given the damaged graph, so that p_e is informative and calibrated (Decision 9 (a)-(d)),
with HypDiff's geometric noise active in training and in every sampling step.

## 1. What changes and what stays

Stays (HypDiff): HGCN encoder + Fermi-Dirac decoder (frozen, trained on train edges), community
prototypes (hyperbolic K-means, k=7), tangent space at the node's prototype, global
standardisation (`normalization` / `renormal`), forward process
x_t = (sqrt(abar_t) + delta*tanh(sqrt(c) t/500)) * x_0 + sqrt(1-abar_t) * eps,
with eps restricted to the community direction (`tran_direction`, angular) and the radial
popularity term, T = 200 linear schedule, x_0-predicting network.

Changes:
1. **Conditioning.** The network sees the damaged graph twice: its edges (GCN message passing,
   as today) and its standardised damaged embedding c as extra input channels:
   x0_hat = f(x_t, c, A_damaged, t/T).
2. **Training target.** The CLEAN train-graph embedding is noised (as in HypDiff's generative
   training), not the damaged one (as in Decision 8). Full t range [0, T).
3. **Sampling.** Stochastic reverse process with FRESH geometric noise at every step (today:
   the predicted noise is re-used, so the reverse loop is deterministic).

## 2. Training (leakage-free; train edges only)

Per epoch:
1. drop f ~ U(0, train_max_drop) of the TRAIN edges -> damaged train graph G_d;
2. x_0 = standardised tangent of encoder(G_train); c = standardised tangent of encoder(G_d)
   (both at the community prototypes, encoder frozen);
3. t ~ U{0..T-1} (config: one t per graph, or per node as upstream);
4. x_t = q_sample(x_0, t, geometric noise);
5. loss = MSE(x0_hat, x_0)  [HypDiff's loss]
        + edge_loss_weight * BCE through the frozen decoder on dropped + kept train edges vs
          sampled non-edges (never validation negatives / control pairs), as in Decision 8.
   edge_loss_weight = 0 is pure HypDiff training; selected on validation.

Leakage: identical guarantees to Decision 8 (only train edges, negatives exclude val
negatives and control). Existing leakage tests are extended to the new mode.

## 3. Architectures (stability/denoisers.py; upstream gcndec untouched)

- `conditional_gcn`: documented copy of HypDiff's gcndec (4 GCN layers + 3 FC, GELU) with
  input 2*dim + 1 (x_t, c, t) instead of dim + 1. Closest to the book.
- `conditional_residual_gcn`: x0_hat = c + correction(x_t, c, A_d, t), output layer zero-init
  (starts as "return the damaged embedding"; the network learns what to add).
Both selected on validation.

## 4. Sampling (generalised DDIM with stochasticity eta; works with marginals only)

HypDiff defines only the marginals q(x_t | x_0) (the radial term breaks the standard DDPM
posterior), so use the DDIM family, which needs only marginals. With
a_t = sqrt(abar_t) + prior(t), s_t = sqrt(1 - abar_t):

    start:  x_{t*} = q_sample(c, t*, geometric noise)          (from the damaged embedding)
    for t in strided schedule t* -> 0:
        x0_hat  = f(x_t, c, A_d, t)            (clip optional)
        eps_hat = (x_t - a_t x0_hat) / s_t
        sigma_t = eta * (s_prev / s_t) * sqrt(s_t^2 - (a_t / a_prev)^2 * s_prev^2)
                  # = standard DDIM sigma when a = sqrt(abar), s = sqrt(1 - abar)
        x_prev  = a_prev x0_hat + sqrt(s_prev^2 - sigma_t^2) eps_hat + sigma_t z,
                  z = FRESH geometric noise (tran_direction, seeded per (series, k))
    output x0_hat at t = 0 -> renormal -> expmap at prototypes -> decode

- eta = 0 reproduces today's iterative loop exactly (deterministic given x_{t*}); eta = 1 is
  ancestral-like. eta is the stochasticity knob; t* is "how much the damaged graph is trusted"
  (large t*: mostly noise; see the valid range below).
- **Valid range of t (checked numerically).** Because of the radial term, the signal
  coefficient a_t = sqrt(abar_t) + delta*tanh(t/500) is NOT monotone: it falls until t ~ 154 and
  then rises again (delta = 1: a_154 = 0.34, a_199 = 0.38). Between consecutive steps above that
  point no valid noising kernel exists (the DDIM variance becomes negative): with delta = 1 the
  stochastic sampler is valid only for t* <= 154; delta = 0.4 (paper) -> t* <= 179; delta = 0 -> all
  t. Options: cap t* at the last valid step (default proposal), or use delta = 0.4. This also
  means HypDiff's own forward process does not end in pure noise (a_199 = 0.38 with delta = 1).
- `steps` (strided schedule, e.g. 20) keeps the cost manageable: K x N x levels x attacks
  reverse passes.
- Correctness check before any training: an ORACLE denoiser (returns the true x_0) must
  reproduce the forward marginals at every step (mean/std of x_t), with HypDiff's non-zero-mean
  restricted noise. If not, the sigma formula is adjusted (e.g. centring z).

## 5. Outputs and selection (validation protocol only)

Per sample: decoded probabilities -> tau (val_fpr, 5%) -> p_e, plus the per-pair mean
probability over K (advisor question). Selection grid (CPU here, 2 workers):
architecture {conditional_gcn, conditional_residual_gcn} x edge_loss_weight {0, 1}
x t* {30, 60, 100, 150} x eta {0, 0.5, 1}, steps 20.
Criteria (Decision 9):
- (a) quality: hit@5%FP not below clamp_only - 0.02 (0.473 - 0.02);
- (b) informativeness: stratified AUC of p_e within deciles of the model's own mean probability
  >= 0.60, and cross-validated LL gain > 0 with a bootstrap 95% interval above 0;
- (c) calibration: reliability curve + ECE of p_e and of the mean probability. Base rate: the
  validation set is 50/50; real candidate pairs are mostly non-edges, so negatives are
  re-weighted to the true non-edge ratio when computing calibration.
Ranking: among variants meeting (a), maximise (b), then (c). eta = 0 is the internal control
(same model, no sampling randomness): the gain of eta > 0 over eta = 0 is the direct evidence
that the distribution contributes.

## 6. Config (configs/cora.toml; loader + tests updated together)

[denoiser] training = "conditional"; architecture += conditional_gcn | conditional_residual_gcn;
train_t_max = 200; edge_loss_weight; t_sampling = "per_graph" | "per_node".
[recovery] sampler += "stochastic"; eta; steps; t_star (selected).
Existing modes remain available and unchanged (backward-compatible defaults).

## 7. Tests

Oracle-marginal test; eta = 0 equals the current iterative sampler; eta > 0 differs across
seeds and is reproducible per seed; conditional training never reads val/pool/control pairs;
shapes and device; config round-trip; upstream hash test unchanged.

## 8. Cost

Training: like edge_reconstruction (~9-15 min CPU per variant; more epochs may be needed for
the full t range). Selection sweep: ~2-3 h CPU here in parallel. Full Cora run with 20-step
sampling: ~20x the current recovery cost -> Colab GPU.

## 9. Risks (reported honestly whatever happens)

1. The community anchor carries most of the signal (clamp_only AUC 0.857); the model may not
   beat it on hit. The claim then rests on (b)-(c), with (a) as "not worse".
2. The network may again learn to ignore x_t (samples collapse). eta = 0 vs eta > 0 and the
   sensitivity metric of Stage 1 detect this immediately.
3. Density shift: training graphs have <= 50% of the edges, evaluation graphs 70-90%
   (same issue as the tau calibration shift).
4. HypDiff's restricted noise is not zero-mean; handled by the oracle test.
5. Departure from the book (conditional training, stochastic DDIM sampler): needs advisor
   approval and a book update.

## 10. Pilot results (2026-11-17, commit after 1da6dbc)

Run: `scripts/sweep_conditional.py` (validation only), 8 variants = architecture
{conditional_gcn, conditional_residual_gcn} x angular_noise {upstream, folded} x clip_x0,
5000 epochs, edge_loss_weight 1, per_graph t, t* {30, 60, 100, 150} x eta {0, 0.5, 1},
20 steps, K = 10. Output: artifacts/sweeps/conditional_pilot/sweep_conditional.json.

Findings:
- Quality is fine: AUC 0.84-0.865, hit up to 0.50 (residual, no clip) vs clamp_only 0.473.
  The mean probability is better calibrated at the real base rate without clip
  (ECE 0.05-0.08 vs 0.18-0.19 with clip).
- Requirement (b) still fails: p_e in (0, 1) for at most 12% of validation edges; stratified
  AUC of p_e within the model's own mean 0.46-0.64; LL gain <= 0.008 nats/pair.
- **eta has no effect** (eta = 0 vs 1 give the same numbers): the sampler's noise does not
  reach the output.
- Cause (direct check on trained checkpoints): the network ignores x_t and predicts from
  the condition alone. Even at t = 2 (noise 0.05) its error is 0.064 (residual) / 0.119
  (gcn) while x_t itself is within ~0.003; the spread of x0_hat across noise draws is
  0.003-0.09. This is the known "condition shortcut" of conditional diffusion: the damaged
  embedding explains most of x_0, so a plain x0-predicting network learns E[x_0 | c] and the
  reverse process collapses to one answer. The clean-vs-damaged difference itself is not
  small (std 0.24-0.49 in standardised units for 10-40% edge drop), so there is real
  uncertainty to represent.

Proposed fix (next step): preconditioned x0 prediction (as in EDM / a Bayesian prior centred
on the condition): x0_hat = k_t * (x_t - s_t E[eps]) / a_t + (1 - k_t) * c + out_t * F(x_t, c, A, t),
k_t = a_t^2 v / (a_t^2 v + s_t^2 Var[eps]), v = Var(x_0 - c) measured on training damage
(~0.09). With F = 0 this is the exact posterior mean under a Gaussian prior x_0 ~ N(c, v);
it forces the model to use x_t at small t and lets the sampler's fresh noise propagate,
while F learns where the damaged graph is uncertain. Test: oracle/prior sampler reproduces
N(c, v); then re-run the pilot and check (b).

## 11. Preconditioned pilot + the geometry of the noise (2026-11-17)

Preconditioned run (db216ee; artifacts/sweeps/conditional_precond): residual network, clip off,
angular_noise {upstream, folded} x edge_loss_weight {0, 1}. v estimated on training damage = 0.10.
- The sampler now really samples: the K reconstructions differ by ~0.21-0.24 (standardised tangent
  units, close to sqrt(v)); with F = 0 the unit test confirms draws from N(c, v).
- But edge probabilities barely move: per-pair probability std across K 0.013-0.015; p_e in (0, 1)
  for 5-11% of validation edges; stratified AUC 0.46-0.56; eta still has no visible effect.
- Quality fell to the no-diffusion level (hit 0.38-0.41, AUC 0.83): without the clamp the output is
  essentially the damaged embedding plus noise (|mean x0_hat - c| = 0.08 vs true |x0 - c| ~ 0.25).

Why (scripts/diagnose_noise_geometry.py): with the SAME total noise energy added to the train-graph
embeddings,

| noise shape | prob std across K | p_e in (0, 1) | hit@5%FP |
|---|---|---|---|
| isotropic, 128 dims | 0.011 | 0.04 | 0.378 |
| fixed 8-dim subspace | 0.017 | 0.07 | 0.369 |
| radial direction of each node only | 0.080 | 0.26 | 0.335 |

In a 128-dimensional latent space, independent isotropic noise mostly adds the same constant to all
pairwise distances (absorbed by tau); its pair-specific part shrinks like 1/sqrt(dim)
(concentration of measure). Uncertainty in the embeddings only becomes uncertainty about EDGES when it
is low-dimensional and structured, e.g. along each node's radial (popularity) direction, which is
exactly HypDiff's geometric decomposition. So the geometric part of the noise is not decoration: it
decides whether the diffusion can produce a meaningful p_e at all.

Implication for the design (proposal, needs approval): let the stochastic part of the conditional
diffusion live in HypDiff's geometric coordinates of each node (radial = popularity, angular =
direction relative to its community prototype; a few dimensions per node), with the remaining
coordinates taken deterministically from the conditional mean. Then check (a)-(c) again. Random
radial noise alone is NOT the answer (hit drops to 0.335, and the spread is not learned); the model
must learn which nodes are uncertain and how much.

## 12. Geometric pilot and the role of the threshold (2026-11-17)

Geometric parameterization (baefaf5; artifacts/sweeps/conditional_geometric; edge_loss_weight 0 / 1):
- hit 0.38-0.41, AUC 0.83 (mean mu barely improves on the damaged embedding);
- p_e in (0, 1) for 5-15% of validation edges, stratified AUC 0.49-0.57: still not informative.
- The noise-geometry diagnostic compared EQUAL ENERGY; the learned uncertainty along the geometric
  directions is small (std ~0.3 standardised units, vs ~3.4 in the radial diagnostic). Given the
  damaged graph, the encoder-space posterior is genuinely narrow: three model families (direct,
  preconditioned, geometric) agree that the K embedding samples carry no edge information beyond
  their mean.

Where the graded probability actually is (scripts/diagnose_probability_calibration.py, pilot-1
conditional residual model, t* = 30, eta = 1, K = 10):

| score (no threshold) | AUC | ECE balanced | ECE at real base rate | mean prob. on non-edges |
|---|---|---|---|---|
| no_diffusion decoder probability | 0.825 | 0.128 | 0.182 | 0.182 |
| clamp_only decoder probability | 0.857 | 0.051 | 0.281 | 0.281 |
| conditional diffusion, mean over K | 0.857 | 0.228 | 0.075 | 0.076 |

Validation edges under the diffusion mean probability: 27% below 0.1, 12% 0.1-0.3, 9% 0.3-0.5,
12% 0.5-0.7, 41% 0.7-0.9: graded, not bimodal.

Interpretation: the near-binary p_e comes from the hard threshold tau applied to every sample, not from
the absence of uncertainty. HypDiff's Fermi-Dirac decoder defines P(A_ij = 1); a reconstruction drawn
from the generative model would sample each edge from that probability (Bernoulli), and then
p_e = frequency over K reconstructions -> E_samples[P(A_ij = 1)], which is graded. The conditional
diffusion contributes a better base-rate calibration (non-edge probability 0.076 vs 0.18 / 0.28), but
absolute calibration is still far off (true base rate ~0.001) and would need a leakage-free
calibration step on validation. This changes the p_e definition (Decision 1) -> advisor decision.

## 13. Decision 10 implemented: Bernoulli reconstructions + calibration (2026-11-17)

Students' decision: a reconstruction draws each evaluated pair from a Bernoulli with the calibrated
decoder probability (replaces the hard tau). Implemented:
- stability/probability_calibration.py: weighted Platt scaling fitted on validation pairs only, weights
  restoring the real base rate (validation edges stand for all edges missing from the validation input,
  validation negatives for all non-edges), prior shift to the evaluation base rate (R removed edges among
  the damaged graph's non-edges), cross-fitted log-loss / Brier / ECE.
- recovery.edge_decision = "bernoulli" | "threshold"; validation_recovery fits a calibrator per method;
  run_recoveries draws with SeedSequence([seed, 10, method, series, level, k]) over all pool / control
  pairs (common random numbers across attacks); deterministic baselines also get K draws; expected_p_e =
  mean calibrated probability (no draw noise) is reported too. Tests: tests/test_probability_calibration.py,
  tests/test_bernoulli_reconstruction.py.

Result (scripts/compare_calibrated_methods.py -> artifacts/sweeps/calibrated_methods.json; validation
only; evaluation-like prevalence with 1056 removed edges = 0.00029):

| method | AUC | cross-fitted log-loss | base-rate-only log-loss | median expected p_e (val edges) | 90th pct |
|---|---|---|---|---|---|
| no_diffusion | 0.825 | 0.00539 | 0.00593 | 0.0009 | 0.0017 |
| clamp_only | 0.857 | 0.00523 | 0.00593 | 0.0013 | 0.0023 |
| edge_reconstruction (Decision 8), t*=7 | 0.856 | 0.00517 | 0.00593 | 0.0014 | 0.0031 |
| upstream generative, iterative t*=50 | 0.832 | 0.00535 | 0.00593 | 0.0011 | 0.0017 |
| conditional residual, stochastic t*=30 | 0.857 | 0.00517 | 0.00593 | 0.0015 | 0.0031 |
| conditional residual folded, t*=100 | 0.840 | 0.00527 | 0.00593 | 0.0012 | 0.0029 |
| conditional gcn, t*=30 | 0.855 | 0.00521 | 0.00593 | 0.0014 | 0.0026 |
| geometric + BCE, t*=60 | 0.832 | 0.00536 | 0.00593 | 0.0010 | 0.0018 |

Findings:
- Calibration works (ECE ~0), but at the real base rate a removed edge gets probability ~0.001-0.003:
  5-10x the base rate, still tiny in absolute terms. With K = 10 and N = 20 almost every edge would get
  p_e = 0 -> the Bernoulli p_e distribution is degenerate. (The earlier oral estimate of 0.05-0.2 was wrong.)
- Reason: AUC 0.85 separates edges from non-edges on average, but with ~3,400 non-edges per removed edge no
  pair is likely in absolute terms. The same fact was hidden in the threshold version: tau at 5% FP on
  3.6M non-edges admits ~180,000 false edges per reconstruction against ~1,000 removed ones.
- Probability quality: the best methods (edge_reconstruction and the conditional residual diffusion) beat the
  base rate by ~13% in log-loss; clamp_only 12%; no_diffusion 9%. Diffusion helps a little, not decisively.
- Default edge_decision set back to "threshold" until the students decide how p_e should be defined.

Options for the p_e definition (decision pending):
1. Relative (base-rate-free) recoverability: in each reconstruction compare the removed edge with a
   randomly drawn non-edge of the same damaged graph; p_e = frequency with which the edge scores higher.
   Graded, threshold-free, control pairs ~0.5 by construction; = per-edge AUC contribution.
2. Lift: expected p_e divided by the base rate ("x times more likely than a random pair").
3. Calibration against a balanced reference (50/50): readable values, but reconstructions are not
   realistic graphs.

## 14. Chosen p_e definition: pairwise (relative) recoverability — first Cora run (2026-11-17)

Students' choice (option 1 of section 13), implemented in f5da64a (edge_decision = "pairwise", default):
in every reconstruction a removed edge "reappears" if its decoded probability is strictly higher than that
of a reference pair drawn uniformly from the control pairs (true non-edges never used in training) of the
same reconstruction; control pairs are compared with other control pairs, so their frequency is ~0.5 by
construction. p_e = r_e / (n_e K) as in Decision 1. No threshold, no calibration. expected_p_e = share of
control pairs scored lower (same quantity without the draw noise). The mean of p_e over edges equals the
AUC of removed edges vs control pairs, so p_e is the per-edge contribution to that AUC.

Run: configs/cora.toml at f5da64a (Decision 8 denoiser: residual, edge_reconstruction, one_step, t* = 7),
CPU, 616 s; artifacts/runs/cora_edge_recovery_20261117T164522Z (not in git). Validation AUC: no_diffusion
0.825, clamp_only 0.852, diffusion 0.848.

| attack | damage | p_e mean (diffusion) | clamp_only | no_diffusion | control | same / different class | share p_e = 1 | 5th pct |
|---|---|---|---|---|---|---|---|---|
| random | 10% | 0.870 | 0.859 | 0.824 | 0.500 | 0.886 / 0.802 | 0.28 | 0.42 |
| random | 20% | 0.866 | 0.856 | 0.830 | 0.500 | 0.882 / 0.799 | 0.17 | 0.40 |
| random | 30% | 0.861 | 0.852 | 0.828 | 0.500 | 0.878 / 0.794 | 0.12 | 0.37 |
| targeted_weighted | 10% | 0.886 | 0.869 | 0.857 | 0.500 | 0.904 / 0.817 | 0.34 | 0.50 |
| targeted_weighted | 20% | 0.873 | 0.862 | 0.850 | 0.500 | 0.890 / 0.802 | 0.20 | 0.42 |
| targeted_weighted | 30% | 0.862 | 0.856 | 0.839 | 0.500 | 0.880 / 0.787 | 0.12 | 0.38 |
| targeted_deterministic | 10% | 0.905 | 0.848 | 0.844 | 0.502 | 0.918 / 0.861 | | |
| targeted_deterministic | 20% | 0.900 | 0.859 | 0.862 | 0.500 | 0.917 / 0.838 | | |
| targeted_deterministic | 30% | 0.895 | 0.866 | 0.866 | 0.500 | 0.914 / 0.824 | | |

Observations (preliminary):
- p_e is graded (random 20%: median 0.95, 5th percentile 0.40, only 0.1% exactly 0), and clearly separated
  from the control distribution (KS 0.59-0.62, Wasserstein 0.36-0.39).
- Ordering diffusion > clamp_only > no_diffusion holds at every level and attack.
- Same-class edges are more recoverable than different-class edges (gap ~0.08-0.09, growing with damage).
- p_e declines mildly with damage (random 0.870 -> 0.861); targeted attacks start higher (edges between
  high-degree nodes are easier to recover) and decline faster.
- Caveat: the share of p_e exactly 1 depends on n_e (edges removed in more series get more draws); compare
  levels with expected_p_e or at matched n_e.
