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
