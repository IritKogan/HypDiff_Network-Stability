# PROTEINS random-versus-targeted stability comparison

## Scope and decoder limit

This post-hoc development comparison adds uniform random edge removal to the
completed degree-product targeted-attack comparison on the same five existing
PROTEINS evaluation graphs. It does not replace the targeted-attack comparison
in Section 4.3 of the project book, and these reused evaluation graphs are not an
independent test set.

The configured Fermi-Dirac decoder is

`P(d) = 1 / (exp((d - r) / t) + 1)`,

where `d` is a squared Poincare distance and is nonnegative. Every saved
checkpoint has `r = 2.0` and `t = 1.0`, so the maximum is
`sigmoid(2) = 0.8807970779778823` (float32: `0.8807970285415649`). The saved
baseline and partial-recovery probabilities reach that float32 maximum. With
the frozen `tau = 1.0`, no candidate can be decoded as an edge, for either
attack. Thus all targeted book scores are zero, and the random-attack scores
are zero by the same bound. No reconstruction samples were needed for the
random masks; reporting those scores does not imply a new recovery run.

This is a negative result for probabilistic recovery at the frozen operating
point: it cannot compare score sensitivity to attack type or recovery
performance. It does not imply that the underlying sub-threshold probabilities
are identical.

## Random-mask protocol

For each graph and each requested damage fraction (10%, 20%, 30%, 40%, 50%), we
sampled eight edge subsets uniformly without replacement from the intact
undirected edge set. The removed-edge count matches the targeted comparison:
`floor(p * E + 0.5)` (nearest integer, half up). Disconnected masks were
retained; there was no connectivity filter.

Random seeds were fixed by
`20261106 + graph_index * 10000 + damage_level_index * 100 + mask_index`,
with graph indices zero-based in the evaluation artifact order, damage-level
indices one-based, and mask indices zero-based. The random comparison records
all seeds and removed edges. The targeted values below are taken unchanged
from the saved deterministic targeted comparison.

Algebraic connectivity is the second-smallest eigenvalue of the unweighted
combinatorial Laplacian over all original nodes, set to zero when disconnected.
Mean node betweenness is the arithmetic mean of normalized unweighted node
betweenness over all original nodes (`endpoints=False`). Random values are
shown as mean [minimum, maximum] over eight masks; targeted values are the
single deterministic mask.

## Results

| Graph | Damage (edges) | Random disconnected | Target disconnected | Random lambda2 mean [min, max] | Target lambda2 | Random betweenness mean [min, max] | Target betweenness |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0555 | 10% (9) | 0/8 | No | 0.0249 [0.0224, 0.0266] | 0.0262 | 0.1412 [0.1346, 0.1466] | 0.1446 |
| 0555 | 20% (18) | 5/8 | No | 0.0080 [0.0000, 0.0232] | 0.0226 | 0.0935 [0.0235, 0.1564] | 0.1532 |
| 0555 | 30% (26) | 8/8 | No | 0.0000 [0.0000, 0.0000] | 0.0200 | 0.0696 [0.0253, 0.1449] | 0.1658 |
| 0555 | 40% (35) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0379 [0.0164, 0.0808] | 0.0907 |
| 0555 | 50% (44) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0205 [0.0059, 0.0796] | 0.0562 |
| 0395 | 10% (4) | 2/8 | No | 0.0491 [0.0000, 0.0747] | 0.0681 | 0.1536 [0.0515, 0.2012] | 0.1904 |
| 0395 | 20% (7) | 4/8 | Yes | 0.0242 [0.0000, 0.0516] | 0.0000 | 0.1235 [0.0108, 0.2105] | 0.1728 |
| 0395 | 30% (11) | 6/8 | Yes | 0.0114 [0.0000, 0.0481] | 0.0000 | 0.1022 [0.0553, 0.2304] | 0.0678 |
| 0395 | 40% (14) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0844 [0.0184, 0.2102] | 0.0515 |
| 0395 | 50% (18) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0241 [0.0070, 0.0404] | 0.0591 |
| 0215 | 10% (6) | 5/8 | No | 0.0072 [0.0000, 0.0207] | 0.0198 | 0.1072 [0.0477, 0.2007] | 0.2040 |
| 0215 | 20% (13) | 7/8 | Yes | 0.0018 [0.0000, 0.0144] | 0.0000 | 0.0816 [0.0160, 0.2236] | 0.0334 |
| 0215 | 30% (19) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0620 [0.0091, 0.1342] | 0.0354 |
| 0215 | 40% (26) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0233 [0.0091, 0.0493] | 0.0218 |
| 0215 | 50% (32) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0213 [0.0033, 0.0600] | 0.0204 |
| 0925 | 10% (7) | 3/8 | No | 0.0185 [0.0000, 0.0305] | 0.0231 | 0.1225 [0.0367, 0.1677] | 0.1808 |
| 0925 | 20% (13) | 4/8 | No | 0.0122 [0.0000, 0.0287] | 0.0164 | 0.1265 [0.0292, 0.1849] | 0.2230 |
| 0925 | 30% (20) | 8/8 | No | 0.0000 [0.0000, 0.0000] | 0.0140 | 0.0354 [0.0118, 0.0895] | 0.2376 |
| 0925 | 40% (27) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0244 [0.0122, 0.0423] | 0.0084 |
| 0925 | 50% (34) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0192 [0.0072, 0.0469] | 0.0069 |
| 1086 | 10% (4) | 1/8 | No | 0.0342 [0.0000, 0.0436] | 0.0412 | 0.1861 [0.0977, 0.2017] | 0.2103 |
| 1086 | 20% (8) | 5/8 | No | 0.0123 [0.0000, 0.0372] | 0.0344 | 0.1280 [0.0456, 0.2309] | 0.2246 |
| 1086 | 30% (12) | 6/8 | Yes | 0.0065 [0.0000, 0.0265] | 0.0000 | 0.0897 [0.0254, 0.2434] | 0.0675 |
| 1086 | 40% (16) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0462 [0.0124, 0.2111] | 0.0122 |
| 1086 | 50% (21) | 8/8 | Yes | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0131 [0.0025, 0.0272] | 0.0071 |

## Interpretation and limits

Random removal produced variable results across masks. By 40% damage all eight
random masks disconnected every graph; at lower damage levels, disconnection
frequency differed substantially by graph. Targeted degree-product removal did
not uniformly disconnect graphs earlier than random removal: for example,
PROTEINS-0555 remained connected at 30% targeted damage while all eight random
masks were disconnected. Conversely, the targeted attack disconnected
PROTEINS-0395 and PROTEINS-0215 at 20%, earlier than most random masks in those
graphs.

The classical measures also vary by graph and attack type; the small number of
random masks is descriptive, not a basis for a general robustness claim. Since
the frozen book score is identically zero for both attacks, this comparison
cannot establish whether probabilistic recovery responds differently to
random and targeted damage.

## Provenance and preservation

- Random comparison: `artifacts/proteins_pilot/random_vs_targeted_classical_comparison/comparison.json`
- Targeted reference: `artifacts/proteins_pilot/targeted_attack_classical_comparison/comparison.json`
- Official calibration SHA-256:
  `9e31bb2aac1de2be11cc7518a3a63be90e4140a4238dccc279cd568cc8e181c8`
- Official evaluation SHA-256:
  `7ebae213f90f5c11541941114253732a9b39cba7698333570c60d9048a41418e`
- Frozen threshold remains `tau = 1.0`.
- No model was trained, no dataset was downloaded, and no threshold was recalibrated.
