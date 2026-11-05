# Development experiments

## Damaged-input autoencoder variant

The paired CPU diagnostic used three fresh 12-node synthetic graphs
(seeds 74001–74003), 30 autoencoder epochs per variant, matched initialization,
and identical graph features, reconstruction targets, decoder architecture, and
evaluation perturbations. The experimental variant trained with a new random
three-edge deletion in the encoder adjacency each epoch while retaining the
intact graph as the reconstruction target. Evaluation was without diffusion.

| Graph seed | Intact-input AP | Damaged-input AP | Intact-input removed edges in top 3 | Damaged-input removed edges in top 3 |
|---|---:|---:|---:|---:|
| 74001 | 0.0928 | 0.1315 | 0 | 0 |
| 74002 | 0.1368 | 0.1199 | 0 | 0 |
| 74003 | 0.1564 | 0.1002 | 1 | 0 |
| Mean | 0.1287 | 0.1172 | 0.33 | 0 |

The augmented variant improved AP on one of three graphs and worsened it on
two; it did not improve top-three recovery on any graph. The variant was not
integrated. The result is a small descriptive development experiment, not
evidence about generalization to unseen graphs.
