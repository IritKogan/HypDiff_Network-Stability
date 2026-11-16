# Phase B source and environment baseline

## Source provenance

- Upstream: `RingBDStack/HypDiff` at commit
  `1620e3cc6ac7a558055e9888257568986765f082`.
- Project repository: `IritKogan/HypDiff_Network-Stability`, local branch
  `main` at `b71c6ab3156cf53dff1badd191f8f06c158b8b83`.
- The public project repository's `HypDiff-main.zip` (GitHub blob
  `16672a23538c2f25a90333d6a7b65c057ca3e946`) was compared with the
  pinned upstream archive in a temporary directory outside the workspace.
  All 157 files matched byte-for-byte. The 85 tracked files in the local
  checkout also matched the pinned upstream tree byte-for-byte. The two
  archives each contain 60 Python files.
- The local worktree was clean before this baseline. No upstream source file
  was changed for WP1.

The local Python 3.11.9 environment imported the selected node-level `lptask`
modules. Its baseline direct dependencies are pinned in
`requirements-phaseb.txt`; DGL is omitted because it belongs to the separate
graph-set path, not this `lptask` import path. No packages were installed and
no global environment was changed.

For a future isolated Windows environment, use:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-phaseb.txt
.\.venv\Scripts\python scripts\validate_environment.py
```

These commands were documented but not run. The available host has a CPU-only
PyTorch build. A GPU environment needs a compatible, explicitly pinned
PyTorch/CUDA wheel and a separate validation; neither is specified or tested
by this CPU baseline.

## Environment observed

| Component | Observed |
|---|---|
| OS | Windows |
| Python | 3.11.9 |
| PyTorch | 2.12.0+cpu; CUDA unavailable |
| PyTorch Geometric | 2.8.0.post1 |
| NumPy / SciPy | 2.4.4 / 1.17.1 |
| NetworkX / scikit-learn | 3.6.1 / 1.8.0 |
| tqdm / pandas / matplotlib | 4.67.3 / 3.0.2 / 3.10.3 |
| RAM / processors | 16 GB / 8 logical processors |
| `nvidia-smi` | unavailable |

`scripts\validate_environment.py` checks these pinned imports and imports the
selected project modules without invoking the training entry point.

## CPU smoke checks

Executed from the repository root without training or downloading data:

- Poincare-ball `expmap` / `logmap` round trip: passed for a small finite
  tangent vector.
- Node-level HGCN encoding and Fermi-Dirac decoding: passed on a four-node
  CPU graph; output latent shape `(4, 8)` and three finite edge probabilities.
- Import smoke check for `config`, `lp_train`, `models.base_models`,
  `models.direction_diffusion`, `utils.data_utils`, and
  `hyperbolic_learning.hyperkmeans`: passed.
- Directional denoiser construction: blocked on CPU by the unconditional
  `.cuda()` calls in `models/direction_diffusion.py` while initializing its
  schedule tensors. This is an upstream device-compatibility issue, not a
  failed model-quality experiment. No patch was applied.

The Poincare code emitted a deprecation warning for a `uint8` condition in
`torch.where`; it did not prevent the smoke check. This warning is recorded
only; no unrelated fix is included in WP1.

## Benchmark path clarification

The book's 1,000-step / IMDB-B performance target describes the proposed
recovery module; it does not identify a runnable upstream path. In the paper,
IMDB-B is used for **graph generation** efficiency. Upstream exposes two
different paths:

- `lptask`: node-level HGCN and Fermi-Dirac link decoding, the plausible
  starting point for the book's fixed-node recovery. Its
  `utils/data_utils.py::load_data_lp` does not load IMDB-B or PROTEINS.
- `graphtask`: graph-collection loading includes IMDB-B and PROTEINS, but
  uses a graph-pooled `HAveEncoder` and fully connected decoder, not the
  node-level HGCN/Fermi-Dirac recovery path.

Both `models/direction_diffusion.py::diffusion` and
`models/direction_diffusionset.py::diffusionset` currently hard-code 200
steps. Thus the book's 1,000-step benchmark cannot be attributed to an
existing, end-to-end Phase B implementation. Selecting or adapting a path,
dataset interpretation, and step-count procedure is deferred for the next
research review; no benchmark was run in WP1.

## Upstream boundary (2026-11-16)

All upstream HypDiff files are byte-identical to RingBDStack/HypDiff @
`1620e3cc6ac7a558055e9888257568986765f082` (line endings aside);
`tests/test_hypdiff_models.py` enforces this with SHA-256 hashes in
`tests/data/upstream_hypdiff_1620e3c.sha256`. The only remaining deviation is
`.gitignore` (project data/artifact paths), pending a decision.

Project changes to model code live in `stability/`:

| Upstream | Project replacement | Change |
|---|---|---|
| `models/direction_diffusion.py::diffusion` | `stability/hypdiff_models.py::StabilityDiffusion` | CPU-safe init (`.to(device)`), curvature parameter, `noise_formula` option (`exact` / `upstream`), t* partial recovery (`sample_from_tstar`) |
| `models/base_models.py::LPModel` + `models/encoders.py::HGCN` | `stability/hypdiff_models.py::StabilityLPModel` | optional `final_activation="leaky_relu"` on the last HGCN layer |
| `hyperbolic_learning/hyperbolic_kmeans/{hkmeans,util_hk}.py` | `stability/hyperbolic_kmeans/` (documented copy) | gradient precedence fix, explicit errors on non-finite values, curvature parameter |

The CPU denoiser block described above is resolved by `StabilityDiffusion`.
The refactor was verified to reproduce the previous code bit-for-bit (encoder
and denoiser weights, embeddings, K-means labels/centroids, training loss,
t*=7 and t*=50 recoveries, for both final activations).
