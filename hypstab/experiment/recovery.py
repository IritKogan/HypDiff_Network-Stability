"""Recovery of every damaged graph and the pairwise edge decisions (Decisions 6, 10, 11).

For one attack (its damage series), every series s and damage level l:

1. damaged graph = train + validation edges + the pool edges NOT removed at (s, l);
   only pool edges are removed and evaluated, and no model has seen them (Decision 6);
2. reconstruction k (k = 0 .. K-1) uses model ``models[k]``:
   - source "noise"   : the same model for every k, diffusion noise draw k;
   - source "ensemble": model k (member k) with noise draw k (Decision 11);
   noise seed = recovery_seed(recovery.seed, s, k) -> common random numbers across levels,
   attacks and sources;
3. three methods per reconstruction (Decision 7): diffusion (recovery from t*), clamp_only
   (clamp step alone), no_diffusion (decode the damaged graph's embedding);
4. way A, pairwise decision (Decision 10): a removed edge "reappears" in reconstruction k if its
   logit is strictly higher than that of a reference pair drawn uniformly from the control pairs
   of the same reconstruction; a control pair is compared with another control pair (never
   itself). Reference draws: SeedSequence([recovery.seed, 11, method, s, l, k]) (same as the old
   code; shared by attacks and sources). Logits are compared instead of the old float32
   probabilities: same order, but no ties from probabilities rounded to 1.0.
   Expected value of one draw (no draw noise) = share of control pairs scored strictly lower;
5. way B, average (Decision 11): per damaged graph, the score of a pair is the mean over the K
   reconstructions of F_k(pair) = (number of control pairs scored strictly lower in reconstruction
   k) / C. For K = 1 this orders pairs exactly like the logits. The averaged score of a removed
   edge is compared with ``experiment.way_b_references`` reference control pairs:
   SeedSequence([recovery.seed, 12, method, s, l]).

Returned arrays (L levels, S series, K reconstructions, P pool edges, C control pairs, per method m):
  removed (L,S,P) bool
  hit_<m> (L,S,K,P) bool              way A outcome of every reconstruction
  share_<m> (L,S,K,P) float32         way A expected share (NaN where not removed)
  prob_<m> (L,S,K,P) float32          decoded probability (NaN where not removed)
  control_hits_<m> (L,S,C) int16      way A control outcomes summed over k
  control_share_sum_<m> (L,S,C) f32   way A control expected shares summed over k
  control_prob_k_sum_<m> (L,K,C) f32  control probabilities of reconstruction k summed over series
  wayb_hits_<m> (L,S,P) int16         way B hits over the R references
  wayb_share_<m> (L,S,P) float32      way B expected share (NaN where not removed)
  wayb_score_<m> (L,S,P) float32      way B averaged score F (NaN where not removed)
  control_wayb_hits_<m> (L,S,C) int16 way B control hits over R references
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from hypstab.config import HypstabConfig
from hypstab.data.graph import GraphData
from hypstab.data.split import EdgeSplit
from hypstab.evaluation.validation import recovery_seed
from hypstab.experiment.damage import DamageSeries
from hypstab.experiment.models import ModelBundle
from hypstab.models.clamp import clamp_to_centers
from hypstab.models.hgcn import normalized_adjacency

METHODS = ("diffusion", "clamp_only", "no_diffusion")
METHOD_CODES = {"diffusion": 0, "no_diffusion": 1, "clamp_only": 2}   # same codes as the old code
PAIRWISE_STREAM = 11   # way A reference draws (same tag as the old code)
WAY_B_STREAM = 12      # way B reference draws


def reference_draws(seed: int, method: str, series: int, level: int, k: int, pool_size: int,
                    control_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Way A reference control pair for every pool edge and for every control pair (never itself)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, PAIRWISE_STREAM, METHOD_CODES[method],
                                                        series, level, k]))
    for_pool = rng.integers(0, control_count, size=pool_size)
    for_control = rng.integers(0, control_count - 1, size=control_count)
    for_control = for_control + (for_control >= np.arange(control_count))
    return for_pool, for_control


def way_b_draws(seed: int, method: str, series: int, level: int, references: int, pool_size: int,
                control_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Way B: ``references`` reference control pairs per pool edge and per control pair (never itself)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, WAY_B_STREAM, METHOD_CODES[method],
                                                        series, level]))
    for_pool = rng.integers(0, control_count, size=(references, pool_size))
    for_control = rng.integers(0, control_count - 1, size=(references, control_count))
    for_control = for_control + (for_control >= np.arange(control_count))
    return for_pool, for_control


def lower_counts(sorted_control: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Number of control scores strictly lower than each value."""
    return np.searchsorted(sorted_control, values, side="left")


def pairwise_decision(logit_removed: np.ndarray, logit_control: np.ndarray, ref_removed: np.ndarray,
                      ref_control: np.ndarray):
    """Way A decision of one reconstruction: (hit_removed, hit_control, share_removed, share_control)."""
    count = len(logit_control)
    ordered = np.sort(logit_control)
    share_removed = lower_counts(ordered, logit_removed) / count
    share_control = lower_counts(ordered, logit_control) / (count - 1)
    return (logit_removed > logit_control[ref_removed], logit_control > logit_control[ref_control],
            share_removed, share_control)


def _logits(model: ModelBundle, embeddings: torch.Tensor, pairs: torch.Tensor) -> np.ndarray:
    logits = model.autoencoder.edge_logits(embeddings, pairs)
    if not torch.isfinite(logits).all():
        raise RuntimeError("decoder produced non-finite logits")
    return logits.double().cpu().numpy()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def run_attack(
    graph: GraphData,
    split: EdgeSplit,
    items: Sequence[DamageSeries],
    models: Sequence[ModelBundle],
    config: HypstabConfig,
    device: torch.device,
    log_prefix: str | None = None,
) -> dict[str, np.ndarray]:
    """All series and levels of one attack with reconstruction k through ``models[k]`` (len = K)."""
    rec = config.recovery
    K = rec.reconstructions
    if len(models) != K:
        raise ValueError(f"need one model per reconstruction: {len(models)} models, K = {K}")
    R = config.experiment.way_b_references
    levels = len(config.perturbation.damage_fractions)
    S, P, C = len(items), len(split.pool), len(split.control_pairs)
    if C < 2:
        raise ValueError("the pairwise decision needs at least two control pairs")
    if [item.series_index for item in items] != list(range(S)):
        raise ValueError("series must be numbered 0 .. S-1 in order")
    c = config.geometry.curvature
    features = graph.features.to(device)
    control = torch.as_tensor(split.control_pairs, device=device)
    kept_edges = np.concatenate([split.train, split.val])
    for bundle in models:
        bundle.autoencoder.eval()
        bundle.diffusion.eval()

    out: dict[str, np.ndarray] = {"removed": np.zeros((levels, S, P), dtype=bool)}
    for m in METHODS:
        out[f"hit_{m}"] = np.zeros((levels, S, K, P), dtype=bool)
        out[f"share_{m}"] = np.full((levels, S, K, P), np.nan, dtype=np.float32)
        out[f"prob_{m}"] = np.full((levels, S, K, P), np.nan, dtype=np.float32)
        out[f"control_hits_{m}"] = np.zeros((levels, S, C), dtype=np.int16)
        out[f"control_share_sum_{m}"] = np.zeros((levels, S, C), dtype=np.float32)
        out[f"control_prob_k_sum_{m}"] = np.zeros((levels, K, C), dtype=np.float32)
        out[f"wayb_hits_{m}"] = np.zeros((levels, S, P), dtype=np.int16)
        out[f"wayb_share_{m}"] = np.full((levels, S, P), np.nan, dtype=np.float32)
        out[f"wayb_score_{m}"] = np.full((levels, S, P), np.nan, dtype=np.float32)
        out[f"control_wayb_hits_{m}"] = np.zeros((levels, S, C), dtype=np.int16)

    for item in items:
        s = item.series_index
        started = time.perf_counter()
        for level in range(levels):
            pool_index = item.removed_pool_indices(level)
            keep = np.ones(P, dtype=bool)
            keep[pool_index] = False
            damaged = np.concatenate([kept_edges, split.pool[keep]])
            adjacency = normalized_adjacency(graph.node_count, damaged, device)
            removed_pairs = torch.as_tensor(split.pool[pool_index], device=device)
            out["removed"][level, s, pool_index] = True

            cache: dict[int, dict] = {}

            def deterministic(bundle: ModelBundle) -> dict:
                """Embedding of the damaged graph + no_diffusion / clamp_only logits, once per model."""
                key = id(bundle)
                if key not in cache:
                    embeddings = bundle.autoencoder.encode(features, adjacency)
                    clamped = clamp_to_centers(embeddings, bundle.context.labels, bundle.context.centers,
                                               c, config.clamp.clip)
                    cache[key] = {
                        "embeddings": embeddings,
                        "no_diffusion": (_logits(bundle, embeddings, removed_pairs),
                                         _logits(bundle, embeddings, control)),
                        "clamp_only": (_logits(bundle, clamped, removed_pairs),
                                       _logits(bundle, clamped, control)),
                    }
                return cache[key]

            score_removed = {m: np.zeros(len(pool_index)) for m in METHODS}   # way B sums of F_k
            score_control = {m: np.zeros(C) for m in METHODS}
            for k, bundle in enumerate(models):
                base = deterministic(bundle)
                generator = torch.Generator(device=device)
                generator.manual_seed(recovery_seed(rec.seed, s, k))
                recovered = bundle.diffusion.recover(base["embeddings"], adjacency, bundle.context.labels,
                                                     bundle.context.centers, rec.t_star, generator,
                                                     rec.sampler, rec.steps)
                logits = {
                    "diffusion": (_logits(bundle, recovered, removed_pairs), _logits(bundle, recovered, control)),
                    "clamp_only": base["clamp_only"],
                    "no_diffusion": base["no_diffusion"],
                }
                for m in METHODS:
                    l_removed, l_control = logits[m]
                    ref_removed, ref_control = reference_draws(rec.seed, m, s, level, k, P, C)
                    hit, control_hit, share, control_share = pairwise_decision(
                        l_removed, l_control, ref_removed[pool_index], ref_control)
                    out[f"hit_{m}"][level, s, k, pool_index] = hit
                    out[f"share_{m}"][level, s, k, pool_index] = share
                    out[f"prob_{m}"][level, s, k, pool_index] = _sigmoid(l_removed)
                    out[f"control_hits_{m}"][level, s] += control_hit
                    out[f"control_share_sum_{m}"][level, s] += control_share
                    out[f"control_prob_k_sum_{m}"][level, k] += _sigmoid(l_control)
                    ordered = np.sort(l_control)
                    score_removed[m] += lower_counts(ordered, l_removed) / C
                    score_control[m] += lower_counts(ordered, l_control) / C
            for m in METHODS:
                comb_removed, comb_control = score_removed[m] / K, score_control[m] / K
                ref_removed, ref_control = way_b_draws(rec.seed, m, s, level, R, P, C)
                hits = (comb_removed[None, :] > comb_control[ref_removed[:, pool_index]]).sum(0)
                control_hits = (comb_control[None, :] > comb_control[ref_control]).sum(0)
                ordered = np.sort(comb_control)
                out[f"wayb_hits_{m}"][level, s, pool_index] = hits
                out[f"wayb_share_{m}"][level, s, pool_index] = lower_counts(ordered, comb_removed) / C
                out[f"wayb_score_{m}"][level, s, pool_index] = comb_removed
                out[f"control_wayb_hits_{m}"][level, s] = control_hits
        if log_prefix:
            print(f"[recovery] {log_prefix} series {s + 1}/{S} ({time.perf_counter() - started:.1f}s)", flush=True)
    return out
