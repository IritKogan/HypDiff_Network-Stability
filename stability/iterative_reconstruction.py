"""Iterative (structural) reconstruction — experiment for Decision 9 (2026-11-17).

Motivation: sampling noise in the 128-dim embedding of the damaged graph does not produce
informative uncertainty (docs/conditional-diffusion-design.md §10-14): once the damaged graph is
seen, the embedding posterior is narrow. The real unknown is WHICH edges are missing. Here each
reconstruction samples a set of edges to add back, re-encodes the completed graph and scores again,
so the K reconstructions are genuinely different graphs and choices interact (an added edge changes
the embeddings and hence the scores of nearby pairs).

Selection of the edges to add: Gumbel-top-R' over all candidate pairs (non-edges of the input
graph), i.e. sampling R' pairs without replacement with probability proportional to
exp(logit(p) / temperature). Uses only the input graph and the model's scores — no labels.
"""

from __future__ import annotations

import numpy as np
import torch

EPSILON = 1e-7


def candidate_pairs(node_count: int, edges: np.ndarray) -> np.ndarray:
    """All unordered pairs (i < j) that are not edges of the input graph, shape (M, 2)."""
    edges = np.sort(np.asarray(edges, dtype=np.int64), axis=1)
    present = np.zeros((node_count, node_count), dtype=bool)
    present[edges[:, 0], edges[:, 1]] = True
    rows, cols = np.triu_indices(node_count, k=1)
    keep = ~present[rows, cols]
    return np.stack((rows[keep], cols[keep]), axis=1)


def score_pairs(encoder, embeddings, pairs: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    """Decoder probabilities for many pairs, in chunks (float64)."""
    out = np.empty(len(pairs), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(pairs), chunk):
            batch = torch.as_tensor(pairs[start:start + chunk], device=embeddings.device)
            out[start:start + chunk] = encoder.decode(embeddings, batch).cpu().numpy()
    return out


def gumbel_top_k(probabilities: np.ndarray, count: int, temperature: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Indices of ``count`` items sampled without replacement with weights
    exp(logit(p) / temperature) (Gumbel-top-k trick). temperature -> 0 gives the top-k."""
    if count < 0 or count > len(probabilities):
        raise ValueError("count must be in [0, len(probabilities)]")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    p = np.clip(probabilities, EPSILON, 1 - EPSILON)
    keys = np.log(p / (1 - p))
    if temperature > 0:
        keys = keys / temperature + rng.gumbel(size=len(keys))
    return np.argpartition(-keys, count - 1)[:count]
