"""clamp_only: the clamp step of the recovery alone (no network, no noise).

Used as a baseline next to the diffusion (Decision 7: diffusion, no_diffusion,
clamp_only are reported in every run). For node i with community centre mu_i:

    v_i = logmap_{mu_i}(h_i)                         (tangent space at the centre)
    z   = (v - mean(v)) / std(v)                     (one scalar mean/std over all
                                                      entries, as in the HypDiff code)
    z   = clip(z, -clip, clip)
    h_i' = expmap_{mu_i}( z_i * std(v) + mean(v) )

Coordinates far from the bulk are pulled in, so outlying nodes move toward their
community centre. Same operation as StabilityDiffusion.shrink_to_prototype in the
old code (tested against it).
"""

from __future__ import annotations

from torch import Tensor

from hypstab.geometry import get_geometry


def standardize(v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(v - mean) / std with ONE scalar mean and (unbiased) std over all entries."""
    mean = v.mean()
    std = v.std()
    return (v - mean) / std, mean, std


def clamp_to_centers(embeddings: Tensor, labels: Tensor, centers: Tensor, c: float, clip: float,
                     manifold: str = "poincare") -> Tensor:
    if clip <= 0:
        raise ValueError("clip must be > 0")
    pb = get_geometry(manifold)
    node_centers = centers.to(embeddings)[labels.to(embeddings.device)]
    tangent = pb.logmap(node_centers, embeddings, c)
    z, mean, std = standardize(tangent)
    return pb.expmap(node_centers, z.clamp(-clip, clip) * std + mean, c)
