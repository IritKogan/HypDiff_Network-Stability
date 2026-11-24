"""Community context: hyperbolic K-means on the TRAIN-graph embeddings (Decision 2: no leakage).

The encoder sees only train edges; the communities are computed once from its
embeddings of the train graph and then kept fixed for every damaged graph.
Class labels are never used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from hypstab.config import HypstabConfig
from hypstab.models.kmeans import KMeansResult, hyperbolic_kmeans
from hypstab.training.autoencoder import HyperbolicAutoencoder


@dataclass(frozen=True)
class CommunityContext:
    labels: Tensor       # (n,) community of every node
    centers: Tensor      # (k, dim) community centres in the ball
    kmeans: KMeansResult

    def summary(self) -> dict:
        sizes = torch.bincount(self.labels, minlength=self.centers.shape[0]).tolist()
        return {
            "sizes": sizes,
            "inertia": self.kmeans.inertia,
            "iterations": self.kmeans.iterations,
            "converged": self.kmeans.converged,
            "center_norms": [round(float(v), 6) for v in self.centers.norm(dim=-1)],
        }


@torch.no_grad()
def fit_communities(
    model: HyperbolicAutoencoder, features: Tensor, train_adjacency: Tensor, config: HypstabConfig
) -> CommunityContext:
    model.eval()
    embeddings = model.encode(features, train_adjacency)
    com = config.community
    result = hyperbolic_kmeans(
        embeddings,
        k=com.n_clusters,
        c=config.geometry.curvature,
        seed=com.seed,
        max_iter=com.max_iter,
        frechet_steps=com.frechet_steps,
        frechet_tol=com.frechet_tol,
    )
    return CommunityContext(labels=result.labels.to(features.device),
                            centers=result.centers.to(features.device), kmeans=result)
