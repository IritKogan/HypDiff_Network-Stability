from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hyperbolic_learning.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from models.base_models import LPModel
from models.direction_diffusion import diffusion
from stability.edge_perturbation import remove_edges
from stability.partial_recovery import (
    compute_community_context,
    reconstruct_k,
    score_reconstructions,
)
from stability.pilot_config import PilotConfig


def make_adjacency(node_count, edges):
    rows = list(range(node_count))
    cols = list(range(node_count))
    for source, target in edges:
        rows.extend((source, target))
        cols.extend((target, source))
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices, values, (node_count, node_count)
    ).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1)
    normalized_values = adjacency.values() / torch.sqrt(
        degree[adjacency.indices()[0]] * degree[adjacency.indices()[1]]
    )
    return torch.sparse_coo_tensor(
        adjacency.indices(), normalized_values, adjacency.shape
    ).coalesce()


def main() -> None:
    torch.manual_seed(1432)
    np.random.seed(19)
    config = PilotConfig(curvature=1.0, tau=0.5)
    args = config.apply_to_args(SimpleNamespace(
        manifold="PoincareBall",
        c=config.curvature,
        cuda=-1,
        device="cpu",
        feat_dim=3,
        dim=4,
        model="HGCN",
        n_nodes=4,
        num_layers=2,
        act="relu",
        dropout=0.0,
        bias=0,
        use_att=0,
        local_agg=0,
        r=2.0,
        t=1.0,
        nb_false_edges=1,
        nb_edges=2,
        hid1=8,
        hid2=4,
    ))
    encoder = LPModel(args).cpu().eval()
    denoiser = diffusion(
        args, dim=args.dim, curvature=config.curvature
    ).cpu().eval()
    clusterer = HyperbolicKMeans(n_clusters=2, curvature=config.curvature)
    node_ids = ("a", "b", "c", "d")
    features = torch.tensor(
        [
            [0.1, 0.2, 0.0],
            [0.2, -0.1, 0.1],
            [-0.1, 0.1, 0.2],
            [0.0, -0.2, 0.1],
        ]
    )
    original_edges = (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))
    original_adjacency = make_adjacency(4, ((0, 1), (1, 2), (2, 3), (3, 0)))

    context = compute_community_context(
        encoder, features, original_adjacency, clusterer, node_ids
    )
    index = {node: position for position, node in enumerate(node_ids)}
    perturbed = remove_edges(node_ids, features, original_edges, 1, seed=4)
    damaged_adjacency = make_adjacency(
        4,
        tuple(
            (index[source], index[target]) for source, target in perturbed.edges
        ),
    )
    with torch.no_grad():
        damaged_embeddings = encoder.encode(features, damaged_adjacency)[2]

    candidates = tuple(
        (source, target)
        for i, source in enumerate(node_ids)
        for target in node_ids[i + 1 :]
    )
    reconstructions = reconstruct_k(
        encoder,
        denoiser,
        damaged_embeddings,
        damaged_adjacency,
        context,
        node_ids,
        candidates,
        t_star=2,
        num_reconstructions=4,
        seed=23,
        config=config,
    )
    score = score_reconstructions(reconstructions, perturbed.removed_edges)
    assert len(reconstructions.edge_sets) == 4
    assert all(
        torch.isfinite(probabilities).all()
        for probabilities in reconstructions.probabilities
    )
    assert 0.0 <= score.scenario_score <= 1.0
    print("Synthetic CPU partial-recovery smoke: PASS")
    print(
        f"curvature={config.curvature}, reconstructions={len(reconstructions.edge_sets)}, "
        f"recovered-edge score={score.scenario_score:.3f}"
    )


if __name__ == "__main__":
    main()
