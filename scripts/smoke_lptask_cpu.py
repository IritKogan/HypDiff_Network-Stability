from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel


def main() -> None:
    torch.manual_seed(1432)
    args = SimpleNamespace(
        manifold="PoincareBall",
        c=0.6,
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
    )
    encoder = StabilityLPModel(args).cpu().eval()
    denoiser = StabilityDiffusion(SimpleNamespace(r=args.r, t=args.t), dim=args.dim).cpu().eval()

    features = torch.tensor(
        [
            [0.1, 0.2, 0.0],
            [0.2, -0.1, 0.1],
            [-0.1, 0.1, 0.2],
            [0.0, -0.2, 0.1],
        ]
    )
    train_edges = ((0, 1), (1, 2), (2, 3))
    rows = list(range(4))
    cols = list(range(4))
    for source, target in train_edges:
        rows.extend((source, target))
        cols.extend((target, source))

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=features.dtype)
    adjacency = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1)
    normalized_values = adjacency.values() / torch.sqrt(
        degree[adjacency.indices()[0]] * degree[adjacency.indices()[1]]
    )
    adjacency = torch.sparse_coo_tensor(
        adjacency.indices(), normalized_values, adjacency.shape
    ).coalesce()

    evaluation_edges = torch.tensor([[0, 3], [0, 2]], dtype=torch.long)
    observed_edges = set(zip(adjacency.indices()[0].tolist(), adjacency.indices()[1].tolist()))
    assert all(
        (source, target) not in observed_edges
        and (target, source) not in observed_edges
        for source, target in evaluation_edges.tolist()
    )

    with torch.no_grad():
        embeddings, _, _, _ = encoder.encode(features, adjacency)
        prototypes = denoiser.cal_u0([embeddings.clone()])
        labels = torch.zeros(features.shape[0], dtype=torch.long)
        recovered = denoiser.p_sample_loop(
            embeddings.clone(), adjacency, labels, prototypes
        )
        probabilities = encoder.decode(recovered, evaluation_edges)

    assert embeddings.shape == recovered.shape == (4, args.dim)
    assert probabilities.shape == (evaluation_edges.shape[0],)
    assert all(
        torch.isfinite(tensor).all()
        for tensor in (embeddings, recovered, probabilities)
    )
    assert all(
        tensor.device.type == "cpu"
        for tensor in (embeddings, recovered, probabilities)
    )
    assert torch.all((probabilities >= 0) & (probabilities <= 1))

    print("Synthetic CPU lptask integration: PASS")
    print(
        f"encoder={tuple(embeddings.shape)}, "
        f"sampler={tuple(recovered.shape)}, "
        f"decoder={tuple(probabilities.shape)}"
    )
    print("Evaluation edges excluded from encoder adjacency: PASS")


if __name__ == "__main__":
    main()
