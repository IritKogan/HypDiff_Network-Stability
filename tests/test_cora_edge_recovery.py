from __future__ import annotations

import itertools

import torch

from stability.cora_edge_recovery import (
    DAMAGE_FRACTIONS,
    cumulative_removal_series,
    edge_recovery_report,
)
from stability.partial_recovery import CommunityContext, reconstruct_k
from stability.pilot_config import PilotConfig


def test_cora_attack_masks_are_cumulative() -> None:
    node_ids = tuple(f"n{index}" for index in range(8))
    edges = tuple(itertools.combinations(node_ids, 2))[:15]

    series = cumulative_removal_series(node_ids, edges, random_seed=42)

    assert set(series) == {"uniform_random", "targeted_degree_product"}
    for levels in series.values():
        assert [row["damage_fraction_requested"] for row in levels] == list(
            DAMAGE_FRACTIONS
        )
        removed = [set(row["removed_edges"]) for row in levels]
        assert removed[0] <= removed[1] <= removed[2]
        assert [row["edges_removed"] for row in levels] == sorted(
            row["edges_removed"] for row in levels
        )


def test_cora_edge_report_has_all_histogram_bins_and_separate_false_positives() -> None:
    original = (("a", "b"), ("a", "c"), ("b", "c"))
    removed = (("a", "b"), ("a", "c"))
    reconstructions = (
        (("a", "b"), ("a", "d")),
        (("a", "b"), ("a", "c"), ("c", "d")),
        (("b", "c"), ("b", "d")),
    )

    report = edge_recovery_report(removed, reconstructions, original)

    recovery = report["removed_edge_recovery"]
    assert [row["recovered_count"] for row in recovery["histogram"]] == [0, 1, 2, 3]
    assert [row["edge_count"] for row in recovery["histogram"]] == [0, 1, 1, 0]
    assert [row["percentage"] for row in recovery["histogram"]] == [0, 50, 50, 0]
    assert [row["r_over_K"] for row in recovery["edges"]] == ["2/3", "1/3"]
    assert not {
        "scenario_score",
        "book_stability_score",
        "graph_level_stability",
        "final_stability_score",
    } & report.keys()
    assert report["false_positive_additions"] == [
        {"reconstruction": 1, "count": 1, "edges": [["a", "d"]]},
        {"reconstruction": 2, "count": 1, "edges": [["c", "d"]]},
        {"reconstruction": 3, "count": 1, "edges": [["b", "d"]]},
    ]


class _DeterministicEncoder:
    c = torch.tensor([1.0])

    def decode(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        reconstruction = int(embeddings[0, 0].item())
        return torch.tensor(
            [
                0.9 if (int(source) + int(target) + reconstruction) % 3 == 0 else 0.1
                for source, target in edge_index.tolist()
            ],
            dtype=embeddings.dtype,
            device=embeddings.device,
        )


class _DeterministicDenoiser:
    c = 1.0
    num_timesteps = 8

    def __init__(self) -> None:
        self.samples = 0

    def sample_from_tstar(self, damaged_embeddings, **kwargs):
        self.samples += 1
        recovered = damaged_embeddings.clone()
        recovered[0, 0] = self.samples
        return recovered


def test_chunked_lazy_candidates_match_unchunked_for_every_reconstruction() -> None:
    node_ids = ("n0", "n1", "n2", "n3")
    candidates = tuple(itertools.combinations(node_ids, 2))
    removed_edges = (("n0", "n1"), ("n1", "n3"))
    original_edges = (*removed_edges, ("n0", "n3"))
    embedding = torch.zeros((len(node_ids), 1))
    context = CommunityContext(
        node_ids=node_ids,
        labels=torch.zeros(len(node_ids), dtype=torch.long),
        centers=torch.tensor([[0.1, 0.1]]),
        curvature=1.0,
    )
    config = PilotConfig(curvature=1.0, tau=0.5)

    unchunked = reconstruct_k(
        encoder=_DeterministicEncoder(),
        denoiser=_DeterministicDenoiser(),
        damaged_embeddings=embedding,
        damaged_adjacency=torch.empty(0),
        context=context,
        node_ids=node_ids,
        candidate_edges=candidates,
        t_star=7,
        num_reconstructions=3,
        seed=11,
        config=config,
    )
    seen_per_reconstruction = []

    def candidate_factory():
        def generate():
            seen = 0
            for edge in itertools.combinations(node_ids, 2):
                seen += 1
                yield edge
            seen_per_reconstruction.append(seen)

        return generate()

    chunked = reconstruct_k(
        encoder=_DeterministicEncoder(),
        denoiser=_DeterministicDenoiser(),
        damaged_embeddings=embedding,
        damaged_adjacency=torch.empty(0),
        context=context,
        node_ids=node_ids,
        candidate_edges=candidate_factory,
        t_star=7,
        num_reconstructions=3,
        seed=11,
        config=config,
        candidate_chunk_size=2,
        retain_probabilities=False,
        validate_candidates=False,
    )

    assert seen_per_reconstruction == [len(candidates)] * 3
    assert chunked.edge_sets == unchunked.edge_sets
    assert all(not values.numel() for values in chunked.probabilities)
    chunked_report = edge_recovery_report(
        removed_edges, chunked.edge_sets, original_edges
    )
    unchunked_report = edge_recovery_report(
        removed_edges, unchunked.edge_sets, original_edges
    )
    assert [
        row["recovered_count"]
        for row in chunked_report["removed_edge_recovery"]["edges"]
    ] == [
        row["recovered_count"]
        for row in unchunked_report["removed_edge_recovery"]["edges"]
    ]
