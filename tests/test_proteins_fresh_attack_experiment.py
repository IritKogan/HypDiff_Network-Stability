import itertools

import torch

from stability.proteins_dataset import ProteinsGraph
from stability.proteins_fresh_attack_experiment import (
    DAMAGE_FRACTIONS,
    random_edge_masks,
    select_calibration_threshold,
    select_fresh_splits,
)


def _graph(index):
    node_ids = tuple(f"g{index}:n{node}" for node in range(4))
    return ProteinsGraph(
        dataset_index=index,
        content_sha256=f"hash-{index}",
        node_ids=node_ids,
        features=torch.ones((4, 2), dtype=torch.float32),
        edges=tuple((node_ids[i], node_ids[i + 1]) for i in range(3)),
    )


def test_fresh_selection_is_seeded_disjoint_and_excludes_prior_graphs():
    graphs = [_graph(index) for index in range(20)]

    calibration, evaluation, eligible, distinct = select_fresh_splits(graphs, seed=3)
    calibration_again, evaluation_again, _, _ = select_fresh_splits(graphs, seed=3)

    assert len(calibration) == len(evaluation) == 5
    assert eligible == distinct == 20
    assert {graph.graph_id for graph in calibration}.isdisjoint(
        {graph.graph_id for graph in evaluation}
    )
    assert [graph.graph_id for graph in calibration] == [
        graph.graph_id for graph in calibration_again
    ]
    assert [graph.graph_id for graph in evaluation] == [
        graph.graph_id for graph in evaluation_again
    ]


def test_random_masks_are_reproducible_distinct_and_fixed_size():
    graph = _graph(1)
    edges = tuple(itertools.combinations(graph.node_ids, 2))

    first = random_edge_masks(graph.node_ids, edges, 2, seed=42, mask_count=8)
    second = random_edge_masks(graph.node_ids, edges, 2, seed=42, mask_count=8)

    assert first == second
    assert len({frozenset(mask) for _, mask in first}) == 8
    assert all(len(mask) == 2 for _, mask in first)


def test_calibration_selects_nonzero_threshold_under_both_fpr_caps():
    node_ids = tuple(f"n{i}" for i in range(5))
    candidates = tuple(itertools.combinations(node_ids, 2))
    original = (
        (node_ids[0], node_ids[1]),
        (node_ids[0], node_ids[2]),
        (node_ids[1], node_ids[2]),
        (node_ids[1], node_ids[3]),
        (node_ids[2], node_ids[3]),
        (node_ids[3], node_ids[4]),
    )
    cases = []
    for fraction in DAMAGE_FRACTIONS:
        for attack_type, mask_count in (
            ("targeted_degree_product", 1),
            ("uniform_random", 8),
        ):
            for mask_index in range(mask_count):
                removed = (original[mask_index % len(original)],)
                partial = [
                    0.9 if frozenset(edge) == frozenset(removed[0]) else 0.1
                    for edge in candidates
                ]
                baseline = [
                    0.8 if frozenset(edge) == frozenset(removed[0]) else 0.1
                    for edge in candidates
                ]
                cases.append(
                    {
                        "damage_fraction_requested": fraction,
                        "attack_type": attack_type,
                        "removed_edges": [list(removed[0])],
                        "candidate_edges": [list(edge) for edge in candidates],
                        "partial_probabilities": [partial] * 8,
                        "baseline_probabilities": baseline,
                    }
                )
    result = select_calibration_threshold(
        [
            {
                "graph_id": "test",
                "node_ids": node_ids,
                "original_edges": [list(edge) for edge in original],
                "cases": cases,
            }
        ]
    )

    assert result["status"] == "threshold_selected"
    assert result["tau"] == 0.9
    assert result["calibration_partial_recovery_book_score"] > 0
    assert result["calibration_partial_recovery_mean_fpr"] == 0
    assert result["calibration_no_diffusion_mean_fpr"] == 0
