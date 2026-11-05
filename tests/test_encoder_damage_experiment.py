import pytest

from stability.encoder_damage_experiment import sample_training_mask
from stability.synthetic_instances import (
    ENCODER_DAMAGE_SEEDS,
    generate_synthetic_graph,
)


def test_training_mask_is_fixed_size_seeded_and_excludes_evaluation_mask():
    graph = generate_synthetic_graph(ENCODER_DAMAGE_SEEDS[0], "encoder_damage")
    evaluation_mask = frozenset(frozenset(edge) for edge in graph.edges[:3])
    first = sample_training_mask(graph.edges, 123, evaluation_mask)
    second = sample_training_mask(graph.edges, 123, evaluation_mask)
    assert first == second
    assert len(first) == 3
    assert frozenset(frozenset(edge) for edge in first) != evaluation_mask


def test_encoder_damage_graph_seeds_are_fresh_instances():
    graphs = [
        generate_synthetic_graph(seed, "encoder_damage")
        for seed in ENCODER_DAMAGE_SEEDS
    ]
    assert len({graph.graph_id for graph in graphs}) == 3
    assert len({graph.edges for graph in graphs}) == 3
    with pytest.raises(ValueError, match="not assigned"):
        generate_synthetic_graph(73001, "encoder_damage")
