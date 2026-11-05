from types import SimpleNamespace

import numpy as np

from stability.sbm_ba_recovery_pilot import (
    _evaluation_sampling_summary,
    generate_pilot_graph,
    make_case_plan,
)


def test_pilot_graphs_retain_disconnected_sbm_and_ba_is_connected():
    disconnected = generate_pilot_graph("SBM", "evaluation", 52002)
    connected = generate_pilot_graph("BA", "evaluation", 62001)

    assert disconnected.initial_component_count == 2
    assert disconnected.initial_lcc_fraction == 23 / 24
    assert connected.initial_component_count == 1
    assert connected.initial_lcc_fraction == 1.0


def test_case_plan_has_fixed_levels_and_distinct_random_masks():
    graph = generate_pilot_graph("BA", "calibration", 61001)

    cases = make_case_plan(graph, graph_index=0, split_code=1)

    assert len(cases) == 45
    for fraction in (0.1, 0.2, 0.3, 0.4, 0.5):
        level_cases = [
            case
            for case in cases
            if case["damage_fraction_requested"] == fraction
        ]
        assert sum(
            case["attack_type"] == "targeted_degree_product"
            for case in level_cases
        ) == 1
        random_cases = [
            case
            for case in level_cases
            if case["attack_type"] == "uniform_random"
        ]
        assert len(random_cases) == 8
        assert len(
            {
                frozenset(frozenset(edge) for edge in case["removed_edges"])
                for case in random_cases
            }
        ) == 8


def test_sampling_summary_accepts_json_lists_for_baseline_probabilities():
    graph = SimpleNamespace(
        graph=SimpleNamespace(node_ids=("a", "b", "c"), edges=(("a", "b"),))
    )
    case = {
        "attack_type": "uniform_random",
        "damage_fraction_requested": 0.1,
        "removed_edges": [["a", "b"]],
        "baseline_probabilities": [0.9, 0.9, 0.1],
        "partial_probabilities": [
            [0.9 if index < 4 else 0.1, 0.9 if index < 2 else 0.1, 0.1]
            for index in range(8)
        ],
    }

    result = _evaluation_sampling_summary(
        [{"graph_id": "fixture", "cases": [case]}],
        {"partial_recovery": 0.85, "no_diffusion_baseline": 0.85},
        {"fixture": graph},
        seed=1,
    )[0]["method_metrics"]

    assert np.isclose(result["partial_recovery"]["score"]["mean"], 0.5)
    assert np.isclose(result["partial_recovery"]["fp"]["mean"], 0.25)
    assert np.isclose(result["partial_recovery"]["fpr"]["mean"], 0.125)
    assert result["no_diffusion_baseline"]["score"]["mean"] == 1.0
    assert result["no_diffusion_baseline"]["fp"]["mean"] == 1.0
    assert result["no_diffusion_baseline"]["fpr"]["mean"] == 0.5
    assert result["no_diffusion_baseline"]["score"]["sampling_ci95"] is None
