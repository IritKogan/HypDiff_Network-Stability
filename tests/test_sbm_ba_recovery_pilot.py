from types import SimpleNamespace

import networkx as nx
import numpy as np

from stability.sbm_ba_recovery_pilot import (
    _evaluation_sampling_summary,
    generate_pilot_graph,
    make_case_plan,
)

SBM_SEEDS = (52001, 52002, 52003, 52004, 52005)
BA_SEEDS = (62001, 62002, 62003, 62004, 62005)


def test_pilot_graphs_retain_disconnected_sbm(monkeypatch):
    """A disconnected SBM draw is kept as-is and its components are reported.

    The generator is replaced by a fixed graph so the test does not depend on
    which seed happens to give a disconnected graph in a given networkx version.
    """
    disconnected = nx.path_graph(23)
    disconnected.add_node(23)  # isolated node -> 2 components, LCC = 23/24
    monkeypatch.setattr(nx, "stochastic_block_model", lambda *args, **kwargs: disconnected)

    pilot_graph = generate_pilot_graph("SBM", "evaluation", 52002)

    assert len(pilot_graph.graph.node_ids) == 24
    assert len(pilot_graph.graph.edges) == 22
    assert pilot_graph.initial_component_count == 2
    assert pilot_graph.initial_lcc_fraction == 23 / 24


def test_pilot_graph_component_summary_matches_its_edges():
    for family, seeds in (("SBM", SBM_SEEDS), ("BA", BA_SEEDS)):
        for seed in seeds:
            pilot_graph = generate_pilot_graph(family, "evaluation", seed)
            rebuilt = nx.Graph()
            rebuilt.add_nodes_from(pilot_graph.graph.node_ids)
            rebuilt.add_edges_from(pilot_graph.graph.edges)
            sizes = [len(c) for c in nx.connected_components(rebuilt)]

            assert pilot_graph.initial_component_count == len(sizes)
            assert pilot_graph.initial_lcc_fraction == max(sizes) / 24


def test_ba_pilot_graphs_are_connected():
    # Barabasi-Albert with m=2 is connected by construction (any networkx version).
    for seed in BA_SEEDS:
        connected = generate_pilot_graph("BA", "evaluation", seed)
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
