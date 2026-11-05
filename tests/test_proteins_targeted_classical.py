import networkx as nx

from stability.proteins_targeted_classical import (
    graph_classical_metrics,
    rounded_removal_count,
    targeted_edge_order,
)


def test_targeted_order_uses_degree_product_then_canonical_node_order():
    nodes = ("a", "b", "c", "d")
    edges = (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"))

    ordered, degrees = targeted_edge_order(nodes, edges)

    assert degrees == {"a": 3, "b": 2, "c": 2, "d": 1}
    assert ordered == (("a", "b"), ("a", "c"), ("b", "c"), ("a", "d"))


def test_removal_count_rounds_nearest_with_half_up():
    assert rounded_removal_count(15, 0.1) == 2
    assert rounded_removal_count(25, 0.1) == 3


def test_classical_metrics_keep_isolated_nodes_and_zero_disconnected_lambda():
    metrics = graph_classical_metrics(("a", "b", "c"), (("a", "b"),))

    assert metrics["connected"] is False
    assert metrics["algebraic_connectivity_lambda2"] == 0.0
    assert metrics["mean_normalized_node_betweenness"] == 0.0


def test_classical_metrics_match_networkx_on_connected_path():
    nodes = ("a", "b", "c")
    edges = (("a", "b"), ("b", "c"))
    expected_graph = nx.Graph()
    expected_graph.add_nodes_from(nodes)
    expected_graph.add_edges_from(edges)
    expected = nx.betweenness_centrality(
        expected_graph, normalized=True, endpoints=False, weight=None
    )

    metrics = graph_classical_metrics(nodes, edges)

    assert metrics["connected"] is True
    assert metrics["algebraic_connectivity_lambda2"] > 0.0
    assert metrics["mean_normalized_node_betweenness"] == sum(expected.values()) / 3
