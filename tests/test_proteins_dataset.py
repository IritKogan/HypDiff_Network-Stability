import pytest
import torch
from torch_geometric.data import Data

from stability.proteins_dataset import adapt_graph, select_graph_splits


def _cycle_graph(index: int, feature_offset: float = 0.0):
    node_count = 12
    sources = list(range(node_count))
    targets = [(node + 1) % node_count for node in range(node_count)]
    edge_index = torch.tensor(
        [sources + targets, targets + sources], dtype=torch.long
    )
    features = (
        torch.arange(node_count * 4, dtype=torch.float32).reshape(node_count, 4)
        / 100
        + feature_offset
    )
    return Data(x=features, edge_index=edge_index, num_nodes=node_count)


def test_adapter_preserves_node_rows_and_canonicalizes_undirected_edges():
    original = _cycle_graph(0)
    graph = adapt_graph(original, dataset_index=17)
    assert graph is not None
    assert graph.node_ids[0] == "PROTEINS-0017:n000"
    assert graph.node_ids[-1] == "PROTEINS-0017:n011"
    assert torch.equal(graph.features, original.x)
    assert len(graph.edges) == 12
    assert len({frozenset(edge) for edge in graph.edges}) == 12


def test_adapter_rejects_self_loops_disconnected_graphs_and_invalid_features():
    self_loop = _cycle_graph(0)
    self_loop.edge_index = torch.cat(
        (self_loop.edge_index, torch.tensor([[0], [0]], dtype=torch.long)), dim=1
    )
    assert adapt_graph(self_loop, 0) is None

    disconnected = Data(
        x=torch.ones((12, 4)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=12,
    )
    assert adapt_graph(disconnected, 1) is None

    invalid = _cycle_graph(0)
    invalid.x[0, 0] = float("nan")
    assert adapt_graph(invalid, 2) is None


def test_graph_split_is_reproducible_distinct_and_disjoint():
    graphs = tuple(
        adapt_graph(_cycle_graph(index, feature_offset=index / 10), index)
        for index in range(10)
    )
    first = select_graph_splits(graphs)
    second = select_graph_splits(graphs)
    assert [g.content_sha256 for g in first.calibration] == [
        g.content_sha256 for g in second.calibration
    ]
    assert [g.content_sha256 for g in first.evaluation] == [
        g.content_sha256 for g in second.evaluation
    ]
    assert len(first.calibration) == 3
    assert len(first.evaluation) == 5
    assert not (
        {g.content_sha256 for g in first.calibration}
        & {g.content_sha256 for g in first.evaluation}
    )


def test_graph_split_fails_instead_of_relaxing_insufficient_eligibility():
    graphs = tuple(
        adapt_graph(_cycle_graph(index, feature_offset=index), index)
        for index in range(7)
    )
    with pytest.raises(ValueError, match="required"):
        select_graph_splits(graphs)
