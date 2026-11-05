import unittest

import torch

from stability.edge_perturbation import remove_edges


class EdgePerturbationTests(unittest.TestCase):
    def setUp(self):
        self.node_ids = ("a", "b", "c", "d")
        self.features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
        )
        self.edges = (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))

    def test_removal_preserves_nodes_features_and_reports_connectivity(self):
        result = remove_edges(self.node_ids, self.features, self.edges, 1, seed=17)

        self.assertEqual(result.node_ids, self.node_ids)
        self.assertTrue(torch.equal(result.features, self.features))
        self.assertIsNot(result.features, self.features)
        self.assertEqual(len(result.removed_edges), 1)
        self.assertEqual(len(result.edges), 3)
        self.assertTrue(result.is_connected)
        self.assertFalse(set(result.edges) & set(result.removed_edges))

    def test_fixed_seed_is_repeatable_and_removal_can_report_disconnection(self):
        first = remove_edges(self.node_ids, self.features, self.edges, 2, seed=3)
        second = remove_edges(self.node_ids, self.features, self.edges, 2, seed=3)
        self.assertEqual(first.removed_edges, second.removed_edges)
        self.assertFalse(first.is_connected)

    def test_reversed_duplicate_edges_are_one_undirected_edge(self):
        result = remove_edges(
            ("a", "b", "c"),
            torch.ones((3, 1)),
            (("a", "b"), ("b", "a"), ("b", "c")),
            1,
            seed=1,
        )
        self.assertEqual(len(result.edges) + len(result.removed_edges), 2)

    def test_rejects_invalid_removal_count(self):
        with self.assertRaisesRegex(ValueError, "cannot remove"):
            remove_edges(self.node_ids, self.features, self.edges, 5, seed=1)


if __name__ == "__main__":
    unittest.main()
