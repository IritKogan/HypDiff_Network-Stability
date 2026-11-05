import unittest

from stability.scoring import score_recovery


class StabilityScoringTests(unittest.TestCase):
    def test_hand_calculated_binary_recovery_score(self):
        result = score_recovery(
            removed_edges=(("a", "b"), ("c", "d")),
            reconstructions=(
                (("a", "b"),),
                (("a", "b"),),
                (("a", "b"),),
                (("c", "d"),),
            ),
        )

        self.assertEqual(result.edge_recovery_frequencies[("a", "b")], 0.75)
        self.assertEqual(result.edge_recovery_frequencies[("c", "d")], 0.25)
        self.assertEqual(result.scenario_score, 0.5)

    def test_reversed_edges_count_as_same_undirected_edge(self):
        result = score_recovery(
            removed_edges=(("a", "b"), ("c", "d")),
            reconstructions=(
                (("b", "a"),),
                (("d", "c"),),
                (("a", "b"), ("d", "c")),
                (),
            ),
        )

        self.assertEqual(result.edge_recovery_frequencies[("a", "b")], 0.5)
        self.assertEqual(result.edge_recovery_frequencies[("c", "d")], 0.5)
        self.assertEqual(result.scenario_score, 0.5)

    def test_rejects_empty_scoring_inputs(self):
        with self.assertRaisesRegex(ValueError, "removed_edges"):
            score_recovery((), ((),))
        with self.assertRaisesRegex(ValueError, "reconstructions"):
            score_recovery((("a", "b"),), ())


if __name__ == "__main__":
    unittest.main()
