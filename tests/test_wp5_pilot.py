import tempfile
from pathlib import Path
import unittest

from stability.pilot_artifacts import (
    count_unique_edge_sets,
    load_pilot_artifact,
    rescore_pilot_artifact,
    save_pilot_artifact,
)


class Wp5PilotArtifactTests(unittest.TestCase):
    def test_artifact_round_trip_rescores_and_counts_identical_sets(self):
        payload = {
            "removed_edges": [["a", "b"], ["c", "d"]],
            "reconstructions": [
                {"seed": 1, "edges": [["a", "b"]]},
                {"seed": 2, "edges": [["a", "b"]]},
                {"seed": 3, "edges": [["a", "b"], ["c", "d"]]},
                {"seed": 4, "edges": [["c", "d"]]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.json"
            save_pilot_artifact(path, payload)
            saved = load_pilot_artifact(path)

        score = rescore_pilot_artifact(saved)
        self.assertEqual(score.edge_recovery_frequencies[("a", "b")], 0.75)
        self.assertEqual(score.edge_recovery_frequencies[("c", "d")], 0.5)
        self.assertEqual(score.scenario_score, 0.625)
        self.assertEqual(count_unique_edge_sets(saved), 3)

    def test_unique_set_count_reports_identical_outputs(self):
        payload = {
            "reconstructions": [
                {"edges": [["a", "b"], ["c", "d"]]},
                {"edges": [["c", "d"], ["a", "b"]]},
                {"edges": [["a", "b"]]},
            ]
        }
        self.assertEqual(count_unique_edge_sets(payload), 2)


if __name__ == "__main__":
    unittest.main()
