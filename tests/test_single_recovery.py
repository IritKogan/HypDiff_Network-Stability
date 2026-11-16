from __future__ import annotations

import unittest

from scripts.run_single_recovery import (
    _validate_imdb_thresholds,
    score_saved_predictions,
)


class SingleRecoveryTests(unittest.TestCase):
    def test_saved_edges_recompute_recovery_score_and_false_positives(self):
        predictions = {
            "reconstructions": [
                {"edges": [["a", "b"], ["a", "d"]]},
                {"edges": [["b", "c"], ["a", "d"]]},
            ]
        }
        evaluation_labels = {
            "node_ids": ["a", "b", "c", "d"],
            "original_edges": [["a", "b"], ["b", "c"], ["a", "c"]],
            "removed_edges": [["a", "b"], ["b", "c"]],
        }

        result = score_saved_predictions(predictions, evaluation_labels)

        self.assertEqual(result["book_stability_score"], 0.5)
        self.assertEqual(
            result["removed_edge_recovery_frequencies"],
            [
                {"edge": ["a", "b"], "frequency": 0.5},
                {"edge": ["b", "c"], "frequency": 0.5},
            ],
        )
        self.assertEqual(result["per_reconstruction_false_positive_edges"], [1, 1])
        self.assertEqual(result["mean_false_positive_edges"], 1.0)

    def test_empty_removed_edge_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no removed edges"):
            score_saved_predictions(
                {"reconstructions": [{"edges": []}]},
                {
                    "node_ids": ["a", "b"],
                    "original_edges": [["a", "b"]],
                    "removed_edges": [],
                },
            )

    def test_imdb_threshold_must_match_frozen_evaluation_metadata(self):
        calibration = {
            "status": "calibration_complete",
            "evaluation_gate_passed_for_both_methods": True,
            "manifest_sha256": "manifest",
            "method_thresholds": {
                "partial_recovery": {
                    "tau": 0.8,
                    "false_positive_rate_cap": 0.05,
                    "calibration_realized_fpr": 0.04,
                }
            },
        }
        evaluation = {
            "status": "evaluation_complete",
            "thresholds_frozen_before_evaluation": True,
            "manifest_sha256": "manifest",
            "frozen_method_thresholds": {"partial_recovery": 0.8},
        }

        self.assertEqual(_validate_imdb_thresholds(calibration, evaluation), (0.8, 0.05))
        evaluation["manifest_sha256"] = "different-manifest"
        with self.assertRaisesRegex(ValueError, "different manifests"):
            _validate_imdb_thresholds(calibration, evaluation)

    def test_imdb_threshold_over_calibration_cap_is_rejected(self):
        calibration = {
            "status": "calibration_complete",
            "evaluation_gate_passed_for_both_methods": True,
            "manifest_sha256": "manifest",
            "method_thresholds": {
                "partial_recovery": {
                    "tau": 0.8,
                    "false_positive_rate_cap": 0.05,
                    "calibration_realized_fpr": 0.06,
                }
            },
        }
        evaluation = {
            "status": "evaluation_complete",
            "thresholds_frozen_before_evaluation": True,
            "manifest_sha256": "manifest",
            "frozen_method_thresholds": {"partial_recovery": 0.8},
        }

        with self.assertRaisesRegex(ValueError, "exceeded its calibration cap"):
            _validate_imdb_thresholds(calibration, evaluation)


if __name__ == "__main__":
    unittest.main()
