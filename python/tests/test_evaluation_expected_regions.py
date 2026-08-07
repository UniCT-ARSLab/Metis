import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.evaluation import (
    build_evaluation_summary,
    episode_reset_raw_values,
    episode_reset_values,
)


def region_records(n_succ, n_total):
    return [{"success": i < n_succ} for i in range(n_total)]


def _reset_info(active_regions):
    """A realistic single-agent reset info as the Godot env returns it at reset."""
    return {
        "agent_info": {
            "reset": {
                "task_reset_mode": "regular",
                "target_region": "hard",
                "target_cell": 6,
                "active_regions": active_regions,
            }
        }
    }


class ExpectedRegionsTests(unittest.TestCase):
    def _summary(self, expected):
        reset_outcomes = {
            "target_region:easy": region_records(2, 3),    # 0.667
            "target_region:medium": region_records(1, 3),  # 0.333
        }
        return build_evaluation_summary(
            [10.0] * 6, [100] * 6, 3, 6,
            reset_outcomes=reset_outcomes, expected_regions=expected)

    def test_absent_expected_region_zeroes_selection_success(self):
        # Hard is expected (active) but produced no episodes -> fail closed.
        summary = self._summary(["easy", "medium", "hard"])
        self.assertEqual(summary["selection_success_rate"], 0.0)
        self.assertEqual(summary["region_success_floor"], 0.0)

    def test_all_expected_present_uses_worst_region(self):
        summary = self._summary(["easy", "medium"])
        self.assertAlmostEqual(summary["region_success_floor"], 1.0 / 3.0, places=3)
        self.assertAlmostEqual(summary["selection_success_rate"], 1.0 / 3.0, places=3)

    def test_no_expected_regions_keeps_min_over_observed(self):
        summary = self._summary(None)
        self.assertAlmostEqual(summary["region_success_floor"], 1.0 / 3.0, places=3)


class ActiveRegionsExtractionTests(unittest.TestCase):
    """Lock the real run.py extraction path: active_regions is a LIST and must survive
    metadata extraction so the expected-regions floor can fire."""

    def test_raw_helper_preserves_list_but_text_helper_flattens(self):
        info = _reset_info(["easy", "medium", "hard"])
        # Raw helper keeps the list intact (one value per agent).
        self.assertEqual(
            episode_reset_raw_values(info, "active_regions"),
            [["easy", "medium", "hard"]],
        )
        # The text helper would stringify it — proving why it was the wrong tool here.
        flattened = episode_reset_values(info, "active_regions")
        self.assertEqual(flattened, ["['easy', 'medium', 'hard']"])
        self.assertNotIsInstance(flattened[0], (list, tuple))

    def test_integrated_absent_hard_zeroes_selection_success(self):
        # 1) Godot reports all three regions active at reset.
        info = _reset_info(["easy", "medium", "hard"])
        # 2) Replicate run.py's aggregation over the extracted metadata.
        expected = set()
        for active in episode_reset_raw_values(info, "active_regions"):
            if isinstance(active, (list, tuple)):
                expected.update(str(a).strip().lower() for a in active)
        self.assertEqual(expected, {"easy", "medium", "hard"})
        # 3) The eval deck only produced easy+medium episodes (Hard missing).
        reset_outcomes = {
            "target_region:easy": region_records(2, 3),
            "target_region:medium": region_records(2, 3),
        }
        summary = build_evaluation_summary(
            [10.0] * 6, [100] * 6, 3, 6,
            reset_outcomes=reset_outcomes, expected_regions=expected)
        # Fail closed: an expected region with no episodes zeroes selection success.
        self.assertEqual(summary["selection_success_rate"], 0.0)
        self.assertEqual(summary["region_success_floor"], 0.0)


if __name__ == "__main__":
    unittest.main()
