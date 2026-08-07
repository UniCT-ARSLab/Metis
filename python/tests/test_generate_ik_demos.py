import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import generate_ik_demos as g

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "godot/scenarios/robotarms/openarm_reach_hold_region_manifest.json"
)


def _summary(cell, seed, accepted, collided, reached, truncated, n_steps, ep_reward,
             final_distance):
    return {
        "cell": cell, "seed": seed,
        "pose_id": g.pose_id("easy", cell, seed),
        "accepted": accepted, "collided": collided, "reached": reached,
        "truncated": truncated, "n_steps": n_steps, "ep_reward": ep_reward,
        "final_distance": final_distance,
    }


class PoseIdTests(unittest.TestCase):
    def test_mirrors_godot_format(self):
        # Godot: "%s:%d:%d" % [region, cell, seed]
        self.assertEqual(g.pose_id("easy", 17, 1000), "easy:17:1000")
        self.assertEqual(g.pose_id("EASY", 41, 5), "easy:41:5")

    def test_train_val_seed_ranges_make_disjoint_pose_ids(self):
        cells = g.load_easy_cells(MANIFEST)
        train_ids = {g.pose_id("easy", c, 1000 + i) for c in cells for i in range(20)}
        val_ids = {g.pose_id("easy", c, 10_000_000 + i) for c in cells for i in range(3)}
        self.assertEqual(train_ids & val_ids, set())


class CellBalancingTests(unittest.TestCase):
    def test_picks_fewest_accepted_then_lowest_id(self):
        cells = [17, 18, 22]
        self.assertEqual(g.next_cell_to_fill({17: 2, 18: 0, 22: 1}, cells, 3), 18)
        # tie on count -> lowest cell id
        self.assertEqual(g.next_cell_to_fill({17: 1, 18: 1, 22: 1}, cells, 3), 17)

    def test_rejected_episode_does_not_advance_a_cell(self):
        # Balancing is by ACCEPTED: an unchanged count keeps the same cell pending.
        cells = [17, 18]
        counts = {17: 0, 18: 3}
        self.assertEqual(g.next_cell_to_fill(counts, cells, 3), 17)  # 18 already met
        # a reject leaves counts[17] == 0 -> still 17 next time
        self.assertEqual(g.next_cell_to_fill(counts, cells, 3), 17)

    def test_none_when_all_met(self):
        cells = [17, 18]
        self.assertIsNone(g.next_cell_to_fill({17: 3, 18: 3}, cells, 3))

    def test_exhausted_cells_skipped(self):
        cells = [17, 18]
        # 17 exhausted below target -> move to 18 even though 17 has fewer accepted
        self.assertEqual(
            g.next_cell_to_fill({17: 0, 18: 1}, cells, 3, exhausted={17}), 18)
        self.assertIsNone(
            g.next_cell_to_fill({17: 0, 18: 3}, cells, 3, exhausted={17}))


class AcceptanceTests(unittest.TestCase):
    def test_clean_hold_success_accepted(self):
        self.assertTrue(g.accept_episode(reached=True, truncated=False, collided=False))

    def test_collision_rejected(self):
        self.assertFalse(g.accept_episode(True, False, True))

    def test_truncation_rejected(self):
        self.assertFalse(g.accept_episode(True, True, False))

    def test_not_reached_rejected(self):
        self.assertFalse(g.accept_episode(False, False, False))


class ManifestCellsTests(unittest.TestCase):
    def test_load_easy_cells_from_real_manifest(self):
        self.assertEqual(
            g.load_easy_cells(MANIFEST),
            [17, 18, 22, 25, 28, 29, 30, 31, 33, 34, 41],
        )


class PerCellStatsTests(unittest.TestCase):
    def test_counts_and_means(self):
        summaries = [
            _summary(17, 1, True, False, True, False, 100, 5.0, 0.02),
            _summary(17, 2, False, True, False, False, 40, -30.0, None),   # collision reject
            _summary(17, 3, True, False, True, False, 120, 6.0, 0.03),
            _summary(18, 4, False, False, False, True, 300, -1.0, None),   # truncation reject
        ]
        stats = g.per_cell_stats(summaries, [17, 18])
        self.assertEqual(stats["17"]["attempts"], 3)
        self.assertEqual(stats["17"]["accepted"], 2)
        self.assertEqual(stats["17"]["collisions"], 1)
        self.assertEqual(stats["17"]["discards"], 1)
        self.assertAlmostEqual(stats["17"]["mean_len_accepted"], 110.0)
        self.assertAlmostEqual(stats["17"]["mean_reward_accepted"], 5.5)
        self.assertAlmostEqual(stats["17"]["mean_final_distance_accepted"], 0.025)
        # cell 18: one attempt, zero accepted -> None means, not a crash
        self.assertEqual(stats["18"]["accepted"], 0)
        self.assertIsNone(stats["18"]["mean_len_accepted"])


class FlattenTests(unittest.TestCase):
    def _ep(self, cell, seed, n):
        return {
            "cell": cell, "seed": seed, "pose_id": g.pose_id("easy", cell, seed),
            "transitions": [
                {
                    "obs": np.zeros(27, dtype=np.float32),
                    "action": np.full(7, 0.1, dtype=np.float32),
                    "reward": float(i),
                    "next_obs": np.ones(27, dtype=np.float32),
                    "terminated": (i == n - 1),
                    "truncated": False,
                }
                for i in range(n)
            ],
        }

    def test_grouping_ordering_and_done_flag(self):
        eps = [self._ep(17, 1000, 3), self._ep(41, 1001, 2)]
        cols = g.flatten_episodes(eps)
        self.assertEqual(cols["episode_indices"], [0, 0, 0, 1, 1])
        self.assertEqual(cols["step_indices"], [0, 1, 2, 0, 1])
        self.assertEqual(cols["cells"], [17, 17, 17, 41, 41])
        self.assertEqual(cols["pose_ids"][0], "easy:17:1000")
        self.assertEqual(cols["pose_ids"][-1], "easy:41:1001")
        # dones = terminated OR truncated; last step of each episode is terminal
        self.assertEqual(cols["dones"], [False, False, True, False, True])
        # obs/action dims preserved
        self.assertEqual(np.asarray(cols["obs"]).shape, (5, 27))
        self.assertEqual(np.asarray(cols["actions"]).shape, (5, 7))


class FiniteActionsTests(unittest.TestCase):
    def test_finite_in_bounds(self):
        a = np.array([[0.5, -0.5, 1.0, -1.0, 0.0, 0.2, -0.2]], dtype=np.float32)
        self.assertTrue(g._finite_actions_ok(a, [-1] * 7, [1] * 7))

    def test_out_of_bounds_false(self):
        a = np.array([[1.5, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        self.assertFalse(g._finite_actions_ok(a, [-1] * 7, [1] * 7))

    def test_nan_false(self):
        a = np.array([[np.nan, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        self.assertFalse(g._finite_actions_ok(a, [-1] * 7, [1] * 7))


class LoadPlansTests(unittest.TestCase):
    def test_groups_by_cell_sorted_by_seed(self):
        import json
        import tempfile
        payload = {
            "planning_failures": {"easy:25:1002": "rrt_no_path"},
            "plans": {
                "easy:17:1001": {"cell": 17, "seed": 1001, "waypoints": [[0] * 7, [1] * 7]},
                "easy:17:1000": {"cell": 17, "seed": 1000, "waypoints": [[0] * 7]},
                "easy:34:1000": {"cell": 34, "seed": 1000, "waypoints": [[0] * 7]},
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        by_cell, fails = g.load_plans(path)
        self.assertEqual(sorted(by_cell.keys()), [17, 34])
        self.assertEqual([p["seed"] for p in by_cell[17]], [1000, 1001])  # sorted
        self.assertEqual(by_cell[17][0]["pose_id"], "easy:17:1000")
        self.assertEqual(fails, {"easy:25:1002": "rrt_no_path"})


class RejectReasonTests(unittest.TestCase):
    def test_reasons(self):
        self.assertEqual(g._reject_reason(True, False, True), "collision")
        self.assertEqual(g._reject_reason(True, True, False), "truncated")
        self.assertEqual(g._reject_reason(False, False, False), "not_reached")
        self.assertEqual(g._reject_reason(True, False, False), "")

    def test_execution_failure_breakdown(self):
        summaries = [
            {"accepted": True, "reject_reason": ""},
            {"accepted": False, "reject_reason": "collision"},
            {"accepted": False, "reject_reason": "collision"},
            {"accepted": False, "reject_reason": "not_reached"},
        ]
        self.assertEqual(
            g.execution_failure_breakdown(summaries),
            {"collision": 2, "not_reached": 1},
        )


class ReplaySummaryTests(unittest.TestCase):
    def _r(self, reached, diff):
        return {"reached": reached, "terminal_reward_abs_diff": diff}

    def test_all_reached_and_coherent(self):
        s = g.summarize_replay([self._r(True, 0.01), self._r(True, 0.1)], reward_tol=0.5)
        self.assertTrue(s["all_reached_target"])
        self.assertTrue(s["terminal_reward_coherent"])

    def test_one_not_reached_fails(self):
        s = g.summarize_replay([self._r(True, 0.0), self._r(False, 0.0)], reward_tol=0.5)
        self.assertFalse(s["all_reached_target"])

    def test_reward_diff_over_tolerance_fails(self):
        s = g.summarize_replay([self._r(True, 0.9)], reward_tol=0.5)
        self.assertFalse(s["terminal_reward_coherent"])


if __name__ == "__main__":
    unittest.main()
