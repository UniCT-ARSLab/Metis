import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.demo_invariants import check_demo_invariants, is_clean, summarize


class DemoInvariantTests(unittest.TestCase):
    def test_consecutive_conflict_flagged(self):
        # obs_0 == obs_1 but action_0 (zero) != action_1 (move) -> the poisonous no-op start.
        obs = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], np.float32)
        act = np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], np.float32)
        v = check_demo_invariants(obs, act, episode_indices=[0, 0, 0], step_indices=[0, 1, 2])
        self.assertIn(0, v["consecutive_conflict"])
        self.assertIn(0, v["first_action_zero"])
        self.assertFalse(is_clean(v))

    def test_clean_dataset_passes(self):
        obs = np.array([[1.0, 2.0], [1.1, 2.1], [3.0, 4.0]], np.float32)
        act = np.array([[0.8, -0.2], [0.5, 0.3], [0.1, 0.9]], np.float32)
        v = check_demo_invariants(obs, act, episode_indices=[0, 0, 0])
        self.assertTrue(is_clean(v), summarize(v))

    def test_duplicate_obs_conflict_across_episodes(self):
        obs = np.array([[1.0, 2.0], [5.0, 6.0], [1.0, 2.0]], np.float32)  # rows 0 and 2 identical
        act = np.array([[0.3, 0.3], [0.1, 0.1], [0.9, -0.9]], np.float32)  # but different actions
        v = check_demo_invariants(obs, act, episode_indices=[0, 0, 1])
        self.assertTrue(len(v["duplicate_obs_conflict"]) >= 1)

    def test_recorded_ne_applied(self):
        obs = np.array([[1.0, 2.0], [1.1, 2.1]], np.float32)
        act = np.array([[0.5, 0.5], [0.3, 0.3]], np.float32)
        applied = np.array([[0.5, 0.5], [0.9, 0.9]], np.float32)  # row 1 differs
        v = check_demo_invariants(obs, act, applied_actions=applied)
        self.assertIn(1, v["recorded_ne_applied"])

    def test_real_m5_v1_is_contaminated(self):
        # The known-bad v1 dataset must trip the invariants (regression guard until v2 replaces it).
        p = Path(__file__).resolve().parents[2] / "python/demos/openarm_reach_hold_m5/train.npz"
        if not p.exists():
            self.skipTest("M5 v1 dataset not present")
        d = np.load(p)
        v = check_demo_invariants(d["obs"], d["actions"], episode_indices=d["episode_indices"],
                                  step_indices=d["step_indices"])
        self.assertGreater(len(v["consecutive_conflict"]), 0)
        self.assertGreater(len(v["first_action_zero"]), 0)


if __name__ == "__main__":
    unittest.main()
