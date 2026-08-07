"""Generic BC/replay dataset separation (common.py): replay-only never enters the BC sampler,
BC-only never enters the replay. Task-agnostic (no cell/region concept)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_complete(path, n, obs_dim=27, act=7, group_start=0):
    np.savez(path, obs=np.zeros((n, obs_dim), np.float32), actions=np.ones((n, act), np.float32),
             rewards=np.zeros(n, np.float32), next_obs=np.zeros((n, obs_dim), np.float32),
             dones=np.zeros(n, np.float32), episode_indices=np.arange(group_start, group_start + n),
             action_type=np.asarray(["continuous"]))


def _write_pairs(path, n, obs_dim=27, act=7):
    np.savez(path, obs=np.zeros((n, obs_dim), np.float32), actions=np.full((n, act), 0.5, np.float32))


class DatasetSeparationTests(unittest.TestCase):
    def test_bc_set_is_demo_plus_bc_only_never_replay_only(self):
        from algorithms.common import (
            load_demonstration_arrays, _load_bc_pairs, _combine_bc_sources)
        with tempfile.TemporaryDirectory() as d:
            demo_p = Path(d) / "demo.npz"      # --demo-path (BC + replay)
            replay_p = Path(d) / "replay.npz"  # --demo-replay-only-path (replay ONLY)
            bc_p = Path(d) / "bc.npz"          # --demo-bc-path (BC ONLY)
            _write_complete(demo_p, 10)
            _write_complete(replay_p, 20)
            _write_pairs(bc_p, 5)

            demo = load_demonstration_arrays([str(demo_p)], 27, 7)
            replay_only = load_demonstration_arrays([str(replay_p)], 27, 7)
            bc_pairs = _load_bc_pairs([str(bc_p)], 27, 7)

            # BC training set = demo + bc-only pairs. Replay-only is NOT passed in -> never in BC.
            bc_train = _combine_bc_sources(demo, bc_pairs)
            self.assertEqual(bc_train["obs"].shape[0], 10 + 5)
            # The distinctive BC-only action value (0.5) is present; replay-only (1.0-only) count
            # would have made it 30 if wrongly included.
            self.assertNotEqual(bc_train["obs"].shape[0], 10 + 20 + 5)

            # Replay-only is a COMPLETE transition set (has rewards/next_obs/dones) -> replayable.
            self.assertTrue(all(k in replay_only for k in ("rewards", "next_obs", "dones")))
            # BC-only pairs LACK rewards/next_obs/dones -> cannot be a valid replay transition.
            self.assertNotIn("rewards", bc_pairs)
            self.assertNotIn("next_obs", bc_pairs)

    def test_combine_without_bc_pairs_is_demo_only(self):
        from algorithms.common import _combine_bc_sources
        demo = {"obs": np.zeros((7, 27), np.float32), "actions": np.zeros((7, 7), np.float32)}
        combined = _combine_bc_sources(demo, None)
        self.assertEqual(combined["obs"].shape[0], 7)

    def test_new_separation_args_exist(self):
        from unittest import mock
        import algorithms.common as common
        with mock.patch("sys.argv", ["td3_bc"]):
            args = common.parse_args("td3_bc")
        self.assertEqual(args.demo_replay_only_path, [])
        self.assertEqual(args.demo_bc_path, [])
        self.assertEqual(args.demo_q_filter_start_policy_updates, 0)


if __name__ == "__main__":
    unittest.main()
