import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BcArgsTests(unittest.TestCase):
    def test_new_bc_args_exist_with_defaults(self):
        from unittest import mock
        import algorithms.sac as sac
        with mock.patch("sys.argv", ["sac"]):
            args = sac.parse_args()
        # BooleanOptionalAction + append/str defaults for the new BC-gate options.
        self.assertEqual(args.demo_validation_path, [])
        self.assertIsNone(args.bc_actor_weights_path)
        self.assertFalse(args.stop_after_bc)
        # existing knob the gate reuses
        self.assertEqual(args.critic_warmup_updates, 2000)


class BcHelperTests(unittest.TestCase):
    def test_bc_actor_path_default(self):
        from algorithms.sac import _bc_actor_path_default
        self.assertEqual(
            _bc_actor_path_default("checkpoints/x/generic_sac_actor.weights.h5"),
            "checkpoints/x/generic_sac_actor_bc.weights.h5",
        )
        self.assertEqual(_bc_actor_path_default("foo.h5"), "foo_bc.h5")
        self.assertEqual(_bc_actor_path_default(None), "bc_actor_bc.weights.h5")

    def test_bc_raw_targets_scaling(self):
        from algorithms.sac import _bc_raw_targets
        low = [-1.0] * 3
        high = [1.0] * 3
        # action 0 -> raw 0; action 1 -> +1 (clipped 0.995); action -1 -> -0.995
        actions = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
        raw = _bc_raw_targets(actions, low, high)
        np.testing.assert_allclose(raw[0], [0.0, 0.995, -0.995], atol=1e-5)

    def test_validation_mse_and_best_epoch_restore(self):
        # Uses a tiny real Keras actor so tf.GradientTape / get_weights / set_weights are exercised.
        import tensorflow as tf
        from algorithms.sac import (
            _bc_raw_targets, _bc_validation_mse, pretrain_actor_behavior_cloning)

        class TinyActor(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.d = tf.keras.layers.Dense(3)

            def call(self, x, training=False):
                return self.d(x), tf.zeros_like(self.d(x))

        actor = TinyActor()
        actor(tf.zeros((1, 5)))  # build

        # _bc_validation_mse: with weights making mean≈0 is not guaranteed; check it's a finite scalar.
        val_obs = np.random.RandomState(0).randn(16, 5).astype(np.float32)
        val_raw = _bc_raw_targets(np.tanh(np.random.RandomState(1).randn(16, 3)), [-1]*3, [1]*3)
        mse = _bc_validation_mse(actor, val_obs, val_raw, 8)
        self.assertTrue(np.isfinite(mse) and mse >= 0.0)

        demo = {"obs": np.random.RandomState(2).randn(64, 5).astype(np.float32),
                "actions": np.tanh(np.random.RandomState(3).randn(64, 3)).astype(np.float32)}
        val = {"obs": val_obs, "actions": np.tanh(np.random.RandomState(1).randn(16, 3)).astype(np.float32)}
        info = pretrain_actor_behavior_cloning(
            actor, demo, epochs=3, batch_size=16, learning_rate=1e-3,
            action_low=[-1]*3, action_high=[1]*3, demo_validation=val)
        # best epoch tracked within range, best val mse finite
        self.assertIn(info["best_epoch"], (1, 2, 3))
        self.assertTrue(np.isfinite(info["best_val_mse"]))


class BcSourcesTests(unittest.TestCase):
    def test_group_batches_balance_by_group(self):
        from algorithms.sac import _bc_group_batches
        # group 0 has 100 members, group 1 has 2. Balanced sampling must pick group 1 often
        # (~half the draws), not proportional to size (which would be ~2%).
        group_ids = np.array([0] * 100 + [1] * 2)
        rng = np.random.default_rng(0)
        picks = np.concatenate(list(_bc_group_batches(group_ids, 102, 51, rng)))
        frac_group1 = np.mean(group_ids[picks] == 1)
        self.assertGreater(frac_group1, 0.3)  # ~0.5 expected; proportional would be ~0.02

    def test_group_batches_none_is_plain_shuffle(self):
        from algorithms.sac import _bc_group_batches
        rng = np.random.default_rng(0)
        picks = np.concatenate(list(_bc_group_batches(None, 10, 4, rng)))
        self.assertEqual(sorted(picks.tolist()), list(range(10)))

    def test_load_bc_pairs_and_combine(self):
        import tempfile
        from algorithms.sac import _load_bc_pairs, _combine_bc_sources
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.npz"
            np.savez(p, obs=np.zeros((5, 27), np.float32),
                     actions=np.ones((5, 7), np.float32),
                     episode_indices=np.array([0, 0, 1, 1, 1]))
            pairs = _load_bc_pairs([str(p)], 27, 7)
            self.assertEqual(pairs["obs"].shape, (5, 27))
            self.assertEqual(sorted(set(pairs["group_ids"].tolist())), [0, 1])
            # combine with a "demo" source -> disjoint groups
            demo = {"obs": np.zeros((3, 27), np.float32), "actions": np.zeros((3, 7), np.float32),
                    "group_ids": np.array([0, 0, 1])}
            combined, groups = _combine_bc_sources(demo, pairs)
            self.assertEqual(combined["obs"].shape, (8, 27))
            # 8 transitions, groups from demo {0,1} then pairs offset -> {2,3}
            self.assertEqual(sorted(set(groups.tolist())), [0, 1, 2, 3])

    def test_load_demonstration_arrays_carries_group_ids(self):
        import tempfile
        from algorithms.common import load_demonstration_arrays
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "demo.npz"
            np.savez(p, obs=np.zeros((4, 27), np.float32), actions=np.zeros((4, 7), np.float32),
                     rewards=np.zeros(4, np.float32), next_obs=np.zeros((4, 27), np.float32),
                     dones=np.zeros(4, np.float32), episode_indices=np.array([5, 5, 6, 6]),
                     action_type=np.asarray(["continuous"]))
            data = load_demonstration_arrays([str(p)], 27, 7)
            self.assertIn("group_ids", data)
            self.assertEqual(sorted(set(data["group_ids"].tolist())), [5, 6])


if __name__ == "__main__":
    unittest.main()
