import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from algorithms.common import (
    build_replay_ready_event,
    should_use_random_exploration,
)
from core.replay_buffer import ReplayBuffer
from core.training import restore_replay_buffer


class ReplayBufferTests(unittest.TestCase):
    @staticmethod
    def transition(value, continuous=False):
        obs = np.asarray([value, value + 0.5], dtype=np.float32)
        action = np.asarray([value / 10.0], dtype=np.float32) if continuous else int(value)
        return obs, action, float(value), obs + 1.0, bool(value % 2)

    def test_ring_snapshot_preserves_chronological_order(self):
        buffer = ReplayBuffer(capacity=4)
        for value in range(6):
            buffer.add(*self.transition(value))
        snapshot = buffer.export_snapshot()
        self.assertEqual(len(buffer), 4)
        self.assertEqual(snapshot["rewards"].tolist(), [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(snapshot["actions"].dtype, np.int32)

    def test_continuous_actions_keep_float_shape(self):
        buffer = ReplayBuffer(capacity=8)
        for value in range(4):
            buffer.add(*self.transition(value, continuous=True))
        _obs, actions, _rewards, _next_obs, _dones = buffer.sample(4, action_dtype=np.float32)
        self.assertEqual(actions.shape, (4, 1))
        self.assertEqual(actions.dtype, np.float32)

    def test_sample_batches_adds_update_dimension(self):
        buffer = ReplayBuffer(capacity=16)
        for value in range(12):
            buffer.add(*self.transition(value))

        obs, actions, rewards, next_obs, dones = buffer.sample_batches(3, 4)

        self.assertEqual(obs.shape, (3, 4, 2))
        self.assertEqual(actions.shape, (3, 4))
        self.assertEqual(rewards.shape, (3, 4))
        self.assertEqual(next_obs.shape, (3, 4, 2))
        self.assertEqual(dones.shape, (3, 4))
        self.assertEqual(actions.dtype, np.int32)
        self.assertEqual(dones.dtype, np.float32)

    def test_loads_legacy_replay_npz(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.npz"
            values = [self.transition(value) for value in range(5)]
            obs, actions, rewards, next_obs, dones = zip(*values)
            np.savez_compressed(
                path,
                version=np.asarray([1], dtype=np.int32),
                capacity=np.asarray([5], dtype=np.int64),
                obs=np.asarray(obs),
                actions=np.asarray(actions),
                rewards=np.asarray(rewards),
                next_obs=np.asarray(next_obs),
                dones=np.asarray(dones),
            )
            buffer = ReplayBuffer(capacity=3)
            restored = buffer.load(path)
            self.assertEqual(restored, 3)
            self.assertEqual(buffer.export_snapshot()["rewards"].tolist(), [2.0, 3.0, 4.0])

    def test_async_snapshot_can_be_restored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replay.npz"
            buffer = ReplayBuffer(capacity=8)
            for value in range(6):
                buffer.add(*self.transition(value))
            count, future = buffer.save_async(path)
            buffer.add(*self.transition(6))
            buffer.wait_for_pending_saves()

            self.assertEqual(count, 6)
            self.assertEqual(future.result(), 6)
            restored = ReplayBuffer(capacity=8)
            restored.load(path)
            self.assertEqual(restored.export_snapshot()["rewards"].tolist(), list(map(float, range(6))))

    def test_missing_resume_replay_collects_with_restored_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                replay_capacity=32,
                replay_warmup=8,
                batch_size=4,
                require_replay_buffer=False,
                random_exploration_episodes=100,
            )
            buffer = ReplayBuffer(capacity=args.replay_capacity)

            restored = restore_replay_buffer(
                args, Path(temp_dir) / "best" / "ckpt-10", buffer)

            self.assertEqual(restored, 0)
            self.assertTrue(args._replay_warmup_uses_restored_policy)
            self.assertFalse(
                should_use_random_exploration(10, args, len(buffer)))
            self.assertTrue(build_replay_ready_event(buffer, args).is_set())

    def test_best_checkpoint_can_restore_matching_parent_replay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = ReplayBuffer(capacity=8)
            for value in range(6):
                source.add(*self.transition(value))
            source.save(root / "replay-10.npz")
            args = SimpleNamespace(
                replay_capacity=8,
                replay_warmup=4,
                batch_size=4,
                require_replay_buffer=False,
                random_exploration_episodes=0,
            )
            restored_buffer = ReplayBuffer(capacity=8)

            restored = restore_replay_buffer(
                args, root / "best" / "ckpt-10", restored_buffer)

            self.assertEqual(restored, 6)
            self.assertFalse(args._replay_warmup_uses_restored_policy)

    def protected_buffer(self, capacity=6, count=2):
        buffer = ReplayBuffer(capacity)
        rows = [self.transition(100 + i, continuous=True) for i in range(count)]
        buffer.add_many(*zip(*rows), is_demo=True, protect=True)
        return buffer

    def test_protected_prefix_survives_mixed_single_and_bulk_online_writes(self):
        buffer = self.protected_buffer()
        expected = []
        for start, count in ((0, 2), (2, 7), (9, 1), (10, 4), (14, 3)):
            rows = [self.transition(i, continuous=True) for i in range(start, start + count)]
            if count == 1:
                buffer.add(*rows[0])
            else:
                buffer.add_many(*zip(*rows))
            expected = (expected + list(range(start, start + count)))[-4:]
            snapshot = buffer.export_snapshot()
            self.assertEqual(snapshot["rewards"].tolist(), [100., 101.] + list(map(float, expected)))
            self.assertEqual(snapshot["is_demo"].tolist(), [True, True] + [False] * len(expected))
            self.assertEqual(buffer.protected_demo_count, 2)

    def test_single_protected_demo_survives_demo_only_save_and_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.protected_buffer(count=1)
            path = Path(temporary) / "replay.npz"
            source.save(path)
            restored = ReplayBuffer(6)
            restored.load(path)
            self.assertEqual(restored.protected_demo_count, 1)
            for i in range(15):
                restored.add(*self.transition(i, continuous=True))
            self.assertEqual(restored.export_snapshot()["rewards"][0], 100.)

    def test_restore_cannot_silently_drop_protected_demos(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.protected_buffer()
            path = Path(temporary) / "replay.npz"
            source.save(path)
            with self.assertRaisesRegex(ValueError, "all protected demonstrations"):
                ReplayBuffer(2).load(path)
            with self.assertRaisesRegex(ValueError, "all protected demonstrations"):
                ReplayBuffer(6).load(path, max_items=2)
            restored = ReplayBuffer(3)
            restored.load(path)
            self.assertEqual(restored.protected_demo_count, 2)

    def test_append_restore_does_not_corrupt_an_existing_protected_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = ReplayBuffer(6)
            source.add(*self.transition(7, continuous=True))
            path = Path(temporary) / "replay.npz"
            source.save(path)
            restored = self.protected_buffer()
            with self.assertRaisesRegex(ValueError, "clear=True"):
                restored.load(path, clear=False)
            self.assertEqual(restored.export_snapshot()["rewards"].tolist(), [100., 101.])

    def test_demo_sampling_telemetry_counts_only_opted_in_draws(self):
        buffer = self.protected_buffer(capacity=8)
        for i in range(6):
            buffer.add(*self.transition(i, continuous=True))
        indices = np.asarray([0, 4, 1, 7])
        with mock.patch.object(np.random, "randint", return_value=indices):
            buffer.sample(4, action_dtype=np.float32)
            self.assertEqual(buffer.demo_sampling_stats["sampled_replay_transitions"], 0)
            buffer.sample(4, action_dtype=np.float32, track_demo_samples=True)
        self.assertEqual(buffer.demo_sampling_stats, {"protected_demo_count": 2, "sampled_demo_transitions": 2, "sampled_replay_transitions": 4, "sampled_demo_fraction": .5})
        buffer.clear_online()
        self.assertEqual(buffer.protected_demo_count, 2)
        self.assertEqual(buffer.demo_sampling_stats["sampled_replay_transitions"], 0)

    def test_tracking_does_not_change_uniform_draws_or_rng_state(self):
        buffer = self.protected_buffer(capacity=8)
        for i in range(6):
            buffer.add(*self.transition(i, continuous=True))
        initial = np.random.get_state()
        expected = buffer.sample(4, action_dtype=np.float32)
        expected_rng = np.random.get_state()
        np.random.set_state(initial)
        actual = buffer.sample(4, action_dtype=np.float32, track_demo_samples=True)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        actual_rng = np.random.get_state()
        np.testing.assert_array_equal(expected_rng[1], actual_rng[1])
        self.assertEqual(expected_rng[2:], actual_rng[2:])


if __name__ == "__main__":
    unittest.main()
