import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
