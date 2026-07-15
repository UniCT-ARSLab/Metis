import tempfile
import unittest
from pathlib import Path

import numpy as np

from replay_buffer import ReplayBuffer


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


if __name__ == "__main__":
    unittest.main()
