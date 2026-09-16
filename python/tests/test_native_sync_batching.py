import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.common import select_actions as select_continuous_actions  # noqa: E402
from algorithms.dqn import process_sync_dqn_updates, select_actions as select_dqn_actions  # noqa: E402


class ArrayResult:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class RecordingModel:
    def __init__(self, output):
        self.output = np.asarray(output)
        self.calls = []

    def __call__(self, observations, training=False):
        self.calls.append((np.asarray(observations).copy(), training))
        return ArrayResult(self.output[: len(observations)])


class NativeActionBatchingTests(unittest.TestCase):
    def test_dqn_greedy_actions_use_one_forward_pass(self):
        model = RecordingModel([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]])
        observations = np.arange(12, dtype=np.float32).reshape(3, 4)
        actions = select_dqn_actions(model, observations, np.zeros(3, dtype=np.bool_), 0.0, 2)
        np.testing.assert_array_equal(actions, [1, 0, 1])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0].shape, (3, 4))

    def test_dqn_exploration_is_independent_per_row(self):
        random.seed(7)
        model = RecordingModel([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0], [4.0, 0.0]])
        actions = select_dqn_actions(model, np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=np.bool_), 1.0, 2)
        self.assertEqual(actions.shape, (4,))
        self.assertEqual(len(model.calls), 0)

    def test_continuous_actions_batch_actor_and_preserve_ou_state(self):
        actor = RecordingModel([[0.0, 0.0], [0.5, -0.5]])
        noise_states = [np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
        with patch("algorithms.common.np.random.normal", return_value=np.asarray([0.1, -0.2], dtype=np.float32)):
            actions = select_continuous_actions(actor, np.zeros((2, 3), dtype=np.float32), np.zeros(2, dtype=np.bool_), np.full(2, -1.0, dtype=np.float32), np.full(2, 1.0, dtype=np.float32), 0.3, noise_state=noise_states)
        self.assertEqual(len(actor.calls), 1)
        np.testing.assert_allclose(noise_states[0], [0.1, -0.2])
        np.testing.assert_allclose(noise_states[1], [0.1, -0.2])
        np.testing.assert_allclose(actions[0], [0.1, -0.2])
        np.testing.assert_allclose(actions[1], [0.6, -0.7])


class DQNSyncUpdateTests(unittest.TestCase):
    def test_transition_schedule_updates_target_and_batches_learner_calls(self):
        target_sync = Mock()
        target_sync.due_after_transitions.return_value = True
        model = Mock()
        model.get_weights.return_value = ["weights"]
        target_model = Mock()
        throttle = Mock()
        throttle.updates_due.return_value = 4
        learner = Mock(return_value=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        losses, updates, target_updated = process_sync_dqn_updates(target_sync=target_sync, target_model=target_model, model=model, sync_throttle=throttle, transitions_added=4, env_steps=4, replay_ready=True, verification_pending=False, learner_step=learner, buffer="buffer", batch_size=128)
        target_sync.due_after_transitions.assert_called_once_with(4)
        target_model.set_weights.assert_called_once_with(["weights"])
        throttle.updates_due.assert_called_once_with(4, env_steps=4)
        learner.assert_called_once_with("buffer", 128, 4)
        np.testing.assert_array_equal(losses, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(updates, 4)
        self.assertTrue(target_updated)

    def test_target_sync_is_not_blocked_by_replay_warmup(self):
        target_sync = Mock()
        target_sync.due_after_transitions.return_value = True
        model = Mock()
        model.get_weights.return_value = ["weights"]
        target_model = Mock()
        learner = Mock()
        target_transition_count = Mock()
        losses, updates, target_updated = process_sync_dqn_updates(target_sync=target_sync, target_model=target_model, model=model, sync_throttle=Mock(), transitions_added=10000, env_steps=4, replay_ready=False, verification_pending=False, learner_step=learner, buffer="buffer", batch_size=128, target_transition_count=target_transition_count)
        target_transition_count.assign_add.assert_called_once_with(10000)
        target_model.set_weights.assert_called_once_with(["weights"])
        learner.assert_not_called()
        self.assertEqual(len(losses), 0)
        self.assertEqual(updates, 0)
        self.assertTrue(target_updated)


if __name__ == "__main__":
    unittest.main()
