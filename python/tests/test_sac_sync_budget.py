import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf

from algorithms import sac
from core.models import default_network_layers, set_network_layers


class NonterminatingEnv:
    instances = []
    obs_dim = 3
    action_size = 1
    action_type = "continuous"
    action_names = ["force"]
    action_low = [-1.0]
    action_high = [1.0]
    action_space_spec = {"force": {"action_type": "continuous", "size": 1, "low": -1.0, "high": 1.0}}
    agent_id = "Agent"
    agent_ids = ["Agent"]
    agent_team_ids = [0]

    def __init__(self, **kwargs):
        self.steps = 0
        self.closed = False
        self.instances.append(self)

    def configure(self, **kwargs):
        pass

    def reset(self, seed=None):
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(self.obs_dim, dtype=np.float32), 0.0, False, False, {}

    def close(self):
        self.closed = True

    def agent_summary(self):
        return "single agent"

    def team_summary(self):
        return "single team"

    def _spec_for_agent(self, agent_id):
        return {"observation_names": ["x", "y", "z"]}


class SacSyncBudgetTests(unittest.TestCase):
    def run_training(self, budget):
        NonterminatingEnv.instances = []
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            command = ["sac", "--collector-mode", "sync", "--num-envs", "2", "--no-parallel-env-steps", "--no-multi-agent", "--num-episodes", "10", "--max-steps-per-episode", "1000", "--sync-batch-steps", "2", "--total-timesteps", str(budget), "--network-layers", "8", "8", "--replay-warmup", "1000", "--no-dashboard", "--no-best-checkpoint", "--no-health-monitor", "--no-save-replay-buffer", "--checkpoint-dir", str(directory / "checkpoints"), "--actor-weights-path", str(directory / "actor.weights.h5"), "--critic1-weights-path", str(directory / "critic1.weights.h5"), "--critic2-weights-path", str(directory / "critic2.weights.h5")]
            output = io.StringIO()
            manager = mock.Mock()
            try:
                with mock.patch("sys.argv", command), mock.patch.object(sac, "GodotProcessManager", return_value=manager), mock.patch.object(sac, "ScenarioGymEnv", NonterminatingEnv), contextlib.redirect_stdout(output):
                    sac.main()
                steps = sum(env.steps for env in NonterminatingEnv.instances)
                self.assertTrue(all(env.closed for env in NonterminatingEnv.instances))
                manager.stop_all.assert_called_once()
                checkpoint = tf.train.latest_checkpoint(str(directory / "checkpoints"))
                self.assertIsNotNone(checkpoint)
                self.assertEqual(int(tf.train.load_variable(checkpoint, "episode/.ATTRIBUTES/VARIABLE_VALUE")), 0)
                metadata = json.loads((directory / "checkpoints/policy.json").read_text())
                self.assertEqual(metadata["episode"], 0)
                self.assertTrue((directory / "checkpoints/policy.keras").is_file())
                return steps, output.getvalue()
            finally:
                set_network_layers(default_network_layers("sac"))

    def test_budget_stops_without_any_finished_episode(self):
        steps, output = self.run_training(8)
        self.assertEqual(steps, 8)
        self.assertEqual(output.count("Transition budget reached: 8/8"), 1)

    def test_nonmultiple_budget_only_finishes_inflight_env_batch(self):
        steps, output = self.run_training(7)
        self.assertEqual(steps, 8)
        self.assertEqual(output.count("Transition budget reached: 8/7"), 1)

    def test_unlimited_budget_does_not_stop_at_empty_reporting_window(self):
        steps, output = self.run_training(0)
        self.assertEqual(steps, 40)
        self.assertNotIn("Transition budget reached", output)


if __name__ == "__main__":
    unittest.main()
