import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

from algorithms import dqn
from core.models import network_layers, set_network_layers


class DiscreteEnv:
    instances = []
    lengths = (2, 7)
    truncations = (False, False)
    obs_dim = 3
    num_actions = 2
    action_size = 1
    action_type = "discrete"
    action_names = ["left", "right"]
    action_space_spec = {"move": {"action_type": "discrete", "size": 2}}
    agent_id = "Agent"
    agent_ids = ["Agent"]
    agent_team_ids = [0]

    def __init__(self, **kwargs):
        self.index = len(self.instances)
        self.steps = 0
        self.episode_steps = 0
        self.seeds = []
        self.configurations = []
        self.closed = False
        self.finished = False
        self.instances.append(self)

    def configure(self, **kwargs):
        self.configurations.append(kwargs)

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.episode_steps = 0
        self.finished = False
        return np.array([self.index, len(self.seeds), 0], dtype=np.float32), {}

    def step(self, action):
        if self.finished:
            raise AssertionError("A terminated episode was stepped without reset")
        self.steps += 1
        self.episode_steps += 1
        length = self.lengths[self.index]
        self.finished = length > 0 and self.episode_steps >= length
        truncated = self.finished and self.truncations[self.index]
        obs = np.array([self.index, len(self.seeds), self.episode_steps], dtype=np.float32)
        return obs, 1.0, self.finished and not truncated, truncated, {"agent_info": {"terminal_reason": "time_limit" if truncated else "terminal" if self.finished else ""}}

    def close(self):
        self.closed = True

    def agent_summary(self):
        return "single agent"

    def team_summary(self):
        return "single team"

    def _spec_for_agent(self, agent_id):
        return {"observation_names": ["env", "reset", "step"]}


class DqnSyncAutoresetTests(unittest.TestCase):
    def run_training(self, budget, *, lengths=(2, 7), truncations=(False, False), reporting_steps=3, episode_limit=100, episode_cap=0, learn=False, interrupt=False):
        original_network_layers = network_layers()
        DiscreteEnv.instances = []
        DiscreteEnv.lengths = lengths
        DiscreteEnv.truncations = truncations
        buffers = []
        requests = []
        original_buffer = dqn.ReplayBuffer
        original_step = dqn.ParallelEnvStepper.step

        def buffer_factory(*args, **kwargs):
            buffer = original_buffer(*args, **kwargs)
            buffers.append(buffer)
            return buffer

        def record_steps(stepper, batch):
            batch = list(batch)
            if interrupt and len(requests) == 2:
                raise KeyboardInterrupt()
            requests.append(len(batch))
            return original_step(stepper, batch)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            command = ["dqn", "--collector-mode", "sync", "--num-envs", "2", "--no-parallel-env-steps", "--no-multi-agent", "--num-episodes", str(episode_limit), "--max-steps-per-episode", str(episode_cap), "--sync-batch-steps", str(reporting_steps), "--total-timesteps", str(budget), "--network-layers", "8", "8", "--replay-warmup", "4" if learn else "1000", "--batch-size", "2", "--epsilon-decay-transitions", "20", "--target-update-steps", "4", "--async-update-every", "1", "--async-updates-per-step", "1", "--async-max-updates-per-env-step", "0", "--no-dashboard", "--no-best-checkpoint", "--no-health-monitor", "--no-save-replay-buffer", "--checkpoint-every", "0", "--log-action-every", "0", "--checkpoint-dir", str(directory / "checkpoints"), "--weights-path", str(directory / "final.weights.h5")]
            output = io.StringIO()
            manager = mock.Mock()
            try:
                with mock.patch("sys.argv", command), mock.patch.object(dqn, "GodotProcessManager", return_value=manager), mock.patch.object(dqn, "ScenarioGymEnv", DiscreteEnv), mock.patch.object(dqn, "ReplayBuffer", side_effect=buffer_factory), mock.patch.object(dqn.ParallelEnvStepper, "step", record_steps), contextlib.redirect_stdout(output):
                    dqn.main()
                checkpoint = tf.train.latest_checkpoint(str(directory / "checkpoints"))
                self.assertIsNotNone(checkpoint)
                self.assertTrue(all(env.closed for env in DiscreteEnv.instances))
                manager.stop_all.assert_called_once()
                self.assertTrue((directory / "checkpoints/policy.keras").is_file())
                values = {key: tf.train.load_variable(checkpoint, key + "/.ATTRIBUTES/VARIABLE_VALUE") for key in ("episode", "epsilon", "epsilon_transition_count", "target_sync_transition_count", "sync_autoreset", "sync_env_resets", "sync_next_episode", "optimizer/_iterations")}
                return {"steps": [env.steps for env in DiscreteEnv.instances], "seeds": [env.seeds for env in DiscreteEnv.instances], "configurations": [env.configurations for env in DiscreteEnv.instances], "requests": requests, "snapshot": buffers[0].export_snapshot(), "checkpoint": values, "output": output.getvalue()}
            finally:
                set_network_layers(original_network_layers)

    def test_short_episode_restarts_without_waiting_for_long_episode(self):
        result = self.run_training(14)
        self.assertEqual(result["requests"], [2] * 7)
        self.assertEqual(result["steps"], [7, 7])
        self.assertEqual(len(result["seeds"][0]), 4)
        self.assertEqual(len(result["seeds"][1]), 1)
        self.assertEqual(int(result["checkpoint"]["episode"]), 4)
        self.assertIn("independent episode resets", result["output"])

    def test_nonmultiple_budget_is_exact_and_does_not_reset_at_stop(self):
        result = self.run_training(13)
        self.assertEqual(sum(result["steps"]), 13)
        self.assertEqual(result["requests"][-1], 1)
        self.assertEqual(int(result["checkpoint"]["epsilon_transition_count"]), 13)
        self.assertEqual(int(result["checkpoint"]["target_sync_transition_count"]), 13)
        np.testing.assert_array_equal(result["checkpoint"]["sync_env_resets"], [4, 1])
        self.assertEqual(result["output"].count("Transition budget reached: 13/13"), 1)

    def test_unfinished_episodes_are_not_counted_as_completed(self):
        result = self.run_training(8, lengths=(0, 0))
        self.assertEqual(int(result["checkpoint"]["episode"]), 0)
        self.assertEqual([len(seeds) for seeds in result["seeds"]], [1, 1])
        self.assertIn("pending", result["output"])

    def test_terminal_observation_is_preserved_before_reset(self):
        result = self.run_training(14)
        snapshot = result["snapshot"]
        for next_obs, done in zip(snapshot["next_obs"], snapshot["dones"]):
            if done:
                self.assertGreater(next_obs[2], 0)
        self.assertEqual(int(sum(snapshot["dones"])), 4)

    def test_truncation_resets_but_keeps_bootstrapping(self):
        result = self.run_training(4, lengths=(2, 2), truncations=(False, True))
        self.assertEqual(int(result["checkpoint"]["episode"]), 2)
        self.assertEqual(int(sum(result["snapshot"]["dones"])), 1)

    def test_python_episode_cap_resets_without_inventing_termination(self):
        result = self.run_training(12, lengths=(0, 0), episode_cap=3)
        self.assertEqual(int(result["checkpoint"]["episode"]), 4)
        self.assertEqual(int(sum(result["snapshot"]["dones"])), 0)

    def test_episode_limit_counts_real_completed_episodes(self):
        result = self.run_training(0, lengths=(1, 3), episode_limit=3)
        self.assertEqual(int(result["checkpoint"]["episode"]), 3)
        self.assertEqual(result["steps"], [2, 3])
        self.assertEqual([len(seeds) for seeds in result["seeds"]], [2, 1])

    def test_reporting_window_does_not_change_transitions_or_seeds(self):
        first = self.run_training(14, reporting_steps=1)
        second = self.run_training(14, reporting_steps=100)
        self.assertEqual(first["steps"], second["steps"])
        self.assertEqual(first["seeds"], second["seeds"])
        np.testing.assert_array_equal(first["snapshot"]["next_obs"], second["snapshot"]["next_obs"])

    def test_updates_and_exploration_use_actual_accepted_transitions(self):
        result = self.run_training(14, learn=True)
        self.assertEqual(int(result["checkpoint"]["optimizer/_iterations"]), 12)
        self.assertAlmostEqual(float(result["checkpoint"]["epsilon"]), dqn.epsilon_at_transition(1.0, 0.05, 20, 14), places=6)

    def test_interrupt_saves_live_episode_and_exploration_counters(self):
        result = self.run_training(100, interrupt=True)
        self.assertEqual(sum(result["steps"]), 4)
        self.assertEqual(int(result["checkpoint"]["episode"]), 1)
        self.assertEqual(int(result["checkpoint"]["epsilon_transition_count"]), 4)
        self.assertAlmostEqual(float(result["checkpoint"]["epsilon"]), dqn.epsilon_at_transition(1.0, 0.05, 20, 4), places=6)

    def test_mode_is_opt_in_and_does_not_change_other_collectors(self):
        args = type("Args", (), {"collector_mode": "sync", "sync_batch_steps": 100, "multi_agent": False, "multi_policy": False, "opponent_pool": False, "auto_recovery": False})()
        self.assertTrue(dqn.uses_sync_autoreset(args))
        for key, value in (("sync_batch_steps", 0), ("collector_mode", "async"), ("multi_agent", True), ("multi_policy", True), ("opponent_pool", True), ("auto_recovery", True)):
            with mock.patch.object(args, key, value):
                self.assertFalse(dqn.uses_sync_autoreset(args))

    def test_resume_rejects_silent_collector_mode_change(self):
        args = type("Args", (), {"collector_mode": "sync", "sync_batch_steps": 100, "multi_agent": False, "multi_policy": False, "opponent_pool": False, "auto_recovery": False, "num_envs": 2})()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = tf.train.Checkpoint(episode=tf.Variable(0))
            path = checkpoint.save(str(Path(temporary) / "old"))
            with self.assertRaisesRegex(RuntimeError, "cannot change"):
                dqn.validate_sync_collection_resume(args, path)

    def test_resume_validates_reset_counter_shape(self):
        args = type("Args", (), {"collector_mode": "sync", "sync_batch_steps": 100, "multi_agent": False, "multi_policy": False, "opponent_pool": False, "auto_recovery": False, "num_envs": 2})()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = tf.train.Checkpoint(sync_autoreset=tf.Variable(True), sync_next_episode=tf.Variable(3), sync_env_resets=tf.Variable([2, 1]))
            path = checkpoint.save(str(Path(temporary) / "new"))
            dqn.validate_sync_collection_resume(args, path)
            args.num_envs = 3
            with self.assertRaisesRegex(RuntimeError, "environment count"):
                dqn.validate_sync_collection_resume(args, path)


if __name__ == "__main__":
    unittest.main()
