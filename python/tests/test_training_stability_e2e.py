import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from algorithms import common, dqn, ppo, sac
from core.models import network_layers, set_network_layers
from core.training import BestCheckpointTracker, PolicyEvaluationResult


class TestEnv:
    instances = []
    obs_dim = 3
    action_type = "continuous"
    action_size = 1
    num_actions = 0
    action_names = ["force"]
    action_low = np.asarray([-1.], np.float32)
    action_high = np.asarray([1.], np.float32)
    action_space_spec = {"force": {"action_type": "continuous", "size": 1, "low": -1., "high": 1.}}
    agent_id = "Agent"
    agent_ids = ["Agent"]
    agent_team_ids = [0]

    def __init__(self, **unused):
        self.steps = 0
        self.closed = False
        self.instances.append(self)

    def configure(self, **unused):
        pass

    def reset(self, seed=None):
        self.steps = 0
        return np.ones(3, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(3, np.float32), 1., self.steps >= 2, False, {"agent_info": {}}

    def close(self):
        self.closed = True

    def agent_summary(self):
        return "single agent"

    def team_summary(self):
        return "single team"

    def _spec_for_agent(self, agent_id):
        return {"observation_names": ["x", "y", "z"], "action_space": self.action_space_spec}


class DiscreteEnv(TestEnv):
    action_type = "discrete"
    action_size = 0
    num_actions = 3
    action_names = ["idle", "left", "right"]
    action_space_spec = {"action": {"action_type": "discrete", "size": 3}}


class StabilityEntrypointTests(unittest.TestCase):
    def run_training(self, algorithm, mode, *, ema=False, fail_first=False, resume=False):
        module = {"sac": sac, "ppo": ppo, "dqn": dqn, "td3": common, "ddpg": common}[algorithm]
        old_layers = network_layers()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                argv = [algorithm, "--collector-mode", mode, "--num-envs", "2", "--no-parallel-env-steps", "--num-episodes", "10", "--max-steps-per-episode", "2", "--network-layers", "8", "8", "--batch-size", "2", "--best-evaluation-every", "1", "--best-evaluation-episodes", "4", "--stop-when-solved", ".95", "--solved-patience", "1", "--no-dashboard", "--no-health-monitor", "--no-auto-recovery", "--checkpoint-dir", str(root / "checkpoints"), "--lr-decay-on-plateau", ".5", "--lr-plateau-patience", "2", "--recovery-lr-cumulative"]
                if algorithm == "ppo":
                    argv.extend(["--ppo-epochs", "1", "--weights-path", str(root / "policy.weights.h5")])
                elif algorithm == "dqn":
                    argv.extend(["--replay-warmup", "2", "--weights-path", str(root / "q.weights.h5")])
                else:
                    argv.extend(["--replay-warmup", "2", "--critic-warmup-updates", "0", "--random-exploration-episodes", "0", "--actor-weights-path", str(root / "actor.weights.h5")])
                    if algorithm == "sac":
                        argv.extend(["--critic1-weights-path", str(root / "critic1.weights.h5"), "--critic2-weights-path", str(root / "critic2.weights.h5")])
                    else:
                        argv.extend(["--critic-weights-path", str(root / "critic.weights.h5")])
                        if algorithm == "td3":
                            argv.extend(["--critic2-weights-path", str(root / "critic2.weights.h5")])
                if ema:
                    argv.extend(["--actor-ema-decay", ".9"])
                attempts = []

                def evaluate(tracker, prefix, episode, *, certification=None):
                    if certification:
                        attempts.append(certification)
                    trials = certification["episodes"] if certification else 4
                    successes = 0 if certification and fail_first and len(attempts) == 1 else trials
                    summary = {"episodes": trials, "trials": trials, "successes": successes, "success_rate": successes / trials, "reward_mean": 1., "steps_mean": 2.}
                    return PolicyEvaluationResult(episode, summary)

                manager = mock.Mock()
                TestEnv.instances = []
                output = io.StringIO()
                with mock.patch("sys.argv", argv), mock.patch.object(module, "GodotProcessManager", return_value=manager), mock.patch.object(module, "ScenarioGymEnv", DiscreteEnv if algorithm == "dqn" else TestEnv), mock.patch.object(BestCheckpointTracker, "evaluate", evaluate), contextlib.redirect_stdout(output):
                    module.main(algorithm) if module is common else module.main()
                self.assertEqual(len(attempts), 2 if fail_first else 1)
                state = json.loads((root / "checkpoints/training_control.json").read_text())
                self.assertTrue(state["certified"]["passed"])
                self.assertLess(state["certified"]["candidate_episode"], 10)
                self.assertTrue(all(env.closed for env in TestEnv.instances))
                manager.stop_all.assert_called_once()
                self.assertIn("Training solved: certified policy=", output.getvalue())
                self.assertIsNotNone(tf.train.latest_checkpoint(str(root / "checkpoints")))
                candidate = Path(state["certified"]["policy"])
                self.assertTrue(candidate.is_file() if ema else Path(str(candidate) + ".index").is_file())
                if ema:
                    loaded = tf.keras.models.load_model(candidate, compile=False)
                    self.assertEqual(loaded.input_shape[-1], 3)
                    latest = tf.train.latest_checkpoint(str(root / "checkpoints"))
                    names = dict(tf.train.list_variables(latest))
                    self.assertIn("actor_ema/updates/.ATTRIBUTES/VARIABLE_VALUE", names)
                if fail_first:
                    self.assertGreater(attempts[1]["seed"], attempts[0]["seed"])
                if resume:
                    latest = tf.train.latest_checkpoint(str(root / "checkpoints"))
                    saved_ema = {name: tf.train.load_variable(latest, name) for name, _shape in tf.train.list_variables(latest) if name.startswith("actor_ema/") and ".ATTRIBUTES/VARIABLE_VALUE" in name}
                    manager.reset_mock()
                    with mock.patch("sys.argv", [*argv, "--resume"]), mock.patch.object(module, "GodotProcessManager", return_value=manager), mock.patch.object(module, "ScenarioGymEnv", TestEnv), mock.patch.object(BestCheckpointTracker, "evaluate", evaluate), contextlib.redirect_stdout(output):
                        module.main(algorithm) if module is common else module.main()
                    latest = tf.train.latest_checkpoint(str(root / "checkpoints"))
                    for name, value in saved_ema.items():
                        np.testing.assert_array_equal(tf.train.load_variable(latest, name), value)
                    manager.stop_all.assert_called_once()
        finally:
            set_network_layers(old_layers)

    def test_sac_sync_ema_certification_stops_cleanly(self):
        self.run_training("sac", "sync", ema=True)

    def test_sac_async_failure_continues_then_certifies(self):
        self.run_training("sac", "async", fail_first=True)

    def test_ppo_sync_certification_stops_cleanly(self):
        self.run_training("ppo", "sync")

    def test_ppo_async_certification_stops_cleanly(self):
        self.run_training("ppo", "async")

    def test_dqn_sync_certification_stops_cleanly(self):
        self.run_training("dqn", "sync")

    def test_dqn_async_certification_stops_cleanly(self):
        self.run_training("dqn", "async")

    def test_td3_sync_ema_certification_stops_cleanly(self):
        self.run_training("td3", "sync", ema=True)

    def test_ddpg_async_ema_certification_stops_cleanly(self):
        self.run_training("ddpg", "async", ema=True)

    def test_sac_ema_survives_real_entrypoint_resume(self):
        self.run_training("sac", "sync", ema=True, resume=True)


if __name__ == "__main__":
    unittest.main()
