import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from algorithms import sac
from core.models import default_network_layers, set_network_layers
from core.replay_buffer import ReplayBuffer
from core.training_health import OffPolicyRecoveryRuntime, RecoveryRequest


class ReplayEnv:
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


def demonstrations(count=2):
    obs = np.full((count, 3), 100., dtype=np.float32)
    return {"obs": obs, "actions": np.zeros((count, 1), np.float32), "rewards": np.ones(count, np.float32), "next_obs": obs + 1., "dones": np.zeros(count, dtype=bool)}


def options(protect=True):
    return SimpleNamespace(demo_protect=protect, demo_prefill=True, demo_path=["expert.npz"], multi_policy=False, resume=False, resume_checkpoint=None, replay_capacity=8, recovery_policy_update_every=4)


class SacDemonstrationTests(unittest.TestCase):
    def test_protection_is_opt_in_and_exported_to_godot_argspec(self):
        with mock.patch("sys.argv", ["sac"]):
            self.assertFalse(sac.parse_args().demo_protect)
        with mock.patch("sys.argv", ["sac", "--demo-protect"]):
            self.assertTrue(sac.parse_args().demo_protect)
        with mock.patch("sys.argv", ["sac", "--demo-protect", "--no-demo-protect"]):
            self.assertFalse(sac.parse_args().demo_protect)
        from core.argspec import spec_for_algorithm
        row = next(a for a in spec_for_algorithm("sac")["arguments"] if a["dest"] == "demo_protect")
        self.assertEqual(row["kind"], "bool")
        self.assertFalse(row["default"])
        self.assertEqual(row["flags"], ["--demo-protect", "--no-demo-protect"])

    def test_protected_prefill_survives_online_overwrite(self):
        buffer = ReplayBuffer(8)
        self.assertEqual(sac.prefill_demonstrations(options(), buffer, demonstrations()), 2)
        for i in range(20):
            buffer.add(np.zeros(3), np.zeros(1), 0., np.zeros(3), False)
        self.assertEqual(buffer.protected_demo_count, 2)
        np.testing.assert_array_equal(buffer.export_snapshot()["obs"][:2], demonstrations()["obs"])
        self.assertEqual(sac.prefill_demonstrations(options(), buffer, demonstrations()), 0)

    def test_default_prefill_remains_overwritable(self):
        buffer = ReplayBuffer(8)
        sac.prefill_demonstrations(options(False), buffer, demonstrations())
        self.assertEqual(buffer.protected_demo_count, 0)
        for _ in range(16):
            buffer.add(np.zeros(3), np.zeros(1), 0., np.zeros(3), False)
        self.assertFalse(np.any(buffer.export_snapshot()["is_demo"]))
        self.assertEqual(sac.demonstration_metrics(buffer), [])

    def test_invalid_protection_options_fail_explicitly(self):
        for field, value in (("demo_prefill", False), ("demo_path", []), ("multi_policy", True), ("replay_capacity", 1)):
            args = options()
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                sac.validate_demo_protection(args)

    def test_empty_or_oversized_protected_sources_are_not_silently_accepted(self):
        for data in (None, demonstrations(0), demonstrations(8), demonstrations(9)):
            with self.subTest(data=None if data is None else len(data["actions"])), self.assertRaises(ValueError):
                sac.prefill_demonstrations(options(), ReplayBuffer(8), data)

    def test_cannot_claim_protection_for_an_existing_unprotected_replay(self):
        buffer = ReplayBuffer(8)
        sac.prefill_demonstrations(options(False), buffer, demonstrations())
        with self.assertRaisesRegex(ValueError, "existing unprotected replay"):
            sac.validate_demo_protection(options(), buffer)

    def test_resume_retains_prefix_without_demo_path_or_repeating_prefill(self):
        with tempfile.TemporaryDirectory() as temporary:
            buffer = ReplayBuffer(8)
            sac.prefill_demonstrations(options(), buffer, demonstrations())
            buffer.add(np.zeros(3), np.zeros(1), 0., np.zeros(3), False)
            path = Path(temporary) / "replay.npz"
            buffer.save(path)
            restored = ReplayBuffer(8)
            restored.load(path)
            args = options()
            args.resume, args.demo_path = True, []
            sac.validate_demo_protection(args)
            sac.validate_demo_protection(args, restored)
            self.assertEqual(sac.prefill_demonstrations(args, restored, None), 0)
            self.assertEqual(len(restored), 3)
            args.demo_protect = False
            sac.validate_demo_protection(args, restored)
            self.assertEqual(restored.protected_demo_count, 2)

    def test_hard_recovery_retains_existing_experts_and_does_not_duplicate_them(self):
        args, buffer = options(), ReplayBuffer(8)
        sac.prefill_demonstrations(args, buffer, demonstrations())
        buffer.add(np.zeros(3), np.zeros(1), 0., np.zeros(3), False)
        refill = mock.Mock(side_effect=lambda: sac.prefill_demonstrations(args, buffer, demonstrations()))
        runtime = OffPolicyRecoveryRuntime(args, buffer, replay_refill=refill)
        runtime.post_restore(RecoveryRequest(episode=1, checkpoint_path="checkpoint", reason="test", attempt=1, learning_rate_factor=.5, minimum_learning_rate_scale=.1, mode="hard"))
        refill.assert_not_called()
        self.assertEqual(len(buffer), buffer.protected_demo_count)
        self.assertEqual(buffer.protected_demo_count, 2)

    def test_hard_recovery_refill_recreates_protection_if_replay_was_empty(self):
        args, buffer = options(), ReplayBuffer(8)
        runtime = OffPolicyRecoveryRuntime(args, buffer, replay_refill=lambda: sac.prefill_demonstrations(args, buffer, demonstrations()))
        runtime.post_restore(RecoveryRequest(episode=1, checkpoint_path="checkpoint", reason="test", attempt=1, learning_rate_factor=.5, minimum_learning_rate_scale=.1, mode="hard"))
        self.assertEqual(buffer.protected_demo_count, 2)

    def test_metrics_report_only_tracked_learner_samples(self):
        buffer = ReplayBuffer(8)
        sac.prefill_demonstrations(options(), buffer, demonstrations())
        buffer.sample(2, action_dtype=np.float32)
        self.assertEqual(dict(sac.demonstration_metrics(buffer))["sampled_replay_transitions"], 0)
        buffer.sample(2, action_dtype=np.float32, track_demo_samples=True)
        self.assertEqual(dict(sac.demonstration_metrics(buffer))["sampled_replay_transitions"], 2)


class SacProtectedTrainingTests(unittest.TestCase):
    def run_training(self, mode, resume=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            demo = root / "expert.npz"
            np.savez_compressed(demo, **demonstrations())
            args = ["sac", "--collector-mode", mode, "--num-envs", "2", "--no-parallel-env-steps", "--no-multi-agent", "--num-episodes", "10", "--max-steps-per-episode", "1000", "--total-timesteps", "8", "--network-layers", "8", "8", "--batch-size", "2", "--replay-warmup", "2", "--replay-capacity", "8", "--critic-warmup-updates", "0", "--random-exploration-episodes", "0", "--demo-path", str(demo), "--demo-protect", "--no-dashboard", "--no-best-checkpoint", "--no-health-monitor", "--no-auto-recovery", "--checkpoint-dir", str(root / "checkpoints"), "--actor-weights-path", str(root / "actor.weights.h5"), "--critic1-weights-path", str(root / "critic1.weights.h5"), "--critic2-weights-path", str(root / "critic2.weights.h5")]
            buffer = ReplayBuffer(8)
            output = io.StringIO()
            ReplayEnv.instances = []
            manager = mock.Mock()
            try:
                with mock.patch("sys.argv", args), mock.patch.object(sac, "GodotProcessManager", return_value=manager), mock.patch.object(sac, "ScenarioGymEnv", ReplayEnv), mock.patch.object(sac, "ReplayBuffer", return_value=buffer), contextlib.redirect_stdout(output):
                    sac.main()
                self.assertTrue(all(e.closed for e in ReplayEnv.instances))
                manager.stop_all.assert_called_once()
                self.assertEqual(buffer.protected_demo_count, 2)
                self.assertGreater(buffer.demo_sampling_stats["sampled_replay_transitions"], 0)
                checkpoint = tf.train.latest_checkpoint(str(root / "checkpoints"))
                replay = root / "checkpoints" / ("replay-" + Path(checkpoint).name.split("-")[-1] + ".npz")
                restored = ReplayBuffer(8)
                restored.load(replay)
                self.assertEqual(restored.protected_demo_count, 2)
                np.testing.assert_array_equal(restored.export_snapshot()["obs"][:2], demonstrations()["obs"])
                self.assertIn("Protected demonstration replay transitions=2 online_capacity=6", output.getvalue())
                if resume:
                    resumed_args = list(args)
                    index = resumed_args.index("--demo-path")
                    del resumed_args[index:index + 2]
                    resumed_args.remove("--demo-protect")
                    resumed_args.extend(("--resume", "--require-replay-buffer", "--no-demo-prefill"))
                    resumed_buffer = ReplayBuffer(8)
                    manager.reset_mock()
                    with mock.patch("sys.argv", resumed_args), mock.patch.object(sac, "GodotProcessManager", return_value=manager), mock.patch.object(sac, "ScenarioGymEnv", ReplayEnv), mock.patch.object(sac, "ReplayBuffer", return_value=resumed_buffer), contextlib.redirect_stdout(output):
                        sac.main()
                    self.assertEqual(resumed_buffer.protected_demo_count, 2)
                    np.testing.assert_array_equal(resumed_buffer.export_snapshot()["obs"][:2], demonstrations()["obs"])
                    self.assertIn("Restored replay buffer:", output.getvalue())
                    manager.stop_all.assert_called_once()
            finally:
                set_network_layers(default_network_layers("sac"))

    def test_sync_training_preserves_demos_and_checkpoints(self):
        self.run_training("sync")

    def test_async_training_preserves_demos_and_checkpoints(self):
        self.run_training("async")

    def test_real_checkpoint_resume_preserves_demos_without_prefill_or_flag(self):
        self.run_training("sync", resume=True)


if __name__ == "__main__":
    unittest.main()
