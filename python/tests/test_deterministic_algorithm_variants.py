import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import build_continuous_actor, build_continuous_critic  # noqa: E402
from core.replay_buffer import ReplayBuffer  # noqa: E402
from algorithms.common import (  # noqa: E402
    scheduled_bc_weight,
    train_deterministic_step,
    variant_uses_joint_bc,
    variant_uses_td3,
)
import train  # noqa: E402
from train import BACKENDS  # noqa: E402


def filled_buffer(count=32, obs_dim=3, action_size=2, **kwargs):
    buffer = ReplayBuffer(capacity=max(64, count + 1), **kwargs)
    rng = np.random.default_rng(7)
    for _ in range(count):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        action = rng.uniform(-1.0, 1.0, size=action_size).astype(np.float32)
        next_obs = (obs + rng.normal(scale=0.1, size=obs_dim)).astype(np.float32)
        buffer.add(obs, action, float(rng.normal()), next_obs, False)
    return buffer


class VariantSelectionTests(unittest.TestCase):
    def test_unified_entrypoint_dispatches_all_deterministic_variants(self):
        expected = {
            "ddpg": "algorithms.ddpg",
            "ddpg_bc": "algorithms.ddpg_bc",
            "ddpgfd": "algorithms.ddpgfd",
            "td3": "algorithms.td3",
            "td3_bc": "algorithms.td3_bc",
        }
        self.assertEqual({name: BACKENDS[name] for name in expected}, expected)
        trainer_dir = Path(__file__).resolve().parents[1] / "algorithms"
        for variant in expected:
            source = (trainer_dir / f"{variant}.py").read_text()
            self.assertIn(f'run_training("{variant}")', source)

    def test_variant_families(self):
        self.assertTrue(variant_uses_td3("td3"))
        self.assertTrue(variant_uses_td3("td3_bc"))
        self.assertFalse(variant_uses_td3("ddpg_bc"))
        self.assertTrue(variant_uses_joint_bc("ddpg_bc"))
        self.assertTrue(variant_uses_joint_bc("td3_bc"))
        self.assertFalse(variant_uses_joint_bc("ddpgfd"))


class UnifiedEntrypointTests(unittest.TestCase):
    def test_dispatch_imports_the_selected_backend_and_forwards_clean_arguments(self):
        backend = SimpleNamespace(main=Mock())
        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "python/train.py",
                "--algorithm",
                "td3",
                "--num-episodes",
                "10",
                "--no-headless",
            ]
            with patch.object(train.importlib, "import_module", return_value=backend) as importer:
                train.main()
        finally:
            forwarded_argv = list(sys.argv)
            sys.argv = original_argv

        importer.assert_called_once_with("algorithms.td3")
        backend.main.assert_called_once_with()
        self.assertEqual(
            forwarded_argv,
            ["python/train.py", "--num-episodes", "10", "--no-headless"],
        )

    def test_bc_schedule_interpolates_and_clamps(self):
        self.assertEqual(scheduled_bc_weight(0, 1.0, 0.1, 100), 1.0)
        self.assertAlmostEqual(scheduled_bc_weight(50, 1.0, 0.1, 100), 0.55)
        self.assertAlmostEqual(scheduled_bc_weight(500, 1.0, 0.1, 100), 0.1)

    def test_policy_path_is_forwarded_to_backend(self):
        backend = SimpleNamespace(main=Mock())
        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "python/train.py",
                "--algorithm",
                "dqn",
                "--policy-path",
                "saved/policy.h5",
            ]
            with patch.object(train.importlib, "import_module", return_value=backend):
                train.main()
        finally:
            forwarded_argv = list(sys.argv)
            sys.argv = original_argv

        self.assertEqual(
            forwarded_argv,
            ["python/train.py", "--policy-path", "saved/policy.h5", "--headless"],
        )

    def test_policy_manifest_selects_backend_without_probing_scenario(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.keras"
            policy_path.touch()
            (Path(temp_dir) / "policy.json").write_text(
                json.dumps({"format": "metis-policy", "algorithm": "ppo"}),
                encoding="utf-8",
            )
            args = SimpleNamespace(algorithm="auto", policy_path=str(policy_path))

            with patch.object(train, "probe_action_type") as probe:
                backend = train.select_backend(args)

        self.assertEqual(backend, "ppo")
        probe.assert_not_called()


class DeterministicLearnerTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(11)
        self.obs_dim = 3
        self.action_size = 2
        self.low = np.full((self.action_size,), -1.0, dtype=np.float32)
        self.high = np.full((self.action_size,), 1.0, dtype=np.float32)

    def models(self, twin=False):
        actor = build_continuous_actor(self.obs_dim, self.action_size)
        critic1 = build_continuous_critic(self.obs_dim, self.action_size)
        target_actor = build_continuous_actor(self.obs_dim, self.action_size)
        target_critic1 = build_continuous_critic(self.obs_dim, self.action_size)
        target_actor.set_weights(actor.get_weights())
        target_critic1.set_weights(critic1.get_weights())
        actor_opt = tf.keras.optimizers.Adam(1e-3)
        critic1_opt = tf.keras.optimizers.Adam(1e-3)
        if not twin:
            return actor, critic1, target_actor, target_critic1, actor_opt, critic1_opt, None, None, None
        critic2 = build_continuous_critic(self.obs_dim, self.action_size)
        target_critic2 = build_continuous_critic(self.obs_dim, self.action_size)
        target_critic2.set_weights(critic2.get_weights())
        critic2_opt = tf.keras.optimizers.Adam(1e-3)
        return (
            actor,
            critic1,
            target_actor,
            target_critic1,
            actor_opt,
            critic1_opt,
            critic2,
            target_critic2,
            critic2_opt,
        )

    def test_td3_updates_both_critics_and_delays_actor(self):
        models = self.models(twin=True)
        buffer = filled_buffer(obs_dim=self.obs_dim, action_size=self.action_size)
        first = train_deterministic_step(
            *models[:6], buffer, 16, 0.99, self.low, self.high,
            variant="td3", learner_update=1,
            critic2=models[6], target_critic2=models[7], critic2_optimizer=models[8],
            policy_delay=2,
        )
        second = train_deterministic_step(
            *models[:6], buffer, 16, 0.99, self.low, self.high,
            variant="td3", learner_update=2,
            critic2=models[6], target_critic2=models[7], critic2_optimizer=models[8],
            policy_delay=2,
        )
        self.assertIsNotNone(first["critic2_loss"])
        self.assertIsNone(first["actor_loss"])
        self.assertFalse(first["target_update_due"])
        self.assertIsNotNone(second["actor_loss"])
        self.assertTrue(second["target_update_due"])

    def test_joint_bc_reports_a_loss_from_the_expert_batch(self):
        models = self.models(twin=False)
        buffer = filled_buffer(obs_dim=self.obs_dim, action_size=self.action_size)
        demo_data = {
            "obs": np.zeros((12, self.obs_dim), dtype=np.float32),
            "actions": np.ones((12, self.action_size), dtype=np.float32),
        }
        result = train_deterministic_step(
            *models[:6], buffer, 16, 0.99, self.low, self.high,
            variant="ddpg_bc", learner_update=1,
            demo_data=demo_data, demo_batch_size=8, bc_weight=1.0,
        )
        self.assertIsNotNone(result["actor_loss"])
        self.assertIsNotNone(result["bc_loss"])
        self.assertGreater(result["bc_loss"], 0.0)


class DemonstrationReplayTests(unittest.TestCase):
    def test_protected_demonstrations_survive_online_wraparound_and_reload(self):
        buffer = ReplayBuffer(
            capacity=6,
            prioritized=True,
            priority_alpha=0.3,
            priority_beta=1.0,
            demo_priority_bonus=1.0,
        )
        demo_obs = np.asarray([[100.0], [101.0]], dtype=np.float32)
        demo_actions = np.zeros((2, 1), dtype=np.float32)
        buffer.add_many(
            demo_obs,
            demo_actions,
            np.zeros((2,), dtype=np.float32),
            demo_obs,
            np.zeros((2,), dtype=np.bool_),
            is_demo=True,
            protect=True,
        )
        for value in range(20):
            obs = np.asarray([float(value)], dtype=np.float32)
            buffer.add(obs, np.zeros((1,), dtype=np.float32), 0.0, obs, False)

        snapshot = buffer.export_snapshot()
        np.testing.assert_array_equal(snapshot["obs"][:2], demo_obs)
        self.assertEqual(buffer.protected_demo_count, 2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.npz"
            buffer.save(path)
            restored = ReplayBuffer(capacity=6, prioritized=True)
            restored.load(path)
            restored_snapshot = restored.export_snapshot()
            self.assertEqual(restored.protected_demo_count, 2)
            np.testing.assert_array_equal(restored_snapshot["obs"][:2], demo_obs)

            smaller = ReplayBuffer(capacity=4, prioritized=True)
            smaller.load(path)
            smaller_snapshot = smaller.export_snapshot()
            self.assertEqual(smaller.protected_demo_count, 2)
            np.testing.assert_array_equal(smaller_snapshot["obs"][:2], demo_obs)

    def test_prioritized_sample_returns_importance_metadata(self):
        buffer = filled_buffer(
            count=16,
            obs_dim=2,
            action_size=1,
            prioritized=True,
            priority_alpha=0.3,
            priority_beta=1.0,
        )
        sample = buffer.sample_prioritized(8, action_dtype=np.float32)
        self.assertEqual(len(sample), 8)
        self.assertEqual(sample[5].shape, (8,))
        self.assertEqual(sample[6].shape, (8,))
        self.assertEqual(sample[7].shape, (8,))


if __name__ == "__main__":
    unittest.main()
