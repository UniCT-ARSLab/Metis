import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf


os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")

from core.models import build_shared_q_network
from core.policy_artifact import (
    PolicyArtifactSaver,
    build_policy_metadata,
    load_policy_into_model,
    load_policy_manifest,
    policy_algorithms_are_compatible,
)
from export import convert_tflite


class FakeEnvironment(SimpleNamespace):
    def _spec_for_agent(self, _agent_id):
        return self.agent_spec


class PolicyArtifactTests(unittest.TestCase):
    def make_env(self):
        return FakeEnvironment(
            agent_id="Paddle",
            obs_dim=5,
            action_type="discrete",
            action_size=3,
            action_names=["idle", "left", "right"],
            action_low=np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
            action_high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            action_space_spec={
                "actions": {
                    "action_type": "discrete",
                    "size": 3,
                    "names": ["idle", "left", "right"],
                }
            },
            agent_spec={"observation_names": ["paddle_x", "ball_x", "ball_y"]},
        )

    def test_saves_reloadable_keras_policy_and_manifest(self):
        model = build_shared_q_network(obs_dim=5, num_actions=3)
        sample = np.arange(5, dtype=np.float32).reshape(1, 5)
        expected = model(sample, training=False).numpy()

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = build_policy_metadata("dqn", self.make_env())
            model_path = PolicyArtifactSaver(model, temp_dir, metadata).save(42)
            restored = tf.keras.models.load_model(model_path, compile=False)
            manifest = load_policy_manifest(model_path)

            np.testing.assert_allclose(restored(sample, training=False).numpy(), expected)
            self.assertEqual(manifest["algorithm"], "dqn")
            self.assertEqual(manifest["episode"], 42)
            self.assertEqual(manifest["inference"]["decoder"], "argmax_q_values")

    def test_tflite_export_is_invokable(self):
        model = build_shared_q_network(obs_dim=5, num_actions=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = convert_tflite(model, Path(temp_dir) / "policy.tflite")
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_manifest_is_plain_json(self):
        model = build_shared_q_network(obs_dim=5, num_actions=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            PolicyArtifactSaver(model, temp_dir, build_policy_metadata("dqn", self.make_env())).save(1)
            payload = json.loads((Path(temp_dir) / "policy.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["observation"]["size"], 5)
            self.assertEqual(payload["action"]["names"], ["idle", "left", "right"])

    def test_warm_start_accepts_complete_h5_model(self):
        source = build_shared_q_network(obs_dim=5, num_actions=3)
        target = build_shared_q_network(obs_dim=5, num_actions=3)
        sample = np.arange(5, dtype=np.float32).reshape(1, 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "legacy_policy.h5"
            source.save(policy_path)
            loaded = load_policy_into_model(target, policy_path, expected_algorithm="dqn")

            self.assertEqual(loaded["source_kind"], "h5_model")
            np.testing.assert_allclose(
                target(sample, training=False).numpy(),
                source(sample, training=False).numpy(),
            )

    def test_warm_start_accepts_h5_weights(self):
        source = build_shared_q_network(obs_dim=5, num_actions=3)
        target = build_shared_q_network(obs_dim=5, num_actions=3)
        sample = np.arange(5, dtype=np.float32).reshape(1, 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "legacy_policy.weights.h5"
            source.save_weights(policy_path)
            loaded = load_policy_into_model(target, policy_path, expected_algorithm="dqn")

            self.assertEqual(loaded["source_kind"], "weights")
            np.testing.assert_allclose(
                target(sample, training=False).numpy(),
                source(sample, training=False).numpy(),
            )

    def test_deterministic_actor_algorithms_share_policy_architecture(self):
        self.assertTrue(policy_algorithms_are_compatible("td3", "ddpg"))
        self.assertFalse(policy_algorithms_are_compatible("sac", "ddpg"))


if __name__ == "__main__":
    unittest.main()
