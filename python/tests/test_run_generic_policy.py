import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")

import numpy as np
import tensorflow as tf


from core.multi_policy import PolicyAssignment
from run import (
    agent_succeeded,
    build_multi_policy_checkpoint,
    checkpoint_episode,
    normalize_checkpoint_path,
    parse_args,
    resolve_algorithm,
    should_preserve_state,
    summarize_episode_outcome,
    wait_for_realtime_tick,
)


class RunGenericPolicyTests(unittest.TestCase):
    @staticmethod
    def _linear_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(1, use_bias=False),
        ])
        model(np.zeros((1, 2), dtype=np.float32), training=False)
        return model

    def test_no_reset_cli_disables_automatic_episode_reset(self):
        self.assertFalse(parse_args(["--no-reset"]).reset)
        self.assertTrue(parse_args([]).reset)

    def test_no_initial_reset_preserves_the_current_scene_state(self):
        self.assertFalse(parse_args(["--no-initial-reset"]).initial_reset)
        self.assertTrue(parse_args([]).initial_reset)

    def test_policy_export_flags_are_explicit(self):
        args = parse_args([
            "--export-policy-dir",
            "checkpoints/exported",
            "--export-policy-only",
        ])
        self.assertEqual(args.export_policy_dir, "checkpoints/exported")
        self.assertTrue(args.export_policy_only)

    def test_no_reset_preserves_state_after_every_terminal_boundary(self):
        args = parse_args(["--no-initial-reset", "--no-reset"])
        self.assertTrue(should_preserve_state(args, 0))
        self.assertTrue(should_preserve_state(args, 1))

        args = parse_args([])
        self.assertFalse(should_preserve_state(args, 0))
        self.assertFalse(should_preserve_state(args, 1))

        args = parse_args(["--no-reset"])
        self.assertFalse(should_preserve_state(args, 0))
        self.assertTrue(should_preserve_state(args, 1))

    def test_normalize_checkpoint_path_accepts_prefix_and_index_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_prefix = Path(temp_dir) / "ckpt-42"
            checkpoint_prefix.with_suffix(".index").touch()

            expected = str(checkpoint_prefix)
            self.assertEqual(normalize_checkpoint_path(expected), expected)
            self.assertEqual(normalize_checkpoint_path(f"{expected}.index"), expected)
            self.assertEqual(checkpoint_episode(expected), 42)

    def test_normalize_checkpoint_path_rejects_missing_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "Checkpoint index not found"):
            normalize_checkpoint_path("missing/ckpt-10")

    def test_success_is_detected_from_terminal_reason_or_event(self):
        self.assertTrue(agent_succeeded({"terminal_reason": "level_cleared"}))
        self.assertTrue(agent_succeeded({"events": {"finish_reached": 1.0}}))
        self.assertFalse(agent_succeeded({"terminal_reason": "life_lost"}))

    def test_multi_agent_outcome_counts_each_agent(self):
        info = {
            "per_agent_infos": [
                {"terminal_reason": "level_cleared"},
                {"terminal_reason": "life_lost"},
            ]
        }

        successes, trials, reasons = summarize_episode_outcome(info, multi_agent=True)

        self.assertEqual(successes, 1)
        self.assertEqual(trials, 2)
        self.assertEqual(reasons, ["level_cleared", "life_lost"])

    def test_realtime_pacing_waits_only_until_the_next_deadline(self):
        sleeps = []
        deadline = wait_for_realtime_tick(
            0.0,
            100.0,
            clock=lambda: 0.004,
            sleeper=sleeps.append,
        )
        self.assertAlmostEqual(deadline, 0.01)
        self.assertAlmostEqual(sleeps[0], 0.006)

        recovered = wait_for_realtime_tick(
            deadline,
            100.0,
            clock=lambda: 0.05,
            sleeper=sleeps.append,
        )
        self.assertAlmostEqual(recovered, 0.05)
        self.assertEqual(len(sleeps), 1)

    def test_manifest_algorithm_wins_over_action_space_heuristic(self):
        env = type("Env", (), {"action_type": "discrete"})()
        manifest = {"algorithm": "ppo"}
        self.assertEqual(resolve_algorithm("auto", env, "legacy.weights.h5", manifest), "ppo")

    def test_explicit_algorithm_must_match_manifest(self):
        env = type("Env", (), {"action_type": "discrete"})()
        with self.assertRaisesRegex(RuntimeError, "does not match policy manifest"):
            resolve_algorithm("dqn", env, "legacy.weights.h5", {"algorithm": "ppo"})

    def test_multi_policy_checkpoint_restores_each_model_by_policy_key(self):
        assignment = PolicyAssignment(
            mode="policy_id",
            policy_ids=("red", "blue"),
            agent_to_policy={"RedAgent": "red", "BlueAgent": "blue"},
            trainable_policy_ids=("red", "blue"),
            policy_keys={"red": "red", "blue": "blue"},
        )
        source = {
            "red": self._linear_model(),
            "blue": self._linear_model(),
        }
        source["red"].set_weights([
            np.full((2, 1), 1.5, dtype=np.float32),
        ])
        source["blue"].set_weights([
            np.full((2, 1), -2.0, dtype=np.float32),
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = tf.train.Checkpoint(
                episode=tf.Variable(12, dtype=tf.int64),
                policies=tf.train.Checkpoint(
                    red=tf.train.Checkpoint(model=source["red"]),
                    blue=tf.train.Checkpoint(model=source["blue"]),
                ),
            )
            saved_path = tf.train.CheckpointManager(
                checkpoint,
                temp_dir,
                max_to_keep=1,
            ).save(checkpoint_number=12)

            restored = {
                "red": self._linear_model(),
                "blue": self._linear_model(),
            }
            build_multi_policy_checkpoint(
                restored,
                assignment,
                "dqn",
            ).restore(saved_path).expect_partial()

            np.testing.assert_allclose(
                restored["red"].get_weights()[0],
                source["red"].get_weights()[0],
            )
            np.testing.assert_allclose(
                restored["blue"].get_weights()[0],
                source["blue"].get_weights()[0],
            )


if __name__ == "__main__":
    unittest.main()
