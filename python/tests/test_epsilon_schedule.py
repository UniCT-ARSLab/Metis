import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.dqn import (  # noqa: E402
    EPSILON_DECAY_FLOOR,
    advance_transition_epsilon,
    epsilon_at_transition,
    epsilon_checkpoint_metadata,
    episodes_to_reach_epsilon,
    resolve_epsilon_decay,
    save_training_checkpoint,
    validate_epsilon_schedule_args,
    validate_restored_transition_epsilon,
)
import tensorflow as tf  # noqa: E402


def make_args(**overrides):
    args = Namespace(
        num_episodes=2500,
        epsilon_start=1.0,
        epsilon_min=0.0,
        epsilon_decay=None,
        epsilon_decay_horizon_fraction=0.5,
        epsilon_decay_transitions=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class ResolveEpsilonDecayTests(unittest.TestCase):
    def test_pinned_decay_is_returned_unchanged(self):
        args = make_args(epsilon_decay=0.9985)
        self.assertEqual(resolve_epsilon_decay(args, 1.0, 0), 0.9985)

    def test_derived_decay_reaches_floor_at_the_horizon(self):
        args = make_args(num_episodes=2500, epsilon_min=0.0)
        decay = resolve_epsilon_decay(args, 1.0, 0)
        # epsilon-min is 0, so the schedule anneals to the floor instead.
        self.assertEqual(episodes_to_reach_epsilon(1.0, EPSILON_DECAY_FLOOR, decay), 1250)

    def test_derived_decay_honours_a_nonzero_epsilon_min(self):
        args = make_args(num_episodes=1000, epsilon_min=0.1)
        decay = resolve_epsilon_decay(args, 1.0, 0)
        self.assertEqual(episodes_to_reach_epsilon(1.0, 0.1, decay), 500)

    def test_resume_horizon_spans_only_the_remaining_episodes(self):
        args = make_args(num_episodes=2000, epsilon_min=0.0)
        # Resuming at 1500 leaves 500 episodes, so the anneal must finish within 250 of
        # them -- not stretch over the full budget as a parse-time constant would.
        decay = resolve_epsilon_decay(args, 0.5, 1500)
        self.assertEqual(episodes_to_reach_epsilon(0.5, EPSILON_DECAY_FLOOR, decay), 250)

    def test_start_already_at_or_below_target_never_decays(self):
        args = make_args(epsilon_min=0.5)
        self.assertEqual(resolve_epsilon_decay(args, 0.5, 0), 1.0)

    def test_horizon_fraction_of_one_spends_the_whole_budget(self):
        args = make_args(num_episodes=800, epsilon_min=0.0, epsilon_decay_horizon_fraction=1.0)
        decay = resolve_epsilon_decay(args, 1.0, 0)
        self.assertEqual(episodes_to_reach_epsilon(1.0, EPSILON_DECAY_FLOOR, decay), 800)

    def test_exhausted_budget_still_yields_a_usable_decay(self):
        # start_episode past num_episodes must not produce a zero/negative horizon.
        args = make_args(num_episodes=100)
        decay = resolve_epsilon_decay(args, 1.0, 500)
        self.assertGreater(decay, 0.0)
        self.assertLess(decay, 1.0)


class EpisodesToReachEpsilonTests(unittest.TestCase):
    def test_returns_zero_when_already_below_target(self):
        self.assertEqual(episodes_to_reach_epsilon(0.01, 0.1, 0.99), 0)

    def test_returns_none_when_decay_never_converges(self):
        self.assertIsNone(episodes_to_reach_epsilon(1.0, 0.1, 1.0))

    def test_matches_a_hand_computed_schedule(self):
        # 0.5**n reaches 0.125 at n=3.
        self.assertEqual(episodes_to_reach_epsilon(1.0, 0.125, 0.5), 3)


class TransitionEpsilonScheduleTests(unittest.TestCase):
    def test_linear_schedule_reaches_the_floor_at_the_exact_transition(self):
        self.assertEqual(epsilon_at_transition(1.0, 0.05, 100, 0), 1.0)
        self.assertAlmostEqual(epsilon_at_transition(1.0, 0.05, 100, 50), 0.525)
        self.assertAlmostEqual(epsilon_at_transition(1.0, 0.05, 100, 100), 0.05)
        self.assertAlmostEqual(epsilon_at_transition(1.0, 0.05, 100, 200), 0.05)

    def test_transition_and_episode_decay_cannot_be_mixed(self):
        args = make_args(epsilon_decay=0.99, epsilon_decay_transitions=100)
        with self.assertRaisesRegex(ValueError, "different schedule units"):
            validate_epsilon_schedule_args(args)

    def test_chunking_does_not_change_transition_progress(self):
        args = make_args(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay_transitions=100,
        )
        values = []
        for chunks in ([37], [10, 20, 7], [1] * 37):
            epsilon = tf.Variable(1.0, dtype=tf.float32)
            count = tf.Variable(0, dtype=tf.int64)
            for chunk in chunks:
                advance_transition_epsilon(args, epsilon, count, chunk)
            values.append((int(count.numpy()), float(epsilon.numpy())))
        self.assertEqual([count for count, _value in values], [37, 37, 37])
        self.assertTrue(all(abs(value - values[0][1]) < 1e-7 for _count, value in values))

    def test_invalid_epsilon_order_is_rejected(self):
        args = make_args(epsilon_start=0.1, epsilon_min=0.2)
        with self.assertRaisesRegex(ValueError, "epsilon-min"):
            validate_epsilon_schedule_args(args)

    def test_counter_and_value_survive_a_checkpoint_round_trip(self):
        args = make_args(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay_transitions=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            epsilon = tf.Variable(args.epsilon_start, dtype=tf.float32)
            count = tf.Variable(0, dtype=tf.int64, trainable=False)
            checkpoint = tf.train.Checkpoint(
                epsilon=epsilon,
                epsilon_transition_count=count,
                **epsilon_checkpoint_metadata(args),
            )
            advance_transition_epsilon(args, epsilon, count, 37)
            path = tf.train.CheckpointManager(
                checkpoint, directory=directory, max_to_keep=1
            ).save()

            restored_epsilon = tf.Variable(args.epsilon_start, dtype=tf.float32)
            restored_count = tf.Variable(0, dtype=tf.int64, trainable=False)
            restored = tf.train.Checkpoint(
                epsilon=restored_epsilon,
                epsilon_transition_count=restored_count,
                **epsilon_checkpoint_metadata(args),
            )
            restored.restore(path).expect_partial()
            validate_restored_transition_epsilon(
                args,
                path,
                restored,
                [
                    (
                        "shared policy",
                        restored_epsilon,
                        restored_count,
                        "epsilon_transition_count/.ATTRIBUTES/VARIABLE_VALUE",
                    )
                ],
            )

            self.assertEqual(int(restored_count.numpy()), 37)
            self.assertAlmostEqual(
                float(restored_epsilon.numpy()),
                epsilon_at_transition(1.0, 0.05, 100, 37),
                places=6,
            )

    def test_checkpoint_does_not_store_warmup_action_epsilon(self):
        args = make_args(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay_transitions=100,
            save_replay_buffer=False,
            collector_mode="async",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = tf.train.Checkpoint(
                episode=tf.Variable(0, dtype=tf.int64),
                epsilon=tf.Variable(1.0, dtype=tf.float32),
                epsilon_transition_count=tf.Variable(37, dtype=tf.int64),
                **epsilon_checkpoint_metadata(args),
            )
            manager = tf.train.CheckpointManager(
                checkpoint, directory=directory, max_to_keep=1
            )
            save_training_checkpoint(
                checkpoint,
                manager,
                buffer=(),
                episode=3,
                epsilon=1.0,
                args=args,
                save_replay=False,
            )

            self.assertAlmostEqual(
                float(checkpoint.epsilon.numpy()),
                epsilon_at_transition(1.0, 0.05, 100, 37),
                places=6,
            )

    def test_resume_rejects_a_changed_transition_horizon(self):
        original = make_args(epsilon_decay_transitions=100)
        changed = make_args(epsilon_decay_transitions=200)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = tf.train.Checkpoint(
                epsilon=tf.Variable(1.0, dtype=tf.float32),
                epsilon_transition_count=tf.Variable(0, dtype=tf.int64),
                **epsilon_checkpoint_metadata(original),
            )
            path = tf.train.CheckpointManager(
                checkpoint, directory=directory, max_to_keep=1
            ).save()
            restored = tf.train.Checkpoint(
                epsilon=tf.Variable(1.0, dtype=tf.float32),
                epsilon_transition_count=tf.Variable(0, dtype=tf.int64),
                **epsilon_checkpoint_metadata(changed),
            )
            restored.restore(path).expect_partial()
            with self.assertRaisesRegex(RuntimeError, "differs from the checkpoint"):
                validate_restored_transition_epsilon(
                    changed,
                    path,
                    restored,
                    [
                        (
                            "shared policy",
                            restored.epsilon,
                            restored.epsilon_transition_count,
                            "epsilon_transition_count/.ATTRIBUTES/VARIABLE_VALUE",
                        )
                    ],
                )

    def test_multi_policy_counter_is_found_inside_policy_checkpoint(self):
        args = make_args(epsilon_decay_transitions=100)
        with tempfile.TemporaryDirectory() as directory:
            epsilon = tf.Variable(1.0, dtype=tf.float32)
            count = tf.Variable(0, dtype=tf.int64)
            policy = tf.train.Checkpoint(
                epsilon=epsilon,
                epsilon_transition_count=count,
            )
            checkpoint = tf.train.Checkpoint(
                policies=tf.train.Checkpoint(red=policy),
                **epsilon_checkpoint_metadata(args),
            )
            advance_transition_epsilon(args, epsilon, count, 25)
            path = tf.train.CheckpointManager(
                checkpoint, directory=directory, max_to_keep=1
            ).save()

            validate_restored_transition_epsilon(
                args,
                path,
                checkpoint,
                [
                    (
                        "red",
                        epsilon,
                        count,
                        "policies/red/epsilon_transition_count/.ATTRIBUTES/VARIABLE_VALUE",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
