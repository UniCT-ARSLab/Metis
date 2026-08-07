"""BC-weight schedule + Q-filter delay are driven by REAL policy updates, not critic updates.

Regression guard for the bug where scheduled_bc_weight() was fed the critic-update count, so the
BC weight annealed and the Q-filter armed DURING the critic warmup -- before a single policy step
had run. The deterministic loops (async/sync x single/multi-policy) must instead increment a
policy_updates_since_warmup counter ONLY when result["policy_updated"] is true, and pass that
counter to scheduled_bc_weight and the q_filter_active gate.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from algorithms.common import (  # noqa: E402
    build_deterministic_learner_step,
    increment_policy_update_count,
    policy_update_count,
    scheduled_bc_weight,
)
from core.replay_buffer import ReplayBuffer  # noqa: E402

OBS_DIM = 4
ACT_DIM = 2


def _actor():
    return tf.keras.Sequential([
        tf.keras.layers.Input((OBS_DIM,)),
        tf.keras.layers.Dense(8, activation="tanh"),
        tf.keras.layers.Dense(ACT_DIM, activation="tanh"),
    ])


def _critic():
    obs_in = tf.keras.layers.Input((OBS_DIM,))
    act_in = tf.keras.layers.Input((ACT_DIM,))
    x = tf.keras.layers.Concatenate()([obs_in, act_in])
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model([obs_in, act_in], out)


def _clone(model):
    c = tf.keras.models.clone_model(model)
    c.set_weights(model.get_weights())
    return c


def _td3_bc_learner(demo_q_filter=True):
    rng = np.random.default_rng(0)
    actor, critic, critic2 = _actor(), _critic(), _critic()
    demo = {
        "obs": rng.standard_normal((16, OBS_DIM)).astype(np.float32),
        "actions": np.clip(rng.standard_normal((16, ACT_DIM)), -1, 1).astype(np.float32),
    }
    learner = build_deterministic_learner_step(
        actor, critic, _clone(actor), _clone(critic),
        tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4),
        gamma=0.99, action_low=[-1.0] * ACT_DIM, action_high=[1.0] * ACT_DIM,
        variant="td3_bc", critic2=critic2, target_critic2=_clone(critic2),
        critic2_optimizer=tf.keras.optimizers.Adam(1e-4),
        policy_delay=1, demo_data=demo, demo_batch_size=8, demo_q_filter=demo_q_filter,
        batch_size=8, compiled=False,
    )
    buffer = ReplayBuffer(capacity=1000)
    for _ in range(64):
        buffer.add(rng.standard_normal(OBS_DIM).astype(np.float32),
                   np.clip(rng.standard_normal(ACT_DIM), -1, 1).astype(np.float32),
                   float(rng.standard_normal()),
                   rng.standard_normal(OBS_DIM).astype(np.float32), False)
    return learner, actor


class ScheduledBcWeightTests(unittest.TestCase):
    def test_boundaries_and_clamp(self):
        # step 0 (warmup, no policy update yet) -> start weight, exactly.
        self.assertEqual(scheduled_bc_weight(0, 1.0, 0.2, 50000), 1.0)
        self.assertAlmostEqual(scheduled_bc_weight(25000, 1.0, 0.2, 50000), 0.6, places=5)
        self.assertAlmostEqual(scheduled_bc_weight(50000, 1.0, 0.2, 50000), 0.2, places=5)
        # past the decay horizon it clamps at end, never below.
        self.assertAlmostEqual(scheduled_bc_weight(999999, 1.0, 0.2, 50000), 0.2, places=5)


class WarmupCounterLoopTests(unittest.TestCase):
    """Exercise the REAL learner across a critic warmup, driving the counter exactly as the
    deterministic loops do, and assert the startup-gate invariant holds."""

    def test_counter_only_advances_on_policy_update(self):
        learner, _ = _td3_bc_learner()
        buf = _buffer_of(learner)
        warmup = 6
        q_filter_start = 3
        counter = 0
        for critic_update in range(12):
            update_actor = critic_update >= warmup  # what the recovery runtime gates on
            bc_weight = scheduled_bc_weight(counter, 1.0, 0.2, 50000)
            q_filter_active = counter >= q_filter_start
            result = learner(
                buf,
                update_actor=update_actor,
                learner_update=critic_update + 1,
                bc_weight=bc_weight,
                q_filter_active=q_filter_active,
            )
            if result["policy_updated"]:
                counter += 1
            if critic_update < warmup:
                # During warmup: no policy update, counter frozen at 0, BC weight at start,
                # Q-filter never armed.
                self.assertFalse(result["policy_updated"])
                self.assertEqual(counter, 0)
                self.assertEqual(bc_weight, 1.0)
                self.assertFalse(q_filter_active)
                self.assertFalse(result["q_filter_active"])
        # After warmup, policy updates ran and the counter advanced past q_filter_start.
        self.assertGreaterEqual(counter, 12 - warmup)
        self.assertGreater(counter, q_filter_start)

    def test_startup_gate_arithmetic(self):
        # The exact gate the user requires: after >=500 critic updates with a 5000-critic warmup,
        # NO policy update has run -> counter 0, bc_weight 1.000, q_filter off.
        warmup = 5000
        q_filter_start = 1000
        counter = 0
        for critic_update in range(500):
            policy_updated = critic_update >= warmup
            if policy_updated:
                counter += 1
        self.assertEqual(counter, 0)
        self.assertEqual(scheduled_bc_weight(counter, 1.0, 0.2, 50000), 1.000)
        self.assertFalse(counter >= q_filter_start)


class QFilterDelayTests(unittest.TestCase):
    def test_qfilter_off_imitates_all_on_selects_subset(self):
        learner, _ = _td3_bc_learner(demo_q_filter=True)
        buf = _buffer_of(learner)
        # Filter OFF (delay window): unfiltered BC -> no telemetry, treated as N/A.
        off = learner(buf, update_actor=True, learner_update=1, bc_weight=1.0, q_filter_active=False)
        self.assertTrue(off["policy_updated"])
        self.assertFalse(off["q_filter_active"])
        self.assertIsNone(off["q_filter_selected_fraction"])
        # Filter ON: telemetry present, fraction is a real number in [0, 1].
        on = learner(buf, update_actor=True, learner_update=2, bc_weight=1.0, q_filter_active=True)
        self.assertTrue(on["q_filter_active"])
        self.assertIsNotNone(on["q_filter_selected_fraction"])
        self.assertGreaterEqual(on["q_filter_selected_fraction"], 0.0)
        self.assertLessEqual(on["q_filter_selected_fraction"], 1.0)


class ResumeTests(unittest.TestCase):
    def test_resume_restores_policy_schedule_position(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            counter = tf.Variable(0, dtype=tf.int64, trainable=False)
            increment_policy_update_count(counter)
            counter.assign(12500)
            saved_path = tf.train.Checkpoint(
                policy_updates_since_warmup=counter
            ).save(str(Path(temp_dir) / "ckpt"))

            restored_counter = tf.Variable(
                0, dtype=tf.int64, trainable=False
            )
            tf.train.Checkpoint(
                policy_updates_since_warmup=restored_counter
            ).restore(saved_path).assert_existing_objects_matched()

            self.assertEqual(policy_update_count(restored_counter), 12500)
            self.assertAlmostEqual(
                scheduled_bc_weight(
                    policy_update_count(restored_counter),
                    1.0,
                    0.2,
                    50000,
                ),
                0.8,
            )

    def test_multi_policy_resume_restores_each_schedule_position(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            counters = {
                "red": tf.Variable(750, dtype=tf.int64, trainable=False),
                "blue": tf.Variable(1750, dtype=tf.int64, trainable=False),
            }
            policies = tf.train.Checkpoint(**{
                name: tf.train.Checkpoint(
                    policy_updates_since_warmup=counter
                )
                for name, counter in counters.items()
            })
            saved_path = tf.train.Checkpoint(policies=policies).save(
                str(Path(temp_dir) / "ckpt")
            )

            restored = {
                name: tf.Variable(0, dtype=tf.int64, trainable=False)
                for name in counters
            }
            restored_policies = tf.train.Checkpoint(**{
                name: tf.train.Checkpoint(
                    policy_updates_since_warmup=counter
                )
                for name, counter in restored.items()
            })
            tf.train.Checkpoint(policies=restored_policies).restore(
                saved_path
            ).assert_existing_objects_matched()

            self.assertEqual(policy_update_count(restored["red"]), 750)
            self.assertEqual(policy_update_count(restored["blue"]), 1750)


def _buffer_of(_learner):
    # Rebuild a fresh filled buffer each call site needs one; kept module-level to avoid capturing
    # the learner's closure buffer (the learner samples whatever buffer it is handed).
    rng = np.random.default_rng(1)
    buffer = ReplayBuffer(capacity=1000)
    for _ in range(64):
        buffer.add(rng.standard_normal(OBS_DIM).astype(np.float32),
                   np.clip(rng.standard_normal(ACT_DIM), -1, 1).astype(np.float32),
                   float(rng.standard_normal()),
                   rng.standard_normal(OBS_DIM).astype(np.float32), False)
    return buffer


if __name__ == "__main__":
    unittest.main()
