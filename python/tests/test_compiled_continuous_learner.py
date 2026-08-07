import os
import unittest

import numpy as np

os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")

import tensorflow as tf

from algorithms.common import build_deterministic_learner_step, soft_update
from algorithms.sac import build_sac_learner_step
from core.models import build_continuous_actor, build_continuous_critic, build_sac_actor


OBS_DIM = 4
ACTION_SIZE = 2
ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)


def _build_learner(compiled, seed=0, actor_anchor_coef=0.0):
    tf.keras.utils.set_random_seed(seed)
    actor = build_sac_actor(OBS_DIM, ACTION_SIZE)
    critic1 = build_continuous_critic(OBS_DIM, ACTION_SIZE)
    critic2 = build_continuous_critic(OBS_DIM, ACTION_SIZE)
    target1 = build_continuous_critic(OBS_DIM, ACTION_SIZE)
    target2 = build_continuous_critic(OBS_DIM, ACTION_SIZE)
    target1.set_weights(critic1.get_weights())
    target2.set_weights(critic2.get_weights())
    actor_optimizer = tf.keras.optimizers.Adam(1e-3)
    critic1_optimizer = tf.keras.optimizers.Adam(1e-3)
    critic2_optimizer = tf.keras.optimizers.Adam(1e-3)
    alpha_optimizer = tf.keras.optimizers.Adam(1e-3)
    log_alpha = tf.Variable(np.log(0.2), dtype=tf.float32, name="log_alpha")
    return build_sac_learner_step(
        actor,
        critic1,
        critic2,
        target1,
        target2,
        actor_optimizer,
        critic1_optimizer,
        critic2_optimizer,
        alpha_optimizer,
        log_alpha,
        -float(ACTION_SIZE),
        0.99,
        ACTION_LOW,
        ACTION_HIGH,
        -20.0,
        2.0,
        compiled=compiled,
        actor_anchor_coef=actor_anchor_coef,
    )


def _fixed_batch(batch=64, seed=1):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((batch, OBS_DIM)).astype(np.float32),
        rng.uniform(-1.0, 1.0, (batch, ACTION_SIZE)).astype(np.float32),
        rng.standard_normal((batch,)).astype(np.float32),
        rng.standard_normal((batch, OBS_DIM)).astype(np.float32),
        np.zeros((batch,), dtype=np.float32),
    )


class CompiledSacLearnerTest(unittest.TestCase):
    def test_soft_update_changes_variables_without_replacing_them(self):
        source = build_continuous_critic(OBS_DIM, ACTION_SIZE)
        target = build_continuous_critic(OBS_DIM, ACTION_SIZE)
        source.set_weights([np.ones_like(value) for value in source.get_weights()])
        target.set_weights([np.zeros_like(value) for value in target.get_weights()])
        target_variable_ids = [id(variable) for variable in target.weights]

        soft_update(target, source, 0.25)

        self.assertEqual(target_variable_ids, [id(variable) for variable in target.weights])
        for value in target.get_weights():
            np.testing.assert_allclose(value, 0.25, rtol=0.0, atol=1e-6)

    def test_output_structure_both_modes(self):
        for compiled in (True, False):
            with self.subTest(compiled=compiled):
                learner = _build_learner(compiled)
                obs, actions, rewards, next_obs, dones = _fixed_batch()

                full = learner(obs, actions, rewards, next_obs, dones, update_policy=True)
                self.assertEqual(len(full), 5)
                for value in full:
                    self.assertIsNotNone(value)
                    self.assertTrue(np.isfinite(value))

                critics_only = learner(obs, actions, rewards, next_obs, dones, update_policy=False)
                self.assertIsNone(critics_only[0])  # actor loss
                self.assertIsNone(critics_only[3])  # alpha loss
                self.assertTrue(np.isfinite(critics_only[1]))
                self.assertTrue(np.isfinite(critics_only[2]))
                self.assertTrue(np.isfinite(critics_only[4]))  # alpha value

    def test_critic_loss_decreases(self):
        for compiled in (True, False):
            with self.subTest(compiled=compiled):
                learner = _build_learner(compiled)
                obs, actions, rewards, next_obs, dones = _fixed_batch()
                critic1_losses = []
                for _ in range(80):
                    losses = learner(obs, actions, rewards, next_obs, dones, update_policy=False)
                    critic1_losses.append(losses[1])
                early = float(np.mean(critic1_losses[:10]))
                late = float(np.mean(critic1_losses[-10:]))
                # Regressing critics toward a bounded target must reduce the fit error.
                self.assertLess(late, early)

    def test_actor_anchor_supports_eager_and_compiled_updates(self):
        for compiled in (True, False):
            with self.subTest(compiled=compiled):
                learner = _build_learner(compiled, actor_anchor_coef=10.0)
                losses = learner(*_fixed_batch(), update_policy=True)
                self.assertEqual(len(losses), 5)
                self.assertTrue(all(np.isfinite(value) for value in losses))


class _FixedBuffer:
    """Returns the same synthetic batch every sample so the critic can fit a fixed target."""

    def __init__(self, batch=64, seed=2):
        rng = np.random.default_rng(seed)
        self._batch = (
            rng.standard_normal((batch, OBS_DIM)).astype(np.float32),
            rng.uniform(-1.0, 1.0, (batch, ACTION_SIZE)).astype(np.float32),
            rng.standard_normal((batch,)).astype(np.float32),
            rng.standard_normal((batch, OBS_DIM)).astype(np.float32),
            np.zeros((batch,), dtype=np.float32),
        )

    def sample(self, batch_size, action_dtype=np.float32):
        return self._batch


def _build_deterministic(variant, compiled, seed=0):
    tf.keras.utils.set_random_seed(seed)
    actor = build_continuous_actor(OBS_DIM, ACTION_SIZE)
    critic = build_continuous_critic(OBS_DIM, ACTION_SIZE)
    target_actor = build_continuous_actor(OBS_DIM, ACTION_SIZE)
    target_critic = build_continuous_critic(OBS_DIM, ACTION_SIZE)
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())
    kwargs = dict(
        variant=variant,
        batch_size=64,
        compiled=compiled,
    )
    if variant == "td3":
        critic2 = build_continuous_critic(OBS_DIM, ACTION_SIZE)
        target_critic2 = build_continuous_critic(OBS_DIM, ACTION_SIZE)
        target_critic2.set_weights(critic2.get_weights())
        kwargs.update(
            critic2=critic2,
            target_critic2=target_critic2,
            critic2_optimizer=tf.keras.optimizers.Adam(1e-3),
        )
    return build_deterministic_learner_step(
        actor,
        critic,
        target_actor,
        target_critic,
        tf.keras.optimizers.Adam(1e-3),
        tf.keras.optimizers.Adam(1e-3),
        0.99,
        ACTION_LOW,
        ACTION_HIGH,
        **kwargs,
    )


class CompiledDeterministicLearnerTest(unittest.TestCase):
    def test_output_structure_and_loss_decreases(self):
        for variant in ("ddpg", "td3"):
            for compiled in (True, False):
                with self.subTest(variant=variant, compiled=compiled):
                    learner = _build_deterministic(variant, compiled)
                    buffer = _FixedBuffer()
                    critic_losses = []
                    for step in range(80):
                        result = learner(buffer, update_actor=True, learner_update=step + 1)
                        self.assertIn("critic_loss", result)
                        self.assertTrue(np.isfinite(result["critic_loss"]))
                        critic_losses.append(result["critic_loss"])
                    if variant == "td3":
                        self.assertIsNotNone(result["critic2_loss"])
                    else:
                        self.assertIsNone(result["critic2_loss"])
                    early = float(np.mean(critic_losses[:10]))
                    late = float(np.mean(critic_losses[-10:]))
                    self.assertLess(late, early)


if __name__ == "__main__":
    unittest.main()
