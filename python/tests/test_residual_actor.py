"""Residual actor for the M6.1 canary: zero-init (delta==0 at update 0), the hard DELTA_SCALE
deviation bound, clip semantics, and that the canary objective's gradient touches ONLY the residual
(the frozen BC actor and critics never move)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.residual_actor import (  # noqa: E402
    DELTA_SCALE, build_residual_actor, canary_actor_loss, combined_action, residual_delta,
    residual_is_zero, assert_within_bound, max_action_deviation)

OBS, ACT = 27, 7


def _bc_fn(seed=0):
    m = tf.keras.Sequential([tf.keras.layers.Input((OBS,)),
                             tf.keras.layers.Dense(ACT, activation="tanh")])
    return m


class ResidualActorTests(unittest.TestCase):
    def test_zero_init_delta_is_zero(self):
        r = build_residual_actor(OBS, ACT)
        obs = np.random.RandomState(0).randn(16, OBS).astype(np.float32)
        self.assertTrue(residual_is_zero(r, obs))
        d = residual_delta(r, obs).numpy()
        self.assertTrue(np.allclose(d, 0.0))

    def test_combined_equals_bc_at_init_within_1e6(self):
        r = build_residual_actor(OBS, ACT)
        bc = _bc_fn()
        obs = np.random.RandomState(1).randn(32, OBS).astype(np.float32)
        action, delta = combined_action(lambda o: bc(o, training=False), r, obs)
        bc_out = bc(obs, training=False).numpy()
        self.assertLess(float(np.max(np.abs(action.numpy() - bc_out))), 1e-6)

    def test_deviation_bound_holds_even_with_large_residual(self):
        r = build_residual_actor(OBS, ACT)
        # force a large residual: set output-layer weights huge -> tanh saturates to +-1 -> delta=+-0.05
        w = r.get_weights(); w[-2] = w[-2] + 5.0; w[-1] = w[-1] + 5.0; r.set_weights(w)
        bc = _bc_fn()
        obs = np.random.RandomState(2).randn(64, OBS).astype(np.float32)
        action, delta = combined_action(lambda o: bc(o, training=False), r, obs)
        bc_out = bc(obs, training=False)
        dev = max_action_deviation(action, bc_out)
        self.assertLessEqual(dev, DELTA_SCALE + 1e-6)          # bounded by construction
        self.assertGreater(float(np.max(np.abs(delta.numpy()))), 0.049)   # residual is actually active
        assert_within_bound(action, bc_out)                    # does not raise

    def test_assert_within_bound_raises_on_violation(self):
        bc = tf.zeros((4, ACT))
        bad = tf.fill((4, ACT), 0.2)                           # deviation 0.2 > 0.05
        with self.assertRaises(AssertionError):
            assert_within_bound(bad, bc)


class CanaryObjectiveTests(unittest.TestCase):
    def _critic_min_q(self):
        # differentiable stand-in critic: Q(s,a) = -sum(a^2) (prefers small actions)
        return lambda o, a: -tf.reduce_sum(a ** 2, axis=1)

    def test_gradient_to_residual_bc_frozen_and_critic_unchanged_after_step(self):
        r = build_residual_actor(OBS, ACT)
        bc = _bc_fn()
        critic = tf.keras.Sequential([tf.keras.layers.Input((ACT,)), tf.keras.layers.Dense(1)])
        def cmq(o, a):
            return tf.reshape(critic(a), [-1])
        obs = np.random.RandomState(3).randn(32, OBS).astype(np.float32)
        crit_before = [w.copy() for w in critic.get_weights()]
        bc_before = [w.copy() for w in bc.get_weights()]
        opt = tf.keras.optimizers.Adam(1e-5)
        with tf.GradientTape(persistent=True) as g:
            loss, delta, action, tel = canary_actor_loss(
                r, lambda o: bc(o, training=False), cmq, obs, q_weight=0.02)
        # residual gets gradient; BC actor gets NONE (stop_gradient blocks the base actor)
        self.assertTrue(any(x is not None for x in g.gradient(loss, r.trainable_variables)))
        self.assertTrue(all(x is None for x in g.gradient(loss, bc.trainable_variables)))
        # apply ONLY to the residual -> base actor + critic weights are bit-identical afterwards
        opt.apply_gradients(zip(g.gradient(loss, r.trainable_variables), r.trainable_variables))
        for a, b in zip(critic.get_weights(), crit_before):
            self.assertTrue(np.array_equal(a, b))          # critic never updated
        for a, b in zip(bc.get_weights(), bc_before):
            self.assertTrue(np.array_equal(a, b))          # BC clone never updated

    def test_anchor_term_is_10_delta_sq(self):
        r = build_residual_actor(OBS, ACT)
        w = r.get_weights(); w[-2] = w[-2] + 3.0; r.set_weights(w)   # nonzero delta
        bc = _bc_fn()
        obs = np.random.RandomState(4).randn(8, OBS).astype(np.float32)
        loss, delta, action, tel = canary_actor_loss(
            r, lambda o: bc(o, training=False), self._critic_min_q(), obs, q_weight=0.0)
        # with q_weight=0, loss == anchor == 10 * mean(delta^2)
        self.assertAlmostEqual(float(loss.numpy()), 10.0 * tel["delta_sq_mean"], places=5)
        self.assertAlmostEqual(tel["anchor_loss"], 10.0 * tel["delta_sq_mean"], places=5)

    def test_action_stays_bounded_in_objective(self):
        r = build_residual_actor(OBS, ACT)
        w = r.get_weights(); w[-2] = w[-2] + 9.0; w[-1] = w[-1] + 9.0; r.set_weights(w)
        bc = _bc_fn()
        obs = np.random.RandomState(5).randn(16, OBS).astype(np.float32)
        _l, _d, action, _t = canary_actor_loss(
            r, lambda o: bc(o, training=False), self._critic_min_q(), obs, q_weight=0.02)
        bc_out = bc(obs, training=False)
        self.assertLessEqual(max_action_deviation(action, bc_out), DELTA_SCALE + 1e-6)


if __name__ == "__main__":
    unittest.main()
