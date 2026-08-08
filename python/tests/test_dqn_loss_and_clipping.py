"""The DQN gradient-clipping and value-loss options added on top of a learner that had neither.

Two properties matter more than the features themselves. First, the defaults must reproduce the
previous behaviour exactly -- an unclipped squared error -- so no existing run, checkpoint or
tutorial silently changes. Second, huber must actually bound the gradient of a large TD error,
because that is the entire reason to offer it.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from core.training import (  # noqa: E402
    describe_gradient_clip,
    make_gradient_clipper,
    value_loss_fn,
)


class ValueLossTests(unittest.TestCase):
    def test_mse_is_the_plain_squared_error(self):
        target = tf.constant([1.0, -2.0, 0.5])
        prediction = tf.constant([0.0, 0.0, 0.0])
        loss = value_loss_fn("mse", 1.0)(target, prediction)
        self.assertAlmostEqual(float(loss), float(np.mean([1.0, 4.0, 0.25])), places=6)

    def test_huber_matches_mse_within_delta_up_to_the_half_factor(self):
        # Inside the quadratic region Huber is 0.5 * err^2, so it is exactly half of MSE there.
        target = tf.constant([0.4, -0.3, 0.1])
        prediction = tf.zeros(3)
        mse = float(value_loss_fn("mse", 1.0)(target, prediction))
        huber = float(value_loss_fn("huber", 1.0)(target, prediction))
        self.assertAlmostEqual(huber, 0.5 * mse, places=6)

    def test_huber_is_linear_outside_delta(self):
        # The property that motivates it: doubling a large error must not quadruple the loss.
        loss = value_loss_fn("huber", 1.0)
        at_10 = float(loss(tf.constant([10.0]), tf.zeros(1)))
        at_20 = float(loss(tf.constant([20.0]), tf.zeros(1)))
        self.assertAlmostEqual(at_10, 1.0 * (10.0 - 0.5), places=6)
        self.assertAlmostEqual(at_20 - at_10, 10.0, places=6)

    def test_huber_bounds_the_gradient_of_a_large_error(self):
        # A 100-wide TD error yields a gradient of delta under huber, but of the error under mse:
        # this is what keeps one outlier transition from dominating an update.
        prediction = tf.Variable([0.0])
        target = tf.constant([100.0])
        for kind, expected in (("mse", 200.0), ("huber", 1.0)):
            with tf.GradientTape() as tape:
                loss = value_loss_fn(kind, 1.0)(target, prediction)
            gradient = float(tape.gradient(loss, prediction)[0])
            self.assertAlmostEqual(abs(gradient), expected, places=4, msg=kind)


class GradientClipperTests(unittest.TestCase):
    @staticmethod
    def gradients(scale):
        return [tf.constant([scale, 0.0]), tf.constant([0.0, scale])]

    def test_disabled_by_default_returns_the_same_objects(self):
        clip = make_gradient_clipper("t")
        grads = self.gradients(1000.0)
        self.assertIs(clip(grads), grads)

    def test_a_hard_cap_bounds_the_global_norm(self):
        clip = make_gradient_clipper("t", grad_clip_norm=1.0)
        clipped = clip(self.gradients(100.0))
        self.assertAlmostEqual(float(tf.linalg.global_norm(clipped)), 1.0, places=5)

    def test_a_hard_cap_leaves_small_gradients_untouched(self):
        clip = make_gradient_clipper("t", grad_clip_norm=10.0)
        original = self.gradients(1.0)
        clipped = clip(original)
        for before, after in zip(original, clipped):
            np.testing.assert_allclose(before.numpy(), after.numpy(), rtol=1e-6)

    def test_adaptive_uses_the_hard_cap_while_the_ema_is_cold(self):
        # Below warmup_steps the EMA has seen too little to be trusted; over-clipping a network that
        # has barely started would stall it.
        clip = make_gradient_clipper(
            "t", grad_clip_norm=5.0, grad_clip_adaptive=True, grad_clip_k=3.0, warmup_steps=10.0)
        clipped = clip(self.gradients(100.0))
        self.assertAlmostEqual(float(tf.linalg.global_norm(clipped)), 5.0, places=5)

    def test_adaptive_tracks_the_running_scale_once_warm(self):
        clip = make_gradient_clipper(
            "t", grad_clip_norm=1e6, grad_clip_adaptive=True, grad_clip_k=2.0,
            grad_clip_decay=0.99, warmup_steps=2.0)
        for _ in range(500):
            clip(self.gradients(1.0))  # global norm sqrt(2) ~= 1.414
        # A spike far above the tracked scale is pulled down towards it, not to the (huge) hard cap:
        # that is what makes the mode adaptive.
        clipped = float(tf.linalg.global_norm(clip(self.gradients(1000.0))))
        self.assertLess(clipped, 100.0)
        self.assertGreater(clipped, 2.0)

    def test_a_spike_loosens_the_adaptive_cap_by_one_minus_decay(self):
        """The EMA absorbs the CURRENT sample before the cap is computed.

        So a spike raises its own ceiling by (1 - decay) * spike * k. With the default decay of 0.99
        that is 1% and harmless, but it is the reason --grad-clip-norm must still be set: the
        adaptive cap alone cannot bound an arbitrarily large single gradient.
        """
        loose = make_gradient_clipper(
            "loose", grad_clip_norm=1e9, grad_clip_adaptive=True, grad_clip_k=1.0,
            grad_clip_decay=0.0, warmup_steps=1.0)
        loose(self.gradients(1.0))
        # decay 0.0 means the EMA IS the current sample, so cap == norm and nothing is clipped.
        spike = self.gradients(1000.0)
        self.assertAlmostEqual(
            float(tf.linalg.global_norm(loose(spike))),
            float(tf.linalg.global_norm(spike)), places=1)

        bounded = make_gradient_clipper(
            "bounded", grad_clip_norm=5.0, grad_clip_adaptive=True, grad_clip_k=1.0,
            grad_clip_decay=0.0, warmup_steps=1.0)
        bounded(self.gradients(1.0))
        self.assertAlmostEqual(float(tf.linalg.global_norm(bounded(spike))), 5.0, places=4)

    def test_a_non_finite_norm_cannot_poison_the_ema(self):
        clip = make_gradient_clipper(
            "t", grad_clip_norm=10.0, grad_clip_adaptive=True, warmup_steps=1.0)
        for _ in range(5):
            clip(self.gradients(1.0))
        clip([tf.constant([float("nan"), 0.0]), tf.constant([0.0, 0.0])])
        clipped = clip(self.gradients(1000.0))
        self.assertTrue(np.isfinite(float(tf.linalg.global_norm(clipped))))


class DescriptionTests(unittest.TestCase):
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def test_reports_each_mode(self):
        self.assertEqual(describe_gradient_clip(self.Args(grad_clip_norm=0.0)), "off")
        self.assertEqual(
            describe_gradient_clip(self.Args(grad_clip_norm=10.0)), "fixed norm=10")
        self.assertEqual(
            describe_gradient_clip(
                self.Args(grad_clip_norm=10.0, grad_clip_adaptive=True, grad_clip_k=3.0)),
            "adaptive k=3 cap=10")


class DqnLearnerDefaultsTests(unittest.TestCase):
    """The regression guard: the wired-up learner must be bit-identical with default flags."""

    @staticmethod
    def make_learner(**options):
        tf.keras.utils.set_random_seed(0)
        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(3,)), tf.keras.layers.Dense(2, use_bias=False)])
        target = tf.keras.models.clone_model(model)
        target.set_weights(model.get_weights())
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        from algorithms.dqn import build_dqn_learner_step
        step = build_dqn_learner_step(
            model, target, optimizer, 0.99, compiled=False, **options)
        return model, step

    @staticmethod
    def batch():
        rng = np.random.default_rng(0)
        return (
            rng.normal(size=(1, 4, 3)).astype(np.float32),
            rng.integers(0, 2, size=(1, 4)).astype(np.int32),
            np.array([[100.0, 0.0, -1.0, 1.0]], dtype=np.float32),  # one huge reward
            rng.normal(size=(1, 4, 3)).astype(np.float32),
            np.zeros((1, 4), dtype=np.float32),
        )

    def run_one(self, **options):
        model, step = self.make_learner(**options)
        tensors = tuple(tf.convert_to_tensor(part) for part in self.batch())

        # build_dqn_learner_step returns a buffer-driven callable; drive the inner graph directly
        # by faking the sampler, which is what the trainer's replay buffer provides.
        class Buffer:
            def sample_batches(self_inner, count, size):
                return tensors

        losses = step(Buffer(), 4, 1)
        return float(losses[0]), [w.copy() for w in model.get_weights()]

    def test_defaults_reproduce_unclipped_mse(self):
        _, tuned = self.run_one()
        _, explicit = self.run_one(critic_loss="mse", grad_clip_norm=0.0)
        for a, b in zip(tuned, explicit):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_huber_changes_the_update_when_a_td_error_is_large(self):
        # If this passed identically the flag would be doing nothing; the batch carries a reward of
        # 100 precisely so the two losses cannot agree.
        _, with_mse = self.run_one(critic_loss="mse")
        _, with_huber = self.run_one(critic_loss="huber", huber_delta=1.0)
        self.assertFalse(
            any(np.allclose(a, b) for a, b in zip(with_mse, with_huber)),
            "huber produced the same weights as mse on a batch with a reward of 100")

    def test_clipping_bounds_the_step_taken_on_that_batch(self):
        model_free, _ = self.make_learner()
        before = [w.copy() for w in model_free.get_weights()]
        _, unclipped = self.run_one(grad_clip_norm=0.0)
        _, clipped = self.run_one(grad_clip_norm=0.01)
        moved_unclipped = max(np.abs(a - b).max() for a, b in zip(before, unclipped))
        moved_clipped = max(np.abs(a - b).max() for a, b in zip(before, clipped))
        self.assertLess(moved_clipped, moved_unclipped)


if __name__ == "__main__":
    unittest.main()
