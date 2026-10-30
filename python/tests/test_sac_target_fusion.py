import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")

import tensorflow as tf

from algorithms.common import soft_update
from algorithms.sac import build_sac_learner_step, sample_actor
from core.models import build_continuous_critic, build_sac_actor


def build_case(*, compiled, fused, interval=3, tau=0.25):
    tf.keras.utils.set_random_seed(81)
    actor = build_sac_actor(4, 2, layers=[8, 8])
    critic1, critic2 = [build_continuous_critic(4, 2, layers=[8, 8]) for _ in range(2)]
    target1, target2 = [build_continuous_critic(4, 2, layers=[8, 8]) for _ in range(2)]
    target1.set_weights(critic1.get_weights())
    target2.set_weights(critic2.get_weights())
    optimizers = [tf.keras.optimizers.Adam(1e-3) for _ in range(4)]
    log_alpha = tf.Variable(np.log(0.2), dtype=tf.float32)
    learner = build_sac_learner_step(
        actor, critic1, critic2, target1, target2, *optimizers, log_alpha,
        -2.0, 0.99, [-1.0, -1.0], [1.0, 1.0], -20.0, 2.0,
        compiled=compiled, grad_clip_norm=10.0,
        tau=tau if fused else None,
        target_update_every=interval if fused else None,
    )
    return learner, (actor, critic1, critic2, target1, target2), optimizers, log_alpha


def fixed_batch():
    rng = np.random.default_rng(82)
    return (
        rng.normal(size=(16, 4)).astype(np.float32),
        rng.uniform(-1, 1, size=(16, 2)).astype(np.float32),
        rng.normal(size=16).astype(np.float32),
        rng.normal(size=(16, 4)).astype(np.float32),
        np.array([False, True] * 8, dtype=np.float32),
    )


def deterministic_sample(*args, **kwargs):
    return sample_actor(*args, **kwargs, deterministic=True)


class SacTargetFusionTests(unittest.TestCase):
    def test_fused_matches_external_updates_and_optimizer_state(self):
        for compiled in (False, True):
            with self.subTest(compiled=compiled), mock.patch("algorithms.sac.sample_actor", side_effect=deterministic_sample):
                legacy, legacy_models, legacy_optimizers, legacy_alpha = build_case(compiled=compiled, fused=False)
                fused, fused_models, fused_optimizers, fused_alpha = build_case(compiled=compiled, fused=True)
                for step in range(1, 7):
                    update_policy = step % 2 == 1
                    old_losses = legacy(*fixed_batch(), update_policy=update_policy)
                    new_losses = fused(*fixed_batch(), update_policy=update_policy)
                    if step % 3 == 0:
                        soft_update(legacy_models[3], legacy_models[1], 0.25)
                        soft_update(legacy_models[4], legacy_models[2], 0.25)
                    for old, new in zip(old_losses, new_losses):
                        if old is None:
                            self.assertIsNone(new)
                        else:
                            self.assertIsInstance(new, float)
                            np.testing.assert_allclose(new, old, rtol=1e-5, atol=1e-6)
                    for old_model, new_model in zip(legacy_models, fused_models):
                        for old, new in zip(old_model.get_weights(), new_model.get_weights()):
                            np.testing.assert_allclose(new, old, rtol=1e-5, atol=1e-6)
                for old_optimizer, new_optimizer in zip(legacy_optimizers, fused_optimizers):
                    for old, new in zip(old_optimizer.variables, new_optimizer.variables):
                        np.testing.assert_allclose(new.numpy(), old.numpy(), rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(fused_alpha.numpy(), legacy_alpha.numpy(), rtol=1e-5, atol=1e-6)

    def test_cadence_uses_restored_optimizer_iteration(self):
        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                learner, models, optimizers, _ = build_case(compiled=compiled, fused=True)
                optimizers[1].iterations.assign(5)
                before = models[3].get_weights()
                learner(*fixed_batch(), update_policy=False)
                for target, old, source in zip(models[3].get_weights(), before, models[1].get_weights()):
                    np.testing.assert_allclose(target, old + 0.25 * (source - old), rtol=1e-5, atol=1e-6)
                before = models[3].get_weights()
                learner(*fixed_batch(), update_policy=False)
                for target, old in zip(models[3].get_weights(), before):
                    np.testing.assert_array_equal(target, old)

    def test_explicit_target_decision_overrides_optimizer_cadence(self):
        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                learner, models, _, _ = build_case(compiled=compiled, fused=True, interval=1)
                before = models[3].get_weights()
                learner(*fixed_batch(), update_targets=False)
                for target, old in zip(models[3].get_weights(), before):
                    np.testing.assert_array_equal(target, old)
                learner(*fixed_batch(), update_targets=True)
                for target, old, source in zip(models[3].get_weights(), before, models[1].get_weights()):
                    np.testing.assert_allclose(target, old + 0.25 * (source - old), rtol=1e-5, atol=1e-6)

    def test_tau_endpoints_and_target_variable_identity(self):
        for tau in (0.0, 1.0):
            with self.subTest(tau=tau):
                learner, models, _, _ = build_case(compiled=True, fused=True, interval=1, tau=tau)
                identities = [id(variable) for variable in models[3].weights]
                before = models[3].get_weights()
                learner(*fixed_batch(), update_policy=False)
                expected = before if tau == 0.0 else models[1].get_weights()
                self.assertEqual(identities, [id(variable) for variable in models[3].weights])
                for target, value in zip(models[3].get_weights(), expected):
                    np.testing.assert_allclose(target, value, rtol=1e-5, atol=1e-6)

    def test_unconfigured_legacy_helper_does_not_update_targets(self):
        learner, models, _, _ = build_case(compiled=True, fused=False)
        before = models[3].get_weights()
        learner(*fixed_batch())
        for target, old in zip(models[3].get_weights(), before):
            np.testing.assert_array_equal(target, old)
        with self.assertRaisesRegex(ValueError, "require tau"):
            learner(*fixed_batch(), update_targets=True)

    def test_each_update_has_one_packed_host_metric_read(self):
        learner, _, _, _ = build_case(compiled=True, fused=True)
        learner(*fixed_batch(), update_policy=True)
        learner(*fixed_batch(), update_policy=False)
        tensor_class = type(tf.constant(0.0))
        original = tensor_class.numpy
        reads = []

        def read(tensor):
            reads.append(tuple(tensor.shape))
            return original(tensor)

        with mock.patch.object(tensor_class, "numpy", read):
            learner(*fixed_batch(), update_policy=True)
            learner(*fixed_batch(), update_policy=False)
        self.assertEqual(reads, [(5,), (5,)])

    def test_invalid_target_configuration_is_rejected(self):
        for tau in (-0.1, 1.1, np.nan):
            with self.subTest(tau=tau), self.assertRaisesRegex(ValueError, "tau"):
                build_case(compiled=True, fused=True, tau=tau)
        with self.assertRaisesRegex(ValueError, "positive interval"):
            build_case(compiled=True, fused=True, interval=0)


if __name__ == "__main__":
    unittest.main()
