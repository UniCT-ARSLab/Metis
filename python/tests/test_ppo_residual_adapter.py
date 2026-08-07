"""Equivalence + correctness tests for the PPO action-adapter refactor.

Verifies (a) the StandardActionAdapter reproduces the pre-adapter continuous path (clip), so
`--policy-mode standard` is bit-identical; (b) the masked PPO policy loss with an all-ones mask equals
the plain mean (identity), and a partial mask changes it; (c) the ResidualActionAdapter maps a raw
sample to base+gated-bounded-residual, stores the raw latent, zeroes the residual out of the gate, and
keeps the frozen base OUT of the trainable variables; (d) a real hybrid model runs
build_update_batch->ppo_update through the adapter path (standard) end-to-end with finite losses.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.policy_action_adapter import (  # noqa: E402
    ResidualActionAdapter, StandardActionAdapter, load_gate_config)


class StandardAdapterTests(unittest.TestCase):
    def test_transform_is_clip_identity(self):
        sa = StandardActionAdapter([-1, -1, -1], [1, 1, 1])
        env, stored, diag = sa.transform(None, np.array([2.0, -3.0, 0.5], np.float32))
        self.assertTrue(np.allclose(env, [1.0, -1.0, 0.5]))     # clipped to bounds
        self.assertTrue(np.allclose(env, stored))               # standard stores what it sends
        self.assertEqual(sa.train_mask(None), 1.0)
        self.assertEqual(sa.trainable_extra(), [])


class MaskedLossIdentityTests(unittest.TestCase):
    def test_all_ones_mask_equals_mean(self):
        surrogate = np.array([0.3, -0.2, 0.5, -0.1], np.float64)
        mask = np.ones(4)
        masked = -np.sum(mask * surrogate) / (np.sum(mask) + 1e-8)
        self.assertAlmostEqual(masked, -np.mean(surrogate), places=6)   # identity at all-ones

    def test_partial_mask_changes_loss(self):
        surrogate = np.array([0.3, -0.2, 0.5, -0.1], np.float64)
        mask = np.array([1.0, 0.0, 1.0, 0.0])
        masked = -np.sum(mask * surrogate) / (np.sum(mask) + 1e-8)
        self.assertAlmostEqual(masked, -np.mean([0.3, 0.5]), places=6)  # only unmasked states count


class ResidualAdapterTests(unittest.TestCase):
    def _adapter(self, update_mask="gate"):
        base = np.array([0.10, -0.20, 0.30], np.float32)
        return ResidualActionAdapter(lambda obs: base, low=[-1, -1, -1], high=[1, 1, 1],
                                     err_start=0, err_size=2, gate_outer=0.11, gate_inner=0.04,
                                     delta_max=0.005, update_mask=update_mask), base

    def test_gate_open_scales_residual_and_stores_raw(self):
        ad, base = self._adapter()
        obs = np.zeros(4, np.float32)                           # ||obs[0:2]|| = 0 -> gate fully open (1.0)
        raw = np.array([0.5, -1.0, 2.0], np.float32)
        env, stored, diag = ad.transform(obs, raw)
        self.assertTrue(np.allclose(stored, raw))               # the LATENT raw is stored (for log-prob)
        expect = np.clip(base + 1.0 * 0.005 * np.tanh(raw), -1, 1)
        self.assertTrue(np.allclose(env, expect, atol=1e-6))
        self.assertLessEqual(diag["delta_absmax"], 0.005 + 1e-9)
        self.assertEqual(ad.train_mask(obs), 1.0)

    def test_gate_closed_is_base_and_masked(self):
        ad, base = self._adapter()
        obs = np.zeros(4, np.float32); obs[0] = 0.5             # ||err|| 0.5 >> outer 0.11 -> gate 0
        env, stored, diag = ad.transform(obs, np.array([9.0, -9.0, 9.0], np.float32))
        self.assertTrue(np.allclose(env, base))                # residual EXACTLY 0 in approach
        self.assertEqual(diag["approach_dev"], 0.0)
        self.assertEqual(ad.train_mask(obs), 0.0)              # no policy update out of gate

    def test_update_mask_all_trains_everywhere(self):
        ad, _ = self._adapter(update_mask="all")
        obs = np.zeros(4, np.float32); obs[0] = 0.5
        self.assertEqual(ad.train_mask(obs), 1.0)

    def test_base_is_frozen_and_not_extra_trainable(self):
        ad, _ = self._adapter()
        self.assertEqual(ad.trainable_extra(), [])             # nothing beyond the PPO model + log_std


class GateConfigTests(unittest.TestCase):
    def test_inline_spec(self):
        gc = load_gate_config("0.11,0.04,14,3")
        self.assertEqual((gc["outer"], gc["inner"], gc["err_start"], gc["err_size"]), (0.11, 0.04, 14, 3))


class WarmStartZeroInitTests(unittest.TestCase):
    def test_fresh_only_zero_inits(self):
        # zero-init ONLY for a truly fresh policy — NOT warm-start (--policy-path) nor resume. The resume
        # signal is an EXPLICIT boolean (is_resuming), never derived from start_episode: a valid resume
        # from ckpt-0 has start_episode==0 yet must keep its restored residual weights.
        import algorithms.ppo as ppo
        self.assertTrue(ppo.residual_should_zero_init(False, None))              # fresh
        self.assertFalse(ppo.residual_should_zero_init(False, "trained.keras"))  # warm-start -> keep it
        self.assertFalse(ppo.residual_should_zero_init(True, None))              # resume -> keep it
        self.assertFalse(ppo.residual_should_zero_init(True, None))              # resume from ckpt-0 (start_episode==0) -> STILL keep it


class SeparateValueOptimizerTests(unittest.TestCase):
    def test_value_optimizer_updates_only_value_tower(self):
        # Correction #1: a separate value optimizer steps the value tower; the actor optimizer steps the
        # actor. The two towers are disjoint, so with an all-inactive batch (policy loss 0) only the value
        # optimizer's iterations advance and only value vars move.
        import tensorflow as tf
        import algorithms.ppo as ppo
        from core.models import build_hybrid_actor_critic
        m = build_hybrid_actor_critic(6, [], 3, continuous_activation="linear", separate_value_tower=True)
        m(np.zeros((1, 6), np.float32))
        log_std = tf.Variable(np.full(3, -1.5, np.float32), trainable=True)
        av, vv = ppo._split_actor_value_vars(m, log_std, 3)
        self.assertEqual(len(vv), 8)                                        # value tower (3 dense + head) x (k,b)
        self.assertTrue(all("value" in getattr(v, "path", v.name) for v in vv))
        self.assertTrue(all("value" not in getattr(v, "path", v.name) for v in av if v is not log_std))
        opt = tf.keras.optimizers.Adam(1e-5); vopt = tf.keras.optimizers.Adam(3e-4)
        am = {"discrete": [], "discrete_sizes": [], "continuous_size": 3, "continuous": [{"slice": slice(0, 3)}],
              "single_discrete": False, "single_continuous": True,
              "continuous_low": np.full(3, -1, np.float32), "continuous_high": np.full(3, 1, np.float32)}
        n = 16
        batch = {"obs": np.random.rand(n, 6).astype(np.float32), "discrete_actions": np.zeros((n, 0), np.int32),
                 "continuous_actions": np.random.rand(n, 3).astype(np.float32), "log_probs": -np.random.rand(n).astype(np.float32),
                 "returns": (np.random.rand(n) * 5).astype(np.float32), "advantages": np.random.rand(n).astype(np.float32),
                 "train_masks": np.zeros(n, np.float32)}

        class A:
            clip_ratio = 0.1; value_loss_coef = 0.5; entropy_coef = 0.002; ppo_epochs = 2; batch_size = 16
            ppo_log_std_min = -5.0; ppo_log_std_max = -0.5; ppo_grad_clip = 1.0; ppo_target_kl = 0.0
            tf_compile_learner = False; tf_xla = False
        probe = np.random.rand(4, 6).astype(np.float32)
        mean0 = m(probe)[-2].numpy().copy(); val0 = m(probe)[-1].numpy().copy()
        ppo.ppo_update(m, log_std, opt, batch, am, A(), vopt)
        self.assertGreater(int(vopt.iterations.numpy()), 0)                 # value optimizer stepped the value tower
        self.assertTrue(np.array_equal(mean0, m(probe)[-2].numpy()))        # actor UNCHANGED (mask all 0 -> zero actor grad)
        self.assertFalse(np.allclose(val0, m(probe)[-1].numpy()))          # value tower LEARNED via its own optimizer


class PPOThroughAdapterTests(unittest.TestCase):
    def test_standard_update_runs_and_base_excluded(self):
        import tensorflow as tf
        import algorithms.ppo as ppo
        from core.models import build_hybrid_actor_critic
        obs_dim, cont = 6, 3
        model = build_hybrid_actor_critic(obs_dim, [], cont)
        model(np.zeros((1, obs_dim), np.float32))
        log_std = tf.Variable(np.full(cont, -0.5, np.float32), trainable=True)
        train_vars = model.trainable_variables + [log_std]
        # a separate "base" model must NOT be in train_vars
        base = build_hybrid_actor_critic(obs_dim, [], cont)
        self.assertTrue(all(id(v) != id(bv) for v in train_vars for bv in base.trainable_variables))

        action_meta = {"discrete": [], "discrete_sizes": [], "continuous_size": cont,
                       "continuous": [{"slice": slice(0, cont)}], "single_discrete": False, "single_continuous": True,
                       "continuous_low": np.full(cont, -1, np.float32), "continuous_high": np.full(cont, 1, np.float32)}
        # build one trajectory through append_transition (adapter=None -> standard), then GAE+update
        traj = ppo.new_trajectory()
        sample_fn = ppo.build_sample_action_fn(model, obs_dim, action_meta)
        for _ in range(8):
            sel = ppo.select_action(sample_fn, log_std.numpy(), np.random.rand(obs_dim).astype(np.float32), action_meta)
            self.assertIn("train_mask", sel)
            ppo.append_transition(traj, np.random.rand(obs_dim).astype(np.float32), sel, 1.0, False)
        self.assertEqual(len(traj["train_masks"]), 8)
        batch = ppo.build_update_batch([traj], action_meta, 0.99, 0.95)
        self.assertIn("train_masks", batch)
        self.assertTrue(np.allclose(batch["train_masks"], 1.0))  # standard -> all trainable

        class A:  # minimal args for ppo_update
            clip_ratio = 0.2; value_loss_coef = 0.5; entropy_coef = 0.01; ppo_epochs = 1
            batch_size = 8; ppo_log_std_min = -20.0; ppo_log_std_max = 2.0
            tf_compile_learner = False; tf_xla = False
        opt = tf.keras.optimizers.Adam(1e-3)
        metrics = ppo.ppo_update(model, log_std, opt, batch, action_meta, A())
        self.assertTrue(np.isfinite(metrics["loss"]) and np.isfinite(metrics["policy_loss"]))


class ResidualModelTests(unittest.TestCase):
    def _residual_model(self, obs_dim=6, cont=3):
        import tensorflow as tf
        from core.models import build_hybrid_actor_critic
        m = build_hybrid_actor_critic(obs_dim, [], cont, continuous_activation="linear", separate_value_tower=True)
        m(np.zeros((1, obs_dim), np.float32))
        ml = m.get_layer("continuous_mean"); ml.set_weights([np.zeros_like(w) for w in ml.get_weights()])
        log_std = tf.Variable(np.full(cont, -1.5, np.float32), trainable=True)
        return m, log_std

    def test_linear_head_reaches_full_delta_max(self):
        # A tanh head would cap the deterministic residual at tanh(1)*delta_max ~= 0.762*delta_max; a
        # LINEAR head lets a large mean drive tanh(mean)->+/-1 so the residual can reach +/-delta_max.
        import tensorflow as tf
        m, _ = self._residual_model()
        mean_layer = m.get_layer("continuous_mean")
        big = [np.full_like(w, 5.0) if w.ndim == 1 else np.full_like(w, 5.0) for w in mean_layer.get_weights()]
        mean_layer.set_weights(big)
        mean = m(np.ones((1, 6), np.float32))[-2].numpy()[0]
        self.assertGreater(float(np.max(np.abs(np.tanh(mean)))), 0.99)   # linear head -> tanh(mean)->~1

    def test_value_loss_does_not_touch_actor_or_log_std(self):
        # Correction #10: with a SEPARATE value tower, an all-inactive (train_mask 0) batch updates the
        # value but leaves the actor output + log_std BIT-IDENTICAL.
        import tensorflow as tf
        import algorithms.ppo as ppo
        m, log_std = self._residual_model()
        opt = tf.keras.optimizers.Adam(1e-2)
        am = {"discrete": [], "discrete_sizes": [], "continuous_size": 3, "continuous": [{"slice": slice(0, 3)}],
              "single_discrete": False, "single_continuous": True,
              "continuous_low": np.full(3, -1, np.float32), "continuous_high": np.full(3, 1, np.float32)}
        probe = np.random.rand(5, 6).astype(np.float32)
        mean0 = m(probe)[-2].numpy().copy(); ls0 = log_std.numpy().copy(); val0 = m(probe)[-1].numpy().copy()
        n = 16
        batch = {"obs": np.random.rand(n, 6).astype(np.float32), "discrete_actions": np.zeros((n, 0), np.int32),
                 "continuous_actions": np.random.rand(n, 3).astype(np.float32), "log_probs": -np.random.rand(n).astype(np.float32),
                 "returns": (np.random.rand(n) * 5).astype(np.float32), "advantages": np.random.rand(n).astype(np.float32),
                 "train_masks": np.zeros(n, np.float32)}   # ENTIRELY out-of-gate

        class A:
            clip_ratio = 0.1; value_loss_coef = 0.5; entropy_coef = 0.002; ppo_epochs = 3; batch_size = 16
            ppo_log_std_min = -5.0; ppo_log_std_max = -0.5; ppo_grad_clip = 1.0; ppo_target_kl = 0.0
            tf_compile_learner = False; tf_xla = False
        ppo.ppo_update(m, log_std, opt, batch, am, A())
        self.assertTrue(np.array_equal(mean0, m(probe)[-2].numpy()))     # actor UNCHANGED
        self.assertTrue(np.array_equal(ls0, log_std.numpy()))           # log_std UNCHANGED
        self.assertFalse(np.allclose(val0, m(probe)[-1].numpy()))       # value LEARNED


if __name__ == "__main__":
    unittest.main()
