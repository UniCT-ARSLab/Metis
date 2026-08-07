"""Critic-only trainer guards: target-policy smoothing forbidden, FROZEN checksums verified, actor
weight hash stable (the actor must never change)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tools.train_critic_offline import _sha256, _verify_checksums, parse_args  # noqa: E402

REQUIRED = ["--m5v2", "m5.npz", "--counterexample-train", "tr.npz", "--selection", "sel.npz",
            "--final-test", "fin.npz", "--clone-weights", "clone.h5", "--checkpoint-dir", "ck",
            "--frozen-manifest", "FROZEN.json", "--m5v2-sha256", "aa", "--clone-sha256", "bb"]


class TrainerGuardTests(unittest.TestCase):
    def test_nonzero_target_noise_rejected(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED + ["--td3-target-policy-noise", "0.1"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_nonzero_noise_clip_rejected(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED + ["--td3-target-noise-clip", "0.2"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_zero_noise_ok(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED + ["--td3-target-policy-noise", "0.0",
                                                        "--td3-target-noise-clip", "0.0"]):
            args = parse_args()
        self.assertEqual(args.td3_target_policy_noise, 0.0)

    def test_v2_stable_defaults(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED):
            args = parse_args()
        self.assertEqual(args.critic_learning_rate, 3e-5)
        self.assertEqual(args.tau, 0.001)
        self.assertEqual(args.critic_loss, "huber")
        self.assertEqual(args.huber_delta, 5.0)
        self.assertEqual(args.grad_clip_norm, 5.0)
        self.assertEqual(args.audit_every, 250)
        self.assertEqual(args.max_updates, 5000)

    def test_huber_choice_only(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED + ["--critic-loss", "bogus"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_verify_checksums_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            for name in ("train.npz", "selection.npz", "final_test.npz", "report.json"):
                (dp / name).write_bytes(name.encode())
            (dp / "m5.npz").write_bytes(b"m5")
            (dp / "clone.h5").write_bytes(b"clone")
            manifest = {"files": {n: {"sha256": _sha256(str(dp / n))}
                                  for n in ("train.npz", "selection.npz", "final_test.npz", "report.json")}}
            (dp / "FROZEN.json").write_text(json.dumps(manifest))
            from types import SimpleNamespace
            good = SimpleNamespace(
                frozen_manifest=str(dp / "FROZEN.json"), counterexample_train=str(dp / "train.npz"),
                selection=str(dp / "selection.npz"), final_test=str(dp / "final_test.npz"),
                m5v2=str(dp / "m5.npz"), clone_weights=str(dp / "clone.h5"),
                m5v2_sha256=_sha256(str(dp / "m5.npz")), clone_sha256=_sha256(str(dp / "clone.h5")))
            _verify_checksums(good)  # passes
            bad = SimpleNamespace(**vars(good))
            bad.m5v2_sha256 = "deadbeef"
            with self.assertRaises(SystemExit):
                _verify_checksums(bad)

    def test_weights_hash_stable_and_sensitive(self):
        import tensorflow as tf
        from tools.train_critic_offline import _weights_sha256
        m = tf.keras.Sequential([tf.keras.layers.Input((3,)), tf.keras.layers.Dense(2)])
        h1 = _weights_sha256(m)
        self.assertEqual(h1, _weights_sha256(m))                 # stable
        w = m.get_weights(); w[0] = w[0] + 1.0; m.set_weights(w)
        self.assertNotEqual(h1, _weights_sha256(m))              # sensitive to any change

    def test_v3_arg_defaults(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED):
            args = parse_args()
        self.assertIsNone(args.warm_start_checkpoint)            # cold start by default (v1/v2)
        self.assertEqual(args.lambda_pair_final, 0.0)            # pair loss OFF by default
        self.assertEqual(args.lambda_pair_ramp_updates, 500)
        self.assertEqual(args.pair_batch_size, 128)

    def test_v4_arg_defaults(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED):
            args = parse_args()
        self.assertEqual(args.pair_loss_mode, "none")           # v4 sign-hinge is opt-in
        self.assertEqual(args.lambda_sign_final, 0.0)
        self.assertEqual(args.sign_margin, 0.25)

    def test_pair_loss_mode_choice_only(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED + ["--pair-loss-mode", "bogus"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_v5_arg_defaults_and_choices(self):
        with mock.patch("sys.argv", ["t"] + REQUIRED):
            args = parse_args()
        self.assertEqual(args.pair_population, "significant")     # v5 paired_bad is opt-in
        self.assertEqual(args.attack_fraction, 0.5)
        self.assertEqual(args.audit_profile, "full")
        for flag, val in (("--pair-population", "bogus"), ("--audit-profile", "bogus")):
            with mock.patch("sys.argv", ["t"] + REQUIRED + [flag, val]):
                with self.assertRaises(SystemExit):
                    parse_args()


class LambdaRampTests(unittest.TestCase):
    def test_linear_ramp_then_constant(self):
        from tools.train_critic_offline import lambda_pair_at
        self.assertAlmostEqual(lambda_pair_at(0, 0.10, 500), 0.0)
        self.assertAlmostEqual(lambda_pair_at(250, 0.10, 500), 0.05)
        self.assertAlmostEqual(lambda_pair_at(500, 0.10, 500), 0.10)
        self.assertAlmostEqual(lambda_pair_at(5000, 0.10, 500), 0.10)   # constant after ramp

    def test_disabled_returns_final(self):
        from tools.train_critic_offline import lambda_pair_at
        self.assertEqual(lambda_pair_at(100, 0.0, 500), 0.0)            # pair loss disabled
        self.assertEqual(lambda_pair_at(100, 0.10, 0), 0.10)           # no ramp -> immediate final


class PairLossTests(unittest.TestCase):
    OBS, ACT = 6, 3

    def _critic(self):
        from core.models import build_continuous_critic
        return build_continuous_critic(obs_dim=self.OBS, action_size=self.ACT)

    def _huber(self):
        import tensorflow as tf
        return tf.keras.losses.Huber(delta=5.0, reduction=tf.keras.losses.Reduction.NONE)

    def test_pair_loss_is_huber_of_deltaq_vs_returndelta(self):
        import numpy as np
        import tensorflow as tf
        from tools.train_critic_offline import pair_loss_terms
        c = self._critic()
        obs = tf.constant(np.random.RandomState(0).randn(5, self.OBS), tf.float32)
        a_c = tf.constant(np.random.RandomState(1).randn(5, self.ACT), tf.float32)
        a_b = tf.constant(np.random.RandomState(2).randn(5, self.ACT), tf.float32)
        ret = tf.constant([-1.0, -2.0, 1.0, 0.5, -0.5], tf.float32)
        loss, dq = pair_loss_terms(c, obs, a_c, a_b, ret, self._huber())
        qc = tf.reshape(c([obs, a_c], training=False), [-1])
        qb = tf.reshape(c([obs, a_b], training=False), [-1])
        self.assertTrue(np.allclose(dq.numpy(), (qc - qb).numpy(), atol=1e-5))
        expect = tf.reduce_mean(self._huber()(ret, dq)).numpy()
        self.assertAlmostEqual(float(loss.numpy()), float(expect), places=5)

    def test_pair_loss_independent_per_critic(self):
        import numpy as np
        import tensorflow as tf
        from tools.train_critic_offline import pair_loss_terms
        c1, c2 = self._critic(), self._critic()          # different random init
        obs = tf.constant(np.ones((4, self.OBS), np.float32))
        a_c = tf.constant(np.full((4, self.ACT), 0.5, np.float32))
        a_b = tf.constant(np.full((4, self.ACT), -0.5, np.float32))
        ret = tf.constant([-1.0, -1.0, 1.0, 1.0], tf.float32)
        _l1, dq1 = pair_loss_terms(c1, obs, a_c, a_b, ret, self._huber())
        _l2, dq2 = pair_loss_terms(c2, obs, a_c, a_b, ret, self._huber())
        self.assertFalse(np.allclose(dq1.numpy(), dq2.numpy()))   # each critic uses its OWN Q

    def test_pair_loss_has_no_gradient_to_actor(self):
        import numpy as np
        import tensorflow as tf
        from core.models import build_continuous_actor
        from tools.train_critic_offline import pair_loss_terms
        # actor left TRAINABLE so the tape auto-watches its variables; the pair loss still must not
        # produce any gradient toward them (it never references the actor).
        actor = build_continuous_actor(obs_dim=self.OBS, action_size=self.ACT)
        c = self._critic()
        obs = tf.constant(np.ones((4, self.OBS), np.float32))
        a_c = tf.constant(np.full((4, self.ACT), 0.5, np.float32))
        a_b = tf.constant(np.full((4, self.ACT), -0.5, np.float32))
        ret = tf.constant([-1.0, -1.0, 1.0, 1.0], tf.float32)
        with tf.GradientTape() as g:
            loss, _dq = pair_loss_terms(c, obs, a_c, a_b, ret, self._huber())
        grads = g.gradient(loss, actor.trainable_variables)
        self.assertTrue(len(grads) > 0 and all(gr is None for gr in grads))   # actor untouched


class SignHingeTests(unittest.TestCase):
    OBS, ACT = 6, 3

    def _critic(self):
        from core.models import build_continuous_critic
        return build_continuous_critic(obs_dim=self.OBS, action_size=self.ACT)

    def _in(self):
        import numpy as np
        import tensorflow as tf
        obs = tf.constant(np.random.RandomState(0).randn(6, self.OBS), tf.float32)
        a_c = tf.constant(np.random.RandomState(1).randn(6, self.ACT), tf.float32)
        a_b = tf.constant(np.random.RandomState(2).randn(6, self.ACT), tf.float32)
        return obs, a_c, a_b

    def test_formula_matches_relu(self):
        import numpy as np
        import tensorflow as tf
        from tools.train_critic_offline import sign_hinge_terms
        c = self._critic(); obs, a_c, a_b = self._in()
        y = tf.constant([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], tf.float32)
        loss, dq, sm = sign_hinge_terms(c, obs, a_c, a_b, y, 0.25)
        qc = tf.reshape(c([obs, a_c], training=False), [-1])
        qb = tf.reshape(c([obs, a_b], training=False), [-1])
        self.assertTrue(np.allclose(dq.numpy(), (qc - qb).numpy(), atol=1e-5))
        self.assertTrue(np.allclose(sm.numpy(), (y.numpy() * dq.numpy()), atol=1e-5))
        expect = float(tf.reduce_mean(tf.nn.relu(0.25 - y * dq)).numpy())
        self.assertAlmostEqual(float(loss.numpy()), expect, places=5)

    def test_aligned_sign_lower_loss_than_flipped(self):
        import numpy as np
        import tensorflow as tf
        from tools.train_critic_offline import sign_hinge_terms
        c = self._critic(); obs, a_c, a_b = self._in()
        _l, dq, _sm = sign_hinge_terms(c, obs, a_c, a_b, tf.ones(6), 0.25)
        y_aligned = tf.constant(np.sign(dq.numpy()), tf.float32)     # y matches current order
        la, _, _ = sign_hinge_terms(c, obs, a_c, a_b, y_aligned, 0.25)
        lf, _, _ = sign_hinge_terms(c, obs, a_c, a_b, -1.0 * y_aligned, 0.25)
        self.assertLess(float(la.numpy()), float(lf.numpy()))        # penalises wrong orders more

    def test_independent_per_critic(self):
        import numpy as np
        import tensorflow as tf
        from tools.train_critic_offline import sign_hinge_terms
        c1, c2 = self._critic(), self._critic()
        obs, a_c, a_b = self._in()
        y = tf.ones(6)
        _l1, dq1, _s1 = sign_hinge_terms(c1, obs, a_c, a_b, y, 0.25)
        _l2, dq2, _s2 = sign_hinge_terms(c2, obs, a_c, a_b, y, 0.25)
        self.assertFalse(np.allclose(dq1.numpy(), dq2.numpy()))


class RestoreRoundtripTests(unittest.TestCase):
    """The v3 warm-start restores critic1/critic2/target1/target2 EXACTLY as saved."""

    def test_exact_restore_of_all_four(self):
        import numpy as np
        import tempfile
        import tensorflow as tf
        from core.models import build_continuous_critic
        obs_dim, act = 6, 3
        crit = lambda: build_continuous_critic(obs_dim=obs_dim, action_size=act)
        c1, c2, t1, t2 = crit(), crit(), crit(), crit()
        # perturb so all four differ from a fresh init
        for m in (c1, c2, t1, t2):
            m.set_weights([w + np.random.RandomState(7).randn(*w.shape).astype(w.dtype) * 0.1
                           for w in m.get_weights()])
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "critic-4250")
            saved = tf.train.Checkpoint(critic=c1, critic2=c2, target_critic=t1,
                                        target_critic2=t2).save(prefix)   # -> prefix-1
            # .save() APPENDS a numeric suffix; the returned path (NOT the bare prefix) is what
            # restore() needs. This is the exact bug that crashed the v6 final-test.
            self.assertTrue(saved.endswith("-1"))
            self.assertNotEqual(saved, prefix)
            r1, r2, rt1, rt2 = crit(), crit(), crit(), crit()
            tf.train.Checkpoint(critic=r1, critic2=r2, target_critic=rt1,
                                target_critic2=rt2).restore(saved).expect_partial()
            for orig, restored in ((c1, r1), (c2, r2), (t1, rt1), (t2, rt2)):
                for wo, wr in zip(orig.get_weights(), restored.get_weights()):
                    self.assertTrue(np.array_equal(wo, wr))


if __name__ == "__main__":
    unittest.main()
