"""Regression tests for the M7.1 online critic-warmup trainer's pure logic (no Godot). Covers the
seven properties the design requires: one-action-then-composite gate semantics, the real correction
gate, disjoint decks/holdout, deterministic audit, missing-cell failure, persistence round-trip, and
the single-use fail-closed final-test."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_m7_critic_warmup import (  # noqa: E402
    GAMMA, HOLDOUT_SUB, ROUND_SUB, SEED_STRIDE, audit_pairs, check_final_test_single_use,
    compute_audit_metrics, correction_basis, effective_counts, final_test_marker,
    full_run_exit_code, gate_scaled_correction, load_critic_checkpoint, save_critic_checkpoint,
    _deck, _disc,
)

CELLS = [17, 25, 28]
HARD = (25, 28)


def _anchor(cell, obs, base_a, perts):
    return {"seed": 1, "k": 5, "band": 0, "dist": 0.025, "obs_k": list(obs), "base_a": list(base_a), "perts": perts}


def _pert(label, pert_action, dret_oneshot, dret_sustained, at_bound):
    return {"label": label, "at_bound": at_bound, "gate": 1.0, "pert_action": list(pert_action),
            "dret_oneshot": dret_oneshot, "dret_sustained": dret_sustained}


def _manifest(anchors_by_cell):
    return {"tag": "t", "seed_base": 0, "anchors": anchors_by_cell}


def _fake_qmin(obs, act):
    # deterministic critic surrogate: Q = sum(action) -> dq tracks the action perturbation
    return np.array([float(np.sum(np.asarray(a))) for a in act])


class GateSemanticsTests(unittest.TestCase):
    def test_audit_uses_one_action_return_not_sustained(self):
        # dret_oneshot=+5 but dret_sustained=-99: the GATE 'dret' must be the one-action value.
        man = _manifest({25: [_anchor(25, np.zeros(7), np.zeros(7),
                                      [_pert("j0+", [0.9] + [0] * 6, +5.0, -99.0, True)])]})
        pairs = audit_pairs(man, _fake_qmin)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["dret"], 5.0)              # one-action-then-composite
        self.assertEqual(pairs[0]["dret_sustained"], -99.0)  # sustained carried only as diagnostic

    def test_correction_gate_scales_and_zeroes(self):
        base = np.zeros(7); unit = np.eye(7)[3]
        e, pa = gate_scaled_correction(base, unit, gate_val=0.5, delta_max=0.005)
        self.assertAlmostEqual(e[3], 0.5 * 0.005)            # gate(distance) * delta_max * unit
        self.assertAlmostEqual(pa[3], 0.5 * 0.005)
        e0, pa0 = gate_scaled_correction(base, unit, gate_val=0.0, delta_max=0.005)
        self.assertTrue(np.allclose(e0, 0.0))                # gate 0 (approach / >outer) -> no correction
        self.assertTrue(np.allclose(pa0, base))

    def test_basis_at_bound_flags(self):
        b = correction_basis(7, 3, np.random.default_rng(0))
        self.assertEqual(len(b), 7 * 2 + 3)
        per_joint = [x for x in b if x[2]]
        gauss = [x for x in b if not x[2]]
        self.assertEqual(len(per_joint), 14)                 # per-joint +/- saturate the residual bound
        self.assertEqual(len(gauss), 3)                      # gaussian dirs are NOT at bound


class DeckDisjointTests(unittest.TestCase):
    def test_train_selection_final_holdout_disjoint(self):
        bases = {"train": 100_000_000, "sel": 200_000_000, "fin": 300_000_000}
        seen = {}
        for name, base in bases.items():
            s = set()
            for ci in range(11):
                for r in range(6):                            # training rounds
                    s |= {_deck(base, ci, r * ROUND_SUB, i) for i in range(200)}
                s |= {_deck(base, ci, HOLDOUT_SUB, i) for i in range(200)}   # holdout sub-range
            seen[name] = s
        self.assertEqual(seen["train"] & seen["sel"], set())
        self.assertEqual(seen["train"] & seen["fin"], set())
        self.assertEqual(seen["sel"] & seen["fin"], set())

    def test_round_and_holdout_subranges_disjoint_within_split(self):
        base, ci = 200_000_000, 3
        rounds = {_deck(base, ci, r * ROUND_SUB, i) for r in range(6) for i in range(1000)}
        hold = {_deck(base, ci, HOLDOUT_SUB, i) for i in range(1000)}
        self.assertEqual(rounds & hold, set())
        self.assertLess(6 * ROUND_SUB + 1000, SEED_STRIDE)   # rounds fit before the cell stride
        self.assertLess(HOLDOUT_SUB + 1000, SEED_STRIDE)     # holdout fits too


class AuditPropertyTests(unittest.TestCase):
    def _two_cell_manifest(self):
        return _manifest({
            25: [_anchor(25, np.zeros(7), np.zeros(7), [_pert("j0+", [0.5] + [0] * 6, +1.0, +1.0, True),
                                                        _pert("j1-", [-0.5] + [0] * 6, -1.0, -1.0, True)])],
            28: [_anchor(28, np.ones(7), np.zeros(7), [_pert("g0", [0.3] * 7, +2.0, +2.0, False)])],
        })

    def test_audit_deterministic(self):
        man = self._two_cell_manifest()
        p1 = audit_pairs(man, _fake_qmin)
        p2 = audit_pairs(man, _fake_qmin)
        self.assertEqual(p1, p2)                              # same manifest + same critic -> identical

    def test_missing_cell_reported_not_skipped(self):
        # manifest only has cells 25,28 but declared cells include 17 -> cell 17 must appear with n=0.
        man = self._two_cell_manifest()
        m = compute_audit_metrics(audit_pairs(man, _fake_qmin), CELLS, HARD)
        self.assertIn(17, m["per_cell"])
        self.assertEqual(m["per_cell"][17]["n"], 0)          # missing -> zero pairs (its gate will fail)

    def test_saturated_bad_pref_nan_when_no_samples(self):
        # all at-bound perturbations HELP return (dret>0) -> no bad-at-bound samples -> NaN, not 0/pass.
        man = _manifest({25: [_anchor(25, np.zeros(7), np.zeros(7),
                                      [_pert("j0+", [0.5] + [0] * 6, +1.0, +1.0, True)])]})
        m = compute_audit_metrics(audit_pairs(man, _fake_qmin), [25], (25,))
        self.assertEqual(m["n_saturated_bad"], 0)
        self.assertTrue(np.isnan(m["saturated_bad_pref"]))   # zero samples is NOT an automatic pass


class PersistenceTests(unittest.TestCase):
    def test_replay_npz_roundtrip(self):
        buf = {"obs": np.random.rand(8, 27).astype(np.float32), "act": np.random.rand(8, 7).astype(np.float32),
               "rew": np.random.rand(8).astype(np.float32), "next": np.random.rand(8, 27).astype(np.float32),
               "done": np.zeros(8, np.float32), "cell": np.arange(8, dtype=np.int32)}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "replay.npz"
            np.savez_compressed(p, **buf)
            back = np.load(p)
            for k in buf:
                self.assertTrue(np.allclose(back[k], buf[k]))


class FinalTestSingleUseTests(unittest.TestCase):
    def test_fail_closed_after_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            allowed, _ = check_final_test_single_use(d, dry_run=False)
            self.assertTrue(allowed)                          # first time: allowed
            final_test_marker(d).write_text("{}")             # consume it
            allowed2, why = check_final_test_single_use(d, dry_run=False)
            self.assertFalse(allowed2)                        # restart: refused
            self.assertIn("already consumed", why)
            allowed3, _ = check_final_test_single_use(d, dry_run=True)
            self.assertTrue(allowed3)                         # dry-run is exempt

    def test_started_marker_blocks_reuse(self):
        # crash-safe: a marker written with status 'started' (BEFORE the audit) already blocks reuse.
        with tempfile.TemporaryDirectory() as d:
            final_test_marker(d).write_text('{"status": "started"}')
            allowed, _ = check_final_test_single_use(d, dry_run=False)
            self.assertFalse(allowed)                         # a mid-final crash cannot re-consume


class DiscountedReturnTests(unittest.TestCase):
    def test_disc_matches_gamma(self):
        self.assertAlmostEqual(_disc([1.0, 1.0, 1.0]), 1 + GAMMA + GAMMA ** 2)
        self.assertAlmostEqual(_disc([0.0, 0.0, 5.0]), GAMMA ** 2 * 5.0)  # late reward discounted
        self.assertNotAlmostEqual(_disc([1.0] * 50), 50.0)     # discounted != undiscounted sum
        self.assertEqual(_disc([]), 0.0)


class CountsTests(unittest.TestCase):
    def test_effective_counts_rounds_down_and_declared(self):
        # 480/6/4: per-round 80, quota 80//12=6, effective 6*12*6=432 (NOT 480).
        self.assertEqual(effective_counts(480, 6, 4), (6, 432))
        # 576/6/4: per-round 96, quota 8, effective 576 (clean default).
        self.assertEqual(effective_counts(576, 6, 4), (8, 576))


class ExitCodeTests(unittest.TestCase):
    def test_full_run_exit_codes(self):
        self.assertEqual(full_run_exit_code(dry_run=False, m7_1_pass=True, hashes_ok=True), 0)
        self.assertEqual(full_run_exit_code(dry_run=False, m7_1_pass=False, hashes_ok=True), 1)  # no PASS -> non-zero
        self.assertEqual(full_run_exit_code(dry_run=True, m7_1_pass=False, hashes_ok=True), 0)   # dry-run exempt
        self.assertEqual(full_run_exit_code(dry_run=False, m7_1_pass=True, hashes_ok=False), 4)  # hash mutated


class CheckpointReloadTests(unittest.TestCase):
    def test_critic_optimizer_roundtrip(self):
        import tensorflow as tf
        from tools.train_m7_critic_warmup import build_continuous_critic
        obs_dim, act = 6, 3
        c1 = build_continuous_critic(obs_dim, act); c2 = build_continuous_critic(obs_dim, act)
        t1 = build_continuous_critic(obs_dim, act); t2 = build_continuous_critic(obs_dim, act)
        opt1 = tf.keras.optimizers.Adam(1e-3); opt2 = tf.keras.optimizers.Adam(1e-3)
        o = np.random.rand(4, obs_dim).astype(np.float32); a = np.random.rand(4, act).astype(np.float32)
        for c, opt in ((c1, opt1), (c2, opt2)):                # a few real steps so weights + slots are non-trivial
            for _ in range(3):
                with tf.GradientTape() as g:
                    loss = tf.reduce_mean((c([o, a]) - 1.0) ** 2)
                opt.apply_gradients(zip(g.gradient(loss, c.trainable_variables), c.trainable_variables))
        probe_in = [tf.convert_to_tensor(o), tf.convert_to_tensor(a)]
        q1_before = c1(probe_in).numpy()
        with tempfile.TemporaryDirectory() as d:
            save_critic_checkpoint(d, c1, c2, t1, t2, opt1, opt2, {"total_updates": 3})
            r = load_critic_checkpoint(d, obs_dim, act, lr=1e-3)
            self.assertTrue(np.allclose(r["c1"](probe_in).numpy(), q1_before, atol=1e-6))  # critic restored
            self.assertEqual(len(r["opt1"].variables), len(opt1.variables))                # optimizer rebuilt
            self.assertTrue(np.allclose(r["opt1"].variables[1].numpy(), opt1.variables[1].numpy(), atol=1e-6))
            self.assertEqual(r["state"]["total_updates"], 3)


if __name__ == "__main__":
    unittest.main()
