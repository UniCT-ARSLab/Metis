"""Counterexample selection audit v2: sign-agreement over ALL significant candidates (|delta|>=
margin), scale-invariant td_nrmse, Q-scale cap, target-critic Bellman, calibration, thresholds."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.counterexample_audit import audit_passes, counterexample_audit  # noqa: E402

OBS, ACT = 4, 2


def _split():
    obs = np.zeros((6, OBS), np.float32)
    src = np.array(["baseline", "baseline", "baseline", "candidate", "candidate", "candidate"])
    return {
        "source": src,
        "label_usable": np.array([True] * 6),
        "paired_bad": np.array([False, False, False, True, True, True]),
        "pose_id": np.array(["p0", "p1", "p2", "p0", "p0", "p0"]),
        "phase": np.array(["mid"] * 6),
        "family": np.array(["baseline", "baseline", "baseline", "pga", "collapsed_v4", "random"]),
        "cell": np.array([17] * 6, np.int32),
        "outcome": np.array(["success", "success", "success",
                             "env_time_limit", "env_time_limit", "terminal_failure"]),
        "obs": obs, "next_obs": obs.copy(),
        "actions": np.array([[0, 0], [0.1, 0], [0.2, 0], [0.6, 0.6], [1, 1], [1, -1]], np.float32),
        "rewards": np.zeros(6, np.float32), "dones": np.ones(6, np.float32),
        "return_delta_episode": np.array([0.0, 0.0, 0.0, -1.0, -2.0, -2.5], np.float32),
        "G_baseline_episode": np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0], np.float32),
        "G_candidate_episode": np.array([3.0, 2.0, 1.0, -1.0, -2.0, -2.5], np.float32),
    }


def _good(obs, act):
    return -np.sum(np.asarray(act, np.float64) ** 2, axis=1)   # peaks at midpoint


def _m(split=None, critic=_good, margin=0.5):
    return counterexample_audit(split or _split(), critic, target_critic_min_q=critic,
                                bc_next_action=lambda n: np.zeros((len(n), ACT), np.float32),
                                gamma=0.99, margin=margin, q1=critic, q2=critic)


def _good_metrics(**over):
    m = dict(ranking_accuracy=1.0, worst_cell_accuracy=1.0, worst_cell_margin=1.0,
             sign_agreement_pga_collapsed=1.0, negative_sign_agreement=1.0,
             spearman_deltaQ_returndelta=0.9,
             delta_q_slope=1.0, baseline_calibration_spearman=0.9, clone_q_mean=0.0,
             td_error=0.1, td_nrmse=0.2, td_nrmse_nonterminal=0.2, q_p99=1.0, g_p99=3.0, finite=True,
             n_terminal=0, terminal_bad_count=0, terminal_bad_ranking_accuracy=float("nan"),
             terminal_worst_q_margin=float("nan"))
    m.update(over)
    return m


class RankingMetricTests(unittest.TestCase):
    def test_good_critic_ranking(self):
        m = _m()
        self.assertEqual(m["ranking_accuracy"], 1.0)
        self.assertEqual(m["sign_agreement_pga_collapsed"], 1.0)
        self.assertGreater(m["delta_q_slope"], 0.0)
        self.assertGreater(m["baseline_calibration_spearman"], 0.0)

    def test_sign_population_all_significant_not_only_paired_bad(self):
        s = _split(); s["paired_bad"] = np.array([False, False, False, False, True, True])
        self.assertEqual(_m(s)["n_sign_population"], 2)   # pga still qualifies (|delta|=1>=margin)

    def test_excludes_small_abs_return_delta(self):
        s = _split(); s["return_delta_episode"] = np.array([0, 0, 0, -0.2, -2.0, -2.5], np.float32)
        self.assertEqual(_m(s, margin=0.5)["n_sign_population"], 1)   # pga (0.2) excluded


def _td_split(reward, cval):
    # 2 non-terminal candidates; target critic returns 0 -> tgt = reward; the critic returns cval.
    obs = np.zeros((3, OBS), np.float32)
    return {
        "source": np.array(["baseline", "candidate", "candidate"]),
        "label_usable": np.array([True, True, True]),
        "paired_bad": np.array([False, True, True]),
        "pose_id": np.array(["p", "p", "p"]), "phase": np.array(["mid"] * 3),
        "family": np.array(["baseline", "pga", "collapsed_v4"]), "cell": np.array([17] * 3, np.int32),
        "outcome": np.array(["success", "env_time_limit", "env_time_limit"]),
        "obs": obs, "next_obs": obs.copy(),
        "actions": np.zeros((3, ACT), np.float32),
        "rewards": np.full(3, reward, np.float32), "dones": np.zeros(3, np.float32),
        "return_delta_episode": np.array([0.0, -1.0, -1.0], np.float32),
        "G_baseline_episode": np.array([reward, reward, reward], np.float32),
        "G_candidate_episode": np.array([reward, reward, reward], np.float32),
    }


class TdNrmseTests(unittest.TestCase):
    def test_perfect_fit_zero_nrmse(self):
        # critic returns reward, target returns 0 -> tgt=reward, resid=0 -> nrmse ~ 0
        r = 2.0
        m = counterexample_audit(_td_split(r, r), lambda o, a: np.full(len(o), r),
                                 target_critic_min_q=lambda o, a: np.zeros(len(o)),
                                 bc_next_action=lambda n: np.zeros((len(n), ACT), np.float32), gamma=0.99)
        self.assertLess(m["td_nrmse"], 1e-3)

    def test_scale_invariant(self):
        def nrmse(scale):
            r, c = 1.0 * scale, 2.0 * scale
            m = counterexample_audit(_td_split(r, c), lambda o, a: np.full(len(o), c),
                                     target_critic_min_q=lambda o, a: np.zeros(len(o)),
                                     bc_next_action=lambda n: np.zeros((len(n), ACT), np.float32), gamma=0.99)
            return m["td_nrmse"]
        self.assertAlmostEqual(nrmse(1.0), nrmse(1000.0), places=5)


class GateTests(unittest.TestCase):
    def test_passes_when_all_good(self):
        passed, reasons = audit_passes(_good_metrics())
        self.assertTrue(passed, reasons)

    def test_high_td_nrmse_rejected_even_with_perfect_ranking(self):
        passed, reasons = audit_passes(_good_metrics(td_nrmse=0.9))
        self.assertFalse(passed)
        self.assertTrue(any("td_nrmse" in r for r in reasons))

    def test_q_scale_cap_rejects_inflated_q(self):
        passed, reasons = audit_passes(_good_metrics(q_p99=1e6, g_p99=3.0))
        self.assertFalse(passed)
        self.assertTrue(any("q_p99" in r for r in reasons))

    def test_low_ranking_rejected(self):
        passed, reasons = audit_passes(_good_metrics(ranking_accuracy=0.85))
        self.assertFalse(passed)
        self.assertTrue(any("ranking" in r for r in reasons))

    def test_negative_slope_rejected(self):
        passed, _ = audit_passes(_good_metrics(delta_q_slope=-0.1))
        self.assertFalse(passed)

    def test_train_sign_gate_enforced_only_when_present(self):
        # absent key -> gate skipped (v1/v2/v3)
        self.assertTrue(audit_passes(_good_metrics())[0])
        # present + low -> rejected with a train_sign reason (v4 sign-hinge)
        passed, reasons = audit_passes(_good_metrics(train_sign_accuracy=0.85))
        self.assertFalse(passed)
        self.assertTrue(any("train_sign" in r for r in reasons))
        # present + high -> passes
        self.assertTrue(audit_passes(_good_metrics(train_sign_accuracy=0.95))[0])


class SignSplitTests(unittest.TestCase):
    def test_split_keys_present_and_dict(self):
        m = _m()
        for k in ("sign_pos", "sign_neg", "sign_pga", "sign_collapsed", "sign_by_cellphase"):
            self.assertIn(k, m)
        self.assertIsInstance(m["sign_by_cellphase"], dict)
        # the good critic ranks correctly -> negative-sign candidates agree
        self.assertEqual(m["sign_neg"], 1.0)


class NegativeSignMetricTests(unittest.TestCase):
    def test_safety_metrics_present_and_correct(self):
        m = _m()
        # _split has pga(-1) + collapsed_v4(-2) as the pga/collapsed significant NEGATIVES
        self.assertEqual(m["n_neg_sign"], 2)
        self.assertEqual(m["n_candidate_better"], 0)
        self.assertEqual(m["negative_sign_agreement"], 1.0)          # good critic ranks both below
        self.assertTrue(np.isnan(m["candidate_better_accuracy"]))    # no positives here
        lo, hi = m["negative_sign_ci"]                               # Wilson CI on 2/2
        self.assertTrue(0.0 <= lo <= hi <= 1.0)


class SafetyProfileTests(unittest.TestCase):
    def test_safety_gates_on_negative_sign_not_inclusive_sign(self):
        # inclusive sign LOW but negative sign HIGH -> safety PASSES (candidate_better is not gated)
        m = _good_metrics(sign_agreement_pga_collapsed=0.4, negative_sign_agreement=0.95)
        self.assertTrue(audit_passes(m, profile="safety")[0])
        self.assertFalse(audit_passes(m, profile="full")[0])         # full still fails on inclusive

    def test_safety_rejects_low_negative_sign(self):
        passed, reasons = audit_passes(_good_metrics(negative_sign_agreement=0.85), profile="safety")
        self.assertFalse(passed)
        self.assertTrue(any("negative_sign_agreement" in r for r in reasons))

    def test_safety_ignores_slope_and_calibration(self):
        # slope<=0 and calibration<=0 would FAIL "full" but are NOT gated under "safety"
        m = _good_metrics(delta_q_slope=-1.0, baseline_calibration_spearman=-0.5)
        self.assertTrue(audit_passes(m, profile="safety")[0])
        self.assertFalse(audit_passes(m, profile="full")[0])

    def test_safety_ignores_train_sign_gate(self):
        # a low train_sign present would fail "full" but must be ignored under "safety"
        m = _good_metrics(train_sign_accuracy=0.5)
        self.assertTrue(audit_passes(m, profile="safety")[0])
        self.assertFalse(audit_passes(m, profile="full")[0])

    def test_safety_bellman_gate_is_nonterminal(self):
        # td_nrmse_all HIGH (terminal-distorted) but nonterminal OK -> safety PASSES; full FAILS
        m = _good_metrics(td_nrmse=0.9, td_nrmse_nonterminal=0.40)
        self.assertTrue(audit_passes(m, profile="safety")[0])
        self.assertFalse(audit_passes(m, profile="full")[0])
        # nonterminal too high -> safety fails on the nonterminal gate
        m2 = _good_metrics(td_nrmse=0.2, td_nrmse_nonterminal=0.6)
        passed, reasons = audit_passes(m2, profile="safety")
        self.assertFalse(passed)
        self.assertTrue(any("td_nrmse_nonterminal" in r for r in reasons))


class TerminalSafetyGateTests(unittest.TestCase):
    def _term(self, **over):
        base = dict(n_terminal=2, terminal_bad_count=2, terminal_bad_ranking_accuracy=1.0,
                    terminal_worst_q_margin=0.30)
        base.update(over)
        return _good_metrics(**base)

    def test_passes_when_terminals_ranked_below_with_margin(self):
        self.assertTrue(audit_passes(self._term(), profile="safety")[0])

    def test_fails_if_a_terminal_not_below_baseline(self):
        passed, reasons = audit_passes(self._term(terminal_bad_ranking_accuracy=0.5), profile="safety")
        self.assertFalse(passed)
        self.assertTrue(any("terminal_bad_ranking_accuracy" in r for r in reasons))

    def test_fails_if_terminal_margin_too_small(self):
        passed, reasons = audit_passes(self._term(terminal_worst_q_margin=0.10), profile="safety")
        self.assertFalse(passed)
        self.assertTrue(any("terminal_worst_q_margin" in r for r in reasons))

    def test_fails_if_terminal_not_classified_paired_bad(self):
        passed, reasons = audit_passes(self._term(terminal_bad_count=1), profile="safety")
        self.assertFalse(passed)
        self.assertTrue(any("not classified paired_bad" in r for r in reasons))

    def test_no_terminals_skips_terminal_gate(self):
        self.assertTrue(audit_passes(_good_metrics(n_terminal=0), profile="safety")[0])


def _terminal_split():
    """3 candidates: 2 nonterminal paired_bad (small residual) + 1 TERMINAL paired_bad with a big
    negative reward (-25). The terminal inflates td_nrmse_all but must NOT gate the nonterminal fit."""
    OBS, ACT = 4, 2
    obs = np.zeros((4, OBS), np.float32)
    return {
        "source": np.array(["baseline", "candidate", "candidate", "candidate"]),
        "label_usable": np.array([True] * 4),
        "paired_bad": np.array([False, True, True, True]),
        "pose_id": np.array(["p", "p", "p", "p"]), "phase": np.array(["mid"] * 4),
        "family": np.array(["baseline", "pga", "collapsed_v4", "random"]),
        "cell": np.array([17] * 4, np.int32),
        "outcome": np.array(["success", "env_time_limit", "env_time_limit", "terminal_failure"]),
        "obs": obs, "next_obs": obs.copy(),
        "actions": np.array([[0, 0], [0.5, 0.5], [0.6, 0.6], [1, 1]], np.float32),
        "rewards": np.array([0.0, -1.0, -1.0, -25.0], np.float32),
        "dones": np.array([0.0, 0.0, 0.0, 1.0], np.float32),        # only the last candidate terminal
        "return_delta_episode": np.array([0.0, -1.0, -2.0, -3.0], np.float32),
        "G_baseline_episode": np.array([1.0, 1.0, 1.0, 1.0], np.float32),
        "G_candidate_episode": np.array([1.0, 0.0, -1.0, -25.0], np.float32),
        "margin": np.array([0.5], np.float32),
    }


class TerminalAuditComputationTests(unittest.TestCase):
    def test_terminal_excluded_from_nonterminal_nrmse_but_caught_by_rank(self):
        # critic that ranks every candidate BELOW baseline (baseline action -> high Q, others low)
        def critic(obs, act):
            a = np.asarray(act, np.float64)
            return -np.sum(a ** 2, axis=1)          # baseline (0,0)->0 ; others negative
        m = counterexample_audit(
            _terminal_split(), critic, target_critic_min_q=lambda o, a: np.zeros(len(o)),
            bc_next_action=lambda n: np.zeros((len(n), 2), np.float32), gamma=0.99, margin=0.5)
        self.assertEqual(m["n_terminal"], 1)
        self.assertEqual(m["terminal_bad_count"], 1)
        self.assertEqual(m["terminal_bad_ranking_accuracy"], 1.0)     # terminal ranked below baseline
        self.assertGreater(m["terminal_worst_q_margin"], 0.25)
        # the big -25 terminal inflates td_nrmse_all far above the nonterminal subset
        self.assertGreater(m["td_nrmse"], m["td_nrmse_nonterminal"])
        self.assertGreater(m["terminal_target_rmse"], 20.0)          # ~|q_cand - (-25)| telemetry


if __name__ == "__main__":
    unittest.main()
