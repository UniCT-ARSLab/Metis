"""Regression test for the M7.0b hovering-vs-success comparison. The FULL remaining-horizon return
must INCLUDE the terminal bonus even when success happens AFTER the short window, so hovering is
compared to success on equivalent horizons (the bug: short-window truncation hid the ~50 bonus and
made hovering look non-dominated)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.m7_0b_reward_preflight import compute_verdict, full_returns  # noqa: E402

H = 25


def _mag(toward_full, toward_reached):
    t = {"one_step": 0.1, "short": 0.5, "short_no_bonus": 0.5, "disc": 0.4,
         "full": toward_full, "full_disc": toward_full * 0.5, "reached": toward_reached, "min_dist": 0.02}
    a = {"one_step": -0.1, "short": -0.5, "short_no_bonus": -0.5, "disc": -0.4,
         "full": -1.0, "full_disc": -0.5, "reached": False, "min_dist": 0.05}
    return {"toward": t, "away": a}


def _mk_row(cell, hover_full, hover_reached, gate_toward_full, gate_toward_reached, seed=0):
    """Minimal m7.0b row honouring the schema compute_verdict consumes (all MAGS present)."""
    m = _mag(gate_toward_full, gate_toward_reached)
    return {"cell": cell, "seed": seed, "bucket": "0.02-0.03", "dist_k": 0.025,
            "hover_undisc": 0.02, "hover_disc": 0.02,
            "hover_full_undisc": hover_full, "hover_full_disc": hover_full * 0.7,
            "hover_reached": hover_reached,
            "mags": {"0.0025": m, "0.005": m, "0.01": m}}


class FullReturnsTests(unittest.TestCase):
    def test_late_success_bonus_in_full_not_short(self):
        # success at step 30 (> H): the +50 bonus is in the FULL return but NOT the short return.
        b = {"tail": [0.001] * 30 + [50.0], "reached": True, "bonus": 50.0}
        f, s = full_returns(b, H)
        self.assertGreater(f["full"], 49.9)                 # full includes the bonus
        self.assertLess(s["short"], 1.0)                    # short (first 25) does not
        self.assertAlmostEqual(s["short_no_bonus"], s["short"], places=6)  # nothing to subtract in short

    def test_early_success_bonus_in_short(self):
        # success at step 5 (<= H): bonus is in short; short_no_bonus removes it.
        b = {"tail": [0.001] * 5 + [50.0], "reached": True, "bonus": 50.0}
        f, s = full_returns(b, H)
        self.assertGreater(s["short"], 49.9)
        self.assertLess(s["short_no_bonus"], 1.0)           # bonus subtracted
        self.assertAlmostEqual(f["full"], s["short"], places=6)  # episode ended within H

    def test_hovering_no_bonus(self):
        b = {"tail": [0.001] * 250, "reached": False, "bonus": 0.0}
        f, s = full_returns(b, H)
        self.assertAlmostEqual(f["full"], 0.25, places=3)
        self.assertAlmostEqual(s["short"], 0.025, places=4)

    def test_full_horizon_makes_hover_dominated_where_short_did_not(self):
        # hovering: 250 steps of +0.001 = 0.25. late-success: bonus 50 at step 30.
        hover = full_returns({"tail": [0.001] * 250, "reached": False, "bonus": 0.0}, H)[0]
        succ = full_returns({"tail": [0.001] * 30 + [50.0], "reached": True, "bonus": 50.0}, H)
        # FULL horizon: hovering (0.25) << success (50.03) -> dominated (the fix)
        self.assertLess(hover["full"], succ[0]["full"])
        # SHORT horizon (the OLD buggy comparison): hovering 0.025 vs success short 0.025 -> NOT dominated
        self.assertGreaterEqual(hover["full"] / max(succ[1]["short"], 1e-9), 1.0)


class ComputeVerdictTests(unittest.TestCase):
    def test_self_reaching_hover_excluded_from_ceiling(self):
        # Row B is a pure-composite branch that reaches on its own (hover_full 52, reached) -> it is a
        # SUCCESS, not hovering. If NOT excluded, hover_max 52 > success_min 50 would break domination.
        rows = [
            _mk_row(25, hover_full=1.0, hover_reached=False, gate_toward_full=50.0, gate_toward_reached=True),
            _mk_row(28, hover_full=52.0, hover_reached=True, gate_toward_full=51.0, gate_toward_reached=True),
        ]
        # sanity: the naive max over ALL hover branches WOULD be the 52 trap
        self.assertEqual(max(r["hover_full_undisc"] for r in rows), 52.0)
        v = compute_verdict(rows)
        self.assertEqual(v["n_hover_selfreached"], 1)
        self.assertEqual(v["hover_max_full_undisc"], 1.0)          # trap excluded -> true stalling ceiling
        self.assertTrue(v["hover_dominated_by_success"])           # 1.0 << 50.0
        self.assertTrue(v["CLOSE_M7_0"])                           # + toward beats away in every cell

    def test_real_stalling_not_dominated_flags_false(self):
        # If a genuine (non-reaching) stalling branch out-earns the worst success, domination must be False.
        rows = [
            _mk_row(25, hover_full=60.0, hover_reached=False, gate_toward_full=50.0, gate_toward_reached=True),
        ]
        v = compute_verdict(rows)
        self.assertFalse(v["hover_dominated_by_success"])
        self.assertFalse(v["CLOSE_M7_0"])


if __name__ == "__main__":
    unittest.main()
