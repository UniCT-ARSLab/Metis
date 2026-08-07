"""Counterexample core (Godot-free): paired Q^BC returns, censoring, empirically_bad, same-anchor
margin, split disjointness, and the typed replay-only NPZ schema."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.counterexample import (  # noqa: E402
    FAMILIES, HORIZONS, META_DTYPES, TRANSITION_DTYPES, CounterexampleDataset,
    anchor_state_key, assert_splits_disjoint, calibrate_margin, classify_rollout,
    classify_termination, discounted_return, paired_verdict, rollout_returns,
    sign_stable_negative, validate_split_files,
)


class FamilyTests(unittest.TestCase):
    def test_nineteen_families(self):
        self.assertEqual(len(FAMILIES), 19)
        self.assertEqual(sum(f.startswith("sat_j") for f in FAMILIES), 14)
        self.assertIn("collapsed_v4", FAMILIES)
        self.assertIn("pga", FAMILIES)
        self.assertEqual(sum(f.startswith("perturbed_") for f in FAMILIES), 2)


class PairedReturnTests(unittest.TestCase):
    def test_discounted_and_horizon_returns(self):
        r = [1.0, 1.0, 1.0, 1.0]
        g = 0.5
        self.assertAlmostEqual(discounted_return(r, g), 1 + 0.5 + 0.25 + 0.125, places=6)
        self.assertAlmostEqual(discounted_return(r, g, horizon=2), 1.5, places=6)
        gh = rollout_returns(r, g, horizons=(1, 2, 4))
        self.assertAlmostEqual(gh[1], 1.0)
        self.assertAlmostEqual(gh[2], 1.5)
        self.assertAlmostEqual(gh[4], 1.875)

    def test_episode_return_matches_full_sum(self):
        # G_episode = discounted return over the whole [action once]+[clone tail] reward sequence.
        rewards = [0.2, -0.1, -0.3, 0.0, 0.5]
        self.assertAlmostEqual(discounted_return(rewards, 0.99),
                               sum(0.99 ** t * x for t, x in enumerate(rewards)), places=6)


class ClassifyRolloutTests(unittest.TestCase):
    def test_success(self):
        oc = classify_rollout(terminated=True, truncated=False, hit_tool_cap=False, is_success=True)
        self.assertEqual(oc["outcome"], "success")
        self.assertTrue(oc["label_usable"] and oc["return_complete"])

    def test_terminal_failure(self):
        oc = classify_rollout(terminated=True, truncated=False, hit_tool_cap=False, is_success=False)
        self.assertEqual(oc["outcome"], "terminal_failure")
        self.assertTrue(oc["label_usable"] and oc["return_complete"])

    def test_env_time_limit_usable(self):
        oc = classify_rollout(terminated=False, truncated=True, hit_tool_cap=False, is_success=False)
        self.assertEqual(oc["outcome"], "env_time_limit")
        self.assertTrue(oc["label_usable"])
        self.assertFalse(oc["return_complete"])

    def test_tool_cap_and_ongoing_unusable(self):
        self.assertFalse(classify_rollout(terminated=False, truncated=False, hit_tool_cap=True,
                                          is_success=False)["label_usable"])
        self.assertFalse(classify_rollout(terminated=False, truncated=False, hit_tool_cap=False,
                                          is_success=False)["label_usable"])


class CensoringTests(unittest.TestCase):
    def test_burst_termination_tag(self):
        self.assertEqual(classify_termination(True, False, False), ("terminated", False))
        self.assertEqual(classify_termination(False, True, False), ("truncated", True))


class SignStabilityTests(unittest.TestCase):
    def test_last_available_horizons_negative(self):
        deltas = {H: -1.0 for H in HORIZONS}
        self.assertTrue(sign_stable_negative(deltas, horizon_steps=200))
        flip = dict(deltas); flip[64] = +0.5  # a late positive flap
        self.assertFalse(sign_stable_negative(flip, horizon_steps=200))
        self.assertTrue(sign_stable_negative(deltas, horizon_steps=0))


class PairedVerdictTests(unittest.TestCase):
    """The A+C ordered rules with the baseline guard."""

    def _v(self, **kw):
        base = dict(base_outcome="success", base_success=True, base_collision=False,
                    cand_outcome="success", cand_success=True, cand_collision=False,
                    cand_label_usable=True, return_delta_episode=0.0,
                    deltas_by_h={H: 0.0 for H in HORIZONS}, margin=0.5, horizon_steps=200)
        base.update(kw)
        return paired_verdict(**base)

    def test_1_baseline_success_candidate_time_limit_is_bad(self):  # required #1
        v = self._v(base_outcome="success", base_success=True,
                    cand_outcome="env_time_limit", cand_success=False)
        self.assertTrue(v["paired_bad"])
        self.assertEqual(v["paired_bad_rule"], "outcome_dominance")

    def test_2_baseline_success_candidate_terminal_failure_is_bad(self):  # required #2
        v = self._v(cand_outcome="terminal_failure", cand_success=False)
        self.assertTrue(v["paired_bad"])
        # dominance fires first (base success, cand not success)
        self.assertEqual(v["paired_bad_rule"], "outcome_dominance")

    def test_2b_baseline_timelimit_candidate_failure_is_bad(self):
        v = self._v(base_outcome="env_time_limit", base_success=False,
                    cand_outcome="terminal_failure", cand_success=False)
        self.assertTrue(v["paired_bad"])
        self.assertEqual(v["paired_bad_rule"], "terminal_failure")

    def test_3_baseline_time_limit_candidate_success_not_bad(self):  # required #3
        v = self._v(base_outcome="env_time_limit", base_success=False,
                    cand_outcome="success", cand_success=True)
        self.assertFalse(v["paired_bad"])
        self.assertTrue(v["candidate_better"])

    def test_4_both_time_limit_return_margin(self):  # required #4
        good = self._v(base_outcome="env_time_limit", base_success=False,
                       cand_outcome="env_time_limit", cand_success=False,
                       return_delta_episode=-1.0, deltas_by_h={H: -1.0 for H in HORIZONS})
        self.assertTrue(good["paired_bad"] and good["paired_bad_rule"] == "return_margin")
        within = self._v(base_outcome="env_time_limit", base_success=False,
                         cand_outcome="env_time_limit", cand_success=False,
                         return_delta_episode=-0.2, deltas_by_h={H: -0.2 for H in HORIZONS})
        self.assertFalse(within["paired_bad"])

    def test_5_both_success_return_comparison(self):  # required #5
        v = self._v(return_delta_episode=-1.0, deltas_by_h={H: -1.0 for H in HORIZONS})
        self.assertTrue(v["paired_bad"] and v["paired_bad_rule"] == "return_margin")

    def test_7_tool_cap_censored(self):  # required #7
        v = self._v(cand_outcome="tool_cap", cand_success=False, cand_label_usable=False)
        self.assertFalse(v["paired_bad"])
        self.assertEqual(v["paired_bad_rule"], "censored")

    def test_10_outcome_dominance_beats_sign_stability(self):  # required #10
        # candidate time-limit with a POSITIVE (better-looking) return delta, but baseline succeeded:
        # dominance must still mark it bad, ignoring margin/sign-stability.
        v = self._v(base_outcome="success", base_success=True, cand_outcome="env_time_limit",
                    cand_success=False, return_delta_episode=+5.0,
                    deltas_by_h={H: +5.0 for H in HORIZONS})
        self.assertTrue(v["paired_bad"])
        self.assertEqual(v["paired_bad_rule"], "outcome_dominance")

    def test_candidate_collision_absent_in_baseline_is_bad(self):
        v = self._v(base_outcome="env_time_limit", base_success=False, base_collision=False,
                    cand_outcome="env_time_limit", cand_success=False, cand_collision=True)
        self.assertTrue(v["paired_bad"])
        self.assertEqual(v["paired_bad_rule"], "terminal_failure")


class MarginTests(unittest.TestCase):
    def test_same_anchor_margin_floor_and_q95(self):
        self.assertEqual(calibrate_margin([10.0, 10.0, 10.0]), 0.5)  # zero noise -> floor
        big = [0.0, 5.0, 10.0, 10.0]  # large spread -> q95 of pairwise |diff| > floor
        self.assertGreater(calibrate_margin(big), 0.5)

    def test_needs_three_replicates(self):
        with self.assertRaises(ValueError):
            calibrate_margin([1.0, 2.0])


class SplitTests(unittest.TestCase):
    def test_disjoint_ok(self):
        assert_splits_disjoint({
            "train": {"pose_id": ["a", "b"], "seed": [1, 2]},
            "selection": {"pose_id": ["c"], "seed": [3]},
            "final_test": {"pose_id": ["d"], "seed": [4]},
        })

    def test_shared_pose_id_raises(self):
        with self.assertRaises(ValueError):
            assert_splits_disjoint({
                "train": {"pose_id": ["a", "b"], "seed": [1, 2]},
                "final_test": {"pose_id": ["b"], "seed": [9]},
            })

    def test_shared_seed_raises(self):
        with self.assertRaises(ValueError):
            assert_splits_disjoint({
                "train": {"pose_id": ["a"], "seed": [1]},
                "selection": {"pose_id": ["z"], "seed": [1]},
            })


def _minimal_row(**over):
    row = dict(
        obs=np.zeros(27, np.float32), actions=np.zeros(7, np.float32), rewards=0.1,
        next_obs=np.zeros(27, np.float32), terminated=False, truncated=False, dones=False,
        branch_id=0, step_in_branch=0, branch_length=8, cell=17, seed=11700000,
        phase="early", pose_id="easy:17:11700000", family="collapsed_v4", source="candidate",
        failure_reason="none", termination_reason="terminated", outcome="success",
        baseline_replicate_id=-1,
        collision=False, self_collision=False, success=True, paired_bad=False,
        paired_bad_rule="none", candidate_better=False, safety_negative=False,
        return_censored=False,
        return_complete=True, matched_horizon=False, label_usable=True,
        censor_reason="none", horizon_steps=50,
        G_baseline=1.0, G_candidate=0.5, return_delta=-0.5,
        G_baseline_episode=1.0, G_candidate_episode=0.5, return_delta_episode=-0.5,
        return_delta_by_h=np.zeros(len(HORIZONS), np.float32),
    )
    row.update(over)
    return row


class SchemaTests(unittest.TestCase):
    def test_forbid_bc_label_keys(self):
        ds = CounterexampleDataset(27, 7)
        with self.assertRaises(ValueError):
            ds.add(**_minimal_row(), is_bc_label=True)

    def test_roundtrip_dtypes_and_replay_only(self):
        ds = CounterexampleDataset(27, 7)
        ds.add(**_minimal_row(source="baseline"))
        # a burst row: NaN returns, label unusable, safety_negative separate from paired_bad
        ds.add(**_minimal_row(source="ood_burst", family="sat_j5_pos", collision=True,
                              self_collision=True, failure_reason="self_collision",
                              termination_reason="self_collision", outcome="n/a",
                              label_usable=False, return_censored=True, paired_bad=False,
                              safety_negative=True, G_baseline=float("nan"),
                              G_candidate=float("nan"), return_delta=float("nan"),
                              G_baseline_episode=float("nan"), G_candidate_episode=float("nan"),
                              return_delta_episode=float("nan"),
                              return_delta_by_h=np.full(len(HORIZONS), np.nan, np.float32)))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ce.npz"
            ds.save(p, action_low=[-1.0] * 7, action_high=[1.0] * 7, margin=0.5,
                    meta={"generator": "test"})
            z = np.load(p, allow_pickle=True)
            self.assertEqual(z["paired_bad"].dtype, np.bool_)
            self.assertEqual(z["safety_negative"].dtype, np.bool_)
            self.assertTrue(z["outcome"].dtype.kind == "U")
            self.assertTrue(np.isnan(z["return_delta_episode"][1]))  # burst row -> NaN return
            self.assertFalse(np.isnan(z["return_delta_episode"][0]))  # baseline -> real
            # dtypes exactly per schema (not everything float32)
            self.assertEqual(z["terminated"].dtype, np.bool_)
            self.assertEqual(z["paired_bad"].dtype, np.bool_)
            self.assertEqual(z["cell"].dtype, np.int32)
            self.assertEqual(z["seed"].dtype, np.int64)
            self.assertTrue(z["family"].dtype.kind == "U")
            self.assertEqual(z["obs"].dtype, np.float32)
            self.assertEqual(z["return_delta_by_h"].shape, (2, len(HORIZONS)))
            self.assertEqual(z["label_usable"].dtype, np.bool_)
            self.assertEqual(z["horizon_steps"].dtype, np.int32)
            self.assertTrue(z["censor_reason"].dtype.kind == "U")
            # replay-only: no BC-label key present
            self.assertNotIn("is_bc_label", z.files)
            self.assertNotIn("is_demo", z.files)
            self.assertEqual(int(z["obs_dim"][0]), 27)

    def test_every_schema_key_present_in_output(self):
        ds = CounterexampleDataset(27, 7)
        ds.add(**_minimal_row())
        arrays = ds.to_arrays()
        for k in list(TRANSITION_DTYPES) + list(META_DTYPES) + ["return_delta_by_h"]:
            self.assertIn(k, arrays)


class PairedBadVsSafetyTests(unittest.TestCase):
    """paired_bad describes ONLY the authoritative candidate paired-Q^BC; safety_negative is the
    separate OOD-burst collision category. The ranking audit must select only candidate rows."""

    def _dataset(self):
        ds = CounterexampleDataset(27, 7)
        ds.add(**_minimal_row(source="candidate", family="collapsed_v4", paired_bad=True,
                              label_usable=True))
        ds.add(**_minimal_row(source="candidate", family="sat_j6_neg", paired_bad=False,
                              label_usable=True))
        # candidate whose label is NOT usable (tool_cap) -> excluded from ranking even if it looked bad
        ds.add(**_minimal_row(source="candidate", family="pga", paired_bad=False,
                              label_usable=False, outcome="tool_cap", return_censored=True))
        # a colliding burst -> safety_negative, NOT paired_bad
        ds.add(**_minimal_row(source="ood_burst", family="sat_j3_neg", collision=True,
                              paired_bad=False, safety_negative=True, label_usable=False,
                              outcome="n/a"))
        return ds.to_arrays()

    def test_ranking_selects_only_candidate_usable_paired_bad(self):
        a = self._dataset()
        sel = (a["source"] == "candidate") & a["label_usable"] & a["paired_bad"]
        self.assertEqual(int(sel.sum()), 1)  # only the collapsed_v4 candidate
        self.assertEqual(a["family"][sel][0], "collapsed_v4")
        # burst safety_negative is never counted as paired_bad
        self.assertEqual(int(a["paired_bad"][a["source"] == "ood_burst"].sum()), 0)
        self.assertEqual(int(a["safety_negative"][a["source"] == "ood_burst"].sum()), 1)


class SplitValidatorTests(unittest.TestCase):
    def _write(self, path, pose_ids, seeds, phases):
        ds = CounterexampleDataset(27, 7)
        for pid, sd, ph in zip(pose_ids, seeds, phases):
            ds.add(**_minimal_row(pose_id=pid, seed=sd, phase=ph))
        ds.save(path, action_low=[-1.0] * 7, action_high=[1.0] * 7, margin=0.5, meta={"g": "t"})

    def test_disjoint_splits_pass_and_report_counts(self):
        with tempfile.TemporaryDirectory() as d:
            tr, se, ft = (Path(d) / f"{n}.npz" for n in ("train", "sel", "final"))
            # same pose across 3 phases -> anchor_states 3, unique_pose_id 1 per split
            self._write(tr, ["p1", "p1", "p1"], [10, 10, 10], ["early", "mid", "late"])
            self._write(se, ["p2", "p2"], [20, 20], ["early", "mid"])
            self._write(ft, ["p3"], [30], ["early"])
            rep = validate_split_files({"train": str(tr), "selection": str(se), "final": str(ft)})
            self.assertTrue(rep["disjoint"])
            self.assertEqual(rep["train"]["unique_pose_id"], 1)
            self.assertEqual(rep["train"]["anchor_states"], 3)
            self.assertEqual(rep["train"]["rows"], 3)

    def test_shared_pose_across_splits_raises(self):
        with tempfile.TemporaryDirectory() as d:
            tr, ft = Path(d) / "train.npz", Path(d) / "final.npz"
            self._write(tr, ["pX"], [10], ["mid"])
            self._write(ft, ["pX"], [99], ["mid"])  # shared pose_id
            with self.assertRaises(ValueError):
                validate_split_files({"train": str(tr), "final": str(ft)})

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            validate_split_files({"train": "/nonexistent/xyz.npz"})

    def test_anchor_state_key_distinguishes_phase(self):
        self.assertNotEqual(anchor_state_key("p", "early"), anchor_state_key("p", "late"))


class MetisAndGuardTests(unittest.TestCase):
    def test_6_baseline_terminal_failure_is_the_reject_outcome(self):
        # The generator discards an anchor whose baseline is terminal_failure. Encode the decision.
        base = classify_rollout(terminated=True, truncated=False, hit_tool_cap=False, is_success=False)
        self.assertEqual(base["outcome"], "terminal_failure")
        # paired_verdict is defensive if such a baseline ever reaches it.
        v = paired_verdict(base_outcome="terminal_failure", base_success=False, base_collision=True,
                           cand_outcome="success", cand_success=True, cand_collision=False,
                           cand_label_usable=True, return_delta_episode=0.0,
                           deltas_by_h={H: 0.0 for H in HORIZONS}, margin=0.5, horizon_steps=200)
        self.assertEqual(v["paired_bad_rule"], "baseline_failure")

    def test_8_truncated_does_not_become_dones(self):
        # Metis replay: a time-limited (truncated, not terminated) transition has dones=False.
        ds = CounterexampleDataset(27, 7)
        ds.add(**_minimal_row(source="baseline", terminated=False, truncated=True, dones=False,
                              outcome="env_time_limit", success=False, return_complete=False))
        z = ds.to_arrays()
        self.assertFalse(bool(z["dones"][0]))
        self.assertTrue(bool(z["truncated"][0]))

    def test_9_matched_horizon_false_still_usable_env_time_limit(self):
        oc = classify_rollout(terminated=False, truncated=True, hit_tool_cap=False, is_success=False)
        self.assertTrue(oc["label_usable"])  # usability does NOT depend on matched_horizon
        v = paired_verdict(base_outcome="env_time_limit", base_success=False, base_collision=False,
                           cand_outcome="env_time_limit", cand_success=False, cand_collision=False,
                           cand_label_usable=True, return_delta_episode=-1.0,
                           deltas_by_h={H: -1.0 for H in HORIZONS}, margin=0.5, horizon_steps=200)
        self.assertTrue(v["paired_bad"])  # unmatched horizon still yields a usable verdict


if __name__ == "__main__":
    unittest.main()
