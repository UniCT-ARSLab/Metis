"""Stratified sampler: quota compliance, branch-balanced safety sampling, and the hard guard that
selection/final_test can never enter the replay."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.counterexample_sampler import CounterexampleSampler  # noqa: E402

OBS, ACT = 27, 7


def _write_m5(path, n=200, pose_prefix="m5"):
    np.savez(path, obs=np.zeros((n, OBS), np.float32), actions=np.zeros((n, ACT), np.float32),
             rewards=np.zeros(n, np.float32), next_obs=np.zeros((n, OBS), np.float32),
             terminated=np.zeros(n, np.float32), dones=np.zeros(n, np.float32),
             cells=np.zeros(n, np.int32), pose_id=np.array([f"{pose_prefix}:{i}" for i in range(n)]))


def _write_ce(path, *, pose_ids, rows):
    """rows: list of dicts with source/paired_bad/safety_negative/cell/phase/family/branch_id."""
    n = len(rows)
    def col(k, dt):
        return np.array([r[k] for r in rows], dtype=dt)
    np.savez(path,
             obs=np.zeros((n, OBS), np.float32), actions=np.zeros((n, ACT), np.float32),
             rewards=np.arange(1, n + 1, dtype=np.float32), next_obs=np.zeros((n, OBS), np.float32),
             terminated=np.zeros(n, np.float32), truncated=np.zeros(n, np.float32),
             dones=np.zeros(n, np.float32),
             source=col("source", "<U10"), paired_bad=col("paired_bad", np.bool_),
             safety_negative=col("safety_negative", np.bool_), cell=col("cell", np.int32),
             phase=col("phase", "<U8"), family=col("family", "<U16"),
             branch_id=col("branch_id", np.int32),
             pose_id=np.array([pose_ids[r["pose"]] for r in rows]))


def _rows():
    rows = []
    bid = 0
    for cell in (17, 18):
        for fam in ("random", "sat_j0_pos"):
            for _ in range(6):
                rows.append(dict(source="candidate", paired_bad=True, safety_negative=False,
                                 cell=cell, phase="mid", family=fam, branch_id=bid, pose=0)); bid += 1
    for _ in range(12):
        rows.append(dict(source="candidate", paired_bad=False, safety_negative=False,
                         cell=17, phase="mid", family="random", branch_id=bid, pose=0)); bid += 1
    # safety: branch BIG (10 transitions) + branch SMALL (1 transition)
    big = bid; bid += 1
    for _ in range(10):
        rows.append(dict(source="ood_burst", paired_bad=False, safety_negative=True, cell=17,
                         phase="mid", family="sat_j3_neg", branch_id=big, pose=0))
    small = bid; bid += 1
    rows.append(dict(source="ood_burst", paired_bad=False, safety_negative=True, cell=18,
                     phase="mid", family="sat_j1_neg", branch_id=small, pose=0))
    for _ in range(8):
        rows.append(dict(source="baseline", paired_bad=False, safety_negative=False, cell=17,
                         phase="mid", family="baseline", branch_id=bid, pose=0)); bid += 1
    # recovery: 5 branches x 4 transitions each (branch-first target)
    for _ in range(5):
        for _ in range(4):
            rows.append(dict(source="recovery", paired_bad=False, safety_negative=False, cell=17,
                             phase="mid", family="baseline", branch_id=bid, pose=0))
        bid += 1
    # non-negative burst (ood_burst & not safety_negative): 5 branches x 4 transitions
    for _ in range(5):
        for _ in range(4):
            rows.append(dict(source="ood_burst", paired_bad=False, safety_negative=False, cell=17,
                             phase="mid", family="sat_j0_neg", branch_id=bid, pose=0))
        bid += 1
    return rows, big, small


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        p = Path(self.d.name)
        self.m5 = p / "m5.npz"
        self.train = p / "ce_train.npz"
        self.selection = p / "ce_sel.npz"
        _write_m5(self.m5)
        rows, self.big, self.small = _rows()
        _write_ce(self.train, pose_ids={0: "train_pose_A"}, rows=rows)
        _write_ce(self.selection, pose_ids={0: "sel_pose_Z"},
                  rows=[dict(source="candidate", paired_bad=True, safety_negative=False, cell=17,
                             phase="mid", family="random", branch_id=0, pose=0)])

    def tearDown(self):
        self.d.cleanup()

    def test_quotas(self):
        s = CounterexampleSampler(str(self.m5), str(self.train), seed=1)
        _b, comp = s.sample(1000)
        self.assertEqual(comp["m5v2"], 500)
        self.assertEqual(comp["cand_bad"], 250)
        self.assertEqual(comp["cand_nonbad"], 100)
        self.assertEqual(comp["safety"], 75)
        self.assertEqual(comp["neutral"], 75)
        self.assertEqual(sum(comp.values()), 1000)

    def test_counts_sum_exact_odd_batch(self):
        s = CounterexampleSampler(str(self.m5), str(self.train), seed=2)
        _b, comp = s.sample(37)
        self.assertEqual(sum(comp.values()), 37)

    def test_safety_branch_balanced_not_row_uniform(self):
        # 2 safety branches (BIG=10 rows, SMALL=1 row). Branch-first sampling picks each branch ~50%,
        # so the SMALL branch's single transition appears ~half the time -- far above the ~1/11
        # frequency a uniform-over-11-rows sampler would give.
        s = CounterexampleSampler(str(self.m5), str(self.train), seed=3)
        rng_reward = s._ce["rewards"]  # unique per-row id
        small_reward = rng_reward[s._safety_branches[self.small][0]]
        hits, total = 0, 0
        for _ in range(60):
            b, comp = s.sample(1000)
            # the safety slice is the last comp['safety'] rows before 'neutral'; identify by reward id
            total += comp["safety"]
            hits += int(np.sum(b["rewards"] == small_reward))
        frac = hits / total
        self.assertGreater(frac, 0.30)   # branch-balanced ~0.5; row-uniform would be ~0.09
        self.assertLess(frac, 0.70)

    def test_guard_rejects_eval_overlap(self):
        # replay == a file whose poses overlap an eval split -> raise. Here train poses overlap
        # themselves if train is also passed as an eval path.
        with self.assertRaises(ValueError):
            CounterexampleSampler(str(self.m5), str(self.train), eval_paths=[str(self.train)])
        # disjoint selection is fine
        CounterexampleSampler(str(self.m5), str(self.train), eval_paths=[str(self.selection)])

    def test_guard_rejects_loading_selection_as_replay(self):
        # loading the selection file AS the replay, with the real selection listed as eval -> overlap
        with self.assertRaises(ValueError):
            CounterexampleSampler(str(self.m5), str(self.selection), eval_paths=[str(self.selection)])

    def test_category_sizes(self):
        s = CounterexampleSampler(str(self.m5), str(self.train), seed=0)
        cs = s.category_sizes()
        self.assertEqual(cs["cand_bad"], 24)     # 2 cells x 2 fams x 6
        self.assertEqual(cs["cand_nonbad"], 12)
        self.assertEqual(cs["safety"], 11)       # 10 + 1
        self.assertEqual(cs["m5v2"], 200)

    def test_checksum_mismatch_raises(self):
        with self.assertRaises(ValueError):
            CounterexampleSampler(str(self.m5), str(self.train), expected_train_sha256="deadbeef")

    def test_neutral_source_balanced(self):
        # baseline (8 rows) vs recovery (20) vs nonneg_burst (20): source-first sampling gives each
        # ~1/3, so baseline appears far above its 8/48 row share.
        s = CounterexampleSampler(str(self.m5), str(self.train), seed=5)
        ce = np.load(self.train, allow_pickle=True)
        src = ce["source"]
        idx = np.concatenate([s._sample_neutral(300) for _ in range(20)])
        base_frac = float(np.mean(src[idx] == "baseline"))
        self.assertGreater(base_frac, 0.22)   # ~1/3 source-balanced (row-uniform would be ~0.17)
        # all three sub-sources represented
        got = set(src[idx].tolist())
        self.assertEqual(got, {"baseline", "recovery", "ood_burst"})

    def test_final_batch_is_shuffled(self):
        # M5 rows have reward 0; counterexample rows have reward >= 1. Unshuffled, the first 50%
        # (m5 quota) would all be reward-0. After the final shuffle they are interspersed.
        s = CounterexampleSampler(str(self.m5), str(self.train), seed=7)
        b, comp = s.sample(1000)
        half = 500
        zeros_in_first_half = int(np.sum(b["rewards"][:half] == 0.0))
        self.assertLess(zeros_in_first_half, comp["m5v2"])   # not all m5 up front
        self.assertGreater(zeros_in_first_half, 50)          # but some m5 present -> genuinely mixed


if __name__ == "__main__":
    unittest.main()
