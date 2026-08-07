"""PairSampler (M6.1 v3): anchor matching, TRAIN-only source, |return_delta|>=margin exclusion,
50/50 sign balance, (cell,phase,family) balance within sign, and the hard guard that selection /
final_test can never enter the pair source."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pair_sampler import PairSampler  # noqa: E402

OBS, ACT = 27, 7
BASE_A = np.full(ACT, 0.5, np.float32)     # distinctive baseline action at anchor A
BASE_B = np.full(ACT, -0.5, np.float32)    # ... and at anchor B


def _write_ce(path, rows, *, pose_map, margin=0.5):
    """rows: dicts with source/label_usable/return_delta/cell/phase/family/pose/action(optional)."""
    n = len(rows)
    def act_of(r):
        if "action" in r:
            return r["action"]
        return np.zeros(ACT, np.float32)
    np.savez(
        path,
        obs=np.arange(n * OBS, dtype=np.float32).reshape(n, OBS),   # unique-ish obs per row
        actions=np.stack([act_of(r) for r in rows]).astype(np.float32),
        source=np.array([r["source"] for r in rows], "<U12"),
        label_usable=np.array([r["label_usable"] for r in rows], np.bool_),
        return_delta_episode=np.array([r["return_delta"] for r in rows], np.float32),
        cell=np.array([r["cell"] for r in rows], np.int32),
        phase=np.array([r["phase"] for r in rows], "<U8"),
        family=np.array([r["family"] for r in rows], "<U16"),
        pose_id=np.array([pose_map[r["pose"]] for r in rows]),
        paired_bad=np.array([r.get("paired_bad", False) for r in rows], np.bool_),
        margin=np.array([margin], np.float32))


def _rows_basic():
    rows = [
        dict(source="baseline", label_usable=True, return_delta=0.0, cell=17, phase="mid",
             family="baseline", pose="A", action=BASE_A),
        dict(source="baseline", label_usable=True, return_delta=0.0, cell=17, phase="mid",
             family="baseline", pose="B", action=BASE_B),
    ]
    # candidates at A: 2 negative (pga, collapsed_v4), 1 positive (candidate_better)
    rows += [
        dict(source="candidate", label_usable=True, return_delta=-1.0, cell=17, phase="mid",
             family="pga", pose="A", action=np.full(ACT, 0.9, np.float32)),
        dict(source="candidate", label_usable=True, return_delta=-2.0, cell=17, phase="mid",
             family="collapsed_v4", pose="A", action=np.full(ACT, 1.0, np.float32)),
        dict(source="candidate", label_usable=True, return_delta=1.5, cell=17, phase="mid",
             family="perturbed_0.25", pose="A", action=np.full(ACT, 0.4, np.float32)),
    ]
    # candidates at B: 1 negative, 1 positive
    rows += [
        dict(source="candidate", label_usable=True, return_delta=-1.2, cell=18, phase="mid",
             family="sat_j0_neg", pose="B", action=np.full(ACT, -0.9, np.float32)),
        dict(source="candidate", label_usable=True, return_delta=0.8, cell=18, phase="mid",
             family="sat_j0_pos", pose="B", action=np.full(ACT, -0.2, np.float32)),
    ]
    # EXCLUDED: |return_delta| < margin(0.5), and label_usable=False
    rows += [
        dict(source="candidate", label_usable=True, return_delta=-0.2, cell=17, phase="mid",
             family="pga", pose="A", action=np.zeros(ACT, np.float32)),
        dict(source="candidate", label_usable=False, return_delta=-3.0, cell=17, phase="mid",
             family="pga", pose="A", action=np.zeros(ACT, np.float32)),
    ]
    return rows


class PairSamplerTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        p = Path(self.d.name)
        self.train = p / "ce_train.npz"
        self.sel = p / "ce_sel.npz"
        self.fin = p / "ce_fin.npz"
        _write_ce(self.train, _rows_basic(), pose_map={"A": "trainA", "B": "trainB"})
        _write_ce(self.sel, [_rows_basic()[0]], pose_map={"A": "selZ", "B": "selZ"})
        _write_ce(self.fin, [_rows_basic()[0]], pose_map={"A": "finY", "B": "finY"})

    def tearDown(self):
        self.d.cleanup()

    def test_excludes_within_margin_and_unusable(self):
        s = PairSampler(str(self.train), seed=0)
        # 7 candidates - 1 (|delta|=0.2 < margin) - 1 (label_usable=False) = 5 significant usable
        self.assertEqual(s.n_pairs(), 5)         # 3 at anchor A + 2 at anchor B
        self.assertEqual((s.n_pos(), s.n_neg()), (2, 3))

    def test_anchor_match_uses_same_pose_phase_baseline_action(self):
        s = PairSampler(str(self.train), seed=0)
        # every pair whose obs came from an A-candidate must carry BASE_A; a B-candidate carries BASE_B
        for i in range(s.n_pairs()):
            base = s._act_base[i]
            self.assertTrue(np.allclose(base, BASE_A) or np.allclose(base, BASE_B))
        # count: 3 A-candidates -> BASE_A, 2 B-candidates -> BASE_B
        n_a = int(np.sum(np.all(np.isclose(s._act_base, BASE_A), axis=1)))
        n_b = int(np.sum(np.all(np.isclose(s._act_base, BASE_B), axis=1)))
        self.assertEqual((n_a, n_b), (3, 2))

    def test_sign_balance_5050(self):
        s = PairSampler(str(self.train), seed=1)
        pos = neg = 0
        for _ in range(200):
            b, comp = s.sample(64)
            self.assertEqual(comp["n_pos"], 32)
            self.assertEqual(comp["n_neg"], 32)
            pos += int(np.sum(b["return_delta"] > 0))
            neg += int(np.sum(b["return_delta"] < 0))
        frac_pos = pos / (pos + neg)
        self.assertGreater(frac_pos, 0.45)
        self.assertLess(frac_pos, 0.55)

    def test_guard_rejects_eval_overlap(self):
        # overlapping pose_id -> raise (selection or final_test may never enter the pair source)
        bad = Path(self.d.name) / "ce_overlap.npz"
        _write_ce(bad, _rows_basic(), pose_map={"A": "shared", "B": "trainB"})
        overlap = Path(self.d.name) / "ce_sel_shared.npz"
        _write_ce(overlap, [_rows_basic()[0]], pose_map={"A": "shared", "B": "shared"})
        with self.assertRaises(ValueError):
            PairSampler(str(bad), eval_paths=[str(overlap)])
        # disjoint selection + final_test are accepted
        PairSampler(str(self.train), eval_paths=[str(self.sel), str(self.fin)])

    def test_final_test_never_merged(self):
        # passing final_test as an eval path must not add any of its rows to the pair source
        a = PairSampler(str(self.train), seed=0).n_pairs()
        b = PairSampler(str(self.train), eval_paths=[str(self.fin)], seed=0).n_pairs()
        self.assertEqual(a, b)

    def test_checksum_mismatch_raises(self):
        with self.assertRaises(ValueError):
            PairSampler(str(self.train), expected_train_sha256="deadbeef")

    def test_all_pairs_full_set_with_family_and_cellphase(self):
        s = PairSampler(str(self.train), seed=0)
        ap = s.all_pairs()
        self.assertEqual(len(ap["obs"]), s.n_pairs())            # full set, no sampling
        self.assertEqual(len(ap["family"]), s.n_pairs())
        self.assertEqual(len(ap["cellphase"]), s.n_pairs())
        self.assertEqual(set(ap["family"]),
                         {"pga", "collapsed_v4", "perturbed_0.25", "sat_j0_neg", "sat_j0_pos"})
        self.assertTrue(all("|" in cp for cp in ap["cellphase"]))   # "cell|phase"


class PairSamplerBalanceTests(unittest.TestCase):
    """Within a sign, (cell,phase,family) groups are drawn uniformly, not row-uniform: a rare group's
    single pair appears far above its row frequency."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.train = Path(self.d.name) / "ce.npz"
        rows = [dict(source="baseline", label_usable=True, return_delta=0.0, cell=17, phase="mid",
                     family="baseline", pose="A", action=BASE_A)]
        # negative sign: BIG group (20 rows, rd=-1) + SMALL group (1 row, rd=-9, unique tag)
        for _ in range(20):
            rows.append(dict(source="candidate", label_usable=True, return_delta=-1.0, cell=17,
                             phase="mid", family="random", pose="A"))
        rows.append(dict(source="candidate", label_usable=True, return_delta=-9.0, cell=18,
                         phase="mid", family="pga", pose="A"))
        # positive sign: one group so 50/50 draw has something to pull
        for _ in range(5):
            rows.append(dict(source="candidate", label_usable=True, return_delta=2.0, cell=17,
                             phase="mid", family="perturbed_0.25", pose="A"))
        _write_ce(self.train, rows, pose_map={"A": "trainA"})

    def tearDown(self):
        self.d.cleanup()

    def test_rare_group_overrepresented_vs_row_share(self):
        s = PairSampler(str(self.train), seed=3)
        hits = total_neg = 0
        for _ in range(80):
            b, _c = s.sample(200)
            neg = b["return_delta"] < 0
            total_neg += int(np.sum(neg))
            hits += int(np.sum(b["return_delta"] == -9.0))
        frac = hits / total_neg
        self.assertGreater(frac, 0.30)   # group-uniform ~0.5; row-uniform would be ~1/21 ~ 0.048
        self.assertLess(frac, 0.70)


class PairedBadPopulationTests(unittest.TestCase):
    """v5 SAFETY: population='paired_bad' keeps only hard-negatives and reserves attack_fraction for
    pga + collapsed_v4 (balanced 50/50), so the rare pga family is never drowned."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.train = Path(self.d.name) / "ce.npz"
        rows = [dict(source="baseline", label_usable=True, return_delta=0.0, cell=17, phase="mid",
                     family="baseline", pose="A", action=BASE_A)]
        # 40 non-attack hard-negatives (random), 4 pga, 20 collapsed_v4 -- all paired_bad, rd<0
        for _ in range(40):
            rows.append(dict(source="candidate", label_usable=True, paired_bad=True, return_delta=-1.0,
                             cell=17, phase="mid", family="random", pose="A"))
        for _ in range(4):
            rows.append(dict(source="candidate", label_usable=True, paired_bad=True, return_delta=-2.0,
                             cell=17, phase="mid", family="pga", pose="A"))
        for _ in range(20):
            rows.append(dict(source="candidate", label_usable=True, paired_bad=True, return_delta=-1.5,
                             cell=18, phase="mid", family="collapsed_v4", pose="A"))
        # a candidate_better (positive, NOT paired_bad) that must be EXCLUDED from the population
        rows.append(dict(source="candidate", label_usable=True, paired_bad=False, return_delta=2.0,
                         cell=17, phase="mid", family="perturbed_0.25", pose="A"))
        _write_ce(self.train, rows, pose_map={"A": "trainA"})

    def tearDown(self):
        self.d.cleanup()

    def test_only_paired_bad_negatives(self):
        s = PairSampler(str(self.train), population="paired_bad", seed=0)
        self.assertEqual(s.n_pairs(), 64)            # 40 + 4 + 20, the candidate_better excluded
        self.assertEqual(s.n_pos(), 0)               # all hard-negatives
        self.assertEqual(s.n_neg(), 64)

    def test_attack_fraction_reserves_pga_and_collapsed(self):
        s = PairSampler(str(self.train), population="paired_bad", attack_fraction=0.5, seed=1)
        # pga rows carry rd=-2, collapsed rd=-1.5, random rd=-1 -> identify families by rd value
        n_attack = n_pga = n_coll = total = 0
        for _ in range(40):
            b, comp = s.sample(256)
            self.assertEqual(comp["n_attack"], 128)  # 0.5 * 256
            total += comp["batch"]
            n_attack += comp["n_attack"]
            n_pga += int(np.sum(b["return_delta"] == -2.0))
            n_coll += int(np.sum(b["return_delta"] == -1.5))
        # attack half split 50/50 pga/collapsed -> each ~25% of the batch despite pga having 4 rows
        self.assertAlmostEqual(n_pga / total, 0.25, delta=0.04)
        self.assertAlmostEqual(n_coll / total, 0.25, delta=0.04)

    def test_paired_bad_requires_the_field(self):
        # significant mode still works when paired_bad present but unused
        PairSampler(str(self.train), population="significant", seed=0)


if __name__ == "__main__":
    unittest.main()
