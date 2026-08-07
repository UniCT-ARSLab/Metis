"""FamilyActions invariants (Godot-free): every family action stays in-bounds; per-joint saturation
sets exactly that joint to its limit; burst hold vs recompute behaves per family."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from tools.generate_counterexamples import FamilyActions  # noqa: E402
from core.counterexample import FAMILIES  # noqa: E402

OBS, ACT = 27, 7


def _actor():
    return tf.keras.Sequential([tf.keras.layers.Input((OBS,)),
                                tf.keras.layers.Dense(ACT, activation="tanh")])


def _critic():
    oi, ai = tf.keras.layers.Input((OBS,)), tf.keras.layers.Input((ACT,))
    x = tf.keras.layers.Concatenate()([oi, ai])
    return tf.keras.Model([oi, ai], tf.keras.layers.Dense(1)(x))


class FamilyActionTests(unittest.TestCase):
    def setUp(self):
        low = np.full(ACT, -1.0, np.float32)
        high = np.full(ACT, 1.0, np.float32)
        self.low, self.high = low, high
        self.fa = FamilyActions(_actor(), _actor(), _critic(), _critic(), low, high, seed=0)
        self.obs = np.zeros(OBS, np.float32)

    def test_all_families_in_bounds(self):
        for fam in FAMILIES + ["baseline"]:
            a = self.fa.first_action(fam, self.obs)
            self.assertEqual(a.shape, (ACT,))
            self.assertTrue(np.all(a >= self.low - 1e-5) and np.all(a <= self.high + 1e-5), fam)

    def test_per_joint_saturation_sets_exactly_that_joint(self):
        for j in range(ACT):
            ap = self.fa.first_action(f"sat_j{j}_pos", self.obs)
            an = self.fa.first_action(f"sat_j{j}_neg", self.obs)
            self.assertAlmostEqual(ap[j], self.high[j], places=5)
            self.assertAlmostEqual(an[j], self.low[j], places=5)

    def test_burst_hold_vs_recompute(self):
        # sat_j holds the s0 command; random holds (fixed_burst); collapsed/pga/perturbed recompute.
        first = self.fa.first_action("sat_j2_pos", self.obs)
        held = self.fa.burst_action("sat_j2_pos", np.ones(OBS, np.float32), first, step_idx=3)
        np.testing.assert_allclose(held, first, atol=1e-6)
        self.assertIn("random", self.fa.HOLD)
        self.assertIn("sat_j2_pos", self.fa.HOLD)
        self.assertNotIn("collapsed_v4", self.fa.HOLD)
        self.assertNotIn("pga", self.fa.HOLD)

    def test_random_per_step_resamples(self):
        fa = FamilyActions(_actor(), _actor(), _critic(), _critic(), self.low, self.high,
                           random_mode="per_step", seed=1)
        first = fa.first_action("random", self.obs)
        step2 = fa.burst_action("random", self.obs, first, step_idx=1)
        self.assertFalse(np.allclose(first, step2))  # new sample each step


if __name__ == "__main__":
    unittest.main()
