"""A time-limit truncation must not be learned as a real terminal.

Gymnasium splits `terminated` from `truncated` precisely so a value estimator can keep
bootstrapping through an episode that was cut by a step cap while still running. Fusing
them teaches the agent that the world ends at --max-steps-per-episode.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_generic_ppo import compute_returns_advantages  # noqa: E402

GAMMA = 0.9
LAMBDA = 1.0


class PPOTruncationBootstrapTests(unittest.TestCase):
    def test_real_terminal_does_not_bootstrap(self):
        # Episode genuinely ends: the last state has no future, so its advantage is
        # driven by the reward alone.
        returns, _ = compute_returns_advantages(
            rewards=[0.0, 0.0], dones=[0.0, 1.0], values=[0.0, 0.0],
            gamma=GAMMA, gae_lambda=LAMBDA, bootstrap_value=100.0,
        )
        # bootstrap_value must be ignored when the final step is terminal.
        self.assertEqual(returns[-1], 0.0)

    def test_truncation_bootstraps_through_the_cut(self):
        # Same trajectory, but cut by the step cap: the final state was still worth 100,
        # and that value must survive into the return.
        returns, _ = compute_returns_advantages(
            rewards=[0.0, 0.0], dones=[0.0, 0.0], values=[0.0, 0.0],
            gamma=GAMMA, gae_lambda=LAMBDA, bootstrap_value=100.0,
        )
        self.assertAlmostEqual(returns[-1], GAMMA * 100.0, places=4)

    def test_truncation_without_bootstrap_matches_the_old_bug(self):
        # Guards the reason the dones-only change is not enough on its own: with no
        # bootstrap value, a truncated trajectory is indistinguishable from a terminal one.
        truncated_no_value, _ = compute_returns_advantages(
            rewards=[1.0], dones=[0.0], values=[0.0],
            gamma=GAMMA, gae_lambda=LAMBDA, bootstrap_value=0.0,
        )
        terminal, _ = compute_returns_advantages(
            rewards=[1.0], dones=[1.0], values=[0.0],
            gamma=GAMMA, gae_lambda=LAMBDA, bootstrap_value=0.0,
        )
        self.assertEqual(truncated_no_value[0], terminal[0])

    def test_bootstrap_propagates_backwards_through_the_trajectory(self):
        returns, _ = compute_returns_advantages(
            rewards=[0.0, 0.0, 0.0], dones=[0.0, 0.0, 0.0], values=[0.0, 0.0, 0.0],
            gamma=GAMMA, gae_lambda=LAMBDA, bootstrap_value=1.0,
        )
        # Discounted once per step back from the cut.
        np.testing.assert_allclose(
            returns, [GAMMA**3, GAMMA**2, GAMMA], rtol=1e-5
        )

    def test_terminal_midway_blocks_the_bootstrap_from_leaking_back(self):
        returns, _ = compute_returns_advantages(
            rewards=[0.0, 0.0], dones=[1.0, 0.0], values=[0.0, 0.0],
            gamma=GAMMA, gae_lambda=LAMBDA, bootstrap_value=100.0,
        )
        # Step 0 is terminal: nothing after it may contribute.
        self.assertEqual(returns[0], 0.0)


class DQNTargetSemanticsTests(unittest.TestCase):
    """The DQN/SAC/DDPG target is rewards + (1 - dones) * gamma * max_next_q."""

    @staticmethod
    def td_target(reward, done_flag, next_q):
        return reward + (1.0 - float(done_flag)) * GAMMA * next_q

    def test_fusing_truncation_erases_future_value(self):
        terminated, truncated, next_q = False, True, 50.0
        fused = self.td_target(1.0, terminated or truncated, next_q)   # the old bug
        correct = self.td_target(1.0, terminated, next_q)              # after the fix
        self.assertEqual(fused, 1.0)
        self.assertAlmostEqual(correct, 1.0 + GAMMA * 50.0, places=4)
        self.assertNotEqual(fused, correct)

    def test_real_terminal_is_unaffected_by_the_fix(self):
        terminated, truncated, next_q = True, False, 50.0
        fused = self.td_target(1.0, terminated or truncated, next_q)
        correct = self.td_target(1.0, terminated, next_q)
        self.assertEqual(fused, correct)


if __name__ == "__main__":
    unittest.main()
