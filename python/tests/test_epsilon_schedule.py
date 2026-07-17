import sys
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_generic_dqn import (  # noqa: E402
    EPSILON_DECAY_FLOOR,
    episodes_to_reach_epsilon,
    resolve_epsilon_decay,
)


def make_args(**overrides):
    args = Namespace(
        num_episodes=2500,
        epsilon_min=0.0,
        epsilon_decay=None,
        epsilon_decay_horizon_fraction=0.5,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class ResolveEpsilonDecayTests(unittest.TestCase):
    def test_pinned_decay_is_returned_unchanged(self):
        args = make_args(epsilon_decay=0.9985)
        self.assertEqual(resolve_epsilon_decay(args, 1.0, 0), 0.9985)

    def test_derived_decay_reaches_floor_at_the_horizon(self):
        args = make_args(num_episodes=2500, epsilon_min=0.0)
        decay = resolve_epsilon_decay(args, 1.0, 0)
        # epsilon-min is 0, so the schedule anneals to the floor instead.
        self.assertEqual(episodes_to_reach_epsilon(1.0, EPSILON_DECAY_FLOOR, decay), 1250)

    def test_derived_decay_honours_a_nonzero_epsilon_min(self):
        args = make_args(num_episodes=1000, epsilon_min=0.1)
        decay = resolve_epsilon_decay(args, 1.0, 0)
        self.assertEqual(episodes_to_reach_epsilon(1.0, 0.1, decay), 500)

    def test_resume_horizon_spans_only_the_remaining_episodes(self):
        args = make_args(num_episodes=2000, epsilon_min=0.0)
        # Resuming at 1500 leaves 500 episodes, so the anneal must finish within 250 of
        # them -- not stretch over the full budget as a parse-time constant would.
        decay = resolve_epsilon_decay(args, 0.5, 1500)
        self.assertEqual(episodes_to_reach_epsilon(0.5, EPSILON_DECAY_FLOOR, decay), 250)

    def test_start_already_at_or_below_target_never_decays(self):
        args = make_args(epsilon_min=0.5)
        self.assertEqual(resolve_epsilon_decay(args, 0.5, 0), 1.0)

    def test_horizon_fraction_of_one_spends_the_whole_budget(self):
        args = make_args(num_episodes=800, epsilon_min=0.0, epsilon_decay_horizon_fraction=1.0)
        decay = resolve_epsilon_decay(args, 1.0, 0)
        self.assertEqual(episodes_to_reach_epsilon(1.0, EPSILON_DECAY_FLOOR, decay), 800)

    def test_exhausted_budget_still_yields_a_usable_decay(self):
        # start_episode past num_episodes must not produce a zero/negative horizon.
        args = make_args(num_episodes=100)
        decay = resolve_epsilon_decay(args, 1.0, 500)
        self.assertGreater(decay, 0.0)
        self.assertLess(decay, 1.0)


class EpisodesToReachEpsilonTests(unittest.TestCase):
    def test_returns_zero_when_already_below_target(self):
        self.assertEqual(episodes_to_reach_epsilon(0.01, 0.1, 0.99), 0)

    def test_returns_none_when_decay_never_converges(self):
        self.assertIsNone(episodes_to_reach_epsilon(1.0, 0.1, 1.0))

    def test_matches_a_hand_computed_schedule(self):
        # 0.5**n reaches 0.125 at n=3.
        self.assertEqual(episodes_to_reach_epsilon(1.0, 0.125, 0.5), 3)


if __name__ == "__main__":
    unittest.main()
