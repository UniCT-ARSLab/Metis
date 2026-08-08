"""Target-network refresh cadence for DQN, in episodes or in transitions.

Episodes are only a sensible unit while their length is roughly constant. On Breakout they grew from
57 to 935 steps inside one run, which moved the real interval between target syncs from 284 to 4345
gradient updates without anyone touching a flag. Counting transitions pins it where it was set.

The property that matters most here is the boring one: with --target-update-steps unset, nothing
changes for any existing run.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training import TargetSyncSchedule  # noqa: E402


class Args:
    def __init__(self, **kwargs):
        self.target_update_every = 20
        self.target_update_steps = 0
        self.__dict__.update(kwargs)


class EpisodeModeTests(unittest.TestCase):
    def test_is_the_default(self):
        schedule = TargetSyncSchedule(Args())
        self.assertFalse(schedule.uses_steps)
        self.assertEqual(schedule.describe(), "every 20 episodes")

    def test_fires_on_every_nth_episode(self):
        schedule = TargetSyncSchedule(Args(target_update_every=3))
        fired = [episode for episode in range(1, 11) if schedule.due_after_episode(episode)]
        self.assertEqual(fired, [3, 6, 9])

    def test_transitions_never_fire_in_episode_mode(self):
        # The two units must not both drive a sync, or the interval silently halves.
        schedule = TargetSyncSchedule(Args(target_update_every=3))
        self.assertFalse(any(schedule.due_after_transitions(10_000) for _ in range(10)))


class StepModeTests(unittest.TestCase):
    def test_fires_once_per_interval_regardless_of_chunking(self):
        # Transitions arrive in whatever batches the collector drains, so the schedule has to carry
        # credit across calls instead of testing each batch on its own.
        for chunk in (1, 3, 7, 100):
            with self.subTest(chunk=chunk):
                schedule = TargetSyncSchedule(Args(target_update_steps=100))
                fired = 0
                for _ in range(1000 // chunk):
                    if schedule.due_after_transitions(chunk):
                        fired += 1
                self.assertEqual(fired, 1000 // chunk * chunk // 100)

    def test_a_single_huge_batch_fires_once_not_once_per_interval(self):
        # A drain larger than the interval still means one sync: syncing repeatedly inside one batch
        # would copy identical weights several times for no reason.
        schedule = TargetSyncSchedule(Args(target_update_steps=100))
        self.assertTrue(schedule.due_after_transitions(1000))
        self.assertFalse(schedule.due_after_transitions(1))

    def test_leftover_credit_carries_over(self):
        schedule = TargetSyncSchedule(Args(target_update_steps=100))
        self.assertFalse(schedule.due_after_transitions(99))
        self.assertTrue(schedule.due_after_transitions(1))

    def test_episodes_never_fire_in_step_mode(self):
        schedule = TargetSyncSchedule(Args(target_update_steps=100, target_update_every=1))
        self.assertFalse(any(schedule.due_after_episode(e) for e in range(1, 20)))

    def test_describe_reports_the_active_unit(self):
        self.assertEqual(
            TargetSyncSchedule(Args(target_update_steps=10_000)).describe(),
            "every 10000 transitions")

    def test_a_negative_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            TargetSyncSchedule(Args(target_update_steps=-1))


class InvariantTests(unittest.TestCase):
    def test_the_interval_does_not_drift_when_episodes_lengthen(self):
        """The whole point: same setting, same spacing, whatever happens to episode length.

        Replays Breakout's own growth -- 57-step episodes early, 935-step episodes late -- and checks
        the transition spacing between syncs stays put, which the episode-based schedule cannot do.
        """
        step_gaps, episode_gaps = [], []
        for length in (57, 935):
            # A fresh schedule per phase: carrying credit across an abrupt change in episode length
            # produces one short gap at the seam, which is correct behaviour but an artefact of
            # switching instantly rather than growing, as a real run does.
            by_steps = TargetSyncSchedule(Args(target_update_steps=10_000))
            by_episodes = TargetSyncSchedule(Args(target_update_every=20))
            transitions_since = 0
            for _episode in range(1, 401):
                transitions_since += length
                if by_steps.due_after_transitions(length):
                    step_gaps.append(transitions_since)
                    transitions_since = 0
            transitions_since = 0
            for episode in range(1, 401):
                transitions_since += length
                if by_episodes.due_after_episode(episode):
                    episode_gaps.append(transitions_since)
                    transitions_since = 0
        # Step mode: every gap within one episode's worth of the target.
        self.assertTrue(all(abs(gap - 10_000) <= 935 for gap in step_gaps), step_gaps)
        # Episode mode: the short-episode gaps and the long-episode ones differ by an order of
        # magnitude. This assertion documents the defect, it is not an endorsement.
        self.assertGreater(max(episode_gaps) / min(episode_gaps), 10)


if __name__ == "__main__":
    unittest.main()
