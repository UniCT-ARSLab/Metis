"""Async DQN drives the opponent pool; the trainers that cannot must still say so.

Wiring the pool into one trainer and leaving the guard off for the rest would turn a
loud, accurate error into a silent no-op -- worse than the gap it replaces.
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training import validate_async_arguments  # noqa: E402

TRAINER_DIR = Path(__file__).resolve().parents[1] / "algorithms"


def async_args(**overrides):
    values = {
        "collector_mode": "async",
        "max_steps_per_episode": 0,
        "async_policy_sync_steps": 100,
        "async_policy_publish_updates": 100,
        "async_updates_per_step": 1,
        "async_update_basis": "transitions",
        "async_update_every": 4,
        "async_max_updates_per_env_step": 1,
        "async_drain_max_events": 64,
        "async_queue_capacity": 256,
        "opponent_pool": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class OpponentPoolGuardTests(unittest.TestCase):
    def test_a_trainer_that_supports_the_pool_is_allowed(self):
        validate_async_arguments(
            async_args(opponent_pool=True), supports_opponent_pool=True
        )  # must not raise

    def test_a_trainer_that_does_not_support_the_pool_still_refuses(self):
        # Defaulting to False is deliberate: a trainer must opt in, so forgetting to
        # thread the pool through fails loudly instead of silently ignoring the flag.
        with self.assertRaises(ValueError) as caught:
            validate_async_arguments(async_args(opponent_pool=True))
        self.assertIn("sync", str(caught.exception), "the error should name the way out")

    def test_the_guard_is_irrelevant_without_the_flag(self):
        validate_async_arguments(async_args(opponent_pool=False))  # must not raise

    def test_sync_mode_is_never_blocked(self):
        validate_async_arguments(
            async_args(collector_mode="sync", opponent_pool=True)
        )  # must not raise


class TrainerWiringTests(unittest.TestCase):
    """Which trainers claim support must match which ones actually thread the pool."""

    def source(self, trainer):
        filename = "common.py" if trainer == "ddpg" else f"{trainer}.py"
        return (TRAINER_DIR / filename).read_text()

    def opts_in(self, trainer):
        source = self.source(trainer)
        return "supports_opponent_pool=True" in source

    def threads_the_pool_through_async(self, trainer):
        source = self.source(trainer)
        marker = source.split(f"def run_async_{trainer}", 1)
        if len(marker) < 2:
            return False
        return "opponent_pool" in marker[1][:4000]

    def test_claimed_support_matches_actual_wiring(self):
        for trainer in ("dqn", "ppo", "sac", "ddpg"):
            self.assertEqual(
                self.opts_in(trainer),
                self.threads_the_pool_through_async(trainer),
                msg=(
                    f"algorithms/{trainer}.py claims opponent-pool support "
                    f"({self.opts_in(trainer)}) but its async path wires it "
                    f"({self.threads_the_pool_through_async(trainer)})"
                ),
            )

    def test_dqn_is_the_one_that_supports_it_today(self):
        self.assertTrue(self.opts_in("dqn"))
        for trainer in ("ppo", "sac", "ddpg"):
            self.assertFalse(self.opts_in(trainer), f"{trainer} opted in without wiring")


if __name__ == "__main__":
    unittest.main()
