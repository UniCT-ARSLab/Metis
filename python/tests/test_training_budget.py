import unittest
from argparse import Namespace

from core.training import TrainingBudget, validate_transition_snapshot_arguments


class TrainingBudgetTests(unittest.TestCase):
    def test_zero_limit_counts_without_stopping(self):
        budget = TrainingBudget(0)

        self.assertFalse(budget.consume(100))
        self.assertEqual(budget.collected, 100)
        self.assertFalse(budget.exhausted)

    def test_positive_limit_stops_when_reached_or_exceeded(self):
        budget = TrainingBudget(10)

        self.assertFalse(budget.consume(4))
        self.assertTrue(budget.consume(6))
        self.assertTrue(budget.exhausted)
        self.assertTrue(budget.consume(2))
        self.assertEqual(budget.collected, 12)

    def test_negative_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            TrainingBudget(-1)


class TransitionSnapshotArgumentTests(unittest.TestCase):
    @staticmethod
    def args(**changes):
        values = {
            "checkpoint_every_transitions": 10,
            "transition_snapshot_dir": "/tmp/snapshots",
            "total_timesteps": 100,
            "collector_mode": "sync",
            "multi_agent": False,
            "multi_policy": False,
            "resume": False,
            "resume_checkpoint": None,
            "policy_path": None,
            "initial_weights_path": None,
        }
        values.update(changes)
        return Namespace(**values)

    def test_valid_fresh_sync_single_policy_run(self):
        self.assertTrue(validate_transition_snapshot_arguments(self.args()))

    def test_disabled_rejects_dangling_output_directory(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_transition_snapshot_arguments(
                self.args(checkpoint_every_transitions=0)
            )

    def test_rejects_resume_async_and_multi_agent(self):
        for changes in (
            {"resume": True},
            {"collector_mode": "async"},
            {"multi_agent": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_transition_snapshot_arguments(self.args(**changes))


if __name__ == "__main__":
    unittest.main()
