import unittest

from core.training import TrainingBudget


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


if __name__ == "__main__":
    unittest.main()
