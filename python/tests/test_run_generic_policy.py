import os
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("GODOT_GYM_TF_LD_READY", "1")

from run_generic_policy import (
    agent_succeeded,
    checkpoint_episode,
    normalize_checkpoint_path,
    summarize_episode_outcome,
    wait_for_realtime_tick,
)


class RunGenericPolicyTests(unittest.TestCase):
    def test_normalize_checkpoint_path_accepts_prefix_and_index_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_prefix = Path(temp_dir) / "ckpt-42"
            checkpoint_prefix.with_suffix(".index").touch()

            expected = str(checkpoint_prefix)
            self.assertEqual(normalize_checkpoint_path(expected), expected)
            self.assertEqual(normalize_checkpoint_path(f"{expected}.index"), expected)
            self.assertEqual(checkpoint_episode(expected), 42)

    def test_normalize_checkpoint_path_rejects_missing_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "Checkpoint index not found"):
            normalize_checkpoint_path("missing/ckpt-10")

    def test_success_is_detected_from_terminal_reason_or_event(self):
        self.assertTrue(agent_succeeded({"terminal_reason": "level_cleared"}))
        self.assertTrue(agent_succeeded({"events": {"finish_reached": 1.0}}))
        self.assertFalse(agent_succeeded({"terminal_reason": "life_lost"}))

    def test_multi_agent_outcome_counts_each_agent(self):
        info = {
            "per_agent_infos": [
                {"terminal_reason": "level_cleared"},
                {"terminal_reason": "life_lost"},
            ]
        }

        successes, trials, reasons = summarize_episode_outcome(info, multi_agent=True)

        self.assertEqual(successes, 1)
        self.assertEqual(trials, 2)
        self.assertEqual(reasons, ["level_cleared", "life_lost"])

    def test_realtime_pacing_waits_only_until_the_next_deadline(self):
        sleeps = []
        deadline = wait_for_realtime_tick(
            0.0,
            100.0,
            clock=lambda: 0.004,
            sleeper=sleeps.append,
        )
        self.assertAlmostEqual(deadline, 0.01)
        self.assertAlmostEqual(sleeps[0], 0.006)

        recovered = wait_for_realtime_tick(
            deadline,
            100.0,
            clock=lambda: 0.05,
            sleeper=sleeps.append,
        )
        self.assertAlmostEqual(recovered, 0.05)
        self.assertEqual(len(sleeps), 1)


if __name__ == "__main__":
    unittest.main()
