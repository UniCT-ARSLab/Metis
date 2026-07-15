import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


from training_support import BestCheckpointTracker, PolicyEvaluationResult


def tracker_args(checkpoint_dir, **overrides):
    values = {
        "best_checkpoint": True,
        "best_checkpoint_dir": None,
        "keep_best_checkpoints": 3,
        "best_evaluation_every": 100,
        "best_evaluation_episodes": 20,
        "best_evaluation_seed": 10_000,
        "best_evaluation_training_episode": None,
        "best_evaluation_max_steps": None,
        "best_evaluation_port": None,
        "best_evaluation_timeout": 30.0,
        "best_evaluation_device": "cpu",
        "best_metric": "auto",
        "checkpoint_dir": str(checkpoint_dir),
        "base_port": 6200,
        "num_envs": 4,
        "num_episodes": 500,
        "max_steps_per_episode": 0,
        "multi_agent": False,
        "godot_bin": "/godot",
        "godot_project": "godot",
        "godot_scene": "res://scenario.tscn",
        "agent_id": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def evaluation(episode, success_rate, reward_mean):
    return PolicyEvaluationResult(
        episode,
        {
            "episodes": 20,
            "successes": int(success_rate * 20),
            "trials": 20,
            "success_rate": success_rate,
            "reward_mean": reward_mean,
            "reward_min": reward_mean,
            "reward_max": reward_mean,
            "steps_mean": 100.0,
            "steps_min": 50,
            "steps_max": 150,
        },
    )


class BestCheckpointTrackerTests(unittest.TestCase):
    def test_evaluation_schedule(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(tracker_args(temp_dir), "dqn")

            self.assertFalse(tracker.should_evaluate(99))
            self.assertTrue(tracker.should_evaluate(100))
            self.assertFalse(tracker.should_evaluate(101))

    def test_auto_metric_prioritizes_success_then_reward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(tracker_args(temp_dir), "dqn")
            baseline = evaluation(100, success_rate=0.10, reward_mean=50.0)
            tracker.record_best(baseline, "best/ckpt-100")

            self.assertFalse(tracker.is_improvement(evaluation(200, 0.05, 100.0)))
            self.assertTrue(tracker.is_improvement(evaluation(200, 0.10, 51.0)))
            self.assertTrue(tracker.is_improvement(evaluation(200, 0.15, -10.0)))

    def test_reward_metric_prioritizes_reward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = tracker_args(temp_dir, best_metric="reward_mean")
            tracker = BestCheckpointTracker(args, "sac")
            tracker.record_best(evaluation(100, 0.50, 10.0), "best/ckpt-100")

            self.assertTrue(tracker.is_improvement(evaluation(200, 0.10, 11.0)))

    def test_metadata_is_restored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = tracker_args(temp_dir)
            tracker = BestCheckpointTracker(args, "ppo")
            tracker.record_best(evaluation(300, 0.25, 7.0), "best/ckpt-300")

            restored = BestCheckpointTracker(args, "ppo")

            self.assertEqual(restored.best_key, (0.25, 7.0))
            self.assertEqual(restored.best_summary["episode"], 300)
            self.assertTrue((Path(temp_dir) / "best" / "best_metrics.json").is_file())

    def test_unlimited_training_gets_a_finite_evaluation_watchdog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(
                tracker_args(temp_dir, max_steps_per_episode=0),
                "dqn",
            )

            command = tracker._evaluation_command("ckpt-100", Path(temp_dir) / "summary.json")

            max_steps_index = command.index("--max-steps") + 1
            self.assertEqual(command[max_steps_index], "10000")


if __name__ == "__main__":
    unittest.main()
