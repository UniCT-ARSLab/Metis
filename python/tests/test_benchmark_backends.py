import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.benchmark_backends import (
    _aggregate,
    _common_train_command,
    _first_number,
    _success_rate,
    parse_args,
    summarize_learning_curve,
)


class BenchmarkBackendsTests(unittest.TestCase):
    def test_metric_parser_accepts_native_reward_formats(self):
        self.assertEqual(_first_number("-2.5 [-5.0, 1.0]"), -2.5)
        self.assertEqual(_first_number("[[1.0, 3.0]]"), 2.0)
        self.assertAlmostEqual(_success_rate({"success": "3/4 (75.00%)"}), 0.75)

    def test_learning_curve_uses_transition_axis(self):
        rows = [
            {"total_timesteps": 10, "reward_mean": -2.0, "success_rate": 0.0, "recorded_at": 5.0},
            {"total_timesteps": 20, "reward_mean": 2.0, "success_rate": 0.8, "recorded_at": 8.0},
        ]

        summary = summarize_learning_curve(rows, success_threshold=0.5)

        self.assertEqual(summary["last_logged_timestep"], 20)
        self.assertAlmostEqual(summary["learning_reward_auc"], 0.0)
        self.assertAlmostEqual(summary["time_to_success_threshold_seconds"], 3.0)

    def test_aggregate_keeps_backends_separate(self):
        records = [
            {
                "backend": backend,
                "algorithm": "dqn",
                "status": "completed",
                "wall_seconds": wall,
                "transitions_per_second": 100.0 / wall,
                "evaluation": {"reward_mean": reward, "success_rate": success, "steps_mean": 10},
                "learning_curve": {
                    "learning_reward_auc": reward,
                    "time_to_success_threshold_seconds": None,
                },
            }
            for backend, wall, reward, success in [
                ("metis", 10.0, 1.0, 0.5),
                ("sb3", 8.0, 2.0, 0.75),
            ]
        ]

        summary = _aggregate(records)

        self.assertEqual([row["backend"] for row in summary], ["metis", "sb3"])
        self.assertEqual(summary[1]["reward_mean_mean"], 2.0)

    def test_training_command_uses_backend_specific_interpreter(self):
        args = SimpleNamespace(
            metis_python="/tmp/metis-python",
            sb3_python="/tmp/sb3-python",
            algorithm="dqn",
            godot_bin="/tmp/godot",
            godot_scene="res://scenario.tscn",
            num_envs=1,
            multi_agent=False,
            sb3_multi_agent_partial_done="error",
            base_port=6200,
            total_timesteps=100,
            max_steps_per_episode=50,
            render_mode="cpu",
            headless=True,
            godot_project=None,
            trainer_args=[],
            evaluation_episodes=5,
            evaluation_seed=1000,
            evaluation_max_steps=50,
        )

        command = _common_train_command(
            args,
            Path("/tmp/repo"),
            "sb3",
            123,
            Path("/tmp/run"),
        )

        self.assertEqual(command[0], "/tmp/sb3-python")

    def test_separator_is_not_forwarded_to_trainer(self):
        args = parse_args(
            [
                "--name",
                "comparison",
                "--algorithm",
                "dqn",
                "--total-timesteps",
                "100",
                "--godot-bin",
                "/tmp/godot",
                "--godot-scene",
                "res://scenario.tscn",
                "--",
                "--batch-size",
                "64",
            ]
        )

        self.assertEqual(args.trainer_args, ["--batch-size", "64"])

    def test_multi_agent_is_a_controlled_benchmark_option(self):
        args = parse_args(
            [
                "--name",
                "pong",
                "--algorithm",
                "dqn",
                "--total-timesteps",
                "100",
                "--godot-bin",
                "/tmp/godot",
                "--godot-scene",
                "res://pong.tscn",
                "--multi-agent",
            ]
        )

        self.assertTrue(args.multi_agent)
        self.assertEqual(args.sb3_multi_agent_partial_done, "error")


if __name__ == "__main__":
    unittest.main()
