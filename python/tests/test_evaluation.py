import json
import tempfile
import unittest
from pathlib import Path

from core.evaluation import (
    build_evaluation_summary,
    classify_episode_failure,
    episode_reset_modes,
    episode_reset_values,
    finalize_episode_agent_diagnostics,
    new_episode_agent_diagnostics,
    update_episode_agent_diagnostics,
    write_evaluation_summary,
)


class EvaluationSummaryTests(unittest.TestCase):
    def test_optional_task_diagnostics_are_aggregated(self):
        summary = build_evaluation_summary(
            rewards=[1.0, 3.0],
            steps=[10, 20],
            successes=0,
            trials=2,
            diagnostics={
                "progress_mean": [0.25, 0.75],
                "progress_max": [0.50, 0.90],
                "position_error_mean": [0.40, 0.20],
                "position_error_min": [0.10, 0.05],
                "orientation_error_mean": [60.0, 20.0],
                "orientation_error_min": [15.0, 5.0],
                "hold_frames_max": [2, 12],
            },
        )

        self.assertAlmostEqual(summary["progress_mean"], 0.5)
        self.assertAlmostEqual(summary["progress_max"], 0.9)
        self.assertAlmostEqual(summary["position_error_mean"], 0.3)
        self.assertAlmostEqual(summary["position_error_min"], 0.05)
        self.assertAlmostEqual(summary["orientation_error_mean"], 40.0)
        self.assertAlmostEqual(summary["orientation_error_min"], 5.0)
        self.assertEqual(summary["hold_frames_max"], 12.0)

    def test_diagnostics_are_written_to_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            summary = build_evaluation_summary(
                rewards=[1.0],
                steps=[10],
                successes=0,
                trials=1,
                diagnostics={"progress_mean": [0.6]},
            )

            write_evaluation_summary(path, summary)

            payload = json.loads(path.read_text())
            self.assertAlmostEqual(payload["progress_mean"], 0.6)

    def test_regular_resets_drive_selection_metrics(self):
        summary = build_evaluation_summary(
            rewards=[10.0, -2.0],
            steps=[20, 100],
            successes=1,
            trials=2,
            diagnostics={"progress_mean": [1.0, 0.2]},
            reset_outcomes={
                "bootstrap": [
                    {"success": True, "reward": 10.0, "progress_mean": 1.0}
                ],
                "regular": [
                    {"success": False, "reward": -2.0, "progress_mean": 0.2}
                ],
            },
        )

        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["selection_success_rate"], 0.0)
        self.assertAlmostEqual(summary["regular_reward_mean"], -2.0)
        self.assertAlmostEqual(summary["regular_progress_mean"], 0.2)
        self.assertAlmostEqual(
            summary["reset_modes"]["bootstrap"]["success_rate"],
            1.0,
        )

    def test_reset_modes_are_read_per_agent(self):
        single_info = {
            "agent_info": {"reset": {"task_reset_mode": "regular"}},
        }
        multi_info = {
            "per_agent_infos": [
                {"reset": {"task_reset_mode": "bootstrap"}},
                {"reset": {"task_reset_mode": "regular"}},
            ],
        }

        self.assertEqual(episode_reset_modes(single_info), ["regular"])
        self.assertEqual(
            episode_reset_modes(multi_info, multi_agent=True),
            ["bootstrap", "regular"],
        )

    def test_worst_target_region_drives_selection_success(self):
        summary = build_evaluation_summary(
            rewards=[1.0, 1.0, 1.0],
            steps=[10, 10, 10],
            successes=2,
            trials=3,
            reset_outcomes={
                "regular": [
                    {"success": True},
                    {"success": True},
                    {"success": False},
                ],
                "target_region:easy": [
                    {"success": True},
                ],
                "target_region:medium": [
                    {"success": True},
                ],
                "target_region:hard": [
                    {"success": False},
                ],
            },
        )

        self.assertAlmostEqual(summary["regular_success_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["region_success_floor"], 0.0)
        self.assertAlmostEqual(summary["selection_success_rate"], 0.0)
        self.assertAlmostEqual(
            summary["target_regions"]["easy"]["success_rate"], 1.0)

    def test_reset_metadata_values_are_read_per_agent(self):
        info = {
            "per_agent_infos": [
                {"reset": {"target_region": "Easy"}},
                {"reset": {"target_region": "Hard"}},
            ],
        }

        self.assertEqual(
            episode_reset_values(
                info, "target_region", multi_agent=True),
            ["easy", "hard"],
        )

    def test_pose_failure_uses_simultaneous_gates(self):
        state = new_episode_agent_diagnostics()
        thresholds = {
            "distance_m": 0.06,
            "orientation_deg": 25.0,
            "max_joint_speed_rad_s": 0.30,
            "hold_physics_frames": 20,
            "require_still": True,
        }
        update_episode_agent_diagnostics(
            state,
            {
                "position_error_m": 0.04,
                "orientation_error_deg": 40.0,
                "max_joint_speed": 0.10,
                "hold_frames": 0,
                "success_thresholds": thresholds,
            },
        )
        update_episode_agent_diagnostics(
            state,
            {
                "position_error_m": 0.04,
                "orientation_error_deg": 20.0,
                "max_joint_speed": 0.50,
                "hold_frames": 0,
                "success_thresholds": thresholds,
            },
        )
        update_episode_agent_diagnostics(
            state,
            {
                "position_error_m": 0.04,
                "orientation_error_deg": 20.0,
                "max_joint_speed": 0.20,
                "hold_frames": 12,
                "success_thresholds": thresholds,
            },
        )

        record = finalize_episode_agent_diagnostics(state)
        record["success"] = False

        self.assertTrue(record["position_gate_reached"])
        self.assertTrue(record["pose_gate_reached"])
        self.assertTrue(record["stillness_gate_reached"])
        self.assertEqual(classify_episode_failure(record), "hold_gate")

    def test_pose_failure_does_not_mix_metrics_from_different_steps(self):
        state = new_episode_agent_diagnostics()
        thresholds = {
            "distance_m": 0.06,
            "orientation_deg": 25.0,
            "max_joint_speed_rad_s": 0.30,
            "hold_physics_frames": 20,
        }
        update_episode_agent_diagnostics(
            state,
            {
                "position_error_m": 0.04,
                "orientation_error_deg": 80.0,
                "max_joint_speed": 0.10,
                "success_thresholds": thresholds,
            },
        )
        update_episode_agent_diagnostics(
            state,
            {
                "position_error_m": 0.20,
                "orientation_error_deg": 10.0,
                "max_joint_speed": 0.10,
                "success_thresholds": thresholds,
            },
        )

        record = finalize_episode_agent_diagnostics(state)
        record["success"] = False

        self.assertTrue(record["position_gate_reached"])
        self.assertFalse(record["pose_gate_reached"])
        self.assertEqual(classify_episode_failure(record), "orientation_gate")

    def test_target_cells_and_failure_reasons_are_aggregated(self):
        common = {
            "target_region": "easy",
            "success_thresholds": {"distance_m": 0.06},
            "position_error_mean": 0.10,
        }
        summary = build_evaluation_summary(
            rewards=[1.0, -1.0, -2.0],
            steps=[10, 20, 30],
            successes=1,
            trials=3,
            episode_records=[
                {
                    **common,
                    "target_cell": "0",
                    "success": True,
                    "reward": 1.0,
                },
                {
                    **common,
                    "target_cell": "0",
                    "success": False,
                    "reward": -1.0,
                    "position_gate_reached": False,
                },
                {
                    **common,
                    "target_cell": "1",
                    "success": False,
                    "reward": -2.0,
                    "collided": True,
                },
            ],
        )

        self.assertEqual(summary["failure_reasons"], {
            "position_gate": 1,
            "collision": 1,
        })
        self.assertAlmostEqual(
            summary["target_cells"]["easy"]["0"]["success_rate"], 0.5
        )
        self.assertEqual(
            summary["target_cells"]["easy"]["1"]["failure_reasons"],
            {"collision": 1},
        )


if __name__ == "__main__":
    unittest.main()
