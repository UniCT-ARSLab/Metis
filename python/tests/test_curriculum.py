import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.curriculum import AdaptiveCurriculumController, scenario_curriculum_config


def curriculum_args(checkpoint_dir, **overrides):
    values = {
        "adaptive_curriculum": True,
        "curriculum_initial_level": 0.0,
        "curriculum_level_step": 0.1,
        "curriculum_promotion_metric": "success_rate",
        "curriculum_promotion_threshold": 0.7,
        "curriculum_promotion_evaluations": 2,
        "curriculum_min_policy_updates": 100,
        "curriculum_demotion_threshold": None,
        "curriculum_demotion_evaluations": 3,
        "checkpoint_dir": str(checkpoint_dir),
        "resume": False,
        "resume_checkpoint": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class AdaptiveCurriculumTests(unittest.TestCase):
    def test_promotes_only_after_consecutive_frozen_evaluations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AdaptiveCurriculumController(curriculum_args(temp_dir))

            self.assertFalse(
                controller.observe_evaluation(100, {"success_rate": 0.8})
            )
            self.assertEqual(controller.level, 0.0)
            self.assertTrue(
                controller.observe_evaluation(200, {"success_rate": 0.75})
            )
            self.assertAlmostEqual(controller.level, 0.1)

    def test_failed_evaluation_resets_confirmation_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AdaptiveCurriculumController(curriculum_args(temp_dir))

            controller.observe_evaluation(100, {"success_rate": 0.8})
            controller.observe_evaluation(200, {"success_rate": 0.5})
            controller.observe_evaluation(300, {"success_rate": 0.8})

            self.assertEqual(controller.confirmations, 1)
            self.assertEqual(controller.level, 0.0)

    def test_assisted_success_does_not_promote_regular_curriculum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AdaptiveCurriculumController(
                curriculum_args(temp_dir, curriculum_promotion_evaluations=1)
            )

            promoted = controller.observe_evaluation(
                100,
                {
                    "success_rate": 0.8,
                    "selection_success_rate": 0.2,
                    "training_updates": 100,
                },
            )

            self.assertFalse(promoted)
            self.assertEqual(controller.level, 0.0)
            self.assertEqual(controller.confirmations, 0)

    def test_policy_without_optimizer_updates_cannot_promote(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AdaptiveCurriculumController(
                curriculum_args(temp_dir, curriculum_promotion_evaluations=1)
            )

            self.assertFalse(
                controller.observe_evaluation(
                    100,
                    {"success_rate": 1.0, "training_updates": 99},
                )
            )
            self.assertEqual(controller.confirmations, 0)
            self.assertEqual(controller.level, 0.0)

            self.assertTrue(
                controller.observe_evaluation(
                    200,
                    {"success_rate": 1.0, "training_updates": 100},
                )
            )
            self.assertEqual(controller.level, 0.1)

    def test_resume_restores_persistent_level(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AdaptiveCurriculumController(
                curriculum_args(temp_dir, curriculum_promotion_evaluations=1)
            )
            controller.observe_evaluation(100, {"success_rate": 1.0})

            restored = AdaptiveCurriculumController(
                curriculum_args(
                    temp_dir,
                    resume=True,
                    curriculum_promotion_evaluations=1,
                )
            )

            self.assertAlmostEqual(restored.level, 0.1)
            payload = json.loads(
                (Path(temp_dir) / "curriculum_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertAlmostEqual(payload["level"], 0.1)

    def test_scenario_config_uses_shared_controller(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = curriculum_args(temp_dir)
            args._adaptive_curriculum_controller = AdaptiveCurriculumController(args)

            self.assertEqual(
                scenario_curriculum_config(args),
                {"curriculum_level": 0.0},
            )

    def test_optional_demotion_requires_consecutive_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AdaptiveCurriculumController(
                curriculum_args(
                    temp_dir,
                    curriculum_initial_level=0.4,
                    curriculum_demotion_threshold=0.4,
                    curriculum_demotion_evaluations=2,
                )
            )

            self.assertFalse(
                controller.observe_evaluation(100, {"success_rate": 0.3})
            )
            self.assertAlmostEqual(controller.level, 0.4)
            self.assertEqual(controller.demotion_confirmations, 1)

            self.assertFalse(
                controller.observe_evaluation(200, {"success_rate": 0.2})
            )
            self.assertAlmostEqual(controller.level, 0.3)
            self.assertEqual(controller.last_transition, "demoted")
            self.assertEqual(controller.last_demotion_episode, 200)

    def test_demotion_confirmation_resets_above_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AdaptiveCurriculumController(
                curriculum_args(
                    temp_dir,
                    curriculum_initial_level=0.4,
                    curriculum_demotion_threshold=0.4,
                    curriculum_demotion_evaluations=2,
                )
            )

            controller.observe_evaluation(100, {"success_rate": 0.3})
            controller.observe_evaluation(200, {"success_rate": 0.5})
            controller.observe_evaluation(300, {"success_rate": 0.3})

            self.assertAlmostEqual(controller.level, 0.4)
            self.assertEqual(controller.demotion_confirmations, 1)

    def test_demotion_threshold_must_leave_hysteresis_gap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "must be lower"):
                AdaptiveCurriculumController(
                    curriculum_args(
                        temp_dir,
                        curriculum_promotion_threshold=0.7,
                        curriculum_demotion_threshold=0.7,
                    )
                )

    def _m4_args(self, temp_dir, **extra):
        return curriculum_args(
            temp_dir,
            curriculum_min_policy_updates=300,
            curriculum_promotion_threshold=0.90,
            curriculum_promotion_evaluations=3,
            curriculum_level_step=0.2,
            curriculum_stage_names="A,B,C,D,E,F",
            **extra,
        )

    def test_promotion_frozen_until_new_updates_since_transition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            c = AdaptiveCurriculumController(self._m4_args(temp_dir))
            promoted = False
            for u in (300, 301, 302):
                promoted = c.observe_evaluation(
                    u, {"success_rate": 0.95, "training_updates": u})
            self.assertTrue(promoted)
            self.assertAlmostEqual(c.level, 0.2)
            self.assertEqual(c.policy_updates_at_last_transition, 302)
            # cooldown: 3x0.95 within <300 NEW updates must not promote
            for u in (303, 400, 500):
                self.assertFalse(c.observe_evaluation(
                    u, {"success_rate": 0.95, "training_updates": u}))
            self.assertAlmostEqual(c.level, 0.2)
            # 300 new updates accrued -> 3 more qualifying evals promote again
            self.assertFalse(c.observe_evaluation(
                602, {"success_rate": 0.95, "training_updates": 602}))
            self.assertFalse(c.observe_evaluation(
                604, {"success_rate": 0.95, "training_updates": 604}))
            self.assertTrue(c.observe_evaluation(
                606, {"success_rate": 0.95, "training_updates": 606}))
            self.assertAlmostEqual(c.level, 0.4)

    def test_demotion_returns_to_exact_previous_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            c = AdaptiveCurriculumController(self._m4_args(
                temp_dir, curriculum_initial_level=0.6,
                curriculum_demotion_threshold=0.70,
                curriculum_demotion_evaluations=3))
            for u in (100, 200, 300):
                c.observe_evaluation(
                    u, {"success_rate": 0.5, "training_updates": u})
            self.assertAlmostEqual(c.level, 0.4)  # exact 0.6-0.2, on the stage grid
            self.assertEqual(round(c.level / 0.2), 2)

    def test_cooldown_and_level_persist_across_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            c = AdaptiveCurriculumController(self._m4_args(temp_dir))
            for u in (300, 301, 302):
                c.observe_evaluation(
                    u, {"success_rate": 0.95, "training_updates": u})
            self.assertAlmostEqual(c.level, 0.2)
            resumed = AdaptiveCurriculumController(
                self._m4_args(temp_dir, resume=True))
            self.assertAlmostEqual(resumed.level, 0.2)
            self.assertEqual(resumed.policy_updates_at_last_transition, 302)
            # still in cooldown after resume
            self.assertFalse(resumed.observe_evaluation(
                400, {"success_rate": 0.95, "training_updates": 400}))
            self.assertAlmostEqual(resumed.level, 0.2)

    def test_snapshot_exposes_stage_display_fields_fresh_and_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            c = AdaptiveCurriculumController(
                self._m4_args(temp_dir, curriculum_initial_level=0.4))
            snap = c.snapshot()
            self.assertEqual(snap["stage_index"], 2)
            self.assertEqual(snap["stage_name"], "C")
            self.assertEqual(snap["promotion_required"], 3)
            self.assertIn("cooldown_active", snap)
            self.assertIn("updates_since_last_transition", snap)
            # survive resume
            resumed = AdaptiveCurriculumController(
                self._m4_args(temp_dir, curriculum_initial_level=0.4, resume=True))
            rsnap = resumed.snapshot()
            self.assertEqual(rsnap["stage_index"], 2)
            self.assertEqual(rsnap["stage_name"], "C")

    def test_stage_name_is_none_without_task_config(self):
        # Framework owns only the generic stage_index; with no names configured the
        # controller must NOT invent an A/B/C nomenclature.
        with tempfile.TemporaryDirectory() as temp_dir:
            c = AdaptiveCurriculumController(
                curriculum_args(
                    temp_dir, curriculum_initial_level=0.4,
                    curriculum_level_step=0.2))
            snap = c.snapshot()
            self.assertEqual(snap["stage_index"], 2)
            self.assertIsNone(snap["stage_name"])

    def test_stage_names_come_from_task_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            c = AdaptiveCurriculumController(
                curriculum_args(
                    temp_dir, curriculum_initial_level=0.6,
                    curriculum_level_step=0.2,
                    curriculum_stage_names="Reach,Grasp,Lift,Place"))
            snap = c.snapshot()
            self.assertEqual(snap["stage_index"], 3)
            self.assertEqual(snap["stage_name"], "Place")
            # Index beyond the supplied names falls back to None, never crashes.
            c.level = 1.0
            self.assertIsNone(c.snapshot()["stage_name"])


if __name__ == "__main__":
    unittest.main()
