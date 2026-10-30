import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from core.training import BestCheckpointTracker, PolicyEvaluationResult, apply_ready_best_checkpoint
from core.training_control import TrainingControl, success_lower_bound
from core.training_health import RecoveryRequest, TensorFlowCheckpointRecovery
from test_best_checkpoint_tracker import tracker_args, evaluation


class Scalar:
    def __init__(self, value):
        self.value = value

    def assign(self, value):
        self.value = float(value)

    def numpy(self):
        return self.value


def optimizer(rate=1.):
    return SimpleNamespace(learning_rate=Scalar(rate))


def checkpoint_files(prefix, content=b"candidate"):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + ".index").write_bytes(content)
    Path(str(prefix) + ".data-00000-of-00001").write_bytes(content)
    return prefix


def cert_summary(successes=300, trials=300, **extras):
    return {"episodes": trials, "trials": trials, "successes": successes, "success_rate": successes / trials, "reward_mean": 1., "steps_mean": 2., **extras}


class ExactCertificationTests(unittest.TestCase):
    def test_all_successes_bound_matches_closed_form(self):
        self.assertAlmostEqual(success_lower_bound(300, 300, .025), .025 ** (1 / 300), places=12)
        self.assertEqual(success_lower_bound(0, 300, .025), 0)
        self.assertLess(success_lower_bound(19, 20, .025), .95)

    def test_nontrivial_bound_and_monotonicity(self):
        self.assertAlmostEqual(success_lower_bound(95, 100, .025), .8871650889, places=8)
        self.assertLess(success_lower_bound(90, 100, .025), success_lower_bound(95, 100, .025))
        self.assertLess(success_lower_bound(95, 100, .01), success_lower_bound(95, 100, .05))

    def test_confidence_rejects_an_observed_rate_at_the_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = TrainingControl(tracker_args(tmp, stop_when_solved=.95, solved_patience=1), "sac")
            request = control.observe(evaluation(1, 1., 1.), True)
            self.assertFalse(control.assess(cert_summary(285), request)["passed"])
            self.assertTrue(control.assess(cert_summary(), request)["passed"])

    def test_regions_must_all_pass_and_missing_regions_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = TrainingControl(tracker_args(tmp, stop_when_solved=.95, solved_patience=1), "sac")
            request = control.observe(evaluation(1, 1., 1., target_regions={"a": {}, "b": {}, "c": {}}), True)
            regions = {name: {"successes": 100, "trials": 100, "success_rate": 1.} for name in "abc"}
            self.assertTrue(control.assess(cert_summary(target_regions=regions), request)["passed"])
            del regions["c"]
            self.assertFalse(control.assess(cert_summary(target_regions=regions), request)["passed"])

    def test_partial_invalid_and_seeded_evaluations_never_certify(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = TrainingControl(tracker_args(tmp, stop_when_solved=.95, solved_patience=1), "sac")
            request = control.observe(evaluation(1, 1., 1.), True)
            for summary in (None, {}, cert_summary(20, 20), cert_summary(success_rate=float("nan")), cert_summary(reset_modes={"seeded": {"successes": 300, "trials": 300}})):
                with self.subTest(summary=summary):
                    self.assertFalse(control.assess(summary, request)["passed"])

    def test_retry_spends_less_alpha_and_never_reuses_deck_even_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracker_args(tmp, stop_when_solved=.95, solved_patience=1)
            first = TrainingControl(args, "sac")
            one = first.observe(evaluation(1, 1., 1.), True)
            first.record_certification(first.assess(cert_summary(0), one))
            args.resume = True
            second = TrainingControl(args, "sac")
            two = second.observe(evaluation(1, 1., 1.), True)
            self.assertGreaterEqual(two["seed"], one["seed"] + one["episodes"])
            self.assertLess(two["alpha"], one["alpha"])
            self.assertEqual(two["attempt"], 2)
            budget = sum(.05 / (index * (index + 1)) for index in range(1, 1000))
            self.assertLessEqual(budget, .05)

    def test_qualification_is_consecutive_and_duplicate_evaluations_do_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = TrainingControl(tracker_args(tmp, stop_when_solved=.95, solved_patience=2), "sac")
            self.assertIsNone(control.observe(evaluation(1, 1., 1.), True))
            self.assertIsNone(control.observe(evaluation(1, 1., 1.), True))
            control.evaluation_failed()
            self.assertIsNone(control.observe(evaluation(2, 1., 1.), True))
            self.assertIsNotNone(control.observe(evaluation(3, 1., 1.), False))

    def test_configuration_and_independence_guards(self):
        cases = [{"stop_when_solved": 1.}, {"stop_when_solved": float("nan")}, {"stop_when_solved": .95, "multi_agent": True}, {"stop_when_solved": .95, "multi_policy": True}, {"stop_when_solved": .95, "best_checkpoint": False}, {"stop_when_solved": .95, "solved_patience": 0}, {"stop_when_solved": .95, "solved_certification_seed": 10000}, {"stop_when_solved": .95, "solved_certification_seed": 1000}, {"stop_when_solved": .95, "solved_certification_confidence": 1.}, {"lr_decay_on_plateau": .3, "lr_decay_on_best": .3}, {"lr_decay_on_plateau": 0.}, {"lr_decay_on_best": float("inf")}, {"lr_min_scale": 0.}]
        for values in cases:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(values=values), self.assertRaises(ValueError):
                TrainingControl(tracker_args(tmp, **values), "sac")

    def test_retry_skips_selection_namespace_without_overlapping_previous_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracker_args(tmp, stop_when_solved=.95, solved_patience=1, solved_certification_seed=1000000000, best_evaluation_seed=1000000400)
            control = TrainingControl(args, "sac")
            first = control.observe(evaluation(1, 1., 1.), True)
            second = control.observe(evaluation(2, 1., 1.), True)
            third = control.observe(evaluation(3, 1., 1.), True)
            self.assertEqual(second["seed"], 1000000420)
            self.assertGreaterEqual(third["seed"], second["seed"] + second["episodes"])
            self.assertGreaterEqual(second["seed"], first["seed"] + first["episodes"])

    def test_exhausted_seed_space_continues_training_without_claiming_solved(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = TrainingControl(tracker_args(tmp, stop_when_solved=.95, solved_patience=1, solved_certification_seed=2**32 - 300), "sac")
            self.assertIsNotNone(control.observe(evaluation(1, 1., 1.), True))
            self.assertIsNone(control.observe(evaluation(2, 1., 1.), True))
            self.assertTrue(control.state["certification_exhausted"])
            self.assertFalse(control.stop_requested)


class LearningRateControlTests(unittest.TestCase):
    def test_default_disabled_creates_no_files_and_changes_no_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = TrainingControl(tracker_args(tmp), "sac")
            actor = optimizer()
            control.bind_optimizers([("actor", actor)])
            control.observe(evaluation(1, 1., 1.), True)
            self.assertEqual(actor.learning_rate.numpy(), 1.)
            self.assertFalse(control.state_path.exists())

    def test_plateau_decays_all_roles_respects_floor_and_does_not_decay_on_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = TrainingControl(tracker_args(tmp, lr_decay_on_plateau=.3, lr_plateau_patience=2, lr_min_scale=.1), "sac")
            actor, critic = optimizer(), optimizer(2.)
            control.bind_optimizers([("actor", actor), ("critic1", critic)])
            control.observe(evaluation(1, .5, 1.), True)
            control.observe(evaluation(2, .5, 1.), False)
            self.assertEqual(actor.learning_rate.numpy(), 1.)
            control.observe(evaluation(3, .5, 1.), False)
            self.assertAlmostEqual(actor.learning_rate.numpy(), .3)
            self.assertAlmostEqual(critic.learning_rate.numpy(), .6)
            for episode in range(4, 10):
                control.observe(evaluation(episode, .5, 1.), False)
            self.assertAlmostEqual(actor.learning_rate.numpy(), .1)
            self.assertAlmostEqual(critic.learning_rate.numpy(), .2)

    def test_on_best_is_separate_opt_in_and_rates_survive_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracker_args(tmp, lr_decay_on_best=.5)
            control = TrainingControl(args, "ppo")
            first = optimizer()
            control.bind_optimizers([("policy", first)])
            control.observe(evaluation(1, .5, 1.), True)
            control.observe(evaluation(2, .5, 1.), False)
            self.assertAlmostEqual(first.learning_rate.numpy(), .5)
            args.resume = True
            resumed = TrainingControl(args, "ppo")
            second = optimizer()
            resumed.bind_optimizers([("policy", second)])
            self.assertAlmostEqual(second.learning_rate.numpy(), .5)
            resumed.observe(evaluation(3, .6, 1.), True)
            self.assertAlmostEqual(second.learning_rate.numpy(), .25)

    def test_cumulative_recovery_does_not_raise_lr_after_restoring_old_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracker_args(tmp, recovery_lr_cumulative=True)
            control = TrainingControl(args, "sac")
            actor = optimizer()
            control.bind_optimizers([("actor", actor)])

            class Checkpoint:
                def write(self, path):
                    checkpoint_files(path)

                def restore(self, path):
                    actor.learning_rate.assign(1.)
                    return self

                def expect_partial(self):
                    return self

            source = checkpoint_files(Path(tmp) / "ckpt-1")
            handler = TensorFlowCheckpointRecovery(Checkpoint(), tmp, [("actor", actor)], learning_rate_control=control)
            for attempt, expected in enumerate((.3, .09, .05, .05), 1):
                handler(RecoveryRequest(episode=attempt, checkpoint_path=str(source), reason="test", attempt=attempt, learning_rate_factor=.3, minimum_learning_rate_scale=.05))
                self.assertAlmostEqual(actor.learning_rate.numpy(), expected)


class CertifiedTrackerTests(unittest.TestCase):
    def run_evaluation(self, tracker, root, episode, content):
        prefix = checkpoint_files(Path(root) / f"ckpt-{episode}", content)
        self.assertTrue(tracker.evaluate_async(prefix, episode))
        # Later learner writes must never replace the staged candidate.
        Path(str(prefix) + ".data-00000-of-00001").write_bytes(b"later learner")
        return apply_ready_best_checkpoint(tracker, wait_timeout=10.)

    def test_pass_stops_and_preserves_exact_candidate_not_live_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = BestCheckpointTracker(tracker_args(tmp, stop_when_solved=.95, solved_patience=2), "sac")

            def evaluate(prefix, episode, *, certification=None):
                return PolicyEvaluationResult(episode, cert_summary() if certification else evaluation(episode, 1., 1.).summary)

            tracker.evaluate = evaluate
            try:
                self.run_evaluation(tracker, tmp, 100, b"first")
                candidate = self.run_evaluation(tracker, tmp, 200, b"qualified")
                self.assertTrue(tracker.stop_requested)
                self.assertFalse(tracker.should_evaluate(300))
                self.assertEqual(Path(str(candidate) + ".data-00000-of-00001").read_bytes(), b"qualified")
                report = json.loads((Path(tmp) / "certification/attempt-1/report.json").read_text())
                self.assertTrue(report["passed"])
            finally:
                tracker.close()

    def test_failed_certificate_continues_and_next_attempt_uses_fresh_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = BestCheckpointTracker(tracker_args(tmp, stop_when_solved=.95, solved_patience=1), "dqn")
            seeds = []

            def evaluate(prefix, episode, *, certification=None):
                if certification:
                    seeds.append(certification["seed"])
                    return PolicyEvaluationResult(episode, cert_summary(0 if len(seeds) == 1 else 300))
                return evaluation(episode, 1., 1.)

            tracker.evaluate = evaluate
            try:
                self.run_evaluation(tracker, tmp, 100, b"weak")
                self.assertFalse(tracker.stop_requested)
                self.assertTrue(tracker.should_evaluate(200))
                self.run_evaluation(tracker, tmp, 200, b"strong")
                self.assertTrue(tracker.stop_requested)
                self.assertGreaterEqual(seeds[1], seeds[0] + 300)
                self.assertFalse(json.loads((Path(tmp) / "certification/attempt-1/report.json").read_text())["passed"])
            finally:
                tracker.close()

    def test_controlled_evaluation_uses_fixed_difficulty_and_regular_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = BestCheckpointTracker(tracker_args(tmp, stop_when_solved=.95), "sac")
            try:
                one = tracker._evaluation_command("ckpt", "summary", 100)
                two = tracker._evaluation_command("ckpt", "summary", 300)
                self.assertEqual(one[one.index("--training-episode") + 1], two[two.index("--training-episode") + 1])
                self.assertIn("--no-reverse-curriculum", one)
            finally:
                tracker.close()

    def test_candidate_mutation_during_certification_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = BestCheckpointTracker(tracker_args(tmp, stop_when_solved=.95, solved_patience=1), "sac")

            def evaluate(prefix, episode, *, certification=None):
                if certification:
                    Path(str(prefix) + ".data-00000-of-00001").write_bytes(b"changed during evaluation")
                    return PolicyEvaluationResult(episode, cert_summary())
                return evaluation(episode, 1., 1.)

            tracker.evaluate = evaluate
            try:
                self.run_evaluation(tracker, tmp, 100, b"candidate")
                self.assertFalse(tracker.stop_requested)
                report = json.loads((Path(tmp) / "certification/attempt-1/report.json").read_text())
                self.assertFalse(report["passed"])
                self.assertIn("changed", report["reason"])
            finally:
                tracker.close()


if __name__ == "__main__":
    unittest.main()
