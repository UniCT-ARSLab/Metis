import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.training_health import (
    OffPolicyRecoveryRuntime,
    RecoveryRequest,
    TensorFlowCheckpointRecovery,
    TrainingHealthMonitor,
    resolved_recovery_learning_rate_factors,
)
from core.replay_buffer import ReplayBuffer


def health_args(directory, **overrides):
    values = {
        "checkpoint_dir": str(directory),
        "health_monitor": True,
        "auto_recovery": False,
        "health_warning_drop": 0.35,
        "health_critical_drop": 0.60,
        "health_collapse_patience": 2,
        "health_plateau_evaluations": 6,
        "health_min_success_baseline": 0.05,
        "health_reward_scale_floor": 1.0,
        "health_state_path": None,
        "recovery_lr_factor": 0.5,
        "recovery_actor_lr_factor": 1.0 / 30.0,
        "recovery_critic_lr_factor": 1.0 / 3.0,
        "recovery_alpha_lr_factor": 1.0 / 30.0,
        "recovery_policy_lr_factor": 0.25,
        "recovery_min_lr_scale": 0.1,
        "recovery_max_attempts": 3,
        "recovery_hard_after_attempt": 2,
        "recovery_critic_warmup_updates": 3,
        "recovery_policy_update_every": 4,
        "recovery_cycle_reset_evaluations": 2,
        "recovery_require_success_baseline": True,
        "recovery_cooldown_evaluations": 2,
        "recovery_min_evaluations": 0,
        "best_evaluation_every": 100,
        "best_metric": "auto",
        "recovery_keep_diagnostics": 3,
        "best_checkpoint": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def summary(success, reward, checkpoint=None):
    return {
        "success_rate": float(success),
        "reward_mean": float(reward),
        "steps_mean": 100.0,
        "checkpoint": checkpoint,
    }


class TrainingHealthMonitorTests(unittest.TestCase):
    def test_learning_rate_factor_resolution_keeps_legacy_override_compatible(self):
        defaults = resolved_recovery_learning_rate_factors(
            SimpleNamespace(recovery_lr_factor=None)
        )
        self.assertAlmostEqual(defaults["actor"], 1.0 / 30.0)
        self.assertAlmostEqual(defaults["critic"], 1.0 / 3.0)
        self.assertAlmostEqual(defaults["policy"], 0.25)

        legacy = resolved_recovery_learning_rate_factors(
            SimpleNamespace(recovery_lr_factor=0.5)
        )
        self.assertEqual(set(legacy.values()), {0.5})

        mixed = resolved_recovery_learning_rate_factors(
            SimpleNamespace(
                recovery_lr_factor=0.5,
                recovery_actor_lr_factor=0.1,
            )
        )
        self.assertEqual(mixed["actor"], 0.1)
        self.assertEqual(mixed["critic"], 0.5)

    def test_collapse_requires_consecutive_frozen_evaluations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingHealthMonitor(health_args(temp_dir), "sac")
            best = summary(0.8, 10.0, "best/ckpt-100")

            monitor.observe_evaluation(
                episode=200,
                summary=summary(0.1, 0.0),
                best_summary=best,
                best_checkpoint=best["checkpoint"],
                improved=False,
            )
            self.assertEqual(monitor.state, "warning")

            monitor.observe_evaluation(
                episode=300,
                summary=summary(0.1, 0.0),
                best_summary=best,
                best_checkpoint=best["checkpoint"],
                improved=False,
            )
            self.assertEqual(monitor.state, "critical")
            self.assertEqual(monitor.recovery_count, 0)

    def test_enabled_recovery_restores_only_after_confirmed_collapse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = health_args(temp_dir, auto_recovery=True)
            monitor = TrainingHealthMonitor(args, "dqn")
            requests = []
            monitor.set_recovery_handler(lambda request: requests.append(request) or {})
            best = summary(0.9, 20.0, "best/ckpt-100")

            for episode in (200, 300):
                monitor.observe_evaluation(
                    episode=episode,
                    summary=summary(0.0, -20.0),
                    best_summary=best,
                    best_checkpoint=best["checkpoint"],
                    improved=False,
                )

            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].checkpoint_path, "best/ckpt-100")
            self.assertEqual(monitor.phase, "verifying")
            self.assertTrue(monitor.verification_pending)
            self.assertEqual(monitor.recovery_count, 1)

    def test_new_best_clears_warning_and_stabilization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingHealthMonitor(health_args(temp_dir), "ppo")
            monitor.state = "warning"
            monitor.phase = "stabilizing"
            monitor.bad_evaluations = 2

            result = summary(0.9, 2.0, "best/ckpt-400")
            monitor.observe_evaluation(
                episode=400,
                summary=result,
                best_summary=result,
                best_checkpoint=result["checkpoint"],
                improved=True,
            )

            self.assertEqual(monitor.state, "healthy")
            self.assertEqual(monitor.phase, "training")
            self.assertEqual(monitor.bad_evaluations, 0)

    def test_plateau_warns_without_requesting_recovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = health_args(
                temp_dir,
                auto_recovery=True,
                health_plateau_evaluations=2,
            )
            monitor = TrainingHealthMonitor(args, "sac")
            requests = []
            monitor.set_recovery_handler(lambda request: requests.append(request))
            best = summary(0.8, 10.0, "best/ckpt-100")

            for episode in (200, 300):
                monitor.observe_evaluation(
                    episode=episode,
                    summary=summary(0.79, 9.9),
                    best_summary=best,
                    best_checkpoint=best["checkpoint"],
                    improved=False,
                )

            self.assertEqual(monitor.state, "warning")
            self.assertIn("No new best policy", monitor.reason)
            self.assertEqual(requests, [])

    def test_recovery_cooldown_prevents_rapid_rollback_loops(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = health_args(
                temp_dir,
                auto_recovery=True,
                health_collapse_patience=1,
                recovery_cooldown_evaluations=2,
            )
            monitor = TrainingHealthMonitor(args, "sac")
            requests = []
            monitor.set_recovery_handler(lambda request: requests.append(request) or {})
            best = summary(0.9, 10.0, "best/ckpt-100")

            for episode in (200, 300, 400):
                monitor.observe_evaluation(
                    episode=episode,
                    summary=summary(0.0, -10.0),
                    best_summary=best,
                    best_checkpoint=best["checkpoint"],
                    improved=False,
                )

            self.assertEqual([request.episode for request in requests], [200, 400])
            self.assertEqual([request.mode for request in requests], ["soft", "hard"])
            self.assertEqual([request.attempt for request in requests], [1, 2])

    def test_healthy_evaluations_reset_cycle_without_erasing_lifetime_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = health_args(
                temp_dir,
                auto_recovery=True,
                health_collapse_patience=1,
                recovery_cooldown_evaluations=1,
            )
            monitor = TrainingHealthMonitor(args, "sac")
            requests = []
            monitor.set_recovery_handler(lambda request: requests.append(request) or {})
            best = summary(0.9, 10.0, "best/ckpt-100")

            monitor.observe_evaluation(
                episode=200,
                summary=summary(0.0, -10.0),
                best_summary=best,
                best_checkpoint=best["checkpoint"],
                improved=False,
            )
            for episode in (300, 400):
                monitor.observe_evaluation(
                    episode=episode,
                    summary=summary(0.85, 9.0),
                    best_summary=best,
                    best_checkpoint=best["checkpoint"],
                    improved=False,
                )

            self.assertEqual(monitor.recovery_count, 1)
            self.assertEqual(monitor.recovery_cycle_attempt, 0)
            self.assertEqual(monitor.phase, "training")

            monitor.observe_evaluation(
                episode=500,
                summary=summary(0.0, -10.0),
                best_summary=best,
                best_checkpoint=best["checkpoint"],
                improved=False,
            )
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[-1].attempt, 1)
            self.assertEqual(requests[-1].mode, "soft")
            self.assertEqual(monitor.recovery_count, 2)

    def test_recovery_waits_for_meaningful_success_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = health_args(
                temp_dir,
                auto_recovery=True,
                health_collapse_patience=1,
                recovery_min_evaluations=1,
            )
            monitor = TrainingHealthMonitor(args, "dqn")
            requests = []
            monitor.set_recovery_handler(lambda request: requests.append(request) or {})
            best = summary(0.0, 10.0, "best/ckpt-100")

            monitor.observe_evaluation(
                episode=100,
                summary=summary(0.0, -10.0),
                best_summary=best,
                best_checkpoint=best["checkpoint"],
                improved=False,
            )

            self.assertEqual(requests, [])
            self.assertEqual(monitor.recovery_count, 0)
            self.assertEqual(monitor.state, "warning")
            self.assertIn("meaningful success-rate baseline", monitor.reason)

    def test_reward_metric_allows_recovery_without_success_signal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = health_args(
                temp_dir,
                auto_recovery=True,
                health_collapse_patience=1,
                recovery_min_evaluations=1,
                best_metric="reward_mean",
            )
            monitor = TrainingHealthMonitor(args, "dqn")
            requests = []
            monitor.set_recovery_handler(lambda request: requests.append(request) or {})
            best = summary(0.0, 10.0, "best/ckpt-100")

            monitor.observe_evaluation(
                episode=100,
                summary=summary(0.0, -10.0),
                best_summary=best,
                best_checkpoint=best["checkpoint"],
                improved=False,
            )

            self.assertEqual(len(requests), 1)

    def test_non_finite_telemetry_is_critical(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingHealthMonitor(health_args(temp_dir), "td3")

            monitor.observe_episode({"episode": 7, "critic_loss": "nan"})

            self.assertEqual(monitor.state, "critical")
            self.assertIn("critic_loss", monitor.reason)
            self.assertTrue(monitor.state_path.is_file())
            self.assertTrue(monitor.events_path.is_file())


class _FakeVariable:
    def __init__(self, value):
        self.value = float(value)

    def numpy(self):
        return self.value

    def assign(self, value):
        self.value = float(value)


class _FakeOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = _FakeVariable(learning_rate)


class _RestoreStatus:
    def expect_partial(self):
        return self


class _FakeCheckpoint:
    def __init__(self):
        self.writes = []
        self.restores = []

    def write(self, prefix):
        self.writes.append(prefix)
        Path(prefix + ".index").write_text("diagnostic", encoding="utf-8")
        Path(prefix + ".data-00000-of-00001").write_text("diagnostic", encoding="utf-8")
        return prefix

    def restore(self, prefix):
        self.restores.append(prefix)
        return _RestoreStatus()


class TensorFlowRecoveryTests(unittest.TestCase):
    def test_recovery_archives_current_state_restores_best_and_republishes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            best = Path(temp_dir) / "best" / "ckpt-100"
            best.parent.mkdir()
            Path(str(best) + ".index").write_text("best", encoding="utf-8")
            checkpoint = _FakeCheckpoint()
            optimizer = _FakeOptimizer(0.001)
            published = []
            recovery = TensorFlowCheckpointRecovery(
                checkpoint,
                temp_dir,
                [("actor", optimizer)],
                policy_publisher=lambda: published.append(True),
            )

            result = recovery(
                RecoveryRequest(
                    episode=300,
                    checkpoint_path=str(best),
                    reason="collapse",
                    attempt=1,
                    learning_rate_factor=0.5,
                    minimum_learning_rate_scale=0.1,
                )
            )

            self.assertEqual(checkpoint.restores, [str(best)])
            self.assertAlmostEqual(optimizer.learning_rate.value, 0.0005)
            self.assertEqual(published, [True])
            self.assertTrue(Path(result["diagnostic_checkpoint"] + ".index").is_file())

    def test_recovery_uses_role_specific_learning_rates_and_post_restore_hook(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            best = Path(temp_dir) / "best" / "ckpt-100"
            best.parent.mkdir()
            Path(str(best) + ".index").write_text("best", encoding="utf-8")
            checkpoint = _FakeCheckpoint()
            actor = _FakeOptimizer(0.003)
            critic = _FakeOptimizer(0.003)
            alpha = _FakeOptimizer(0.003)
            requests = []
            recovery = TensorFlowCheckpointRecovery(
                checkpoint,
                temp_dir,
                [("actor", actor), ("critic1", critic), ("alpha", alpha)],
                post_restore=lambda request: requests.append(request) or {"hook": True},
            )

            result = recovery(
                RecoveryRequest(
                    episode=300,
                    checkpoint_path=str(best),
                    reason="collapse",
                    attempt=2,
                    total_attempt=4,
                    mode="hard",
                    critic_warmup_updates=3000,
                    learning_rate_factor=0.5,
                    learning_rate_factors={
                        "actor": 1.0 / 30.0,
                        "critic": 1.0 / 3.0,
                        "alpha": 1.0 / 30.0,
                    },
                    minimum_learning_rate_scale=0.01,
                )
            )

            self.assertAlmostEqual(actor.learning_rate.value, 0.0001)
            self.assertAlmostEqual(critic.learning_rate.value, 0.001)
            self.assertAlmostEqual(alpha.learning_rate.value, 0.0001)
            self.assertEqual(requests[0].mode, "hard")
            self.assertTrue(result["hook"])


class OffPolicyRecoveryRuntimeTests(unittest.TestCase):
    def test_hard_recovery_clears_online_replay_and_stages_policy_updates(self):
        buffer = ReplayBuffer(capacity=8)
        demonstrations = [
            (
                [float(value)],
                [float(value)],
                float(value),
                [float(value + 1)],
                False,
            )
            for value in range(2)
        ]
        obs, actions, rewards, next_obs, dones = zip(*demonstrations)
        buffer.add_many(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            is_demo=True,
            protect=True,
        )
        buffer.add([9.0], [9.0], 9.0, [10.0], False)
        target_syncs = []
        args = SimpleNamespace(recovery_policy_update_every=4)
        runtime = OffPolicyRecoveryRuntime(
            args,
            buffer,
            normal_policy_update_every=2,
            target_sync=lambda: target_syncs.append(True),
        )

        outcome = runtime.post_restore(
            RecoveryRequest(
                episode=10,
                checkpoint_path="best/ckpt-10",
                reason="collapse",
                attempt=2,
                mode="hard",
                critic_warmup_updates=3,
                learning_rate_factor=0.5,
                minimum_learning_rate_scale=0.1,
            )
        )

        self.assertEqual(len(buffer), 2)
        self.assertEqual(target_syncs, [True])
        self.assertIn("online replay cleared", outcome["replay_buffer"])
        self.assertFalse(runtime.should_update_policy())
        runtime.record_critic_update(3)
        self.assertTrue(runtime.should_update_policy())
        self.assertFalse(runtime.should_update_policy())


if __name__ == "__main__":
    unittest.main()
