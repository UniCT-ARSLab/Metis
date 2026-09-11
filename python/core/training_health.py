"""Training health monitoring and conservative automatic recovery.

The monitor deliberately treats deterministic/frozen policy evaluation as the authoritative signal.
"""

import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RECOVERY_LR_FACTORS = {
    "actor": 1.0 / 30.0,
    "critic": 1.0 / 3.0,
    "alpha": 1.0 / 30.0,
    "policy": 0.25,
}


def resolved_recovery_learning_rate_factors(args):
    """Resolve role overrides, a legacy common override, and conservative defaults."""
    legacy = getattr(args, "recovery_lr_factor", None)
    resolved = {}
    for role, default in DEFAULT_RECOVERY_LR_FACTORS.items():
        explicit = getattr(args, f"recovery_{role}_lr_factor", None)
        value = explicit if explicit is not None else legacy
        resolved[role] = float(default if value is None else value)
    return resolved


@dataclass(frozen=True)
class RecoveryRequest:
    episode: int
    checkpoint_path: str
    reason: str
    attempt: int
    learning_rate_factor: float
    minimum_learning_rate_scale: float
    total_attempt: int = 1
    mode: str = "soft"
    critic_warmup_updates: int = 0
    learning_rate_factors: dict | None = None


class OffPolicyRecoveryRuntime:
    """Mutable learner state shared by sync and async off-policy recovery hooks."""

    def __init__(
        self,
        args,
        buffer,
        *,
        initial_critic_warmup=0,
        normal_policy_update_every=1,
        replay_refill=None,
        target_sync=None,
    ):
        self.args = args
        self.health_monitor = getattr(args, "_training_health_monitor", None)
        self.buffer = buffer
        self.replay_refill = replay_refill
        self.target_sync = target_sync
        self.critic_warmup_target = max(0, int(initial_critic_warmup))
        self.critic_updates = 0
        self.policy_candidates = 0
        self.normal_policy_update_every = max(1, int(normal_policy_update_every))
        self.policy_update_every = self.normal_policy_update_every
        self.pool = None
        self.scheduler = None
        self.snapshot = None
        self.minimum_policy_version = None
        self.last_mode = None

    def attach_async(self, pool, scheduler, snapshot):
        self.pool = pool
        self.scheduler = scheduler
        self.snapshot = snapshot

    def post_restore(self, request):
        self.last_mode = request.mode
        self.critic_updates = 0
        self.policy_candidates = 0
        self.critic_warmup_target = max(0, int(request.critic_warmup_updates))
        self.policy_update_every = max(
            self.normal_policy_update_every,
            int(getattr(self.args, "recovery_policy_update_every", 4)),
        )

        replay_description = "preserved (soft recovery)"
        if request.mode == "hard":
            retained = self.buffer.clear_online(preserve_protected_demos=True)
            refilled = 0
            if self.replay_refill is not None and retained == 0:
                refilled = int(self.replay_refill() or 0)
            replay_description = (
                f"online replay cleared; retained={retained} protected demos; "
                f"refilled={refilled} demonstrations"
            )

        target_description = None
        if self.target_sync is not None:
            self.target_sync()
            target_description = "synchronized with restored online networks"

        queue_description = None
        if self.scheduler is not None and self.pool is not None:
            dropped = self.scheduler.reset_after_recovery(self.pool)
            queue_description = (
                f"dropped {dropped['events']} events/"
                f"{dropped['transitions']} transitions"
            )
        if self.snapshot is not None:
            self.minimum_policy_version = self.snapshot.version

        outcome = {
            "replay_buffer": replay_description,
            "stabilization": (
                f"policy frozen for {self.critic_warmup_target} critic updates; "
                f"policy update interval={self.policy_update_every}"
                if self.critic_warmup_target > 0
                else f"policy update interval={self.policy_update_every}"
            ),
        }
        if queue_description:
            outcome["async_queue"] = queue_description
        if target_description:
            outcome["target_networks"] = target_description
        return outcome

    def accepts(self, event):
        if self.minimum_policy_version is None:
            return True
        version = getattr(event, "policy_version", None)
        return version is None or int(version) >= int(self.minimum_policy_version)

    @property
    def warmup_complete(self):
        return self.critic_updates >= self.critic_warmup_target

    @property
    def warmup_left(self):
        return max(0, self.critic_warmup_target - self.critic_updates)

    def should_update_policy(self):
        if (
            self.last_mode is not None
            and self.health_monitor is not None
            and self.health_monitor.recovery_cycle_attempt == 0
            and not self.health_monitor.verification_pending
        ):
            self.policy_update_every = self.normal_policy_update_every
            self.last_mode = None
        if not self.warmup_complete:
            return False
        due = self.policy_candidates % self.policy_update_every == 0
        self.policy_candidates += 1
        return due

    def record_critic_update(self, count=1):
        previous = self.critic_updates
        self.critic_updates += int(count)
        return (
            self.critic_warmup_target > 0
            and previous < self.critic_warmup_target <= self.critic_updates
        )


class TrainingHealthMonitor:
    """State machine fed by episode telemetry and frozen policy evaluations."""

    def __init__(self, args, algorithm, event_sink=None):
        self.args = args
        self.algorithm = str(algorithm)
        self.enabled = bool(getattr(args, "health_monitor", True))
        self.auto_recovery = bool(getattr(args, "auto_recovery", False))
        self.state = "warming_up" if self.enabled else "disabled"
        self.phase = "training"
        self.reason = (
            "Waiting for the first frozen policy evaluation."
            if self.enabled
            else "Training health monitoring is disabled."
        )
        self.bad_evaluations = 0
        self.critical_evaluations = 0
        self.evaluation_count = 0
        self.evaluations_without_improvement = 0
        # recovery_count is lifetime telemetry. Limits apply to the current cycle so a
        # successful policy era does not permanently exhaust automatic recovery.
        self.recovery_count = 0
        self.recovery_cycle = 0
        self.recovery_cycle_attempt = 0
        self.healthy_evaluations_since_recovery = 0
        self.last_recovery_episode = None
        self.last_recovery_mode = None
        self.last_evaluation_episode = None
        self.verification_pending = False
        self.current_evaluation = None
        self.best_evaluation = None
        self._recovery_handler = None
        self._event_sink = event_sink
        self._last_event_signature = None
        self._telemetry_seen = False
        self._validate_arguments()
        checkpoint_dir = Path(getattr(args, "checkpoint_dir", "checkpoints"))
        configured_path = getattr(args, "health_state_path", None)
        self.state_path = Path(configured_path) if configured_path else checkpoint_dir / "training_health.json"
        self.events_path = self.state_path.with_name(
            self.state_path.stem + "_events.jsonl"
        )
        if self.enabled:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            is_resume = bool(
                getattr(args, "resume", False)
                or getattr(args, "resume_checkpoint", None)
            )
            if is_resume:
                self._load_previous_state()
            else:
                self.events_path.unlink(missing_ok=True)
            self._persist()

    def _validate_arguments(self):
        if self.auto_recovery and not self.enabled:
            raise ValueError("--auto-recovery requires --health-monitor")
        warning = float(getattr(self.args, "health_warning_drop", 0.35))
        critical = float(getattr(self.args, "health_critical_drop", 0.60))
        if not 0.0 <= warning < critical:
            raise ValueError(
                "--health-warning-drop must be non-negative and smaller than "
                "--health-critical-drop"
            )
        if int(getattr(self.args, "health_collapse_patience", 2)) < 1:
            raise ValueError("--health-collapse-patience must be at least 1")
        factor = getattr(self.args, "recovery_lr_factor", None)
        factor = 0.5 if factor is None else float(factor)
        minimum = float(getattr(self.args, "recovery_min_lr_scale", 0.01))
        if not 0.0 < factor <= 1.0:
            raise ValueError("--recovery-lr-factor must be in (0, 1]")
        if not 0.0 < minimum <= 1.0:
            raise ValueError("--recovery-min-lr-scale must be in (0, 1]")
        if int(getattr(self.args, "recovery_max_attempts", 3)) < 1:
            raise ValueError("--recovery-max-attempts must be at least 1")
        if int(getattr(self.args, "recovery_hard_after_attempt", 2)) < 1:
            raise ValueError("--recovery-hard-after-attempt must be at least 1")
        if int(getattr(self.args, "recovery_critic_warmup_updates", 3000)) < 0:
            raise ValueError("--recovery-critic-warmup-updates cannot be negative")
        if int(getattr(self.args, "recovery_policy_update_every", 4)) < 1:
            raise ValueError("--recovery-policy-update-every must be at least 1")
        if int(getattr(self.args, "recovery_cycle_reset_evaluations", 2)) < 1:
            raise ValueError("--recovery-cycle-reset-evaluations must be at least 1")
        if int(getattr(self.args, "recovery_min_evaluations", 5)) < 0:
            raise ValueError("--recovery-min-evaluations cannot be negative")
        for role, value in resolved_recovery_learning_rate_factors(self.args).items():
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"--recovery-{role}-lr-factor must be in (0, 1]"
                )

    def _load_previous_state(self):
        if not self.state_path.is_file():
            return
        try:
            previous = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if previous.get("algorithm") not in (None, self.algorithm):
            return
        self.recovery_count = int(previous.get("recovery_count", 0))
        # Old state files only had a lifetime counter. Migrating that value into a
        # fresh cycle would immediately lock the upgraded recovery system.
        self.recovery_cycle = int(previous.get("recovery_cycle", 0))
        self.recovery_cycle_attempt = int(previous.get("recovery_cycle_attempt", 0))
        self.healthy_evaluations_since_recovery = int(
            previous.get("healthy_evaluations_since_recovery", 0)
        )
        self.evaluation_count = int(previous.get("evaluation_count", 0))
        self.last_recovery_episode = previous.get("last_recovery_episode")
        self.last_recovery_mode = previous.get("last_recovery_mode")
        self.last_evaluation_episode = previous.get("last_evaluation_episode")
        self.verification_pending = bool(previous.get("verification_pending", False))
        self.current_evaluation = previous.get("current_evaluation")
        self.best_evaluation = previous.get("best_evaluation")
        self.reason = "Health history restored; waiting for a new frozen evaluation."

    def set_event_sink(self, event_sink):
        self._event_sink = event_sink

    def set_recovery_handler(self, handler):
        self._recovery_handler = handler

    @property
    def recovery_handler(self):
        return self._recovery_handler

    def snapshot(self):
        return {
            "algorithm": self.algorithm,
            "enabled": self.enabled,
            "auto_recovery": self.auto_recovery,
            "state": self.state,
            "phase": self.phase,
            "reason": self.reason,
            "bad_evaluations": self.bad_evaluations,
            "critical_evaluations": self.critical_evaluations,
            "evaluation_count": self.evaluation_count,
            "evaluations_without_improvement": self.evaluations_without_improvement,
            "recovery_count": self.recovery_count,
            "recovery_cycle": self.recovery_cycle,
            "recovery_cycle_attempt": self.recovery_cycle_attempt,
            "recovery_max_attempts": int(
                getattr(self.args, "recovery_max_attempts", 3)
            ),
            "healthy_evaluations_since_recovery": self.healthy_evaluations_since_recovery,
            "last_recovery_episode": self.last_recovery_episode,
            "last_recovery_mode": self.last_recovery_mode,
            "last_evaluation_episode": self.last_evaluation_episode,
            "verification_pending": self.verification_pending,
            "current_evaluation": self.current_evaluation,
            "best_evaluation": self.best_evaluation,
            "updated_at": time.time(),
        }

    def episode_fields(self):
        return {
            "health_state": self.state,
            "training_phase": self.phase,
            "recoveries": self.recovery_count,
            "recovery_cycle": self.recovery_cycle,
            "recovery_attempt": self.recovery_cycle_attempt,
            "recovery_mode": self.last_recovery_mode or "none",
        }

    def emit_snapshot(self):
        if self._event_sink is not None:
            self._event_sink({"event": "snapshot", **self.snapshot()})

    def replay_events(self, event_sink, limit=100):
        """Replay recent persisted transitions to a newly attached dashboard."""
        if not self.events_path.is_file():
            return
        try:
            with self.events_path.open("r", encoding="utf-8") as stream:
                lines = deque(stream, maxlen=max(1, int(limit)))
            for line in lines:
                try:
                    event_sink(json.loads(line))
                except (TypeError, ValueError):
                    continue
        except OSError:
            return

    def observe_episode(self, metrics):
        """Catch numerical failures without interpreting ordinary noisy losses."""
        if not self.enabled:
            return
        bad_fields = [
            key
            for key, value in metrics.items()
            if key != "episode" and self._contains_non_finite(value)
        ]
        if not bad_fields:
            if (
                not self._telemetry_seen
                and not bool(getattr(self.args, "best_checkpoint", False))
            ):
                self._telemetry_seen = True
                self.state = "healthy"
                self.reason = (
                    "Episode telemetry is finite. Frozen-checkpoint collapse "
                    "evaluation is unavailable or disabled for this run."
                )
                self._emit("telemetry_healthy", int(metrics.get("episode", -1)), severity="info")
            return
        episode = int(metrics.get("episode", -1))
        reason = "Non-finite training telemetry: " + ", ".join(sorted(bad_fields))
        self.state = "critical"
        self.reason = reason
        self._emit("numerical_failure", episode, severity="critical")
        self._maybe_recover(episode, reason, force=True)

    def observe_evaluation(
        self,
        *,
        episode,
        summary,
        best_summary,
        best_checkpoint,
        improved,
    ):
        if not self.enabled:
            return
        episode = int(episode)
        self.evaluation_count += 1
        was_verification = self.verification_pending
        self.verification_pending = False
        self.last_evaluation_episode = episode
        self.current_evaluation = self._evaluation_view(summary, episode)
        if best_summary:
            self.best_evaluation = self._evaluation_view(
                best_summary,
                int(best_summary.get("episode", episode)),
            )
        if any(
            self._contains_non_finite(summary.get(key))
            for key in (
                "success_rate",
                "reward_mean",
                "steps_mean",
                "progress_mean",
                "position_error_mean",
                "orientation_error_mean",
            )
        ):
            self.state = "critical"
            self.reason = "Frozen policy evaluation produced a non-finite metric."
            self._emit("evaluation_numerical_failure", episode, severity="critical")
            self._maybe_recover(
                episode,
                self.reason,
                checkpoint_path=best_checkpoint,
            )
            return

        if improved or not best_summary:
            self.bad_evaluations = 0
            self.critical_evaluations = 0
            self.evaluations_without_improvement = 0
            self.state = "healthy"
            self.phase = "training"
            self.reason = "Frozen policy evaluation established a new best result."
            if self.recovery_cycle_attempt:
                self._reset_recovery_cycle(
                    episode,
                    "A new validated best policy confirmed recovery.",
                )
            self._emit("evaluation_improved", episode, severity="info")
            return

        self.evaluations_without_improvement += 1
        drop, signal = self._quality_drop(summary, best_summary)
        warning_drop = float(getattr(self.args, "health_warning_drop", 0.35))
        critical_drop = float(getattr(self.args, "health_critical_drop", 0.60))
        patience = int(getattr(self.args, "health_collapse_patience", 2))

        if drop < warning_drop:
            self.bad_evaluations = 0
            self.critical_evaluations = 0
            self.state = "healthy"
            self.reason = (
                f"Frozen evaluation remains within {drop:.1%} of the best "
                f"{signal} result."
            )
            if self.recovery_cycle_attempt:
                self.healthy_evaluations_since_recovery += 1
                reset_after = int(
                    getattr(self.args, "recovery_cycle_reset_evaluations", 2)
                )
                if self.healthy_evaluations_since_recovery >= reset_after:
                    self._reset_recovery_cycle(
                        episode,
                        f"{reset_after} healthy frozen evaluations confirmed recovery.",
                    )
            if self.phase in {"stabilizing", "verifying"}:
                self.phase = "training"
            plateau_after = int(
                getattr(self.args, "health_plateau_evaluations", 6)
            )
            if (
                plateau_after > 0
                and self.evaluations_without_improvement >= plateau_after
            ):
                self.state = "warning"
                self.reason = (
                    f"No new best policy in {self.evaluations_without_improvement} "
                    "frozen evaluations. Performance is stable, so rollback is not "
                    "triggered."
                )
                self._emit("plateau_detected", episode, severity="warning")
                return
            self._emit(
                "recovery_verified" if was_verification else "evaluation_healthy",
                episode,
                severity="info",
            )
            return

        self.bad_evaluations += 1
        if was_verification:
            self.phase = "stabilizing"
        if drop >= critical_drop:
            self.critical_evaluations += 1
        else:
            self.critical_evaluations = 0
        self.healthy_evaluations_since_recovery = 0
        if drop >= critical_drop and self.critical_evaluations >= patience:
            self.state = "critical"
            self.reason = (
                f"Frozen {signal} dropped {drop:.1%} from the best result for "
                f"{self.critical_evaluations} consecutive critical evaluation(s)."
            )
            self._emit("collapse_detected", episode, severity="critical")
            self._maybe_recover(
                episode,
                self.reason,
                checkpoint_path=best_checkpoint,
            )
            return

        self.state = "warning"
        self.reason = (
            f"Frozen {signal} is {drop:.1%} below the best result "
            f"({self.critical_evaluations}/{patience} critical confirmations)."
        )
        self._emit(
            "recovery_verification_failed" if was_verification else "evaluation_warning",
            episode,
            severity="warning",
        )

    def mark_finished(self):
        if not self.enabled:
            return
        self.phase = "finished"
        self._emit("training_finished", self.last_evaluation_episode, severity="info")

    def reset_evaluation_baseline(self, reason):
        """Forget cross-stage comparisons after a curriculum promotion."""
        if not self.enabled:
            return
        self.state = "warming_up"
        self.phase = "curriculum"
        self.reason = str(reason)
        self.bad_evaluations = 0
        self.critical_evaluations = 0
        self.evaluations_without_improvement = 0
        self.healthy_evaluations_since_recovery = 0
        self.current_evaluation = None
        self.best_evaluation = None
        self.verification_pending = False
        self._emit(
            "curriculum_promoted",
            self.last_evaluation_episode,
            severity="info",
        )

    def _maybe_recover(self, episode, reason, checkpoint_path=None, force=False):
        if not self.auto_recovery:
            self._emit(
                "recovery_skipped",
                episode,
                severity="warning",
                detail="Automatic recovery is disabled.",
            )
            return
        if not checkpoint_path and self.best_evaluation:
            checkpoint_path = self.best_evaluation.get("checkpoint")
        if not checkpoint_path:
            self._emit(
                "recovery_skipped",
                episode,
                severity="critical",
                detail="No validated best checkpoint is available.",
            )
            return
        if self._recovery_handler is None:
            self._emit(
                "recovery_skipped",
                episode,
                severity="critical",
                detail="This backend did not register a recovery handler.",
            )
            return
        maturity_reason = None if force else self._recovery_maturity_reason()
        if maturity_reason:
            self.state = "warning"
            self.phase = "training"
            self.reason = maturity_reason
            self._emit(
                "recovery_deferred",
                episode,
                severity="warning",
                detail="Collapse monitoring remains active without consuming a recovery attempt.",
            )
            return
        maximum = int(getattr(self.args, "recovery_max_attempts", 3))
        if self.recovery_cycle_attempt >= maximum:
            self._emit(
                "recovery_skipped",
                episode,
                severity="critical",
                detail=f"Maximum recovery attempts reached for cycle {self.recovery_cycle} ({maximum}).",
            )
            return
        cooldown = int(getattr(self.args, "recovery_cooldown_evaluations", 2))
        evaluation_every = max(1, int(getattr(self.args, "best_evaluation_every", 1)))
        if (
            self.last_recovery_episode is not None
            and int(episode) - self.last_recovery_episode < cooldown * evaluation_every
        ):
            self._emit(
                "recovery_skipped",
                episode,
                severity="warning",
                detail="Recovery cooldown is still active.",
            )
            return

        if self.recovery_cycle_attempt == 0:
            self.recovery_cycle += 1
        attempt = self.recovery_cycle_attempt + 1
        total_attempt = self.recovery_count + 1
        hard_after = int(getattr(self.args, "recovery_hard_after_attempt", 2))
        mode = "hard" if attempt >= hard_after else "soft"
        legacy_factor = getattr(self.args, "recovery_lr_factor", None)
        legacy_factor = 0.5 if legacy_factor is None else float(legacy_factor)
        learning_rate_factors = resolved_recovery_learning_rate_factors(self.args)
        request = RecoveryRequest(
            episode=int(episode),
            checkpoint_path=str(checkpoint_path),
            reason=str(reason),
            attempt=attempt,
            total_attempt=total_attempt,
            mode=mode,
            critic_warmup_updates=int(
                getattr(self.args, "recovery_critic_warmup_updates", 3000)
            ),
            learning_rate_factor=legacy_factor,
            learning_rate_factors=learning_rate_factors,
            minimum_learning_rate_scale=float(
                getattr(self.args, "recovery_min_lr_scale", 0.01)
            ),
        )
        self.phase = "recovering"
        self._emit(
            "recovery_started",
            episode,
            severity="critical",
            detail=(
                f"Restoring {checkpoint_path} with {mode} recovery "
                f"(cycle {self.recovery_cycle}, attempt {attempt}/{maximum}, "
                f"lifetime recovery {total_attempt})."
            ),
        )
        try:
            outcome = self._recovery_handler(request) or {}
        except Exception as exc:
            self.phase = "training"
            self.reason = f"Automatic recovery failed: {exc}"
            self._emit("recovery_failed", episode, severity="critical")
            return

        self.recovery_count = total_attempt
        self.recovery_cycle_attempt = attempt
        self.healthy_evaluations_since_recovery = 0
        self.last_recovery_episode = int(episode)
        self.last_recovery_mode = mode
        self.bad_evaluations = 0
        self.critical_evaluations = 0
        self.verification_pending = True
        self.phase = "verifying"
        self.state = "warning"
        learning_rates = outcome.get("learning_rates", {})
        diagnostic = outcome.get("diagnostic_checkpoint")
        detail = (
            f"Best checkpoint restored with {mode} recovery; an immediate frozen "
            "verification has been requested."
        )
        if learning_rates:
            detail += f" Learning rates: {learning_rates}."
        if diagnostic:
            detail += f" Pre-recovery snapshot: {diagnostic}."
        for key in ("replay_buffer", "stabilization", "async_queue", "target_networks"):
            value = outcome.get(key)
            if value:
                detail += f" {key.replace('_', ' ').title()}: {value}."
        self.reason = detail
        self._emit("recovery_completed", episode, severity="warning", detail=detail)

    def _recovery_maturity_reason(self):
        minimum_evaluations = int(getattr(self.args, "recovery_min_evaluations", 5))
        if self.evaluation_count < minimum_evaluations:
            return (
                f"Automatic recovery is waiting for {minimum_evaluations} frozen "
                f"evaluations ({self.evaluation_count} available)."
            )
        require_success = bool(
            getattr(self.args, "recovery_require_success_baseline", True)
        )
        metric = str(getattr(self.args, "best_metric", "auto"))
        if require_success and metric == "success_rate":
            best_success = float((self.best_evaluation or {}).get("success_rate", 0.0))
            minimum_success = float(
                getattr(self.args, "health_min_success_baseline", 0.05)
            )
            if best_success < minimum_success:
                return (
                    "Automatic recovery is waiting for a meaningful success-rate "
                    f"baseline ({best_success:.1%}/{minimum_success:.1%}). Use "
                    "--best-metric reward_mean for tasks without a success signal."
                )
        return None

    def _reset_recovery_cycle(self, episode, reason):
        previous_cycle = self.recovery_cycle
        previous_attempts = self.recovery_cycle_attempt
        self.recovery_cycle_attempt = 0
        self.healthy_evaluations_since_recovery = 0
        self.bad_evaluations = 0
        self.critical_evaluations = 0
        self.verification_pending = False
        self.phase = "training"
        self._emit(
            "recovery_cycle_reset",
            episode,
            severity="info",
            detail=(
                f"{reason} Recovery cycle {previous_cycle} closed after "
                f"{previous_attempts} attempt(s); future collapses receive a fresh budget."
            ),
        )

    def _quality_drop(self, current, best):
        metric = str(getattr(self.args, "best_metric", "auto"))
        uses_selection_success = (
            "selection_success_rate" in best
            or "selection_success_rate" in current
        )
        best_success = float(
            best.get("selection_success_rate", best.get("success_rate", 0.0))
        )
        current_success = float(
            current.get(
                "selection_success_rate",
                current.get("success_rate", 0.0),
            )
        )
        minimum_success = float(
            getattr(self.args, "health_min_success_baseline", 0.05)
        )
        if metric != "reward_mean" and best_success >= minimum_success:
            return (
                max(
                    0.0,
                    (best_success - current_success) / max(best_success, 1e-9),
                ),
                (
                    "selection success rate"
                    if uses_selection_success
                    else "success rate"
                ),
            )

        if metric in {"auto", "task_progress"} and "progress_mean" in best:
            best_progress = float(best.get("progress_mean", 0.0))
            current_progress = float(current.get("progress_mean", 0.0))
            return (
                max(
                    0.0,
                    (best_progress - current_progress)
                    / max(abs(best_progress), 0.05),
                ),
                "task progress",
            )

        best_reward = float(best.get("reward_mean", 0.0))
        current_reward = float(current.get("reward_mean", 0.0))
        reward_floor = float(getattr(self.args, "health_reward_scale_floor", 1.0))
        reward_drop = max(
            0.0,
            (best_reward - current_reward) / max(abs(best_reward), reward_floor),
        )
        return reward_drop, "mean reward"

    @staticmethod
    def _evaluation_view(summary, episode):
        success_rate = float(
            summary.get(
                "selection_success_rate",
                summary.get("success_rate", 0.0),
            )
        )
        return {
            "episode": int(episode),
            "checkpoint": summary.get("checkpoint"),
            "success_rate": success_rate,
            "overall_success_rate": float(summary.get("success_rate", success_rate)),
            "reward_mean": float(
                summary.get("regular_reward_mean", summary.get("reward_mean", 0.0))
            ),
            "overall_reward_mean": float(summary.get("reward_mean", 0.0)),
            "steps_mean": float(summary.get("steps_mean", 0.0)),
            **(
                {
                    "progress_mean": float(
                        summary.get(
                            "regular_progress_mean",
                            summary.get("progress_mean", 0.0),
                        )
                    )
                }
                if "progress_mean" in summary or "regular_progress_mean" in summary
                else {}
            ),
            **(
                {
                    "position_error_mean": float(
                        summary.get(
                            "regular_position_error_mean",
                            summary.get("position_error_mean", 0.0),
                        )
                    )
                }
                if (
                    "position_error_mean" in summary
                    or "regular_position_error_mean" in summary
                )
                else {}
            ),
            **(
                {
                    "orientation_error_mean": float(
                        summary.get(
                            "regular_orientation_error_mean",
                            summary.get("orientation_error_mean", 0.0),
                        )
                    )
                }
                if (
                    "orientation_error_mean" in summary
                    or "regular_orientation_error_mean" in summary
                )
                else {}
            ),
            **(
                {
                    "hold_frames_max": float(
                        summary.get(
                            "regular_hold_frames_max",
                            summary.get("hold_frames_max", 0.0),
                        )
                    )
                }
                if "hold_frames_max" in summary or "regular_hold_frames_max" in summary
                else {}
            ),
        }

    @staticmethod
    def _contains_non_finite(value):
        if isinstance(value, (float, int)):
            return isinstance(value, float) and not math.isfinite(value)
        if isinstance(value, (list, tuple, dict)):
            values = value.values() if isinstance(value, dict) else value
            return any(TrainingHealthMonitor._contains_non_finite(item) for item in values)
        if isinstance(value, str):
            return bool(
                re.search(
                    r"(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])",
                    value,
                    flags=re.IGNORECASE,
                )
            )
        return False

    def _emit(self, event, episode, *, severity, detail=None):
        payload = {
            "event": event,
            "severity": severity,
            "episode": None if episode is None else int(episode),
            **self.snapshot(),
        }
        if detail:
            payload["detail"] = str(detail)
        signature = (
            event,
            payload["episode"],
            self.state,
            self.phase,
            self.reason,
            payload.get("detail"),
        )
        if signature == self._last_event_signature:
            return
        self._last_event_signature = signature
        self._persist(payload)
        label = {
            "info": "TRAINING HEALTH",
            "warning": "TRAINING HEALTH WARNING",
            "critical": "TRAINING HEALTH CRITICAL",
        }.get(severity, "TRAINING HEALTH")
        suffix = f" {detail}" if detail and detail != self.reason else ""
        print(
            f"{label}: state={self.state} phase={self.phase} "
            f"episode={payload['episode']} {self.reason}{suffix}",
            flush=True,
        )
        if self._event_sink is not None:
            self._event_sink(payload)

    def _persist(self, event=None):
        if not self.enabled:
            return
        snapshot = self.snapshot()
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)
        if event is not None:
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")


class TensorFlowCheckpointRecovery:
    """Restore a full TensorFlow checkpoint and invoke algorithm-specific stabilization."""

    def __init__(
        self,
        checkpoint,
        checkpoint_dir,
        optimizers,
        *,
        keep_diagnostics=3,
        policy_publisher=None,
        post_restore=None,
    ):
        self.checkpoint = checkpoint
        self.directory = Path(checkpoint_dir) / "recovery"
        self.optimizers = [
            (str(name), optimizer, self._read_learning_rate(optimizer))
            for name, optimizer in optimizers
            if optimizer is not None
        ]
        self.keep_diagnostics = max(1, int(keep_diagnostics))
        self.policy_publisher = policy_publisher
        self.post_restore = post_restore

    def set_policy_publisher(self, callback):
        self.policy_publisher = callback

    def set_post_restore(self, callback):
        self.post_restore = callback

    def __call__(self, request):
        checkpoint_path = str(request.checkpoint_path)
        if not Path(checkpoint_path + ".index").is_file():
            raise FileNotFoundError(
                f"Validated recovery checkpoint is missing: {checkpoint_path}.index"
            )

        self.directory.mkdir(parents=True, exist_ok=True)
        diagnostic_prefix = self.directory / (
            f"pre-recovery-ep{request.episode}-attempt{request.attempt}"
        )
        self.checkpoint.write(str(diagnostic_prefix))
        self._prune_diagnostics()

        self.checkpoint.restore(checkpoint_path).expect_partial()
        learning_rates = {}
        for name, optimizer, original_rate in self.optimizers:
            factor = self._learning_rate_factor(name, request)
            scale = max(request.minimum_learning_rate_scale, factor)
            target = original_rate * scale
            optimizer.learning_rate.assign(target)
            learning_rates[name] = target

        if self.policy_publisher is not None:
            self.policy_publisher()
        outcome = {
            "diagnostic_checkpoint": str(diagnostic_prefix),
            "learning_rates": learning_rates,
            "replay_buffer": "preserved (soft recovery)",
        }
        if self.post_restore is not None:
            extra = self.post_restore(request) or {}
            outcome.update(extra)
        return outcome

    @staticmethod
    def _learning_rate_factor(name, request):
        factors = request.learning_rate_factors or {}
        normalized = str(name).lower()
        if normalized.startswith("critic"):
            role = "critic"
        elif normalized.startswith(("alpha", "entropy", "temperature")):
            role = "alpha"
        elif normalized.startswith("actor"):
            role = "actor"
        else:
            role = "policy"
        return float(factors.get(role, request.learning_rate_factor))

    @staticmethod
    def _read_learning_rate(optimizer):
        value = optimizer.learning_rate
        if hasattr(value, "numpy"):
            value = value.numpy()
        return float(value)

    def _prune_diagnostics(self):
        index_paths = sorted(
            self.directory.glob("pre-recovery-*.index"),
            key=lambda path: path.stat().st_mtime,
        )
        for index_path in index_paths[:-self.keep_diagnostics]:
            prefix = index_path.with_suffix("")
            for shard in prefix.parent.glob(prefix.name + ".*"):
                shard.unlink(missing_ok=True)
