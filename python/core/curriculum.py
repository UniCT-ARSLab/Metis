import argparse
import json
import threading
from pathlib import Path

import numpy as np


def add_adaptive_curriculum_arguments(parser):
    # Title kept in sync with core.training.GROUP_ADAPTIVE_CURRICULUM by hand: importing it here
    # would close an import cycle, since core.training already imports this module.
    parser = parser.add_argument_group("adaptive curriculum")
    parser.add_argument(
        "--adaptive-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Expose a persistent 0..1 curriculum level to Godot and promote it only "
            "after frozen policy evaluations meet the configured mastery threshold."
        ),
    )
    parser.add_argument(
        "--curriculum-initial-level",
        type=float,
        default=0.0,
        help="Initial adaptive curriculum level for a fresh run.",
    )
    parser.add_argument(
        "--curriculum-level-step",
        type=float,
        default=0.1,
        help="Curriculum increment after a stage is mastered.",
    )
    parser.add_argument(
        "--curriculum-promotion-metric",
        choices=("success_rate", "task_progress", "reward_mean"),
        default="success_rate",
        help="Frozen-evaluation metric used to promote the curriculum.",
    )
    parser.add_argument(
        "--curriculum-promotion-threshold",
        type=float,
        default=0.70,
        help="Required frozen-evaluation metric value for a promotion confirmation.",
    )
    parser.add_argument(
        "--curriculum-promotion-evaluations",
        type=int,
        default=2,
        help="Consecutive qualifying frozen evaluations required to advance one stage.",
    )
    parser.add_argument(
        "--curriculum-min-policy-updates",
        type=int,
        default=100,
        help=(
            "Policy optimizer updates required before frozen evaluations may "
            "promote the curriculum."
        ),
    )
    parser.add_argument(
        "--curriculum-demotion-threshold",
        type=float,
        default=None,
        help=(
            "Optional frozen-evaluation metric threshold below which the adaptive "
            "curriculum moves back one level. Disabled by default; when enabled it "
            "must be lower than --curriculum-promotion-threshold."
        ),
    )
    parser.add_argument(
        "--curriculum-demotion-evaluations",
        type=int,
        default=3,
        help=(
            "Consecutive frozen evaluations below the demotion threshold required "
            "to move back one curriculum level."
        ),
    )
    parser.add_argument(
        "--curriculum-stage-names",
        type=str,
        default="",
        help=(
            "Optional comma-separated display names for the discrete curriculum stages, "
            "ordered by ascending level (e.g. 'A,B,C,D,E,F'). The framework only owns the "
            "generic stage_index; these names come from the task and are surfaced verbatim "
            "in logs and the dashboard. Empty leaves stage_name unset (index only)."
        ),
    )


class AdaptiveCurriculumController:
    """Thread-safe, checkpoint-directory scoped curriculum state.

    Godot scenarios remain free to interpret ``curriculum_level`` in a task-specific
    way. The trainer only owns promotion: a level is advanced after deterministic
    frozen evaluations demonstrate mastery, never because a wall-clock episode number
    happened to be reached.
    """

    STATE_VERSION = 2

    def __init__(self, args):
        self.enabled = bool(getattr(args, "adaptive_curriculum", False))
        self._lock = threading.Lock()
        self.level_step = float(getattr(args, "curriculum_level_step", 0.1))
        self.metric = str(
            getattr(args, "curriculum_promotion_metric", "success_rate")
        )
        self.threshold = float(
            getattr(args, "curriculum_promotion_threshold", 0.70)
        )
        self.required_evaluations = int(
            getattr(args, "curriculum_promotion_evaluations", 2)
        )
        self.min_policy_updates = int(
            getattr(args, "curriculum_min_policy_updates", 100)
        )
        raw_demotion_threshold = getattr(
            args, "curriculum_demotion_threshold", None
        )
        self.demotion_threshold = (
            None
            if raw_demotion_threshold is None
            else float(raw_demotion_threshold)
        )
        self.demotion_required_evaluations = int(
            getattr(args, "curriculum_demotion_evaluations", 3)
        )
        # Optional, task-supplied stage labels. The framework owns only stage_index;
        # names (A-F for OpenArm, anything else for another task) come from config.
        self.stage_names = self._parse_stage_names(
            getattr(args, "curriculum_stage_names", None)
        )
        self.level = float(
            np.clip(getattr(args, "curriculum_initial_level", 0.0), 0.0, 1.0)
        )
        self.confirmations = 0
        self.demotion_confirmations = 0
        # Policy-optimizer updates at the last stage transition. Promotion is frozen until
        # min_policy_updates NEW updates accrue since then (cumulative updates would satisfy the
        # requirement forever after the first promotion). Persisted + restored from checkpoint.
        self.policy_updates_at_last_transition = 0
        self.last_training_updates = 0
        self.last_transition = None
        self.last_evaluation_episode = None
        self.last_metric_value = None
        self.last_promotion_episode = None
        self.last_promotion_checkpoint = None
        self.last_demotion_episode = None
        self.state_path = Path(args.checkpoint_dir) / "curriculum_state.json"

        self._validate()
        if self.enabled and self._is_resume(args):
            self._load()
        if self.enabled:
            self._save()

    @staticmethod
    def _is_resume(args):
        return bool(
            getattr(args, "resume", False)
            or getattr(args, "resume_checkpoint", None)
        )

    def _validate(self):
        if not self.enabled:
            return
        if not 0.0 < self.level_step <= 1.0:
            raise ValueError("--curriculum-level-step must be in (0, 1]")
        if self.required_evaluations < 1:
            raise ValueError("--curriculum-promotion-evaluations must be at least 1")
        if self.min_policy_updates < 0:
            raise ValueError("--curriculum-min-policy-updates cannot be negative")
        if not np.isfinite(self.threshold):
            raise ValueError("--curriculum-promotion-threshold must be finite")
        if self.demotion_required_evaluations < 1:
            raise ValueError("--curriculum-demotion-evaluations must be at least 1")
        if self.demotion_threshold is not None:
            if not np.isfinite(self.demotion_threshold):
                raise ValueError("--curriculum-demotion-threshold must be finite")
            if self.demotion_threshold >= self.threshold:
                raise ValueError(
                    "--curriculum-demotion-threshold must be lower than "
                    "--curriculum-promotion-threshold"
                )

    def scenario_config(self):
        if not self.enabled:
            return {}
        with self._lock:
            return {"curriculum_level": float(self.level)}

    def evaluation_metric(self, summary):
        if self.metric == "success_rate":
            return float(
                summary.get(
                    "selection_success_rate",
                    summary.get("success_rate", 0.0),
                )
            )
        if self.metric == "reward_mean":
            return float(
                summary.get(
                    "regular_reward_mean",
                    summary.get("reward_mean", -np.inf),
                )
            )
        return float(
            summary.get(
                "regular_progress_mean",
                summary.get("progress_mean", -np.inf),
            )
        )

    def _snap_level(self, level):
        """Snap to the exact discrete stage grid so a transition lands on a stage, never a float
        intermediate (e.g. 0.6 - 0.2 -> exactly 0.4)."""
        steps = round(float(level) / self.level_step)
        return float(np.clip(steps * self.level_step, 0.0, 1.0))

    def updates_since_last_transition(self, training_updates):
        return int(training_updates) - int(self.policy_updates_at_last_transition)

    @staticmethod
    def _parse_stage_names(raw):
        if isinstance(raw, str):
            return [n.strip() for n in raw.split(",") if n.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(n).strip() for n in raw if str(n).strip()]
        return []

    def _stage_index(self):
        span = max(1, round(1.0 / self.level_step))
        return int(np.clip(round(self.level / self.level_step), 0, span))

    def _stage_name(self, idx):
        """Task-supplied label for a stage, or None when the task configured no names."""
        if 0 <= idx < len(self.stage_names):
            return self.stage_names[idx]
        return None

    def _display_fields(self):
        """Canonical derived fields for logs + dashboard so nothing is recomputed downstream."""
        idx = self._stage_index()
        updates_since = int(self.last_training_updates) - int(
            self.policy_updates_at_last_transition
        )
        return {
            "stage_index": idx,
            "stage_name": self._stage_name(idx),
            "promotion_required": self.required_evaluations,
            "demotion_required": self.demotion_required_evaluations,
            "last_training_updates": self.last_training_updates,
            "updates_since_last_transition": updates_since,
            "cooldown_active": bool(updates_since < self.min_policy_updates),
        }

    def observe_evaluation(self, episode, summary):
        """Return True exactly when this evaluation promotes the curriculum. Demotion is always
        active; PROMOTION is frozen until min_policy_updates NEW optimizer updates have accrued
        since the last transition (not cumulative)."""
        if not self.enabled:
            return False
        training_updates_raw = summary.get("training_updates")
        has_updates = training_updates_raw is not None
        training_updates = int(training_updates_raw) if has_updates else 0
        value = self.evaluation_metric(summary)
        with self._lock:
            self.last_transition = None
            self.last_evaluation_episode = int(episode)
            self.last_metric_value = value
            if has_updates:
                self.last_training_updates = training_updates
            if not np.isfinite(value):
                self.confirmations = 0
                self.demotion_confirmations = 0
                self._save_locked()
                return False

            # When the update count is not reported, do not freeze (updates_since = min).
            updates_since = (
                training_updates - self.policy_updates_at_last_transition
                if has_updates
                else self.min_policy_updates
            )

            # Demotion (safety) stays active even during the promotion cooldown.
            if (
                self.demotion_threshold is not None
                and self.level > 0.0
                and value < self.demotion_threshold
            ):
                self.confirmations = 0
                self.demotion_confirmations += 1
                if self.demotion_confirmations < self.demotion_required_evaluations:
                    self._save_locked()
                    return False
                old_level = self.level
                self.level = self._snap_level(self.level - self.level_step)
                self.demotion_confirmations = 0
                if has_updates:
                    self.policy_updates_at_last_transition = training_updates
                self.last_demotion_episode = int(episode)
                self.last_transition = "demoted"
                self._save_locked()
                print(
                    "Adaptive curriculum demoted "
                    f"{old_level:.3f} -> {self.level:.3f} at episode={episode} "
                    f"after {self.metric}={value:.4f}.",
                    flush=True,
                )
                return False

            self.demotion_confirmations = 0
            if value < self.threshold:
                self.confirmations = 0
                self._save_locked()
                return False

            # Promotion cooldown: freeze (and do not bank confirmations) until enough NEW updates.
            if updates_since < self.min_policy_updates:
                self.confirmations = 0
                self._save_locked()
                return False

            self.confirmations += 1
            if self.level >= 1.0 or self.confirmations < self.required_evaluations:
                self._save_locked()
                return False

            old_level = self.level
            self.level = self._snap_level(self.level + self.level_step)
            self.confirmations = 0
            if has_updates:
                self.policy_updates_at_last_transition = training_updates
            self.last_promotion_episode = int(episode)
            self.last_promotion_checkpoint = None
            self.last_transition = "promoted"
            self._save_locked()
            print(
                "Adaptive curriculum promoted "
                f"{old_level:.3f} -> {self.level:.3f} at episode={episode} "
                f"after {self.metric}={value:.4f} "
                f"(+{updates_since} updates since last transition).",
                flush=True,
            )
            return True

    def record_promotion_checkpoint(self, checkpoint_path):
        if not self.enabled:
            return
        with self._lock:
            self.last_promotion_checkpoint = (
                str(checkpoint_path) if checkpoint_path is not None else None
            )
            self._save_locked()

    def snapshot(self):
        with self._lock:
            return {
                "version": self.STATE_VERSION,
                "enabled": self.enabled,
                "level": self.level,
                "level_step": self.level_step,
                "metric": self.metric,
                "threshold": self.threshold,
                "required_evaluations": self.required_evaluations,
                "min_policy_updates": self.min_policy_updates,
                "confirmations": self.confirmations,
                "demotion_threshold": self.demotion_threshold,
                "demotion_required_evaluations": self.demotion_required_evaluations,
                "demotion_confirmations": self.demotion_confirmations,
                "policy_updates_at_last_transition": self.policy_updates_at_last_transition,
                "last_transition": self.last_transition,
                "last_evaluation_episode": self.last_evaluation_episode,
                "last_metric_value": self.last_metric_value,
                "last_promotion_episode": self.last_promotion_episode,
                "last_promotion_checkpoint": self.last_promotion_checkpoint,
                "last_demotion_episode": self.last_demotion_episode,
                **self._display_fields(),
            }

    def _load(self):
        if not self.state_path.is_file():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(
                f"WARNING: could not restore adaptive curriculum state: {exc}",
                flush=True,
            )
            return
        self.level = float(np.clip(data.get("level", self.level), 0.0, 1.0))
        self.confirmations = max(0, int(data.get("confirmations", 0)))
        self.demotion_confirmations = max(
            0, int(data.get("demotion_confirmations", 0))
        )
        self.policy_updates_at_last_transition = max(
            0, int(data.get("policy_updates_at_last_transition", 0))
        )
        self.last_transition = data.get("last_transition")
        self.last_evaluation_episode = data.get("last_evaluation_episode")
        self.last_metric_value = data.get("last_metric_value")
        self.last_promotion_episode = data.get("last_promotion_episode")
        self.last_promotion_checkpoint = data.get("last_promotion_checkpoint")
        self.last_demotion_episode = data.get("last_demotion_episode")
        print(
            f"Restored adaptive curriculum level={self.level:.3f} "
            f"from {self.state_path}.",
            flush=True,
        )

    def _save(self):
        with self._lock:
            self._save_locked()

    def _save_locked(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.STATE_VERSION,
            "enabled": self.enabled,
            "level": self.level,
            "level_step": self.level_step,
            "metric": self.metric,
            "threshold": self.threshold,
            "required_evaluations": self.required_evaluations,
            "min_policy_updates": self.min_policy_updates,
            "confirmations": self.confirmations,
            "demotion_threshold": self.demotion_threshold,
            "demotion_required_evaluations": self.demotion_required_evaluations,
            "demotion_confirmations": self.demotion_confirmations,
            "policy_updates_at_last_transition": self.policy_updates_at_last_transition,
            "last_transition": self.last_transition,
            "last_evaluation_episode": self.last_evaluation_episode,
            "last_metric_value": self.last_metric_value,
            "last_promotion_episode": self.last_promotion_episode,
            "last_promotion_checkpoint": self.last_promotion_checkpoint,
            "last_demotion_episode": self.last_demotion_episode,
            **self._display_fields(),
        }
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)


def ensure_adaptive_curriculum(args):
    controller = getattr(args, "_adaptive_curriculum_controller", None)
    if controller is None:
        controller = AdaptiveCurriculumController(args)
        args._adaptive_curriculum_controller = controller
    return controller


def scenario_curriculum_config(args):
    controller = getattr(args, "_adaptive_curriculum_controller", None)
    if controller is None:
        return {}
    return controller.scenario_config()
