import json
from pathlib import Path

import numpy as np


SUCCESS_EVENTS = {
    "level_cleared",
    "target_reached",
    "finish_reached",
    "goal_scored",
    "point_won",
    "team_won",
    "success",
}
SUCCESS_TERMINAL_REASONS = SUCCESS_EVENTS
POSE_THRESHOLD_KEYS = (
    "distance_m",
    "orientation_deg",
    "hold_physics_frames",
    "max_joint_speed_rad_s",
)


def _finite_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def new_episode_agent_diagnostics():
    """Create per-agent state used to diagnose a complete evaluation episode."""
    return {
        "progress": [],
        "position_error": [],
        "orientation_error": [],
        "max_joint_speed": [],
        "hold_frames": [],
        "success_thresholds": {},
        "position_gate_reached": False,
        "pose_gate_reached": False,
        "stillness_gate_reached": False,
        "collided": False,
        "progress_stalled": False,
        "collision_sources": [],
        "terminal_reasons": [],
    }


def update_episode_agent_diagnostics(state, agent_info):
    """Accumulate metrics and simultaneous pose gates from one environment step."""
    if not isinstance(agent_info, dict):
        return state

    values = {
        "progress": _finite_float(agent_info.get("track_progress")),
        "position_error": _finite_float(agent_info.get("position_error_m")),
        "orientation_error": _finite_float(agent_info.get("orientation_error_deg")),
        "max_joint_speed": _finite_float(
            agent_info.get("max_joint_speed", agent_info.get("max_joint_speed_rad_s"))
        ),
        "hold_frames": _finite_float(agent_info.get("hold_frames")),
    }
    for key, value in values.items():
        if value is not None:
            state[key].append(value)

    thresholds = agent_info.get("success_thresholds", {})
    if isinstance(thresholds, dict):
        for key in POSE_THRESHOLD_KEYS:
            value = _finite_float(thresholds.get(key))
            if value is not None:
                state["success_thresholds"][key] = value
        if "require_still" in thresholds:
            state["success_thresholds"]["require_still"] = bool(
                thresholds["require_still"]
            )

    distance_limit = state["success_thresholds"].get("distance_m")
    angle_limit = state["success_thresholds"].get("orientation_deg")
    speed_limit = state["success_thresholds"].get("max_joint_speed_rad_s")
    require_still = state["success_thresholds"].get("require_still", True)
    position_ok = (
        values["position_error"] is not None
        and distance_limit is not None
        and values["position_error"] <= distance_limit
    )
    orientation_ok = (
        values["orientation_error"] is not None
        and angle_limit is not None
        and values["orientation_error"] <= angle_limit
    )
    speed_ok = (
        not require_still
        or (
            values["max_joint_speed"] is not None
            and speed_limit is not None
            and values["max_joint_speed"] <= speed_limit
        )
    )
    state["position_gate_reached"] |= position_ok
    state["pose_gate_reached"] |= position_ok and orientation_ok
    state["stillness_gate_reached"] |= position_ok and orientation_ok and speed_ok
    state["collided"] |= bool(agent_info.get("collided", False))
    state["progress_stalled"] |= bool(agent_info.get("progress_stalled", False))

    collision_source = str(agent_info.get("collision_source", "")).strip()
    if collision_source and collision_source not in state["collision_sources"]:
        state["collision_sources"].append(collision_source)
    terminal_reason = str(agent_info.get("terminal_reason", "")).strip()
    if terminal_reason and terminal_reason not in state["terminal_reasons"]:
        state["terminal_reasons"].append(terminal_reason)
    return state


def finalize_episode_agent_diagnostics(state):
    """Return JSON-safe metrics for one agent after an evaluation episode."""
    return {
        "progress_mean": (
            float(np.mean(state["progress"])) if state["progress"] else None
        ),
        "progress_max": (
            float(np.max(state["progress"])) if state["progress"] else None
        ),
        "progress_final": (
            float(state["progress"][-1]) if state["progress"] else None
        ),
        "position_error_mean": (
            float(np.mean(state["position_error"]))
            if state["position_error"]
            else None
        ),
        "position_error_min": (
            float(np.min(state["position_error"]))
            if state["position_error"]
            else None
        ),
        "position_error_final": (
            float(state["position_error"][-1])
            if state["position_error"]
            else None
        ),
        "orientation_error_mean": (
            float(np.mean(state["orientation_error"]))
            if state["orientation_error"]
            else None
        ),
        "orientation_error_min": (
            float(np.min(state["orientation_error"]))
            if state["orientation_error"]
            else None
        ),
        "orientation_error_final": (
            float(state["orientation_error"][-1])
            if state["orientation_error"]
            else None
        ),
        "max_joint_speed_mean": (
            float(np.mean(state["max_joint_speed"]))
            if state["max_joint_speed"]
            else None
        ),
        "max_joint_speed_min": (
            float(np.min(state["max_joint_speed"]))
            if state["max_joint_speed"]
            else None
        ),
        "max_joint_speed_final": (
            float(state["max_joint_speed"][-1])
            if state["max_joint_speed"]
            else None
        ),
        "hold_frames_max": (
            float(np.max(state["hold_frames"])) if state["hold_frames"] else None
        ),
        "hold_frames_final": (
            float(state["hold_frames"][-1]) if state["hold_frames"] else None
        ),
        "success_thresholds": dict(state["success_thresholds"]),
        "position_gate_reached": bool(state["position_gate_reached"]),
        "pose_gate_reached": bool(state["pose_gate_reached"]),
        "stillness_gate_reached": bool(state["stillness_gate_reached"]),
        "collided": bool(state["collided"]),
        "progress_stalled": bool(state["progress_stalled"]),
        "collision_sources": list(state["collision_sources"]),
        "terminal_reasons": list(state["terminal_reasons"]),
    }


def classify_episode_failure(record):
    """Explain the first unsatisfied gate of a failed evaluation trial."""
    if bool(record.get("success", False)):
        return "success"
    reasons = {
        str(value).strip().lower()
        for value in record.get("terminal_reasons", [])
        if str(value).strip()
    }
    if bool(record.get("collided", False)) or any(
        "collision" in reason for reason in reasons
    ):
        return "collision"

    thresholds = record.get("success_thresholds", {})
    if isinstance(thresholds, dict) and thresholds:
        if (
            thresholds.get("distance_m") is not None
            and not bool(record.get("position_gate_reached", False))
        ):
            return "position_gate"
        if (
            thresholds.get("orientation_deg") is not None
            and not bool(record.get("pose_gate_reached", False))
        ):
            return "orientation_gate"
        if (
            bool(thresholds.get("require_still", True))
            and thresholds.get("max_joint_speed_rad_s") is not None
            and not bool(record.get("stillness_gate_reached", False))
        ):
            return "stillness_gate"
        required_hold = _finite_float(thresholds.get("hold_physics_frames"))
        achieved_hold = _finite_float(record.get("hold_frames_max"))
        if (
            required_hold is not None
            and (achieved_hold is None or achieved_hold < required_hold)
        ):
            return "hold_gate"

    if bool(record.get("progress_stalled", False)) or any(
        "stall" in reason for reason in reasons
    ):
        return "stall"
    if any(
        "time_limit" in reason or "truncated" in reason or "max_step" in reason
        for reason in reasons
    ):
        return "time_limit"
    return "other"


def _records_summary(records):
    records = [record for record in records if isinstance(record, dict)]
    successes = sum(bool(record.get("success", False)) for record in records)
    summary = {
        "successes": int(successes),
        "trials": len(records),
        "success_rate": float(successes / max(len(records), 1)),
    }
    failures = {}
    for record in records:
        failure = str(
            record.get("failure_reason") or classify_episode_failure(record)
        )
        if failure != "success":
            failures[failure] = failures.get(failure, 0) + 1
    if failures:
        summary["failure_reasons"] = failures
    for key, reducer in {
        "reward": np.mean,
        "progress_mean": np.mean,
        "progress_max": np.max,
        "position_error_mean": np.mean,
        "position_error_min": np.min,
        "orientation_error_mean": np.mean,
        "orientation_error_min": np.min,
        "hold_frames_max": np.max,
    }.items():
        values = np.asarray(
            [
                float(record[key])
                for record in records
                if record.get(key) is not None
            ],
            dtype=np.float32,
        )
        values = values[np.isfinite(values)]
        if values.size:
            summary["reward_mean" if key == "reward" else key] = float(
                reducer(values)
            )
    return summary


def episode_agent_infos(info, multi_agent=False):
    if multi_agent:
        return [item for item in info.get("per_agent_infos", []) if isinstance(item, dict)]
    agent_info = info.get("agent_info", {})
    return [agent_info] if isinstance(agent_info, dict) else []


def episode_reset_modes(info, multi_agent=False):
    """Return one task reset mode per controlled agent when the scenario exposes it."""
    return episode_reset_values(info, "task_reset_mode", multi_agent)


def episode_reset_values(info, key, multi_agent=False):
    """Return one normalized reset metadata value per controlled agent.

    Scalar text labels only: the value is coerced with ``str(...).strip().lower()``.
    For structured metadata (lists/dicts such as ``active_regions``) use
    ``episode_reset_raw_values`` instead — this helper would stringify them.
    """
    reset_infos = episode_agent_infos(info, multi_agent)
    if not multi_agent and not reset_infos:
        reset_infos = [info] if isinstance(info, dict) else []
    values = []
    for agent_info in reset_infos:
        reset_info = agent_info.get("reset", {})
        value = (
            str(reset_info.get(key, "")).strip().lower()
            if isinstance(reset_info, dict)
            else ""
        )
        values.append(value)
    return values


def episode_reset_raw_values(info, key, multi_agent=False):
    """Return one RAW reset metadata value per controlled agent, preserving structure.

    Unlike ``episode_reset_values`` (which flattens everything into a lowercase text
    label), this keeps lists/dicts intact — e.g. ``active_regions=["easy","medium",
    "hard"]`` stays a list instead of becoming the string ``"['easy', ...]"``. Missing
    values yield ``None`` so callers can tell absent apart from empty.
    """
    reset_infos = episode_agent_infos(info, multi_agent)
    if not multi_agent and not reset_infos:
        reset_infos = [info] if isinstance(info, dict) else []
    values = []
    for agent_info in reset_infos:
        reset_info = agent_info.get("reset", {})
        value = reset_info.get(key, None) if isinstance(reset_info, dict) else None
        values.append(value)
    return values


def agent_succeeded(agent_info):
    events = agent_info.get("events", {})
    successful_event = isinstance(events, dict) and any(
        isinstance(events.get(name, 0.0), (bool, int, float, np.number))
        and float(events.get(name, 0.0)) > 0.0
        for name in SUCCESS_EVENTS
    )
    return (
        bool(agent_info.get("target_reached", False))
        or bool(agent_info.get("finish_reached", False))
        or str(agent_info.get("terminal_reason", "")) in SUCCESS_TERMINAL_REASONS
        or successful_event
    )


def summarize_episode_outcome(info, multi_agent=False):
    agent_infos = episode_agent_infos(info, multi_agent)
    successes = sum(int(agent_succeeded(agent_info)) for agent_info in agent_infos)
    reasons = [
        str(agent_info.get("terminal_reason"))
        for agent_info in agent_infos
        if agent_info.get("terminal_reason")
    ]
    return successes, len(agent_infos), reasons


def build_evaluation_summary(
    rewards,
    steps,
    successes,
    trials,
    diagnostics=None,
    reset_outcomes=None,
    episode_records=None,
    expected_regions=None,
):
    if not rewards:
        return None
    reward_values = np.asarray(rewards, dtype=np.float32)
    step_values = np.asarray(steps, dtype=np.float32)
    collided_count = sum(
        1 for record in (episode_records or []) if bool(record.get("collided", False))
    )
    collision_rate = float(collided_count / max(len(episode_records or []), 1))
    summary = {
        "episodes": len(rewards),
        "successes": int(successes),
        "trials": int(trials),
        "success_rate": float(successes / max(trials, 1)),
        "collided": int(collided_count),
        "collision_rate": collision_rate,
        "reward_mean": float(np.mean(reward_values)),
        "reward_min": float(np.min(reward_values)),
        "reward_max": float(np.max(reward_values)),
        "steps_mean": float(np.mean(step_values)),
        "steps_min": int(np.min(step_values)),
        "steps_max": int(np.max(step_values)),
    }
    diagnostics = diagnostics or {}
    reducers = {
        "progress_mean": np.mean,
        "progress_max": np.max,
        "position_error_mean": np.mean,
        "position_error_min": np.min,
        "orientation_error_mean": np.mean,
        "orientation_error_min": np.min,
        "hold_frames_max": np.max,
    }
    for key, reducer in reducers.items():
        values = np.asarray(diagnostics.get(key, []), dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            summary[key] = float(reducer(values))
    reset_modes = {}
    target_regions = {}
    reset_metric_reducers = {
        "reward": np.mean,
        "progress_mean": np.mean,
        "progress_max": np.max,
        "position_error_mean": np.mean,
        "position_error_min": np.min,
        "orientation_error_mean": np.mean,
        "orientation_error_min": np.min,
        "hold_frames_max": np.max,
    }
    for label, outcomes in sorted((reset_outcomes or {}).items()):
        clean_label = str(label).strip().lower()
        if not clean_label:
            continue
        records = [
            value if isinstance(value, dict) else {"success": bool(value)}
            for value in outcomes
        ]
        if not records:
            continue
        mode_successes = sum(bool(record.get("success", False)) for record in records)
        mode_summary = {
            "successes": int(mode_successes),
            "trials": len(records),
            "success_rate": float(mode_successes / len(records)),
        }
        for key, reducer in reset_metric_reducers.items():
            values = np.asarray(
                [
                    float(record[key])
                    for record in records
                    if record.get(key) is not None
                ],
                dtype=np.float32,
            )
            values = values[np.isfinite(values)]
            if values.size:
                output_key = "reward_mean" if key == "reward" else key
                mode_summary[output_key] = float(reducer(values))
        if clean_label.startswith("target_region:"):
            region_name = clean_label.partition(":")[2].strip()
            if region_name:
                target_regions[region_name] = mode_summary
        else:
            reset_modes[clean_label] = mode_summary
    if reset_modes:
        summary["reset_modes"] = reset_modes
    regular = reset_modes.get("regular")
    if regular is not None:
        summary["regular_success_rate"] = float(regular["success_rate"])
        summary["selection_success_rate"] = float(regular["success_rate"])
        for key in (
            "reward_mean",
            "progress_mean",
            "progress_max",
            "position_error_mean",
            "position_error_min",
            "orientation_error_mean",
            "orientation_error_min",
            "hold_frames_max",
        ):
            if key in regular:
                summary[f"regular_{key}"] = float(regular[key])
    else:
        summary["selection_success_rate"] = float(summary["success_rate"])
    expected = {str(r).lower() for r in (expected_regions or [])}
    if target_regions:
        summary["target_regions"] = target_regions
        region_success_floor = min(
            float(values["success_rate"])
            for values in target_regions.values()
        )
        # Fail closed: a required active region that produced no evaluation episodes drops the
        # floor to 0 instead of taking the minimum only over the regions that happened to appear.
        if expected and not expected.issubset(set(target_regions.keys())):
            region_success_floor = 0.0
        summary["region_success_floor"] = region_success_floor
        summary["selection_success_rate"] = min(
            float(summary["selection_success_rate"]),
            region_success_floor,
        )
    elif expected:
        summary["region_success_floor"] = 0.0
        summary["selection_success_rate"] = 0.0
    episode_records = [
        dict(record)
        for record in (episode_records or [])
        if isinstance(record, dict)
    ]
    if episode_records:
        failure_reasons = {}
        target_cells = {}
        for record in episode_records:
            failure = classify_episode_failure(record)
            record["failure_reason"] = failure
            if failure != "success":
                failure_reasons[failure] = failure_reasons.get(failure, 0) + 1
            region = str(record.get("target_region", "")).strip().lower()
            cell = str(record.get("target_cell", "")).strip().lower()
            if region in {"", "bootstrap", "marker"} or cell in {"", "-1"}:
                continue
            target_cells.setdefault(region, {}).setdefault(cell, []).append(record)
        summary["episode_records"] = episode_records
        if failure_reasons:
            summary["failure_reasons"] = failure_reasons
        if target_cells:
            summary["target_cells"] = {
                region: {
                    cell: _records_summary(records)
                    for cell, records in sorted(cells.items())
                }
                for region, cells in sorted(target_cells.items())
            }
    return summary


def print_evaluation_summary(summary):
    if summary is None:
        return
    details = ""
    if "progress_mean" in summary:
        details += (
            f"\n  progress  mean:{summary['progress_mean']:.4f} "
            f"max:{summary.get('progress_max', summary['progress_mean']):.4f}"
        )
    if "position_error_mean" in summary:
        details += (
            f"\n  pose      position mean:{summary['position_error_mean']:.4f}m "
            f"min:{summary.get('position_error_min', summary['position_error_mean']):.4f}m"
        )
    if "orientation_error_mean" in summary:
        details += (
            f"  orientation mean:{summary['orientation_error_mean']:.1f}deg "
            f"min:{summary.get('orientation_error_min', summary['orientation_error_mean']):.1f}deg"
        )
    if "hold_frames_max" in summary:
        details += f"  hold_max:{summary['hold_frames_max']:.0f}"
    reset_modes = summary.get("reset_modes", {})
    if reset_modes:
        reset_details = []
        for mode, values in sorted(reset_modes.items()):
            mode_details = (
                f"{mode}:{values['successes']}/{values['trials']} "
                f"({values['success_rate']:.2%})"
            )
            if "progress_mean" in values:
                mode_details += f" progress:{values['progress_mean']:.3f}"
            if "reward_mean" in values:
                mode_details += f" reward:{values['reward_mean']:.3f}"
            reset_details.append(mode_details)
        details += f"\n  resets    {' | '.join(reset_details)}"
    target_regions = summary.get("target_regions", {})
    if target_regions:
        region_details = []
        for region, values in sorted(target_regions.items()):
            region_details.append(
                f"{region}:{values['successes']}/{values['trials']} "
                f"({values['success_rate']:.2%})"
            )
        details += (
            f"\n  regions   {' | '.join(region_details)} "
            f"| floor:{summary['region_success_floor']:.2%}"
        )
    failure_reasons = summary.get("failure_reasons", {})
    if failure_reasons:
        details += "\n  failures  " + " | ".join(
            f"{name}:{count}"
            for name, count in sorted(
                failure_reasons.items(), key=lambda item: (-item[1], item[0])
            )
        )
    target_cells = summary.get("target_cells", {})
    cell_summaries = [
        (float(values.get("success_rate", 0.0)), region, cell, values)
        for region, cells in target_cells.items()
        if isinstance(cells, dict)
        for cell, values in cells.items()
        if isinstance(values, dict)
    ]
    if cell_summaries:
        _, region, cell, values = min(cell_summaries)
        details += (
            f"\n  worst cell {region}/{cell}:"
            f"{values['successes']}/{values['trials']} "
            f"({values['success_rate']:.2%})"
        )
    print(
        "Evaluation summary\n"
        f"  episodes  {summary['episodes']}\n"
        f"  success   {summary['successes']}/{summary['trials']} ({summary['success_rate']:.2%})\n"
        f"  reward    mean:{summary['reward_mean']:.4f} "
        f"range:[{summary['reward_min']:.4f},{summary['reward_max']:.4f}]\n"
        f"  steps     mean:{summary['steps_mean']:.1f} "
        f"range:[{summary['steps_min']},{summary['steps_max']}]"
        f"{details}",
        flush=True,
    )


def write_evaluation_summary(path, summary):
    if not path or summary is None:
        return
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
