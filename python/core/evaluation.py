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


def episode_agent_infos(info, multi_agent=False):
    if multi_agent:
        return [item for item in info.get("per_agent_infos", []) if isinstance(item, dict)]
    agent_info = info.get("agent_info", {})
    return [agent_info] if isinstance(agent_info, dict) else []


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


def build_evaluation_summary(rewards, steps, successes, trials):
    if not rewards:
        return None
    reward_values = np.asarray(rewards, dtype=np.float32)
    step_values = np.asarray(steps, dtype=np.float32)
    return {
        "episodes": len(rewards),
        "successes": int(successes),
        "trials": int(trials),
        "success_rate": float(successes / max(trials, 1)),
        "reward_mean": float(np.mean(reward_values)),
        "reward_min": float(np.min(reward_values)),
        "reward_max": float(np.max(reward_values)),
        "steps_mean": float(np.mean(step_values)),
        "steps_min": int(np.min(step_values)),
        "steps_max": int(np.max(step_values)),
    }


def print_evaluation_summary(summary):
    if summary is None:
        return
    print(
        "Evaluation summary\n"
        f"  episodes  {summary['episodes']}\n"
        f"  success   {summary['successes']}/{summary['trials']} ({summary['success_rate']:.2%})\n"
        f"  reward    mean:{summary['reward_mean']:.4f} "
        f"range:[{summary['reward_min']:.4f},{summary['reward_max']:.4f}]\n"
        f"  steps     mean:{summary['steps_mean']:.1f} "
        f"range:[{summary['steps_min']},{summary['steps_max']}]",
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
