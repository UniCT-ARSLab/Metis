"""M5 - generate IK expert reach-and-hold demonstrations for the stable OpenArm scene.

The IK controller is ONLY the expert action generator: it drives the arm from home to a
Stage-A target through the NORMAL Agent/Bridge path (apply_action -> set_joint_target_velocity),
never by setting joints/transforms directly. What lands in the dataset is exactly what the policy
would see and do: the 27-dim policy observation and the 7-dim joint-velocity action that the IK
produced (info["agent_info"]["applied_action"]). No IK pose / privileged state is ever recorded.

Contract enforced here (M5):
  - trajectories are home -> Stage-A only (the stable scene has reverse/bootstrap disabled);
  - every Easy cell is balanced by ACCEPTED demos (a rejected episode never advances a cell);
  - a trajectory is accepted ONLY if it terminates in a real hold-success (target_reached),
    with no collision and no truncation; rejected trajectories are counted, not stored;
  - train and validation use DISJOINT seed ranges, so pose_id (= region:cell:seed) never
    collides -> no trajectory is ever shared across the two sets;
  - the dataset stores complete transitions (obs, action, reward, next_obs, terminated,
    truncated) plus per-episode reset metadata (seed, region, cell, pose_id);
  - a manifest records seed ranges, scene + Godot version, obs/action dims + bounds, the
    live Stage-A gate, the reward contract and per-cell statistics;
  - after writing, the dataset is reloaded and a sample is replayed through the SAME M3 task
    (recorded actions, same reset) to confirm it reproduces target_reached + a coherent
    terminal reward before it is considered valid.

Stop point: this script generates + replay-validates + reports. It does NOT train BC or SAC.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python/

from core.evaluation import (  # noqa: E402
    agent_succeeded,
    finalize_episode_agent_diagnostics,
    new_episode_agent_diagnostics,
    update_episode_agent_diagnostics,
)
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402

SCENE = "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"
REGION = "easy"
DEFAULT_MANIFEST = (
    "godot/scenarios/robotarms/openarm_reach_hold_region_manifest.json"
)

# Reward contract mirror (authoritative source = the scene/agent .tscn, guarded by
# godot/tests/test_openarm_reach_hold_reward.gd). Recorded in the dataset manifest so a
# trainer knows exactly which reward shaped these demos.
REWARD_CONTRACT = {
    "authoritative_source": (
        "openarm_reach_hold_scenario.tscn + openarm_reach_hold_agent.tscn "
        "(guarded by godot/tests/test_openarm_reach_hold_reward.gd)"
    ),
    "scenario_rewards": {
        "progress_delta": {
            "forward": 10.0,
            "backward": 10.0,
            "orientation_weight": 0.10,
            "mode": "POSITION_GATED",
        },
        "target_distance_potential": {
            "scale": 5.0,
            "radius": 0.06,
            "retreat_multiplier": 1.25,
        },
        "goal_reward": 50.0,
        "collision_penalty": -25.0,
        "self_collision_penalty": -35.0,
        "terminal_failure": {
            "base": -10.0,
            "reasons": ["progress_stalled"],
            "remaining_progress_penalty": 0.0,
            "penalize_truncation": False,
            "penalize_unclassified_terminal": False,
        },
        "no_progress_step_penalty": 0.0,
    },
    "agent_rewards": {
        "joint_limit_weight": 0.01,
        "smoothness_penalty_scale": -0.001,
        "time_penalty_per_decision_step": -0.001,
    },
}


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested without Godot)
# --------------------------------------------------------------------------------------
def pose_id(region, cell, seed):
    """Deterministic identity of a pose. MUST mirror the Godot side
    (openarm_scenario.gd: "%s:%d:%d" % [region, cell, seed])."""
    return "%s:%d:%d" % (str(region).strip().lower(), int(cell), int(seed))


def next_cell_to_fill(accepted_counts, cells, target_n, exhausted=None):
    """Cell with the fewest ACCEPTED demos still below target_n; None when all met/exhausted.

    Balancing is by accepted demos, not attempts: a rejected episode leaves the count
    unchanged so the cell keeps being retried until it reaches target_n or is exhausted.
    Deterministic: ties break by ascending cell id.
    """
    exhausted = exhausted or set()
    pending = [
        (int(accepted_counts.get(c, 0)), int(c))
        for c in cells
        if c not in exhausted and int(accepted_counts.get(c, 0)) < target_n
    ]
    if not pending:
        return None
    pending.sort()
    return pending[0][1]


def accept_episode(reached, truncated, collided):
    """Accept only a clean, complete hold-success: reached target, no truncation, no collision."""
    return bool(reached) and not bool(truncated) and not bool(collided)


def load_easy_cells(manifest_path):
    """Derive the Easy cell set from the region manifest (never hardcoded)."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cells = data["active_shells_path_confirmed"]["easy"]
    return sorted(int(c) for c in cells)


def per_cell_stats(summaries, cells):
    """Per-cell dataset statistics over ALL attempts of one set."""
    stats = {}
    for c in cells:
        ce = [e for e in summaries if int(e["cell"]) == int(c)]
        acc = [e for e in ce if e["accepted"]]
        finals = [
            e["final_distance"] for e in acc if e.get("final_distance") is not None
        ]
        stats[str(c)] = {
            "attempts": len(ce),
            "accepted": len(acc),
            "collisions": int(sum(1 for e in ce if e["collided"])),
            "discards": len(ce) - len(acc),
            "mean_len_accepted": float(np.mean([e["n_steps"] for e in acc]))
            if acc
            else None,
            "mean_reward_accepted": float(np.mean([e["ep_reward"] for e in acc]))
            if acc
            else None,
            "mean_final_distance_accepted": float(np.mean(finals)) if finals else None,
        }
    return stats


def episode_summary(ep):
    """The stat-relevant subset of a run_episode result (no transition arrays)."""
    return {
        key: ep[key]
        for key in (
            "cell",
            "seed",
            "pose_id",
            "accepted",
            "collided",
            "reached",
            "truncated",
            "reject_reason",
            "n_steps",
            "ep_reward",
            "final_distance",
        )
    }


def execution_failure_breakdown(summaries):
    """Counts of execution reject reasons over all attempted (planned) poses."""
    out = {}
    for e in summaries:
        if not e["accepted"]:
            out[e["reject_reason"]] = out.get(e["reject_reason"], 0) + 1
    return out


def flatten_episodes(accepted_eps):
    """Flatten accepted episodes into per-transition arrays with episode grouping intact."""
    cols = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "next_obs": [],
        "terminated": [],
        "truncated": [],
        "dones": [],
        "episode_indices": [],
        "step_indices": [],
        "cells": [],
        "seeds": [],
        "pose_ids": [],
        "regions": [],
    }
    for ep_index, ep in enumerate(accepted_eps):
        for step_index, t in enumerate(ep["transitions"]):
            cols["obs"].append(t["obs"])
            cols["actions"].append(t["action"])
            cols["rewards"].append(t["reward"])
            cols["next_obs"].append(t["next_obs"])
            cols["terminated"].append(t["terminated"])
            cols["truncated"].append(t["truncated"])
            cols["dones"].append(bool(t["terminated"] or t["truncated"]))
            cols["episode_indices"].append(ep_index)
            cols["step_indices"].append(step_index)
            cols["cells"].append(int(ep["cell"]))
            cols["seeds"].append(int(ep["seed"]))
            cols["pose_ids"].append(ep["pose_id"])
            cols["regions"].append(REGION)
    return cols


# --------------------------------------------------------------------------------------
# Godot-driven generation
# --------------------------------------------------------------------------------------
def _demo_config(agent_id, cell, curriculum_level, waypoints=None):
    cfg = {
        "demo_mode": True,
        "demo_forced_region": REGION,
        "demo_forced_cell": int(cell),
        "curriculum_level": float(curriculum_level),
        "evaluation_mode": False,
        "recording_mode": True,
        "recording_agent_id": agent_id,
        "manage_agent_cameras": True,
    }
    if waypoints is not None:
        cfg["demo_plan"] = [list(map(float, w)) for w in waypoints]
    return cfg


def load_plans(path):
    """Load the planner output; group collision-free plans by cell (sorted by seed)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_cell = {}
    for pose_id, plan in data.get("plans", {}).items():
        by_cell.setdefault(int(plan["cell"]), []).append(
            {"seed": int(plan["seed"]), "pose_id": pose_id,
             "waypoints": plan["waypoints"]})
    for cell in by_cell:
        by_cell[cell].sort(key=lambda p: p["seed"])
    return by_cell, data.get("planning_failures", {})


def _applied_action(agent_info, action_size):
    applied = agent_info.get("applied_action")
    if applied is None:
        return np.zeros((action_size,), dtype=np.float32)
    return np.asarray(applied, dtype=np.float32).reshape(action_size)


def _reject_reason(reached, truncated, collided):
    if collided:
        return "collision"
    if truncated:
        return "truncated"
    if not reached:
        return "not_reached"
    return ""


def run_episode(env, agent_id, cell, seed, waypoints, pose_id_hint, curriculum_level,
                max_steps):
    """Execute one planned reach-and-hold episode (closed-loop plan-follower) and record it."""
    env.configure(**_demo_config(agent_id, cell, curriculum_level, waypoints))
    obs, info = env.reset(seed=int(seed))
    reset_info = info.get("agent_info", {}).get("reset", {})
    got_cell = int(reset_info.get("target_cell", -1))
    if got_cell != int(cell):
        raise RuntimeError(
            f"forced cell {cell} not honoured (scenario returned {got_cell}); fail-closed"
        )
    pid = str(reset_info.get("target_pose_id", pose_id_hint))
    if pid != pose_id_hint:
        raise RuntimeError(
            f"pose_id mismatch: plan {pose_id_hint} vs scenario {pid} (target drift)"
        )

    diag = new_episode_agent_diagnostics()
    transitions = []
    reached = False
    terminated = False
    truncated = False
    action_bound_violation = False
    for _step in range(max_steps):
        next_obs, reward, terminated, truncated, info = env.step("manual")
        agent_info = info.get("agent_info", {})
        action = _applied_action(agent_info, env.action_size)
        if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0 + 1e-4):
            action_bound_violation = True
        transitions.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": action,
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        update_episode_agent_diagnostics(diag, agent_info)
        if agent_succeeded(agent_info):
            reached = True
        obs = next_obs
        if terminated or truncated:
            break

    final = finalize_episode_agent_diagnostics(diag)
    collided = bool(final["collided"])
    reached = reached or ("target_reached" in list(final["terminal_reasons"]))
    accepted = accept_episode(reached, truncated, collided) and not action_bound_violation
    return {
        "cell": int(cell),
        "seed": int(seed),
        "pose_id": pid,
        "accepted": accepted,
        "collided": collided,
        "reached": reached,
        "truncated": bool(truncated),
        "terminated": bool(terminated),
        "reject_reason": (
            "action_out_of_bounds" if action_bound_violation
            else _reject_reason(reached, truncated, collided)),
        "n_steps": len(transitions),
        "ep_reward": float(sum(t["reward"] for t in transitions)),
        "final_distance": final.get("position_error_final"),
        "success_thresholds": dict(final.get("success_thresholds", {})),
        "transitions": transitions if accepted else [],
    }


def generate_set(
    env, agent_id, planned_by_cell, target_n, curriculum_level, max_steps, label,
):
    """Execute the PLANNED poses per cell until target_n are ACCEPTED (or the pool runs out).

    Balance is by accepted demos: a rejected planned pose does not advance the cell; the cell
    keeps consuming its planned pool until target_n accepted or the pool is exhausted (shortfall).
    """
    cells = sorted(planned_by_cell.keys())
    accepted_counts = {c: 0 for c in cells}
    attempts = {c: 0 for c in cells}
    summaries = []
    accepted_eps = []
    stage_a = {}
    for cell in cells:
        for pose in planned_by_cell[cell]:
            if accepted_counts[cell] >= target_n:
                break
            ep = run_episode(
                env, agent_id, cell, pose["seed"], pose["waypoints"],
                pose["pose_id"], curriculum_level, max_steps)
            attempts[cell] += 1
            summaries.append(episode_summary(ep))
            if ep["accepted"]:
                accepted_counts[cell] += 1
                accepted_eps.append(ep)
                if not stage_a and ep.get("success_thresholds"):
                    stage_a = ep["success_thresholds"]
            done = sum(min(accepted_counts[c], target_n) for c in cells)
            print(
                f"[{label}] cell={cell} seed={pose['seed']} "
                f"{'ACCEPT' if ep['accepted'] else 'reject:' + ep['reject_reason']} "
                f"(steps={ep['n_steps']} rew={ep['ep_reward']:.2f}) "
                f"cell_progress={accepted_counts[cell]}/{target_n} "
                f"total={done}/{target_n * len(cells)}",
                flush=True,
            )
        if accepted_counts[cell] < target_n:
            print(
                f"[{label}] SHORTFALL cell {cell}: {accepted_counts[cell]}/{target_n} "
                f"accepted from {len(planned_by_cell[cell])} planned poses",
                flush=True,
            )
    shortfall = {
        int(c): int(accepted_counts[c]) for c in cells if accepted_counts[c] < target_n
    }
    return {
        "summaries": summaries,
        "accepted_eps": accepted_eps,
        "accepted_counts": {int(c): int(accepted_counts[c]) for c in cells},
        "attempts": {int(c): int(attempts[c]) for c in cells},
        "planned_per_cell": {int(c): len(planned_by_cell[c]) for c in cells},
        "shortfall": shortfall,
        "execution_failures": execution_failure_breakdown(summaries),
        "stage_a": stage_a,
    }


def save_set(path, accepted_eps, env):
    """Write one set (train or val) to an npz with complete transitions + metadata."""
    cols = flatten_episodes(accepted_eps)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "obs": np.asarray(cols["obs"], dtype=np.float32),
        "actions": np.asarray(cols["actions"], dtype=np.float32),
        "rewards": np.asarray(cols["rewards"], dtype=np.float32),
        "next_obs": np.asarray(cols["next_obs"], dtype=np.float32),
        "terminated": np.asarray(cols["terminated"], dtype=np.float32),
        "truncated": np.asarray(cols["truncated"], dtype=np.float32),
        "dones": np.asarray(cols["dones"], dtype=np.float32),
        "episode_indices": np.asarray(cols["episode_indices"], dtype=np.int32),
        "step_indices": np.asarray(cols["step_indices"], dtype=np.int32),
        "cells": np.asarray(cols["cells"], dtype=np.int32),
        "seeds": np.asarray(cols["seeds"], dtype=np.int64),
        "pose_ids": np.asarray(cols["pose_ids"], dtype=str),
        "regions": np.asarray(cols["regions"], dtype=str),
        "action_names": np.asarray(env.action_names, dtype=str),
        "action_type": np.asarray([env.action_type], dtype=str),
        "obs_dim": np.asarray([env.obs_dim], dtype=np.int32),
        "action_size": np.asarray([env.action_size], dtype=np.int32),
    }
    if env.action_type == "continuous":
        arrays["action_low"] = np.asarray(env.action_low, dtype=np.float32)
        arrays["action_high"] = np.asarray(env.action_high, dtype=np.float32)
    np.savez_compressed(output, **arrays)
    return output, len(arrays["actions"]), len(accepted_eps)


def _finite_actions_ok(actions, low, high):
    a = np.asarray(actions, dtype=np.float32)
    if a.size == 0:
        return True
    if not np.all(np.isfinite(a)):
        return False
    lo = np.asarray(low, dtype=np.float32) if low is not None else -1.0
    hi = np.asarray(high, dtype=np.float32) if high is not None else 1.0
    return bool(np.all(a >= lo - 1e-4) and np.all(a <= hi + 1e-4))


# --------------------------------------------------------------------------------------
# Replay-validation: reload the dataset and replay a sample through the M3 task
# --------------------------------------------------------------------------------------
def replay_validate(env, agent_id, dataset_path, curriculum_level, max_steps, sample):
    """Replay recorded actions from the same reset and confirm target_reached + coherent reward."""
    data = np.load(dataset_path, allow_pickle=False)
    episode_indices = data["episode_indices"]
    unique_eps = sorted(set(int(e) for e in episode_indices))
    if sample > 0:
        step = max(1, len(unique_eps) // sample)
        chosen = unique_eps[::step][:sample]
    else:
        chosen = unique_eps
    results = []
    for ep in chosen:
        mask = episode_indices == ep
        actions = data["actions"][mask]
        cell = int(data["cells"][mask][0])
        seed = int(data["seeds"][mask][0])
        recorded_reward = float(np.sum(data["rewards"][mask]))
        recorded_terminal_reward = float(data["rewards"][mask][-1])

        env.configure(**_demo_config(agent_id, cell, curriculum_level))
        env.reset(seed=seed)
        replay_reward = 0.0
        terminal_reward = 0.0
        reached = False
        terminated = truncated = False
        for i in range(min(len(actions), max_steps)):
            _obs, reward, terminated, truncated, info = env.step(
                np.asarray(actions[i], dtype=np.float32)
            )
            replay_reward += float(reward)
            terminal_reward = float(reward)
            agent_info = info.get("agent_info", {})
            if agent_succeeded(agent_info) or str(
                agent_info.get("terminal_reason", "")
            ) == "target_reached":
                reached = True
            if terminated or truncated:
                break
        results.append(
            {
                "episode": int(ep),
                "cell": cell,
                "seed": seed,
                "pose_id": pose_id(REGION, cell, seed),
                "reached": bool(reached),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "recorded_total_reward": recorded_reward,
                "replay_total_reward": replay_reward,
                "recorded_terminal_reward": recorded_terminal_reward,
                "replay_terminal_reward": terminal_reward,
                "terminal_reward_abs_diff": abs(
                    recorded_terminal_reward - terminal_reward
                ),
            }
        )
        print(
            f"[replay] ep={ep} cell={cell} seed={seed} reached={reached} "
            f"rec_rew={recorded_reward:.3f} replay_rew={replay_reward:.3f} "
            f"term_diff={abs(recorded_terminal_reward - terminal_reward):.4f}",
            flush=True,
        )
    return results


def summarize_replay(results, reward_tol):
    reached_all = bool(results) and all(r["reached"] for r in results)
    reward_coherent = all(
        r["terminal_reward_abs_diff"] <= reward_tol for r in results
    )
    return {
        "episodes_replayed": len(results),
        "all_reached_target": reached_all,
        "terminal_reward_coherent": bool(reward_coherent),
        "reward_tolerance": reward_tol,
        "max_terminal_reward_diff": max(
            (r["terminal_reward_abs_diff"] for r in results), default=0.0
        ),
        "details": results,
    }


# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    p.add_argument("--godot-project", default="godot")
    p.add_argument("--godot-scene", default=SCENE)
    p.add_argument("--manifest-cells", default=DEFAULT_MANIFEST,
                   help="Region manifest to derive the Easy cell set from.")
    p.add_argument("--out-dir", default="python/demos/openarm_reach_hold_m5")
    p.add_argument("--train-plans", required=False,
                   help="planner JSON (train seed base) produced by openarm_ik_demo_planner.gd")
    p.add_argument("--val-plans", required=False,
                   help="planner JSON (val seed base, disjoint from train)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6400)
    p.add_argument("--curriculum-level", type=float, default=0.0,
                   help="Stage A = level 0.0 (4cm/45deg/hold20/0.30).")
    p.add_argument("--train-per-cell", type=int, default=20)
    p.add_argument("--val-per-cell", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--replay-sample", type=int, default=8)
    p.add_argument("--replay-reward-tol", type=float, default=0.5)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--connect-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--debug-godot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--replay-only", default=None,
                   help="Skip generation; reload + replay-validate this train npz.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    cells = load_easy_cells(args.manifest_cells)
    print(f"Easy cells (from {args.manifest_cells}): {cells}", flush=True)

    manager = None
    if not args.connect_only:
        manager = GodotProcessManager(
            godot_bin=args.godot_bin,
            project_dir=args.godot_project,
            scene_path=args.godot_scene,
        )
        manager.start_many(
            [args.port], headless=args.headless, debug=args.debug_godot,
            user_args=["--recording-mode", "--disable-agent-replication"],
        )
        print(f"Started Godot on port {args.port}", flush=True)

    env = None
    try:
        env = ScenarioGymEnv(host=args.host, port=args.port, seed=0)
        agent_id = env.agent_id
        if env.obs_dim != 27 or env.action_size != 7:
            raise RuntimeError(
                f"unexpected dims obs={env.obs_dim} action={env.action_size} (want 27/7)"
            )

        if args.replay_only:
            results = replay_validate(
                env, agent_id, args.replay_only, args.curriculum_level,
                args.max_steps, args.replay_sample,
            )
            summary = summarize_replay(results, args.replay_reward_tol)
            print(json.dumps(summary, indent=2), flush=True)
            return

        if not args.train_plans or not args.val_plans:
            raise RuntimeError("--train-plans and --val-plans are required (run the planner first)")
        train_planned, train_plan_fail = load_plans(args.train_plans)
        val_planned, val_plan_fail = load_plans(args.val_plans)
        print(f"Planned poses: train={sum(len(v) for v in train_planned.values())} "
              f"val={sum(len(v) for v in val_planned.values())} "
              f"(train plan-fails={len(train_plan_fail)}, val plan-fails={len(val_plan_fail)})",
              flush=True)

        train = generate_set(
            env, agent_id, train_planned, args.train_per_cell,
            args.curriculum_level, args.max_steps, "train")
        val = generate_set(
            env, agent_id, val_planned, args.val_per_cell,
            args.curriculum_level, args.max_steps, "val")

        train_path, train_tx, train_ep = save_set(
            out_dir / "train.npz", train["accepted_eps"], env)
        val_path, val_tx, val_ep = save_set(
            out_dir / "val.npz", val["accepted_eps"], env)

        # Disjointness of trajectories across sets: pose_id sets must not intersect.
        train_ids = {e["pose_id"] for e in train["accepted_eps"]}
        val_ids = {e["pose_id"] for e in val["accepted_eps"]}
        overlap = sorted(train_ids & val_ids)

        # Reload + replay-validate a sample through the M3 task before declaring valid.
        replay = replay_validate(
            env, agent_id, str(train_path), args.curriculum_level,
            args.max_steps, args.replay_sample,
        )
        replay_summary = summarize_replay(replay, args.replay_reward_tol)

        train_actions = np.concatenate(
            [t["action"][None, :] for e in train["accepted_eps"]
             for t in e["transitions"]], axis=0
        ) if train_tx else np.zeros((0, env.action_size), dtype=np.float32)

        covered = {int(c) for c in cells if train["accepted_counts"].get(c, 0) >= 1}
        manifest = {
            "schema": "openarm-reach-hold-ik-demos/2",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scene": args.godot_scene,
            "godot_version": "4.7.1 (cross-checked on 4.6.2)",
            "region": REGION,
            "easy_cells": cells,
            "easy_cells_source": args.manifest_cells,
            "expert": "RRT-Connect joint-space planner (openarm_ik_demo_planner.gd) + "
                      "closed-loop joint_velocity plan-follower",
            "obs_dim": int(env.obs_dim),
            "action_size": int(env.action_size),
            "action_type": env.action_type,
            "action_names": list(env.action_names),
            "action_low": list(map(float, env.action_low))
            if env.action_type == "continuous" else None,
            "action_high": list(map(float, env.action_high))
            if env.action_type == "continuous" else None,
            "stage_a_gate": train["stage_a"] or val["stage_a"],
            "reward_contract": REWARD_CONTRACT,
            "acceptance_rule": (
                "terminated in hold-success (target_reached) AND not truncated AND no collision "
                "AND all actions finite in [-1,1]"
            ),
            "planning_failures": {"train": train_plan_fail, "val": val_plan_fail},
            "execution_failures": {
                "train": train["execution_failures"], "val": val["execution_failures"]},
            "train": {
                "path": str(train_path), "plans": args.train_plans,
                "episodes": train_ep, "transitions": train_tx,
                "target_per_cell": args.train_per_cell,
                "planned_per_cell": train["planned_per_cell"],
                "accepted_per_cell": train["accepted_counts"],
                "attempts_per_cell": train["attempts"],
                "shortfall_cells": train["shortfall"],
                "per_cell_stats": per_cell_stats(train["summaries"], cells),
            },
            "val": {
                "path": str(val_path), "plans": args.val_plans,
                "episodes": val_ep, "transitions": val_tx,
                "target_per_cell": args.val_per_cell,
                "planned_per_cell": val["planned_per_cell"],
                "accepted_per_cell": val["accepted_counts"],
                "attempts_per_cell": val["attempts"],
                "shortfall_cells": val["shortfall"],
                "per_cell_stats": per_cell_stats(val["summaries"], cells),
            },
            "gates": {
                "quota_complete": (
                    all(train["accepted_counts"].get(c, 0) >= args.train_per_cell
                        for c in cells)
                    and all(val["accepted_counts"].get(c, 0) >= args.val_per_cell
                            for c in cells)),
                "quota_target": {"train": args.train_per_cell, "val": args.val_per_cell},
                "all_easy_cells_covered": len(covered) == len(cells),
                "uncovered_cells": sorted(set(int(c) for c in cells) - covered),
                "no_collision_in_accepted": all(
                    not e["collided"] for e in train["accepted_eps"] + val["accepted_eps"]),
                "all_accepted_target_reached": all(
                    e["reached"] for e in train["accepted_eps"] + val["accepted_eps"]),
                "actions_finite_in_bounds": _finite_actions_ok(
                    train_actions,
                    env.action_low if env.action_type == "continuous" else None,
                    env.action_high if env.action_type == "continuous" else None),
                "obs_dim_27": int(env.obs_dim) == 27,
                "action_dim_7": int(env.action_size) == 7,
                "train_val_pose_ids_disjoint": len(overlap) == 0,
                "pose_id_overlap": overlap,
                "replay_all_reached_target": replay_summary["all_reached_target"],
                "replay_terminal_reward_coherent": replay_summary[
                    "terminal_reward_coherent"],
            },
            "replay_validation": replay_summary,
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote {train_path} ({train_tx} tx / {train_ep} ep), "
              f"{val_path} ({val_tx} tx / {val_ep} ep)", flush=True)
        print(f"Wrote {manifest_path}", flush=True)
        print("PLANNING failures: train=%d val=%d" % (
            len(train_plan_fail), len(val_plan_fail)), flush=True)
        print("EXECUTION failures: train=%s val=%s" % (
            train["execution_failures"], val["execution_failures"]), flush=True)
        print("GATES: " + json.dumps(manifest["gates"], indent=2), flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if manager is not None:
            manager.stop_all()


if __name__ == "__main__":
    main()
