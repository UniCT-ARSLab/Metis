"""M5.5 SafeDAgger Round-1 collection on the stable reach-hold task (Stage A, unchanged).

The BC actor (novice, deterministic) rolls out. At every learner-visited state the EXPERT action is
computed from the precomputed collision-free joint plan STARTING AT THE CURRENT config (the M5
plan-follower: a velocity collinear with (current_waypoint - current_joints)). A safety monitor
takes over BEFORE a collision when the learner deviates from the plan or stops making progress;
while the expert drives, its action is REALLY applied via env.step() and the complete transition is
recorded; control returns to the learner once it rejoins the plan.

Two STRICTLY SEPARATED datasets (never fabricate transitions):
  - dagger_labels_r1.npz   : (obs visited by the learner, expert_action).  BC imitation ONLY.
  - dagger_recovery_r1.npz : complete transitions where the expert action was REALLY applied
                             (obs, action, reward, next_obs, done).        Valid for replay.
An expert_action is NEVER paired with a next_obs produced by the learner action.

Task / reward / curriculum are untouched; the expert never plans from an already-colliding state
(a ring buffer of the last valid states diagnoses late interventions).
"""
import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.sac import scale_action_numpy  # noqa: E402
from core.models import build_sac_actor  # noqa: E402
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402

import tensorflow as tf  # noqa: E402

SCENE = "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"
REGION = "easy"
# openarm right-arm joint limits (from openarm_bimanual.urdf) used to de-normalise obs[0:7]
# (get_joint_position_observation = remap(pos, lower, upper, -1, 1)).
JOINT_LO = np.array([-1.396263, -0.174533, -1.570796, 0.0, -1.570796, -0.785398, -1.570796])
JOINT_HI = np.array([3.490659, 3.316125, 1.570796, 2.443461, 1.570796, 0.785398, 1.570796])


def denorm_joints(obs):
    n = np.clip(np.asarray(obs[:7], dtype=np.float64), -1.0, 1.0)
    return JOINT_LO + (n + 1.0) * 0.5 * (JOINT_HI - JOINT_LO)


def load_plans_by_cell(path):
    data = json.loads(Path(path).read_text())
    by_cell = {}
    for pose_id, plan in data.get("plans", {}).items():
        by_cell.setdefault(int(plan["cell"]), []).append(
            {"seed": int(plan["seed"]), "pose_id": pose_id,
             "waypoints": [np.asarray(w, dtype=np.float64) for w in plan["waypoints"]]})
    for c in by_cell:
        by_cell[c].sort(key=lambda p: p["seed"])
    return by_cell


class PlanFollower:
    """Python mirror of the agent's demo-plan follower: collinear joint_velocity toward the current
    waypoint, advancing when close. State = wp index; deviation = distance to the tracked waypoint."""

    def __init__(self, waypoints, max_speed, dt_step):
        self.wp = waypoints
        self.i = 0
        self.k = max_speed * dt_step

    def action(self, joints):
        goal = self.wp[self.i]
        gap = goal - joints
        raw = gap / self.k
        m = np.max(np.abs(raw))
        scale = (1.0 / m) if m > 1.0 else 1.0
        act = np.clip(raw * scale, -1.0, 1.0)
        if np.max(np.abs(gap)) < 0.03 and self.i < len(self.wp) - 1:
            self.i += 1
        return act.astype(np.float32)

    def deviation(self, joints):
        # Joint-space distance to the planned POLYLINE (home->goal), not to the current waypoint:
        # the learner may sit between two sparse waypoints yet still be on the collision-free path.
        # Small while the learner tracks the plan; grows as it drifts off the safe corridor.
        if len(self.wp) == 1:
            return float(np.linalg.norm(joints - self.wp[0]))
        best = np.inf
        for a, b in zip(self.wp[:-1], self.wp[1:]):
            ab = b - a
            denom = float(np.dot(ab, ab))
            t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(joints - a, ab) / denom, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(joints - (a + t * ab))))
        return best


def bc_greedy(actor, obs, low, high):
    mean, _ = actor(tf.convert_to_tensor(obs[None, :], dtype=tf.float32), training=False)
    return scale_action_numpy(np.tanh(mean.numpy()[0]), low, high).astype(np.float32)


def agent_info_of(info):
    return info.get("agent_info", {}) if isinstance(info, dict) else {}


def rollout_pose(env, agent_id, actor, low, high, cell, pose, curriculum_level, max_steps,
                 dt_step, max_speed, tau_takeover, tau_rejoin, noprogress_patience):
    """One SafeDAgger episode. Returns (labels[], recovery[], stats)."""
    env.configure(demo_mode=True, demo_forced_region=REGION, demo_forced_cell=int(cell),
                  curriculum_level=float(curriculum_level), evaluation_mode=False,
                  recording_mode=True, recording_agent_id=agent_id, manage_agent_cameras=True)
    obs, info = env.reset(seed=int(pose["seed"]))
    reset_info = agent_info_of(info).get("reset", {})
    if int(reset_info.get("target_cell", -1)) != int(cell):
        raise RuntimeError(f"forced cell {cell} not honoured ({reset_info.get('target_cell')})")

    follower = PlanFollower(pose["waypoints"], max_speed, dt_step)
    labels, recovery = [], []
    valid_ring = deque(maxlen=16)
    valid_ring.append(denorm_joints(obs))  # home is a valid state (never intervene from nothing)
    mode = "learner"
    interventions = 0
    late_intervention = False
    best_dist = None
    stagnate = 0
    collided = False
    reached = False

    for _step in range(max_steps):
        joints = denorm_joints(obs)
        a_nov = bc_greedy(actor, obs, low, high)
        a_exp = follower.action(joints)
        dev = follower.deviation(joints)

        cur_dist = float(agent_info_of(info).get("position_error_m", np.nan))
        if best_dist is None or (np.isfinite(cur_dist) and cur_dist < best_dist - 1e-3):
            best_dist = cur_dist if np.isfinite(cur_dist) else best_dist
            stagnate = 0
        else:
            stagnate += 1
        no_progress = stagnate >= noprogress_patience

        # Safety switching: take over BEFORE collision on deviation or stall; hand back once rejoined.
        if mode == "learner" and (dev > tau_takeover or no_progress):
            mode = "expert"
            interventions += 1
            if not valid_ring:
                late_intervention = True
        elif mode == "expert" and dev < tau_rejoin:
            mode = "learner"

        if mode == "learner":
            labels.append((obs.astype(np.float32), a_exp))          # imitation-only label
            applied = a_nov
        else:
            applied = a_exp

        next_obs, reward, terminated, truncated, info = env.step(applied)
        ai = agent_info_of(info)
        if bool(ai.get("collided", False)):
            collided = True
        if str(ai.get("terminal_reason", "")) == "target_reached":
            reached = True

        if mode == "expert":
            # complete transition, expert action REALLY applied -> valid for replay
            recovery.append((obs.astype(np.float32), a_exp, float(reward),
                             next_obs.astype(np.float32), bool(terminated or truncated)))

        if not collided:
            valid_ring.append(joints)

        obs = next_obs
        if terminated or truncated:
            break

    stats = {"cell": int(cell), "pose_id": pose["pose_id"], "seed": int(pose["seed"]),
             "n_labels": len(labels), "n_recovery": len(recovery),
             "interventions": interventions, "late_intervention": late_intervention,
             "collided": collided, "reached": reached}
    return labels, recovery, stats


def save_labels(path, rows, group_offset=0):
    if not rows:
        obs = np.zeros((0, 27), np.float32); act = np.zeros((0, 7), np.float32)
        grp = np.zeros((0,), np.int64)
    else:
        obs = np.stack([r[0] for r in rows]); act = np.stack([r[1] for r in rows])
        grp = np.asarray([r[2] + group_offset for r in rows], dtype=np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=obs, actions=act, group_ids=grp,
                        action_type=np.asarray(["continuous"]))
    return len(act)


def save_recovery(path, rows):
    if not rows:
        z = lambda d: np.zeros((0,) + d, np.float32)
        np.savez_compressed(path, obs=z((27,)), actions=z((7,)), rewards=z(()),
                            next_obs=z((27,)), dones=z(()), group_ids=np.zeros((0,), np.int64),
                            action_type=np.asarray(["continuous"]))
        return 0
    obs = np.stack([r[0] for r in rows]); act = np.stack([r[1] for r in rows])
    rew = np.asarray([r[2] for r in rows], np.float32)
    nobs = np.stack([r[3] for r in rows]); done = np.asarray([r[4] for r in rows], np.float32)
    grp = np.asarray([r[5] for r in rows], np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=obs, actions=act, rewards=rew, next_obs=nobs, dones=done,
                        group_ids=grp, action_type=np.asarray(["continuous"]))
    return len(act)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    p.add_argument("--godot-project", default="godot")
    p.add_argument("--godot-scene", default=SCENE)
    p.add_argument("--plans", required=True, help="plan_dagger_r1.json (DAgger target plans)")
    p.add_argument("--bc-actor", required=True, help="actor_bc.weights.h5 (novice, deterministic)")
    p.add_argument("--out-dir", default="python/demos/openarm_reach_hold_dagger")
    p.add_argument("--port", type=int, default=6460)
    p.add_argument("--curriculum-level", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--physics-frames-per-step", type=int, default=3)
    p.add_argument("--max-speed", type=float, default=0.5)
    p.add_argument("--tau-takeover", type=float, default=0.35,
                   help="joint-space deviation (rad) from the plan that triggers expert takeover")
    p.add_argument("--tau-rejoin", type=float, default=0.12,
                   help="deviation under which control returns to the learner")
    p.add_argument("--noprogress-patience", type=int, default=25)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    by_cell = load_plans_by_cell(args.plans)
    cells = sorted(by_cell.keys())
    dt_step = args.physics_frames_per_step / 60.0
    print(f"DAgger targets: {sum(len(v) for v in by_cell.values())} over cells {cells}", flush=True)

    manager = None
    if args.godot_bin:
        manager = GodotProcessManager(godot_bin=args.godot_bin, project_dir=args.godot_project,
                                      scene_path=args.godot_scene)
        manager.start_many([args.port], headless=args.headless, debug=False,
                            user_args=["--recording-mode", "--disable-agent-replication"])
    env = None
    try:
        env = ScenarioGymEnv(host="127.0.0.1", port=args.port, seed=0)
        agent_id = env.agent_id
        low = np.asarray(env.action_low, np.float32)
        high = np.asarray(env.action_high, np.float32)
        actor = build_sac_actor(obs_dim=env.obs_dim, action_size=env.action_size)
        actor(tf.zeros((1, env.obs_dim)))
        actor.load_weights(args.bc_actor)
        print(f"Loaded BC novice actor: {args.bc_actor}", flush=True)

        all_labels, all_recovery, stats = [], [], []
        gid = 0
        for cell in cells:
            for pose in by_cell[cell]:
                labels, recovery, st = rollout_pose(
                    env, agent_id, actor, low, high, cell, pose, args.curriculum_level,
                    args.max_steps, dt_step, args.max_speed, args.tau_takeover, args.tau_rejoin,
                    args.noprogress_patience)
                # tag each label with the pose group id (for group-balanced BC), recovery too
                all_labels.extend((o, a, gid) for (o, a) in labels)
                all_recovery.extend((o, a, r, no, d, gid) for (o, a, r, no, d) in recovery)
                stats.append(st)
                gid += 1
                print(f"[dagger] cell={cell} {pose['pose_id']} labels={st['n_labels']} "
                      f"recovery={st['n_recovery']} interv={st['interventions']} "
                      f"reached={st['reached']} coll={st['collided']}"
                      + (" LATE" if st["late_intervention"] else ""), flush=True)

        n_lab = save_labels(out / "dagger_labels_r1.npz", all_labels)
        n_rec = save_recovery(out / "dagger_recovery_r1.npz", all_recovery)
        manifest = {
            "schema": "openarm-reach-hold-dagger/1",
            "round": 1,
            "plans": args.plans,
            "bc_actor": args.bc_actor,
            "n_targets": len(stats),
            "n_labels": n_lab,
            "n_recovery": n_rec,
            "labels_path": str(out / "dagger_labels_r1.npz"),
            "recovery_path": str(out / "dagger_recovery_r1.npz"),
            "tau_takeover": args.tau_takeover,
            "tau_rejoin": args.tau_rejoin,
            "late_interventions": int(sum(s["late_intervention"] for s in stats)),
            "reached_rollouts": int(sum(s["reached"] for s in stats)),
            "collided_rollouts": int(sum(s["collided"] for s in stats)),
            "per_target": stats,
        }
        (out / "dagger_r1_manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"\nWrote labels={n_lab} recovery={n_rec} manifest -> {out}", flush=True)
        print(f"late_interventions={manifest['late_interventions']} "
              f"reached={manifest['reached_rollouts']}/{len(stats)} "
              f"collided={manifest['collided_rollouts']}/{len(stats)}", flush=True)
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
