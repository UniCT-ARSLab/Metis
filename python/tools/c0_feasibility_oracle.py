"""M6.2 Phase C0: bounded-settling FEASIBILITY oracle (NO training, NO dataset build).

For each pose: run the FROZEN clone until it is near the target (position error <= ~10 cm), then
activate a BOUNDED expert correction PROJECTED around the clone action:

    correction = clip(expert_action - clone_action, -delta_max, +delta_max)
    action     = clip(clone_action + correction, -1, 1)          # hard: |action-clone| <= delta_max

The expert drives toward the RRT goal config (last waypoint) -> precise settling. Actions are REALLY
executed (no teleport, no theoretical labels); success = the scenario's own target_reached-after-hold.
Sweeps delta_max in {0.02, 0.05, 0.10} to find the MINIMUM that yields real hold-success without
collisions. Reports per bound: success / collisions / final distance / hold, per cell. Builds NOTHING.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.models import build_continuous_actor  # noqa: E402
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402
from tools.dagger_round import PlanFollower, agent_info_of, denorm_joints  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN, SCENE  # noqa: E402

REGION = "easy"
CLONE = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"


def load_goal_cfgs(path):
    data = json.loads(Path(path).read_text())
    out = {}
    for pid, pl in data.get("plans", {}).items():
        wps = pl.get("waypoints", [])
        if wps:
            out[(int(pl["cell"]), int(pl["seed"]))] = np.asarray(wps[-1], np.float64)
    return out


def rollout(env, greedy, goal_cfg, cell, seed, delta, *, near_dist, max_steps, max_speed, dt):
    env.configure(demo_mode=True, demo_forced_region=REGION, demo_forced_cell=int(cell),
                  curriculum_level=0.0, evaluation_mode=False, recording_mode=False,
                  manage_agent_cameras=False)
    obs, info = env.reset(seed=int(seed))
    if int(agent_info_of(info).get("reset", {}).get("target_cell", -1)) != int(cell):
        raise RuntimeError(f"forced cell {cell} not honoured")
    follower = PlanFollower([goal_cfg], max_speed, dt)          # settle toward the goal config
    activated = reached = collided = False
    hold = hold_max = 0
    dist_min = dist_final = 9.0
    max_dev = 0.0
    for _step in range(max_steps):
        joints = denorm_joints(obs)
        clone_a = np.clip(greedy(obs), -1.0, 1.0)
        cur = float(agent_info_of(info).get("position_error_m", np.nan))
        if np.isfinite(cur) and cur <= near_dist:
            activated = True
        if activated:
            exp = follower.action(joints)
            corr = np.clip(exp - clone_a, -delta, delta)
            action = np.clip(clone_a + corr, -1.0, 1.0)
            max_dev = max(max_dev, float(np.max(np.abs(action - clone_a))))
        else:
            action = clone_a
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        ai = agent_info_of(info)
        collided = collided or bool(ai.get("collided", False))
        if str(ai.get("terminal_reason", "")) == "target_reached":
            reached = True
        d = float(ai.get("position_error_m", np.nan))
        if np.isfinite(d):
            dist_final = d
            dist_min = min(dist_min, d)
            if d <= 0.04 and float(ai.get("max_joint_speed", 9)) < 0.3:
                hold += 1; hold_max = max(hold_max, hold)
            else:
                hold = 0
        if terminated or truncated:
            break
    assert max_dev <= delta + 1e-4, f"bounded-correction violation {max_dev} > {delta}"
    return {"cell": int(cell), "seed": int(seed), "activated": activated, "success": reached,
            "collided": collided, "dist_final": dist_final, "dist_min": dist_min,
            "hold_max": int(hold_max), "max_dev": max_dev}


def summarize(rows):
    act = [r for r in rows if r["activated"]]
    n = len(act)
    succ = sum(r["success"] for r in act)
    coll = sum(r["collided"] for r in act)
    per_cell = {}
    for c in sorted(set(r["cell"] for r in act)):
        cr = [r for r in act if r["cell"] == c]
        per_cell[int(c)] = {"n": len(cr), "success": sum(r["success"] for r in cr),
                            "collisions": sum(r["collided"] for r in cr),
                            "dist_final_median": float(np.median([r["dist_final"] for r in cr])),
                            "hold_max_median": float(np.median([r["hold_max"] for r in cr]))}
    return {"n_activated": n, "n_total": len(rows),
            "success_rate": succ / n if n else float("nan"),
            "collision_rate": coll / n if n else float("nan"),
            "dist_final_median": float(np.median([r["dist_final"] for r in act])) if n else float("nan"),
            "hold_max_median": float(np.median([r["hold_max"] for r in act])) if n else float("nan"),
            "per_cell": per_cell}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=6500)
    ap.add_argument("--plans", default="python/demos/openarm_reach_hold_c0/plan_c0.json")
    ap.add_argument("--clone-weights", default=CLONE)
    ap.add_argument("--deltas", default="0.02,0.05,0.10")
    ap.add_argument("--near-dist", type=float, default=0.10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--physics-frames-per-step", type=int, default=3)
    ap.add_argument("--max-speed", type=float, default=0.5)
    ap.add_argument("--out", default="python/demos/openarm_reach_hold_c0/c0_feasibility.json")
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]
    goal_cfgs = load_goal_cfgs(args.plans)
    dt = args.physics_frames_per_step / 60.0
    print(f"C0: {len(goal_cfgs)} poses; deltas {deltas}; near_dist {args.near_dist}", flush=True)

    mgr = GodotProcessManager(godot_bin=args.godot_bin, project_dir="godot", scene_path=SCENE)
    mgr.start_many([args.port], headless=True)
    env = ScenarioGymEnv(host="127.0.0.1", port=args.port, seed=0, timeout=60.0)
    actor = build_continuous_actor(obs_dim=env.obs_dim, action_size=env.action_size)
    actor.load_weights(args.clone_weights)

    def greedy(obs):
        return actor(tf.convert_to_tensor(obs[None, :], tf.float32), training=False).numpy()[0]

    report = {"deltas": deltas, "near_dist": args.near_dist, "per_delta": {}}
    try:
        for delta in deltas:
            rows = []
            for (cell, seed), goal in sorted(goal_cfgs.items()):
                rows.append(rollout(env, greedy, goal, cell, seed, delta, near_dist=args.near_dist,
                                    max_steps=args.max_steps, max_speed=args.max_speed, dt=dt))
            s = summarize(rows)
            s["rows"] = rows
            report["per_delta"][f"{delta}"] = s
            print(f"DELTA {delta}: activated {s['n_activated']}/{s['n_total']} "
                  f"success {s['success_rate']:.3f} collisions {s['collision_rate']:.3f} "
                  f"dist_final_med {s['dist_final_median']:.4f} hold_med {s['hold_max_median']:.0f}", flush=True)
    finally:
        env.close()
        mgr.stop_all()

    # feasibility verdict: minimum delta with significant no-collision recovery (>=0.80 success, 0 coll)
    feasible = None
    for delta in deltas:
        s = report["per_delta"][f"{delta}"]
        if s["collision_rate"] == 0.0 and s["success_rate"] >= 0.80:
            feasible = delta
            break
    report["feasible_min_delta"] = feasible
    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print("C0_FEASIBILITY " + json.dumps({"feasible_min_delta": feasible,
          "per_delta": {d: {"success": round(report["per_delta"][d]["success_rate"], 3),
                            "coll": round(report["per_delta"][d]["collision_rate"], 3)}
                        for d in report["per_delta"]}}), flush=True)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
