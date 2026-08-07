"""M6.2 Phase A: diagnostic deck + CLONE failure records (READ-ONLY eval).

Rolls the FROZEN BC clone over a NEW diagnostic deck (never-used seeds, balanced over the 11 Easy
cells) and records, for every FAILURE: min/final position error, orientation, hold, max joint speed,
terminal reason, the full action trajectory, and the cell. Also stores each failing pose's joint
config at the point of deviation (last obs) so a later recoverability check can plan from there.

NO changes to observation / reward / network / dataset.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.evaluation import (  # noqa: E402
    agent_succeeded, finalize_episode_agent_diagnostics, new_episode_agent_diagnostics,
    update_episode_agent_diagnostics)
from core.models import build_continuous_actor  # noqa: E402
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN, DEFAULT_MANIFEST, SCENE, build_deck, load_easy_cells  # noqa: E402

BC_WEIGHTS = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"
REGION = "easy"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=5640)
    ap.add_argument("--bc-weights", default=BC_WEIGHTS)
    ap.add_argument("--seed-base", type=int, default=600000, help="NEVER-used seeds for the diagnostic deck")
    ap.add_argument("--per-cell", type=int, default=16)
    ap.add_argument("--curriculum-level", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="checkpoints/openarm_reach_hold_m6_2_diag/clone_failure_diagnostic.json")
    args = ap.parse_args()

    cells = load_easy_cells(DEFAULT_MANIFEST)
    seeds = [args.seed_base + i for i in range(args.per_cell)]
    deck = build_deck(cells, seeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    bc = build_continuous_actor(obs_dim=27, action_size=7)
    bc.load_weights(args.bc_weights)
    bc.trainable = False

    def bc_action(obs):
        return bc(tf.convert_to_tensor(np.asarray(obs, np.float32).reshape(1, -1)),
                  training=False).numpy().reshape(-1)

    mgr = GodotProcessManager(godot_bin=args.godot_bin, project_dir="godot", scene_path=SCENE)
    mgr.start_many([args.port], headless=True)
    env = ScenarioGymEnv(host="127.0.0.1", port=args.port, seed=0, timeout=60.0)

    poses = []
    try:
        for d in deck:
            cell, seed = d["cell"], d["seed"]
            env.configure(demo_mode=True, demo_forced_region=REGION, demo_forced_cell=int(cell),
                          curriculum_level=args.curriculum_level, evaluation_mode=False, recording_mode=False,
                          manage_agent_cameras=False)
            obs, info = env.reset(seed=int(seed))
            got = int(info.get("agent_info", {}).get("reset", {}).get("target_cell", -1))
            if got != int(cell):
                raise RuntimeError(f"forced cell {cell} not honoured (got {got})")
            diag = new_episode_agent_diagnostics()
            actions, obs_traj = [], []
            reached = terminated = truncated = False
            steps = 0
            for step in range(args.max_steps):
                a = bc_action(obs)
                obs_traj.append(np.asarray(obs, np.float32).tolist())
                actions.append(a.tolist())
                obs, reward, terminated, truncated, info = env.step(a)
                ai = info.get("agent_info", {})
                update_episode_agent_diagnostics(diag, ai)
                if agent_succeeded(ai):
                    reached = True
                steps = step + 1
                if terminated or truncated:
                    break
            fin = finalize_episode_agent_diagnostics(diag)
            rec = {"cell": int(cell), "seed": int(seed), "success": bool(reached), "steps": int(steps),
                   "terminal_reasons": list(fin["terminal_reasons"]), "collided": bool(fin["collided"]),
                   "collision_sources": list(fin["collision_sources"]),
                   "dist_final": fin["position_error_final"], "dist_min": fin["position_error_min"],
                   "orient_final": fin["orientation_error_final"], "orient_min": fin["orientation_error_min"],
                   "vel_final": fin["max_joint_speed_final"], "hold_max": fin["hold_frames_max"],
                   "success_thresholds": fin["success_thresholds"]}
            if not reached:
                rec["action_trajectory"] = actions        # full action traj kept for FAILURES only
                rec["last_obs"] = obs_traj[-1] if obs_traj else None   # deviation-state seed for recoverability
            poses.append(rec)
    finally:
        env.close()
        mgr.stop_all()

    # aggregate
    def cell_stats():
        out = {}
        for c in cells:
            ps = [p for p in poses if p["cell"] == c]
            s = sum(p["success"] for p in ps)
            out[int(c)] = {"n": len(ps), "successes": s, "success_rate": s / len(ps) if ps else float("nan"),
                           "collisions": sum(p["collided"] for p in ps)}
        return out

    fails = [p for p in poses if not p["success"]]
    # failure attribution (position > orientation > hold), from min/max diagnostics
    def attribute(p):
        thr = p["success_thresholds"]
        dmin = p["dist_min"]; omin = p["orient_min"]; hold = p["hold_max"] or 0
        if dmin is None or dmin > thr.get("distance_m", 0.08):
            return "position"
        if omin is not None and omin > thr.get("orientation_deg", 45.0):
            return "orientation"
        if (hold or 0) < thr.get("hold_physics_frames", 1):
            return "hold_or_stillness"
        return "other"
    attr = {}
    for p in fails:
        a = attribute(p); attr[a] = attr.get(a, 0) + 1
    n = len(poses); succ = sum(p["success"] for p in poses)
    report = {"deck_poses": n, "success_rate": succ / n if n else float("nan"),
              "collision_rate": sum(p["collided"] for p in poses) / n if n else float("nan"),
              "per_cell": cell_stats(), "n_failures": len(fails), "failure_attribution": attr,
              "poses": poses}
    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print("=== CLONE FAILURE DIAGNOSTIC ===", flush=True)
    print(f"deck {n} poses ({len(cells)} cells x {args.per_cell} seeds); success_rate {report['success_rate']:.3f} "
          f"collision_rate {report['collision_rate']:.3f}", flush=True)
    print("per-cell success:", {c: round(v["success_rate"], 3) for c, v in report["per_cell"].items()}, flush=True)
    print(f"failures {len(fails)}; attribution {attr}", flush=True)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
