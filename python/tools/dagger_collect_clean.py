"""M6.2 Phase B: CLEAN SafeDAgger corrective collection (Round 1) for the td3_bc continuous clone.

Rolls the FROZEN clone (actor_bc_final) on a NEW disjoint DAgger deck. Learner drives; when it
deviates from the collision-free plan (dev>tau) or stalls, the EXPERT (PlanFollower toward the plan,
computed from the CURRENT joint state) takes over with REAL joint_velocity actions and its next_obs
is recorded. ONLY recoveries from episodes that end in a REAL hold-success (target_reached, no
collision) are kept -- contaminated/incomplete recoveries are discarded. Each kept transition is
tagged with its cell so the BC fine-tune can weight the hard cells (25/41/31/28).

Clone + M5 v2 are read-only; corrective dataset is written to its own dir. Reuses the proven
PlanFollower / denorm_joints from dagger_round.py (the SAC-specific bc_greedy is NOT reused; the clone
is a continuous tanh actor).
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.models import build_continuous_actor  # noqa: E402
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402
from tools.dagger_round import PlanFollower, agent_info_of, denorm_joints, load_plans_by_cell  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN, SCENE  # noqa: E402

REGION = "easy"
CLONE = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"


def rollout(env, greedy, cell, pose, *, max_steps, dt_step, max_speed, tau_takeover, tau_rejoin,
            noprogress_patience):
    env.configure(demo_mode=True, demo_forced_region=REGION, demo_forced_cell=int(cell),
                  curriculum_level=0.0, evaluation_mode=False, recording_mode=False,
                  manage_agent_cameras=False)
    obs, info = env.reset(seed=int(pose["seed"]))
    if int(agent_info_of(info).get("reset", {}).get("target_cell", -1)) != int(cell):
        raise RuntimeError(f"forced cell {cell} not honoured")
    follower = PlanFollower(pose["waypoints"], max_speed, dt_step)
    recovery = []
    mode = "learner"
    best_dist = None
    stagnate = 0
    collided = reached = False
    for _step in range(max_steps):
        joints = denorm_joints(obs)
        a_nov = greedy(obs)
        a_exp = follower.action(joints)
        dev = follower.deviation(joints)
        cur = float(agent_info_of(info).get("position_error_m", np.nan))
        if best_dist is None or (np.isfinite(cur) and cur < best_dist - 1e-3):
            best_dist = cur if np.isfinite(cur) else best_dist
            stagnate = 0
        else:
            stagnate += 1
        no_progress = stagnate >= noprogress_patience
        if mode == "learner" and (dev > tau_takeover or no_progress):
            mode = "expert"
        elif mode == "expert" and dev < tau_rejoin:
            mode = "learner"
        applied = a_nov if mode == "learner" else a_exp
        next_obs, reward, terminated, truncated, info = env.step(applied)
        ai = agent_info_of(info)
        collided = collided or bool(ai.get("collided", False))
        if str(ai.get("terminal_reason", "")) == "target_reached":
            reached = True
        if mode == "expert":
            recovery.append((obs.astype(np.float32), a_exp.astype(np.float32), float(reward),
                             next_obs.astype(np.float32), bool(terminated or truncated)))
        obs = next_obs
        if terminated or truncated:
            break
    return recovery, {"cell": int(cell), "pose_id": pose["pose_id"], "seed": int(pose["seed"]),
                      "n_recovery": len(recovery), "reached": reached, "collided": collided}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=6470)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--clone-weights", default=CLONE)
    ap.add_argument("--out-dir", default="python/demos/openarm_reach_hold_dagger_r1")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--physics-frames-per-step", type=int, default=3)
    ap.add_argument("--max-speed", type=float, default=0.5)
    ap.add_argument("--tau-takeover", type=float, default=0.35)
    ap.add_argument("--tau-rejoin", type=float, default=0.12)
    ap.add_argument("--noprogress-patience", type=int, default=25)
    args = ap.parse_args()

    by_cell = load_plans_by_cell(args.plans)
    dt_step = args.physics_frames_per_step / 60.0
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    mgr = GodotProcessManager(godot_bin=args.godot_bin, project_dir="godot", scene_path=SCENE)
    mgr.start_many([args.port], headless=True)
    env = ScenarioGymEnv(host="127.0.0.1", port=args.port, seed=0, timeout=60.0)
    actor = build_continuous_actor(obs_dim=env.obs_dim, action_size=env.action_size)
    actor.load_weights(args.clone_weights)

    def greedy(obs):
        return np.clip(actor(tf.convert_to_tensor(obs[None, :], tf.float32), training=False).numpy()[0],
                       -1.0, 1.0).astype(np.float32)

    kept, stats = [], []
    try:
        for cell in sorted(by_cell):
            for pose in by_cell[cell]:
                recovery, st = rollout(env, greedy, cell, pose, max_steps=args.max_steps,
                                       dt_step=dt_step, max_speed=args.max_speed,
                                       tau_takeover=args.tau_takeover, tau_rejoin=args.tau_rejoin,
                                       noprogress_patience=args.noprogress_patience)
                # HOLD-SUCCESS FILTER: keep corrections only from real hold-success, no-collision episodes
                keep = st["reached"] and not st["collided"]
                st["kept"] = bool(keep and recovery)
                if keep:
                    for (o, a, r, no, d) in recovery:
                        kept.append((o, a, r, no, d, int(cell)))
                stats.append(st)
                print(f"[dagger] cell={cell} {pose['pose_id']} rec={st['n_recovery']} "
                      f"reached={st['reached']} coll={st['collided']} kept={st['kept']}", flush=True)
    finally:
        env.close()
        mgr.stop_all()

    if kept:
        obs = np.stack([r[0] for r in kept]); act = np.stack([r[1] for r in kept])
        rew = np.asarray([r[2] for r in kept], np.float32); nobs = np.stack([r[3] for r in kept])
        done = np.asarray([r[4] for r in kept], np.float32); cells = np.asarray([r[5] for r in kept], np.int32)
    else:
        obs = np.zeros((0, 27), np.float32); act = np.zeros((0, 7), np.float32)
        rew = np.zeros((0,), np.float32); nobs = np.zeros((0, 27), np.float32)
        done = np.zeros((0,), np.float32); cells = np.zeros((0,), np.int32)
    corr_path = out / "corrective_r1.npz"
    np.savez_compressed(corr_path, obs=obs, actions=act, rewards=rew, next_obs=nobs, dones=done,
                        cells=cells, action_type=np.asarray(["continuous"]))
    import collections
    per_cell = collections.Counter(cells.tolist())
    manifest = {"schema": "openarm-reach-hold-safedagger-clean/1", "plans": args.plans,
                "clone": args.clone_weights, "n_poses": len(stats),
                "n_kept_episodes": int(sum(s["kept"] for s in stats)),
                "n_corrective_transitions": int(len(act)),
                "per_cell_transitions": {int(k): int(v) for k, v in sorted(per_cell.items())},
                "reached_episodes": int(sum(s["reached"] for s in stats)),
                "collided_episodes": int(sum(s["collided"] for s in stats))}
    (out / "corrective_r1_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DAGGER_COLLECT_DONE " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
