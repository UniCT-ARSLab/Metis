"""M6.2 Phase C1 (collection): record the BOUNDED SETTLING corrective dataset.

Re-runs the C0 mechanism at a fixed delta_max on the C0 deck and records ONLY the transitions really
applied in the near-target settling phase (position error <= near_dist): (obs, applied_correction,
cell, pos_err_m, d_obs=||obs[14:17]||). NO aggressive approach/recovery segments are recorded (the
expert is off during approach). Also samples M5/clone ANCHOR obs (residual target = 0). The residual
imitation trainer (C1) fits gate(d)*delta_max*tanh(residual) to these corrections.

Actions are REALLY executed (no teleport). Clone + M5 v2 read-only.
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
from tools.c0_feasibility_oracle import load_goal_cfgs  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN, SCENE  # noqa: E402

REGION = "easy"
CLONE = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"
M5V2 = "python/demos/openarm_reach_hold_m5_v2/train.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=6510)
    ap.add_argument("--plans", default="python/demos/openarm_reach_hold_c0/plan_c0.json")
    ap.add_argument("--clone-weights", default=CLONE)
    ap.add_argument("--m5v2", default=M5V2)
    ap.add_argument("--delta-max", type=float, default=0.02)
    ap.add_argument("--near-dist", type=float, default=0.10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--physics-frames-per-step", type=int, default=3)
    ap.add_argument("--max-speed", type=float, default=0.5)
    ap.add_argument("--n-anchors", type=int, default=6000, help="M5 anchor obs (residual target 0)")
    ap.add_argument("--out", default="python/demos/openarm_reach_hold_c1/settling_corrective.npz")
    args = ap.parse_args()

    goal_cfgs = load_goal_cfgs(args.plans)
    dt = args.physics_frames_per_step / 60.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    mgr = GodotProcessManager(godot_bin=args.godot_bin, project_dir="godot", scene_path=SCENE)
    mgr.start_many([args.port], headless=True)
    env = ScenarioGymEnv(host="127.0.0.1", port=args.port, seed=0, timeout=60.0)
    actor = build_continuous_actor(obs_dim=env.obs_dim, action_size=env.action_size)
    actor.load_weights(args.clone_weights)

    def greedy(obs):
        return np.clip(actor(tf.convert_to_tensor(obs[None, :], tf.float32), training=False).numpy()[0],
                       -1.0, 1.0)

    rec_obs, rec_corr, rec_cell, rec_derr, rec_dobs = [], [], [], [], []
    reached_n = coll_n = 0
    try:
        for (cell, seed), goal in sorted(goal_cfgs.items()):
            env.configure(demo_mode=True, demo_forced_region=REGION, demo_forced_cell=int(cell),
                          curriculum_level=0.0, evaluation_mode=False, recording_mode=False,
                          manage_agent_cameras=False)
            obs, info = env.reset(seed=int(seed))
            if int(agent_info_of(info).get("reset", {}).get("target_cell", -1)) != int(cell):
                raise RuntimeError(f"forced cell {cell} not honoured")
            follower = PlanFollower([goal], args.max_speed, dt)
            activated = reached = collided = False
            ep_obs, ep_corr, ep_derr, ep_dobs = [], [], [], []
            for _step in range(args.max_steps):
                joints = denorm_joints(obs)
                clone_a = greedy(obs)
                cur = float(agent_info_of(info).get("position_error_m", np.nan))
                if np.isfinite(cur) and cur <= args.near_dist:
                    activated = True
                if activated:
                    exp = follower.action(joints)
                    corr = np.clip(exp - clone_a, -args.delta_max, args.delta_max)
                    action = np.clip(clone_a + corr, -1.0, 1.0)
                    ep_obs.append(obs.astype(np.float32)); ep_corr.append(corr.astype(np.float32))
                    ep_derr.append(cur if np.isfinite(cur) else np.nan)
                    ep_dobs.append(float(np.linalg.norm(obs[14:17])))
                else:
                    action = clone_a
                obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
                ai = agent_info_of(info)
                collided = collided or bool(ai.get("collided", False))
                if str(ai.get("terminal_reason", "")) == "target_reached":
                    reached = True
                if terminated or truncated:
                    break
            reached_n += int(reached); coll_n += int(collided)
            # keep the settling transitions ONLY from clean hold-success episodes
            if reached and not collided:
                rec_obs.extend(ep_obs); rec_corr.extend(ep_corr)
                rec_cell.extend([int(cell)] * len(ep_obs)); rec_derr.extend(ep_derr); rec_dobs.extend(ep_dobs)
    finally:
        env.close()
        mgr.stop_all()

    obs_arr = np.asarray(rec_obs, np.float32) if rec_obs else np.zeros((0, 27), np.float32)
    corr_arr = np.asarray(rec_corr, np.float32) if rec_corr else np.zeros((0, 7), np.float32)
    cell_arr = np.asarray(rec_cell, np.int32)
    derr_arr = np.asarray(rec_derr, np.float32)
    dobs_arr = np.asarray(rec_dobs, np.float32)
    # M5 anchor obs (residual target zero) -- from the clone's normal distribution
    m5 = np.load(args.m5v2, allow_pickle=True)
    m5o = np.asarray(m5["obs"], np.float32)
    rng = np.random.default_rng(0)
    anc = m5o[rng.choice(len(m5o), size=min(args.n_anchors, len(m5o)), replace=False)]

    np.savez_compressed(args.out, obs=obs_arr, correction=corr_arr, cells=cell_arr,
                        pos_err_m=derr_arr, d_obs=dobs_arr, anchor_obs=anc,
                        delta_max=np.asarray([args.delta_max], np.float32),
                        near_dist=np.asarray([args.near_dist], np.float32))
    # calibrate the obs-space gate boundary: d_obs vs pos_err_m (linear ~ /workspace_scale)
    if len(derr_arr):
        finite = np.isfinite(derr_arr) & (derr_arr > 1e-6)
        ratio = float(np.median(dobs_arr[finite] / derr_arr[finite])) if finite.any() else float("nan")
    else:
        ratio = float("nan")
    manifest = {"delta_max": args.delta_max, "near_dist": args.near_dist,
                "n_settling_transitions": int(len(obs_arr)), "n_anchors": int(len(anc)),
                "reached_episodes": reached_n, "collided_episodes": coll_n,
                "d_obs_per_meter": ratio,
                "gate_outer_d_obs": float(args.near_dist * ratio) if ratio == ratio else None,
                "per_cell": {int(c): int((cell_arr == c).sum()) for c in np.unique(cell_arr)} if len(cell_arr) else {}}
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print("C1_COLLECT_DONE " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
