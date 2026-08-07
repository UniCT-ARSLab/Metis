"""M6.2 Phase C1: rollout eval of the gated bounded-settling residual policy vs the frozen clone.

Runs the SAME forced-cell deck for both policies (paired per cell+seed). Records success, collision,
dist_min/final, hold_max per pose + cell. For the residual it also logs the max |action - clone|
observed in the APPROACH zone (gate==0) -- must be 0 (behavior identical to the clone outside the
near-target zone) -- and the global max |action - clone| (must be <= delta_max). Clone read-only.
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
from core.settling_residual import (  # noqa: E402
    GATE_INNER, GATE_OUTER, build_settling_residual, gate, settling_correction)
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN, DEFAULT_MANIFEST, SCENE, build_deck, load_easy_cells  # noqa: E402

REGION = "easy"
CLONE = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=6520)
    ap.add_argument("--clone-weights", default=CLONE)
    ap.add_argument("--residual-weights", required=True)
    ap.add_argument("--delta-max", type=float, default=0.02)
    ap.add_argument("--gate-outer", type=float, default=GATE_OUTER)
    ap.add_argument("--gate-inner", type=float, default=GATE_INNER)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--curriculum-level", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells = load_easy_cells(DEFAULT_MANIFEST)
    deck = build_deck(cells, [args.seed_base + i for i in range(args.per_cell)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    clone = build_continuous_actor(obs_dim=27, action_size=7); clone.load_weights(args.clone_weights)
    residual = build_settling_residual(27, 7); residual.load_weights(args.residual_weights)

    def policy(obs):
        o = obs[None, :].astype(np.float32)
        clone_a = np.clip(clone(tf.convert_to_tensor(o), training=False).numpy()[0], -1.0, 1.0)
        corr = settling_correction(residual, o, delta_max=args.delta_max,
                                   outer=args.gate_outer, inner=args.gate_inner).numpy()[0]
        action = np.clip(clone_a + corr, -1.0, 1.0)
        g = float(gate(np.linalg.norm(obs[14:17]), outer=args.gate_outer, inner=args.gate_inner))
        dev = float(np.max(np.abs(action - clone_a)))
        return action.astype(np.float32), dev, g

    mgr = GodotProcessManager(godot_bin=args.godot_bin, project_dir="godot", scene_path=SCENE)
    mgr.start_many([args.port], headless=True)
    env = ScenarioGymEnv(host="127.0.0.1", port=args.port, seed=0, timeout=60.0)

    poses = []
    try:
        for dpose in deck:
            cell, seed = dpose["cell"], dpose["seed"]
            env.configure(demo_mode=True, demo_forced_region=REGION, demo_forced_cell=int(cell),
                          curriculum_level=args.curriculum_level, evaluation_mode=False, recording_mode=False,
                          manage_agent_cameras=False)
            obs, info = env.reset(seed=int(seed))
            if int(info.get("agent_info", {}).get("reset", {}).get("target_cell", -1)) != int(cell):
                raise RuntimeError(f"forced cell {cell} not honoured")
            diag = new_episode_agent_diagnostics()
            reached = terminated = truncated = False
            max_dev = max_dev_approach = 0.0
            for _ in range(args.max_steps):
                action, dev, g = policy(obs)
                max_dev = max(max_dev, dev)
                if g == 0.0:
                    max_dev_approach = max(max_dev_approach, dev)
                obs, reward, terminated, truncated, info = env.step(action)
                ai = info.get("agent_info", {})
                update_episode_agent_diagnostics(diag, ai)
                if agent_succeeded(ai):
                    reached = True
                if terminated or truncated:
                    break
            fin = finalize_episode_agent_diagnostics(diag)
            poses.append({"cell": int(cell), "seed": int(seed), "success": bool(reached),
                          "collided": bool(fin["collided"]), "dist_min": fin["position_error_min"],
                          "dist_final": fin["position_error_final"], "hold_max": fin["hold_frames_max"],
                          "orient_final": fin["orientation_error_final"], "orient_min": fin["orientation_error_min"],
                          "terminal_reasons": list(fin["terminal_reasons"]),
                          "max_dev": max_dev, "max_dev_approach": max_dev_approach})
    finally:
        env.close()
        mgr.stop_all()

    n = len(poses)
    per_cell = {}
    for c in cells:
        ps = [p for p in poses if p["cell"] == c]
        s = sum(p["success"] for p in ps)
        per_cell[int(c)] = {"n": len(ps), "successes": s, "success_rate": s / len(ps) if ps else float("nan"),
                            "collisions": sum(p["collided"] for p in ps),
                            "dist_min_median": float(np.median([p["dist_min"] for p in ps if p["dist_min"] is not None])) if ps else None,
                            "hold_max_median": float(np.median([p["hold_max"] or 0 for p in ps])) if ps else None}
    report = {"policy": "clone+settling_residual", "residual": args.residual_weights,
              "delta_max": args.delta_max, "n": n,
              "success_rate": sum(p["success"] for p in poses) / n if n else float("nan"),
              "collision_rate": sum(p["collided"] for p in poses) / n if n else float("nan"),
              "max_dev_global": max((p["max_dev"] for p in poses), default=0.0),
              "max_dev_approach": max((p["max_dev_approach"] for p in poses), default=0.0),
              "per_cell": per_cell, "poses": poses}
    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print("EVAL_RESIDUAL " + json.dumps({k: report[k] for k in
          ("success_rate", "collision_rate", "max_dev_global", "max_dev_approach")}, default=float), flush=True)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
