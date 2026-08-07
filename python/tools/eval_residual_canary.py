"""Paired headless evaluator for the M6.1 canary. Runs a FIXED forced-cell deck over the 11 Easy
cells (Stage A), deterministic by seed, driving Godot with an arbitrary action_fn(obs) -> (action,
max_deviation). The SAME deck evaluates the frozen BC clone and the residual policy, so results are
paired per (cell, seed). Reuses the demo forced-cell mechanism (demo_mode=True + demo_forced_cell,
NO demo_plan -> the policy drives, the cell is pinned). Read-only w.r.t. weights.

Selection deck and final deck must use DISJOINT seed ranges (pose_id = "easy:cell:seed" never
collides), mirroring generate_ik_demos train/val disjointness.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.evaluation import (  # noqa: E402
    agent_succeeded, finalize_episode_agent_diagnostics, new_episode_agent_diagnostics,
    update_episode_agent_diagnostics)
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402

SCENE = "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"
REGION = "easy"
DEFAULT_MANIFEST = "godot/scenarios/robotarms/openarm_reach_hold_region_manifest.json"
DEFAULT_GODOT_BIN = "/home/fedyfausto/Godot/Godot_v4.7.1-stable_linux.x86_64"


def load_easy_cells(manifest=DEFAULT_MANIFEST):
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    cells = data["active_shells_path_confirmed"]["easy"]
    return [int(c) for c in cells]


def build_deck(cells, seeds):
    """Balanced deck: every (cell, seed) pair. seeds is a list of ints (one bucket per cell)."""
    return [{"cell": int(c), "seed": int(s)} for c in cells for s in seeds]


def _summarize(poses):
    by_cell = {}
    for p in poses:
        by_cell.setdefault(p["cell"], []).append(p)
    per_cell = {}
    for c, ps in sorted(by_cell.items()):
        n = len(ps)
        succ = sum(int(p["success"]) for p in ps)
        coll = sum(int(p["collided"]) for p in ps)
        prog = [p["progress_final"] for p in ps if p["progress_final"] is not None]
        per_cell[int(c)] = {"n": n, "successes": succ, "success_rate": succ / n if n else float("nan"),
                            "collisions": coll, "collision_rate": coll / n if n else float("nan"),
                            "progress_mean": float(np.mean(prog)) if prog else None}
    n = len(poses)
    succ = sum(int(p["success"]) for p in poses)
    coll = sum(int(p["collided"]) for p in poses)
    prog = [p["progress_final"] for p in poses if p["progress_final"] is not None]
    return {
        "n": n, "success_rate": succ / n if n else float("nan"),
        "collision_rate": coll / n if n else float("nan"),
        "progress_mean": float(np.mean(prog)) if prog else None,
        "max_deviation": max((p["max_deviation"] for p in poses), default=0.0),
        "per_cell": per_cell, "poses": poses,
    }


class CanaryEvaluator:
    def __init__(self, *, godot_bin=DEFAULT_GODOT_BIN, project_dir="godot", scene=SCENE,
                 port=5599, curriculum_level=0.0, max_steps=300, host="127.0.0.1", timeout=60.0):
        self.manager = GodotProcessManager(godot_bin=godot_bin, project_dir=project_dir, scene_path=scene)
        self.manager.start_many([port], headless=True)
        self.env = ScenarioGymEnv(host=host, port=port, seed=0, timeout=timeout)
        self.agent_id = self.env.agent_id
        self.curriculum_level = float(curriculum_level)
        self.max_steps = int(max_steps)
        if self.env.obs_dim != 27 or self.env.action_size != 7:
            raise RuntimeError(f"unexpected dims obs={self.env.obs_dim} action={self.env.action_size}")

    def _run_pose(self, action_fn, cell, seed):
        self.env.configure(demo_mode=True, demo_forced_region=REGION, demo_forced_cell=int(cell),
                           curriculum_level=self.curriculum_level, evaluation_mode=False,
                           recording_mode=False, manage_agent_cameras=False)
        obs, info = self.env.reset(seed=int(seed))
        reset_info = info.get("agent_info", {}).get("reset", {})
        got = int(reset_info.get("target_cell", -1))
        if got != int(cell):
            raise RuntimeError(f"forced cell {cell} not honoured (got {got}); fail-closed")
        pose_id = str(reset_info.get("target_pose_id", f"{REGION}:{cell}:{seed}"))
        diag = new_episode_agent_diagnostics()
        reached = terminated = truncated = False
        steps = 0
        max_dev = 0.0
        for step in range(self.max_steps):
            action, dev = action_fn(obs)
            if dev is not None:
                max_dev = max(max_dev, float(dev))
            obs, reward, terminated, truncated, info = self.env.step(action)
            ai = info.get("agent_info", {})
            update_episode_agent_diagnostics(diag, ai)
            if agent_succeeded(ai):
                reached = True
            steps = step + 1
            if terminated or truncated:
                break
        fin = finalize_episode_agent_diagnostics(diag)
        return {"cell": int(cell), "seed": int(seed), "pose_id": pose_id, "success": bool(reached),
                "collided": bool(fin["collided"]), "collision_sources": list(fin["collision_sources"]),
                "progress_final": fin["progress_final"], "steps": int(steps),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "max_deviation": float(max_dev)}

    def eval_deck(self, action_fn, deck):
        return _summarize([self._run_pose(action_fn, d["cell"], d["seed"]) for d in deck])

    def close(self):
        try:
            self.env.close()
        finally:
            self.manager.stop_all() if hasattr(self.manager, "stop_all") else self.manager.stop()


def bc_action_fn(bc_actor):
    """Clone action: BC actor output in [-1,1]. Deviation is 0 by definition."""
    import tensorflow as tf
    def fn(obs):
        a = bc_actor(tf.convert_to_tensor(np.asarray(obs, np.float32).reshape(1, -1)),
                     training=False).numpy().reshape(-1)
        return a, 0.0
    return fn


def residual_action_fn(bc_actor, residual_model, *, delta_scale=0.05):
    """Residual action: clip(bc + delta, -1, 1); returns (action, max|action-bc|)."""
    import tensorflow as tf
    def fn(obs):
        o = tf.convert_to_tensor(np.asarray(obs, np.float32).reshape(1, -1))
        bc = bc_actor(o, training=False).numpy().reshape(-1)
        delta = delta_scale * np.tanh(residual_model(o, training=False).numpy().reshape(-1))
        action = np.clip(bc + delta, -1.0, 1.0)
        return action.astype(np.float32), float(np.max(np.abs(action - bc)))
    return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--godot-project", default="godot")
    ap.add_argument("--godot-scene", default=SCENE)
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--bc-weights", default="checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5")
    ap.add_argument("--residual-weights", default=None)
    ap.add_argument("--cells", type=int, nargs="*", default=None, help="default = all 11 Easy cells")
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--curriculum-level", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args()

    import tensorflow as tf
    from core.models import build_continuous_actor
    from core.residual_actor import build_residual_actor

    cells = args.cells or load_easy_cells(args.manifest)
    seeds = [args.seed_base + i for i in range(args.per_cell)]
    deck = build_deck(cells, seeds)

    bc = build_continuous_actor(obs_dim=27, action_size=7)
    bc.load_weights(args.bc_weights)
    bc.trainable = False
    if args.residual_weights:
        res = build_residual_actor(27, 7)
        res.load_weights(args.residual_weights)
        action_fn = residual_action_fn(bc, res)
        label = "residual"
    else:
        action_fn = bc_action_fn(bc)
        label = "clone"

    ev = CanaryEvaluator(godot_bin=args.godot_bin, project_dir=args.godot_project,
                         scene=args.godot_scene, port=args.port,
                         curriculum_level=args.curriculum_level, max_steps=args.max_steps)
    try:
        summary = ev.eval_deck(action_fn, deck)
    finally:
        ev.close()
    summary["label"] = label
    summary["deck_size"] = len(deck)
    out = {k: summary[k] for k in ("label", "deck_size", "n", "success_rate", "collision_rate",
                                   "progress_mean", "max_deviation", "per_cell")}
    print("CANARY_EVAL " + json.dumps(out), flush=True)
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, default=float))
        print("WROTE", args.summary_json, flush=True)


if __name__ == "__main__":
    main()
