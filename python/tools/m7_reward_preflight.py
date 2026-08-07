"""M7.0: reward / RL PREFLIGHT (NO training). At real Stage-C near-gate states (2-3 cm), branch REAL
Godot transitions and compare the return of: pure composite; a small sustained bias TOWARD the target;
the OPPOSITE bias; symmetric per-joint biases. Verifies the reward actually distinguishes the last
millimetre: approaching -> higher return, moving away -> lower, success-after-hold -> a real bonus,
and NO positive reward accumulates by stalling at ~2.14 cm. If the signal does not separate the last
mm (globally OR on cells 25/28/31/41), STOP before TD3. Composite frozen; nothing trained.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.policy_artifact import load_policy_model  # noqa: E402
from tools.c0_feasibility_oracle import load_goal_cfgs  # noqa: E402
from tools.dagger_round import agent_info_of, denorm_joints  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN, SCENE  # noqa: E402
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402

REGION = "easy"
POLICY = "checkpoints/openarm_reach_hold_m6_3_policy/policy.keras"
HARD = (25, 28, 31, 41)


class Probe:
    def __init__(self, *, godot_bin, port, curriculum, max_steps, policy=POLICY, region=REGION,
                 scene=SCENE, project_dir="godot", distance_key="position_error_m",
                 terminal_key="terminal_reason", success_value="target_reached"):
        self.mgr = GodotProcessManager(godot_bin=godot_bin, project_dir=project_dir, scene_path=scene)
        self.mgr.start_many([port], headless=True)
        self.env = ScenarioGymEnv(host="127.0.0.1", port=port, seed=0, timeout=60.0)
        self.curriculum = float(curriculum)
        self.max_steps = int(max_steps)
        self.region = str(region)
        self.policy_path = str(policy)
        self.distance_key = str(distance_key)
        self.terminal_key = str(terminal_key)
        self.success_value = str(success_value)
        self.model, *_ = load_policy_model(self.policy_path)

    def close(self):
        try:
            self.env.close()
        finally:
            self.mgr.stop_all()

    def _act(self, obs):
        return np.clip(self.model(tf.convert_to_tensor(obs[None, :], tf.float32),
                                  training=False).numpy()[0], -1.0, 1.0)

    def _reset(self, cell, seed):
        self.env.configure(demo_mode=True, demo_forced_region=self.region, demo_forced_cell=int(cell),
                           curriculum_level=self.curriculum, evaluation_mode=False,
                           recording_mode=False, manage_agent_cameras=False)
        obs, info = self.env.reset(seed=int(seed))
        if int(agent_info_of(info).get("reset", {}).get("target_cell", -1)) != int(cell):
            raise RuntimeError("forced cell not honoured")
        return obs, info

    def rollout(self, cell, seed):
        """Deterministic composite rollout; returns per-step obs, dist, reward + success flag."""
        obs, info = self._reset(cell, seed)
        obss, dists, rews = [], [], []
        reached = False
        for _ in range(self.max_steps):
            obss.append(obs.astype(np.float32))
            dists.append(float(agent_info_of(info).get(self.distance_key, np.nan)))
            obs, r, term, trunc, info = self.env.step(self._act(obs).astype(np.float32))
            rews.append(float(r))
            if str(agent_info_of(info).get(self.terminal_key, "")) == self.success_value:
                reached = True
            if term or trunc:
                break
        return {"obs": obss, "dist": dists, "rew": rews, "reached": reached}

    def branch_return(self, cell, seed, k, bias, window):
        """Replay composite prefix [0,k); apply clip(composite+bias) for a SHORT window [k,k+window)
        (a small nudge, not a sustained slam), then pure composite. Returns tail return, mean reward,
        whether it reaches success, and the MIN distance reached in the tail."""
        obs, _info = self._reset(cell, seed)
        rews = []
        reached = False
        min_d = np.inf
        for step in range(self.max_steps):
            a = self._act(obs)
            if k <= step < k + window:
                a = np.clip(a + bias, -1.0, 1.0)
            obs, r, term, trunc, info = self.env.step(a.astype(np.float32))
            ai = agent_info_of(info)
            dd = float(ai.get(self.distance_key, np.nan))
            if np.isfinite(dd):
                min_d = min(min_d, dd)
            if step >= k:
                rews.append(float(r))
            if str(ai.get(self.terminal_key, "")) == self.success_value:
                reached = True
            if term or trunc:
                break
        return {"return": float(np.sum(rews)), "mean_reward": float(np.mean(rews)) if rews else float("nan"),
                "reached": reached, "min_dist": float(min_d)}

    def branch_full(self, cell, seed, k, bias, window, max_tail=None):
        """Like branch_return but returns the full tail reward LIST + the terminal-bonus reward (the
        reward at the target_reached step, if any) + min distance + reached, so the caller can compute
        one-step, short-horizon, with/without-bonus and discounted/undiscounted returns.

        max_tail (optional): stop the branch `max_tail` steps after k (tail length <= max_tail) instead
        of running to episode end. Default None = run to episode end (unchanged for m7_0/m7_0b callers).
        """
        obs, _info = self._reset(cell, seed)
        tail = []
        reached = False
        bonus = 0.0
        min_d = np.inf
        for step in range(self.max_steps):
            a = self._act(obs)
            if k <= step < k + window:
                a = np.clip(a + bias, -1.0, 1.0)
            obs, r, term, trunc, info = self.env.step(a.astype(np.float32))
            ai = agent_info_of(info)
            dd = float(ai.get(self.distance_key, np.nan))
            if np.isfinite(dd):
                min_d = min(min_d, dd)
            if step >= k:
                tail.append(float(r))
            if str(ai.get(self.terminal_key, "")) == self.success_value:
                reached = True
                bonus = float(r)                 # the terminal-bonus reward at the reaching step
            if term or trunc:
                break
            if max_tail is not None and step >= k + max_tail - 1:
                break                             # bounded audit horizon (cost control)
        return {"tail": tail, "reached": reached, "bonus": bonus, "min_dist": float(min_d)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=6570)
    ap.add_argument("--plans", default="python/demos/openarm_reach_hold_m7/plan_m7.json")
    ap.add_argument("--curriculum-level", type=float, default=0.4)
    ap.add_argument("--near-lo", type=float, default=0.02)
    ap.add_argument("--near-hi", type=float, default=0.03)
    ap.add_argument("--bias-eps", type=float, default=0.08)
    ap.add_argument("--perjoint-eps", type=float, default=0.08)
    ap.add_argument("--bias-window", type=int, default=25, help="steps the nudge is applied (short, no slam)")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-poses", type=int, default=44, help="cap near-gate poses probed (cost control)")
    ap.add_argument("--out", default="python/demos/openarm_reach_hold_m7/m7_0_reward_preflight.json")
    args = ap.parse_args()

    goals = load_goal_cfgs(args.plans)
    pr = Probe(godot_bin=args.godot_bin, port=args.port, curriculum=args.curriculum_level,
               max_steps=args.max_steps)
    rows = []
    try:
        for (cell, seed), goal in sorted(goals.items()):
            roll = pr.rollout(cell, seed)
            d = np.asarray(roll["dist"], np.float64)
            near = np.where((d >= args.near_lo) & (d <= args.near_hi))[0]
            if len(near) == 0:
                continue                                        # composite never entered 2-3cm
            k = int(near[len(near) // 2])                       # a representative near-gate decision step
            joints = denorm_joints(roll["obs"][k])
            direction = goal - joints
            nrm = float(np.linalg.norm(direction))
            if nrm < 1e-6:
                continue
            toward = args.bias_eps * (direction / nrm)
            W = args.bias_window
            comp = pr.branch_return(cell, seed, k, np.zeros(7), W)
            tw = pr.branch_return(cell, seed, k, toward, W)
            aw = pr.branch_return(cell, seed, k, -toward, W)
            perjoint = []
            if cell in HARD:                                    # per-joint symmetry only on hard cells (cost)
                for j in range(7):
                    e = np.zeros(7); e[j] = args.perjoint_eps
                    perjoint.append({"j": j, "plus": pr.branch_return(cell, seed, k, e, W)["return"],
                                     "minus": pr.branch_return(cell, seed, k, -e, W)["return"]})
            # stall reward = composite tail mean per-step reward when it does NOT reach (hovering ~2.14cm)
            stall_rew = comp["mean_reward"] if not comp["reached"] else float("nan")
            rows.append({"cell": int(cell), "seed": int(seed), "k": k, "dist_k": float(d[k]),
                         "ret_composite": comp["return"], "ret_toward": tw["return"], "ret_away": aw["return"],
                         "toward_reached": tw["reached"], "composite_reached": comp["reached"],
                         "toward_min_dist": tw["min_dist"], "composite_min_dist": comp["min_dist"],
                         "away_min_dist": aw["min_dist"], "stall_mean_reward": stall_rew, "perjoint": perjoint})
            print(f"[m7.0] cell {cell} seed {seed} d_k {d[k]:.4f}  ret comp {comp['return']:.3f} "
                  f"toward {tw['return']:.3f} away {aw['return']:.3f}  stall_r {stall_rew:.4f} "
                  f"toward_reached {tw['reached']}", flush=True)
            if len(rows) >= args.max_poses:
                break
    finally:
        pr.close()

    stalling = [r for r in rows if not r["composite_reached"]]   # the last-mm question lives here

    def frac(pred, rs):
        return (float(np.mean([pred(r) for r in rs])) if rs else float("nan")), len(rs)

    # directionality: the toward-nudge must actually reduce distance vs away (else it is not 'toward')
    dir_ok, nd = frac(lambda r: r["toward_min_dist"] < r["away_min_dist"], stalling)
    # reward gradient on STALLING poses where the nudge was directionally valid
    valid = [r for r in stalling if r["toward_min_dist"] < r["away_min_dist"]]
    grad_ok, n = frac(lambda r: r["ret_toward"] > r["ret_composite"] > r["ret_away"], valid)
    toward_better, _ = frac(lambda r: r["ret_toward"] > r["ret_composite"], valid)
    away_worse, _ = frac(lambda r: r["ret_away"] < r["ret_composite"], valid)
    grad_hard, nh = frac(lambda r: r["ret_toward"] > r["ret_composite"] > r["ret_away"],
                         [r for r in valid if r["cell"] in HARD])
    stall_positive, ns = frac(lambda r: r["stall_mean_reward"] > 0.0,
                              [r for r in stalling if np.isfinite(r["stall_mean_reward"])])
    success_bonus, _ = frac(lambda r: r["toward_reached"], valid)
    per_cell = {}
    for c in sorted(set(r["cell"] for r in valid)):
        g, ncell = frac(lambda r: r["ret_toward"] > r["ret_composite"] > r["ret_away"],
                        [r for r in valid if r["cell"] == c])
        per_cell[int(c)] = {"n": ncell, "gradient_ok_frac": g}
    verdict = {
        "n_probed": len(rows), "n_stalling": len(stalling), "n_valid_direction": n,
        "nudge_directionality_frac": dir_ok, "gradient_ordering_frac": grad_ok,
        "toward_better_frac": toward_better, "away_worse_frac": away_worse,
        "gradient_hard_cells_frac": grad_hard, "n_hard": nh,
        "stall_positive_reward_frac": stall_positive, "n_stall_measured": ns,
        "toward_reaches_success_frac": success_bonus, "per_cell": per_cell,
        # PASS: on directionally-valid stalling states the reward separates the last mm globally AND on
        # hard cells, and stalling at ~2.14cm does NOT accumulate positive reward.
        "PASS": bool(grad_ok >= 0.8 and grad_hard >= 0.8 and stall_positive <= 0.1),
    }
    report = {"config": vars(args), "verdict": verdict, "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print("M7_0_VERDICT " + json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
