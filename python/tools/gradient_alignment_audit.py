"""READ-ONLY gradient-alignment audit for the M6.1 critic v6 vs the canary v2 residual checkpoints.

Question: does the critic's Q-ascent direction match the REAL return effect of the residual action?
The canary showed rising Q but falling success; this audit measures the (mis)alignment directly.

NO changes to reward / dataset / critic / actor / thresholds. NO new final deck: it re-uses the SAME
selection deck already consumed (seeds 710000+, 11 Easy cells) and the residual checkpoints
0/250/500/1000/1500/2000 saved by the canary. Everything here is evaluation only.

Phase 1 (per pose x checkpoint): roll out clone + each residual dose; record success, total reward,
steps, terminal reason, final/min distance, orientation error, max joint speed, max hold; along the
CLONE trajectory compute Q(s,a_clone), Q(s,a_res), delta_Q and ||a_res-a_clone||; classify each pose
gained / lost / unchanged vs the clone.

Phase 2 (causal, on lost+gained poses at the max dose): replay the clone prefix to a decision state
s_k, branch A = clone action then clone tail, branch B = ONE residual action then clone tail; measure
the REAL delta_return at horizons {1,2,4,8,episode}; compare sign(delta_Q) with sign(delta_return).
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
from core.models import build_continuous_actor, build_continuous_critic  # noqa: E402
from core.residual_actor import build_residual_actor  # noqa: E402
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN, DEFAULT_MANIFEST, SCENE, build_deck, load_easy_cells  # noqa: E402

BC_WEIGHTS = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"
CRITIC_CKPT = "checkpoints/openarm_reach_hold_m6_1_critic_v6/critic-3750-1"
CANARY_DIR = "checkpoints/openarm_reach_hold_m6_1_canary_v2"
REGION = "easy"
DELTA_SCALE = 0.05
HORIZONS = (1, 2, 4, 8)


def _spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64); rx -= rx.mean()
    ry = np.argsort(np.argsort(y)).astype(np.float64); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


class Auditor:
    def __init__(self, *, godot_bin, port, max_steps):
        self.manager = GodotProcessManager(godot_bin=godot_bin, project_dir="godot", scene_path=SCENE)
        self.manager.start_many([port], headless=True)
        self.env = ScenarioGymEnv(host="127.0.0.1", port=port, seed=0, timeout=60.0)
        self.max_steps = int(max_steps)
        self.bc = build_continuous_actor(obs_dim=27, action_size=7); self.bc.load_weights(BC_WEIGHTS)
        self.bc.trainable = False
        self.c1 = build_continuous_critic(obs_dim=27, action_size=7)
        self.c2 = build_continuous_critic(obs_dim=27, action_size=7)
        tc = build_continuous_critic(obs_dim=27, action_size=7)
        tc2 = build_continuous_critic(obs_dim=27, action_size=7)
        tf.train.Checkpoint(critic=self.c1, critic2=self.c2, target_critic=tc,
                            target_critic2=tc2).restore(CRITIC_CKPT).expect_partial()

    def close(self):
        try:
            self.env.close()
        finally:
            self.manager.stop_all()

    # --- policies (numpy obs -> numpy action) ---
    def bc_action(self, obs):
        return self.bc(tf.convert_to_tensor(np.asarray(obs, np.float32).reshape(1, -1)),
                       training=False).numpy().reshape(-1)

    def res_action(self, residual, obs):
        o = tf.convert_to_tensor(np.asarray(obs, np.float32).reshape(1, -1))
        bc = self.bc(o, training=False).numpy().reshape(-1)
        delta = DELTA_SCALE * np.tanh(residual(o, training=False).numpy().reshape(-1))
        return np.clip(bc + delta, -1.0, 1.0).astype(np.float32)

    def qmin(self, obs, act):
        o = tf.convert_to_tensor(np.asarray(obs, np.float32), tf.float32)
        a = tf.convert_to_tensor(np.asarray(act, np.float32), tf.float32)
        return tf.minimum(self.c1([o, a], training=False), self.c2([o, a], training=False)).numpy().reshape(-1)

    def _configure_reset(self, cell, seed):
        self.env.configure(demo_mode=True, demo_forced_region=REGION, demo_forced_cell=int(cell),
                           curriculum_level=0.0, evaluation_mode=False, recording_mode=False,
                           manage_agent_cameras=False)
        obs, info = self.env.reset(seed=int(seed))
        got = int(info.get("agent_info", {}).get("reset", {}).get("target_cell", -1))
        if got != int(cell):
            raise RuntimeError(f"forced cell {cell} not honoured (got {got})")
        return obs

    def rollout(self, action_of, cell, seed, *, record=True):
        """action_of(obs)->action. Returns episode metrics + (if record) per-step obs/reward."""
        obs = self._configure_reset(cell, seed)
        diag = new_episode_agent_diagnostics()
        traj_obs, rewards = [], []
        reached = terminated = truncated = False
        steps = 0
        for step in range(self.max_steps):
            if record:
                traj_obs.append(np.asarray(obs, np.float32))
            action = action_of(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            ai = info.get("agent_info", {})
            update_episode_agent_diagnostics(diag, ai)
            rewards.append(float(reward))
            if agent_succeeded(ai):
                reached = True
            steps = step + 1
            if terminated or truncated:
                break
        fin = finalize_episode_agent_diagnostics(diag)
        return {
            "cell": int(cell), "seed": int(seed), "success": bool(reached),
            "total_reward": float(np.sum(rewards)), "steps": int(steps),
            "terminal_reasons": list(fin["terminal_reasons"]), "collided": bool(fin["collided"]),
            "dist_final": fin["position_error_final"], "dist_min": fin["position_error_min"],
            "orient_final": fin["orientation_error_final"], "orient_min": fin["orientation_error_min"],
            "vel_final": fin["max_joint_speed_final"], "hold_max": fin["hold_frames_max"],
            "success_thresholds": fin["success_thresholds"],
            "traj_obs": np.asarray(traj_obs, np.float32) if record else None,
            "rewards": np.asarray(rewards, np.float64),
        }

    def branch_return(self, residual, cell, seed, k, first_is_residual):
        """Replay clone prefix [0,k), take ONE action at k (clone if not first_is_residual else
        residual), then clone tail. Returns per-step rewards from step 0 (rewards[k:] is the branch
        tail). Deterministic: same reset seed + deterministic clone prefix reproduces s_k."""
        obs = self._configure_reset(cell, seed)
        rewards = []
        for step in range(self.max_steps):
            if step == k and first_is_residual:
                action = self.res_action(residual, obs)
            else:
                action = self.bc_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        return np.asarray(rewards, np.float64)


def _attribute_failure(pose, thr):
    """On a LOST pose, which Stage-A gate component blocked success (best-effort from per-episode
    min/max diagnostics): position > orientation > hold."""
    dmin = pose["dist_min"]; omin = pose["orient_min"]; hold = pose["hold_max"] or 0
    d_gate = thr.get("distance_m", 0.08); a_gate = thr.get("orientation_deg", 45.0)
    hold_gate = thr.get("hold_physics_frames", 1)
    if dmin is None or dmin > d_gate:
        return "position"
    if omin is not None and omin > a_gate:
        return "orientation"
    if (hold or 0) < hold_gate:
        return "hold_or_stillness"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=5630)
    ap.add_argument("--canary-dir", default=CANARY_DIR)
    ap.add_argument("--doses", default="0,250,500,1000,1500,2000")
    ap.add_argument("--causal-dose", type=int, default=2000, help="dose whose lost/gained poses get the causal audit")
    ap.add_argument("--selection-seed-base", type=int, default=710000)
    ap.add_argument("--selection-per-cell", type=int, default=8)
    ap.add_argument("--causal-points", type=int, default=5, help="decision points per lost/gained pose")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="checkpoints/openarm_reach_hold_m6_1_canary_v2/gradient_alignment_audit.json")
    args = ap.parse_args()

    doses = [int(x) for x in args.doses.split(",")]
    cells = load_easy_cells(DEFAULT_MANIFEST)
    seeds = [args.selection_seed_base + i for i in range(args.selection_per_cell)]
    deck = build_deck(cells, seeds)
    print(f"AUDIT deck: {len(cells)} cells x {len(seeds)} seeds = {len(deck)} poses; doses {doses}", flush=True)

    au = Auditor(godot_bin=args.godot_bin, port=args.port, max_steps=args.max_steps)
    residuals = {}
    for dz in doses:
        r = build_residual_actor(27, 7)
        r.load_weights(str(Path(args.canary_dir) / f"residual-{dz}.weights.h5"))
        residuals[dz] = r

    report = {"deck_poses": len(deck), "doses": doses, "phase1": {}, "phase2": {}}
    try:
        # ---- PHASE 1 ----
        clone = {}
        for d in deck:
            key = f"{d['cell']}:{d['seed']}"
            clone[key] = au.rollout(au.bc_action, d["cell"], d["seed"], record=True)
        clone_sr = float(np.mean([c["success"] for c in clone.values()]))
        print(f"CLONE selection: sr={clone_sr:.3f}", flush=True)

        per_dose = {}
        for dz in doses:
            res = residuals[dz]
            gained = lost = unchanged = 0
            dq_all, dact_all = [], []
            rows = {}
            for d in deck:
                key = f"{d['cell']}:{d['seed']}"
                ro = au.rollout(lambda o, rr=res: au.res_action(rr, o), d["cell"], d["seed"], record=False)
                cl = clone[key]
                # delta_Q along the CLONE trajectory (states the causal test will branch from)
                obs = cl["traj_obs"]
                a_clone = np.stack([au.bc_action(o) for o in obs]) if len(obs) else np.zeros((0, 7))
                a_res = np.stack([au.res_action(res, o) for o in obs]) if len(obs) else np.zeros((0, 7))
                if len(obs):
                    dq = au.qmin(obs, a_res) - au.qmin(obs, a_clone)
                    dact = np.linalg.norm(a_res - a_clone, axis=1)
                    dq_all.extend(dq.tolist()); dact_all.extend(dact.tolist())
                    dq_mean = float(np.mean(dq)); dact_mean = float(np.mean(dact))
                else:
                    dq_mean = dact_mean = float("nan")
                cls = ("gained" if ro["success"] and not cl["success"] else
                       "lost" if cl["success"] and not ro["success"] else "unchanged")
                gained += cls == "gained"; lost += cls == "lost"; unchanged += cls == "unchanged"
                rows[key] = {"cell": d["cell"], "seed": d["seed"], "class": cls,
                             "clone_success": cl["success"], "res_success": ro["success"],
                             "clone_reward": cl["total_reward"], "res_reward": ro["total_reward"],
                             "res_dist_min": ro["dist_min"], "res_orient_min": ro["orient_min"],
                             "res_hold_max": ro["hold_max"], "delta_q_mean": dq_mean,
                             "dact_mean": dact_mean}
            res_sr = float(np.mean([r["res_success"] for r in rows.values()]))
            per_dose[dz] = {"res_sr": res_sr, "succ_drop": clone_sr - res_sr,
                            "gained": gained, "lost": lost, "unchanged": unchanged,
                            "delta_q_mean": float(np.mean(dq_all)) if dq_all else float("nan"),
                            "dact_mean": float(np.mean(dact_all)) if dact_all else float("nan"),
                            "rows": rows}
            print(f"DOSE {dz}: res_sr={res_sr:.3f} drop={clone_sr-res_sr:.3f} "
                  f"gained={gained} lost={lost} unchanged={unchanged} "
                  f"dQ_mean={per_dose[dz]['delta_q_mean']:.4f} dact_mean={per_dose[dz]['dact_mean']:.4f}",
                  flush=True)
        report["phase1"] = {"clone_sr": clone_sr,
                            "per_dose": {dz: {k: v for k, v in per_dose[dz].items() if k != "rows"}
                                         for dz in doses}}

        # ---- PHASE 2: causal on lost+gained poses at causal_dose ----
        cd = args.causal_dose
        res = residuals[cd]
        focus = [k for k, r in per_dose[cd]["rows"].items() if r["class"] in ("lost", "gained")]
        print(f"CAUSAL (dose {cd}): {len(focus)} lost/gained poses x {args.causal_points} decision points",
              flush=True)
        pairs = []          # (delta_Q, delta_return_episode, cell, class, horizons dict)
        fail_attr = {}
        reward_up_success_down = 0
        for key in focus:
            r0 = per_dose[cd]["rows"][key]
            cell, seed = r0["cell"], r0["seed"]
            cl = clone[f"{cell}:{seed}"]
            klen = len(cl["traj_obs"])
            if klen < 2:
                continue
            if r0["class"] == "lost":
                fa = _attribute_failure(
                    {"dist_min": r0["res_dist_min"], "orient_min": r0["res_orient_min"],
                     "hold_max": r0["res_hold_max"]}, cl["success_thresholds"])
                fail_attr[fa] = fail_attr.get(fa, 0) + 1
                if r0["res_reward"] >= r0["clone_reward"]:
                    reward_up_success_down += 1
            pts = sorted(set(int(f * (klen - 1)) for f in np.linspace(0.1, 0.9, args.causal_points)))
            for k in pts:
                s_k = cl["traj_obs"][k]
                a_clone = au.bc_action(s_k); a_res = au.res_action(res, s_k)
                dq = float(au.qmin(s_k.reshape(1, -1), a_res.reshape(1, -1))[0]
                           - au.qmin(s_k.reshape(1, -1), a_clone.reshape(1, -1))[0])
                rew_B = au.branch_return(res, cell, seed, k, first_is_residual=True)
                rew_A = cl["rewards"]                              # clone reference (branch A)
                dret = {}
                for h in HORIZONS:
                    a = float(np.sum(rew_A[k:k + h])); b = float(np.sum(rew_B[k:k + h]))
                    dret[str(h)] = b - a
                dret["episode"] = float(np.sum(rew_B[k:]) - np.sum(rew_A[k:]))
                pairs.append({"key": key, "cell": cell, "class": r0["class"], "k": k,
                              "delta_q": dq, "delta_return": dret})
        # ---- directional accuracy + Spearman ----
        def dir_acc(items):
            ok = [np.sign(p["delta_q"]) == np.sign(p["delta_return"]["episode"]) for p in items
                  if abs(p["delta_return"]["episode"]) > 1e-9]
            return float(np.mean(ok)) if ok else float("nan"), len(ok)
        acc_all, n_all = dir_acc(pairs)
        acc_lost, n_lost = dir_acc([p for p in pairs if p["class"] == "lost"])
        acc_gained, n_gained = dir_acc([p for p in pairs if p["class"] == "gained"])
        sp = _spearman([p["delta_q"] for p in pairs], [p["delta_return"]["episode"] for p in pairs])
        per_horizon = {}
        for h in [str(x) for x in HORIZONS] + ["episode"]:
            ok = [np.sign(p["delta_q"]) == np.sign(p["delta_return"][h]) for p in pairs
                  if abs(p["delta_return"][h]) > 1e-9]
            per_horizon[h] = {"dir_acc": float(np.mean(ok)) if ok else float("nan"), "n": len(ok),
                              "spearman": _spearman([p["delta_q"] for p in pairs],
                                                    [p["delta_return"][h] for p in pairs])}
        per_cell_acc = {}
        for c in sorted(set(p["cell"] for p in pairs)):
            per_cell_acc[c], _ = dir_acc([p for p in pairs if p["cell"] == c])
        report["phase2"] = {
            "causal_dose": cd, "n_focus_poses": len(focus), "n_pairs": len(pairs),
            "directional_accuracy_global": acc_all, "n_significant": n_all,
            "directional_accuracy_lost": acc_lost, "directional_accuracy_gained": acc_gained,
            "spearman_dQ_dReturn": sp, "directional_accuracy_per_cell": per_cell_acc,
            "directional_accuracy_per_horizon": per_horizon,
            "failure_attribution": fail_attr, "reward_up_success_down": reward_up_success_down,
            "pairs": pairs,
        }
        print(f"CAUSAL RESULT: dir_acc_global={acc_all:.3f} (n={n_all}) lost={acc_lost:.3f} "
              f"gained={acc_gained:.3f} spearman={sp:.3f} fail_attr={fail_attr} "
              f"reward_up_success_down={reward_up_success_down}", flush=True)
    finally:
        au.close()

    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print("WROTE " + args.out, flush=True)
    # verdict hint (read-only; decision is the user's)
    p2 = report["phase2"]
    if p2["directional_accuracy_global"] == p2["directional_accuracy_global"]:  # not nan
        verdict = ("ANTI-ALIGNED -> retire v6 from actor optimization"
                   if p2["directional_accuracy_global"] < 0.5 else
                   "reward-vs-gate: check reward_up_success_down / failure_attribution")
        print("AUDIT_VERDICT_HINT " + verdict, flush=True)
    print("AUDIT_END", flush=True)


if __name__ == "__main__":
    main()
