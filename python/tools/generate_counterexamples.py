"""Godot-driven counterexample generator (M6.1). Replay the expert plan prefix from reset (NO
teleport), branch into the 19 OOD families, run paired Q^BC one-step probes + K=8 OOD stress bursts
+ expert recovery, compute paired episode returns, and write the typed replay-only NPZ.

STOP-BEFORE-FULL-GENERATION: default `--smoke` limits to 1 cell x 1 phase x all 19 families so the
paired-return / censoring / split / counts can be inspected first.

Reaching an anchor: `configure(demo_mode, forced_cell, demo_plan)` + `reset(seed)` + N `step("manual")`
(the closed-loop plan-follower). Physics is deterministic in lockstep, so every branch re-reaches the
SAME s0 by replaying the same prefix. Branch actions are numpy (override the plan-follower);
`step("manual")` after a branch resumes the plan = expert recovery.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.counterexample import (  # noqa: E402
    FAMILIES, HORIZONS, K_MAX_BURST, CounterexampleDataset, assert_splits_disjoint,
    calibrate_margin, classify_rollout, discounted_return, paired_verdict, rollout_returns,
)
from core.evaluation import agent_succeeded  # noqa: E402
from core.models import build_continuous_actor, build_continuous_critic  # noqa: E402
from envs.process_manager import GodotProcessManager  # noqa: E402
from envs.scenario import ScenarioGymEnv  # noqa: E402

START_OBS_TOL = 1e-4  # baseline & candidate must start from the identical anchor state

REGION = "easy"
PHASE_FRAC = {"early": 0.25, "mid": 0.55, "late": 0.85}


def _scale(raw, low, high):
    return np.clip(low + 0.5 * (np.asarray(raw) + 1.0) * (high - low), low, high).astype(np.float32)


def _demo_config(agent_id, cell, waypoints, max_steps=300):
    return {
        "demo_mode": True, "demo_forced_region": REGION, "demo_forced_cell": int(cell),
        "curriculum_level": 1.0, "evaluation_mode": False, "recording_mode": True,
        "recording_agent_id": agent_id, "manage_agent_cameras": True,
        "max_steps": int(max_steps),  # the ScenarioController time-limit (scene default 300)
        "demo_plan": [list(map(float, w)) for w in waypoints],
    }


def _agent(info):
    return info.get("agent_info", {})


def _applied(ai, action_size):
    """The action Godot ACTUALLY applied this step (already the effective command, not re-scaled)."""
    a = ai.get("applied_action")
    if a is None:
        return np.zeros((action_size,), np.float32)
    return np.asarray(a, np.float32).reshape(action_size)


def _collision(ai):
    coll = bool(ai.get("collided", False))
    src = str(ai.get("collision_source", "") or "")
    return coll, ("self" in src.lower()), (src or "none")


def _ending(terminated, truncated, collided, self_coll):
    """Per-rollout/step ending tag for termination_reason (distinct from source and outcome)."""
    if self_coll:
        return "self_collision"
    if collided:
        return "collision"
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return "ongoing"


class FamilyActions:
    """Per-state action for each family + per-step burst rule (hold vs recompute)."""

    HOLD = {"random"}  # families whose burst holds the s0 command (+ sat_j*, added in __init__)

    def __init__(self, clone_model, collapsed_model, critic, critic2, low, high, *,
                 random_mode="fixed_burst", seed=0):
        self.clone, self.collapsed = clone_model, collapsed_model
        self.critic, self.critic2 = critic, critic2
        self.low, self.high = np.asarray(low, np.float32), np.asarray(high, np.float32)
        self.half = 0.5 * (self.high - self.low)
        self.random_mode = random_mode
        self.rng = np.random.default_rng(seed)
        self.HOLD = set(self.HOLD) | {f for f in FAMILIES if f.startswith("sat_j")}

    def clone_action(self, obs):
        return _scale(self.clone(np.expand_dims(obs, 0), training=False).numpy()[0], self.low, self.high)

    def _pga(self, obs):
        base = self.clone_action(obs)
        a = tf.Variable(base.reshape(1, -1))
        o = tf.convert_to_tensor(np.expand_dims(obs, 0), tf.float32)
        with tf.GradientTape() as t:
            q = tf.minimum(self.critic([o, a]), self.critic2([o, a]))
        g = t.gradient(q, a).numpy()[0]
        d = g / max(np.linalg.norm(g), 1e-12)
        return np.clip(base + 0.1 * self.half * d, self.low, self.high).astype(np.float32)

    def first_action(self, family, obs):
        if family == "collapsed_v4":
            return _scale(self.collapsed(np.expand_dims(obs, 0), training=False).numpy()[0], self.low, self.high)
        if family == "pga":
            return self._pga(obs)
        if family == "random":
            return self.rng.uniform(self.low, self.high).astype(np.float32)
        if family.startswith("perturbed_"):
            s = float(family.split("_")[1])
            return np.clip(self.clone_action(obs) + self.rng.normal(size=7) * s * self.half,
                           self.low, self.high).astype(np.float32)
        if family.startswith("sat_j"):
            parts = family.split("_")  # ["sat", "j<idx>", "pos"|"neg"]
            j = int(parts[1][1:])
            sign = parts[2]
            a = self.clone_action(obs).copy()
            a[j] = self.high[j] if sign == "pos" else self.low[j]
            return a
        if family == "baseline":
            return self.clone_action(obs)
        raise ValueError(family)

    def burst_action(self, family, obs, first_action, step_idx):
        """Action to apply at burst step `step_idx` (0-based)."""
        if step_idx == 0:
            return first_action
        if family in self.HOLD:
            if family == "random" and self.random_mode == "per_step":
                return self.rng.uniform(self.low, self.high).astype(np.float32)
            return first_action  # hold the selected command
        return self.first_action(family, obs)  # recompute (collapsed/pga/perturbed)


def _reach_anchor(env, cfg, seed, prefix_len):
    """reset(seed) + `prefix_len` plan-follower steps -> (obs at anchor s0). No teleport."""
    env.configure(**cfg)
    obs, info = env.reset(seed=int(seed))
    ai = _agent(info).get("reset", {})
    if int(ai.get("target_cell", -1)) != cfg["demo_forced_cell"]:
        raise RuntimeError(f"forced cell not honoured: {ai.get('target_cell')} != {cfg['demo_forced_cell']}")
    done = False
    for _ in range(prefix_len):
        obs, _r, term, trunc, info = env.step("manual")
        if term or trunc:
            done = True
            break
    return obs, done


def _scout_length(env, cfg, seed, max_steps):
    """Full plan-follower rollout length (steps until terminated/plan-exhausted)."""
    env.configure(**cfg)
    env.reset(seed=int(seed))
    n = 0
    for _ in range(max_steps):
        _obs, _r, term, trunc, _info = env.step("manual")
        n += 1
        if term or trunc:
            break
    return n


def _measure_env_max_steps(env, cfg, seed, hard_cap=2000):
    """Hold a neutral (zero) action from reset until the ENV ends the episode; return the step count
    at which `truncated`/`terminated` fires = the env's max_steps for this scene."""
    env.configure(**cfg)
    env.reset(seed=int(seed))
    zero = np.zeros(env.action_size, np.float32)
    for n in range(1, hard_cap + 1):
        _obs, _r, term, trunc, _info = env.step(zero)
        if term or trunc:
            return n
    return hard_cap


def _qbc_probe(env, cfg, seed, prefix_len, first_action, fa, gamma, tool_cap):
    """[first_action once] + [clone BC until env-terminated/truncated or the tool_cap backstop].
    Distinguishes `terminated` / env `truncated` (time-limit) / tool_cap so labels can be classified.
    Records the anchor start obs for the same-anchor check."""
    obs, done = _reach_anchor(env, cfg, seed, prefix_len)
    if done:
        return None
    start_obs = np.asarray(obs, np.float32)
    rewards, collided, self_coll = [], False, False
    terminated = env_truncated = hit_tool_cap = False
    is_success = False
    terminal_reason = ""
    first_tr = None
    fidelity_violations = 0  # recorded == applied (the GATE invariant); 0 by construction
    godot_clamps = 0         # requested != applied (the sim clamped an OOD action; diagnostic)
    a = np.asarray(first_action, np.float32)
    steps = 0
    for t in range(tool_cap):
        nobs, r, term, trunc, info = env.step(a)
        ai = _agent(info)
        c, sc, _src = _collision(ai)
        applied = _applied(ai, len(a)).astype(np.float32)
        rewards.append(float(r))
        collided = collided or c
        self_coll = self_coll or sc
        steps += 1
        if t == 0:
            recorded = applied  # record what Godot actually applied, never a re-scaled input
            if float(np.max(np.abs(recorded - applied))) >= 1e-6:
                fidelity_violations += 1              # regression guard: stored must equal applied
            if float(np.max(np.abs(a - applied))) >= 1e-6:
                godot_clamps += 1                     # sim modified the requested action (info only)
            first_tr = dict(obs=start_obs, actions=recorded, rewards=float(r),
                            next_obs=np.asarray(nobs, np.float32), terminated=bool(term),
                            truncated=bool(trunc), collision=bool(c), self_collision=bool(sc),
                            termination_reason=_ending(term, trunc, c, sc))
        if term or trunc:
            terminated, env_truncated = bool(term), bool(trunc)
            is_success = bool(agent_succeeded(ai)) if term else False
            terminal_reason = str(ai.get("terminal_reason", "") or "")
            break
        obs = nobs
        a = fa.clone_action(obs)  # tail follows the frozen clone
    else:
        hit_tool_cap = True
        is_success = False
        terminal_reason = ""
    oc = classify_rollout(terminated=terminated, truncated=env_truncated,
                          hit_tool_cap=hit_tool_cap, is_success=is_success)
    return dict(first_tr=first_tr, start_obs=start_obs, steps=steps,
                fidelity_violations=fidelity_violations, godot_clamps=godot_clamps,
                g_h=rollout_returns(rewards, gamma, HORIZONS),
                g_episode=discounted_return(rewards, gamma),
                collided=collided, self_collision=self_coll,
                terminated=terminated, env_truncated=env_truncated, hit_tool_cap=hit_tool_cap,
                success=is_success, terminal_reason=terminal_reason, outcome=oc["outcome"],
                label_usable=oc["label_usable"], return_complete=oc["return_complete"],
                failure_reason=(terminal_reason if (terminated and not is_success)
                                else "self_collision" if self_coll else "collision" if collided
                                else oc["outcome"]))


def _burst(env, cfg, seed, prefix_len, family, fa, k, do_recovery, recovery_steps):
    """Apply the family per-step rule for up to k steps (ood_burst); if no collision, resume the
    plan-follower for recovery. Returns (burst_rows, recovery_rows, fidelity_violations) as DISTINCT
    branches -- each row records the APPLIED action + its own step index + ending reason."""
    obs, done = _reach_anchor(env, cfg, seed, prefix_len)
    if done:
        return [], [], 0
    first_action = fa.first_action(family, obs)
    burst, recovery, fidelity, clamps = [], [], 0, 0
    collided = False
    for step_idx in range(k):
        a = np.asarray(fa.burst_action(family, obs, first_action, step_idx), np.float32)
        nobs, r, term, trunc, info = env.step(a)
        ai = _agent(info)
        c, sc, _src = _collision(ai)
        applied = _applied(ai, len(a)).astype(np.float32)
        recorded = applied
        if float(np.max(np.abs(recorded - applied))) >= 1e-6:
            fidelity += 1
        if float(np.max(np.abs(a - applied))) >= 1e-6:
            clamps += 1
        collided = collided or c
        burst.append(dict(obs=np.asarray(obs, np.float32), actions=recorded,
                          rewards=float(r), next_obs=np.asarray(nobs, np.float32),
                          terminated=bool(term), truncated=bool(trunc), step_in_branch=step_idx,
                          collision=c, self_collision=sc,
                          termination_reason=_ending(term, trunc, c, sc)))
        obs = nobs
        if term or trunc or c:
            break
    if do_recovery and not collided:
        for r_idx in range(recovery_steps):
            nobs, r, term, trunc, info = env.step("manual")
            ai = _agent(info)
            c, sc, _src = _collision(ai)
            applied = _applied(ai, env.action_size)  # plan-follower command, already the applied one
            recovery.append(dict(obs=np.asarray(obs, np.float32), actions=applied.astype(np.float32),
                                 rewards=float(r), next_obs=np.asarray(nobs, np.float32),
                                 terminated=bool(term), truncated=bool(trunc), step_in_branch=r_idx,
                                 collision=c, self_collision=sc,
                                 termination_reason=_ending(term, trunc, c, sc)))
            obs = nobs
            if term or trunc or c:
                break
    return burst, recovery, fidelity, clamps


def load_anchor_pool(plan_path, cells, phases, block_start, block_size):
    """Per (cell, phase) bucket return the ORDERED candidate-anchor pool = the split's disjoint pose
    BLOCK poses[block_start:block_start+block_size]. The block gives slack so a baseline that ends in
    terminal_failure can be discarded and the next pose substituted while keeping the quota. Splits
    use non-overlapping blocks -> pose_id/seed stay disjoint (validated by validate_split_files)."""
    plans = json.load(open(plan_path))["plans"]
    by_cell = {}
    for pid, p in plans.items():
        if int(p["cell"]) in cells:
            by_cell.setdefault(int(p["cell"]), []).append((pid, p))
    pool = {}
    for cell in cells:
        block = sorted(by_cell.get(cell, []), key=lambda kp: int(kp[1]["seed"]))[block_start: block_start + block_size]
        for phase in phases:
            pool[(cell, phase)] = [dict(cell=cell, phase=phase, pose_id=pid, seed=int(p["seed"]),
                                        waypoints=p["waypoints"]) for pid, p in block]
    return pool


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--clone-weights", required=True)
    ap.add_argument("--collapsed-actor", required=True)
    ap.add_argument("--critic-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="17")
    ap.add_argument("--phases", default="mid")
    ap.add_argument("--n-per-bucket", type=int, default=1)
    ap.add_argument("--pose-block-start", type=int, default=0,
                    help="first pose index of this split's disjoint block (per cell).")
    ap.add_argument("--pose-block-size", type=int, default=24,
                    help="block width = quota + substitution slack for baseline-failure discards.")
    ap.add_argument("--k-max", type=int, default=K_MAX_BURST)
    ap.add_argument("--env-max-steps", type=int, default=300,
                    help="ScenarioController time-limit set via config (scene default 300).")
    ap.add_argument("--tool-cap-margin", type=int, default=32,
                    help="tool_cap = env_max_steps + this margin (never equal -> no race).")
    ap.add_argument("--recovery-steps", type=int, default=8)
    ap.add_argument("--margin-replicates", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--random-mode", choices=["fixed_burst", "per_step"], default="fixed_burst")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--godot-bin", required=True)
    ap.add_argument("--godot-project", default="godot")
    ap.add_argument("--godot-scene", default="res://scenarios/robotarms/openarm_reach_hold_scenario.tscn")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6600)
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main():
    args = parse_args()
    cells = [int(c) for c in str(args.cells).split(",")]
    phases = [p for p in str(args.phases).split(",")]
    if args.smoke:
        cells, phases, args.n_per_bucket = cells[:1], phases[:1], min(args.n_per_bucket, 1)
        print(f"SMOKE: cells={cells} phases={phases} families={len(FAMILIES)} n_per_bucket={args.n_per_bucket}", flush=True)

    manager = GodotProcessManager(godot_bin=args.godot_bin, project_dir=args.godot_project,
                                  scene_path=args.godot_scene)
    manager.start_many([args.port], headless=args.headless, debug=False,
                       user_args=["--recording-mode", "--disable-agent-replication"])
    env = None
    try:
        env = ScenarioGymEnv(host=args.host, port=args.port, seed=0)
        low, high = env.action_low, env.action_high
        clone = build_continuous_actor(obs_dim=env.obs_dim, action_size=env.action_size)
        clone.load_weights(args.clone_weights)
        collapsed = build_continuous_actor(obs_dim=env.obs_dim, action_size=env.action_size)
        collapsed.load_weights(args.collapsed_actor)
        critic = build_continuous_critic(obs_dim=env.obs_dim, action_size=env.action_size)
        critic2 = build_continuous_critic(obs_dim=env.obs_dim, action_size=env.action_size)
        tf.train.Checkpoint(critic=critic, critic2=critic2).restore(args.critic_checkpoint).expect_partial()
        fa = FamilyActions(clone, collapsed, critic, critic2, low, high,
                           random_mode=args.random_mode, seed=args.seed)

        pool = load_anchor_pool(args.plan, cells, phases, args.pose_block_start, args.pose_block_size)
        env_max_steps = int(args.env_max_steps)
        tool_cap = env_max_steps + int(args.tool_cap_margin)
        print(f"env_max_steps={env_max_steps}  tool_cap={tool_cap}  "
              f"pose_block=[{args.pose_block_start}:{args.pose_block_start+args.pose_block_size}]", flush=True)

        ds = CounterexampleDataset(env.obs_dim, env.action_size, gamma=args.gamma)
        branch_id = 0
        margins = []
        pose_ids_seen, seeds_seen = set(), set()
        invalid_anchor_branches = 0
        fidelity_violations = 0
        godot_clamps = 0
        margin_by_bucket = {}
        bucket_anchor_counts = collections.Counter()
        baseline_outcomes = collections.Counter()
        candidate_matrix = collections.Counter()
        paired_bad_by_rule = collections.Counter()
        discarded_baseline_failures = 0
        candidate_better_count = 0
        NAN = float("nan")
        nan_h = np.full(len(HORIZONS), NAN, np.float32)

        def _oc_na():
            return dict(outcome="n/a", label_usable=False, return_complete=False,
                        matched_horizon=False, censor_reason="n/a")

        def _row(tr, *, anc, fam, source, brid, step, blen, oc, base_rep_id, horizon_steps,
                 success=False, paired_bad=False, paired_bad_rule="none", candidate_better=False,
                 safety_negative=False, g_base=NAN, g_cand=NAN, g_base_ep=NAN, g_cand_ep=NAN, g_by_h=None):
            # Metis replay: dones = terminated ONLY (truncation is stored but NOT a Bellman terminal).
            g_by_h = nan_h if g_by_h is None else g_by_h
            rd = (g_cand - g_base) if (g_cand == g_cand and g_base == g_base) else NAN
            rd_ep = (g_cand_ep - g_base_ep) if (g_cand_ep == g_cand_ep and g_base_ep == g_base_ep) else NAN
            ds.add(obs=tr["obs"], actions=tr["actions"], rewards=tr["rewards"],
                   next_obs=tr["next_obs"], terminated=bool(tr["terminated"]),
                   truncated=bool(tr["truncated"]), dones=bool(tr["terminated"]),
                   branch_id=brid, step_in_branch=step, branch_length=blen, cell=anc["cell"],
                   seed=anc["seed"], phase=anc["phase"], pose_id=anc["pose_id"], family=fam,
                   source=source, failure_reason=tr["termination_reason"],
                   termination_reason=tr["termination_reason"], outcome=oc["outcome"],
                   baseline_replicate_id=base_rep_id,
                   collision=bool(tr.get("collision", False)), self_collision=bool(tr.get("self_collision", False)),
                   success=bool(success), paired_bad=bool(paired_bad), paired_bad_rule=str(paired_bad_rule),
                   candidate_better=bool(candidate_better), safety_negative=bool(safety_negative),
                   return_censored=(not oc["label_usable"]), return_complete=oc["return_complete"],
                   matched_horizon=bool(oc.get("matched_horizon", False)),
                   label_usable=oc["label_usable"], censor_reason=oc.get("censor_reason", "n/a"),
                   horizon_steps=int(horizon_steps), G_baseline=float(g_base), G_candidate=float(g_cand),
                   return_delta=float(rd), G_baseline_episode=float(g_base_ep),
                   G_candidate_episode=float(g_cand_ep), return_delta_episode=float(rd_ep),
                   return_delta_by_h=g_by_h)

        def _bucket_margin(cfg, seed, prefix_len):
            reps = []
            for _rep in range(args.margin_replicates):
                s0, d0 = _reach_anchor(env, cfg, seed, prefix_len)
                if d0:
                    break
                pr = _qbc_probe(env, cfg, seed, prefix_len, fa.clone_action(s0), fa, args.gamma, tool_cap)
                if pr is not None:
                    reps.append(pr["g_episode"])
            return calibrate_margin(reps) if len(reps) >= args.margin_replicates else 0.5

        for bcell in cells:
            for phase in phases:
                bkey = (bcell, phase)
                anchor_pool = pool.get(bkey, [])
                accepted = 0
                for anc in anchor_pool:
                    if accepted >= args.n_per_bucket:
                        break
                    cfg = _demo_config(env.agent_id, anc["cell"], anc["waypoints"], max_steps=env_max_steps)
                    L = _scout_length(env, cfg, anc["seed"], tool_cap)
                    prefix_len = max(1, min(L - 1, int(round(PHASE_FRAC[phase] * L))))
                    s0, done0 = _reach_anchor(env, cfg, anc["seed"], prefix_len)
                    if done0:
                        continue  # prefix already terminal -> substitute next pose
                    base = _qbc_probe(env, cfg, anc["seed"], prefix_len, fa.clone_action(s0), fa,
                                      args.gamma, tool_cap)
                    fidelity_violations += base["fidelity_violations"]
                    godot_clamps += base["godot_clamps"]
                    baseline_outcomes[base["outcome"]] += 1
                    # ---- baseline guard ----
                    if base["outcome"] in ("tool_cap", "ongoing"):
                        raise SystemExit(f"FAIL-CLOSED: baseline {base['outcome']} at {anc['pose_id']} "
                                         f"(tool_cap must sit above the env limit).")
                    if base["outcome"] == "terminal_failure":
                        discarded_baseline_failures += 1
                        print(f"  DISCARD {anc['pose_id']} baseline terminal_failure "
                              f"({base['terminal_reason'] or 'failure'}); substituting.", flush=True)
                        continue  # not a valid 'good' reference -> substitute next pose
                    # ---- accept anchor (baseline is success or env_time_limit) ----
                    if bkey not in margin_by_bucket:
                        margin_by_bucket[bkey] = _bucket_margin(cfg, anc["seed"], prefix_len)
                    margin = margin_by_bucket[bkey]
                    margins.append(margin)
                    pose_ids_seen.add(anc["pose_id"])
                    seeds_seen.add(anc["seed"])
                    print(f"anchor cell={anc['cell']} phase={phase} pose={anc['pose_id']} L={L} "
                          f"prefix={prefix_len} base_outcome={base['outcome']} success={base['success']} "
                          f"base_steps={base['steps']} margin={margin:.4f}", flush=True)

                    base_oc = dict(outcome=base["outcome"], label_usable=base["label_usable"],
                                   return_complete=base["return_complete"], matched_horizon=False,
                                   censor_reason="none")
                    _row(base["first_tr"], anc=anc, fam="baseline", source="baseline", brid=branch_id,
                         step=0, blen=1, oc=base_oc, base_rep_id=0, success=base["success"],
                         g_base=base["g_h"][8], g_cand=base["g_h"][8], g_base_ep=base["g_episode"],
                         g_cand_ep=base["g_episode"], g_by_h=np.zeros(len(HORIZONS), np.float32),
                         horizon_steps=base["steps"])
                    branch_id += 1
                    accepted += 1
                    bucket_anchor_counts[bkey] += 1

                    for fam in FAMILIES:
                        a0 = fa.first_action(fam, base["start_obs"])
                        probe = _qbc_probe(env, cfg, anc["seed"], prefix_len, a0, fa, args.gamma, tool_cap)
                        if probe is None:
                            continue
                        fidelity_violations += probe["fidelity_violations"]
                        godot_clamps += probe["godot_clamps"]
                        drift = float(np.max(np.abs(probe["start_obs"] - base["start_obs"])))
                        if drift > START_OBS_TOL:
                            invalid_anchor_branches += 1
                            print(f"  INVALID branch {fam}: start drift {drift:.2e}", flush=True)
                            continue
                        deltas = {H: probe["g_h"][H] - base["g_h"][H] for H in HORIZONS}
                        rd_h = np.asarray([deltas[H] for H in HORIZONS], np.float32)
                        v = paired_verdict(base_outcome=base["outcome"], base_success=base["success"],
                                           base_collision=base["collided"], cand_outcome=probe["outcome"],
                                           cand_success=probe["success"], cand_collision=probe["collided"],
                                           cand_label_usable=probe["label_usable"],
                                           return_delta_episode=probe["g_episode"] - base["g_episode"],
                                           deltas_by_h=deltas, margin=margin, horizon_steps=probe["steps"])
                        candidate_matrix[(base["outcome"], probe["outcome"])] += 1
                        paired_bad_by_rule[v["paired_bad_rule"]] += 1
                        candidate_better_count += int(v["candidate_better"])
                        matched = bool(base["env_truncated"] and probe["env_truncated"]
                                       and base["steps"] == probe["steps"])
                        cand_oc = dict(outcome=probe["outcome"], label_usable=probe["label_usable"],
                                       return_complete=probe["return_complete"], matched_horizon=matched,
                                       censor_reason=("none" if probe["label_usable"] else probe["outcome"]))
                        _row(probe["first_tr"], anc=anc, fam=fam, source="candidate", brid=branch_id,
                             step=0, blen=1, oc=cand_oc, base_rep_id=-1, success=probe["success"],
                             paired_bad=v["paired_bad"], paired_bad_rule=v["paired_bad_rule"],
                             candidate_better=v["candidate_better"], g_base=base["g_h"][8],
                             g_cand=probe["g_h"][8], g_base_ep=base["g_episode"],
                             g_cand_ep=probe["g_episode"], g_by_h=rd_h, horizon_steps=probe["steps"])
                        branch_id += 1

                        burst, recovery, fid, clmp = _burst(env, cfg, anc["seed"], prefix_len, fam, fa,
                                                            args.k_max, True, args.recovery_steps)
                        fidelity_violations += fid
                        godot_clamps += clmp
                        safety_neg = any(t["collision"] for t in burst)
                        for t in burst:
                            _row(t, anc=anc, fam=fam, source="ood_burst", brid=branch_id,
                                 step=t["step_in_branch"], blen=len(burst), oc=_oc_na(), base_rep_id=-1,
                                 safety_negative=safety_neg, horizon_steps=len(burst))
                        branch_id += 1
                        if recovery:
                            for t in recovery:
                                _row(t, anc=anc, fam=fam, source="recovery", brid=branch_id,
                                     step=t["step_in_branch"], blen=len(recovery), oc=_oc_na(),
                                     base_rep_id=-1, horizon_steps=len(recovery))
                            branch_id += 1

        assert_splits_disjoint({"this_split": {"pose_id": pose_ids_seen, "seed": seeds_seen}})
        global_margin = float(max(margins) if margins else 0.5)
        arr = ds.to_arrays()

        # ---- FAIL-CLOSED per-split validation (before declaring success) ----------------------
        fc = []
        expected_buckets = {(c, p) for c in cells for p in phases}
        for bk in sorted(expected_buckets):
            got = bucket_anchor_counts.get(bk, 0)
            if got != int(args.n_per_bucket):
                fc.append(f"bucket cell={bk[0]} phase={bk[1]} incomplete: {got}/{args.n_per_bucket} anchors")
        if invalid_anchor_branches > 0:
            fc.append(f"invalid_branches={invalid_anchor_branches}")
        if fidelity_violations > 0:
            fc.append(f"action_fidelity_violations={fidelity_violations}")
        # branch metadata: branch_length == rows-in-branch AND steps == 0..N-1
        for b in np.unique(arr["branch_id"]):
            mb = arr["branch_id"] == b
            n = int(mb.sum())
            if int(arr["branch_length"][mb][0]) != n or sorted(arr["step_in_branch"][mb].tolist()) != list(range(n)):
                fc.append(f"branch {int(b)} metadata incoherent (len {n})")
                break
        if int(arr["paired_bad"][arr["source"] != "candidate"].sum()) > 0:
            fc.append("paired_bad appears outside source==candidate")
        cand_m = arr["source"] == "candidate"
        base_m = arr["source"] == "baseline"
        # gate: no baseline terminal_failure survived the guard; every candidate reached a NATURAL
        # end (no tool_cap/ongoing).
        if int((arr["outcome"][base_m] == "terminal_failure").sum()) > 0:
            fc.append("baseline terminal_failure present in dataset (guard failed)")
        cand_out = set(arr["outcome"][cand_m].tolist())
        bad_cand = cand_out - {"success", "terminal_failure", "env_time_limit"}
        if bad_cand:
            fc.append(f"candidate tool_cap/ongoing/unclassifiable outcomes: {sorted(bad_cand)}")
        if fc:
            for msg in fc:
                print("FAIL_CLOSED:", msg, flush=True)
            raise SystemExit(f"FAIL-CLOSED: split {args.out} rejected ({len(fc)} issue(s)); not accepted.")

        manifest = dict(generator="generate_counterexamples", version=4, plan=args.plan,
                        cells=cells, phases=phases, n_per_bucket=int(args.n_per_bucket),
                        pose_block_start=int(args.pose_block_start), pose_block_size=int(args.pose_block_size),
                        families=list(FAMILIES), k_max=args.k_max,
                        gamma=args.gamma, horizons=list(HORIZONS), env_max_steps=env_max_steps,
                        tool_cap=tool_cap, margins=margins, global_margin=global_margin,
                        margin_by_bucket={f"{c}:{p}": float(m) for (c, p), m in margin_by_bucket.items()},
                        random_mode=args.random_mode,
                        sampling={"p_ce": 0.5, "bad_fraction": 0.5, "safety_negative_quota": "separate"},
                        invalid_anchor_branches=invalid_anchor_branches,
                        fidelity_violations=fidelity_violations, godot_clamps=godot_clamps,
                        baseline_outcomes=dict(baseline_outcomes),
                        candidate_matrix={f"{b}->{c}": n for (b, c), n in candidate_matrix.items()},
                        paired_bad_by_rule=dict(paired_bad_by_rule),
                        discarded_baseline_failures=int(discarded_baseline_failures),
                        candidate_better=int(candidate_better_count),
                        splits={"pose_ids": sorted(pose_ids_seen), "seeds": sorted(seeds_seen)})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        ds.save(args.out, action_low=low, action_high=high, margin=global_margin, meta=manifest)

        print(f"\nWROTE {args.out}  transitions={len(ds)}  margin={global_margin:.4f}", flush=True)
        print("counts by source:", dict(collections.Counter(arr["source"].tolist())), flush=True)
        cand = arr["source"] == "candidate"
        burst_m = arr["source"] == "ood_burst"
        print("candidate outcomes:", dict(collections.Counter(arr["outcome"][cand].tolist())), flush=True)
        print("candidate termination_reason:", dict(collections.Counter(arr["termination_reason"][cand].tolist())), flush=True)
        print(f"invalid_branches={invalid_anchor_branches}  action_fidelity_violations(recorded==applied)="
              f"{fidelity_violations}  godot_clamps(requested!=applied, diagnostic)={godot_clamps}", flush=True)
        print("replay dones(terminated)=", int(arr["dones"].sum()), " truncated=", int(arr["truncated"].sum()), flush=True)
        print("candidate label_usable:", int(arr["label_usable"][cand].sum()),
              " paired_bad(candidate only):", int(arr["paired_bad"][cand].sum()),
              " paired_bad(all):", int(arr["paired_bad"].sum()), flush=True)
        print("safety_negative(ood_burst):", int(arr["safety_negative"][burst_m].sum()),
              " safety_negative(all):", int(arr["safety_negative"].sum()), flush=True)
        print("baseline_outcomes:", dict(baseline_outcomes), flush=True)
        print("candidate_matrix(base->cand):", {f"{b}->{c}": n for (b, c), n in sorted(candidate_matrix.items())}, flush=True)
        print("paired_bad_by_rule:", dict(paired_bad_by_rule), flush=True)
        print(f"discarded_baseline_failures={discarded_baseline_failures}  candidate_better={candidate_better_count}", flush=True)
    finally:
        if env is not None:
            env.close()
        manager.stop_all()


if __name__ == "__main__":
    main()
