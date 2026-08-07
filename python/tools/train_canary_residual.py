"""M6.1 CANARY: bounded residual actor around the FROZEN BC clone, guided by the FROZEN accepted
critic v6 (critic-3750-1). Approved spec. Offline actor updates only (no env interaction for the
gradient); in-sim PAIRED eval vs the clone drives rollback / selection / final. STOP after the
report -- even on PASS this does NOT start online SAC/TD3.

Objective (only the residual is trainable):
    action = clip(bc(obs) + 0.05*tanh(residual(obs)), -1, 1)
    L = -q_weight * mean(min(Q1,Q2)(s, action)) + 10.0 * mean(delta^2)
q_weight ramps 0 -> 0.02 linearly over 2000 actor updates. actor LR 1e-5, batch 256.
No critic update, no replay mutation, no curriculum, no auto-recovery, no alpha.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.models import build_continuous_actor, build_continuous_critic  # noqa: E402
from core.residual_actor import (  # noqa: E402
    DELTA_SCALE, ANCHOR_COEF, assert_within_bound, build_residual_actor, canary_actor_loss,
    combined_action, residual_is_zero)
from tools.eval_residual_canary import (  # noqa: E402
    CanaryEvaluator, DEFAULT_GODOT_BIN, DEFAULT_MANIFEST, bc_action_fn, build_deck,
    load_easy_cells, residual_action_fn)

EVAL_POINTS = [0, 100, 250, 500, 1000, 1500, 2000]
BC_WEIGHTS = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"
CRITIC_CKPT = "checkpoints/openarm_reach_hold_m6_1_critic_v6/critic-3750-1"


def _weights_sha256(model):
    import hashlib
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m5v2", default="python/demos/openarm_reach_hold_m5_v2/train.npz",
                    help="offline STATE buffer (on-distribution BC observations) for the Q objective.")
    ap.add_argument("--bc-weights", default=BC_WEIGHTS)
    ap.add_argument("--critic-ckpt", default=CRITIC_CKPT)
    ap.add_argument("--checkpoint-dir", default="checkpoints/openarm_reach_hold_m6_1_canary")
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=5610)
    ap.add_argument("--actor-learning-rate", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--q-weight-final", type=float, default=0.02)
    ap.add_argument("--q-weight-ramp", type=int, default=2000)
    ap.add_argument("--max-updates", type=int, default=2000)
    ap.add_argument("--anchor-coef", type=float, default=ANCHOR_COEF)
    ap.add_argument("--delta-scale", type=float, default=DELTA_SCALE)
    ap.add_argument("--selection-per-cell", type=int, default=3)
    ap.add_argument("--selection-seed-base", type=int, default=700000)
    ap.add_argument("--final-per-cell", type=int, default=22)
    ap.add_argument("--final-seed-base", type=int, default=800000)   # DISJOINT from selection
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    # rollback / selection thresholds (approved spec)
    ap.add_argument("--rollback-success-1eval", type=float, default=0.10)
    ap.add_argument("--rollback-success-2eval", type=float, default=0.05)
    ap.add_argument("--rollback-progress-drop", type=float, default=0.02)
    ap.add_argument("--final-noninferiority", type=float, default=0.05)
    # ---- v2 DOSE-ESCALATION (pre-registered). Do NOT stop at first pair; test real q_weight. ----
    ap.add_argument("--dose-escalation", action="store_true",
                    help="v2: eval 0/250/500/1000/1500/2000, candidates only >=1000, run to 2000, "
                         "pick lexicographically (success, progress, -deviation), classify.")
    ap.add_argument("--eval-points", default="0,250,500,1000,1500,2000")
    ap.add_argument("--min-candidate-update", type=int, default=1000)
    ap.add_argument("--candidate-progress-drop", type=float, default=0.002)
    ap.add_argument("--selection-cell-max-drop", type=float, default=0.13,   # ~1/8 on an 8-seed deck
                    help="max per-cell success-rate drop tolerated for a selection candidate.")
    ap.add_argument("--final-progress-drop", type=float, default=0.005)
    ap.add_argument("--final-cell-max-loss", type=int, default=2,
                    help="max episodes a single cell may lose vs the clone on the final deck.")
    ap.add_argument("--effective-progress-gain", type=float, default=0.002)
    return ap.parse_args()


def _q_weight(update, final, ramp):
    return float(final) * min(1.0, float(update) / float(ramp)) if ramp > 0 else float(final)


def _paired_deltas(clone_sum, res_sum):
    """Paired clone-vs-residual deltas by pose_id. Returns global + per-cell + new-collision list."""
    clone_by = {p["pose_id"]: p for p in clone_sum["poses"]}
    new_collisions = []          # residual collides where the clone did NOT (same pose)
    for p in res_sum["poses"]:
        c = clone_by.get(p["pose_id"])
        if c is not None and p["collided"] and not c["collided"]:
            new_collisions.append(p["pose_id"])
    per_cell = {}
    for cell, rc in res_sum["per_cell"].items():
        cc = clone_sum["per_cell"].get(cell, {})
        per_cell[cell] = {
            "clone_sr": cc.get("success_rate"), "res_sr": rc["success_rate"],
            "sr_drop": (cc.get("success_rate", 0.0) - rc["success_rate"]),
            "clone_succ": cc.get("successes", 0), "res_succ": rc["successes"],
            "episode_loss": int(cc.get("successes", 0)) - int(rc["successes"]),  # clone - residual
            "clone_coll": cc.get("collision_rate"), "res_coll": rc["collision_rate"],
        }
    return {
        "success_drop": clone_sum["success_rate"] - res_sum["success_rate"],
        "collision_increase": res_sum["collision_rate"] - clone_sum["collision_rate"],
        "progress_drop": ((clone_sum["progress_mean"] or 0.0) - (res_sum["progress_mean"] or 0.0)),
        "new_collision_poses": new_collisions,
        "per_cell": per_cell,
        "worst_cell_sr_drop": max((v["sr_drop"] for v in per_cell.values()), default=0.0),
        "worst_cell_episode_loss": max((v["episode_loss"] for v in per_cell.values()), default=0),
    }


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist = open(ckpt_dir / "canary_history.jsonl", "w")

    def log(rec):
        hist.write(json.dumps(rec, default=float) + "\n"); hist.flush()

    # ---- frozen base actor + frozen critic v6 + trainable residual ----
    bc = build_continuous_actor(obs_dim=27, action_size=7)
    bc.load_weights(args.bc_weights)
    bc.trainable = False
    c1 = build_continuous_critic(obs_dim=27, action_size=7)
    c2 = build_continuous_critic(obs_dim=27, action_size=7)
    t1 = build_continuous_critic(obs_dim=27, action_size=7)
    t2 = build_continuous_critic(obs_dim=27, action_size=7)
    tf.train.Checkpoint(critic=c1, critic2=c2, target_critic=t1, target_critic2=t2) \
        .restore(args.critic_ckpt).expect_partial()
    for m in (c1, c2, t1, t2):
        m.trainable = False
    residual = build_residual_actor(27, 7)
    opt = tf.keras.optimizers.Adam(args.actor_learning_rate)

    bc_hash0, c1_hash0, c2_hash0 = _weights_sha256(bc), _weights_sha256(c1), _weights_sha256(c2)

    obs_buf = np.asarray(np.load(args.m5v2, allow_pickle=True)["obs"], np.float32)

    def bc_fn_tf(o):
        return bc(o, training=False)

    def critic_min_q(o, a):
        return tf.minimum(c1([o, a], training=False), c2([o, a], training=False))[:, 0]

    def _hashes_ok():
        return (_weights_sha256(bc) == bc_hash0 and _weights_sha256(c1) == c1_hash0
                and _weights_sha256(c2) == c2_hash0)

    # ---- PREFLIGHT gates (before any update) ----
    pf_obs = obs_buf[rng.integers(len(obs_buf), size=512)]
    preflight = {
        "residual_zero": residual_is_zero(residual, pf_obs),
        "bc_hash_stable": _weights_sha256(bc) == bc_hash0,
        "critic_hash_stable": _weights_sha256(c1) == c1_hash0 and _weights_sha256(c2) == c2_hash0,
    }
    act0, _d0 = combined_action(bc_fn_tf, residual, pf_obs, delta_scale=args.delta_scale)
    preflight["action_equals_clone_1e6"] = bool(
        np.max(np.abs(act0.numpy() - bc(tf.convert_to_tensor(pf_obs), training=False).numpy())) < 1e-6)
    print("PREFLIGHT " + json.dumps(preflight), flush=True)
    log({"tag": "preflight", **preflight})
    if not all(preflight.values()):
        print("PREFLIGHT FAILED; abort, no updates.", flush=True)
        raise SystemExit("preflight gate failed")

    # ---- decks (selection + DISJOINT final) ----
    cells = load_easy_cells(DEFAULT_MANIFEST)
    sel_seeds = [args.selection_seed_base + i for i in range(args.selection_per_cell)]
    fin_seeds = [args.final_seed_base + i for i in range(args.final_per_cell)]
    assert set(sel_seeds).isdisjoint(fin_seeds), "selection/final seeds must be disjoint"
    sel_deck = build_deck(cells, sel_seeds)
    fin_deck = build_deck(cells, fin_seeds)
    print(f"DECKS: {len(cells)} Easy cells; selection {len(sel_deck)} poses, final {len(fin_deck)} poses",
          flush=True)

    ev = CanaryEvaluator(godot_bin=args.godot_bin, port=args.port, curriculum_level=0.0,
                         max_steps=args.max_steps)
    accepted = None
    try:
        clone_sel = ev.eval_deck(bc_action_fn(bc), sel_deck)          # clone baseline (selection)
        print(f"CLONE selection baseline: sr={clone_sel['success_rate']:.3f} "
              f"coll={clone_sel['collision_rate']:.3f} prog={clone_sel['progress_mean']}", flush=True)
        log({"tag": "clone_selection_baseline", "success_rate": clone_sel["success_rate"],
             "collision_rate": clone_sel["collision_rate"], "progress_mean": clone_sel["progress_mean"],
             "per_cell": clone_sel["per_cell"]})

        res_fn = residual_action_fn(bc, residual, delta_scale=args.delta_scale)
        eval_points = sorted({int(x) for x in args.eval_points.split(",")})

        # fixed obs sample for per-dose telemetry (critic Q clone vs residual, residual stats, saturation)
        tel_obs = tf.convert_to_tensor(obs_buf[rng.integers(len(obs_buf), size=2048)], tf.float32)
        clone_act_tel = bc(tel_obs, training=False).numpy()
        clone_q_tel = float(np.mean(critic_min_q(tel_obs, tf.convert_to_tensor(clone_act_tel)).numpy()))
        clone_sat = float(np.mean(np.abs(clone_act_tel) >= 0.999))

        def dose_telemetry(update, qw):
            delta = args.delta_scale * np.tanh(residual(tel_obs, training=False).numpy())
            res_act = np.clip(clone_act_tel + delta, -1.0, 1.0).astype(np.float32)
            dev = res_act - clone_act_tel
            q_res = float(np.mean(critic_min_q(tel_obs, tf.convert_to_tensor(res_act)).numpy()))
            return {"update": update, "q_weight": float(qw), "q_clone": clone_q_tel, "q_residual": q_res,
                    "q_gain": q_res - clone_q_tel, "residual_rms": float(np.sqrt(np.mean(dev ** 2))),
                    "residual_mean_abs": float(np.mean(np.abs(dev))), "residual_max": float(np.max(np.abs(dev))),
                    "saturation_clone": clone_sat, "saturation_residual": float(np.mean(np.abs(res_act) >= 0.999))}

        def do_eval(update, qw):
            res_sel = ev.eval_deck(res_fn, sel_deck)
            dev = res_sel["max_deviation"]
            assert dev <= args.delta_scale + 1e-6, f"deviation {dev} > bound"
            d = _paired_deltas(clone_sel, res_sel)
            wpath = str(ckpt_dir / f"residual-{update}.weights.h5")
            residual.save_weights(wpath)                             # separate ckpt per eval point
            dose = dose_telemetry(update, qw)
            log({"tag": "eval", "update": update, "q_weight": float(qw),
                 "success_rate": res_sel["success_rate"], "collision_rate": res_sel["collision_rate"],
                 "progress_mean": res_sel["progress_mean"], "max_deviation": dev, "deltas": d,
                 "dose": dose, "ckpt": wpath, "clone_sr": clone_sel["success_rate"]})
            print(f"EVAL u={update} qw={qw:.4f} res_sr={res_sel['success_rate']:.3f} "
                  f"clone_sr={clone_sel['success_rate']:.3f} succ_drop={d['success_drop']:.3f} "
                  f"prog_drop={d['progress_drop']:.4f} new_coll={len(d['new_collision_poses'])} "
                  f"worst_cell_drop={d['worst_cell_sr_drop']:.3f} dev={dev:.4f} | "
                  f"q_clone={dose['q_clone']:.3f} q_res={dose['q_residual']:.3f} gain={dose['q_gain']:.4f} "
                  f"res_rms={dose['residual_rms']:.4f} res_max={dose['residual_max']:.4f} "
                  f"sat={dose['saturation_residual']:.3f}", flush=True)
            return res_sel, d, dose

        def rollback_reason(d, dev, prev_drop):
            if d["new_collision_poses"]:
                return f"new collision where clone safe: {d['new_collision_poses'][:3]}"
            if d["success_drop"] > args.rollback_success_1eval:
                return f"success drop {d['success_drop']:.3f} > {args.rollback_success_1eval} in one eval"
            if prev_drop is not None and prev_drop > args.rollback_success_2eval \
                    and d["success_drop"] > args.rollback_success_2eval:
                return f"success drop > {args.rollback_success_2eval} for two consecutive evals"
            if d["progress_drop"] > args.rollback_progress_drop:
                return f"progress drop {d['progress_drop']:.3f} > {args.rollback_progress_drop}"
            if dev > args.delta_scale:
                return f"deviation {dev} > {args.delta_scale}"
            return None

        # eval @0 must reproduce the clone baseline exactly (delta == 0)
        res0, d0, dose0 = do_eval(0, 0.0)
        if abs(d0["success_drop"]) > 1e-9 or res0["max_deviation"] > 1e-9:
            print("PREFLIGHT eval@0 does NOT reproduce baseline; abort.", flush=True)
            raise SystemExit("eval@0 != clone baseline")

        # ---- DOSE ESCALATION: run to max_updates (unless rollback); collect ALL evals ----
        evals = {0: (res0, d0, dose0)}
        prev_drop = None
        rolled_back = rb_reason = None
        for update in range(1, args.max_updates + 1):
            idx = rng.integers(len(obs_buf), size=args.batch_size)
            batch = tf.convert_to_tensor(obs_buf[idx], tf.float32)
            qw = _q_weight(update, args.q_weight_final, args.q_weight_ramp)
            with tf.GradientTape() as g:
                loss, delta, action, tel = canary_actor_loss(
                    residual, bc_fn_tf, critic_min_q, batch, q_weight=qw,
                    anchor_coef=args.anchor_coef, delta_scale=args.delta_scale)
            opt.apply_gradients(zip(g.gradient(loss, residual.trainable_variables),
                                    residual.trainable_variables))
            bc_batch = bc(batch, training=False)
            assert_within_bound(action, bc_batch, tol=args.delta_scale + 1e-4)   # hard per-batch bound
            if not (np.isfinite(tel["q_mean"]) and _hashes_ok()):
                rolled_back, rb_reason = True, "NaN or frozen-hash violation during training"
                break
            if update in eval_points:
                res_sel, d, dose = do_eval(update, qw)
                rb = rollback_reason(d, res_sel["max_deviation"], prev_drop)
                if rb is not None:
                    rolled_back, rb_reason = True, rb
                    break
                evals[update] = (res_sel, d, dose)
                prev_drop = d["success_drop"]

        if rolled_back:
            print(f"ROLLBACK: {rb_reason}", flush=True)
            log({"tag": "rollback", "reason": rb_reason})
            residual2 = build_residual_actor(27, 7)                  # zero residual (restore clone)
            residual2.save_weights(str(ckpt_dir / "residual-rolledback-zero.weights.h5"))
            _finalize(ckpt_dir, hist, accepted=False, classification="rollback",
                      reason=f"ROLLBACK during escalation: {rb_reason} (residual zeroed, canary stopped)",
                      bc_ok=_weights_sha256(bc) == bc_hash0)
            return

        # ---- candidate selection among ELIGIBLE (update >= min_candidate_update) ----
        def meets_candidate(d):
            return (d["success_drop"] <= 0.0 and not d["new_collision_poses"]
                    and d["collision_increase"] <= 0.0
                    and d["progress_drop"] <= args.candidate_progress_drop
                    and d["worst_cell_sr_drop"] <= args.selection_cell_max_drop)

        eligible = [u for u in sorted(evals)
                    if u >= args.min_candidate_update and meets_candidate(evals[u][1])]
        print(f"ELIGIBLE candidates (>= {args.min_candidate_update}): {eligible}", flush=True)
        if not eligible:
            _finalize(ckpt_dir, hist, accepted=False, classification="rejected",
                      reason=f"no eligible dose (>= update {args.min_candidate_update}) met selection criteria",
                      bc_ok=_weights_sha256(bc) == bc_hash0, extra={"eligible": []})
            return

        # lexicographic: best success (max sr), then best progress (min progress_drop), then min deviation
        def _key(u):
            res_sel, d, _dose = evals[u]
            return (-res_sel["success_rate"], d["progress_drop"], res_sel["max_deviation"])
        cand_update = min(eligible, key=_key)
        cand_ckpt = str(ckpt_dir / f"residual-{cand_update}.weights.h5")
        print(f"SELECTION candidate = @update={cand_update} (lexicographic among {eligible})", flush=True)
        log({"tag": "selection", "candidate_update": cand_update, "eligible": eligible,
             "dose": evals[cand_update][2]})

        # ---- FINAL: one consultation on the DISJOINT final deck ----
        residual.load_weights(cand_ckpt)
        res_final_fn = residual_action_fn(bc, residual, delta_scale=args.delta_scale)
        clone_fin = ev.eval_deck(bc_action_fn(bc), fin_deck)
        res_fin = ev.eval_deck(res_final_fn, fin_deck)
        df = _paired_deltas(clone_fin, res_fin)
        hashes_ok = _hashes_ok()
        final_pass = (df["success_drop"] <= args.final_noninferiority
                      and not df["new_collision_poses"] and df["collision_increase"] <= 0.0
                      and df["progress_drop"] <= args.final_progress_drop
                      and df["worst_cell_episode_loss"] <= args.final_cell_max_loss
                      and hashes_ok)
        if not final_pass:
            classification = "rejected"
        elif df["success_drop"] < 0.0 or (-df["progress_drop"]) > args.effective_progress_gain:
            classification = "safe_effective"      # non-inferior AND improves success or progress>0.002
        else:
            classification = "safe_neutral"        # non-inferior but essentially equal to the clone
        print(f"FINAL u={cand_update} res_sr={res_fin['success_rate']:.3f} "
              f"clone_sr={clone_fin['success_rate']:.3f} succ_drop={df['success_drop']:.3f} "
              f"prog_drop={df['progress_drop']:.4f} new_coll={len(df['new_collision_poses'])} "
              f"worst_cell_loss={df['worst_cell_episode_loss']} hashes_ok={hashes_ok} "
              f"PASS={final_pass} class={classification}", flush=True)
        log({"tag": "final", "update": cand_update, "classification": classification, "pass": bool(final_pass),
             "clone": {"sr": clone_fin["success_rate"], "coll": clone_fin["collision_rate"],
                       "prog": clone_fin["progress_mean"]},
             "residual": {"sr": res_fin["success_rate"], "coll": res_fin["collision_rate"],
                          "prog": res_fin["progress_mean"]}, "deltas": df, "dose": evals[cand_update][2]})
        _finalize(ckpt_dir, hist, accepted=bool(final_pass), classification=classification,
                  reason=(f"final {classification}" if final_pass else f"final rejected: {df}"),
                  bc_ok=_weights_sha256(bc) == bc_hash0,
                  extra={"candidate_update": cand_update, "candidate_ckpt": cand_ckpt,
                         "final_deltas": df, "candidate_dose": evals[cand_update][2],
                         "safety_demonstrated": True,
                         "critic_guidance_tested": bool(evals[cand_update][2]["residual_max"] > 1e-3)})
    finally:
        ev.close()


def _finalize(ckpt_dir, hist, *, accepted, reason, bc_ok, classification=None, extra=None):
    result = {"accepted": bool(accepted and bc_ok), "classification": classification,
              "reason": reason if bc_ok else f"{reason} | BC CLONE CHANGED (must never happen)",
              "bc_unchanged": bool(bc_ok), **(extra or {})}
    (Path(ckpt_dir) / "canary_result.json").write_text(json.dumps(result, indent=2, default=float))
    hist.close()
    print("CANARY_RESULT " + json.dumps({k: result[k] for k in
          ("accepted", "classification", "reason", "bc_unchanged")}, default=str), flush=True)
    print("STOP after report. Even on safe_effective, online TD3/SAC is NOT started automatically.",
          flush=True)


if __name__ == "__main__":
    main()
