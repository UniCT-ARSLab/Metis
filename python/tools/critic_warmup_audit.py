"""Audit-only: score a warmed critic against a fixed validation batch, NO Godot, NO training.

Rebuilds the actor + twin critics, restores a critic-warmup checkpoint (actor still the frozen BC
clone), loads the validation demonstrations, verifies the integrity invariants, and runs the
strengthened q_ranking_audit (per-sample win rate, margin quantiles, per-cell worst margin). For
OpenArm the absence of per-cell ids is FAIL-CLOSED -- it must not silently fall back to the global
gate. Writes a complete JSON report. Exits non-zero if any invariant or the gate fails.

Example:
  python tools/critic_warmup_audit.py \
    --checkpoint checkpoints/openarm_reach_hold_td3bc_audit_v1/ckpt-98 \
    --val-path python/demos/openarm_reach_hold_m5_v2/val.npz \
    --clone-weights checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5 \
    --output-json checkpoints/openarm_reach_hold_td3bc_audit_v1/audit_report.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.critic_audit import format_audit, q_ranking_audit  # noqa: E402
from core.models import build_continuous_actor, build_continuous_critic  # noqa: E402


def _weights_hash(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ckpt_scalar(ckpt_path, name):
    try:
        return tf.train.load_variable(ckpt_path, name)
    except Exception:
        return None


def _scale_np(raw, low, high):
    low = low.reshape(1, -1)
    high = high.reshape(1, -1)
    return low + (np.asarray(raw, np.float64) + 1.0) * 0.5 * (high - low)


def _min_q_np(critic, critic2, obs, act):
    q1 = np.asarray(critic([obs, act.astype(np.float32)]), np.float64).reshape(-1)
    q2 = np.asarray(critic2([obs, act.astype(np.float32)]), np.float64).reshape(-1)
    return np.minimum(q1, q2)


def _saturation_rate(act_scaled, low, high, frac=0.95):
    """Fraction of action components sitting within `frac` of a limit (|a-mid| >= frac*half)."""
    low = low.reshape(1, -1)
    high = high.reshape(1, -1)
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low)
    return float(np.mean(np.abs(np.asarray(act_scaled) - mid) >= frac * half))


def run_extended_diagnosis(critic, critic2, actor_fn, collapsed_fn, obs, low, high,
                           *, pga_step=0.1, perturb_scales=(0.05, 0.1, 0.25, 0.5), seed=0):
    """Probe the critic's OOD behaviour beyond the six gate families. Returns {family: stats}.

    Families: the v4 collapsed actor's actions; single-joint saturation +/- for each joint (14);
    multi-scale Gaussian perturbations of the clone; and one projected-gradient-ascent step on the
    ACTION (start at the clone, step toward higher min(Q1,Q2), clip). For each: mean min(Q1,Q2),
    mean L2 deviation from the clone action, and saturation rate.
    """
    rng = np.random.default_rng(seed)
    obs = np.asarray(obs, np.float32)
    low = np.asarray(low, np.float64)
    high = np.asarray(high, np.float64)
    n, act_dim = obs.shape[0], low.shape[0]
    half = 0.5 * (high - low)

    clone = _scale_np(actor_fn(obs), low, high)

    families = {}
    if collapsed_fn is not None:
        families["collapsed_v4"] = _scale_np(collapsed_fn(obs), low, high)
    for j in range(act_dim):
        for sign, tag in ((1.0, "pos"), (-1.0, "neg")):
            a = clone.copy()
            a[:, j] = high[j] if sign > 0 else low[j]
            families[f"sat_j{j}_{tag}"] = a
    for s in perturb_scales:
        a = np.clip(clone + rng.normal(size=clone.shape) * float(s) * half.reshape(1, -1),
                    low.reshape(1, -1), high.reshape(1, -1))
        families[f"perturb_{s}"] = a

    # one projected-gradient-ascent step on the action, maximising min(Q1,Q2) from the clone.
    a_var = tf.Variable(clone.astype(np.float32))
    obs_t = tf.convert_to_tensor(obs, tf.float32)
    with tf.GradientTape() as tape:
        q = tf.minimum(critic([obs_t, a_var]), critic2([obs_t, a_var]))
        obj = tf.reduce_sum(q)
    grad = tape.gradient(obj, a_var).numpy()
    gnorm = np.linalg.norm(grad, axis=1, keepdims=True)
    direction = np.divide(grad, np.maximum(gnorm, 1e-12))
    pga = np.clip(clone + float(pga_step) * half.reshape(1, -1) * direction,
                  low.reshape(1, -1), high.reshape(1, -1))
    families["pga_1step"] = pga

    clone_q = float(np.mean(_min_q_np(critic, critic2, obs, clone)))
    out = {"clone": {"q_mean": clone_q, "deviation": 0.0,
                     "saturation_rate": _saturation_rate(clone, low, high)}}
    for name, act in families.items():
        q = _min_q_np(critic, critic2, obs, act)
        dev = float(np.mean(np.linalg.norm(act - clone, axis=1)))
        out[name] = {
            "q_mean": float(np.mean(q)),
            "q_vs_clone": float(np.mean(q) - clone_q),
            "deviation": dev,
            "saturation_rate": _saturation_rate(act, low, high),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="checkpoint prefix, e.g. .../ckpt-98")
    ap.add_argument("--val-path", required=True)
    ap.add_argument("--clone-weights", required=True,
                    help="actor_bc_final.weights.h5 the restored actor must equal")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--win-rate-threshold", type=float, default=0.9)
    ap.add_argument("--cell-margin-tol", type=float, default=0.0)
    ap.add_argument("--perturb-scale", type=float, default=0.1)
    ap.add_argument("--expected-critic-iterations", type=int, default=5000)
    ap.add_argument("--expected-samples", type=int, default=11952)
    ap.add_argument("--expected-cells", type=int, default=11)
    ap.add_argument("--require-cells", action=argparse.BooleanOptionalAction, default=True,
                    help="OpenArm: fail-closed if the validation batch has no per-cell ids "
                         "(no silent fallback to the global-only gate).")
    ap.add_argument("--collapsed-actor", default=None,
                    help="v4 collapsed actor weights, scored as an OOD family in the extended "
                         "diagnosis (diagnosis only -- never trained).")
    ap.add_argument("--pga-step", type=float, default=0.1,
                    help="projected-gradient-ascent step size (fraction of the action half-range).")
    args = ap.parse_args()

    # ---- validation batch -------------------------------------------------------------------
    val = np.load(args.val_path, allow_pickle=True)
    obs = np.asarray(val["obs"], dtype=np.float32)
    expert = np.asarray(val["actions"], dtype=np.float32)
    obs_dim, action_size = obs.shape[1], expert.shape[1]
    cells = np.asarray(val["cells"]) if "cells" in val.files else None
    action_low = (np.asarray(val["action_low"], dtype=np.float32) if "action_low" in val.files
                  else -np.ones(action_size, np.float32))
    action_high = (np.asarray(val["action_high"], dtype=np.float32) if "action_high" in val.files
                   else np.ones(action_size, np.float32))

    # ---- rebuild + restore (weights via checkpoint; scalars read directly) ------------------
    actor = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
    critic = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
    critic2 = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
    ckpt = tf.train.Checkpoint(actor=actor, critic=critic, critic2=critic2)
    ckpt.restore(args.checkpoint).expect_partial()

    crit_iter = _ckpt_scalar(args.checkpoint, "critic_optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE")
    crit2_iter = _ckpt_scalar(args.checkpoint, "critic2_optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE")
    actor_iter = _ckpt_scalar(args.checkpoint, "actor_optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE")
    psw = _ckpt_scalar(args.checkpoint, "policy_updates_since_warmup/.ATTRIBUTES/VARIABLE_VALUE")
    crit_iter = int(crit_iter) if crit_iter is not None else -1
    psw = int(psw) if psw is not None else -1

    clone = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
    clone.load_weights(args.clone_weights)
    actor_hash, clone_hash = _weights_hash(actor), _weights_hash(clone)

    n_samples = int(len(obs))
    cells_available = cells is not None
    distinct_cells = int(len(np.unique(cells))) if cells_available else 0

    checks = {
        "critic_iterations": crit_iter,
        "critic2_iterations": (int(crit2_iter) if crit2_iter is not None else -1),
        "actor_optimizer_iterations": (int(actor_iter) if actor_iter is not None else -1),
        "critic_iterations_ok": crit_iter == args.expected_critic_iterations,
        "policy_updates_since_warmup": psw,
        "psw_ok": psw == 0,
        "actor_hash": actor_hash,
        "clone_hash": clone_hash,
        "actor_matches_clone": actor_hash == clone_hash,
        "validation_samples": n_samples,
        "samples_ok": n_samples == args.expected_samples,
        "cells_available": cells_available,
        "distinct_cells": distinct_cells,
        "cells_ok": distinct_cells == args.expected_cells,
    }
    print("CRITIC_WARMUP_AUDIT_CHECKS", flush=True)
    print(f"  critic_optimizer.iterations = {crit_iter} "
          f"(expect {args.expected_critic_iterations}) -> {'OK' if checks['critic_iterations_ok'] else 'FAIL'}")
    print(f"  critic2.iterations={checks['critic2_iterations']}  "
          f"actor_optimizer.iterations={checks['actor_optimizer_iterations']} (expect 0: actor frozen)")
    print(f"  policy_updates_since_warmup = {psw} (expect 0) -> {'OK' if checks['psw_ok'] else 'FAIL'}")
    print(f"  actor_hash   = {actor_hash}")
    print(f"  clone_hash   = {clone_hash}")
    print(f"  actor == actor_bc_final -> {'OK' if checks['actor_matches_clone'] else 'FAIL'}")
    print(f"  validation_samples = {n_samples} (expect {args.expected_samples}) -> {'OK' if checks['samples_ok'] else 'FAIL'}")
    print(f"  cells_available = {cells_available}  distinct_cells = {distinct_cells} "
          f"(expect {args.expected_cells}) -> {'OK' if checks['cells_ok'] else 'FAIL'}")

    # ---- fail-closed preconditions ----------------------------------------------------------
    hard_failures = []
    if args.require_cells and not cells_available:
        hard_failures.append("no per-cell ids in the validation batch (OpenArm requires cells; "
                             "fail-closed, no global-only fallback)")
    if not checks["critic_iterations_ok"]:
        hard_failures.append(f"critic iterations {crit_iter} != {args.expected_critic_iterations}")
    if not checks["psw_ok"]:
        hard_failures.append(f"policy_updates_since_warmup {psw} != 0")
    if not checks["actor_matches_clone"]:
        hard_failures.append("restored actor does not match actor_bc_final.weights.h5")
    if not checks["samples_ok"]:
        hard_failures.append(f"validation samples {n_samples} != {args.expected_samples}")
    if not checks["cells_ok"]:
        hard_failures.append(f"distinct cells {distinct_cells} != {args.expected_cells}")

    # ---- audit ------------------------------------------------------------------------------
    def _actor_fn(o):
        return actor(o, training=False).numpy()

    def _critic_fn(oa):
        return critic(oa, training=False).numpy()

    def _critic2_fn(oa):
        return critic2(oa, training=False).numpy()

    report = q_ranking_audit(
        _critic_fn, _actor_fn, obs, expert, action_low, action_high,
        critic2=_critic2_fn, group_ids=cells, perturb_scale=args.perturb_scale, seed=args.seed,
        win_rate_threshold=args.win_rate_threshold, cell_margin_tol=args.cell_margin_tol,
    )
    print(format_audit(report), flush=True)

    # ---- extended OOD diagnosis (not part of the gate) --------------------------------------
    collapsed_fn = None
    if args.collapsed_actor:
        collapsed = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
        collapsed.load_weights(args.collapsed_actor)

        def collapsed_fn(o):
            return collapsed(o, training=False).numpy()

    diagnosis = run_extended_diagnosis(
        critic, critic2, _actor_fn, collapsed_fn, obs, action_low, action_high,
        pga_step=args.pga_step, seed=args.seed,
    )
    print("CRITIC_OOD_DIAGNOSIS (min(Q1,Q2) | q_vs_clone | deviation | saturation_rate):", flush=True)
    clone_q = diagnosis["clone"]["q_mean"]
    print(f"  clone                {clone_q:+.4f} |   0.0000 | 0.000 | sat={diagnosis['clone']['saturation_rate']:.3f}")
    for name in sorted(diagnosis):
        if name == "clone":
            continue
        d = diagnosis[name]
        flag = "  <-- Q>=clone" if d["q_vs_clone"] >= 0 else ""
        print(f"  {name:<20} {d['q_mean']:+.4f} | {d['q_vs_clone']:+.4f} | "
              f"{d['deviation']:.3f} | sat={d['saturation_rate']:.3f}{flag}")

    pairs_out = {}
    for key, p in report["pairs"].items():
        pairs_out[key] = {
            "win_rate": p["win_rate"], "mean": p["mean"], "median": p["median"],
            "p10": p["p10"], "p90": p["p90"],
            "worst_cell": p["worst_cell"], "worst_cell_margin": p["worst_cell_margin"],
            "n_cells": p["n_cells"], "n_cells_negative": p["n_cells_negative"],
        }
    out = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "validation_path": args.val_path,
        "validation_checksum_sha256": _file_sha256(args.val_path),
        "validation_samples": n_samples,
        "distinct_cells": distinct_cells,
        "checks": checks,
        "hard_failures": hard_failures,
        "q_mean": report["q_mean"],
        "pairs": pairs_out,
        "diagnostic": {k: {kk: vv for kk, vv in v.items() if kk != "cell_means"}
                       for k, v in report["diagnostic"].items()},
        "gate": report["gate"],
        "min_win_rate": report["gate"]["min_win_rate"],
        "worst_cell_margin": report["gate"]["worst_cell_margin"],
        "gate_passed_raw": report["gate_passed"],
        "gate_passed": bool(report["gate_passed"] and not hard_failures),
        "extended_diagnosis": diagnosis,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote audit JSON: {args.output_json}", flush=True)

    # ---- verdict ----------------------------------------------------------------------------
    if hard_failures:
        print("CRITIC_WARMUP_AUDIT VERDICT: FAIL-CLOSED (precondition failures):", flush=True)
        for f in hard_failures:
            print(f"  - {f}", flush=True)
        raise SystemExit(2)

    if not report["gate_passed"]:
        print("CRITIC_WARMUP_AUDIT VERDICT: GATE FAILED -- NO actor update.", flush=True)
        bad_pairs = [k for k, p in report["pairs"].items() if p["win_rate"] < args.win_rate_threshold]
        neg_pairs = [(k, p["worst_cell"], p["worst_cell_margin"]) for k, p in report["pairs"].items()
                     if p["worst_cell_margin"] is not None and p["worst_cell_margin"] < -args.cell_margin_tol]
        if bad_pairs:
            print(f"  low-win-rate pairs (< {args.win_rate_threshold}): {bad_pairs}", flush=True)
        if neg_pairs:
            print("  pairs with a clearly-negative cell (pair, cell, mean_margin):", flush=True)
            for k, c, m in neg_pairs:
                print(f"    {k}: cell {c} margin {m:+.4f}", flush=True)
        raise SystemExit(3)

    print("CRITIC_WARMUP_AUDIT VERDICT: GATE PASSED (strengthened). Actor updates would be allowed.",
          flush=True)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
