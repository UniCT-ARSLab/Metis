"""M6.1 critic-only OFFLINE training. Actor = frozen BC clone (never updated, no alpha, no Q-filter,
no CQL). Fresh twin critics + fresh optimizer. Fixed offline replay = M5 v2 + counterexample TRAIN
(stratified sampler). Standard TD3 off-policy target with the frozen BC as the next-action policy:

    y = r + gamma * (1 - dones) * min(Q1_t(s', BC(s')), Q2_t(s', BC(s')))      dones = terminated

Audit the SELECTION split before update 0 and every 1000 critic updates (actor always frozen). Keep
the FIRST checkpoint of two CONSECUTIVE passing selection audits; then consult FINAL_TEST exactly
once. No actor update ever. STOP at the critic gate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.counterexample_audit import audit_passes, counterexample_audit  # noqa: E402
from core.counterexample_sampler import CounterexampleSampler  # noqa: E402
from core.models import build_continuous_actor, build_continuous_critic  # noqa: E402
from core.pair_sampler import PairSampler  # noqa: E402


def _scale(raw, low, high):
    return tf.clip_by_value(low + 0.5 * (raw + 1.0) * (high - low), low, high)


def lambda_pair_at(update, final, ramp_updates):
    """Linear ramp 0 -> `final` across the first `ramp_updates` v3 updates; constant `final` after.
    Returns `final` immediately when the pair loss is disabled or no ramp is requested."""
    if final <= 0.0 or ramp_updates <= 0:
        return float(final)
    return float(final) * min(1.0, float(update) / float(ramp_updates))


def pair_loss_terms(critic, obs, act_cand, act_base, return_delta, huber):
    """Per-critic VALUE pairwise supervision (v3). delta_Q = Q(s, a_candidate) - Q(s, a_baseline) at
    the SAME state s; regress it toward the MEASURED return_delta with a Huber. Independent per critic
    (each uses its OWN Q). No actor, no target critic, no min() -- this only trains the online critic
    it is given. Returns (loss, delta_q). SUPERSEDED by sign_hinge_terms for v4 (value regression
    inflated Q and degraded worst-cell accuracy)."""
    qc = tf.reshape(critic([obs, act_cand], training=True), [-1])
    qb = tf.reshape(critic([obs, act_base], training=True), [-1])
    dq = qc - qb
    tgt = tf.reshape(tf.cast(return_delta, tf.float32), [-1])
    loss = tf.reduce_mean(huber(tgt, dq))
    return loss, dq


def sign_hinge_terms(critic, obs, act_cand, act_base, y_sign, q_margin):
    """Per-critic SIGN-HINGE supervision (v4). Only the ORDER matters, not the magnitude:
        delta_Q = Q(s, a_candidate) - Q(s, a_baseline)
        L_sign  = mean(relu(q_margin - y_sign * delta_Q))     y_sign = sign(return_delta_episode)
    Pushes y_sign*delta_Q above +q_margin (correct order with a buffer) WITHOUT anchoring delta_Q to
    the (large, noisy) return magnitude -- avoids the Q inflation the value loss caused. No actor, no
    target, no min(). Returns (loss, delta_q, signed_margin=y_sign*delta_Q)."""
    qc = tf.reshape(critic([obs, act_cand], training=True), [-1])
    qb = tf.reshape(critic([obs, act_base], training=True), [-1])
    dq = qc - qb
    ys = tf.reshape(tf.cast(y_sign, tf.float32), [-1])
    signed_margin = ys * dq
    loss = tf.reduce_mean(tf.nn.relu(float(q_margin) - signed_margin))
    return loss, dq, signed_margin


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m5v2", required=True)
    ap.add_argument("--counterexample-train", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--final-test", required=True)
    ap.add_argument("--clone-weights", required=True, help="frozen BC actor (actor_bc_final).")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--frozen-manifest", required=True,
                    help="FROZEN.json; train/selection/final_test/report.json checksums verified.")
    ap.add_argument("--m5v2-sha256", required=True)
    ap.add_argument("--clone-sha256", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--critic-learning-rate", type=float, default=3e-5)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--critic-loss", choices=["mse", "huber"], default="huber")
    ap.add_argument("--huber-delta", type=float, default=5.0)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    # M6.1 target is EXACTLY min(Qt1,Qt2)(s',BC(s')) -- NO TD3 target-policy smoothing. Both must be 0.
    ap.add_argument("--td3-target-policy-noise", type=float, default=0.0)
    ap.add_argument("--td3-target-noise-clip", type=float, default=0.0)
    ap.add_argument("--audit-every", type=int, default=250)
    ap.add_argument("--max-updates", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    # ---- v3 controlled fine-tuning: warm-start + pairwise critic supervision ----
    ap.add_argument("--warm-start-checkpoint", default=None,
                    help="tf.train.Checkpoint prefix to restore critic1/critic2/target1/target2 from "
                         "(v3). Optimizers are ALWAYS fresh. Omit for a cold start (v1/v2 behaviour).")
    ap.add_argument("--pair-loss-mode", choices=["none", "value", "sign"], default="none",
                    help="none=v1/v2; value=v3 (Huber(delta_Q, return_delta), SUPERSEDED); "
                         "sign=v4 (sign-hinge on delta_Q order, no magnitude anchor).")
    ap.add_argument("--lambda-pair-final", type=float, default=0.0,
                    help="value mode: final weight of the VALUE pair loss added to L_TD.")
    ap.add_argument("--lambda-sign-final", type=float, default=0.0,
                    help="sign mode: final weight of the SIGN-HINGE loss added to L_TD.")
    ap.add_argument("--sign-margin", type=float, default=0.25,
                    help="sign mode: hinge margin q_margin in relu(q_margin - y_sign*delta_Q).")
    ap.add_argument("--lambda-pair-ramp-updates", type=int, default=500,
                    help="linear ramp 0 -> final lambda across the first N updates (value & sign).")
    ap.add_argument("--pair-batch-size", type=int, default=128)
    ap.add_argument("--pair-population", choices=["significant", "paired_bad"], default="significant",
                    help="significant=v3/v4 (usable & |ret|>=margin, 50/50 sign); "
                         "paired_bad=v5 SAFETY (hard-negatives only; attack-family balanced).")
    ap.add_argument("--attack-fraction", type=float, default=0.5,
                    help="paired_bad population: batch fraction reserved for pga+collapsed_v4.")
    ap.add_argument("--audit-profile", choices=["full", "safety"], default="full",
                    help="full=v1-v4 gate set; safety=v5 (negative-sign gate, no slope/calibration/"
                         "train_sign; candidate_better is telemetry only).")
    args = ap.parse_args()
    if float(args.td3_target_policy_noise) != 0.0 or float(args.td3_target_noise_clip) != 0.0:
        ap.error("M6.1 critic-only forbids target-policy smoothing: "
                 "--td3-target-policy-noise and --td3-target-noise-clip must both be 0.0")
    return args


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_sha256(model):
    import hashlib
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()


def _verify_checksums(args):
    manifest = json.load(open(args.frozen_manifest))
    base = Path(args.frozen_manifest).parent
    want = {"train.npz": args.counterexample_train, "selection.npz": args.selection,
            "final_test.npz": args.final_test, "report.json": str(base / "report.json")}
    for name, expected in manifest["files"].items():
        path = want.get(name, str(base / name))
        got = _sha256(path)
        if got != expected["sha256"]:
            raise SystemExit(f"FROZEN checksum mismatch for {name}: {got} != {expected['sha256']}")
    for label, path, exp in (("m5v2", args.m5v2, args.m5v2_sha256),
                             ("clone", args.clone_weights, args.clone_sha256)):
        got = _sha256(path)
        if got != exp:
            raise SystemExit(f"{label} checksum mismatch: {got} != {exp}")
    print("CHECKSUMS OK: train/selection/final_test/report.json + m5v2 + clone verified.", flush=True)


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _verify_checksums(args)

    sel = dict(np.load(args.selection, allow_pickle=True))
    fin = dict(np.load(args.final_test, allow_pickle=True))
    obs_dim = sel["obs"].shape[1]
    act_dim = sel["actions"].shape[1]
    low = tf.constant(np.asarray(sel["action_low"], np.float32).reshape(1, -1))
    high = tf.constant(np.asarray(sel["action_high"], np.float32).reshape(1, -1))

    # frozen BC actor
    actor = build_continuous_actor(obs_dim=obs_dim, action_size=act_dim)
    actor.load_weights(args.clone_weights)
    actor.trainable = False
    actor_hash_before = _weights_sha256(actor)
    # twin critics + targets. v1/v2: fresh (cold start). v3: warm-start from a prior checkpoint.
    c1 = build_continuous_critic(obs_dim=obs_dim, action_size=act_dim)
    c2 = build_continuous_critic(obs_dim=obs_dim, action_size=act_dim)
    t1 = build_continuous_critic(obs_dim=obs_dim, action_size=act_dim)
    t2 = build_continuous_critic(obs_dim=obs_dim, action_size=act_dim)
    if args.warm_start_checkpoint:
        # restore ALL FOUR (critic1/critic2/target1/target2) exactly as saved; optimizers stay fresh.
        tf.train.Checkpoint(critic=c1, critic2=c2, target_critic=t1, target_critic2=t2) \
            .restore(args.warm_start_checkpoint).expect_partial()
        print(f"WARM-START restored critics+targets from {args.warm_start_checkpoint} "
              "(optimizers FRESH)", flush=True)
    else:
        t1.set_weights(c1.get_weights())
        t2.set_weights(c2.get_weights())
    opt1 = tf.keras.optimizers.Adam(args.critic_learning_rate)   # FRESH optimizer (always)
    opt2 = tf.keras.optimizers.Adam(args.critic_learning_rate)

    manifest = json.load(open(args.frozen_manifest))
    train_sha = manifest["files"]["train.npz"]["sha256"]
    sampler = CounterexampleSampler(
        args.m5v2, args.counterexample_train,
        eval_paths=[args.selection, args.final_test], seed=args.seed,
        expected_train_sha256=train_sha)
    # v3/v4 pairwise supervision source: counterexample TRAIN ONLY (selection/final_test barred).
    pair_sampler = None
    lambda_final = {"value": args.lambda_pair_final, "sign": args.lambda_sign_final,
                    "none": 0.0}[args.pair_loss_mode]
    if args.pair_loss_mode != "none":
        pair_sampler = PairSampler(
            args.counterexample_train, eval_paths=[args.selection, args.final_test],
            seed=args.seed, expected_train_sha256=train_sha,
            population=args.pair_population, attack_fraction=args.attack_fraction)
        extra = f"sign_margin={args.sign_margin}" if args.pair_loss_mode == "sign" else ""
        if args.pair_population == "paired_bad":
            extra += f" attack_fraction={args.attack_fraction}"
        print(f"PAIR SAMPLER ({args.pair_loss_mode}/{args.pair_population}): {pair_sampler.n_pairs()} "
              f"pairs (pos={pair_sampler.n_pos()} neg={pair_sampler.n_neg()}, margin={pair_sampler.margin}); "
              f"lambda_final={lambda_final} ramp={args.lambda_pair_ramp_updates} "
              f"pair_batch={args.pair_batch_size} profile={args.audit_profile} {extra}", flush=True)

    def bc_action(obs):
        # EXACT target policy: the frozen BC clone, NO TD3 target-policy smoothing.
        return _scale(actor(obs, training=False), low, high)

    def critic_min_q(obs, act):
        obs = tf.convert_to_tensor(obs, tf.float32)
        act = tf.convert_to_tensor(act, tf.float32)
        return tf.minimum(c1([obs, act], training=False), c2([obs, act], training=False)).numpy().reshape(-1)

    def target_min_q(obs, act):
        obs = tf.convert_to_tensor(obs, tf.float32)
        act = tf.convert_to_tensor(act, tf.float32)
        return tf.minimum(t1([obs, act], training=False), t2([obs, act], training=False)).numpy().reshape(-1)

    def q1_np(obs, act):
        return c1([tf.convert_to_tensor(obs, tf.float32), tf.convert_to_tensor(act, tf.float32)],
                  training=False).numpy().reshape(-1)

    def q2_np(obs, act):
        return c2([tf.convert_to_tensor(obs, tf.float32), tf.convert_to_tensor(act, tf.float32)],
                  training=False).numpy().reshape(-1)

    def bc_next_action_np(nobs):
        return bc_action(tf.convert_to_tensor(nobs, tf.float32)).numpy()

    _huber = tf.keras.losses.Huber(delta=float(args.huber_delta),
                                   reduction=tf.keras.losses.Reduction.NONE)

    def _critic_loss(pred, y):
        if args.critic_loss == "huber":
            return tf.reduce_mean(_huber(y, pred))
        return tf.reduce_mean(tf.square(pred - y))

    def _value_stats(dq, pret_np):
        dqn = dq.numpy().reshape(-1)
        return {"mae": float(np.mean(np.abs(dqn - pret_np))),
                "pred_mean": float(np.mean(dqn)), "tgt_mean": float(np.mean(pret_np)),
                "sign_acc": float(np.mean(np.sign(dqn) == np.sign(pret_np)))}

    def _sign_stats(dq, sm, ys_np):
        dqn = dq.numpy().reshape(-1); smn = sm.numpy().reshape(-1)
        return {"sign_acc": float(np.mean(np.sign(dqn) == ys_np)),
                "hinge_active_frac": float(np.mean((float(args.sign_margin) - smn) > 0.0)),
                "signed_margin_mean": float(np.mean(smn))}

    def critic_step(batch, pair_batch=None, lambda_pair=0.0):
        obs = tf.convert_to_tensor(batch["obs"], tf.float32)
        act = tf.convert_to_tensor(batch["actions"], tf.float32)
        rew = tf.convert_to_tensor(batch["rewards"].reshape(-1, 1), tf.float32)
        nobs = tf.convert_to_tensor(batch["next_obs"], tf.float32)
        dones = tf.convert_to_tensor(batch["dones"].reshape(-1, 1), tf.float32)
        na = bc_action(nobs)                                     # frozen BC, NO smoothing
        qt = tf.minimum(t1([nobs, na], training=False), t2([nobs, na], training=False))
        y = rew + args.gamma * (1.0 - dones) * qt                # dones = terminated (Metis)
        use_pair = pair_batch is not None and lambda_pair > 0.0
        if pair_batch is not None:
            p_obs = tf.convert_to_tensor(pair_batch["obs"], tf.float32)
            p_ac = tf.convert_to_tensor(pair_batch["act_cand"], tf.float32)
            p_ab = tf.convert_to_tensor(pair_batch["act_base"], tf.float32)
            p_ret_np = np.asarray(pair_batch["return_delta"], np.float64)
            p_ysign_np = np.sign(p_ret_np)
            p_ysign = tf.convert_to_tensor(p_ysign_np, tf.float32)

        def _one(critic, opt):
            with tf.GradientTape() as g:
                td = _critic_loss(critic([obs, act], training=True), y)
                dq = sm = None
                pl = tf.constant(0.0)
                if use_pair and args.pair_loss_mode == "value":
                    pl, dq = pair_loss_terms(critic, p_obs, p_ac, p_ab,
                                             pair_batch["return_delta"], _huber)
                elif use_pair and args.pair_loss_mode == "sign":
                    pl, dq, sm = sign_hinge_terms(critic, p_obs, p_ac, p_ab, p_ysign, args.sign_margin)
                loss = td + lambda_pair * pl if use_pair else td
            gr, gn = tf.clip_by_global_norm(g.gradient(loss, critic.trainable_variables),
                                            args.grad_clip_norm)
            opt.apply_gradients(zip(gr, critic.trainable_variables))
            return float(td.numpy()), float(pl.numpy()), dq, sm, float(gn.numpy())

        td1, pl1, dq1, sm1, gn1 = _one(c1, opt1)
        td2, pl2, dq2, sm2, gn2 = _one(c2, opt2)
        for tv, sv in ((t1, c1), (t2, c2)):                      # soft-update AFTER both critic steps
            for wt, ws in zip(tv.weights, sv.weights):
                wt.assign(args.tau * ws + (1.0 - args.tau) * wt)
        tel = {"td_loss": [td1, td2], "pair_loss": [pl1, pl2], "grad_norm": [gn1, gn2],
               "lambda_pair": float(lambda_pair), "mode": args.pair_loss_mode}
        if use_pair and args.pair_loss_mode == "value":
            tel["pair1"] = _value_stats(dq1, p_ret_np)
            tel["pair2"] = _value_stats(dq2, p_ret_np)
        elif use_pair and args.pair_loss_mode == "sign":
            tel["pair1"] = _sign_stats(dq1, sm1, p_ysign_np)
            tel["pair2"] = _sign_stats(dq2, sm2, p_ysign_np)
        return tel

    def save_critic(tag):
        p = str(ckpt_dir / f"critic-{tag}")
        # .save() APPENDS a numeric suffix (e.g. "-1"); return the ACTUAL prefix so restore() works.
        return tf.train.Checkpoint(critic=c1, critic2=c2, target_critic=t1, target_critic2=t2).save(p)

    def _audit(split):
        return counterexample_audit(split, critic_min_q, target_critic_min_q=target_min_q,
                                    bc_next_action=bc_next_action_np, gamma=args.gamma,
                                    margin=float(split["margin"][0]), q1=q1_np, q2=q2_np,
                                    huber_delta=args.huber_delta)

    def train_sign_report():
        """Stable train-side sign accuracy over the FULL pair set (min-Q delta_Q, mirroring the
        selection audit). Splits by family / sign / (cell,phase); + hinge-active + signed-margin."""
        ap = pair_sampler.all_pairs()
        dq = critic_min_q(ap["obs"], ap["act_cand"]) - critic_min_q(ap["obs"], ap["act_base"])
        ys = np.sign(ap["return_delta"])
        match = np.sign(dq) == ys
        fam, cp = ap["family"], ap["cellphase"]

        def acc(mask):
            return float(np.mean(match[mask])) if mask.any() else float("nan")
        overall = float(np.mean(match))
        sm = ys * dq
        return {
            "train_sign_accuracy": overall,
            "train_sign_pga": acc(fam == "pga"),
            "train_sign_collapsed": acc(fam == "collapsed_v4"),
            "train_sign_pos": acc(ys > 0), "train_sign_neg": acc(ys < 0),
            "train_sign_by_cellphase": {str(k): acc(cp == k) for k in np.unique(cp)},
            "train_hinge_active_frac": float(np.mean((float(args.sign_margin) - sm) > 0.0)),
            "train_signed_margin_mean": float(np.mean(sm)),
        }

    _KEYS = ("ranking_accuracy", "worst_cell_accuracy", "worst_cell_margin",
             "sign_agreement_pga_collapsed", "sign_pos", "sign_neg", "sign_pga", "sign_collapsed",
             "negative_sign_agreement", "candidate_better_accuracy", "n_neg_sign", "n_candidate_better",
             "neg_sign_pga", "neg_sign_collapsed", "negative_sign_ci", "candidate_better_ci",
             "spearman_deltaQ_returndelta", "delta_q_slope",
             "baseline_calibration_spearman", "td_nrmse", "td_nrmse_nonterminal",
             "td_nrmse_paired_bad", "td_nrmse_candidate_nonbad", "td_nrmse_pga_collapsed",
             "huber_residual_normalized", "n_terminal", "terminal_bad_count",
             "terminal_bad_ranking_accuracy", "terminal_worst_q_margin",
             "terminal_target_mae", "terminal_target_rmse", "q_p99", "g_p99",
             "clone_q_mean", "td_error", "q1_mean", "q2_mean", "n_sign_population", "finite")

    hist = open(ckpt_dir / "audit_history.jsonl", "w")

    def _log_audit(tag, update, m, *, passed=None, reasons=None, comp=None, tel=None, pair_comp=None,
                   train_sign=None):
        rec = {"tag": tag, "update": update, "passed": passed, "reasons": reasons,
               "composition": comp, "train_telemetry": tel, "pair_composition": pair_comp,
               "train_sign": train_sign, "sign_by_cellphase": m.get("sign_by_cellphase"),
               **{k: m[k] for k in _KEYS}, "q_bad_mean": m["q_bad_mean"], "q_nonbad_mean": m["q_nonbad_mean"]}
        hist.write(json.dumps(rec) + "\n"); hist.flush()
        return rec

    def _finalize(accepted, reason, *, sel_update=None, pass_metrics=None, final_metrics=None,
                  accepted_critics=None):
        actor_hash_after = _weights_sha256(actor)
        actor_ok = (actor_hash_after == actor_hash_before)
        result = {
            "accepted": bool(accepted and actor_ok),
            "reason": reason if actor_ok else f"{reason} | ACTOR CHANGED (must never happen)",
            "config": {k: getattr(args, k) for k in vars(args)},
            "checksums": {"train": manifest["files"]["train.npz"]["sha256"],
                          "selection": manifest["files"]["selection.npz"]["sha256"],
                          "final_test": manifest["files"]["final_test.npz"]["sha256"],
                          "m5v2": args.m5v2_sha256, "clone": args.clone_sha256},
            "seed": args.seed, "selected_update": sel_update,
            "selection_pass_metrics": pass_metrics, "final_test_metrics": final_metrics,
            "actor_hash_before": actor_hash_before, "actor_hash_after": actor_hash_after,
            "accepted_critics": accepted_critics,
        }
        (ckpt_dir / "critic_selection.json").write_text(json.dumps(result, indent=2, default=float))
        hist.close()
        print("CRITIC_SELECTION " + json.dumps({k: result[k] for k in
              ("accepted", "reason", "selected_update", "actor_hash_after")}, default=str), flush=True)
        if not actor_ok:
            raise SystemExit(f"ACTOR CHANGED during critic-only training: "
                             f"{actor_hash_before} -> {actor_hash_after}")
        return result

    # ---- baseline audit (update 0) ----
    m0 = _audit(sel)
    _log_audit("baseline", 0, m0)
    print("AUDIT update=0 (baseline) " + json.dumps({k: m0[k] for k in _KEYS}, default=float), flush=True)

    prev_pass_update = prev_pass_ckpt = prev_pass_metrics = None
    selected_update = selected_ckpt = pass_metrics = None
    # coverage-gap diagnostic: train_sign learned (>=0.90) but selection sign lags (<0.85) for N in a row
    COV_GAP_TRAIN, COV_GAP_SEL, COV_GAP_N = 0.90, 0.85, 3
    cov_gap_streak = 0

    for update in range(1, args.max_updates + 1):
        batch, comp = sampler.sample(args.batch_size)
        lam = lambda_pair_at(update, lambda_final, args.lambda_pair_ramp_updates)
        pair_batch = pair_comp = None
        if pair_sampler is not None:
            pair_batch, pair_comp = pair_sampler.sample(args.pair_batch_size)
        tel = critic_step(batch, pair_batch, lam)
        if update % args.audit_every == 0:
            m = _audit(sel)
            train_sign = train_sign_report() if pair_sampler is not None else None
            # train_sign is a GATE only under the "full" profile (v4); under "safety" it is telemetry.
            if train_sign is not None and args.audit_profile == "full":
                m["train_sign_accuracy"] = train_sign["train_sign_accuracy"]
            passed, reasons = audit_passes(m, profile=args.audit_profile)
            cur_ckpt = save_critic(update)                       # checkpoint EVERY audit (even fail)
            _log_audit("selection", update, m, passed=passed, reasons=reasons, comp=comp,
                       tel=tel, pair_comp=pair_comp, train_sign=train_sign)
            print(f"AUDIT update={update} passed={passed} " + json.dumps({
                "mode": tel["mode"], "profile": args.audit_profile,
                "td_loss": [round(x, 4) for x in tel["td_loss"]],
                "pair_loss": [round(x, 4) for x in tel["pair_loss"]], "lambda": round(lam, 4),
                "grad_norm": [round(x, 3) for x in tel["grad_norm"]],
                "train_sign_acc": round(train_sign["train_sign_accuracy"], 4) if train_sign else None,
                "hinge_active": round(train_sign["train_hinge_active_frac"], 3) if train_sign else None,
                "GATE_neg_sign": round(m["negative_sign_agreement"], 4), "n_neg": m["n_neg_sign"],
                "neg_sign_pga": round(m["neg_sign_pga"], 3) if not np.isnan(m["neg_sign_pga"]) else None,
                "neg_sign_collapsed": round(m["neg_sign_collapsed"], 3) if not np.isnan(m["neg_sign_collapsed"]) else None,
                "cand_better_acc": round(m["candidate_better_accuracy"], 3)
                if not np.isnan(m["candidate_better_accuracy"]) else None,
                "cand_better_ci": [round(x, 3) for x in m["candidate_better_ci"]], "n_cand_better": m["n_candidate_better"],
                "sel_sign_all_inclPos": round(m["sign_agreement_pga_collapsed"], 4),
                "ranking_accuracy": round(m["ranking_accuracy"], 4),
                "worst_cell_accuracy": round(m["worst_cell_accuracy"], 4),
                "worst_cell_margin": round(m["worst_cell_margin"], 4),
                "spearman": round(m["spearman_deltaQ_returndelta"], 4),
                "GATE_td_nrmse_nonterm": round(m["td_nrmse_nonterminal"], 4),
                "td_nrmse_all": round(m["td_nrmse"], 4),
                "td_nrmse_pga_coll": round(m["td_nrmse_pga_collapsed"], 4),
                "huber_resid_norm": round(m["huber_residual_normalized"], 4),
                "n_term": m["n_terminal"], "term_rank_acc": m["terminal_bad_ranking_accuracy"],
                "term_worst_margin": round(m["terminal_worst_q_margin"], 3)
                if not np.isnan(m["terminal_worst_q_margin"]) else None,
                "term_tgt_rmse": round(m["terminal_target_rmse"], 2)
                if not np.isnan(m["terminal_target_rmse"]) else None,
                "q_p99": round(m["q_p99"], 2),
                "g_p99": round(m["g_p99"], 2), "q1": round(m["q1_mean"], 3), "q2": round(m["q2_mean"], 3),
                "reasons": reasons}), flush=True)
            # coverage-gap early stop: train sign learned but the GATED selection sign lags.
            if train_sign is not None:
                ts = train_sign["train_sign_accuracy"]
                ss = m["negative_sign_agreement"] if args.audit_profile == "safety" \
                    else m["sign_agreement_pga_collapsed"]
                if ts >= COV_GAP_TRAIN and (np.isnan(ss) or ss < COV_GAP_SEL):
                    cov_gap_streak += 1
                else:
                    cov_gap_streak = 0
                if cov_gap_streak >= COV_GAP_N:
                    _finalize(False, f"COVERAGE GAP: train_sign_accuracy>={COV_GAP_TRAIN} but "
                              f"selection sign<{COV_GAP_SEL} for {COV_GAP_N} consecutive audits "
                              "(NOT underfitting -> next step: generate counterexample-train v2 with "
                              "more poses per bucket; keep selection+final_test FROZEN)")
                    raise SystemExit("COVERAGE GAP: critic NOT selected; actor stays LOCKED. "
                                     "Regenerate counterexample-train (more poses/bucket).")
            if passed:
                if prev_pass_update is not None:
                    selected_update, selected_ckpt = prev_pass_update, prev_pass_ckpt
                    pass_metrics = {"first": prev_pass_metrics, "second": m}
                    print(f"SELECTED critic = first of consecutive pair @update={selected_update}", flush=True)
                    break
                prev_pass_update, prev_pass_ckpt, prev_pass_metrics = update, cur_ckpt, m
            else:
                prev_pass_update = prev_pass_ckpt = prev_pass_metrics = None

    if selected_ckpt is None:
        _finalize(False, f"no consecutive passing pair within {args.max_updates} updates")
        raise SystemExit("Critic NOT selected; actor stays LOCKED.")

    # ---- final test ONCE, under the SAME declared gates (incl. train_sign for v4) ----
    tf.train.Checkpoint(critic=c1, critic2=c2, target_critic=t1, target_critic2=t2).restore(selected_ckpt).expect_partial()
    mf = _audit(fin)
    ft_train_sign = train_sign_report() if pair_sampler is not None else None
    if ft_train_sign is not None and args.audit_profile == "full":
        mf["train_sign_accuracy"] = ft_train_sign["train_sign_accuracy"]
    fpass, freasons = audit_passes(mf, profile=args.audit_profile)   # SAME gates as selection
    _log_audit("final_test", selected_update, mf, passed=fpass, reasons=freasons, train_sign=ft_train_sign)
    print("FINAL_TEST " + json.dumps({k: mf[k] for k in _KEYS} | {"passed": fpass, "reasons": freasons}), flush=True)

    if fpass:
        p1 = str(ckpt_dir / "m6_1_accepted_critic1.weights.h5")
        p2 = str(ckpt_dir / "m6_1_accepted_critic2.weights.h5")
        c1.save_weights(p1); c2.save_weights(p2)
        accepted_critics = {"critic1": {"path": p1, "sha256": _sha256(p1)},
                            "critic2": {"path": p2, "sha256": _sha256(p2)}}
        _finalize(True, f"final_test passed (selected @update={selected_update})",
                  sel_update=selected_update, pass_metrics=pass_metrics, final_metrics=mf,
                  accepted_critics=accepted_critics)
        print("M6.1 CRITIC ACCEPTED. STOP before any actor update.", flush=True)
    else:
        _finalize(False, f"final_test FAILED: {freasons}", sel_update=selected_update,
                  pass_metrics=pass_metrics, final_metrics=mf)
        raise SystemExit("FINAL_TEST FAILED: experiment rejected. Do NOT re-query final_test or tune "
                         "to it. No actor unlock.")


if __name__ == "__main__":
    main()
