"""Selection/final-test audit for the M6.1 critic-only experiment. Read-only over a counterexample
split: paired ranking accuracy, worst-cell accuracy/margin, PGA/collapsed sign agreement, Spearman
of delta_Q vs empirical return_delta, a selection Bellman residual, and the clone Q level.

`critic_min_q(obs, actions) -> (N,)` returns min(Q1,Q2) for the CURRENT critics; `bc_next_action(
next_obs) -> (N, act)` returns the frozen-BC action for the TD target (Metis: truncated bootstraps,
dones=terminated). Framework-light so it can be unit-tested with stand-in callables.
"""
import numpy as np


def _spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n. Honest small-sample uncertainty --
    e.g. candidate_better has only ~10 selection cases, so its point estimate needs a wide band."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return (float(center - half), float(center + half))


def _linreg_slope(x, y):
    """OLS slope of y on x (delta_Q on return_delta)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3:
        return float("nan")
    vx = np.var(x)
    return float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 0 else float("nan")


def _nrmse(resid, tgt):
    """Scale-invariant relative RMSE: ||resid|| / ||tgt||. nan on an empty subset."""
    resid, tgt = np.asarray(resid, np.float64), np.asarray(tgt, np.float64)
    if len(resid) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(resid ** 2)) / (np.sqrt(np.mean(tgt ** 2)) + 1e-6))


def counterexample_audit(split, critic_min_q, *, target_critic_min_q=None, bc_next_action=None,
                         gamma=0.99, margin=0.5, q1=None, q2=None, huber_delta=5.0):
    """`split`: dict of arrays from a counterexample npz. `critic_min_q`(obs,act) uses the ONLINE
    critics; `target_critic_min_q`(obs,act) uses the TARGET critics (honest Bellman residual). `q1`,
    `q2` (optional) give per-critic Q for telemetry. `margin` filters the sign-agreement population
    to statistically-significant candidates. Returns a metrics dict; the trainer owns the criteria."""
    src = split["source"]
    cand = src == "candidate"
    base = src == "baseline"
    usable = split["label_usable"].astype(bool)
    pbad = split["paired_bad"].astype(bool)

    # candidate -> its anchor's baseline action (same s0; matched by pose_id + phase)
    base_action = {}
    for i in np.where(base)[0]:
        base_action[(str(split["pose_id"][i]), str(split["phase"][i]))] = split["actions"][i]

    ci = np.where(cand)[0]
    obs_c = split["obs"][ci].astype(np.float32)
    act_c = split["actions"][ci].astype(np.float32)
    key = [(str(split["pose_id"][i]), str(split["phase"][i])) for i in ci]
    have_base = np.array([k in base_action for k in key])
    act_b = np.stack([base_action[k] if k in base_action else act_c[j]
                      for j, k in enumerate(key)]).astype(np.float32)

    q_cand = np.asarray(critic_min_q(obs_c, act_c), np.float64).reshape(-1)
    q_base = np.asarray(critic_min_q(obs_c, act_b), np.float64).reshape(-1)
    delta_q = q_cand - q_base
    ret_delta = split["return_delta_episode"][ci].astype(np.float64)
    fam = split["family"][ci]
    cell = split["cell"][ci]
    use_c = usable[ci] & have_base
    bad_c = pbad[ci] & use_c

    # 1. ranking accuracy on paired_bad: baseline must outrank the (empirically worse) candidate.
    rank_ok = (q_base > q_cand)
    ranking_accuracy = float(np.mean(rank_ok[bad_c])) if bad_c.any() else float("nan")

    # 2. worst-cell accuracy + margin (paired_bad only)
    cell_acc, cell_margin = {}, {}
    for c in np.unique(cell[bad_c]) if bad_c.any() else []:
        m = bad_c & (cell == c)
        cell_acc[int(c)] = float(np.mean(rank_ok[m]))
        cell_margin[int(c)] = float(np.mean(q_base[m] - q_cand[m]))
    worst_cell_accuracy = min(cell_acc.values()) if cell_acc else float("nan")
    worst_cell_margin = min(cell_margin.values()) if cell_margin else float("nan")

    # 3. PGA/collapsed sign agreement over SIGNIFICANT candidates: source=candidate & usable &
    #    family in {pga,collapsed_v4} & |return_delta| >= margin (near-zero delta is not stable).
    pc = use_c & np.isin(fam, ["pga", "collapsed_v4"]) & (np.abs(ret_delta) >= float(margin))

    def _sign_acc(mask):
        return float(np.mean(np.sign(delta_q[mask]) == np.sign(ret_delta[mask]))) if mask.any() \
            else float("nan")

    sign_agreement = _sign_acc(pc)
    # split telemetry over the SAME pga+collapsed significant population (diagnostics for v4).
    sign_pos = _sign_acc(pc & (ret_delta > 0))
    sign_neg = _sign_acc(pc & (ret_delta < 0))
    sign_pga = _sign_acc(pc & (fam == "pga"))
    sign_collapsed = _sign_acc(pc & (fam == "collapsed_v4"))
    ph_c = split["phase"][ci]
    keys_cp = np.array([f"{int(c)}|{str(p)}" for c, p in zip(cell.tolist(), ph_c.tolist())])
    sign_by_cellphase = {str(kk): _sign_acc(pc & (keys_cp == kk))
                         for kk in (np.unique(keys_cp[pc]) if pc.any() else [])}

    # v5 SAFETY framing: the critic's job is to rank the WORSE/OOD (negative return_delta) pga/collapsed
    # actions BELOW the baseline. negative_sign_agreement is the GATE; candidate_better is telemetry
    # ONLY (few cases, wide CI -- never a hard gate here).
    pc_neg = pc & (ret_delta < 0)
    pc_pos = pc & (ret_delta > 0)
    negative_sign_agreement = _sign_acc(pc_neg)
    candidate_better_accuracy = _sign_acc(pc_pos)
    n_neg_sign = int(pc_neg.sum())
    n_candidate_better = int(pc_pos.sum())
    neg_sign_pga = _sign_acc(pc_neg & (fam == "pga"))
    neg_sign_collapsed = _sign_acc(pc_neg & (fam == "collapsed_v4"))
    neg_k = int(np.sum(np.sign(delta_q[pc_neg]) == np.sign(ret_delta[pc_neg]))) if pc_neg.any() else 0
    cb_k = int(np.sum(np.sign(delta_q[pc_pos]) == np.sign(ret_delta[pc_pos]))) if pc_pos.any() else 0
    negative_sign_ci = _wilson_ci(neg_k, n_neg_sign)
    candidate_better_ci = _wilson_ci(cb_k, n_candidate_better)

    # 3b. TERMINAL safety (v6). Terminal candidates (dones==1) bootstrap y=reward -- often a large
    #     negative (~ -25). Huber(delta=5) does NOT optimise their squared residual, so they distort a
    #     MSE-based aggregate td_nrmse. Handle them by RANK, not by Bellman fit: a terminal paired_bad
    #     action MUST be ranked below its baseline with a margin. q_base - q_cand is the rank margin.
    dones_ci = split["dones"][ci].astype(np.float64)
    term = (dones_ci == 1.0) & have_base
    nonterm = (dones_ci == 0.0) & have_base
    rew_ci = split["rewards"][ci].astype(np.float64)
    n_terminal = int(term.sum())
    term_bad = term & pbad[ci]
    terminal_bad_count = int(term_bad.sum())
    terminal_bad_ranking_accuracy = float(np.mean(q_base[term_bad] > q_cand[term_bad])) \
        if term_bad.any() else float("nan")
    terminal_worst_q_margin = float(np.min((q_base - q_cand)[term_bad])) if term_bad.any() else float("nan")
    # terminal target error is TELEMETRY ONLY (Huber does not minimise it): resid = q_cand - reward.
    term_resid = (q_cand - rew_ci)[term_bad]
    terminal_target_mae = float(np.mean(np.abs(term_resid))) if term_bad.any() else float("nan")
    terminal_target_rmse = float(np.sqrt(np.mean(term_resid ** 2))) if term_bad.any() else float("nan")

    # 4. Spearman + regression SLOPE of delta_Q on return_delta over usable candidates
    spearman = _spearman(delta_q[use_c], ret_delta[use_c]) if use_c.sum() >= 3 else float("nan")
    delta_q_slope = _linreg_slope(ret_delta[use_c], delta_q[use_c]) if use_c.sum() >= 3 else float("nan")

    # 5. absolute calibration on COMPLETE-SUCCESS baselines ONLY (never env_time_limit baselines:
    #    the replay bootstraps past truncation while G_episode stops at the limit -> not comparable).
    bi = np.where(base)[0]
    b_out = split["outcome"][bi]
    b_success = (b_out == "success")
    obs_b = split["obs"][bi].astype(np.float32)
    act_bb = split["actions"][bi].astype(np.float32)
    q_clone = np.asarray(critic_min_q(obs_b, act_bb), np.float64).reshape(-1)
    clone_q_mean = float(np.mean(q_clone)) if len(q_clone) else float("nan")  # telemetry ONLY
    if b_success.sum() >= 3:
        g_base_ep = split["G_baseline_episode"][bi][b_success].astype(np.float64)
        baseline_calibration_spearman = _spearman(q_clone[b_success], g_base_ep)
    else:
        baseline_calibration_spearman = float("nan")

    # 6. Bellman residual with the TARGET critic + scale-invariant td_nrmse (relative error).
    #    v6: the GATE is td_nrmse_NONTERMINAL; td_nrmse_all is telemetry (terminals distort MSE that
    #    Huber does not optimise). Per-subset NRMSE + a normalised Huber residual are DIAGNOSTICS ONLY.
    td_error = td_nrmse = td_nrmse_nonterminal = float("nan")
    td_nrmse_paired_bad = td_nrmse_candidate_nonbad = td_nrmse_pga_collapsed = float("nan")
    huber_residual_normalized = float("nan")
    target_finite = True
    if bc_next_action is not None and target_critic_min_q is not None:
        nobs = split["next_obs"][ci].astype(np.float32)
        na = np.asarray(bc_next_action(nobs), np.float32)
        dones = split["dones"][ci].astype(np.float64)  # dones == terminated
        tgt = split["rewards"][ci].astype(np.float64) + gamma * (1.0 - dones) * \
            np.asarray(target_critic_min_q(nobs, na), np.float64).reshape(-1)
        resid = q_cand - tgt
        td_error = float(np.mean(resid ** 2))
        td_nrmse = _nrmse(resid, tgt)                                  # ALL -> telemetry (v6)
        nt = dones == 0.0
        td_nrmse_nonterminal = _nrmse(resid[nt], tgt[nt])             # GATE (v6 safety)
        pbad_c = pbad[ci]
        pgc = np.isin(fam, ["pga", "collapsed_v4"])
        td_nrmse_paired_bad = _nrmse(resid[nt & pbad_c], tgt[nt & pbad_c])
        td_nrmse_candidate_nonbad = _nrmse(resid[nt & ~pbad_c], tgt[nt & ~pbad_c])
        td_nrmse_pga_collapsed = _nrmse(resid[nt & pgc], tgt[nt & pgc])
        # normalised Huber residual over nonterminal (what the loss ACTUALLY minimises).
        d = float(huber_delta)
        ar = np.abs(resid[nt])
        hub = np.where(ar <= d, 0.5 * resid[nt] ** 2, d * (ar - 0.5 * d))
        huber_residual_normalized = float(np.mean(hub) / (np.mean(np.abs(tgt[nt])) + 1e-6)) \
            if nt.any() else float("nan")
        target_finite = bool(np.all(np.isfinite(tgt)))

    # Q-scale cap: p99(|Q|) over all evaluated actions vs p99(|G_episode|) in the split.
    all_q = np.concatenate([np.abs(q_cand), np.abs(q_base), np.abs(q_clone)])
    q_p99 = float(np.percentile(all_q, 99)) if len(all_q) else float("nan")
    g_all = np.abs(np.concatenate([
        split["G_baseline_episode"].astype(np.float64), split["G_candidate_episode"].astype(np.float64)]))
    g_all = g_all[np.isfinite(g_all)]
    g_p99 = float(np.percentile(g_all, 99)) if len(g_all) else float("nan")

    # per-critic telemetry (beyond min-Q)
    q1_mean = q2_mean = float("nan")
    if q1 is not None and q2 is not None:
        q1_mean = float(np.mean(np.asarray(q1(obs_c, act_c), np.float64)))
        q2_mean = float(np.mean(np.asarray(q2(obs_c, act_c), np.float64)))

    finite = bool(np.all(np.isfinite(q_cand)) and np.all(np.isfinite(q_base))
                  and np.all(np.isfinite(q_clone)) and target_finite)
    return {
        "ranking_accuracy": ranking_accuracy,
        "worst_cell_accuracy": worst_cell_accuracy,
        "worst_cell_margin": worst_cell_margin,
        "cell_accuracy": cell_acc,
        "sign_agreement_pga_collapsed": sign_agreement,
        "sign_pos": sign_pos, "sign_neg": sign_neg,
        "sign_pga": sign_pga, "sign_collapsed": sign_collapsed,
        "sign_by_cellphase": sign_by_cellphase,
        # v5 safety metrics
        "negative_sign_agreement": negative_sign_agreement,
        "candidate_better_accuracy": candidate_better_accuracy,   # telemetry ONLY (few cases)
        "n_neg_sign": n_neg_sign, "n_candidate_better": n_candidate_better,
        "neg_sign_pga": neg_sign_pga, "neg_sign_collapsed": neg_sign_collapsed,
        "negative_sign_ci": negative_sign_ci, "candidate_better_ci": candidate_better_ci,
        "spearman_deltaQ_returndelta": spearman,
        "delta_q_slope": delta_q_slope,
        "baseline_calibration_spearman": baseline_calibration_spearman,
        "clone_q_mean": clone_q_mean,   # DIAGNOSTIC ONLY
        "td_error": td_error,           # DIAGNOSTIC ONLY (raw, scale-dependent)
        "td_nrmse": td_nrmse,           # v1-v4 GATE / v6 telemetry (ALL candidates, terminal-distorted)
        "td_nrmse_nonterminal": td_nrmse_nonterminal,   # v6 safety Bellman GATE
        "td_nrmse_paired_bad": td_nrmse_paired_bad,             # diagnostics (nonterminal subsets)
        "td_nrmse_candidate_nonbad": td_nrmse_candidate_nonbad,
        "td_nrmse_pga_collapsed": td_nrmse_pga_collapsed,
        "huber_residual_normalized": huber_residual_normalized,
        # terminal safety (v6): rank-based, NOT Bellman-fit
        "n_terminal": n_terminal, "terminal_bad_count": terminal_bad_count,
        "terminal_bad_ranking_accuracy": terminal_bad_ranking_accuracy,
        "terminal_worst_q_margin": terminal_worst_q_margin,
        "terminal_target_mae": terminal_target_mae, "terminal_target_rmse": terminal_target_rmse,
        "q_p99": q_p99, "g_p99": g_p99,  # Q-scale cap
        "q1_mean": q1_mean, "q2_mean": q2_mean,
        "finite": finite,
        "n_paired_bad": int(bad_c.sum()),
        "n_usable_candidates": int(use_c.sum()),
        "n_success_baselines": int(b_success.sum()),
        "n_sign_population": int(pc.sum()),
        "q_bad_mean": float(np.mean(q_cand[bad_c])) if bad_c.any() else float("nan"),
        "q_nonbad_mean": float(np.mean(q_cand[use_c & ~pbad[ci]])) if (use_c & ~pbad[ci]).any() else float("nan"),
    }


def audit_passes(metrics, *, td_nrmse_max=0.50, q_scale_slack=10.0, train_sign_min=0.90,
                 profile="full"):
    """Apply the FIXED M6.1 criteria (never tuned to results). Two gate PROFILES:

    - "full" (v1-v4): ranking, worst-cell acc/margin, sign_agreement over ALL pga/collapsed
      significant candidates (incl. positives), Spearman, delta_q_slope>0, baseline calibration>0,
      td_nrmse, Q-cap, finite, + OPTIONAL train_sign gate when the key is present.
    - "safety" (v5 SAFETY critic): the critic must recognise/rank-down the WORSE OOD actions. Gates:
      ranking (bad-ranking) >=0.90, worst-cell acc >=0.80, worst-cell margin >=0,
      NEGATIVE sign agreement on pga/collapsed >=0.90, Spearman >=0.50, td_nrmse<=max, Q-cap, finite.
      candidate_better (positive) is telemetry ONLY -- never a hard gate here. NO slope/calibration/
      train_sign gates.

    td_nrmse (relative Bellman error) + a Q-scale cap replace the near-unsatisfiable update-0 td gate.
    Returns (passed, reasons)."""
    reasons = []
    def need(cond, msg):
        if not cond:
            reasons.append(msg)
    m = metrics
    need(m["finite"], "NaN/Inf in Q or target")
    need(m["ranking_accuracy"] >= 0.90, f"ranking_accuracy {m['ranking_accuracy']:.3f} < 0.90")
    need(m["worst_cell_accuracy"] >= 0.80, f"worst_cell_accuracy {m['worst_cell_accuracy']:.3f} < 0.80")
    need(m["worst_cell_margin"] >= 0.0, f"worst_cell_margin {m['worst_cell_margin']:.3f} < 0")
    if profile == "safety":
        need(not np.isnan(m["negative_sign_agreement"]) and m["negative_sign_agreement"] >= 0.90,
             f"negative_sign_agreement {m['negative_sign_agreement']} < 0.90")
    else:
        need(not np.isnan(m["sign_agreement_pga_collapsed"]) and m["sign_agreement_pga_collapsed"] >= 0.90,
             f"sign_agreement {m['sign_agreement_pga_collapsed']} < 0.90")
    need(not np.isnan(m["spearman_deltaQ_returndelta"]) and m["spearman_deltaQ_returndelta"] >= 0.50,
         f"spearman {m['spearman_deltaQ_returndelta']} < 0.50")
    if profile == "full":
        need(not np.isnan(m["delta_q_slope"]) and m["delta_q_slope"] > 0.0,
             f"delta_q_slope {m['delta_q_slope']} <= 0")
        need(not np.isnan(m["baseline_calibration_spearman"]) and m["baseline_calibration_spearman"] > 0.0,
             f"baseline_calibration_spearman {m['baseline_calibration_spearman']} <= 0")
    if profile == "safety":
        # v6: the Bellman gate is NONTERMINAL nrmse (Huber does not optimise the big-reward terminals'
        # MSE, so they must not gate the fit). Terminals are handled by RANK below.
        need(not np.isnan(m["td_nrmse_nonterminal"]) and m["td_nrmse_nonterminal"] <= td_nrmse_max,
             f"td_nrmse_nonterminal {m['td_nrmse_nonterminal']} > {td_nrmse_max}")
        # TERMINAL safety gate: if the split has terminal candidates, EVERY one must be classified
        # paired_bad AND ranked below its baseline (acc==1.0) with worst rank-margin >= 0.25.
        if m.get("n_terminal", 0) > 0:
            need(m["terminal_bad_count"] == m["n_terminal"],
                 f"terminal not classified paired_bad ({m['terminal_bad_count']}/{m['n_terminal']})")
            need(m["terminal_bad_ranking_accuracy"] == 1.0,
                 f"terminal_bad_ranking_accuracy {m['terminal_bad_ranking_accuracy']} != 1.0")
            need(not np.isnan(m["terminal_worst_q_margin"]) and m["terminal_worst_q_margin"] >= 0.25,
                 f"terminal_worst_q_margin {m['terminal_worst_q_margin']} < 0.25")
    else:
        need(not np.isnan(m["td_nrmse"]) and m["td_nrmse"] <= td_nrmse_max,
             f"td_nrmse {m['td_nrmse']} > {td_nrmse_max}")
    if np.isfinite(m["q_p99"]) and np.isfinite(m["g_p99"]):
        cap = 2.0 * m["g_p99"] + q_scale_slack
        need(m["q_p99"] <= cap, f"q_p99 {m['q_p99']:.3f} > cap {cap:.3f} (2*g_p99+{q_scale_slack})")
    else:
        need(False, "q_p99/g_p99 not finite")
    if profile == "full" and "train_sign_accuracy" in m:   # v4 sign-hinge only (optional gate)
        need(not np.isnan(m["train_sign_accuracy"]) and m["train_sign_accuracy"] >= train_sign_min,
             f"train_sign_accuracy {m['train_sign_accuracy']} < {train_sign_min}")
    return (len(reasons) == 0), reasons
