"""Critic-warmup audit: does the just-warmed critic actually rank good actions above bad ones?

After a critic warmup (actor frozen), the very first actor update maximises this critic's Q. If the
critic assigns HIGHER value to random / saturated actions than to the expert demos or the BC clone,
that first update drives the actor straight into garbage (the TD3+BC v4 collapse: actor unfroze ->
100% self-collision). This module scores min(Q1,Q2) on a FIXED validation batch for several action
families and -- crucially -- looks BEYOND the global means: per-sample win rates, margin quantiles,
and per-cell worst-group margins, so a critic that is right on average but wrong on a whole region
of the workspace cannot pass. That per-region blind spot is exactly how a mean-only gate lets a
collapse-prone critic through.

Pure and framework-light: `critic`/`critic2` are callables taking ``[obs, action]`` and returning a
(N,1) value; `actor` is a callable taking ``obs`` and returning raw tanh actions in [-1, 1]. That
keeps it unit-testable with tiny stand-in models.
"""
import numpy as np

GOOD_FAMILIES = ("expert", "clone")
# random + saturated are the "bad" reference actions the gate must beat. perturbed is DIAGNOSTIC
# only (a small step off the clone is not automatically bad), so it is never in the gate.
BAD_FAMILIES = ("random", "saturated_pos", "saturated_neg")


def _scale(raw, low, high):
    """Map raw tanh actions in [-1, 1] to the environment action range [low, high]."""
    raw = np.asarray(raw, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64).reshape(1, -1)
    high = np.asarray(high, dtype=np.float64).reshape(1, -1)
    return low + (raw + 1.0) * 0.5 * (high - low)


def _per_sample_min_q(critic, critic2, obs, actions):
    """Per-sample twin-clipped critic value min(Q1, Q2) (Q1 alone if no twin), shape (N,)."""
    q1 = np.asarray(critic([obs, actions.astype(np.float32)]), dtype=np.float64).reshape(-1)
    if critic2 is not None:
        q2 = np.asarray(critic2([obs, actions.astype(np.float32)]), dtype=np.float64).reshape(-1)
        return np.minimum(q1, q2)
    return q1


def _pair_stats(good_q, bad_q, group_ids, cell_margin_tol):
    """Per-sample margin statistics for one good-vs-bad family pair."""
    margin = good_q - bad_q
    stats = {
        "win_rate": float(np.mean(margin > 0.0)),
        "mean": float(np.mean(margin)),
        "median": float(np.median(margin)),
        "p10": float(np.quantile(margin, 0.10)),
        "p90": float(np.quantile(margin, 0.90)),
        "worst_cell": None,
        "worst_cell_margin": None,
        "n_cells": 0,
        "n_cells_negative": 0,
    }
    if group_ids is not None and len(group_ids) == len(margin):
        cells = np.asarray(group_ids)
        uniq = np.unique(cells)
        cell_means = {int(c) if np.issubdtype(cells.dtype, np.number) else str(c):
                      float(np.mean(margin[cells == c])) for c in uniq}
        worst = min(cell_means, key=cell_means.get)
        stats["worst_cell"] = worst
        stats["worst_cell_margin"] = cell_means[worst]
        stats["n_cells"] = len(uniq)
        stats["n_cells_negative"] = int(sum(1 for v in cell_means.values() if v < -cell_margin_tol))
        stats["cell_means"] = cell_means
    return stats


def q_ranking_audit(critic, actor, obs, expert_actions, action_low, action_high,
                    *, critic2=None, group_ids=None, perturb_scale=0.1, seed=0,
                    win_rate_threshold=0.9, cell_margin_tol=0.0):
    """Return per-family Q, per-pair margin statistics, and a robust pass/fail gate.

    Families scored on the same fixed ``obs`` batch: expert / clone / perturbed / random /
    saturated_pos / saturated_neg (see module docstring).

    The gate requires, for EVERY good-vs-bad pair (expert & clone) x (random & saturated±):
      - win_rate >= win_rate_threshold  (the good action beats the bad one on the vast majority of
        samples, not merely on average), AND
      - no cell whose MEAN margin is clearly negative (worst_cell_margin >= -cell_margin_tol) when
        ``group_ids`` (e.g. the validation ``cells``) is provided.
    perturbed is reported for diagnosis but never gated.
    """
    obs = np.asarray(obs, dtype=np.float32)
    expert_actions = np.asarray(expert_actions, dtype=np.float64)
    low = np.asarray(action_low, dtype=np.float64).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float64).reshape(1, -1)
    n, act_dim = expert_actions.shape
    rng = np.random.default_rng(seed)

    clone = _scale(np.asarray(actor(obs)), low, high)
    half = 0.5 * (high - low)
    perturbed = np.clip(clone + rng.normal(size=clone.shape) * float(perturb_scale) * half, low, high)
    random = rng.uniform(low=low, high=high, size=(n, act_dim))
    sat_pos = np.broadcast_to(high, (n, act_dim)).copy()
    sat_neg = np.broadcast_to(low, (n, act_dim)).copy()

    families = {
        "expert": expert_actions, "clone": clone, "perturbed": perturbed,
        "random": random, "saturated_pos": sat_pos, "saturated_neg": sat_neg,
    }
    q = {name: _per_sample_min_q(critic, critic2, obs, act) for name, act in families.items()}
    q_mean = {name: float(np.mean(v)) for name, v in q.items()}

    pairs = {}
    for good in GOOD_FAMILIES:
        for bad in BAD_FAMILIES:
            pairs[f"{good}_vs_{bad}"] = _pair_stats(q[good], q[bad], group_ids, cell_margin_tol)
    # perturbed kept diagnostic-only.
    diagnostic = {
        "perturbed_vs_random": _pair_stats(q["perturbed"], q["random"], group_ids, cell_margin_tol),
        "clone_vs_perturbed": _pair_stats(q["clone"], q["perturbed"], group_ids, cell_margin_tol),
    }

    win_rates = [p["win_rate"] for p in pairs.values()]
    all_win_ok = all(w >= win_rate_threshold for w in win_rates)
    have_cells = any(p["worst_cell_margin"] is not None for p in pairs.values())
    worst_cell_margins = [p["worst_cell_margin"] for p in pairs.values()
                          if p["worst_cell_margin"] is not None]
    no_negative_cell = (not have_cells) or all(m >= -cell_margin_tol for m in worst_cell_margins)
    gate_passed = bool(all_win_ok and no_negative_cell)

    return {
        "q_mean": q_mean,
        "pairs": pairs,
        "diagnostic": diagnostic,
        "gate": {
            "win_rate_threshold": float(win_rate_threshold),
            "cell_margin_tol": float(cell_margin_tol),
            "min_win_rate": float(min(win_rates)),
            "all_win_rates_ok": bool(all_win_ok),
            "worst_cell_margin": (float(min(worst_cell_margins)) if worst_cell_margins else None),
            "no_negative_cell": bool(no_negative_cell),
            "cells_available": bool(have_cells),
        },
        "n": int(n),
        "gate_passed": gate_passed,
    }


def format_audit(report):
    """One compact human-readable block for the log."""
    q = report["q_mean"]
    g = report["gate"]
    lines = ["CRITIC_WARMUP_AUDIT min(Q1,Q2) on %d samples:" % report["n"]]
    for name in ("expert", "clone", "perturbed", "random", "saturated_pos", "saturated_neg"):
        lines.append(f"  mean Q  {name:<14} {q[name]:+.4f}")
    lines.append("  good-vs-bad pairs (win_rate | mean | median | p10 | p90 | worst_cell:margin):")
    for key, p in report["pairs"].items():
        wc = "" if p["worst_cell"] is None else f" | cell {p['worst_cell']}:{p['worst_cell_margin']:+.4f} ({p['n_cells_negative']}/{p['n_cells']} neg)"
        lines.append(f"    {key:<26} wr={p['win_rate']:.3f} | {p['mean']:+.4f} | "
                     f"{p['median']:+.4f} | {p['p10']:+.4f} | {p['p90']:+.4f}{wc}")
    d = report["diagnostic"]["perturbed_vs_random"]
    lines.append(f"  [diag] perturbed_vs_random wr={d['win_rate']:.3f} mean={d['mean']:+.4f}")
    lines.append(f"  GATE min_win_rate={g['min_win_rate']:.3f}(>= {g['win_rate_threshold']:.2f}?) "
                 f"win_ok={g['all_win_rates_ok']}  "
                 f"worst_cell_margin={g['worst_cell_margin']}  no_neg_cell={g['no_negative_cell']}")
    lines.append(f"  GATE {'PASS' if report['gate_passed'] else 'FAIL'} "
                 f"(actor updates {'allowed' if report['gate_passed'] else 'BLOCKED'})")
    return "\n".join(lines)
