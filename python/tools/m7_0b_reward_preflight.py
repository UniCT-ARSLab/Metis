"""M7.0b: rigorous reward preflight on the REAL Stage-C failure distribution. Stratified 2-6cm
coverage (hard cells 25/28/31/41 mandatory), bounded nudges 0.0025 / 0.005 (0.01 diagnostic only),
one-step AND short-horizon returns with the terminal-bonus contribution separated, per-cell
directional accuracy + Wilson CI, and the max discounted/undiscounted return obtainable by HOVERING
for the remaining horizon. Composite frozen; reward untouched; nothing trained.

Close M7.0 (stop before M7.1) iff, at nudge 0.005, toward > away in ALL cells AND hovering stays
clearly dominated by success.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.counterexample_audit import _wilson_ci  # noqa: E402
from tools.c0_feasibility_oracle import load_goal_cfgs  # noqa: E402
from tools.dagger_round import denorm_joints  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN  # noqa: E402
from tools.m7_reward_preflight import Probe  # noqa: E402

HARD = (25, 28, 31, 41)
BUCKETS = [(0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.06)]
MAGS = [0.0025, 0.005, 0.01]     # 0.01 is DIAGNOSTIC ONLY
GATE_MAG = 0.005
GAMMA = 0.99
H = 25                            # short horizon


def full_returns(b, H, gamma=GAMMA):
    """From a branch tail (k..episode end): short-horizon (first H) AND FULL remaining-horizon returns,
    discounted + undiscounted. The FULL return INCLUDES the terminal bonus when success occurs (the
    bonus reward sits in the tail at the reaching step). short_no_bonus subtracts the bonus only when
    the reaching (episode end) fell inside the short window."""
    tf_ = np.asarray(b["tail"], np.float64)
    ts = tf_[:H]
    short = float(np.sum(ts))
    bonus_in_short = bool(b["reached"]) and len(tf_) <= H
    s = {"one_step": float(ts[0]) if len(ts) else float("nan"), "short": short,
         "short_no_bonus": short - (b["bonus"] if bonus_in_short else 0.0),
         "disc": float(np.sum([gamma ** i * r for i, r in enumerate(ts)]))}
    f = {"full": float(np.sum(tf_)),
         "full_disc": float(np.sum([gamma ** i * r for i, r in enumerate(tf_)]))}
    return f, s


def compute_verdict(rows):
    """Aggregate the M7.0b verdict from bucket rows (pure function -> re-runnable on a saved report
    without Godot). Per-cell directional accuracy at each magnitude + the hovering-vs-success
    domination check on EQUIVALENT full remaining horizons, with self-reaching hover branches excluded
    from the hovering ceiling (they are successes, not stalling)."""
    def diracc(cell, m, key):
        rs = [r for r in rows if r["cell"] == cell and m in r["mags"]]
        wins = [r["mags"][m]["toward"][key] > r["mags"][m]["away"][key] for r in rs]
        k = int(np.sum(wins)); n = len(wins)
        return {"acc": (k / n if n else float("nan")), "n": n, "ci": _wilson_ci(k, n)}
    cells = sorted(set(r["cell"] for r in rows))
    per_cell = {}
    for c in cells:
        per_cell[int(c)] = {m: {"short": diracc(c, m, "short"),
                                "short_no_bonus": diracc(c, m, "short_no_bonus"),
                                "one_step": diracc(c, m, "one_step")} for m in [f"{x}" for x in MAGS]}
    gm = f"{GATE_MAG}"
    # A pure-composite branch that REACHES the target on its own is a SUCCESS, not hovering; exclude it
    # from the hovering ceiling (else a self-solving pose's ~50 terminal bonus masquerades as a hovering
    # return and defeats the comparison). Hovering = STALLING = pure composite that never reaches.
    stalling = [r for r in rows if not r["hover_reached"]]
    hov_max_full = max((r["hover_full_undisc"] for r in stalling), default=float("nan"))
    hov_max_full_disc = max((r["hover_full_disc"] for r in stalling), default=float("nan"))
    succ_full = [r["mags"][gm]["toward"]["full"] for r in rows if r["mags"][gm]["toward"]["reached"]]
    succ_full_disc = [r["mags"][gm]["toward"]["full_disc"] for r in rows if r["mags"][gm]["toward"]["reached"]]
    succ_min_full = float(np.min(succ_full)) if succ_full else float("nan")
    succ_min_full_disc = float(np.min(succ_full_disc)) if succ_full_disc else float("nan")
    all_cells_toward = all(per_cell[c][gm]["short"]["acc"] > 0.5 for c in cells)
    # dominated: the BEST STALLING return (full horizon, non-reaching pure composite) is below the WORST
    # success return (full horizon, bonus included), both undiscounted AND discounted.
    hover_dominated = bool(succ_full and hov_max_full < succ_min_full
                           and hov_max_full_disc < succ_min_full_disc)
    return {
        "n_rows": len(rows), "cells": cells, "gate_magnitude": GATE_MAG, "horizon": "full_remaining",
        "toward_beats_away_all_cells_at_gate": bool(all_cells_toward),
        "n_stalling_nonreaching": len(stalling), "n_hover_selfreached": len(rows) - len(stalling),
        "hover_max_full_undisc": hov_max_full, "hover_max_full_disc": hov_max_full_disc,
        "success_min_full_undisc": succ_min_full, "success_min_full_disc": succ_min_full_disc,
        "n_success": len(succ_full), "hover_dominated_by_success": hover_dominated,
        "per_cell_directional_accuracy": per_cell,
        "CLOSE_M7_0": bool(all_cells_toward and hover_dominated),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=6580)
    ap.add_argument("--plans", default="python/demos/openarm_reach_hold_m7/plan_m7.json")
    ap.add_argument("--curriculum-level", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-poses", type=int, default=30)
    ap.add_argument("--out", default="python/demos/openarm_reach_hold_m7/m7_0b_reward_preflight.json")
    args = ap.parse_args()

    goals = load_goal_cfgs(args.plans)
    # hard cells FIRST (mandatory), then the rest
    order = sorted(goals, key=lambda cs: (cs[0] not in HARD, cs[0], cs[1]))
    pr = Probe(godot_bin=args.godot_bin, port=args.port, curriculum=args.curriculum_level,
               max_steps=args.max_steps)
    rows = []
    try:
        for (cell, seed) in order:
            roll = pr.rollout(cell, seed)
            d = np.asarray(roll["dist"], np.float64)
            goal = goals[(cell, seed)]
            for (lo, hi) in BUCKETS:
                idx = np.where((d >= lo) & (d <= hi))[0]
                if len(idx) == 0:
                    continue
                k = int(idx[len(idx) // 2])
                joints = denorm_joints(roll["obs"][k])
                direction = goal - joints
                nrm = float(np.linalg.norm(direction))
                if nrm < 1e-6:
                    continue
                u = direction / nrm
                hov = pr.branch_full(cell, seed, k, np.zeros(7), 0)        # pure composite = hovering
                hf, hs = full_returns(hov, H)
                rec = {"cell": int(cell), "seed": int(seed), "bucket": f"{lo}-{hi}", "dist_k": float(d[k]),
                       "hover_undisc": hs["short"], "hover_disc": hs["disc"],
                       "hover_full_undisc": hf["full"], "hover_full_disc": hf["full_disc"],
                       "hover_reached": bool(hov["reached"]), "mags": {}}
                for m in MAGS:
                    tw = pr.branch_full(cell, seed, k, m * u, H)
                    aw = pr.branch_full(cell, seed, k, -m * u, H)
                    def pack(b):
                        f, s = full_returns(b, H)
                        return {**s, **f, "reached": bool(b["reached"]), "min_dist": b["min_dist"]}
                    rec["mags"][f"{m}"] = {"toward": pack(tw), "away": pack(aw)}
                rows.append(rec)
                print(f"[m7.0b] cell {cell} seed {seed} bucket {lo}-{hi} d {d[k]:.4f} "
                      f"hov_undisc {rec['hover_undisc']:.3f} "
                      f"tw0.005-short {rec['mags']['0.005']['toward']['short']:.3f} "
                      f"aw {rec['mags']['0.005']['away']['short']:.3f}", flush=True)
            if len({r["cell"] for r in rows}) >= 11 and len(rows) >= args.max_poses:
                break
    finally:
        pr.close()

    # ---- aggregate verdict (pure function; re-runnable on saved rows without Godot) ----
    verdict = compute_verdict(rows)
    Path(args.out).write_text(json.dumps({"config": vars(args), "verdict": verdict, "rows": rows},
                                         indent=2, default=float))
    print("M7_0B_VERDICT " + json.dumps({k: verdict[k] for k in
          ("toward_beats_away_all_cells_at_gate", "hover_max_full_undisc", "success_min_full_undisc",
           "hover_dominated_by_success", "CLOSE_M7_0")}), flush=True)


if __name__ == "__main__":
    main()
