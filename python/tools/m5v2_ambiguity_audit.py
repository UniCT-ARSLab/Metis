"""M6.2 Phase A: demonstration AMBIGUITY audit of M5 v2 (READ-ONLY, offline, numpy only).

Question: are there very-close observations paired with DIFFERENT expert actions? If different RRT
paths pass near-identical states but command incompatible joint velocities, the BC clone is forced to
AVERAGE them -> multimodal target, irreducible cloning error. This measures it directly.

Method: standardize obs (z-score per dim). For a sample of query steps, find the nearest CROSS-EPISODE
neighbours (same-episode neighbours are excluded -- consecutive steps of one trajectory are trivially
near-obs with near-actions; that is temporal continuity, NOT ambiguity). For each query record the
obs distance, the action L2 gap and the action cosine similarity to its nearest cross-episode
neighbour, plus a tight-neighbourhood conditional action std (the averaging error BC cannot avoid).
Compare against random cross-episode pairs. Break down per cell and per phase (episode step fraction).

NO changes to observation / reward / network / dataset. Pure analysis.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

M5V2 = "python/demos/openarm_reach_hold_m5_v2/train.npz"


def _phase_of(step_idx, ep_idx):
    """Per-episode normalized step -> approach/mid/settle."""
    phase = np.empty(len(step_idx), dtype="<U8")
    for ep in np.unique(ep_idx):
        m = ep_idx == ep
        s = step_idx[m].astype(np.float64)
        frac = (s - s.min()) / max(1.0, (s.max() - s.min()))
        lab = np.where(frac <= 1 / 3, "approach", np.where(frac <= 2 / 3, "mid", "settle"))
        phase[m] = lab
    return phase


def _knn_cross_episode(qobs, allobs, q_ep, all_ep, k, block=1024):
    """For each query row, the k nearest rows in allobs whose episode != the query's episode.
    Returns (idx [Q,k], dist [Q,k]) in the standardized space. Blocked BLAS matmul."""
    Q = len(qobs)
    idx = np.empty((Q, k), np.int64)
    dist = np.empty((Q, k), np.float64)
    all_sq = np.sum(allobs ** 2, axis=1)
    for b in range(0, Q, block):
        qb = qobs[b:b + block]
        d2 = np.sum(qb ** 2, axis=1)[:, None] + all_sq[None, :] - 2.0 * qb @ allobs.T
        d2 = np.maximum(d2, 0.0)
        # forbid same-episode neighbours (set their distance to +inf)
        same = q_ep[b:b + block][:, None] == all_ep[None, :]
        d2[same] = np.inf
        part = np.argpartition(d2, k, axis=1)[:, :k]
        rows = np.arange(len(qb))[:, None]
        dpart = d2[rows, part]
        order = np.argsort(dpart, axis=1)
        idx[b:b + block] = part[rows, order]
        dist[b:b + block] = np.sqrt(dpart[rows, order])
    return idx, dist


def _act_stats(a_i, a_j):
    l2 = np.linalg.norm(a_i - a_j, axis=1)
    dot = np.sum(a_i * a_j, axis=1)
    ni = np.linalg.norm(a_i, axis=1) + 1e-9
    nj = np.linalg.norm(a_j, axis=1) + 1e-9
    cos = dot / (ni * nj)
    return l2, cos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m5v2", default=M5V2)
    ap.add_argument("--sample", type=int, default=8000, help="query steps sampled (neighbours = all)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tight-percentile", type=float, default=10.0,
                    help="'very close' = nearest-neighbour obs distance below this percentile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="python/demos/openarm_reach_hold_m5_v2/ambiguity_audit.json")
    args = ap.parse_args()

    z = np.load(args.m5v2, allow_pickle=True)
    obs = np.asarray(z["obs"], np.float64)
    act = np.asarray(z["actions"], np.float64)
    cells = np.asarray(z["cells"]).astype(int)
    ep = np.asarray(z["episode_indices"]).astype(int)
    step = np.asarray(z["step_indices"]).astype(int)
    phase = _phase_of(step, ep)
    n = len(obs)

    mu = obs.mean(0); sd = obs.std(0); sd[sd < 1e-8] = 1.0
    obs_std = (obs - mu) / sd

    rng = np.random.default_rng(args.seed)
    qi = rng.choice(n, size=min(args.sample, n), replace=False)

    idx, dist = _knn_cross_episode(obs_std[qi], obs_std, ep[qi], ep, args.k)
    nn = idx[:, 0]                       # nearest cross-episode neighbour
    nn_dist = dist[:, 0]
    l2_nn, cos_nn = _act_stats(act[qi], act[nn])

    # random cross-episode baseline (match count)
    rj = rng.choice(n, size=len(qi), replace=True)
    ok = ep[rj] != ep[qi]
    l2_rand, cos_rand = _act_stats(act[qi][ok], act[rj][ok])

    # tight neighbourhood: queries whose NN is very close -> the ambiguous ones
    thr = float(np.percentile(nn_dist, args.tight_percentile))
    tight = nn_dist <= thr

    # conditional action std over the tight k-neighbourhood (averaging error BC cannot avoid)
    cond_std = []
    for r in np.where(tight)[0]:
        neigh = idx[r][dist[r] <= thr]
        if len(neigh) >= 2:
            aset = np.vstack([act[qi[r]][None, :], act[neigh]])
            cond_std.append(float(np.linalg.norm(aset.std(0))))
    cond_std = np.asarray(cond_std)

    def _summ(l2, cos):
        return {"n": int(len(l2)), "act_l2_mean": float(np.mean(l2)), "act_l2_median": float(np.median(l2)),
                "act_cos_mean": float(np.mean(cos)), "act_cos_median": float(np.median(cos))}

    report = {
        "n_total": n, "n_query": int(len(qi)), "k": args.k,
        "obs_nn_dist_median_std_space": float(np.median(nn_dist)),
        "tight_threshold_std_space": thr, "n_tight": int(tight.sum()),
        "nearest_neighbour_pairs": _summ(l2_nn, cos_nn),
        "tight_pairs": _summ(l2_nn[tight], cos_nn[tight]),
        "random_pairs_baseline": _summ(l2_rand, cos_rand),
        "ambiguity_ratio_tight_vs_random_actL2": (float(np.mean(l2_nn[tight]) / (np.mean(l2_rand) + 1e-9))),
        "conditional_action_std_tight": {
            "n": int(len(cond_std)), "mean": float(np.mean(cond_std)) if len(cond_std) else float("nan"),
            "p90": float(np.percentile(cond_std, 90)) if len(cond_std) else float("nan"),
            "max": float(np.max(cond_std)) if len(cond_std) else float("nan")},
    }
    # per cell / per phase (tight pairs only -> where ambiguity bites)
    def _group(labels):
        out = {}
        for g in sorted(set(labels[qi].tolist())):
            m = tight & (labels[qi] == g)
            if m.any():
                out[str(g)] = {"n": int(m.sum()), "act_l2_mean": float(np.mean(l2_nn[m])),
                               "act_cos_mean": float(np.mean(cos_nn[m])),
                               "nn_dist_mean": float(np.mean(nn_dist[m]))}
        return out
    report["per_cell_tight"] = _group(cells)
    report["per_phase_tight"] = _group(phase)

    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print("=== M5 v2 AMBIGUITY AUDIT ===", flush=True)
    print(f"n={n} query={len(qi)} k={args.k}", flush=True)
    print(f"nearest cross-episode pair: act_L2 mean {report['nearest_neighbour_pairs']['act_l2_mean']:.3f} "
          f"cos {report['nearest_neighbour_pairs']['act_cos_mean']:.3f}", flush=True)
    print(f"TIGHT (very-close obs, n={report['n_tight']}): act_L2 mean {report['tight_pairs']['act_l2_mean']:.3f} "
          f"cos {report['tight_pairs']['act_cos_mean']:.3f}", flush=True)
    print(f"RANDOM baseline: act_L2 mean {report['random_pairs_baseline']['act_l2_mean']:.3f} "
          f"cos {report['random_pairs_baseline']['act_cos_mean']:.3f}", flush=True)
    print(f"ambiguity_ratio (tight actL2 / random actL2) = "
          f"{report['ambiguity_ratio_tight_vs_random_actL2']:.3f}  "
          f"(near 1 = obs does NOT determine action = multimodal; near 0 = obs determines action)", flush=True)
    print(f"conditional action std | tight neighbourhood: mean "
          f"{report['conditional_action_std_tight']['mean']:.3f} p90 "
          f"{report['conditional_action_std_tight']['p90']:.3f}", flush=True)
    print("per-cell tight act_L2:", {k: round(v["act_l2_mean"], 3) for k, v in report["per_cell_tight"].items()}, flush=True)
    print("per-phase tight act_L2:", {k: round(v["act_l2_mean"], 3) for k, v in report["per_phase_tight"].items()}, flush=True)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
