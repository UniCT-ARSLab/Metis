"""M7.1 ONLINE CRITIC WARMUP — OpenArm ADAPTER over a generic twin-critic warmup.

This script is the OpenArm reach-hold ADAPTER: it binds the generic critic-warmup logic (collection,
manifest audit, gates, persistence — all parameterised by `--task-config`) to the OpenArm Godot scene
via the `Probe` (scene/project/region/cell/distance keys come from the task-config).

Trains ONLY twin critics on FRESH ONLINE transitions collected around a FROZEN base policy. The RL
residual actor and its optimizer receive ZERO updates. Before any actor unlock the critics face a
PAIRED audit on a PRECOMPUTED IMMUTABLE manifest of REAL rollouts:

  GATE metric = ONE candidate action at the anchor, THEN the frozen policy to NATURAL termination,
                DISCOUNTED with the same gamma as the critic (this IS Q(s,a), not a truncated Q_H).
                dQ = min(Q1,Q2)(s, a) - min(Q1,Q2)(s, base).
  DIAGNOSTIC  = the same perturbation SUSTAINED for a window (discounted); never gates.

Perturbations use the REAL residual gate: correction = gate(‖obs[err]‖)·delta_max·unit (gate
outer/inner from the composite manifest, obs-space — do NOT change). train/selection/final decks are
disjoint; the Bellman holdout is disjoint from train. Persistence is transactional and the final-test
is single-use + crash-safe (marker written BEFORE consulting the final). Composite + residual hashes
are checked before/after AND against the composite manifest. NO SAC alpha/entropy; frozen-policy target
with NO target-policy noise. Even on PASS: STOP, do NOT update the actor (M7.2 needs a new approval).
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.counterexample_audit import _spearman, _wilson_ci  # noqa: E402
from core.models import build_continuous_critic  # noqa: E402
from core.settling_residual import build_settling_residual  # noqa: E402
from tools.dagger_round import agent_info_of  # noqa: E402
from tools.eval_residual_canary import DEFAULT_GODOT_BIN  # noqa: E402
from tools.m7_reward_preflight import Probe  # noqa: E402

GAMMA = 0.99
SEED_STRIDE = 1_000_000       # per-cell seed span; splits 100M apart; rounds 100k apart; holdout at 900k
ROUND_SUB = 100_000
HOLDOUT_SUB = 900_000


class TaskCfg:
    def __init__(self, d):
        self.name = d["name"]; self.region = d.get("region", "easy")
        self.obs_dim = int(d["obs_dim"]); self.action_size = int(d["action_size"])
        self.cells = list(d["cells"]); self.hard = tuple(d["hard_cells"])
        self.bands = [tuple(b) for b in d["bands_m"]]
        self.gate_outer = float(d["gate_outer"]); self.gate_inner = float(d["gate_inner"])
        self.err_start = int(d["err_start"]); self.err_size = int(d["err_size"])
        self.anchor_lo = float(d.get("anchor_dist_lo", 0.02)); self.anchor_hi = float(d.get("anchor_dist_hi", 0.10))
        self.scene = d.get("godot_scene"); self.project = d.get("godot_project", "godot")
        self.distance_key = d.get("distance_key", "position_error_m")
        self.terminal_key = d.get("terminal_key", "terminal_reason")
        self.success_value = d.get("success_value", "target_reached")

    def gate(self, obs):
        o = np.atleast_2d(np.asarray(obs, np.float64))
        d = np.linalg.norm(o[:, self.err_start:self.err_start + self.err_size], axis=-1)
        g = np.clip((self.gate_outer - d) / (self.gate_outer - self.gate_inner), 0.0, 1.0)
        return g if g.shape[0] > 1 else float(g[0])

    def band_of(self, dd):
        for i, (lo, hi) in enumerate(self.bands):
            if lo <= dd <= hi:
                return i
        return -1


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _wsha(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w, np.float32).tobytes())
    return h.hexdigest()


def _disc(tail, gamma=GAMMA):
    """Discounted return of a reward list (Σ gamma^i r_i) — same gamma as the critic bootstrap."""
    t = np.asarray(tail, np.float64)
    return float(np.sum(gamma ** np.arange(len(t)) * t)) if len(t) else 0.0


def _atomic_write(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)             # atomic rename on POSIX


def effective_counts(per_cell, rounds, n_bands, n_beh=3):
    """True per-cell collection total once the per-(band,behaviour) quota rounds DOWN — so the declared
    count matches what is actually collected (e.g. 480/6/4 -> quota 6 -> 432/cell, not 480)."""
    quota = max(1, (per_cell // rounds) // (n_bands * n_beh))
    return quota, quota * n_bands * n_beh * rounds


def full_run_exit_code(dry_run, m7_1_pass, hashes_ok):
    """Fail-closed exit code: 4 if a frozen hash mutated; 1 if a full (non-dry) run did NOT reach
    M7_1_PASS (including selected=False); 0 otherwise."""
    if not hashes_ok:
        return 4
    if not dry_run and not m7_1_pass:
        return 1
    return 0


def correction_basis(action_size, n_gauss, rng):
    dirs = []
    for j in range(action_size):
        for s in (+1, -1):
            u = np.zeros(action_size, np.float64); u[j] = s
            dirs.append((f"j{j}{s:+d}", u, True))
    for g in range(n_gauss):
        v = rng.normal(0, 1, action_size)
        dirs.append((f"g{g}", v / (np.linalg.norm(v) + 1e-9), False))
    return dirs


def _deck(seed_base, ci, sub, i):
    return seed_base + ci * SEED_STRIDE + sub + i


def gate_scaled_correction(base_a, unit, gate_val, delta_max):
    """Real residual correction e = gate(distance)·delta_max·unit and clip(base + e, -1, 1)."""
    e = np.asarray(float(gate_val) * float(delta_max) * np.asarray(unit, np.float64), np.float64)
    return e, np.clip(np.asarray(base_a, np.float64) + e, -1.0, 1.0)


def audit_pairs(manifest, qmin_fn):
    """PURE paired audit on a FIXED manifest. dReturn = ONE-ACTION-then-composite DISCOUNTED return
    (the GATE metric, matching the critic's bootstrap); the sustained nudge is carried as diagnostic.
    Deterministic given (manifest, qmin_fn)."""
    pairs = []
    for cell, ancs in manifest["anchors"].items():
        for a in ancs:
            q_base = float(qmin_fn([a["obs_k"]], [a["base_a"]])[0])
            for p in a["perts"]:
                dq = float(qmin_fn([a["obs_k"]], [p["pert_action"]])[0] - q_base)
                pairs.append({"cell": int(cell), "dq": dq, "dret": p["dret_oneshot"],
                              "dret_sustained": p["dret_sustained"], "at_bound": bool(p["at_bound"])})
    return pairs


def compute_audit_metrics(pairs, cells, hard):
    def acc(ps):
        sig = [p for p in ps if abs(p["dret"]) > 1e-9]
        k = sum(np.sign(p["dq"]) == np.sign(p["dret"]) for p in sig)
        return (k / len(sig) if sig else float("nan")), len(sig)
    per_cell = {}
    for c in cells:                                            # DECLARED cells -> missing surfaces as n=0
        a, n = acc([p for p in pairs if p["cell"] == c])
        per_cell[int(c)] = {"dir_acc": a, "n": int(n), "ci": _wilson_ci(int(a * n) if a == a else 0, n)}
    g_acc, g_n = acc(pairs)
    sp = _spearman([p["dq"] for p in pairs], [p["dret"] for p in pairs])
    hp = [p for p in pairs if p["cell"] in hard]
    sp_hard = _spearman([p["dq"] for p in hp], [p["dret"] for p in hp])
    wm = []
    for c in cells:
        up = [p["dq"] for p in pairs if p["cell"] == c and p["dret"] > 0]
        dn = [p["dq"] for p in pairs if p["cell"] == c and p["dret"] < 0]
        if up and dn:
            wm.append(float(np.mean(up) - np.mean(dn)))
    worst_q_margin = min(wm) if wm else float("nan")
    bad = [p for p in pairs if p["at_bound"] and p["dret"] < -1e-9]
    sat_pref = float(np.mean([p["dq"] > 0 for p in bad])) if bad else float("nan")
    return {"dir_acc_global": g_acc, "n_global": int(g_n), "spearman_global": sp, "spearman_hard": sp_hard,
            "worst_cell_q_margin": worst_q_margin, "saturated_bad_pref": sat_pref, "n_saturated_bad": len(bad),
            "per_cell": per_cell}


def final_test_marker(ckdir):
    return Path(ckdir) / "final_test_consumed.json"


def check_final_test_single_use(ckdir, dry_run):
    """Fail-closed across restarts: ANY marker on disk (started OR completed) => already consumed."""
    m = final_test_marker(ckdir)
    if m.exists() and not dry_run:
        return False, f"final-test already consumed: {m}"
    return True, ""


def save_critic_checkpoint(ckdir, c1, c2, t1, t2, opt1, opt2, state):
    """Transactional critic/target/optimizer persistence + a state.json (selected update, hashes)."""
    ckdir = Path(ckdir)
    for m, n in ((c1, "critic1"), (c2, "critic2"), (t1, "target1"), (t2, "target2")):
        m.save_weights(ckdir / f"{n}.weights.h5")
    np.savez_compressed(ckdir / "opt_state.npz",
                        **{f"o1_{i}": v.numpy() for i, v in enumerate(opt1.variables)},
                        **{f"o2_{i}": v.numpy() for i, v in enumerate(opt2.variables)})
    _atomic_write(ckdir / "critic_state.json", json.dumps(state, default=float))


def load_critic_checkpoint(ckdir, obs_dim, action_size, lr=3e-5):
    """Rebuild twin critics + targets + optimizers from a saved checkpoint (TESTED reload)."""
    ckdir = Path(ckdir)
    c1 = build_continuous_critic(obs_dim, action_size); c2 = build_continuous_critic(obs_dim, action_size)
    t1 = build_continuous_critic(obs_dim, action_size); t2 = build_continuous_critic(obs_dim, action_size)
    opt1 = tf.keras.optimizers.Adam(lr); opt2 = tf.keras.optimizers.Adam(lr)
    z = np.zeros((1, obs_dim), np.float32); za = np.zeros((1, action_size), np.float32)
    for c, opt in ((c1, opt1), (c2, opt2)):                    # materialise optimizer slots first
        with tf.GradientTape() as g:
            loss = tf.reduce_sum(c([z, za]))
        opt.apply_gradients(zip(g.gradient(loss, c.trainable_variables), c.trainable_variables))
    for m, n in ((c1, "critic1"), (c2, "critic2"), (t1, "target1"), (t2, "target2")):
        m.load_weights(ckdir / f"{n}.weights.h5")              # restore exact weights (over the dummy step)
    data = np.load(ckdir / "opt_state.npz")
    for i, v in enumerate(opt1.variables):
        if f"o1_{i}" in data:
            v.assign(data[f"o1_{i}"])
    for i, v in enumerate(opt2.variables):
        if f"o2_{i}" in data:
            v.assign(data[f"o2_{i}"])
    state = {}
    if (ckdir / "critic_state.json").exists():
        state = json.loads((ckdir / "critic_state.json").read_text())
    return {"c1": c1, "c2": c2, "t1": t1, "t2": t2, "opt1": opt1, "opt2": opt2, "state": state}


def _load_manifest(path):
    m = json.loads(Path(path).read_text())
    m["anchors"] = {int(c): v for c, v in m["anchors"].items()}
    return m


def _load_holdout(path):
    return {k: np.asarray(v) for k, v in np.load(path).items()}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-config", default="python/demos/openarm_reach_hold_m7/task_m7_1.json")
    ap.add_argument("--policy", default="checkpoints/openarm_reach_hold_m6_3_policy/policy.keras")
    ap.add_argument("--residual-weights", default="checkpoints/openarm_reach_hold_m6_2_c1/residual_settling.weights.h5")
    ap.add_argument("--composite-manifest", default="checkpoints/openarm_reach_hold_m6_3_policy/composite_manifest.json")
    ap.add_argument("--checkpoint-dir", default="checkpoints/openarm_reach_hold_m7_1_critic")
    ap.add_argument("--godot-bin", default=DEFAULT_GODOT_BIN)
    ap.add_argument("--port", type=int, default=6590)
    ap.add_argument("--curriculum-level", type=float, default=0.4)
    ap.add_argument("--delta-max", type=float, default=0.005)
    ap.add_argument("--critic-learning-rate", type=float, default=3e-5)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--huber-delta", type=float, default=5.0)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--utd", type=float, default=1.0)
    ap.add_argument("--collection-rounds", type=int, default=6)
    ap.add_argument("--collect-per-cell", type=int, default=576, help="near-target transitions/cell TOTAL (must divide cleanly by rounds*bands*3)")
    ap.add_argument("--max-updates", type=int, default=20000, help="hard ceiling only; actual = utd * transitions")
    ap.add_argument("--perturb-sigma", type=float, default=0.4)
    ap.add_argument("--train-seed-base", type=int, default=100_000_000)
    ap.add_argument("--selection-seed-base", type=int, default=200_000_000)
    ap.add_argument("--final-seed-base", type=int, default=300_000_000)
    ap.add_argument("--anchors-per-cell", type=int, default=4)
    ap.add_argument("--audit-gauss", type=int, default=4)
    ap.add_argument("--audit-window", type=int, default=25, help="SUSTAINED diagnostic window (NOT the gate)")
    ap.add_argument("--holdout-per-cell", type=int, default=40)
    ap.add_argument("--dir-acc-global", type=float, default=0.80)
    ap.add_argument("--dir-acc-cell", type=float, default=0.65)
    ap.add_argument("--dir-acc-hard", type=float, default=0.70)
    ap.add_argument("--spearman-global", type=float, default=0.50)
    ap.add_argument("--spearman-hard", type=float, default=0.40)
    ap.add_argument("--worst-cell-q-margin-min", type=float, default=0.0)
    ap.add_argument("--td-nrmse-max", type=float, default=0.60)
    ap.add_argument("--min-pairs-per-cell", type=int, default=30)
    ap.add_argument("--min-anchors-per-cell", type=int, default=3)
    ap.add_argument("--min-saturated-bad", type=int, default=5)
    ap.add_argument("--q-p99-max", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-manifests", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = TaskCfg(json.loads(Path(args.task_config).read_text()))
    ckdir = Path(args.checkpoint_dir)
    if args.dry_run:
        ckdir = ckdir / "dryrun"
        cfg.cells = [c for c in cfg.cells if c in (25,)] or cfg.cells[:1]
        args.collection_rounds = 2; args.collect_per_cell = 24; args.max_updates = 200
        args.anchors_per_cell = 2; args.audit_gauss = 1; args.holdout_per_cell = 8
        args.min_pairs_per_cell = 1; args.min_anchors_per_cell = 1; args.min_saturated_bad = 1
    ckdir.mkdir(parents=True, exist_ok=True)

    comp_hash0 = _sha(args.policy); c1_res_hash0 = _sha(args.residual_weights)
    cmanifest = json.loads(Path(args.composite_manifest).read_text())
    mh = cmanifest.get("hashes", {})
    manifest_consistent = bool(mh.get("policy_keras") == comp_hash0 and mh.get("residual_weights") == c1_res_hash0)
    if not manifest_consistent:
        print(f"FATAL manifest hash mismatch: policy {mh.get('policy_keras','?')[:12]} vs {comp_hash0[:12]}, "
              f"residual {mh.get('residual_weights','?')[:12]} vs {c1_res_hash0[:12]}", flush=True)
        sys.exit(3)

    probe = Probe(godot_bin=args.godot_bin, port=args.port, curriculum=args.curriculum_level, max_steps=300,
                  policy=args.policy, region=cfg.region, scene=cfg.scene, project_dir=cfg.project,
                  distance_key=cfg.distance_key, terminal_key=cfg.terminal_key, success_value=cfg.success_value)
    composite = probe.model
    rl_actor = build_settling_residual(cfg.obs_dim, cfg.action_size)
    rl_actor_hash0 = _wsha(rl_actor)
    c1 = build_continuous_critic(obs_dim=cfg.obs_dim, action_size=cfg.action_size)
    c2 = build_continuous_critic(obs_dim=cfg.obs_dim, action_size=cfg.action_size)
    t1 = build_continuous_critic(obs_dim=cfg.obs_dim, action_size=cfg.action_size)
    t2 = build_continuous_critic(obs_dim=cfg.obs_dim, action_size=cfg.action_size)
    t1.set_weights(c1.get_weights()); t2.set_weights(c2.get_weights())
    opt1 = tf.keras.optimizers.Adam(args.critic_learning_rate)
    opt2 = tf.keras.optimizers.Adam(args.critic_learning_rate)

    def base_action(obs):
        return np.clip(composite(tf.convert_to_tensor(obs[None, :], tf.float32), training=False).numpy()[0], -1, 1)

    def base_action_batch(ob):
        return np.clip(composite(tf.convert_to_tensor(np.asarray(ob, np.float32), tf.float32), training=False).numpy(), -1, 1)

    def qmin(obs, act):
        o = tf.convert_to_tensor(np.asarray(obs, np.float32)); a = tf.convert_to_tensor(np.asarray(act, np.float32))
        return tf.minimum(c1([o, a], training=False), c2([o, a], training=False)).numpy().reshape(-1)

    def q_p99_of(obs, act):
        """|Q| 99th percentile over the WHOLE provided set (chunked), not just a leading slice."""
        obs = np.asarray(obs); act = np.asarray(act)
        qs = [np.abs(qmin(obs[i:i + 8192], act[i:i + 8192])) for i in range(0, len(obs), 8192)] or [np.array([0.0])]
        q = np.concatenate(qs)
        return float(np.percentile(q, 99)), bool(np.all(np.isfinite(q)))

    def collect(seed_base, per_cell, round_offset):
        nb = len(cfg.bands); quota = max(1, per_cell // (nb * 3))
        buf = {k: [] for k in ("obs", "act", "rew", "next", "done", "cell", "band", "beh")}
        counts = defaultdict(int)
        for ci, cell in enumerate(cfg.cells):
            def full():
                return all(counts[(cell, b, z)] >= quota for b in range(nb) for z in range(3))
            i = 0
            while not full() and i < per_cell * 3:
                seed = _deck(seed_base, ci, round_offset * ROUND_SUB, i); i += 1
                try:
                    obs, info = probe._reset(cell, seed)
                except RuntimeError:
                    continue
                for _ in range(300):
                    dd = float(agent_info_of(info).get(cfg.distance_key, np.nan))
                    b = cfg.band_of(dd)
                    a = base_action(obs); beh = -1
                    if b >= 0:
                        cand = [z for z in range(3) if counts[(cell, b, z)] < quota]
                        if cand:
                            beh = min(cand, key=lambda z: counts[(cell, b, z)])
                            g = cfg.gate(obs)
                            if beh == 1:
                                a = np.clip(a + g * np.clip(rng.normal(0, args.perturb_sigma * args.delta_max, cfg.action_size),
                                                            -args.delta_max, args.delta_max), -1, 1)
                            elif beh == 2:
                                j = rng.integers(cfg.action_size); a = a.copy()
                                a[j] = np.clip(a[j] + g * rng.choice([-1, 1]) * args.delta_max, -1, 1)
                    nobs, r, term, trunc, info = probe.env.step(a.astype(np.float32))
                    if beh >= 0:
                        buf["obs"].append(obs.astype(np.float32)); buf["act"].append(a.astype(np.float32))
                        buf["rew"].append(float(r)); buf["next"].append(nobs.astype(np.float32))
                        buf["done"].append(float(term)); buf["cell"].append(int(cell))
                        buf["band"].append(int(b)); buf["beh"].append(int(beh))
                        counts[(cell, b, beh)] += 1
                    obs = nobs
                    if term or trunc or full():
                        break
        buf = {k: np.asarray(v, np.float32 if k in ("obs", "act", "rew", "next", "done") else np.int32) for k, v in buf.items()}
        buf["_counts"] = {f"{c}_{b}_{z}": int(counts[(c, b, z)]) for c in cfg.cells for b in range(nb) for z in range(3)}
        return buf

    def collect_holdout(seed_base, per_cell):
        return collect(seed_base, per_cell, round_offset=HOLDOUT_SUB // ROUND_SUB)

    # ---- IMMUTABLE audit manifest: discounted one-action-then-composite returns (critic-independent) ----
    def build_manifest(seed_base, tag):
        basis = correction_basis(cfg.action_size, args.audit_gauss, rng)
        anchors = {}
        for ci, cell in enumerate(cfg.cells):
            got = []
            for i in range(args.anchors_per_cell * 8):
                if len(got) >= args.anchors_per_cell:
                    break
                seed = _deck(seed_base, ci, 0, i)
                try:
                    roll = probe.rollout(cell, seed)
                except RuntimeError:
                    continue
                d = np.asarray(roll["dist"], np.float64)
                idx = np.where((d >= cfg.anchor_lo) & (d <= cfg.anchor_hi))[0]
                if len(idx) == 0:
                    continue
                k = int(idx[len(idx) // 2]); obs_k = roll["obs"][k]
                g = cfg.gate(obs_k)
                if g <= 0.0:
                    continue
                base_a = base_action(obs_k)
                base_disc = _disc(probe.branch_full(cell, seed, k, np.zeros(cfg.action_size), 0)["tail"])  # to natural termination
                perts = []
                for label, u, at_bound in basis:
                    e, pert_action = gate_scaled_correction(base_a, u, g, args.delta_max)
                    one = _disc(probe.branch_full(cell, seed, k, e, 1)["tail"]) - base_disc                # GATE (one action then composite)
                    sus = _disc(probe.branch_full(cell, seed, k, e, args.audit_window)["tail"]) - base_disc  # DIAGNOSTIC
                    perts.append({"label": label, "at_bound": bool(at_bound), "gate": float(g),
                                  "pert_action": pert_action.tolist(), "dret_oneshot": one, "dret_sustained": sus})
                got.append({"seed": int(seed), "k": k, "band": int(cfg.band_of(float(d[k]))),
                            "dist": float(d[k]), "obs_k": obs_k.tolist(), "base_a": base_a.tolist(), "perts": perts})
            if len(got) < args.min_anchors_per_cell:
                raise RuntimeError(f"{tag}: cell {cell} only {len(got)} anchors < {args.min_anchors_per_cell}")
            anchors[int(cell)] = got
        missing = [c for c in cfg.cells if c not in anchors]
        if missing:
            raise RuntimeError(f"{tag}: missing cells {missing}")
        return {"tag": tag, "seed_base": seed_base, "gamma": GAMMA, "return": "discounted_to_termination", "anchors": anchors}

    def td_nrmse(hold):
        o = hold["obs"]; a = hold["act"]; r = hold["rew"].reshape(-1, 1)
        no = hold["next"]; dn = hold["done"].reshape(-1, 1)
        na = base_action_batch(no)
        qt = np.minimum(t1([no, na], training=False).numpy(), t2([no, na], training=False).numpy())
        y = r + GAMMA * (1.0 - dn) * qt
        q = np.minimum(c1([o, a], training=False).numpy(), c2([o, a], training=False).numpy())
        nt = dn.reshape(-1) == 0.0
        if not np.any(nt):
            return float("nan")
        resid = (q - y).reshape(-1)[nt]
        return float(np.sqrt(np.mean(resid ** 2)) / (np.std(y.reshape(-1)[nt]) + 1e-9))

    def gate_pass(m, td, q_p99, q_finite):
        reasons = []
        def need(c, msg):
            if not c:
                reasons.append(msg)
        need(not np.isnan(m["dir_acc_global"]) and m["dir_acc_global"] >= args.dir_acc_global, f"dir_acc_global {m['dir_acc_global']}")
        for c, v in m["per_cell"].items():
            need(v["n"] >= args.min_pairs_per_cell, f"cell {c} pairs {v['n']}<{args.min_pairs_per_cell}")
            thr = args.dir_acc_hard if c in cfg.hard else args.dir_acc_cell
            need(not np.isnan(v["dir_acc"]) and v["dir_acc"] >= thr, f"cell {c} dir_acc {v['dir_acc']}<{thr}")
        need(not np.isnan(m["spearman_global"]) and m["spearman_global"] >= args.spearman_global, f"spearman_global {m['spearman_global']}")
        need(not np.isnan(m["spearman_hard"]) and m["spearman_hard"] >= args.spearman_hard, f"spearman_hard {m['spearman_hard']}")
        need(not np.isnan(m["worst_cell_q_margin"]) and m["worst_cell_q_margin"] > args.worst_cell_q_margin_min, f"worst_q_margin {m['worst_cell_q_margin']}")
        need(m["n_saturated_bad"] >= args.min_saturated_bad, f"saturated bad n {m['n_saturated_bad']}<{args.min_saturated_bad} (untested)")
        need(not np.isnan(m["saturated_bad_pref"]) and m["saturated_bad_pref"] <= 0.5, f"saturated_bad_pref {m['saturated_bad_pref']}")
        need(not np.isnan(td) and td <= args.td_nrmse_max, f"td_nrmse {td}")
        need(q_finite and q_p99 <= args.q_p99_max, f"q_p99 {q_p99} finite {q_finite}")
        return (len(reasons) == 0), reasons

    _huber = tf.keras.losses.Huber(delta=args.huber_delta, reduction=tf.keras.losses.Reduction.NONE)

    def critic_step(buf, bi):
        o = tf.convert_to_tensor(buf["obs"][bi]); a = tf.convert_to_tensor(buf["act"][bi])
        r = tf.convert_to_tensor(buf["rew"][bi].reshape(-1, 1))
        no = tf.convert_to_tensor(buf["next"][bi]); dn = tf.convert_to_tensor(buf["done"][bi].reshape(-1, 1))
        na = tf.convert_to_tensor(base_action_batch(buf["next"][bi]), tf.float32)
        qt = tf.minimum(t1([no, na], training=False), t2([no, na], training=False))
        y = r + GAMMA * (1.0 - dn) * qt                          # frozen-policy target, NO noise
        loss_last = 0.0
        for c, opt in ((c1, opt1), (c2, opt2)):
            with tf.GradientTape() as g:
                loss = tf.reduce_mean(_huber(y, c([o, a], training=True)))
            gr, _ = tf.clip_by_global_norm(g.gradient(loss, c.trainable_variables), args.grad_clip_norm)
            opt.apply_gradients(zip(gr, c.trainable_variables)); loss_last = float(loss.numpy())
        for tv, sv in ((t1, c1), (t2, c2)):
            for wt, ws in zip(tv.weights, sv.weights):
                wt.assign(args.tau * ws + (1 - args.tau) * wt)
        return loss_last

    def run_audit(man, hold, buf_obs, buf_act, update, tag):
        pairs = audit_pairs(man, qmin); m = compute_audit_metrics(pairs, cfg.cells, cfg.hard)
        q_p99, q_fin = q_p99_of(buf_obs, buf_act)
        td = td_nrmse(hold)
        ok, reasons = gate_pass(m, td, q_p99, q_fin)
        rec = {"tag": tag, "update": update, "pass": ok, "dir_acc_global": m["dir_acc_global"],
               "spearman_global": m["spearman_global"], "spearman_hard": m["spearman_hard"],
               "worst_cell_q_margin": m["worst_cell_q_margin"], "saturated_bad_pref": m["saturated_bad_pref"],
               "n_saturated_bad": m["n_saturated_bad"], "td_nrmse": td, "q_p99": q_p99, "reasons": reasons,
               "per_cell": m["per_cell"]}
        print(f"M7.1 {tag} up={update} pass={ok} dirAcc {m['dir_acc_global']} sp {m['spearman_global']} "
              f"spH {m['spearman_hard']} wQm {m['worst_cell_q_margin']} satN {m['n_saturated_bad']} "
              f"td {td} qP99 {q_p99} reasons {reasons[:3]}", flush=True)
        return ok, rec, pairs

    nb = len(cfg.bands)
    per_round_cell = args.collect_per_cell // args.collection_rounds
    quota, eff_per_cell = effective_counts(args.collect_per_cell, args.collection_rounds, nb)
    eff_total = eff_per_cell * len(cfg.cells)

    print(f"M7.1 {'DRY-RUN ' if args.dry_run else ''}(OpenArm adapter) building manifests + holdouts...", flush=True)
    result = {"dry_run": args.dry_run, "manifest_consistent": manifest_consistent,
              "config": {k: getattr(args, k) for k in vars(args)}, "task": cfg.name, "cells": cfg.cells,
              "effective_per_cell": eff_per_cell, "effective_total": eff_total, "band_behaviour_quota": quota}
    m7_1_pass = False; comp_ok = actor_ok = c1_ok = True
    try:
        if args.reuse_manifests and (ckdir / "selection_manifest.json").exists():
            print("M7.1 reusing saved manifests + holdouts (no rebuild)", flush=True)
            sel_manifest = _load_manifest(ckdir / "selection_manifest.json")
            fin_manifest = _load_manifest(ckdir / "final_manifest.json")
            sel_hold = _load_holdout(ckdir / "holdout_selection.npz")
            fin_hold = _load_holdout(ckdir / "holdout_final.npz")
        else:
            sel_manifest = build_manifest(args.selection_seed_base, "selection")
            fin_manifest = build_manifest(args.final_seed_base, "final")
            sel_hold = collect_holdout(args.selection_seed_base, args.holdout_per_cell)
            fin_hold = collect_holdout(args.final_seed_base, args.holdout_per_cell)
            _atomic_write(ckdir / "selection_manifest.json", json.dumps(sel_manifest, default=float))
            _atomic_write(ckdir / "final_manifest.json", json.dumps(fin_manifest, default=float))
            np.savez_compressed(ckdir / "holdout_selection.npz", **{k: sel_hold[k] for k in ("obs", "act", "rew", "next", "done")})
            np.savez_compressed(ckdir / "holdout_final.npz", **{k: fin_hold[k] for k in ("obs", "act", "rew", "next", "done")})

        # UPDATE-0 baseline audit (untrained critic) — uses holdout obs for the Q cap (no buffer yet).
        audit_history = []
        _, rec0, _ = run_audit(sel_manifest, sel_hold, sel_hold["obs"], sel_hold["act"], 0, "SELECTION")
        audit_history.append(rec0)

        buffer = None
        total_updates = 0
        prev_pass = False
        selected = False
        agg_counts = defaultdict(int)
        for r in range(args.collection_rounds):
            new = collect(args.train_seed_base, per_round_cell, round_offset=r)
            for kk, vv in new["_counts"].items():
                agg_counts[kk] += vv
            if buffer is None:
                buffer = {k: new[k] for k in ("obs", "act", "rew", "next", "done", "cell", "band", "beh")}
            else:
                for k in ("obs", "act", "rew", "next", "done", "cell", "band", "beh"):
                    buffer[k] = np.concatenate([buffer[k], new[k]], axis=0)
            n_up = min(int(args.utd * len(new["obs"])), max(1, args.max_updates - total_updates))
            for _ in range(n_up):
                bi = rng.integers(len(buffer["obs"]), size=min(args.batch_size, len(buffer["obs"])))
                critic_step(buffer, bi); total_updates += 1
            ok, rec, _ = run_audit(sel_manifest, sel_hold, buffer["obs"], buffer["act"], total_updates, "SELECTION")
            audit_history.append(rec)
            if ok and prev_pass:
                selected = True
                print("M7.1 two consecutive selection passes -> FINAL-TEST", flush=True)
                break
            prev_pass = ok

        utd_effective = total_updates / max(1, len(buffer["obs"]))
        comp_ok = _sha(args.policy) == comp_hash0
        actor_ok = _wsha(rl_actor) == rl_actor_hash0
        c1_ok = _sha(args.residual_weights) == c1_res_hash0

        # ---- TRANSACTIONAL candidate persistence BEFORE consulting the final ----
        np.savez_compressed(ckdir / "replay_online.npz", **{k: buffer[k] for k in ("obs", "act", "rew", "next", "done", "cell", "band", "beh")})
        save_critic_checkpoint(ckdir, c1, c2, t1, t2, opt1, opt2,
                               {"selected": bool(selected), "total_updates": total_updates,
                                "utd_effective": utd_effective, "composite_sha": comp_hash0,
                                "c1_residual_sha": c1_res_hash0, "rl_actor_sha": rl_actor_hash0,
                                "obs_dim": cfg.obs_dim, "action_size": cfg.action_size})
        result.update({"selected": bool(selected), "utd_declared": args.utd, "utd_effective": utd_effective,
                       "total_updates": total_updates, "n_online_transitions": int(len(buffer["obs"])),
                       "collection_counts_aggregated": dict(agg_counts), "audit_history": audit_history,
                       "actor_untouched": actor_ok, "composite_unchanged": comp_ok, "c1_residual_unchanged": c1_ok})
        _atomic_write(ckdir / "m7_1_result.json", json.dumps({**result, "final_test": None, "M7_1_PASS": False}, indent=2, default=float))

        # ---- FINAL-TEST: crash-safe (marker BEFORE audit) + single-use ----
        final_test = None
        if args.dry_run:
            selected = True
        if selected:
            allowed, why = check_final_test_single_use(ckdir, args.dry_run)
            if not allowed:
                print(f"FATAL {why}", flush=True); sys.exit(2)
            if not args.dry_run:
                _atomic_write(final_test_marker(ckdir), json.dumps({"status": "started", "policy_sha": comp_hash0}))
            ok, frec, fpairs = run_audit(fin_manifest, fin_hold, buffer["obs"], buffer["act"], total_updates, "FINAL-TEST")
            final_test = frec
            _atomic_write(ckdir / "final_pairs.json", json.dumps(fpairs, default=float))
            if not args.dry_run:
                _atomic_write(final_test_marker(ckdir), json.dumps({"status": "completed", "pass": bool(ok), "policy_sha": comp_hash0}))

        m7_1_pass = bool(selected and final_test and final_test.get("pass") and comp_ok and actor_ok and c1_ok)
        result.update({"M7_1_PASS": m7_1_pass, "final_test": final_test,
                       "hashes": {"composite_before": comp_hash0, "composite_after": _sha(args.policy),
                                  "c1_residual_before": c1_res_hash0, "c1_residual_after": _sha(args.residual_weights),
                                  "rl_actor_before": rl_actor_hash0, "rl_actor_after": _wsha(rl_actor)},
                       "note": "critic warmup ONLY; actor NEVER updated. Even on M7_1_PASS, STOP before actor unlock (M7.2 = new approval)."})
        _atomic_write(ckdir / "m7_1_result.json", json.dumps(result, indent=2, default=float))
        print("M7_1_DONE " + json.dumps({k: result.get(k) for k in ("dry_run", "selected", "M7_1_PASS", "utd_effective",
              "actor_untouched", "composite_unchanged")}, default=float), flush=True)
    finally:
        probe.close()

    # ---- fail-closed exit codes (reached only on clean completion) ----
    code = full_run_exit_code(args.dry_run, m7_1_pass, comp_ok and actor_ok and c1_ok)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
