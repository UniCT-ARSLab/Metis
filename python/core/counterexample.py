"""Counterexample dataset core: paired Q^BC returns, label classification, same-anchor margin
calibration, and the typed replay-only NPZ schema. Framework-light + Godot-free so the
return/label/schema logic is unit-testable without the simulator (the Godot-driven rollouts live in
the generator tool and only FEED reward/collision/termination sequences into here).

Authoritative return = G_episode ([candidate ONCE] + [frozen clone BC to a real terminal OR the env
time-limit]). Label usability (classify_outcome):
  - `terminated`             -> complete return, authoritative.
  - `matched_env_time_limit` -> baseline AND candidate hit the SAME env time-limit horizon -> a
                                comparable finite-horizon return, label USABLE (finite_horizon=True).
  - `tool_cap` / `horizon_mismatch` -> label NOT usable.
G@{1,2,4,8,16,32,64,128} are sign-stability checkpoints. Metis replay semantics: `dones=terminated`
only; `truncated` (time-limit) is stored but is NOT a Bellman terminal.

Two DISTINCT bad-labels (never conflated): `paired_bad` describes ONLY the authoritative paired Q^BC
comparison on `source=="candidate"` rows; `safety_negative` marks OOD-burst branches whose repeated
action leads to a collision/failure. Ranking audit uses `source=="candidate" & label_usable &
paired_bad`; the replay sampler may prioritise `safety_negative` as a SEPARATE quota.
"""
import numpy as np

GAMMA_DEFAULT = 0.99
# Return checkpoints for sign-stability. "end" (the full-length episode return) is G_*_episode.
HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128)
K_MAX_BURST = 8

# 19 candidate families: 1 collapsed_v4, 1 pga, 1 random, 2 perturbed, 14 per-joint saturation.
PERTURBED_SCALES = (0.25, 0.50)
N_JOINTS = 7
FAMILIES = (
    ["collapsed_v4", "pga", "random"]
    + [f"perturbed_{s:.2f}" for s in PERTURBED_SCALES]
    + [f"sat_j{j}_{sign}" for j in range(N_JOINTS) for sign in ("pos", "neg")]
)
assert len(FAMILIES) == 19, len(FAMILIES)

SOURCES = ("baseline", "candidate", "ood_burst", "recovery")

# Exact per-transition NPZ dtypes (schema §5). Transition arrays first, then metadata.
TRANSITION_DTYPES = {
    "obs": np.float32, "actions": np.float32, "rewards": np.float32, "next_obs": np.float32,
    "terminated": np.bool_, "truncated": np.bool_, "dones": np.bool_,
}
META_DTYPES = {
    "branch_id": np.int32, "step_in_branch": np.int32, "branch_length": np.int32,
    "cell": np.int32, "seed": np.int64,
    "phase": "<U8", "pose_id": "<U24", "family": "<U16", "source": "<U10",
    # termination_reason = how the ROLLOUT ended (ongoing/collision/self_collision/terminated/
    # truncated) -- NOT the label class (that is `outcome`) and NOT the source.
    "failure_reason": "<U24", "termination_reason": "<U16", "outcome": "<U24",
    "baseline_replicate_id": np.int32,
    "collision": np.bool_, "self_collision": np.bool_,
    # this rollout's own success flag (terminated & target_reached).
    "success": np.bool_,
    # paired_bad: authoritative candidate paired-Q^BC verdict (source==candidate only) + its rule.
    # candidate_better: candidate reached the target where the baseline hit only the time-limit.
    # safety_negative: OOD-burst branch hit a collision/failure (separate quota, NOT paired).
    "paired_bad": np.bool_, "paired_bad_rule": "<U20", "candidate_better": np.bool_,
    "safety_negative": np.bool_, "return_censored": np.bool_,
    # matched_horizon kept as a DIAGNOSTIC (never a usability gate); return_complete for terminals.
    "return_complete": np.bool_, "matched_horizon": np.bool_,
    "label_usable": np.bool_, "censor_reason": "<U16", "horizon_steps": np.int32,
    "G_baseline": np.float32, "G_candidate": np.float32, "return_delta": np.float32,
    "G_baseline_episode": np.float32, "G_candidate_episode": np.float32,
    "return_delta_episode": np.float32,
}
# return_delta_by_h is (N, len(HORIZONS)) float32, handled separately.


def discounted_return(rewards, gamma=GAMMA_DEFAULT, horizon=None):
    r = list(rewards) if horizon is None else list(rewards)[:horizon]
    return float(sum((gamma ** t) * float(x) for t, x in enumerate(r)))


def rollout_returns(rewards, gamma=GAMMA_DEFAULT, horizons=HORIZONS):
    """Local diagnostic returns G@H for H in horizons (truncated to available rewards)."""
    return {H: discounted_return(rewards, gamma, H) for H in horizons}


def classify_termination(terminated, truncated, hit_cap):
    """Simple per-rollout termination tag (used by the K=8 OOD bursts, which are always censored)."""
    if terminated:
        return "terminated", False
    if truncated:
        return "truncated", True
    if hit_cap:
        return "cap", True
    return "ongoing", True


# Endpoints the environment itself defines. A rollout that reaches ANY of these has a true
# episode return (absorbing after early termination: future rewards are 0, no artificial padding);
# the env time-limit is a valid empirical horizon. tool_cap/ongoing are the generator's own cut and
# are NOT usable. horizon MATCH is now diagnostic only, never a usability gate.
NATURAL_END_OUTCOMES = ("success", "terminal_failure", "env_time_limit")


def classify_rollout(*, terminated, truncated, hit_tool_cap, is_success):
    """Classify ONE rollout by the env's own endpoint. success = terminated & target_reached;
    terminal_failure = terminated & not success; env_time_limit = truncated (env max_steps);
    tool_cap = generator backstop; ongoing = never ended. label_usable iff a natural end."""
    if terminated:
        outcome = "success" if is_success else "terminal_failure"
    elif truncated:
        outcome = "env_time_limit"
    elif hit_tool_cap:
        outcome = "tool_cap"
    else:
        outcome = "ongoing"
    usable = outcome in NATURAL_END_OUTCOMES
    return dict(outcome=outcome, label_usable=usable,
                return_complete=outcome in ("success", "terminal_failure"),
                is_natural_end=usable)


def sign_stable_negative(deltas_by_h, horizon_steps, min_stable=2):
    """True if the last available horizon deltas (H <= horizon_steps) are ALL negative. Guards
    against a delta that is negative at the end but flapped sign along the way."""
    avail = [H for H in HORIZONS if H <= int(horizon_steps)]
    last = avail[-min_stable:] if len(avail) >= min_stable else avail
    if not last:
        return True  # too short for horizon checkpoints -> defer to the end-delta test
    return all(float(deltas_by_h[H]) < 0.0 for H in last)


def paired_verdict(*, base_outcome, base_success, base_collision, cand_outcome, cand_success,
                   cand_collision, cand_label_usable, return_delta_episode, deltas_by_h, margin,
                   horizon_steps, min_stable=2):
    """A+C ordered paired-Q^BC verdict. Baseline is assumed already GUARDED (never terminal_failure).
    Returns {paired_bad, paired_bad_rule, candidate_better}. Outcome-dominance / terminal-failure
    prevail over the return-margin test; matched_horizon plays no role here."""
    if not cand_label_usable:
        return dict(paired_bad=False, paired_bad_rule="censored", candidate_better=False)
    if base_outcome == "terminal_failure":  # defensive: guard should have rejected this anchor
        return dict(paired_bad=False, paired_bad_rule="baseline_failure", candidate_better=False)
    cand_failure = (cand_outcome == "terminal_failure")
    # 1. baseline success, candidate not success -> task-outcome dominance (ignores margin/stability)
    if base_success and not cand_success:
        return dict(paired_bad=True, paired_bad_rule="outcome_dominance", candidate_better=False)
    # 2. baseline non-failure, candidate terminal_failure (incl. a candidate-only collision)
    if cand_failure or (cand_collision and not base_collision):
        return dict(paired_bad=True, paired_bad_rule="terminal_failure", candidate_better=False)
    # 3/5. candidate reaches the target where the baseline only hit the time-limit -> better, never bad
    if cand_success and base_outcome == "env_time_limit":
        return dict(paired_bad=False, paired_bad_rule="candidate_better", candidate_better=True)
    # 4. both success OR both env_time_limit -> return margin + sign stability
    both_success = base_success and cand_success
    both_tl = base_outcome == "env_time_limit" and cand_outcome == "env_time_limit"
    if both_success or both_tl:
        bad = (float(return_delta_episode) < -float(margin)
               and sign_stable_negative(deltas_by_h, horizon_steps, min_stable))
        return dict(paired_bad=bool(bad), paired_bad_rule=("return_margin" if bad else "none"),
                    candidate_better=False)
    return dict(paired_bad=False, paired_bad_rule="none", candidate_better=False)


def anchor_state_key(pose_id, phase):
    """An anchor STATE is (pose, phase): the same pose sampled at a different trajectory phase is a
    different state. Splits must be disjoint at the POSE level (a pose never crosses splits)."""
    return f"{pose_id}::{phase}"


def validate_split_files(paths, *, required_keys=None):
    """Load train/selection/final-test NPZs TOGETHER and validate: files exist, schema keys present,
    zero shared pose_id, zero shared seed. Returns a report with per-split anchor-state count (rows
    over distinct branch/pose/phase) and unique-pose_id count (a pose reused across phases is one
    pose). `paths`: {split_name: path}."""
    import hashlib
    import os

    required_keys = set(required_keys or (list(TRANSITION_DTYPES) + list(META_DTYPES)
                                          + ["return_delta_by_h"]))
    report = {}
    pose_sets, seed_sets = {}, {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"split {name}: missing file {path}")
        z = np.load(path, allow_pickle=True)
        missing = required_keys - set(z.files)
        if missing:
            raise ValueError(f"split {name}: schema missing keys {sorted(missing)}")
        poses = set(map(str, z["pose_id"].tolist()))
        seeds = set(int(s) for s in z["seed"].tolist())
        anchor_states = set(zip(map(str, z["pose_id"].tolist()), map(str, z["phase"].tolist())))
        with open(path, "rb") as fh:
            checksum = hashlib.sha256(fh.read()).hexdigest()
        pose_sets[name], seed_sets[name] = poses, seeds
        report[name] = {"rows": int(len(z["obs"])), "unique_pose_id": len(poses),
                        "unique_seed": len(seeds), "anchor_states": len(anchor_states),
                        "checksum_sha256": checksum}
    names = list(paths)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sp = pose_sets[names[i]] & pose_sets[names[j]]
            ss = seed_sets[names[i]] & seed_sets[names[j]]
            if sp:
                raise ValueError(f"splits {names[i]}/{names[j]} share pose_id: {sorted(sp)[:5]}")
            if ss:
                raise ValueError(f"splits {names[i]}/{names[j]} share seed: {sorted(ss)[:5]}")
    report["disjoint"] = True
    return report


def calibrate_margin(baseline_episode_returns, *, floor=0.5, q=0.95):
    """Same-anchor noise: >=3 independent baseline replays of ONE fixed anchor at the SAME horizon.
    margin = max(floor, q95(|pairwise differences|)). Different seeds/targets or horizons would
    measure difficulty, not noise."""
    vals = [float(v) for v in baseline_episode_returns]
    if len(vals) < 3:
        raise ValueError("margin calibration needs >= 3 same-anchor baseline replicates")
    diffs = [abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]
    return float(max(floor, np.quantile(diffs, q)))


def assert_splits_disjoint(splits):
    """splits: {split_name: {"pose_id": iterable, "seed": iterable}}. Raise if any pose_id or seed
    is shared across splits (train / selection / final-test must be fully disjoint)."""
    names = list(splits)
    for key in ("pose_id", "seed"):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = set(map(str, splits[names[i]].get(key, [])))
                b = set(map(str, splits[names[j]].get(key, [])))
                shared = a & b
                if shared:
                    raise ValueError(
                        f"splits {names[i]} and {names[j]} share {key}: {sorted(shared)[:5]}")


class CounterexampleDataset:
    """Accumulate typed, replay-only transitions + metadata and write the schema-§5 NPZ. Refuses any
    BC-label key (this dataset must never enter the BC sampler; M5 v2 is the only BC source)."""

    _BC_LABEL_FORBIDDEN = {"is_bc", "is_bc_label", "bc_label", "is_demo"}

    def __init__(self, obs_dim, action_size, *, gamma=GAMMA_DEFAULT, horizons=HORIZONS):
        self.obs_dim = int(obs_dim)
        self.action_size = int(action_size)
        self.gamma = float(gamma)
        self.horizons = tuple(horizons)
        self._rows = []  # list of dicts

    def add(self, **row):
        bad = self._BC_LABEL_FORBIDDEN & set(row)
        if bad:
            raise ValueError(f"counterexample rows are replay-only; forbidden BC keys: {bad}")
        missing = (set(TRANSITION_DTYPES) | set(META_DTYPES) | {"return_delta_by_h"}) - set(row)
        if missing:
            raise ValueError(f"row missing keys: {sorted(missing)}")
        self._rows.append(row)

    def __len__(self):
        return len(self._rows)

    def to_arrays(self):
        if not self._rows:
            raise ValueError("no rows")
        out = {}
        for key, dt in TRANSITION_DTYPES.items():
            out[key] = np.asarray([r[key] for r in self._rows], dtype=dt)
        for key, dt in META_DTYPES.items():
            out[key] = np.asarray([r[key] for r in self._rows], dtype=dt)
        out["return_delta_by_h"] = np.asarray(
            [r["return_delta_by_h"] for r in self._rows], dtype=np.float32)
        return out

    def save(self, path, *, action_low, action_high, margin, meta):
        import json
        arrays = self.to_arrays()
        arrays.update(
            action_low=np.asarray(action_low, np.float32),
            action_high=np.asarray(action_high, np.float32),
            obs_dim=np.asarray([self.obs_dim], np.int32),
            action_size=np.asarray([self.action_size], np.int32),
            action_type=np.asarray(["continuous"]),
            gamma=np.asarray([self.gamma], np.float32),
            horizons=np.asarray(self.horizons, np.int32),
            margin=np.asarray([float(margin)], np.float32),
            meta=np.asarray([json.dumps(meta)]),
        )
        np.savez(path, **arrays)
        return path
