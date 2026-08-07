"""Generic building blocks for PROTECTED RESIDUAL policy learning around a frozen base policy.

Task-agnostic (no OpenArm names): a task-config JSON supplies dims, cells, hard tags, bands, the
near-target gate, and the Godot scene / protocol keys. Shared by the OpenArm residual experiments
(critic-warmup + residual-PPO canary). Pure functions here are unit-tested; nothing touches an env.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

# Disjoint seed geometry: splits 100M apart; per cell 1M; rounds/generations 100k apart; holdout at 900k.
SEED_STRIDE = 1_000_000
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
        """Real near-target gate value(s) in [0,1] from the obs-space error slice (batched or single).
        gate = clip((outer - ||err||)/(outer - inner), 0, 1); EXACTLY 0 for ||err|| >= outer (approach)."""
        o = np.atleast_2d(np.asarray(obs, np.float64))
        d = np.linalg.norm(o[:, self.err_start:self.err_start + self.err_size], axis=-1)
        g = np.clip((self.gate_outer - d) / (self.gate_outer - self.gate_inner), 0.0, 1.0)
        return g if g.shape[0] > 1 else float(g[0])

    def band_of(self, dd):
        for i, (lo, hi) in enumerate(self.bands):
            if lo <= dd <= hi:
                return i
        return -1

    @staticmethod
    def load(path):
        return TaskCfg(json.loads(Path(path).read_text()))


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sha_weights(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w, np.float32).tobytes())
    return h.hexdigest()


def deck(seed_base, ci, sub, i):
    return seed_base + ci * SEED_STRIDE + sub + i


def atomic_write(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)             # atomic rename on POSIX


def final_test_marker(ckdir):
    return Path(ckdir) / "final_test_consumed.json"


def check_final_test_single_use(ckdir, dry_run):
    """Fail-closed across restarts: ANY marker on disk (started OR completed) => already consumed."""
    m = final_test_marker(ckdir)
    if m.exists() and not dry_run:
        return False, f"final-test already consumed: {m}"
    return True, ""


def gate_scaled_correction(base_a, unit, gate_val, delta_max):
    """Real residual correction e = gate(distance)*delta_max*unit and clip(base + e, -1, 1)."""
    e = np.asarray(float(gate_val) * float(delta_max) * np.asarray(unit, np.float64), np.float64)
    return e, np.clip(np.asarray(base_a, np.float64) + e, -1.0, 1.0)


def correction_basis(action_size, n_gauss, rng):
    """Unit correction directions: per-joint +/- (each saturates ONE joint's residual bound) + Gaussian."""
    dirs = []
    for j in range(action_size):
        for s in (+1, -1):
            u = np.zeros(action_size, np.float64); u[j] = s
            dirs.append((f"j{j}{s:+d}", u, True))
    for g in range(n_gauss):
        v = rng.normal(0, 1, action_size)
        dirs.append((f"g{g}", v / (np.linalg.norm(v) + 1e-9), False))
    return dirs


# NOTE: GAE, PPO log-prob/KL and the actor/value models are NOT defined here. Residual PPO uses the
# SHARED implementation in python/algorithms/ppo.py (compute_returns_advantages, ppo_update,
# gaussian_log_prob, evaluate_actions, build_hybrid_actor_critic) through core.policy_action_adapter.
# This module holds only task/gate/deck/hash/marker orchestration helpers.
