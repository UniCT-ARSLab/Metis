"""Generic policy/action adapters for PPO.

PPO's action SEMANTICS live in one narrow seam: how a raw sampled continuous vector becomes the ENV
action, what gets STORED for the log-prob, and which states are TRAINABLE. Everything downstream (GAE,
clipped surrogate, optimizer, checkpoint) sees only scalar log_prob / advantage / return and never the
action meaning. An adapter encapsulates exactly that seam so the SAME PPO can drive either an absolute
action (standard) or a bounded RESIDUAL around a frozen base policy — without a second PPO.

Contract (all numpy at the select_action call site):
  transform(obs, raw_continuous) -> (env_continuous, stored_continuous, diagnostics)
      env_continuous   : what the environment receives
      stored_continuous: the LATENT the Gaussian log-prob is scored on (must match what the PPO
                         continuous head parameterises) -- for residual this is the raw pre-transform
                         sample, so PPO's existing gaussian_log_prob / evaluate_actions stay unchanged
  train_mask(obs) -> 1.0 | 0.0   (states that may drive the policy loss; value loss stays unmasked)
  trainable_extra() -> []        (extra trainable vars beyond the PPO model + log_std; none so far)

The standard adapter reproduces the pre-adapter behaviour EXACTLY (clip to the action bounds, mask 1),
so `--policy-mode standard` is bit-identical to the old PPO.
"""
import hashlib

import numpy as np


class StandardActionAdapter:
    """Absolute continuous action: env = clip(raw, low, high); store the clipped action; always train.
    Identical to the pre-adapter PPO continuous path."""

    mode = "standard"

    def __init__(self, low, high):
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)

    def transform(self, obs, raw_continuous):
        a = np.clip(np.asarray(raw_continuous, np.float32), self.low, self.high).astype(np.float32)
        return a, a, {}

    def train_mask(self, obs):
        return 1.0

    def trainable_extra(self):
        return []

    def describe(self):
        return {"mode": self.mode}


def _gate(err_norm, outer, inner):
    return float(np.clip((outer - err_norm) / (outer - inner), 0.0, 1.0))


class ResidualActionAdapter:
    """PPO controls a bounded residual around a FROZEN base policy:
        base   = base_policy(obs)                              # frozen, NOT trainable / checkpointed
        delta  = gate(||obs[err]||) * delta_max * tanh(raw)    # gate EXACTLY 0 in the approach zone
        env    = clip(base + delta, low, high)
    The STORED latent is `raw` (pre-tanh), so PPO's Gaussian log-prob scores the residual head directly
    (the tanh + gate + delta_max + base are a deterministic squash outside the scored distribution, the
    same role the plain clip plays in the standard adapter). train_mask = 1 only where the gate is open
    (update_mask="gate") so the policy loss never acts on approach states where the residual is 0.
    """

    mode = "residual"

    def __init__(self, base_act_fn, *, low, high, err_start, err_size, gate_outer, gate_inner,
                 delta_max, update_mask="gate", base_sha=None):
        self._base = base_act_fn                                # callable obs(np) -> base action(np), FROZEN
        self.low = np.asarray(low, np.float32); self.high = np.asarray(high, np.float32)
        self.err_start = int(err_start); self.err_size = int(err_size)
        self.gate_outer = float(gate_outer); self.gate_inner = float(gate_inner)
        self.delta_max = float(delta_max)
        self.update_mask = str(update_mask)
        self.base_sha = base_sha

    def gate(self, obs):
        o = np.asarray(obs, np.float64)
        err = float(np.linalg.norm(o[self.err_start:self.err_start + self.err_size]))
        return _gate(err, self.gate_outer, self.gate_inner)

    def base_action(self, obs):
        return np.asarray(self._base(obs), np.float32)

    def transform(self, obs, raw_continuous):
        base = self.base_action(obs); g = self.gate(obs)
        delta = g * self.delta_max * np.tanh(np.asarray(raw_continuous, np.float32))
        raw_env = base + delta
        env = np.clip(raw_env, self.low, self.high).astype(np.float32)
        diag = {"gate": g, "delta_absmax": float(np.max(np.abs(delta))) if delta.size else 0.0,
                "clipped": bool(np.any((raw_env < self.low) | (raw_env > self.high))),
                "approach_dev": float(np.max(np.abs(env - base))) if g == 0.0 else 0.0}
        return env, np.asarray(raw_continuous, np.float32), diag

    def train_mask(self, obs):
        if self.update_mask == "all":
            return 1.0
        return 1.0 if self.gate(obs) > 0.0 else 0.0             # "gate": only where the residual can act

    def trainable_extra(self):
        return []                                              # base is frozen; nothing extra to train

    def describe(self):
        return {"mode": self.mode, "delta_max": self.delta_max, "gate_outer": self.gate_outer,
                "gate_inner": self.gate_inner, "err_start": self.err_start, "err_size": self.err_size,
                "update_mask": self.update_mask, "base_sha": self.base_sha}


def sha_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_gate_config(spec):
    """--residual-gate-config: a JSON path or an inline `outer,inner,err_start,err_size` string.
    Returns a dict with those four keys. Generic (no task literals)."""
    import json
    from pathlib import Path
    if spec is None:
        raise ValueError("residual policy-mode requires --residual-gate-config")
    p = Path(spec)
    if p.exists():
        d = json.loads(p.read_text())
        return {"outer": float(d.get("gate_outer", d.get("outer"))),
                "inner": float(d.get("gate_inner", d.get("inner"))),
                "err_start": int(d.get("err_start")), "err_size": int(d.get("err_size", 3))}
    parts = [float(x) for x in str(spec).split(",")]
    return {"outer": parts[0], "inner": parts[1], "err_start": int(parts[2]),
            "err_size": int(parts[3]) if len(parts) > 3 else 3}
