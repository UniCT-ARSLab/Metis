"""M6.3: FRESH-PROCESS reload check. Loads policy.keras exactly the way run.py does (via
core.policy_artifact.load_policy_model, which registers the Metis composite layer -- no manual residual
rebuild, no OpenArm knowledge) and verifies it reproduces the C1 implementation within 1e-6, with the
gate/bound/approach invariants. Run this as its OWN process.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.models import build_continuous_actor  # noqa: E402
from core.policy_artifact import load_policy_model  # noqa: E402  (registers the composite layer)
from core.settling_residual import build_settling_residual, gate, settling_action  # noqa: E402

POLICY = "checkpoints/openarm_reach_hold_m6_3_policy/policy.keras"
CLONE = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"
RESIDUAL = "checkpoints/openarm_reach_hold_m6_2_c1/residual_settling.weights.h5"
M5V2 = "python/demos/openarm_reach_hold_m5_v2/train.npz"
DELTA, OUTER, INNER = 0.02, 0.11, 0.04


def main():
    model, manifest, kind, path = load_policy_model(POLICY)
    print(f"loaded {kind} from {path} (no custom_objects, no residual rebuild)", flush=True)

    obs = np.asarray(np.load(M5V2, allow_pickle=True)["obs"], np.float32)
    obs = obs[np.random.default_rng(1).choice(len(obs), size=4096, replace=False)]

    comp = model(tf.convert_to_tensor(obs), training=False).numpy()
    clone = build_continuous_actor(obs_dim=27, action_size=7); clone.load_weights(CLONE)
    res = build_settling_residual(27, 7); res.load_weights(RESIDUAL)
    c1_clone = np.clip(clone(tf.convert_to_tensor(obs), training=False).numpy(), -1.0, 1.0)
    c1 = settling_action(c1_clone, res, obs, delta_max=DELTA, outer=OUTER, inner=INNER)[0].numpy()

    g = gate(np.linalg.norm(obs[:, 14:17], axis=1), outer=OUTER, inner=INNER).numpy()
    approach = g == 0.0
    checks = {
        "max_diff_vs_c1": float(np.max(np.abs(comp - c1))),
        "max_correction": float(np.max(np.abs(comp - c1_clone))),
        "approach_dev_gate0": float(np.max(np.abs(comp[approach] - c1_clone[approach]))) if approach.any() else 0.0,
        "finite_and_bounded": bool(np.all(np.isfinite(comp)) and np.all(np.abs(comp) <= 1.0 + 1e-6)),
    }
    ok = (checks["max_diff_vs_c1"] <= 1e-6 and checks["max_correction"] <= DELTA + 1e-6
          and checks["approach_dev_gate0"] == 0.0 and checks["finite_and_bounded"])
    print("RELOAD_VERIFY " + json.dumps({**checks, "PASS": ok}), flush=True)
    if not ok:
        raise SystemExit("reload verification FAILED")


if __name__ == "__main__":
    main()
