"""M6.3: package the M6.2-C1 composite (frozen clone + gated bounded settling residual) into ONE
self-contained Metis policy.keras + an immutable manifest, and export TFLite. Verifies in-process
parity vs the C1 implementation (max diff <= 1e-6), the gate/bound invariants, and TFLite parity.
The critic v6 is NOT part of the policy.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.composite_policy import build_composite_policy  # noqa: E402
from core.models import build_continuous_actor  # noqa: E402
from core.settling_residual import build_settling_residual, gate, settling_action  # noqa: E402

CLONE = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"
CLONE_MANIFEST = "checkpoints/openarm_reach_hold_m6_td3bc_final/policy.json"
RESIDUAL = "checkpoints/openarm_reach_hold_m6_2_c1/residual_settling.weights.h5"
DATASET = "python/demos/openarm_reach_hold_c1/settling_corrective.npz"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clone-weights", default=CLONE)
    ap.add_argument("--residual-weights", default=RESIDUAL)
    ap.add_argument("--delta-max", type=float, default=0.02)
    ap.add_argument("--gate-outer", type=float, default=0.11)
    ap.add_argument("--gate-inner", type=float, default=0.04)
    ap.add_argument("--out-dir", default="checkpoints/openarm_reach_hold_m6_3_policy")
    ap.add_argument("--m5v2", default="python/demos/openarm_reach_hold_m5_v2/train.npz")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    clone_manifest = json.loads(Path(CLONE_MANIFEST).read_text())
    obs_dim = int(clone_manifest["observation"]["size"]); act = int(clone_manifest["action"]["size"])

    clone = build_continuous_actor(obs_dim=obs_dim, action_size=act); clone.load_weights(args.clone_weights)
    residual = build_settling_residual(obs_dim, act); residual.load_weights(args.residual_weights)
    # keep a private C1-reference copy of the residual for parity (build_composite renames sub-models)
    ref_res = build_settling_residual(obs_dim, act); ref_res.load_weights(args.residual_weights)
    ref_clone = build_continuous_actor(obs_dim=obs_dim, action_size=act); ref_clone.load_weights(args.clone_weights)

    composite = build_composite_policy(clone, residual, delta_max=args.delta_max,
                                       gate_outer=args.gate_outer, gate_inner=args.gate_inner)
    policy_path = out / "policy.keras"
    composite.save(policy_path)
    print(f"saved {policy_path}", flush=True)

    # ---- in-process parity vs the C1 implementation + invariants ----
    m5 = np.load(args.m5v2, allow_pickle=True)
    obs = np.asarray(m5["obs"], np.float32)
    obs = obs[np.random.default_rng(0).choice(len(obs), size=4096, replace=False)]
    comp_a = composite(tf.convert_to_tensor(obs), training=False).numpy()
    c1_clone = np.clip(ref_clone(tf.convert_to_tensor(obs), training=False).numpy(), -1.0, 1.0)
    c1_a, c1_corr = settling_action(c1_clone, ref_res, obs, delta_max=args.delta_max,
                                    outer=args.gate_outer, inner=args.gate_inner)
    c1_a = c1_a.numpy()
    max_diff = float(np.max(np.abs(comp_a - c1_a)))
    max_corr = float(np.max(np.abs(comp_a - c1_clone)))
    g = gate(np.linalg.norm(obs[:, 14:17], axis=1), outer=args.gate_outer, inner=args.gate_inner).numpy()
    approach = g == 0.0
    approach_dev = float(np.max(np.abs(comp_a[approach] - c1_clone[approach]))) if approach.any() else 0.0
    finite = bool(np.all(np.isfinite(comp_a)) and np.all(np.abs(comp_a) <= 1.0 + 1e-6))
    checks = {"max_diff_vs_c1": max_diff, "max_correction": max_corr,
              "approach_dev_gate0": approach_dev, "finite_and_bounded": finite,
              "n_approach": int(approach.sum()), "n_settle": int((~approach).sum())}
    print("PARITY " + json.dumps(checks), flush=True)
    assert max_diff <= 1e-6, f"composite != C1 ({max_diff})"
    assert max_corr <= args.delta_max + 1e-6, f"correction {max_corr} > delta_max"
    assert approach_dev == 0.0, f"approach not clone-identical ({approach_dev})"
    assert finite, "non-finite or out-of-bound output"

    # ---- TFLite export + parity ----
    conv = tf.lite.TFLiteConverter.from_keras_model(composite)
    tfl = conv.convert()
    tfl_path = out / "policy.tflite"; tfl_path.write_bytes(tfl)
    interp = tf.lite.Interpreter(model_content=tfl); interp.allocate_tensors()
    inp_i = interp.get_input_details()[0]; out_i = interp.get_output_details()[0]
    tfl_out = []
    for i in range(0, 512):
        interp.set_tensor(inp_i["index"], obs[i:i + 1])
        interp.invoke()
        tfl_out.append(interp.get_tensor(out_i["index"])[0])
    tfl_diff = float(np.max(np.abs(np.asarray(tfl_out) - comp_a[:512])))
    print(f"TFLITE parity max_diff {tfl_diff:.2e}", flush=True)

    # ---- Metis policy.json manifest (same contract as the clone; algorithm td3_bc) ----
    manifest = dict(clone_manifest)
    manifest["model_file"] = "policy.keras"
    manifest["composite"] = {"kind": "frozen_clone + gated_settling_residual",
                             "delta_max": args.delta_max, "gate_outer": args.gate_outer,
                             "gate_inner": args.gate_inner}
    (out / "policy.json").write_text(json.dumps(manifest, indent=2))

    # ---- immutable manifest with hashes ----
    final = {}
    for k, f in (("final_clone", "checkpoints/openarm_reach_hold_m6_2_c1/final_clone.json"),
                 ("final_residual", "checkpoints/openarm_reach_hold_m6_2_c1/final_residual.json")):
        if Path(f).is_file():
            d = json.loads(Path(f).read_text())
            final[k] = {"success_rate": d.get("success_rate"), "collision_rate": d.get("collision_rate")}
    immutable = {
        "schema": "openarm-reach-hold-composite-policy/1",
        "algorithm": "td3_bc", "delta_max": args.delta_max,
        "gate": {"outer": args.gate_outer, "inner": args.gate_inner, "err_slice": [14, 17]},
        "observation_action_contract": {"obs": manifest["observation"], "action": manifest["action"]},
        "hashes": {
            "clone_weights": _sha256(args.clone_weights),
            "residual_weights": _sha256(args.residual_weights),
            "settling_dataset": _sha256(DATASET) if Path(DATASET).is_file() else None,
            "policy_keras": _sha256(str(policy_path)),
            "policy_tflite": _sha256(str(tfl_path)),
        },
        "final_results": final,
        "parity": checks, "tflite_parity_max_diff": tfl_diff,
        "critic_v6_included": False,
    }
    (out / "composite_manifest.json").write_text(json.dumps(immutable, indent=2, default=float))
    print("PACKAGE_DONE " + json.dumps({"policy": str(policy_path), "max_diff": max_diff,
          "tflite_diff": tfl_diff, "clone_hash": immutable["hashes"]["clone_weights"][:12]}), flush=True)


if __name__ == "__main__":
    main()
