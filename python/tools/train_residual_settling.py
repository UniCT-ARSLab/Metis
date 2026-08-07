"""M6.2 Phase C1: train the gated bounded-settling RESIDUAL head (clone FROZEN).

Loss ONLY on the residual: on settling states fit gate(d)*delta_max*tanh(residual(obs)) to the
applied bounded correction; on M5/clone ANCHOR states fit it to ZERO (residual must not fire where
the clone is already fine). The gate makes the approach provably clone-identical, so the correction
is only learned in the near-target zone. Clone hash NEVER changes; the residual is saved separately.
No critic / Q-loss / PPO / SAC / TD3.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.settling_residual import (  # noqa: E402
    GATE_INNER, GATE_OUTER, build_settling_residual, settling_correction)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrective", default="python/demos/openarm_reach_hold_c1/settling_corrective.npz")
    ap.add_argument("--out", default="checkpoints/openarm_reach_hold_m6_2_c1/residual_settling.weights.h5")
    ap.add_argument("--gate-outer", type=float, default=GATE_OUTER)
    ap.add_argument("--gate-inner", type=float, default=GATE_INNER)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--anchor-fraction", type=float, default=0.5)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.corrective, allow_pickle=True)
    s_obs = np.asarray(d["obs"], np.float32)
    s_corr = np.asarray(d["correction"], np.float32)
    anchors = np.asarray(d["anchor_obs"], np.float32)
    delta_max = float(np.asarray(d["delta_max"])[0])
    if len(s_obs) == 0:
        raise SystemExit("empty settling corrective set; refuse to train")

    obs_dim, act = s_obs.shape[1], s_corr.shape[1]
    residual = build_settling_residual(obs_dim, act)
    opt = tf.keras.optimizers.Adam(args.learning_rate)

    # preflight: zero-init residual -> correction is exactly zero everywhere
    c0 = settling_correction(residual, s_obs[:256], delta_max=delta_max,
                             outer=args.gate_outer, inner=args.gate_inner).numpy()
    assert float(np.max(np.abs(c0))) == 0.0, "residual not zero-initialized"
    print(f"C1 train: settling {len(s_obs)} + anchors {len(anchors)}; delta_max {delta_max}; "
          f"gate [{args.gate_inner},{args.gate_outer}] obs-space", flush=True)

    n_anc = int(round(args.anchor_fraction * args.batch_size))
    n_set = args.batch_size - n_anc

    @tf.function
    def step(so, sc, ao):
        with tf.GradientTape() as g:
            cp = settling_correction(residual, so, delta_max=delta_max,
                                     outer=args.gate_outer, inner=args.gate_inner, training=True)
            l_set = tf.reduce_mean(tf.square(cp - sc))
            ap_ = settling_correction(residual, ao, delta_max=delta_max,
                                      outer=args.gate_outer, inner=args.gate_inner, training=True)
            l_anc = tf.reduce_mean(tf.square(ap_))
            loss = l_set + args.anchor_weight * l_anc
        opt.apply_gradients(zip(g.gradient(loss, residual.trainable_variables),
                                residual.trainable_variables))
        return loss, l_set, l_anc

    for u in range(1, args.updates + 1):
        si = rng.integers(len(s_obs), size=n_set)
        ai = rng.integers(len(anchors), size=n_anc)
        loss, l_set, l_anc = step(tf.convert_to_tensor(s_obs[si]), tf.convert_to_tensor(s_corr[si]),
                                  tf.convert_to_tensor(anchors[ai]))
        if u % 1000 == 0:
            print(f"  update {u}: loss {float(loss):.6f} (settle {float(l_set):.6f} anchor {float(l_anc):.6f})",
                  flush=True)

    # hard check: |correction| <= delta_max on a large sample (guaranteed by construction)
    chk = settling_correction(residual, s_obs, delta_max=delta_max,
                              outer=args.gate_outer, inner=args.gate_inner).numpy()
    max_corr = float(np.max(np.abs(chk)))
    assert max_corr <= delta_max + 1e-5, f"correction {max_corr} exceeds delta_max {delta_max}"
    residual.save_weights(args.out)
    result = {"out": args.out, "out_sha256": _sha256(args.out), "delta_max": delta_max,
              "gate_outer": args.gate_outer, "gate_inner": args.gate_inner,
              "n_settling": int(len(s_obs)), "n_anchors": int(len(anchors)),
              "max_correction": max_corr, "updates": args.updates}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=2))
    print("C1_TRAIN_DONE " + json.dumps({"out": args.out, "max_corr": round(max_corr, 5)}), flush=True)


if __name__ == "__main__":
    main()
