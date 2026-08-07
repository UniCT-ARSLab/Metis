"""M6.2 Phase B: BC fine-tune of the FROZEN clone on M5 v2 + a SafeDAgger corrective dataset.

Supervised regression actor(obs) -> target_action (MSE). Starts from the clone (actor_bc_final) and
fine-tunes. Each batch mixes >=80% M5 v2 and <=20% corrections (the corrections are already weighted
toward the hard cells because that is where the clone deviates). M5 v2 and the clone are NEVER
modified; the candidate is saved SEPARATELY.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf  # noqa: E402

from core.models import build_continuous_actor  # noqa: E402

M5V2 = "python/demos/openarm_reach_hold_m5_v2/train.npz"
CLONE = "checkpoints/openarm_reach_hold_m6_td3bc_final/actor_bc_final.weights.h5"
HARD_CELLS = (25, 41, 31, 28)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m5v2", default=M5V2)
    ap.add_argument("--corrections", required=True, help="SafeDAgger corrective npz (obs, actions[, cells])")
    ap.add_argument("--clone-weights", default=CLONE)
    ap.add_argument("--out", default="checkpoints/openarm_reach_hold_m6_2_bc/actor_bc_dagger_r1.weights.h5")
    ap.add_argument("--updates", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--correction-fraction", type=float, default=0.20, help="<=0.20 per the constraint")
    ap.add_argument("--hard-cell-weight", type=float, default=2.0,
                    help="oversampling weight for corrections on cells 25/41/31/28 (1.0 = uniform).")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    assert args.correction_fraction <= 0.20 + 1e-9, "corrections must be <=20% of BC batches"
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    m5 = np.load(args.m5v2, allow_pickle=True)
    m5_obs = np.asarray(m5["obs"], np.float32)
    m5_act = np.asarray(m5["actions"], np.float32)
    cor = np.load(args.corrections, allow_pickle=True)
    c_obs = np.asarray(cor["obs"], np.float32)
    c_act = np.asarray(cor["actions"], np.float32)
    if len(c_obs) == 0:
        raise SystemExit("empty corrective dataset; refuse to fine-tune")
    # correction sampling weights: oversample the hard cells if a 'cells' field is present
    if "cells" in cor.files and args.hard_cell_weight != 1.0:
        cells = np.asarray(cor["cells"]).astype(int)
        w = np.where(np.isin(cells, HARD_CELLS), args.hard_cell_weight, 1.0).astype(np.float64)
        c_w = w / w.sum()
    else:
        c_w = None

    actor = build_continuous_actor(obs_dim=m5_obs.shape[1], action_size=m5_act.shape[1])
    actor.load_weights(args.clone_weights)                       # fine-tune FROM the clone
    opt = tf.keras.optimizers.Adam(args.learning_rate)

    n_cor = max(1, int(round(args.correction_fraction * args.batch_size)))
    n_m5 = args.batch_size - n_cor

    @tf.function
    def step(obs, tgt):
        with tf.GradientTape() as g:
            pred = actor(obs, training=True)
            loss = tf.reduce_mean(tf.square(pred - tgt))
        opt.apply_gradients(zip(g.gradient(loss, actor.trainable_variables), actor.trainable_variables))
        return loss

    print(f"BC fine-tune: M5v2 {len(m5_obs)} + corrections {len(c_obs)} (hard-cell weighted={c_w is not None}); "
          f"batch {n_m5} M5 / {n_cor} cor; updates {args.updates}", flush=True)
    for u in range(1, args.updates + 1):
        mi = rng.integers(len(m5_obs), size=n_m5)
        ci = rng.choice(len(c_obs), size=n_cor, p=c_w)
        obs = np.concatenate([m5_obs[mi], c_obs[ci]]).astype(np.float32)
        tgt = np.concatenate([m5_act[mi], c_act[ci]]).astype(np.float32)
        loss = step(tf.convert_to_tensor(obs), tf.convert_to_tensor(tgt))
        if u % 1000 == 0:
            print(f"  update {u}: mse {float(loss.numpy()):.5f}", flush=True)

    actor.save_weights(args.out)
    result = {"out": args.out, "out_sha256": _sha256(args.out),
              "m5v2": args.m5v2, "corrections": args.corrections,
              "n_m5": int(len(m5_obs)), "n_corrections": int(len(c_obs)),
              "correction_fraction": args.correction_fraction, "updates": args.updates,
              "clone_weights": args.clone_weights, "clone_sha256": _sha256(args.clone_weights)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=2))
    print("BC_FINETUNE_DONE " + json.dumps({"out": args.out, "sha": result["out_sha256"][:12]}), flush=True)


if __name__ == "__main__":
    main()
