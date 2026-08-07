#!/usr/bin/env python3
"""Convert IK-recorder JSON demos (from godot/tools/ik_demo_recorder.gd) into a
single Metis demonstration .npz (keys: obs, actions, rewards, next_obs, dones).

Usage:
    convert_ik_demos.py [IN_DIR_OR_GLOB] [OUT_NPZ]

Defaults to the Godot user:// demo dir and python/demos/ik_demos.npz.
Feed the result to training with --demo-path python/demos/ik_demos.npz.
"""
import glob
import json
import os
import sys

import numpy as np

KEYS = ["obs", "actions", "rewards", "next_obs", "dones"]
DEFAULT_IN = os.path.expanduser(
    "~/snap/code/254/.local/share/godot/app_userdata/Metis/ik_demos"
)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "demos", "ik_demos.npz")


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    files = sorted(glob.glob(src if os.path.splitext(src)[1] else os.path.join(src, "*.json")))
    if not files:
        print(f"No JSON demos found at {src!r}")
        sys.exit(1)

    acc = {k: [] for k in KEYS}
    for path in files:
        with open(path) as handle:
            demo = json.load(handle)
        missing = [k for k in KEYS if k not in demo]
        if missing:
            print(f"  skip {os.path.basename(path)}: missing {missing}")
            continue
        for k in KEYS:
            acc[k].extend(demo[k])
        print(f"  {os.path.basename(path)}: {len(demo['obs'])} transitions")

    arrays = {
        "obs": np.asarray(acc["obs"], dtype=np.float32),
        "actions": np.asarray(acc["actions"], dtype=np.float32),
        "rewards": np.asarray(acc["rewards"], dtype=np.float32),
        "next_obs": np.asarray(acc["next_obs"], dtype=np.float32),
        "dones": np.asarray(acc["dones"], dtype=np.float32),
    }
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(
        f"saved {out}\n  obs{arrays['obs'].shape} actions{arrays['actions'].shape} "
        f"rewards[{arrays['rewards'].min():.3f},{arrays['rewards'].max():.3f}] "
        f"dones_sum={int(arrays['dones'].sum())}"
    )


if __name__ == "__main__":
    main()
