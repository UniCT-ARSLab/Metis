"""Check that a lockstep tuning change does not alter simulation dynamics.

Replays one fixed, seeded action sequence under two Godot configurations and diffs
the observation trajectories. Pacing knobs (idle sleep, spin polls) change *when*
the bridge polls, never the physics dt, so their diff must be exactly 0.0. A knob
that changes dt (Engine.time_scale, Engine.physics_ticks_per_second) would instead
diverge slowly and compound -- which no throughput number would ever reveal, and
which would mean the policy learned physics that differ from evaluation.

Usage:
  python python/tools/bench_determinism.py --godot-bin /path/to/godot \
      --config 2000 0 --config 2000 64
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv

SEED = 42


def record_trajectory(args, sleep_usec, spin_polls):
    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
        scene_path=args.godot_scene,
    )
    manager.start_many(
        [args.port],
        headless=True,
        user_args=[
            f"--lockstep-idle-sleep-usec={int(sleep_usec)}",
            f"--lockstep-spin-polls={int(spin_polls)}",
        ],
    )
    try:
        env = ScenarioGymEnv(port=args.port)
        rng = np.random.default_rng(SEED)
        actions = [int(a) for a in rng.integers(env.num_actions, size=args.steps)]

        obs, _ = env.reset(seed=SEED)
        trajectory = [np.asarray(obs, dtype=np.float32)]
        for action in actions:
            obs, _, terminated, truncated, _ = env.step(action)
            trajectory.append(np.asarray(obs, dtype=np.float32))
            if terminated or truncated:
                # Re-seeding keeps both runs on the same episode boundaries.
                obs, _ = env.reset(seed=SEED)
                trajectory.append(np.asarray(obs, dtype=np.float32))
        env.close()
        return np.stack(trajectory)
    finally:
        manager.stop_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot-bin", required=True)
    parser.add_argument("--godot-project", default="godot")
    parser.add_argument("--godot-scene", default="res://scenarios/breakout/breakout_scenario.tscn")
    parser.add_argument("--port", type=int, default=6400)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--config",
        type=int,
        nargs=2,
        action="append",
        metavar=("SLEEP_USEC", "SPIN_POLLS"),
        required=True,
        help="A (idle_sleep_usec, spin_polls) pair. Pass twice: the first is the baseline.",
    )
    args = parser.parse_args()

    if len(args.config) < 2:
        parser.error("pass --config at least twice (baseline first)")

    baseline = None
    for sleep_usec, spin_polls in args.config:
        label = f"sleep={sleep_usec} spin={spin_polls}"
        trajectory = record_trajectory(args, sleep_usec, spin_polls)
        if baseline is None:
            baseline = trajectory
            print(f"{label:<24} baseline  shape={trajectory.shape}", flush=True)
            continue

        if trajectory.shape != baseline.shape:
            print(f"{label:<24} SHAPE MISMATCH {trajectory.shape} vs {baseline.shape} "
                  f"-- episodes diverged, dynamics are NOT preserved")
            continue

        diff = np.abs(trajectory - baseline)
        exceeds = np.argwhere(diff.max(axis=1) > 1e-5)
        first = int(exceeds[0][0]) if len(exceeds) else -1
        verdict = "IDENTICAL" if diff.max() == 0.0 else (
            "OK (<1e-6)" if diff.max() < 1e-6 else "DIVERGED")
        print(f"{label:<24} max_abs_diff={diff.max():.3e}  mean_abs_diff={diff.mean():.3e}  "
              f"first_step_over_1e-5={first}  -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
