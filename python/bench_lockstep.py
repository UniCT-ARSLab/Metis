"""Measure Godot<->Python bridge round-trip latency in isolation.

No TensorFlow, no replay buffer, no learner: just the bridge. Actions come from a
seeded RNG, so a run is reproducible and the only cost measured is the transport
plus Godot's own step. Instance startup happens before the timing window opens,
which is what makes the aggregate number trustworthy -- driving this from the
training logs instead gives per-episode windows too short to be a throughput.

Usage:
  python python/bench_lockstep.py --godot-bin /path/to/godot --num-envs 1 4 8
"""

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from godot_process_manager import GodotProcessManager
from scenario_gym_env import ScenarioGymEnv

WARMUP_STEPS = 200


def _proc_cpu_seconds(pid):
    """utime+stime for a pid, in seconds. Returns None if the process is gone."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    ticks = 100.0  # os.sysconf("SC_CLK_TCK"), constant at 100 on Linux
    return (int(fields[11]) + int(fields[12])) / ticks


def drive_env(env, steps, rng_seed, latencies):
    """Step one env with a fixed random action sequence, recording latencies."""
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    env.reset(seed=rng_seed)
    for step in range(steps):
        action = int(rng.integers(env.num_actions))
        started = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        elapsed = time.perf_counter() - started
        if step >= WARMUP_STEPS:
            latencies.append(elapsed)
        if terminated or truncated:
            env.reset(seed=rng_seed)


def run_cell(args, num_envs, sleep_usec, spin_polls):
    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
        scene_path=args.godot_scene,
    )
    ports = [args.base_port + i for i in range(num_envs)]
    user_args = [
        f"--lockstep-idle-sleep-usec={int(sleep_usec)}",
        f"--lockstep-spin-polls={int(spin_polls)}",
    ]
    manager.start_many(ports, headless=True, user_args=user_args)

    envs = [ScenarioGymEnv(port=port) for port in ports]
    pids = [item["proc"].pid for item in manager.processes]

    # Startup is deliberately outside the window: including it makes the
    # denominator depend on num_envs and the aggregate number meaningless.
    cpu_before = [_proc_cpu_seconds(pid) for pid in pids]
    per_env_latencies = [[] for _ in envs]
    threads = [
        threading.Thread(
            target=drive_env,
            args=(env, args.steps, 1000 + idx, per_env_latencies[idx]),
            daemon=True,
        )
        for idx, env in enumerate(envs)
    ]

    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    cpu_after = [_proc_cpu_seconds(pid) for pid in pids]

    for env in envs:
        env.close()
    manager.stop_all()

    timed_steps = sum(len(lat) for lat in per_env_latencies)
    latencies = sorted(lat for env_lat in per_env_latencies for lat in env_lat)
    godot_cpu = sum(
        (after - before)
        for before, after in zip(cpu_before, cpu_after)
        if before is not None and after is not None
    )

    def pct(fraction):
        if not latencies:
            return float("nan")
        return latencies[min(int(len(latencies) * fraction), len(latencies) - 1)] * 1e3

    return {
        "num_envs": num_envs,
        "sleep_usec": sleep_usec,
        "spin_polls": spin_polls,
        "aggregate": timed_steps / wall,
        "per_env": timed_steps / wall / num_envs,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        # Cores' worth of CPU the Godot instances burned, summed. With
        # sleep=0 each instance spins, so this is what a low sleep costs.
        "godot_cores": godot_cpu / wall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot-bin", required=True)
    parser.add_argument("--godot-project", default="godot")
    parser.add_argument("--godot-scene", default="res://scenarios/breakout/breakout_scenario.tscn")
    parser.add_argument("--base-port", type=int, default=6300)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--idle-sleep-usec", type=int, nargs="+", default=[2000, 0])
    parser.add_argument("--spin-polls", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--steps",
        type=int,
        default=1200,
        help=f"Steps per env, of which the first {WARMUP_STEPS} are discarded.",
    )
    args = parser.parse_args()

    if args.steps <= WARMUP_STEPS:
        parser.error(f"--steps must exceed the {WARMUP_STEPS}-step warmup")

    print(f"{'envs':>4} {'sleep':>6} {'spin':>5} {'agg/s':>8} {'per-env/s':>10} "
          f"{'p50ms':>7} {'p90ms':>7} {'p99ms':>7} {'godot_cores':>12}")
    for spin_polls in args.spin_polls:
        for sleep_usec in args.idle_sleep_usec:
            for num_envs in args.num_envs:
                row = run_cell(args, num_envs, sleep_usec, spin_polls)
                print(f"{row['num_envs']:>4} {row['sleep_usec']:>6} {row['spin_polls']:>5} "
                      f"{row['aggregate']:>8.1f} {row['per_env']:>10.1f} {row['p50']:>7.2f} "
                      f"{row['p90']:>7.2f} {row['p99']:>7.2f} {row['godot_cores']:>12.2f}",
                      flush=True)


if __name__ == "__main__":
    main()
