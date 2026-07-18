import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


BACKENDS = {
    "dqn": "algorithms.dqn",
    "ddpg": "algorithms.ddpg",
    "ddpg_bc": "algorithms.ddpg_bc",
    "ddpgfd": "algorithms.ddpgfd",
    "td3": "algorithms.td3",
    "td3_bc": "algorithms.td3_bc",
    "sac": "algorithms.sac",
    "ppo": "algorithms.ppo",
}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Unified Godot training entrypoint. With --algorithm auto it inspects "
            "the scenario action space and delegates to DQN for discrete actions, "
            "DDPG for continuous actions, or PPO for hybrid actions. SAC can be "
            "selected explicitly for continuous action spaces, as can TD3 and the "
            "demonstration-aware deterministic variants."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        choices=["auto", "dqn", "ddpg", "ddpg_bc", "ddpgfd", "td3", "td3_bc", "sac", "ppo"],
        default="auto",
    )
    parser.add_argument("--probe-port", type=int, default=None)
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Warm-start from a Metis .keras/.h5 policy or a legacy .weights.h5 file.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run Godot without a window. Training instances otherwise spin up the full "
            "renderer and contend with TensorFlow for the GPU, and only headless instances "
            "get --fixed-fps, without which physics stays gated to wall-clock 60Hz. Use "
            "--no-headless to watch, and expect a large throughput drop."
        ),
    )
    parser.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collector-mode", choices=["sync", "async"], default="async")
    parser.add_argument("--async-queue-capacity", type=int, default=256)
    parser.add_argument("--async-policy-sync-steps", type=int, default=100)
    parser.add_argument("--async-policy-publish-updates", type=int, default=100)
    parser.add_argument("--async-updates-per-step", type=int, default=1)
    parser.add_argument(
        "--async-update-basis",
        choices=["transitions", "env_steps"],
        default="transitions",
    )
    parser.add_argument(
        "--async-update-every",
        "--async-update-every-steps",
        dest="async_update_every",
        type=int,
        default=4,
    )
    parser.add_argument("--async-max-updates-per-env-step", type=int, default=1)
    parser.add_argument("--async-drain-max-events", type=int, default=64)
    parser.add_argument("--async-replay-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-env-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-env-count", type=int, default=None)
    parser.add_argument("--gpu-memory-growth", action=argparse.BooleanOptionalAction, default=True)
    args, _ = parser.parse_known_args(argv)
    return args


def strip_frontend_args(argv):
    stripped = []
    skip_next = False
    value_options = {"--algorithm", "--probe-port"}

    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if any(arg.startswith(option + "=") for option in value_options):
            continue
        if arg in value_options:
            skip_next = True
            continue
        stripped.append(arg)

    return stripped


def has_bool_option(argv, option):
    negative = "--no-" + option.removeprefix("--")
    return any(arg == option or arg == negative for arg in argv)


def probe_action_type(args):
    probe_port = args.probe_port
    if probe_port is None:
        probe_port = args.base_port + max(args.num_envs, 1)

    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
        scene_path=args.godot_scene,
    )
    env = None
    try:
        manager.start_many([probe_port], headless=True, debug=args.godot_debug)
        env = ScenarioGymEnv(
            port=probe_port,
            seed=args.env_seed_base,
            timeout=args.env_timeout,
            agent_id=args.agent_id,
            multi_agent=args.multi_agent,
        )
        print(
            f"Detected scenario: action_type={env.action_type} obs_dim={env.obs_dim} "
            f"action_size={env.action_size} {env.agent_summary()}",
            flush=True,
        )
        return env.action_type
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        manager.stop_all()


def select_backend(args):
    if args.algorithm != "auto":
        return args.algorithm

    policy_algorithm = policy_manifest_algorithm(args.policy_path)
    if policy_algorithm:
        if policy_algorithm not in BACKENDS:
            raise RuntimeError(f"Unsupported algorithm={policy_algorithm!r} in policy manifest")
        print(f"Detected policy algorithm from manifest: {policy_algorithm}", flush=True)
        return policy_algorithm

    action_type = probe_action_type(args)
    if action_type == "discrete":
        return "dqn"
    if action_type == "continuous":
        return "ddpg"
    if action_type == "hybrid":
        return "ppo"
    raise RuntimeError(f"Unsupported action_type={action_type!r}; expected 'discrete', 'continuous' or 'hybrid'.")


def policy_manifest_algorithm(policy_path):
    if not policy_path:
        return None
    path = Path(policy_path)
    manifest_path = path / "policy.json" if path.is_dir() else path.with_name("policy.json")
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read policy manifest: {manifest_path}") from exc
    if payload.get("format") != "metis-policy":
        raise RuntimeError(f"Unsupported policy manifest format: {manifest_path}")
    return payload.get("algorithm")


def main():
    args = parse_args(sys.argv[1:])
    try:
        backend = select_backend(args)
    except KeyboardInterrupt:
        print("\nInterrupted while probing the Godot scenario; processes were closed.", flush=True)
        return
    backend_args = strip_frontend_args(sys.argv[1:])
    if not has_bool_option(backend_args, "--headless") and args.headless:
        backend_args.append("--headless")
    if not has_bool_option(backend_args, "--godot-debug") and args.godot_debug:
        backend_args.append("--godot-debug")

    backend_module_name = BACKENDS[backend]
    print(f"Using training backend: {backend} ({backend_module_name})", flush=True)

    # Keep orig_argv untouched: CUDA relaunches must return through this public CLI.
    sys.argv = [sys.argv[0], *backend_args]
    backend_module = importlib.import_module(backend_module_name)
    backend_module.main()


if __name__ == "__main__":
    main()
