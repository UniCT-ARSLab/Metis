import argparse
import os
import sys
from pathlib import Path

from godot_process_manager import GodotProcessManager
from scenario_gym_env import ScenarioGymEnv


BACKENDS = {
    "dqn": "train_generic_dqn.py",
    "ddpg": "train_generic_ddpg.py",
    "ppo": "train_generic_ppo.py",
}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Unified Godot training entrypoint. With --algorithm auto it inspects "
            "the scenario action space and delegates to DQN for discrete actions, "
            "DDPG for continuous actions, or PPO for hybrid actions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algorithm", choices=["auto", "dqn", "ddpg", "ppo"], default="auto")
    parser.add_argument("--probe-port", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default=False)
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
            f"action_size={env.action_size} agents={env.agent_ids}",
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

    action_type = probe_action_type(args)
    if action_type == "discrete":
        return "dqn"
    if action_type == "continuous":
        return "ddpg"
    if action_type == "hybrid":
        return "ppo"
    raise RuntimeError(f"Unsupported action_type={action_type!r}; expected 'discrete', 'continuous' or 'hybrid'.")


def main():
    args = parse_args(sys.argv[1:])
    backend = select_backend(args)
    backend_script = Path(__file__).resolve().parent / BACKENDS[backend]
    backend_args = strip_frontend_args(sys.argv[1:])

    print(f"Using training backend: {backend} ({backend_script.name})", flush=True)
    os.execv(sys.executable, [sys.executable, str(backend_script)] + backend_args)


if __name__ == "__main__":
    main()
