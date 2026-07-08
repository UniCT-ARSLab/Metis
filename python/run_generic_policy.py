import argparse
import os
import platform
import random
import sys
import sysconfig
import time
from pathlib import Path


def configure_tensorflow_runtime():
    if platform.system() == "Linux":
        ensure_nvidia_pip_libs_on_path()


def ensure_nvidia_pip_libs_on_path():
    if os.environ.get("GODOT_GYM_TF_LD_READY") == "1":
        return

    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia_dir = purelib / "nvidia"
    if not nvidia_dir.exists():
        return

    lib_dirs = [
        str(package_dir / "lib")
        for package_dir in nvidia_dir.iterdir()
        if (package_dir / "lib").is_dir()
    ]
    if not lib_dirs:
        return

    current_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing_paths = [p for p in lib_dirs if p not in current_paths]
    os.environ["GODOT_GYM_TF_LD_READY"] = "1"
    if missing_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing_paths + current_paths)
        os.execv(sys.executable, [sys.executable] + sys.argv)


configure_tensorflow_runtime()

import numpy as np
import tensorflow as tf

from godot_process_manager import GodotProcessManager
from models import build_shared_q_network
from scenario_gym_env import ScenarioGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained DQN policy on any Godot BridgeServer scenario.")
    parser.add_argument("--weights-path", default="generic_dqn_weights.weights.h5")
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6200)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--connect-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def greedy_action(model, obs):
    q_values = model(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    return int(np.argmax(q_values)), q_values


def select_action(model, obs, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1), None
    return greedy_action(model, obs)


def select_multi_actions(model, obs_batch, epsilon, num_actions):
    actions = []
    q_values = []
    for obs in obs_batch:
        action, q = select_action(model, obs, epsilon, num_actions)
        actions.append(action)
        q_values.append(q)
    return np.asarray(actions, dtype=np.int32), q_values


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    manager = None
    if not args.connect_only:
        manager = GodotProcessManager(
            godot_bin=args.godot_bin,
            project_dir=args.godot_project,
        )
        manager.start_many([args.port], headless=args.headless)
        print(f"Started Godot on port {args.port}", flush=True)

    env = None
    try:
        env = ScenarioGymEnv(
            host=args.host,
            port=args.port,
            seed=args.seed,
            agent_id=args.agent_id,
            multi_agent=args.multi_agent,
        )
        model = build_shared_q_network(obs_dim=env.obs_dim, num_actions=env.num_actions)
        model.load_weights(args.weights_path)

        print(
            f"Loaded policy weights={args.weights_path} obs_dim={env.obs_dim} "
            f"num_actions={env.num_actions} agents={env.agent_ids}",
            flush=True,
        )

        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0

            for step in range(args.max_steps):
                if args.multi_agent:
                    action, _ = select_multi_actions(model, obs, args.epsilon, env.num_actions)
                else:
                    action, _ = select_action(model, obs, args.epsilon, env.num_actions)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

                if args.print_every > 0 and step % args.print_every == 0:
                    action_label = action.tolist() if hasattr(action, "tolist") else int(action)
                    print(
                        f"episode={episode:04d} step={step:04d} action={action_label} "
                        f"reward={reward:.4f} total={total_reward:.4f} "
                        f"terminated={terminated} truncated={truncated}",
                        flush=True,
                    )

                if args.delay > 0:
                    time.sleep(args.delay)

                if terminated or truncated:
                    break

            print(
                f"episode={episode:04d} finished total_reward={total_reward:.4f} steps={step + 1}",
                flush=True,
            )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if manager is not None:
            manager.stop_all()


if __name__ == "__main__":
    main()
