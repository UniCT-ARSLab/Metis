import argparse
import os
import platform
import sys
import sysconfig
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

NO_TIME_LIMIT_STEPS = 2_147_483_647


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained Keras DQN policy in Godot.")
    parser.add_argument("--load-from", choices=["auto", "weights", "checkpoint"], default="auto")
    parser.add_argument("--weights-path", default="generic_dqn_weights.weights.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_dqn")
    parser.add_argument("--port", type=int, default=6400)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--connect-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--infinite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def iter_episode_numbers(args):
    episode = 0
    while args.infinite or episode < args.episodes:
        yield episode
        episode += 1


def configured_max_steps(args):
    if args.time_limit:
        return args.max_steps
    return NO_TIME_LIMIT_STEPS


def load_policy(model, args):
    weights_path = Path(args.weights_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    latest_checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))

    if args.load_from == "weights" or (args.load_from == "auto" and weights_path.exists()):
        model.load_weights(str(weights_path))
        print(f"Loaded weights: {weights_path}", flush=True)
        return

    if args.load_from == "checkpoint" or (args.load_from == "auto" and latest_checkpoint):
        if not latest_checkpoint:
            raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"Loaded checkpoint: {latest_checkpoint}", flush=True)
        return

    raise RuntimeError(
        f"No model found. Checked weights={weights_path} and checkpoint_dir={checkpoint_dir}"
    )


def select_action(model, obs, epsilon, num_actions):
    if epsilon > 0.0 and np.random.random() < epsilon:
        return int(np.random.randint(0, num_actions))

    q_values = model(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    return int(np.argmax(q_values))


def select_actions(model, obs_batch, active_mask, epsilon, num_actions):
    actions = np.zeros((obs_batch.shape[0],), dtype=np.int32)
    q_values = model(obs_batch, training=False).numpy()

    for idx in range(obs_batch.shape[0]):
        if not active_mask[idx]:
            actions[idx] = 0
        elif epsilon > 0.0 and np.random.random() < epsilon:
            actions[idx] = int(np.random.randint(0, num_actions))
        else:
            actions[idx] = int(np.argmax(q_values[idx]))
    return actions


def make_env(args):
    return ScenarioGymEnv(
        host=args.host,
        port=args.port,
        seed=args.seed,
        timeout=args.env_timeout,
        agent_id=args.agent_id,
        multi_agent=args.multi_agent,
    )


def main():
    args = parse_args()
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}", flush=True)

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
        env = make_env(args)
        if args.multi_agent:
            print(
                f"Scenario spec: agent_ids={env.agent_ids} obs_dim={env.obs_dim} num_actions={env.num_actions} actions={env.action_names}",
                flush=True,
            )
        else:
            print(
                f"Scenario spec: agent_id={env.agent_id} obs_dim={env.obs_dim} num_actions={env.num_actions} actions={env.action_names}",
                flush=True,
            )

        env_max_steps = configured_max_steps(args)
        env.configure(max_steps=env_max_steps)

        model = build_shared_q_network(obs_dim=env.obs_dim, num_actions=env.num_actions)
        load_policy(model, args)

        if args.infinite:
            print("Running indefinitely. Stop with Ctrl+C.", flush=True)
        if not args.time_limit:
            print("Episode time limit disabled; matches end only on terminal game state.", flush=True)

        for episode in iter_episode_numbers(args):
            obs, info = env.reset(seed=args.seed + episode)
            if args.multi_agent:
                active_mask = np.logical_not(np.asarray(info["per_agent_done"], dtype=np.bool_))
                total_reward = np.zeros((len(info["agent_ids"]),), dtype=np.float32)
            else:
                total_reward = 0.0
            step = 0

            while True:
                if args.multi_agent:
                    actions = select_actions(model, obs, active_mask, args.epsilon, env.num_actions)
                    obs, reward, terminated, truncated, info = env.step(actions)
                    per_agent_rewards = np.asarray(info["per_agent_rewards"], dtype=np.float32)
                    active_mask = np.logical_not(np.asarray(info["per_agent_done"], dtype=np.bool_))
                    total_reward += per_agent_rewards
                else:
                    action = select_action(model, obs, args.epsilon, env.num_actions)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                step += 1

                if terminated or truncated or (args.time_limit and step >= args.max_steps):
                    break

            if args.multi_agent:
                print(
                    f"episode={episode:04d} steps={step} mean_reward={float(np.mean(total_reward)):.3f} total_reward={total_reward.tolist()} info={info}",
                    flush=True,
                )
            else:
                print(
                    f"episode={episode:04d} steps={step} total_reward={total_reward:.3f} info={info}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
    finally:
        if env is not None:
            env.close()
        if manager is not None:
            manager.stop_all()


if __name__ == "__main__":
    main()
