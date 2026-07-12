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
from models import build_continuous_actor, build_sac_actor, build_shared_q_network
from scenario_gym_env import ScenarioGymEnv

NO_TIME_LIMIT_STEPS = 2_147_483_647


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained policy on any Godot BridgeServer scenario.")
    parser.add_argument("--algorithm", choices=["auto", "dqn", "ddpg", "sac"], default="auto")
    parser.add_argument("--load-from", choices=["auto", "weights", "checkpoint"], default="auto")
    parser.add_argument("--weights-path", default="generic_dqn_weights.weights.h5")
    parser.add_argument("--actor-weights-path", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic")
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6200)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--infinite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--connect-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-every", type=int, default=1)
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


def scale_action_numpy(raw_action, low, high):
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def select_continuous_action(model, obs, env, algorithm):
    output = model(np.expand_dims(obs, axis=0), training=False)
    if algorithm == "sac":
        raw_action = tf.tanh(output[0]).numpy()[0]
    else:
        raw_action = output.numpy()[0]
    return np.clip(scale_action_numpy(raw_action, env.action_low, env.action_high), env.action_low, env.action_high).astype(np.float32)


def select_multi_continuous_actions(model, obs_batch, env, algorithm):
    actions = [
        select_continuous_action(model, obs, env, algorithm)
        for obs in np.asarray(obs_batch, dtype=np.float32)
    ]
    return np.asarray(actions, dtype=np.float32)


def resolve_algorithm(requested, env, weights_path):
    if requested != "auto":
        return requested
    if env.action_type == "discrete":
        return "dqn"
    if env.action_type == "continuous":
        if "sac" in Path(weights_path).name.lower():
            return "sac"
        return "ddpg"
    raise RuntimeError(f"run_generic_policy.py does not support action_type={env.action_type!r}")


def policy_weights_path(args, algorithm):
    if algorithm in {"ddpg", "sac"} and args.actor_weights_path:
        return args.actor_weights_path
    return args.weights_path


def load_policy(model, args, algorithm):
    weights_path = Path(policy_weights_path(args, algorithm))
    checkpoint_dir = Path(args.checkpoint_dir)
    latest_checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))

    if args.load_from == "weights" or (args.load_from == "auto" and weights_path.exists()):
        model.load_weights(str(weights_path))
        print(f"Loaded policy weights: {weights_path}", flush=True)
        return

    if args.load_from == "checkpoint" or (args.load_from == "auto" and latest_checkpoint):
        if not latest_checkpoint:
            raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")
        if algorithm == "dqn":
            checkpoint = tf.train.Checkpoint(model=model)
        elif algorithm in {"ddpg", "sac"}:
            checkpoint = tf.train.Checkpoint(actor=model)
        else:
            raise RuntimeError(f"Unsupported checkpoint load for algorithm={algorithm!r}")
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"Loaded policy checkpoint: {latest_checkpoint}", flush=True)
        return

    raise RuntimeError(
        f"No policy found. Checked weights={weights_path} and checkpoint_dir={checkpoint_dir}"
    )


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
            scene_path=args.godot_scene,
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
        algorithm = resolve_algorithm(args.algorithm, env, args.actor_weights_path or args.weights_path)
        if algorithm == "dqn":
            model = build_shared_q_network(obs_dim=env.obs_dim, num_actions=env.num_actions)
        elif algorithm == "ddpg":
            model = build_continuous_actor(obs_dim=env.obs_dim, action_size=env.action_size)
        elif algorithm == "sac":
            model = build_sac_actor(obs_dim=env.obs_dim, action_size=env.action_size)
        else:
            raise RuntimeError(f"Unsupported algorithm={algorithm!r}")
        load_policy(model, args, algorithm)

        env_max_steps = configured_max_steps(args)
        env.configure(max_steps=env_max_steps)

        print(
            f"Running policy algorithm={algorithm} obs_dim={env.obs_dim} "
            f"action_type={env.action_type} action_size={env.action_size} agents={env.agent_ids}",
            flush=True,
        )
        if args.infinite:
            print("Running indefinitely. Stop with Ctrl+C.", flush=True)
        if not args.time_limit:
            print("Episode time limit disabled; episodes end only on terminal scenario state.", flush=True)

        for episode in iter_episode_numbers(args):
            obs, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            if args.multi_agent:
                total_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)

            for step in range(env_max_steps):
                if env.action_type == "continuous":
                    if args.multi_agent:
                        action = select_multi_continuous_actions(model, obs, env, algorithm)
                    else:
                        action = select_continuous_action(model, obs, env, algorithm)
                elif args.multi_agent:
                    action, _ = select_multi_actions(model, obs, args.epsilon, env.num_actions)
                else:
                    action, _ = select_action(model, obs, args.epsilon, env.num_actions)

                obs, reward, terminated, truncated, info = env.step(action)
                if args.multi_agent:
                    total_reward += np.asarray(info.get("per_agent_rewards", []), dtype=np.float32)
                else:
                    total_reward += float(reward)

                if args.print_every > 0 and step % args.print_every == 0:
                    action_label = action.tolist() if hasattr(action, "tolist") else int(action)
                    total_label = total_reward.tolist() if hasattr(total_reward, "tolist") else f"{total_reward:.4f}"
                    print(
                        f"episode={episode:04d} step={step:04d} action={action_label} "
                        f"reward={reward:.4f} total={total_label} "
                        f"terminated={terminated} truncated={truncated}",
                        flush=True,
                    )

                if args.delay > 0:
                    time.sleep(args.delay)

                if terminated or truncated:
                    break

            total_label = total_reward.tolist() if hasattr(total_reward, "tolist") else f"{total_reward:.4f}"
            print(
                f"episode={episode:04d} finished total_reward={total_label} steps={step + 1}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
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
