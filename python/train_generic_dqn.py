import argparse
import os
import platform
import random
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
from replay_buffer import ReplayBuffer
from scenario_gym_env import ScenarioGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Generic DQN trainer for Godot scenarios using BridgeServer.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-update-every", type=int, default=20)
    parser.add_argument("--replay-warmup", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--weights-path", default="generic_dqn_weights.weights.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_dqn")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def describe_tensorflow_backend():
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}", flush=True)
    if platform.system() == "Darwin":
        print(f"macOS machine: {platform.machine()}", flush=True)


def select_action(model, obs, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    q = model(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    return int(np.argmax(q))


def train_step(model, target_model, optimizer, buffer, batch_size, gamma):
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)

    next_q = target_model(next_obs, training=False).numpy()
    max_next_q = np.max(next_q, axis=1)
    targets = rewards + (1.0 - dones) * gamma * max_next_q

    with tf.GradientTape() as tape:
        q_values = model(obs, training=True)
        action_mask = tf.one_hot(actions, q_values.shape[-1])
        q_selected = tf.reduce_sum(q_values * action_mask, axis=1)
        loss = tf.reduce_mean(tf.square(targets - q_selected))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy())


def main():
    args = parse_args()
    describe_tensorflow_backend()

    ports = [args.base_port + i for i in range(args.num_envs)]
    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
    )
    manager.start_many(ports, headless=args.headless)
    print(f"Started Godot instances on ports {ports}", flush=True)

    envs = []
    try:
        envs = [
            ScenarioGymEnv(
                port=port,
                seed=args.env_seed_base + idx,
                timeout=args.env_timeout,
                agent_id=args.agent_id,
            )
            for idx, port in enumerate(ports)
        ]

        obs_dim = envs[0].obs_dim
        num_actions = envs[0].num_actions
        agent_id = envs[0].agent_id
        print(
            f"Scenario spec: agent_id={agent_id} obs_dim={obs_dim} num_actions={num_actions} actions={envs[0].action_names}",
            flush=True,
        )

        for env in envs[1:]:
            if env.obs_dim != obs_dim or env.num_actions != num_actions:
                raise RuntimeError("All parallel environments must expose the same obs_dim and num_actions")

        model = build_shared_q_network(obs_dim=obs_dim, num_actions=num_actions)
        target_model = build_shared_q_network(obs_dim=obs_dim, num_actions=num_actions)
        target_model.set_weights(model.get_weights())
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        buffer = ReplayBuffer(capacity=args.replay_capacity)
        epsilon = args.epsilon_start
        start_episode = 0

        checkpoint = tf.train.Checkpoint(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            episode=tf.Variable(0, dtype=tf.int64),
            epsilon=tf.Variable(args.epsilon_start, dtype=tf.float32),
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.checkpoint_dir,
            max_to_keep=args.keep_checkpoints,
        )
        if args.resume and checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            epsilon = float(checkpoint.epsilon.numpy())
            print(
                f"Resumed checkpoint {checkpoint_manager.latest_checkpoint} from episode={start_episode} epsilon={epsilon:.3f}",
                flush=True,
            )

        for episode in range(start_episode, args.num_episodes):
            env_states = []
            for env_idx, env in enumerate(envs):
                obs, info = env.reset(seed=args.episode_seed_multiplier * episode + env_idx)
                env_states.append({
                    "obs": obs,
                    "done": False,
                    "ep_reward": 0.0,
                })

            losses = []
            for step_idx in range(args.max_steps_per_episode):
                if all(state["done"] for state in env_states):
                    break

                for env, state in zip(envs, env_states):
                    if state["done"]:
                        continue

                    action = select_action(model, state["obs"], epsilon, num_actions)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)

                    buffer.add(
                        state["obs"],
                        int(action),
                        float(reward),
                        next_obs,
                        done,
                    )

                    state["obs"] = next_obs
                    state["done"] = done
                    state["ep_reward"] += float(reward)

                    if len(buffer) >= args.replay_warmup:
                        loss = train_step(model, target_model, optimizer, buffer, args.batch_size, args.gamma)
                        losses.append(loss)

            if (episode + 1) % args.target_update_every == 0:
                target_model.set_weights(model.get_weights())

            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
            rewards_summary = [state["ep_reward"] for state in env_states]
            mean_loss = float(np.mean(losses)) if losses else 0.0
            print(
                f"episode={episode:04d} epsilon={epsilon:.3f} mean_loss={mean_loss:.5f} rewards={rewards_summary}",
                flush=True,
            )

            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                checkpoint.episode.assign(episode + 1)
                checkpoint.epsilon.assign(epsilon)
                saved_path = checkpoint_manager.save(checkpoint_number=episode + 1)
                print(f"Saved checkpoint: {saved_path}", flush=True)

        checkpoint.episode.assign(args.num_episodes)
        checkpoint.epsilon.assign(epsilon)
        saved_path = checkpoint_manager.save(checkpoint_number=args.num_episodes)
        print(f"Saved final checkpoint: {saved_path}", flush=True)
        model.save_weights(args.weights_path)
        print(f"Saved weights: {args.weights_path}", flush=True)
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        manager.stop_all()


if __name__ == "__main__":
    main()
