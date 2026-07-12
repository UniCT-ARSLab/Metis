import argparse
import os
import platform
import random
import sys
import sysconfig
from pathlib import Path


def configure_tensorflow_runtime():
    system = platform.system()
    if system == "Linux":
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
from replay_buffer import ReplayBuffer
from models import build_shared_q_network
from team_battle_gym_env import TeamBattleGymEnv

NUM_ENVS = 4
BASE_PORT = 6200
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 400
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-3
TARGET_UPDATE_EVERY = 20
REPLAY_WARMUP = 500
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
OBS_DIM = 18
NUM_AGENTS = 2
NUM_ACTIONS = 8
REPLAY_CAPACITY = 100000
ENV_SEED_BASE = 100
EPISODE_SEED_MULTIPLIER = 1000
WEIGHTS_PATH = "shared_dqn_weights.weights.h5"
ENV_TIMEOUT = 30.0
CHECKPOINT_DIR = "checkpoints/shared_dqn"
CHECKPOINT_EVERY = 25
KEEP_CHECKPOINTS = 5


def describe_tensorflow_backend():
    system = platform.system()
    machine = platform.machine()
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPU devices: {gpus}", flush=True)
    if system == "Darwin":
        if gpus:
            print(
                "TensorFlow Metal backend appears available via GPU device(s).",
                flush=True,
            )
        else:
            print(
                "No TensorFlow GPU device detected on macOS. On Apple Silicon install: pip install tensorflow tensorflow-metal",
                flush=True,
            )
        print(f"macOS machine: {machine}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a shared DQN policy against Godot Gym environments.")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--base-port", type=int, default=BASE_PORT)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps-per-episode", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--target-update-every", type=int, default=TARGET_UPDATE_EVERY)
    parser.add_argument("--replay-warmup", type=int, default=REPLAY_WARMUP)
    parser.add_argument("--epsilon-start", type=float, default=EPSILON_START)
    parser.add_argument("--epsilon-min", type=float, default=EPSILON_MIN)
    parser.add_argument("--epsilon-decay", type=float, default=EPSILON_DECAY)
    parser.add_argument("--obs-dim", type=int, default=OBS_DIM)
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS)
    parser.add_argument("--num-actions", type=int, default=NUM_ACTIONS)
    parser.add_argument("--replay-capacity", type=int, default=REPLAY_CAPACITY)
    parser.add_argument("--env-seed-base", type=int, default=ENV_SEED_BASE)
    parser.add_argument("--episode-seed-multiplier", type=int, default=EPISODE_SEED_MULTIPLIER)
    parser.add_argument("--env-timeout", type=float, default=ENV_TIMEOUT)
    parser.add_argument("--weights-path", default=WEIGHTS_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--keep-checkpoints", type=int, default=KEEP_CHECKPOINTS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def select_actions(model, obs_batch, alive_mask, epsilon, num_actions=8):
    actions = np.zeros((obs_batch.shape[0],), dtype=np.int32)
    for i in range(obs_batch.shape[0]):
        if not alive_mask[i]:
            actions[i] = 0
            continue
        if random.random() < epsilon:
            actions[i] = random.randint(0, num_actions - 1)
        else:
            q = model(np.expand_dims(obs_batch[i], axis=0), training=False).numpy()[0]
            actions[i] = int(np.argmax(q))
    return actions


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
    manager = GodotProcessManager(godot_bin=args.godot_bin)
    manager.start_many(ports, headless=args.headless)
    print(f"Started Godot instances on ports {ports}", flush=True)

    try:
        envs = [
            TeamBattleGymEnv(
                port=p,
                seed=args.env_seed_base + i,
                num_agents=args.num_agents,
                obs_dim=args.obs_dim,
                num_actions=args.num_actions,
                timeout=args.env_timeout,
            )
            for i, p in enumerate(ports)
        ]
        print("Connected Gym environments to Godot instances", flush=True)
        model = build_shared_q_network(obs_dim=args.obs_dim, num_actions=args.num_actions)
        target_model = build_shared_q_network(obs_dim=args.obs_dim, num_actions=args.num_actions)
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
            print(f"Starting episode={episode:04d} epsilon={epsilon:.3f}", flush=True)
            env_states = []
            for env_idx, env in enumerate(envs):
                obs, info = env.reset(seed=args.episode_seed_multiplier * episode + env_idx)
                env_states.append({
                    "obs": obs,
                    "alive": np.asarray(info["alive_mask"], dtype=np.bool_),
                    "done": False,
                    "ep_reward": np.zeros((args.num_agents,), dtype=np.float32)
                })

            losses = []

            for step in range(args.max_steps_per_episode):
                if all(state["done"] for state in env_states):
                    break

                for env, state in zip(envs, env_states):
                    if state["done"]:
                        continue

                    actions = select_actions(model, state["obs"], state["alive"], epsilon, args.num_actions)
                    try:
                        next_obs, mean_reward, terminated, truncated, info = env.step(actions)
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"Godot env on port {env.port} failed during episode={episode} step={step}"
                        ) from exc

                    per_agent_rewards = np.asarray(info["per_agent_rewards"], dtype=np.float32)
                    per_agent_done = np.asarray(info["per_agent_done"], dtype=np.bool_)
                    next_alive = np.asarray(info["alive_mask"], dtype=np.bool_)

                    for agent_idx in range(args.num_agents):
                        if not state["alive"][agent_idx] and not next_alive[agent_idx]:
                            continue
                        buffer.add(
                            state["obs"][agent_idx],
                            int(actions[agent_idx]),
                            float(per_agent_rewards[agent_idx]),
                            next_obs[agent_idx],
                            bool(per_agent_done[agent_idx] or terminated or truncated),
                        )
                        state["ep_reward"][agent_idx] += per_agent_rewards[agent_idx]

                    state["obs"] = next_obs
                    state["alive"] = next_alive
                    state["done"] = bool(terminated or truncated)

                    if len(buffer) >= args.replay_warmup:
                        loss = train_step(model, target_model, optimizer, buffer, args.batch_size, args.gamma)
                        losses.append(loss)

            if (episode + 1) % args.target_update_every == 0:
                target_model.set_weights(model.get_weights())

            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
            rewards_summary = [state["ep_reward"].tolist() for state in env_states]
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
    finally:
        for env in locals().get("envs", []):
            try:
                env.close()
            except Exception:
                pass
        manager.stop_all()


if __name__ == "__main__":
    main()
