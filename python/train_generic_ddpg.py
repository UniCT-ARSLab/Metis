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
from models import build_continuous_actor, build_continuous_critic
from replay_buffer import ReplayBuffer
from scenario_gym_env import ScenarioGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Generic DDPG trainer for continuous Godot scenarios.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration-noise", type=float, default=0.2)
    parser.add_argument("--exploration-noise-min", type=float, default=0.02)
    parser.add_argument("--exploration-noise-decay", type=float, default=0.995)
    parser.add_argument("--replay-warmup", type=int, default=500)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--target-update-every", type=int, default=1)
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-weights-path", default="generic_ddpg_actor.weights.h5")
    parser.add_argument("--critic-weights-path", default="generic_ddpg_critic.weights.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_ddpg")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--demo-path", action="append", default=[])
    parser.add_argument("--demo-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-max-transitions", type=int, default=0)
    parser.add_argument("--demo-bc-epochs", type=int, default=0)
    parser.add_argument("--demo-bc-batch-size", type=int, default=128)
    parser.add_argument("--demo-bc-learning-rate", type=float, default=None)
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def describe_tensorflow_backend():
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}", flush=True)
    if platform.system() == "Darwin":
        print(f"macOS machine: {platform.machine()}", flush=True)


def select_action(actor, obs, low, high, noise_std):
    action = actor(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    if noise_std > 0:
        action = action + np.random.normal(0.0, noise_std, size=action.shape)
    return np.clip(action, low, high).astype(np.float32)


def select_actions(actor, obs_batch, done_mask, low, high, noise_std):
    actions = np.zeros((obs_batch.shape[0], low.shape[0]), dtype=np.float32)
    for idx, obs in enumerate(obs_batch):
        if not done_mask[idx]:
            actions[idx] = select_action(actor, obs, low, high, noise_std)
    return actions


def soft_update(target_model, source_model, tau):
    target_weights = target_model.get_weights()
    source_weights = source_model.get_weights()
    updated = [
        (1.0 - tau) * target_weight + tau * source_weight
        for target_weight, source_weight in zip(target_weights, source_weights)
    ]
    target_model.set_weights(updated)


def train_step(actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, buffer, batch_size, gamma):
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, action_dtype=np.float32)
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards.reshape(-1, 1), dtype=tf.float32)
    next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones.reshape(-1, 1), dtype=tf.float32)

    next_actions = target_actor(next_obs, training=False)
    target_q = target_critic([next_obs, next_actions], training=False)
    y = rewards + (1.0 - dones) * gamma * target_q

    with tf.GradientTape() as tape:
        q = critic([obs, actions], training=True)
        critic_loss = tf.reduce_mean(tf.square(y - q))
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    with tf.GradientTape() as tape:
        policy_actions = actor(obs, training=True)
        actor_loss = -tf.reduce_mean(critic([obs, policy_actions], training=False))
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    return float(actor_loss.numpy()), float(critic_loss.numpy())


def load_demonstration_arrays(paths, obs_dim, action_size, max_transitions=0):
    obs_parts = []
    action_parts = []
    reward_parts = []
    next_obs_parts = []
    done_parts = []
    remaining = int(max_transitions)

    for path in paths:
        with np.load(path) as data:
            required = ("obs", "actions", "rewards", "next_obs", "dones")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"Demonstration file {path!r} is missing arrays: {missing}")

            obs = np.asarray(data["obs"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            rewards = np.asarray(data["rewards"], dtype=np.float32)
            next_obs = np.asarray(data["next_obs"], dtype=np.float32)
            dones = np.asarray(data["dones"], dtype=np.float32)
            action_type = str(np.asarray(data.get("action_type", ["continuous"]))[0])

        if action_type != "continuous":
            raise ValueError(f"Demo {path!r} has action_type={action_type!r}; DDPG requires continuous demos")
        if obs.ndim != 2 or obs.shape[1] != obs_dim:
            raise ValueError(f"Demo {path!r} obs shape {obs.shape} does not match obs_dim={obs_dim}")
        if next_obs.shape != obs.shape:
            raise ValueError(f"Demo {path!r} next_obs shape {next_obs.shape} does not match obs shape {obs.shape}")
        if actions.ndim != 2 or actions.shape[1] != action_size:
            raise ValueError(f"Demo {path!r} actions shape {actions.shape} does not match action_size={action_size}")

        count = len(actions)
        if remaining > 0:
            count = min(count, remaining)
            remaining -= count

        obs_parts.append(obs[:count])
        action_parts.append(actions[:count])
        reward_parts.append(rewards[:count])
        next_obs_parts.append(next_obs[:count])
        done_parts.append(dones[:count])

        if remaining == 0 and max_transitions > 0:
            break

    if not obs_parts:
        return None

    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "actions": np.concatenate(action_parts, axis=0),
        "rewards": np.concatenate(reward_parts, axis=0),
        "next_obs": np.concatenate(next_obs_parts, axis=0),
        "dones": np.concatenate(done_parts, axis=0),
    }


def pretrain_actor_behavior_cloning(actor, demo_data, epochs, batch_size, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    obs = demo_data["obs"]
    actions = demo_data["actions"]
    count = len(actions)

    for epoch in range(int(epochs)):
        order = np.random.permutation(count)
        losses = []

        for start in range(0, count, batch_size):
            idx = order[start:start + batch_size]
            batch_obs = tf.convert_to_tensor(obs[idx], dtype=tf.float32)
            batch_actions = tf.convert_to_tensor(actions[idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                predicted_actions = actor(batch_obs, training=True)
                loss = tf.reduce_mean(tf.square(batch_actions - predicted_actions))

            grads = tape.gradient(loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            losses.append(float(loss.numpy()))

        print(
            f"demo_bc_epoch={epoch + 1:04d}/{epochs:04d} "
            f"actor_mse={float(np.mean(losses)):.6f}",
            flush=True,
        )


def main():
    args = parse_args()
    describe_tensorflow_backend()
    random.seed(args.env_seed_base)
    np.random.seed(args.env_seed_base)
    tf.random.set_seed(args.env_seed_base)

    ports = [args.base_port + i for i in range(args.num_envs)]
    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
        scene_path=args.godot_scene,
    )
    manager.start_many(ports, headless=args.headless, debug=args.godot_debug)
    print(f"Started Godot instances on ports {ports}", flush=True)

    envs = []
    try:
        envs = [
            ScenarioGymEnv(
                port=port,
                seed=args.env_seed_base + idx,
                timeout=args.env_timeout,
                agent_id=args.agent_id,
                multi_agent=args.multi_agent,
            )
            for idx, port in enumerate(ports)
        ]

        env0 = envs[0]
        if env0.action_type != "continuous":
            raise RuntimeError(f"train_generic_ddpg.py requires action_type='continuous', got {env0.action_type!r}")

        obs_dim = env0.obs_dim
        action_size = env0.action_size
        action_low = env0.action_low
        action_high = env0.action_high
        print(
            f"Scenario spec: agent_id={env0.agent_id} agents={env0.agent_ids} multi_agent={args.multi_agent} "
            f"obs_dim={obs_dim} action_size={action_size} action_names={env0.action_names}",
            flush=True,
        )

        for env in envs[1:]:
            if env.obs_dim != obs_dim or env.action_type != "continuous" or env.action_size != action_size:
                raise RuntimeError("All parallel environments must expose the same obs_dim and continuous action_size")

        actor = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
        critic = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        target_actor = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
        target_critic = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        target_actor.set_weights(actor.get_weights())
        target_critic.set_weights(critic.get_weights())

        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_learning_rate)
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate)
        buffer = ReplayBuffer(capacity=args.replay_capacity)
        noise_std = args.exploration_noise
        start_episode = 0

        checkpoint = tf.train.Checkpoint(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            episode=tf.Variable(0, dtype=tf.int64),
            noise_std=tf.Variable(args.exploration_noise, dtype=tf.float32),
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.checkpoint_dir,
            max_to_keep=args.keep_checkpoints,
        )
        if args.resume and checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            noise_std = float(checkpoint.noise_std.numpy())
            print(
                f"Resumed checkpoint {checkpoint_manager.latest_checkpoint} from episode={start_episode} "
                f"noise_std={noise_std:.3f}",
                flush=True,
            )

        demo_data = None
        if args.demo_path:
            demo_data = load_demonstration_arrays(
                args.demo_path,
                obs_dim=obs_dim,
                action_size=action_size,
                max_transitions=args.demo_max_transitions,
            )
            if demo_data is not None:
                print(
                    f"Loaded continuous demonstrations transitions={len(demo_data['actions'])} "
                    f"paths={args.demo_path}",
                    flush=True,
                )

        if demo_data is not None and args.demo_prefill:
            added = buffer.add_many(
                demo_data["obs"],
                demo_data["actions"],
                demo_data["rewards"],
                demo_data["next_obs"],
                demo_data["dones"],
            )
            print(f"Prefilled replay buffer with demonstration transitions={added}", flush=True)

        if demo_data is not None and args.demo_bc_epochs > 0:
            pretrain_actor_behavior_cloning(
                actor,
                demo_data,
                epochs=args.demo_bc_epochs,
                batch_size=args.demo_bc_batch_size,
                learning_rate=args.demo_bc_learning_rate or args.actor_learning_rate,
            )
            target_actor.set_weights(actor.get_weights())

        for episode in range(start_episode, args.num_episodes):
            env_states = []
            for env_idx, env in enumerate(envs):
                obs, info = env.reset(seed=args.episode_seed_multiplier * episode + env_idx)
                if args.multi_agent:
                    done_mask = np.asarray(info.get("per_agent_done", np.zeros((len(env.agent_ids),), dtype=np.bool_)), dtype=np.bool_)
                    ep_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)
                    action_sum = np.zeros((len(env.agent_ids), action_size), dtype=np.float32)
                    action_count = np.zeros((len(env.agent_ids), 1), dtype=np.float32)
                else:
                    done_mask = None
                    ep_reward = 0.0
                    action_sum = np.zeros((action_size,), dtype=np.float32)
                    action_count = 0.0
                env_states.append({
                    "obs": obs,
                    "done": False,
                    "done_mask": done_mask,
                    "ep_reward": ep_reward,
                    "action_sum": action_sum,
                    "action_count": action_count,
                })

            actor_losses = []
            critic_losses = []
            for step_idx in range(args.max_steps_per_episode):
                if all(state["done"] for state in env_states):
                    break

                for env, state in zip(envs, env_states):
                    if state["done"]:
                        continue

                    if args.multi_agent:
                        action = select_actions(actor, state["obs"], state["done_mask"], action_low, action_high, noise_std)
                    else:
                        action = select_action(actor, state["obs"], action_low, action_high, noise_std)

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)

                    if args.multi_agent:
                        per_agent_rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
                        per_agent_done = np.asarray(info.get("per_agent_done"), dtype=np.bool_)
                        for agent_idx in range(len(env.agent_ids)):
                            if state["done_mask"][agent_idx] and per_agent_done[agent_idx]:
                                continue
                            buffer.add(
                                state["obs"][agent_idx],
                                action[agent_idx],
                                float(per_agent_rewards[agent_idx]),
                                next_obs[agent_idx],
                                bool(per_agent_done[agent_idx] or done),
                            )
                            state["ep_reward"][agent_idx] += per_agent_rewards[agent_idx]
                            state["action_sum"][agent_idx] += action[agent_idx]
                            state["action_count"][agent_idx, 0] += 1.0
                        state["done_mask"] = per_agent_done
                    else:
                        buffer.add(state["obs"], action, float(reward), next_obs, done)
                        state["ep_reward"] += float(reward)
                        state["action_sum"] += action
                        state["action_count"] += 1.0

                    state["obs"] = next_obs
                    state["done"] = done

                    if len(buffer) >= args.replay_warmup:
                        actor_loss, critic_loss = train_step(
                            actor,
                            critic,
                            target_actor,
                            target_critic,
                            actor_optimizer,
                            critic_optimizer,
                            buffer,
                            args.batch_size,
                            args.gamma,
                        )
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)

                if len(buffer) >= args.replay_warmup and (step_idx + 1) % args.target_update_every == 0:
                    soft_update(target_actor, actor, args.tau)
                    soft_update(target_critic, critic, args.tau)

            noise_std = max(args.exploration_noise_min, noise_std * args.exploration_noise_decay)
            rewards_summary = [
                state["ep_reward"].tolist() if hasattr(state["ep_reward"], "tolist") else state["ep_reward"]
                for state in env_states
            ]
            if args.multi_agent:
                mean_actions = [
                    (state["action_sum"] / np.maximum(state["action_count"], 1.0)).tolist()
                    for state in env_states
                ]
            else:
                mean_actions = [
                    (state["action_sum"] / max(float(state["action_count"]), 1.0)).tolist()
                    for state in env_states
                ]
            print(
                f"episode={episode:04d} noise_std={noise_std:.3f} "
                f"actor_loss={float(np.mean(actor_losses)) if actor_losses else 0.0:.5f} "
                f"critic_loss={float(np.mean(critic_losses)) if critic_losses else 0.0:.5f} "
                f"rewards={rewards_summary} mean_actions={mean_actions}",
                flush=True,
            )

            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                checkpoint.episode.assign(episode + 1)
                checkpoint.noise_std.assign(noise_std)
                saved_path = checkpoint_manager.save(checkpoint_number=episode + 1)
                print(f"Saved checkpoint: {saved_path}", flush=True)

        checkpoint.episode.assign(args.num_episodes)
        checkpoint.noise_std.assign(noise_std)
        saved_path = checkpoint_manager.save(checkpoint_number=args.num_episodes)
        print(f"Saved final checkpoint: {saved_path}", flush=True)
        actor.save_weights(args.actor_weights_path)
        critic.save_weights(args.critic_weights_path)
        print(f"Saved actor weights: {args.actor_weights_path}", flush=True)
        print(f"Saved critic weights: {args.critic_weights_path}", flush=True)
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        manager.stop_all()


if __name__ == "__main__":
    main()
