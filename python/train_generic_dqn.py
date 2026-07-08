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
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-action-every", type=int, default=1)
    parser.add_argument("--weights-path", default="generic_dqn_weights.weights.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_dqn")
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
    parser.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default= False)
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


def select_actions(model, obs_batch, done_mask, epsilon, num_actions):
    actions = np.zeros((obs_batch.shape[0],), dtype=np.int32)
    for idx, obs in enumerate(obs_batch):
        if done_mask[idx]:
            actions[idx] = 0
        else:
            actions[idx] = select_action(model, obs, epsilon, num_actions)
    return actions


def action_counts_summary(action_counts, action_names):
    parts = []
    for idx, count in enumerate(action_counts):
        if count <= 0:
            continue
        name = action_names[idx] if idx < len(action_names) else str(idx)
        parts.append(f"{idx}:{name}={int(count)}")
    return "{" + ", ".join(parts) + "}"


def action_last_summary(last_actions, action_names):
    result = []
    for action in last_actions:
        action_id = int(action)
        action_name = action_names[action_id] if action_id < len(action_names) else str(action_id)
        result.append(f"{action_id}:{action_name}")
    return result


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


def load_demonstration_arrays(paths, obs_dim, num_actions, max_transitions=0):
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
            actions = np.asarray(data["actions"], dtype=np.int32)
            rewards = np.asarray(data["rewards"], dtype=np.float32)
            next_obs = np.asarray(data["next_obs"], dtype=np.float32)
            dones = np.asarray(data["dones"], dtype=np.float32)

        if obs.ndim != 2 or obs.shape[1] != obs_dim:
            raise ValueError(f"Demo {path!r} obs shape {obs.shape} does not match obs_dim={obs_dim}")
        if next_obs.shape != obs.shape:
            raise ValueError(f"Demo {path!r} next_obs shape {next_obs.shape} does not match obs shape {obs.shape}")
        if np.any(actions < 0) or np.any(actions >= num_actions):
            raise ValueError(f"Demo {path!r} contains actions outside [0, {num_actions})")

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


def pretrain_behavior_cloning(model, demo_data, epochs, batch_size, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    obs = demo_data["obs"]
    actions = demo_data["actions"]
    count = len(actions)

    for epoch in range(int(epochs)):
        order = np.random.permutation(count)
        losses = []
        accuracies = []

        for start in range(0, count, batch_size):
            idx = order[start:start + batch_size]
            batch_obs = tf.convert_to_tensor(obs[idx], dtype=tf.float32)
            batch_actions = tf.convert_to_tensor(actions[idx], dtype=tf.int32)

            with tf.GradientTape() as tape:
                logits = model(batch_obs, training=True)
                loss = loss_fn(batch_actions, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, batch_actions), tf.float32))
            losses.append(float(loss.numpy()))
            accuracies.append(float(accuracy.numpy()))

        print(
            f"demo_bc_epoch={epoch + 1:04d}/{epochs:04d} "
            f"loss={float(np.mean(losses)):.5f} accuracy={float(np.mean(accuracies)):.3f}",
            flush=True,
        )


def main():
    args = parse_args()
    describe_tensorflow_backend()

    ports = [args.base_port + i for i in range(args.num_envs)]
    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
        scene_path=args.godot_scene
    )
    manager.start_many(ports, headless=args.headless, debug = args.godot_debug)
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

        obs_dim = envs[0].obs_dim
        num_actions = envs[0].num_actions
        agent_id = envs[0].agent_id
        agent_ids = envs[0].agent_ids
        if envs[0].action_type != "discrete":
            raise RuntimeError(
                f"train_generic_dqn.py supports only discrete action spaces, got action_type={envs[0].action_type!r}. "
                "Use a continuous-control trainer for Box actions."
            )
        print(
            f"Scenario spec: agent_id={agent_id} agents={agent_ids} multi_agent={args.multi_agent} "
            f"obs_dim={obs_dim} num_actions={num_actions} actions={envs[0].action_names}",
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
        restored_checkpoint = False

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
            restored_checkpoint = True
            print(
                f"Resumed checkpoint {checkpoint_manager.latest_checkpoint} from episode={start_episode} epsilon={epsilon:.3f}",
                flush=True,
            )

        demo_data = None
        if args.demo_path:
            demo_data = load_demonstration_arrays(
                args.demo_path,
                obs_dim=obs_dim,
                num_actions=num_actions,
                max_transitions=args.demo_max_transitions,
            )
            if demo_data is not None:
                print(
                    f"Loaded demonstrations transitions={len(demo_data['actions'])} paths={args.demo_path}",
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
            pretrain_behavior_cloning(
                model,
                demo_data,
                epochs=args.demo_bc_epochs,
                batch_size=args.demo_bc_batch_size,
                learning_rate=args.demo_bc_learning_rate or args.learning_rate,
            )
            target_model.set_weights(model.get_weights())
        elif not restored_checkpoint:
            target_model.set_weights(model.get_weights())

        for episode in range(start_episode, args.num_episodes):
            env_states = []
            for env_idx, env in enumerate(envs):
                obs, info = env.reset(seed=args.episode_seed_multiplier * episode + env_idx)
                if args.multi_agent:
                    done_mask = np.asarray(info.get("per_agent_done", np.zeros((len(env.agent_ids),), dtype=np.bool_)), dtype=np.bool_)
                    ep_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)
                    target_reached = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    target_seen = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    action_counts = np.zeros((len(env.agent_ids), num_actions), dtype=np.int32)
                    last_action = np.zeros((len(env.agent_ids),), dtype=np.int32)
                else:
                    done_mask = None
                    ep_reward = 0.0
                    target_reached = False
                    target_seen = False
                    action_counts = np.zeros((num_actions,), dtype=np.int32)
                    last_action = 0
                env_states.append({
                    "obs": obs,
                    "done": False,
                    "done_mask": done_mask,
                    "ep_reward": ep_reward,
                    "target_reached": target_reached,
                    "target_seen": target_seen,
                    "action_counts": action_counts,
                    "last_action": last_action,
                })

            losses = []
            for step_idx in range(args.max_steps_per_episode):
                if all(state["done"] for state in env_states):
                    break

                for env, state in zip(envs, env_states):
                    if state["done"]:
                        continue

                    if args.multi_agent:
                        action = select_actions(model, state["obs"], state["done_mask"], epsilon, num_actions)
                        for agent_idx, action_id in enumerate(action):
                            state["action_counts"][agent_idx, int(action_id)] += 1
                            state["last_action"][agent_idx] = int(action_id)
                    else:
                        action = select_action(model, state["obs"], epsilon, num_actions)
                        state["action_counts"][int(action)] += 1
                        state["last_action"] = int(action)

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)

                    if args.multi_agent:
                        per_agent_rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
                        per_agent_done = np.asarray(info.get("per_agent_done"), dtype=np.bool_)
                        per_agent_infos = info.get("per_agent_infos", [{} for _ in env.agent_ids])

                        for agent_idx in range(len(env.agent_ids)):
                            agent_info = per_agent_infos[agent_idx] if agent_idx < len(per_agent_infos) else {}
                            if bool(agent_info.get("target_reached", False)):
                                state["target_reached"][agent_idx] = True
                            if bool(agent_info.get("target_first_seen", False)):
                                state["target_seen"][agent_idx] = True

                            if state["done_mask"][agent_idx] and per_agent_done[agent_idx]:
                                continue
                            buffer.add(
                                state["obs"][agent_idx],
                                int(action[agent_idx]),
                                float(per_agent_rewards[agent_idx]),
                                next_obs[agent_idx],
                                bool(per_agent_done[agent_idx] or done),
                            )
                            state["ep_reward"][agent_idx] += per_agent_rewards[agent_idx]

                        state["done_mask"] = per_agent_done
                    else:
                        agent_info = info.get("agent_info", {})
                        if bool(agent_info.get("target_reached", False)):
                            state["target_reached"] = True
                        if bool(agent_info.get("target_first_seen", False)):
                            state["target_seen"] = True

                        buffer.add(
                            state["obs"],
                            int(action),
                            float(reward),
                            next_obs,
                            done,
                        )
                        state["ep_reward"] += float(reward)

                    state["obs"] = next_obs
                    state["done"] = done

                    if len(buffer) >= args.replay_warmup:
                        loss = train_step(model, target_model, optimizer, buffer, args.batch_size, args.gamma)
                        losses.append(loss)

            if (episode + 1) % args.target_update_every == 0:
                target_model.set_weights(model.get_weights())

            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
            rewards_summary = [
                state["ep_reward"].tolist() if hasattr(state["ep_reward"], "tolist") else state["ep_reward"]
                for state in env_states
            ]
            if args.multi_agent:
                reached_summary = [state["target_reached"].astype(np.int32).tolist() for state in env_states]
                seen_summary = [state["target_seen"].astype(np.int32).tolist() for state in env_states]
                reached_total = sum(int(np.sum(state["target_reached"])) for state in env_states)
                seen_total = sum(int(np.sum(state["target_seen"])) for state in env_states)
                metric_count = sum(len(state["target_reached"]) for state in env_states)
            else:
                reached_summary = [int(state["target_reached"]) for state in env_states]
                seen_summary = [int(state["target_seen"]) for state in env_states]
                reached_total = sum(int(state["target_reached"]) for state in env_states)
                seen_total = sum(int(state["target_seen"]) for state in env_states)
                metric_count = len(env_states)
            success_rate = reached_total / max(metric_count, 1)
            seen_rate = seen_total / max(metric_count, 1)
            if args.multi_agent:
                action_summary = [
                    {
                        env.agent_ids[agent_idx]: action_counts_summary(state["action_counts"][agent_idx], env.action_names)
                        for agent_idx in range(len(env.agent_ids))
                    }
                    for env, state in zip(envs, env_states)
                ]
                last_action_summary = [
                    {
                        env.agent_ids[agent_idx]: action_last_summary([state["last_action"][agent_idx]], env.action_names)[0]
                        for agent_idx in range(len(env.agent_ids))
                    }
                    for env, state in zip(envs, env_states)
                ]
            else:
                action_summary = [
                    action_counts_summary(state["action_counts"], env.action_names)
                    for env, state in zip(envs, env_states)
                ]
                last_action_summary = [
                    action_last_summary([state["last_action"]], env.action_names)[0]
                    for env, state in zip(envs, env_states)
                ]
            mean_loss = float(np.mean(losses)) if losses else 0.0
            print(
                f"episode={episode:04d} epsilon={epsilon:.3f} mean_loss={mean_loss:.5f} "
                f"rewards={rewards_summary} reached={reached_summary} seen={seen_summary} "
                f"success_rate={success_rate:.3f} seen_rate={seen_rate:.3f}",
                flush=True,
            )
            if args.log_action_every > 0 and episode % args.log_action_every == 0:
                print(
                    f"episode={episode:04d} actions={action_summary} last_actions={last_action_summary}",
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
