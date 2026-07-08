import argparse
import json
import os
import platform
import random
import sys
import sysconfig
from collections import OrderedDict
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
from models import build_hybrid_actor_critic
from scenario_gym_env import ScenarioGymEnv


LOG_2PI = np.float32(np.log(2.0 * np.pi))


def parse_args():
    parser = argparse.ArgumentParser(description="Generic PPO trainer for hybrid Godot action spaces.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--initial-log-std", type=float, default=-0.5)
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weights-path", default="generic_ppo_hybrid.weights.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_ppo_hybrid")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
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


def build_action_metadata(action_space_spec):
    components = OrderedDict(action_space_spec)
    discrete = []
    continuous = []
    continuous_low = []
    continuous_high = []
    offset = 0

    for name, component in components.items():
        action_type = str(component.get("action_type", component.get("type", "discrete")))
        size = int(component.get("size", 1))
        if action_type == "discrete":
            discrete.append({"name": str(name), "size": size})
        elif action_type == "continuous":
            low = _component_bound(component.get("low", -1.0), size, -1.0)
            high = _component_bound(component.get("high", 1.0), size, 1.0)
            continuous.append({
                "name": str(name),
                "size": size,
                "slice": slice(offset, offset + size),
            })
            continuous_low.extend(low.tolist())
            continuous_high.extend(high.tolist())
            offset += size
        else:
            raise RuntimeError(f"Unsupported action component type {action_type!r} for component {name!r}")

    return {
        "discrete": discrete,
        "discrete_sizes": [item["size"] for item in discrete],
        "continuous": continuous,
        "continuous_size": int(offset),
        "continuous_low": np.asarray(continuous_low, dtype=np.float32),
        "continuous_high": np.asarray(continuous_high, dtype=np.float32),
    }


def _component_bound(value, size, default):
    array = np.asarray(value if isinstance(value, (list, tuple)) else [value], dtype=np.float32)
    if array.size == 0:
        array = np.asarray([default], dtype=np.float32)
    if array.size == 1:
        return np.full((size,), float(array[0]), dtype=np.float32)
    if array.size != size:
        raise RuntimeError(f"Action bound has size {array.size}, expected {size}")
    return array.astype(np.float32)


def split_model_outputs(outputs, action_meta):
    discrete_count = len(action_meta["discrete"])
    logits = list(outputs[:discrete_count])
    cursor = discrete_count
    if action_meta["continuous_size"] > 0:
        continuous_mean = outputs[cursor]
        cursor += 1
    else:
        continuous_mean = None
    value = outputs[cursor]
    return logits, continuous_mean, tf.squeeze(value, axis=-1)


def pack_action(discrete_actions, continuous_action, action_meta):
    action = {}
    for idx, component in enumerate(action_meta["discrete"]):
        action[component["name"]] = int(discrete_actions[idx])

    for component in action_meta["continuous"]:
        values = continuous_action[component["slice"]]
        if component["size"] == 1:
            action[component["name"]] = float(values[0])
        else:
            action[component["name"]] = values.astype(np.float32).tolist()
    return action


def zero_env_action(action_meta):
    discrete_actions = np.zeros((len(action_meta["discrete"]),), dtype=np.int32)
    continuous_action = np.zeros((action_meta["continuous_size"],), dtype=np.float32)
    return pack_action(discrete_actions, continuous_action, action_meta)


def select_action(model, log_std, obs, action_meta):
    obs_tensor = tf.convert_to_tensor(np.expand_dims(obs, axis=0), dtype=tf.float32)
    outputs = model(obs_tensor, training=False)
    logits, continuous_mean, value = split_model_outputs(outputs, action_meta)

    discrete_actions = []
    log_prob_parts = []
    for component_logits in logits:
        action = tf.random.categorical(component_logits, 1, dtype=tf.int32)
        action = tf.squeeze(action, axis=-1)
        discrete_actions.append(int(action.numpy()[0]))
        selected_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=action,
            logits=component_logits,
        )
        log_prob_parts.append(selected_log_prob)

    if action_meta["continuous_size"] > 0:
        mean = continuous_mean[0]
        std = tf.exp(log_std)
        raw_action = mean + tf.random.normal(tf.shape(mean)) * std
        low = tf.convert_to_tensor(action_meta["continuous_low"], dtype=tf.float32)
        high = tf.convert_to_tensor(action_meta["continuous_high"], dtype=tf.float32)
        continuous_action = tf.clip_by_value(raw_action, low, high)
        continuous_log_prob = gaussian_log_prob(raw_action[None, :], continuous_mean, log_std)
        log_prob_parts.append(continuous_log_prob)
        continuous_np = continuous_action.numpy().astype(np.float32)
    else:
        continuous_np = np.zeros((0,), dtype=np.float32)

    if log_prob_parts:
        log_prob = float(tf.reduce_sum(tf.stack(log_prob_parts, axis=0)).numpy())
    else:
        log_prob = 0.0

    return {
        "env_action": pack_action(discrete_actions, continuous_np, action_meta),
        "discrete_actions": np.asarray(discrete_actions, dtype=np.int32),
        "continuous_action": continuous_np,
        "log_prob": log_prob,
        "value": float(value.numpy()[0]),
    }


def gaussian_log_prob(actions, means, log_std):
    std = tf.exp(log_std)
    return tf.reduce_sum(
        -0.5 * (((actions - means) / std) ** 2 + 2.0 * log_std + LOG_2PI),
        axis=-1,
    )


def gaussian_entropy(log_std):
    return tf.reduce_sum(log_std + 0.5 * (1.0 + LOG_2PI))


def evaluate_actions(model, log_std, obs, discrete_actions, continuous_actions, action_meta):
    outputs = model(obs, training=True)
    logits, continuous_mean, values = split_model_outputs(outputs, action_meta)

    log_prob_parts = []
    entropy_parts = []
    for idx, component_logits in enumerate(logits):
        labels = discrete_actions[:, idx]
        log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=component_logits,
        )
        probs = tf.nn.softmax(component_logits, axis=-1)
        log_probs = tf.nn.log_softmax(component_logits, axis=-1)
        entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
        log_prob_parts.append(log_prob)
        entropy_parts.append(entropy)

    if action_meta["continuous_size"] > 0:
        log_prob_parts.append(gaussian_log_prob(continuous_actions, continuous_mean, log_std))
        entropy_parts.append(tf.ones((tf.shape(obs)[0],), dtype=tf.float32) * gaussian_entropy(log_std))

    log_probs = tf.add_n(log_prob_parts) if log_prob_parts else tf.zeros((tf.shape(obs)[0],), dtype=tf.float32)
    entropy = tf.add_n(entropy_parts) if entropy_parts else tf.zeros((tf.shape(obs)[0],), dtype=tf.float32)
    return log_probs, entropy, values


def compute_returns_advantages(rewards, dones, values, gamma, gae_lambda):
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = 0.0
    for idx in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[idx]
        delta = rewards[idx] + gamma * next_value * nonterminal - values[idx]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[idx] = last_advantage
        next_value = values[idx]

    returns = advantages + values
    return returns.astype(np.float32), advantages.astype(np.float32)


def new_trajectory():
    return {
        "obs": [],
        "discrete_actions": [],
        "continuous_actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "dones": [],
    }


def append_transition(trajectory, obs, selected, reward, done):
    trajectory["obs"].append(np.asarray(obs, dtype=np.float32))
    trajectory["discrete_actions"].append(selected["discrete_actions"])
    trajectory["continuous_actions"].append(selected["continuous_action"])
    trajectory["log_probs"].append(selected["log_prob"])
    trajectory["values"].append(selected["value"])
    trajectory["rewards"].append(float(reward))
    trajectory["dones"].append(float(done))


def trajectory_has_samples(trajectory):
    return len(trajectory["rewards"]) > 0


def build_update_batch(trajectories, action_meta, gamma, gae_lambda):
    obs_parts = []
    discrete_parts = []
    continuous_parts = []
    log_prob_parts = []
    return_parts = []
    advantage_parts = []

    for trajectory in trajectories:
        if not trajectory_has_samples(trajectory):
            continue

        returns, advantages = compute_returns_advantages(
            trajectory["rewards"],
            trajectory["dones"],
            trajectory["values"],
            gamma,
            gae_lambda,
        )
        obs_parts.append(np.asarray(trajectory["obs"], dtype=np.float32))
        discrete_parts.append(
            np.asarray(trajectory["discrete_actions"], dtype=np.int32).reshape(
                -1,
                len(action_meta["discrete"]),
            )
        )
        continuous_parts.append(
            np.asarray(trajectory["continuous_actions"], dtype=np.float32).reshape(
                -1,
                action_meta["continuous_size"],
            )
        )
        log_prob_parts.append(np.asarray(trajectory["log_probs"], dtype=np.float32))
        return_parts.append(returns)
        advantage_parts.append(advantages)

    if not obs_parts:
        return None

    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "discrete_actions": np.concatenate(discrete_parts, axis=0),
        "continuous_actions": np.concatenate(continuous_parts, axis=0),
        "log_probs": np.concatenate(log_prob_parts, axis=0),
        "returns": np.concatenate(return_parts, axis=0),
        "advantages": np.concatenate(advantage_parts, axis=0),
    }


def ppo_update(model, log_std, optimizer, batch, action_meta, args):
    obs = tf.convert_to_tensor(batch["obs"], dtype=tf.float32)
    discrete_actions = tf.convert_to_tensor(batch["discrete_actions"], dtype=tf.int32)
    continuous_actions = tf.convert_to_tensor(batch["continuous_actions"], dtype=tf.float32)
    old_log_probs = tf.convert_to_tensor(batch["log_probs"], dtype=tf.float32)
    returns = tf.convert_to_tensor(batch["returns"], dtype=tf.float32)
    advantages = tf.convert_to_tensor(batch["advantages"], dtype=tf.float32)

    advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
    count = obs.shape[0]
    indices = np.arange(count)
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []

    train_vars = model.trainable_variables + ([log_std] if action_meta["continuous_size"] > 0 else [])
    for _ in range(args.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, count, args.batch_size):
            idx = indices[start:start + args.batch_size]
            with tf.GradientTape() as tape:
                new_log_probs, entropy, values = evaluate_actions(
                    model,
                    log_std,
                    tf.gather(obs, idx),
                    tf.gather(discrete_actions, idx),
                    tf.gather(continuous_actions, idx),
                    action_meta,
                )
                ratio = tf.exp(new_log_probs - tf.gather(old_log_probs, idx))
                batch_advantages = tf.gather(advantages, idx)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                policy_loss = -tf.reduce_mean(tf.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages))
                value_loss = tf.reduce_mean(tf.square(tf.gather(returns, idx) - values))
                entropy_bonus = tf.reduce_mean(entropy)
                loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_bonus

            grads = tape.gradient(loss, train_vars)
            optimizer.apply_gradients(zip(grads, train_vars))
            losses.append(float(loss.numpy()))
            policy_losses.append(float(policy_loss.numpy()))
            value_losses.append(float(value_loss.numpy()))
            entropies.append(float(entropy_bonus.numpy()))

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


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
        if env0.action_type != "hybrid":
            raise RuntimeError(f"train_generic_ppo.py requires action_type='hybrid', got {env0.action_type!r}")

        obs_dim = env0.obs_dim
        action_meta = build_action_metadata(env0.action_space_spec)
        expected_action_space = json.dumps(env0.action_space_spec, sort_keys=True)
        print(
            f"Scenario spec: agent_id={env0.agent_id} agents={env0.agent_ids} multi_agent={args.multi_agent} obs_dim={obs_dim} "
            f"discrete={action_meta['discrete']} continuous={action_meta['continuous']}",
            flush=True,
        )

        for env in envs:
            if env.obs_dim != obs_dim or env.action_type != "hybrid":
                raise RuntimeError("All parallel environments must expose the same obs_dim and hybrid action_type")
            specs_to_check = env.agent_specs if args.multi_agent else [env._spec_for_agent(env.agent_id)]
            for spec in specs_to_check:
                action_space = spec.get("action_space", {})
                if json.dumps(action_space, sort_keys=True) != expected_action_space:
                    raise RuntimeError("Hybrid PPO currently requires all controlled agents to share the same action_space spec")

        model = build_hybrid_actor_critic(
            obs_dim=obs_dim,
            discrete_sizes=action_meta["discrete_sizes"],
            continuous_size=action_meta["continuous_size"],
        )
        model(np.zeros((1, obs_dim), dtype=np.float32), training=False)
        log_std = tf.Variable(
            np.full((action_meta["continuous_size"],), args.initial_log_std, dtype=np.float32),
            name="continuous_log_std",
            trainable=True,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        start_episode = 0

        checkpoint = tf.train.Checkpoint(
            model=model,
            log_std=log_std,
            optimizer=optimizer,
            episode=tf.Variable(0, dtype=tf.int64),
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.checkpoint_dir,
            max_to_keep=args.keep_checkpoints,
        )
        if args.resume and checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            print(f"Resumed checkpoint {checkpoint_manager.latest_checkpoint} from episode={start_episode}", flush=True)

        for episode in range(start_episode, args.num_episodes):
            trajectories = []
            rewards_summary = []

            for env_idx, env in enumerate(envs):
                obs, _ = env.reset(seed=args.episode_seed_multiplier * episode + env_idx)
                if args.multi_agent:
                    env_trajectories = {
                        agent_id: new_trajectory()
                        for agent_id in env.agent_ids
                    }
                    done_mask = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    episode_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)

                    for _step in range(args.max_steps_per_episode):
                        action_payload = {}
                        selected_by_agent = {}
                        for agent_idx, agent_id in enumerate(env.agent_ids):
                            if done_mask[agent_idx]:
                                action_payload[agent_id] = zero_env_action(action_meta)
                                continue
                            selected = select_action(model, log_std, obs[agent_idx], action_meta)
                            action_payload[agent_id] = selected["env_action"]
                            selected_by_agent[agent_id] = (agent_idx, selected, obs[agent_idx].copy())

                        next_obs, _reward, terminated, truncated, info = env.step(action_payload)
                        global_done = bool(terminated or truncated)
                        per_agent_rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
                        per_agent_done = np.asarray(info.get("per_agent_done"), dtype=np.bool_)

                        for agent_id, (agent_idx, selected, agent_obs) in selected_by_agent.items():
                            done = bool(global_done or per_agent_done[agent_idx])
                            append_transition(
                                env_trajectories[agent_id],
                                agent_obs,
                                selected,
                                float(per_agent_rewards[agent_idx]),
                                done,
                            )
                            episode_reward[agent_idx] += per_agent_rewards[agent_idx]

                        obs = next_obs
                        done_mask = np.logical_or(done_mask, per_agent_done)
                        if global_done or bool(np.all(done_mask)):
                            break

                    trajectories.extend(
                        trajectory
                        for trajectory in env_trajectories.values()
                        if trajectory_has_samples(trajectory)
                    )
                    rewards_summary.append(episode_reward.tolist())
                else:
                    trajectory = new_trajectory()
                    episode_reward = 0.0
                    for _step in range(args.max_steps_per_episode):
                        selected = select_action(model, log_std, obs, action_meta)
                        next_obs, reward, terminated, truncated, _info = env.step(selected["env_action"])
                        done = bool(terminated or truncated)

                        append_transition(trajectory, obs, selected, float(reward), done)
                        episode_reward += float(reward)
                        obs = next_obs
                        if done:
                            break
                    if trajectory_has_samples(trajectory):
                        trajectories.append(trajectory)
                    rewards_summary.append(episode_reward)

            update_batch = build_update_batch(
                trajectories,
                action_meta,
                args.gamma,
                args.gae_lambda,
            )
            if update_batch is None:
                print(f"episode={episode:04d} skipped_update=no_samples rewards={rewards_summary}", flush=True)
                continue

            metrics = ppo_update(model, log_std, optimizer, update_batch, action_meta, args)
            print(
                f"episode={episode:04d} rewards={rewards_summary} "
                f"samples={len(update_batch['rewards']) if 'rewards' in update_batch else len(update_batch['obs'])} "
                f"loss={metrics['loss']:.5f} policy_loss={metrics['policy_loss']:.5f} "
                f"value_loss={metrics['value_loss']:.5f} entropy={metrics['entropy']:.5f}",
                flush=True,
            )

            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                checkpoint.episode.assign(episode + 1)
                saved_path = checkpoint_manager.save(checkpoint_number=episode + 1)
                print(f"Saved checkpoint: {saved_path}", flush=True)

        checkpoint.episode.assign(args.num_episodes)
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
