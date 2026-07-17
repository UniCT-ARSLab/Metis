import argparse
import json
import os
import platform
import random
import sys
import sysconfig
from collections import OrderedDict
from pathlib import Path
from queue import Empty


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
        original_args = list(getattr(sys, "orig_argv", sys.argv))
        os.execv(sys.executable, [sys.executable] + original_args[1:])


configure_tensorflow_runtime()

import numpy as np
import tensorflow as tf

from core.models import build_hybrid_actor_critic
from core.opponent_pool import OpponentPool, add_opponent_pool_arguments, validate_team_layout
from core.training import (
    AsyncCollectorPool,
    AsyncEpisodeEvent,
    AsyncWorkerDoneEvent,
    AsyncWorkerErrorEvent,
    ParallelEnvStepper,
    PolicySnapshot,
    BestCheckpointTracker,
    add_best_checkpoint_arguments,
    apply_ready_best_checkpoint,
    add_collector_arguments,
    add_log_format_argument,
    add_lockstep_tuning_arguments,
    add_parallel_env_arguments,
    add_tensorflow_runtime_arguments,
    build_lockstep_user_args,
    configure_tensorflow_devices,
    episode_step_indices,
    print_episode_metrics,
    resolve_resume_checkpoint,
    validate_async_arguments,
)
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


LOG_2PI = np.float32(np.log(2.0 * np.pi))

# 'hybrid' mixes discrete and continuous heads; 'discrete' is the same model with the
# continuous head absent. Both are driven from action_space_spec components.
SUPPORTED_ACTION_TYPES = {"hybrid", "discrete"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generic PPO trainer for hybrid Godot action spaces.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=500,
        help="Maximum episode steps shared with Godot; use 0 to rely only on terminal conditions.",
    )
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
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run Godot without a window. Rendering training instances contends with "
            "TensorFlow for the GPU, and only headless instances get --fixed-fps, without "
            "which physics stays gated to wall-clock 60Hz. Use --no-headless to watch."
        ),
    )
    parser.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default=False)
    add_collector_arguments(parser)
    add_opponent_pool_arguments(parser)
    add_best_checkpoint_arguments(parser)
    add_parallel_env_arguments(parser)
    add_lockstep_tuning_arguments(parser)
    add_log_format_argument(parser)
    add_tensorflow_runtime_arguments(parser)
    return parser.parse_args()


def save_training_checkpoint(checkpoint, checkpoint_manager, episode, final=False):
    checkpoint.episode.assign(episode)
    saved_path = checkpoint_manager.save(checkpoint_number=episode)
    print(f"Saved {'final checkpoint' if final else 'checkpoint'}: {saved_path}", flush=True)
    return saved_path


def request_best_checkpoint_evaluation(tracker, candidate_path, episode):
    tracker.evaluate_async(candidate_path, episode)




def describe_tensorflow_backend(args):
    devices, memory_growth = configure_tensorflow_devices(
        tf,
        memory_growth=args.gpu_memory_growth,
        system_name=platform.system(),
    )
    print(
        f"TensorFlow GPU devices: {devices} memory_growth={'enabled' if memory_growth else 'disabled'}",
        flush=True,
    )
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
        # One discrete component and nothing else: the env exposes Discrete(n) and wants
        # a bare int, not the hybrid dict. See pack_action.
        "single_discrete": len(discrete) == 1 and not continuous,
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
    """Build the action in whatever shape this scenario's env expects.

    Hybrid scenarios take a {component_name: value} dict. A plain discrete scenario --
    Breakout, for instance -- exposes Discrete(n) and its env does `int(action)`, which
    would raise on a dict. Emitting the bare int here keeps the difference contained to
    one function instead of leaking into every call site.
    """
    if action_meta["single_discrete"]:
        return int(discrete_actions[0])

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


def value_of(sample_fn, log_std, obs, action_meta):
    """V(obs) from the critic head, for bootstrapping a time-limit-truncated trajectory.

    Goes through the same traced sampler as select_action so bootstrapping shares its
    thread-safety; the sampled action is discarded, and bootstrapping only happens on a
    truncation, so the extra draw is negligible.
    """
    obs_batch = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)
    log_std_tensor = tf.convert_to_tensor(log_std, dtype=tf.float32)
    _discrete, _continuous, _log_prob, value_t = sample_fn(obs_batch, log_std_tensor)
    return float(value_t.numpy())


def build_sample_action_fn(model, obs_dim, action_meta, device="/CPU:0"):
    """Trace the stochastic action sample into one graph per collector.

    The eager op-by-op version corrupts tensor shapes when several collector threads
    run it concurrently -- a 0-D tensor where a 1-D one is expected -- crashing roughly
    two runs in five at --num-envs 4. A traced concrete function is safe to call from
    multiple threads where eager execution is not, the same fix build_greedy_action_fn
    applies to DQN. Pinning to `device` keeps the ops with the collector's CPU weights.
    """
    continuous_size = int(action_meta["continuous_size"])
    low = tf.constant(action_meta["continuous_low"], dtype=tf.float32)
    high = tf.constant(action_meta["continuous_high"], dtype=tf.float32)

    @tf.function(input_signature=[
        tf.TensorSpec([1, obs_dim], tf.float32),
        tf.TensorSpec([continuous_size], tf.float32),
    ])
    def sample(obs_batch, log_std):
        with tf.device(device):
            outputs = model(obs_batch, training=False)
            logits, continuous_mean, value = split_model_outputs(outputs, action_meta)

            discrete_actions = []
            log_prob = tf.zeros((), dtype=tf.float32)
            for component_logits in logits:
                action = tf.random.categorical(component_logits, 1, dtype=tf.int32)
                action = tf.squeeze(action, axis=[0, 1])  # scalar
                discrete_actions.append(action)
                component_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=action[None], logits=component_logits
                )
                log_prob += component_log_prob[0]

            discrete_out = (
                tf.stack(discrete_actions, axis=0)
                if discrete_actions
                else tf.zeros((0,), dtype=tf.int32)
            )

            # continuous_size is a Python constant, so this branch is resolved at trace
            # time -- a discrete-only scenario never traces the continuous ops at all.
            if continuous_size > 0:
                mean = continuous_mean[0]
                std = tf.exp(log_std)
                raw_action = mean + tf.random.normal(tf.shape(mean)) * std
                continuous_out = tf.clip_by_value(raw_action, low, high)
                log_prob += gaussian_log_prob(raw_action[None, :], continuous_mean, log_std)[0]
            else:
                continuous_out = tf.zeros((0,), dtype=tf.float32)

            return discrete_out, continuous_out, log_prob, value[0]

    return sample


def select_action(sample_fn, log_std, obs, action_meta):
    obs_batch = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)
    log_std_tensor = tf.convert_to_tensor(log_std, dtype=tf.float32)
    discrete_t, continuous_t, log_prob_t, value_t = sample_fn(obs_batch, log_std_tensor)
    discrete_actions = discrete_t.numpy().astype(np.int32)
    continuous_np = continuous_t.numpy().astype(np.float32)

    return {
        "env_action": pack_action(discrete_actions, continuous_np, action_meta),
        "discrete_actions": discrete_actions,
        "continuous_action": continuous_np,
        "log_prob": float(log_prob_t.numpy()),
        "value": float(value_t.numpy()),
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


def compute_returns_advantages(rewards, dones, values, gamma, gae_lambda, bootstrap_value=0.0):
    """GAE over one trajectory.

    `dones` must mark real terminals only. `bootstrap_value` is V(s_T) for a trajectory
    that was cut while still running (a time-limit truncation); it stays 0.0 for one that
    ended terminally, where there genuinely is no future value.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = float(bootstrap_value)
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
        # V(s_T), set only when the episode was cut by the step cap rather than ending.
        "bootstrap_value": 0.0,
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
            trajectory.get("bootstrap_value", 0.0),
        )
        obs_parts.append(np.asarray(trajectory["obs"], dtype=np.float32))
        discrete_parts.append(
            np.asarray(trajectory["discrete_actions"], dtype=np.int32).reshape(
                -1,
                len(action_meta["discrete"]),
            )
        )
        # A discrete-only scenario has continuous_size == 0, and numpy cannot infer a -1
        # row count against a zero-width column, so state the rows explicitly.
        continuous_parts.append(
            np.asarray(trajectory["continuous_actions"], dtype=np.float32).reshape(
                len(trajectory["rewards"]),
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


def run_async_ppo(
    args,
    envs,
    model,
    log_std,
    optimizer,
    action_meta,
    obs_dim,
    checkpoint,
    checkpoint_manager,
    best_tracker,
    start_episode,
):
    validate_async_arguments(args)
    with tf.device("/CPU:0"):
        local_models = [
            build_hybrid_actor_critic(
                obs_dim=obs_dim,
                discrete_sizes=action_meta["discrete_sizes"],
                continuous_size=action_meta["continuous_size"],
            )
            for _env in envs
        ]
    # One traced sampler per collector: they run concurrently, and eager sampling is not
    # thread-safe. set_weights on the closed-over model keeps each trace current.
    sample_fns = [build_sample_action_fn(m, obs_dim, action_meta) for m in local_models]
    snapshot = PolicySnapshot(model.get_weights(), state=log_std.numpy())

    def worker(worker_id, env, _allocator, put, stop_event):
        local_model = local_models[worker_id]
        sample_fn = sample_fns[worker_id]
        policy_version = -1
        for episode in range(start_episode, args.num_episodes):
            if stop_event.is_set():
                return
            policy_version, local_log_std = snapshot.sync_model_with_state(
                local_model,
                policy_version,
            )
            local_log_std = tf.convert_to_tensor(local_log_std, dtype=tf.float32)
            env.configure(
                training_episode=episode,
                max_steps=args.max_steps_per_episode,
                physics_frames_per_step=args.physics_frames_per_step,
            )
            obs, info = env.reset(seed=args.episode_seed_multiplier * episode + worker_id)

            if args.multi_agent:
                done_mask = np.asarray(
                    info.get("per_agent_done", np.zeros((len(env.agent_ids),), dtype=np.bool_)),
                    dtype=np.bool_,
                )
                trajectories = {agent_id: new_trajectory() for agent_id in env.agent_ids}
                episode_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)
            else:
                done_mask = None
                trajectory = new_trajectory()
                episode_reward = 0.0

            global_done = False
            for _step_idx in episode_step_indices(args.max_steps_per_episode):
                if stop_event.is_set() or global_done:
                    break
                if args.multi_agent:
                    action_payload = {}
                    selected_by_agent = {}
                    for agent_idx, agent_id in enumerate(env.agent_ids):
                        if done_mask[agent_idx]:
                            action_payload[agent_id] = zero_env_action(action_meta)
                            continue
                        selected = select_action(
                            sample_fn,
                            local_log_std,
                            obs[agent_idx],
                            action_meta,
                        )
                        action_payload[agent_id] = selected["env_action"]
                        selected_by_agent[agent_id] = (
                            agent_idx,
                            selected,
                            obs[agent_idx].copy(),
                        )
                    step_result = env.step(action_payload)
                else:
                    selected = select_action(sample_fn, local_log_std, obs, action_meta)
                    selected_obs = obs.copy()
                    step_result = env.step(selected["env_action"])

                next_obs, reward, terminated, truncated, step_info = step_result
                # `global_done` ends the rollout; only `terminated` cuts the GAE chain. A
                # step-cap truncation leaves real future value behind, captured below as
                # the trajectory's bootstrap value.
                global_done = bool(terminated or truncated)
                cut_short = bool(truncated and not terminated)
                if args.multi_agent:
                    per_agent_rewards = np.asarray(step_info.get("per_agent_rewards"), dtype=np.float32)
                    per_agent_done = np.asarray(step_info.get("per_agent_done"), dtype=np.bool_)
                    per_agent_terminated = np.asarray(
                        step_info.get("per_agent_terminated", per_agent_done), dtype=np.bool_
                    )
                    episode_reward += per_agent_rewards
                    for agent_id, (agent_idx, selected, agent_obs) in selected_by_agent.items():
                        append_transition(
                            trajectories[agent_id],
                            agent_obs,
                            selected,
                            float(per_agent_rewards[agent_idx]),
                            bool(terminated or per_agent_terminated[agent_idx]),
                        )
                        if cut_short and not per_agent_terminated[agent_idx]:
                            trajectories[agent_id]["bootstrap_value"] = value_of(
                                sample_fn, local_log_std, next_obs[agent_idx], action_meta
                            )
                    done_mask = np.logical_or(done_mask, per_agent_done)
                    global_done = global_done or bool(np.all(done_mask))
                else:
                    append_transition(
                        trajectory,
                        selected_obs,
                        selected,
                        float(reward),
                        bool(terminated),
                    )
                    if cut_short:
                        trajectory["bootstrap_value"] = value_of(
                            sample_fn, local_log_std, next_obs, action_meta
                        )
                    episode_reward += float(reward)
                obs = next_obs

            if stop_event.is_set():
                return
            if args.multi_agent:
                rollout_trajectories = [
                    trajectory
                    for trajectory in trajectories.values()
                    if trajectory_has_samples(trajectory)
                ]
                rewards = episode_reward.tolist()
            else:
                rollout_trajectories = [trajectory] if trajectory_has_samples(trajectory) else []
                rewards = episode_reward
            payload = {
                "policy_version": policy_version,
                "trajectories": rollout_trajectories,
                "rewards": rewards,
            }
            if not put(AsyncEpisodeEvent(worker_id, episode, payload)):
                return
            if episode + 1 < args.num_episodes:
                if not snapshot.wait_for_newer(policy_version, stop_event):
                    return

    pool = AsyncCollectorPool(
        envs,
        worker,
        start_episode,
        args.num_episodes,
        queue_capacity=args.async_queue_capacity,
    )
    print(
        f"Collector mode: async on-policy workers={len(envs)} queue={args.async_queue_capacity} "
        "barrier=policy_generation",
        flush=True,
    )
    completed = int(start_episode)
    pending = {}
    last_saved_episode = None
    interrupted = False
    done_workers = 0
    pool.start()
    try:
        while completed < args.num_episodes:
            apply_ready_best_checkpoint(best_tracker)
            try:
                event = pool.get(timeout=0.2)
            except Empty:
                if done_workers == len(envs):
                    raise RuntimeError("All PPO collectors stopped before training completed")
                continue
            if isinstance(event, AsyncWorkerErrorEvent):
                raise RuntimeError(f"Async PPO collector {event.worker_id} failed") from event.error
            if isinstance(event, AsyncWorkerDoneEvent):
                done_workers += 1
                continue
            if not isinstance(event, AsyncEpisodeEvent):
                continue

            generation = pending.setdefault(event.episode, {})
            generation[event.worker_id] = event.payload
            if event.episode != completed or len(generation) < len(envs):
                continue

            payloads = [generation[worker_id] for worker_id in range(len(envs))]
            versions = {payload["policy_version"] for payload in payloads}
            if len(versions) != 1 or next(iter(versions)) != snapshot.version:
                raise RuntimeError(
                    f"PPO generation {completed} mixed policy versions: {sorted(versions)} "
                    f"current={snapshot.version}"
                )
            trajectories = [
                trajectory
                for payload in payloads
                for trajectory in payload["trajectories"]
            ]
            rewards_summary = [payload["rewards"] for payload in payloads]
            update_batch = build_update_batch(
                trajectories,
                action_meta,
                args.gamma,
                args.gae_lambda,
            )
            if update_batch is None:
                metrics = None
            else:
                metrics = ppo_update(model, log_std, optimizer, update_batch, action_meta, args)

            completed += 1
            policy_version = snapshot.publish(model.get_weights(), state=log_std.numpy())
            training_metrics = [
                ("completed", f"{completed}/{args.num_episodes}"),
                ("queue", pool.events.qsize()),
                ("policy_version", policy_version),
            ]
            if metrics is None:
                training_metrics.append(("skipped", "no_samples"))
            else:
                training_metrics.extend([
                    ("samples", len(update_batch["obs"])),
                    ("loss", f"{metrics['loss']:.5f}"),
                    ("policy_loss", f"{metrics['policy_loss']:.5f}"),
                    ("value_loss", f"{metrics['value_loss']:.5f}"),
                    ("entropy", f"{metrics['entropy']:.5f}"),
                ])
            print_episode_metrics(event.episode, [
                ("mode", [("collector", "async_on_policy"), ("workers", len(envs))]),
                ("outcome", [("rewards", rewards_summary)]),
                ("training", training_metrics),
            ], args.log_format)
            pending.pop(event.episode, None)

            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                saved_path = save_training_checkpoint(checkpoint, checkpoint_manager, completed)
                last_saved_episode = completed
            else:
                saved_path = None
            if best_tracker.should_evaluate(completed):
                if saved_path is None:
                    saved_path = save_training_checkpoint(checkpoint, checkpoint_manager, completed)
                    last_saved_episode = completed
                request_best_checkpoint_evaluation(best_tracker, saved_path, completed)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupt received: stopping async PPO collectors...", flush=True)
    finally:
        pool.close()

    if last_saved_episode != completed:
        save_training_checkpoint(checkpoint, checkpoint_manager, completed, final=True)
    if interrupted:
        print(f"Interrupted async PPO training saved at episode={completed}", flush=True)
    else:
        # Collect the evaluation requested near the last episode. Skipped on interrupt:
        # Ctrl-C should exit, not wait.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
    return completed


def main():
    args = parse_args()
    validate_async_arguments(args)
    best_tracker = BestCheckpointTracker(args, "ppo")
    describe_tensorflow_backend(args)
    random.seed(args.env_seed_base)
    np.random.seed(args.env_seed_base)
    tf.random.set_seed(args.env_seed_base)

    ports = [args.base_port + i for i in range(args.num_envs)]
    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
        scene_path=args.godot_scene,
    )
    envs = []
    model = None
    checkpoint = None
    checkpoint_manager = None
    stepper = None
    start_episode = 0
    last_completed_episode = None
    try:
        manager.start_many(
            ports,
            headless=args.headless,
            debug=args.godot_debug,
            render_env_count=args.render_env_count,
            user_args=build_lockstep_user_args(args),
        )
        print(f"Started Godot instances on ports {ports}", flush=True)
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
        if args.collector_mode == "sync":
            stepper = ParallelEnvStepper(len(envs), args.parallel_env_steps)
            print(f"Environment stepping: {'parallel' if stepper.enabled else 'sequential'}", flush=True)

        env0 = envs[0]
        # 'hybrid' is the general case; 'discrete' is the same policy with no continuous
        # head, which build_action_metadata and build_hybrid_actor_critic already handle.
        # 'continuous' is not: PPO's action space here is built from action_space_spec
        # components, and a bare continuous env exposes no discrete head to sample from.
        if env0.action_type not in SUPPORTED_ACTION_TYPES:
            raise RuntimeError(
                f"algorithms/ppo.py supports action_type in {sorted(SUPPORTED_ACTION_TYPES)}, "
                f"got {env0.action_type!r}"
            )

        obs_dim = env0.obs_dim
        action_meta = build_action_metadata(env0.action_space_spec)
        expected_action_space = json.dumps(env0.action_space_spec, sort_keys=True)
        print(
            f"Scenario spec: agent_id={env0.agent_id} {env0.agent_summary()} multi_agent={args.multi_agent} "
            f"{env0.team_summary()} obs_dim={obs_dim} "
            f"discrete={action_meta['discrete']} continuous={action_meta['continuous']}",
            flush=True,
        )

        for env in envs:
            if env.obs_dim != obs_dim or env.action_type != env0.action_type:
                raise RuntimeError(
                    "All parallel environments must expose the same obs_dim and action_type"
                )
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
        resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_manager)
        if resume_checkpoint:
            checkpoint.restore(resume_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            print(f"Resumed checkpoint {resume_checkpoint} from episode={start_episode}", flush=True)
        optimizer.learning_rate.assign(args.learning_rate)

        opponent_teams = validate_team_layout(envs, args.opponent_pool)
        opponent_pool = OpponentPool(
            args,
            algorithm="ppo",
            model_factory=lambda: build_hybrid_actor_critic(
                obs_dim=obs_dim,
                discrete_sizes=action_meta["discrete_sizes"],
                continuous_size=action_meta["continuous_size"],
            ),
            metadata={"obs_dim": obs_dim, "action_space": env0.action_space_spec},
            state_getter=lambda: log_std.numpy(),
        )
        if opponent_pool.enabled:
            print(
                f"Opponent pool: dir={opponent_pool.directory} teams={opponent_teams} "
                f"snapshots={len(opponent_pool.entries)} sampling={opponent_pool.sampling}",
                flush=True,
            )

        if args.collector_mode == "async":
            last_completed_episode = run_async_ppo(
                args,
                envs,
                model,
                log_std,
                optimizer,
                action_meta,
                obs_dim,
                checkpoint,
                checkpoint_manager,
                best_tracker,
                start_episode,
            )
            model.save_weights(args.weights_path)
            print(f"Saved weights: {args.weights_path}", flush=True)
            return

        # The sync loop forwards on the main thread, so eager would be safe here, but it
        # goes through the same traced sampler as async for one code path. One per model:
        # the learner's, plus any opponent-pool snapshot models, which the pool reuses.
        sample_fns = {}

        def sample_fn_for(sampled_model):
            fn = sample_fns.get(id(sampled_model))
            if fn is None:
                fn = build_sample_action_fn(sampled_model, obs_dim, action_meta)
                sample_fns[id(sampled_model)] = fn
            return fn

        last_saved_episode = None
        for episode in range(start_episode, args.num_episodes):
            opponent_match = opponent_pool.start_episode(model, episode)
            opponent_log_std = (
                tf.convert_to_tensor(opponent_match.state, dtype=tf.float32)
                if opponent_match.state is not None
                else log_std
            )
            trajectories = []
            rewards_summary = []
            env_states = []
            for env_idx, env in enumerate(envs):
                env.configure(
                    training_episode=episode,
                    max_steps=args.max_steps_per_episode,
                    physics_frames_per_step=args.physics_frames_per_step,
                )
                obs, _ = env.reset(seed=args.episode_seed_multiplier * episode + env_idx)
                if args.multi_agent:
                    learner_mask = opponent_pool.learner_mask(
                        env.agent_team_ids,
                        opponent_teams,
                        episode,
                        env_idx,
                        use_current_policy=opponent_match.use_current_policy,
                    )
                    env_trajectories = {
                        agent_id: new_trajectory()
                        for agent_idx, agent_id in enumerate(env.agent_ids)
                        if learner_mask[agent_idx]
                    }
                    done_mask = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    episode_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)
                    env_states.append({
                        "obs": obs,
                        "done": False,
                        "learner_mask": learner_mask,
                        "trajectories": env_trajectories,
                        "done_mask": done_mask,
                        "episode_reward": episode_reward,
                    })
                else:
                    env_states.append({
                        "obs": obs,
                        "done": False,
                        "trajectory": new_trajectory(),
                        "episode_reward": 0.0,
                    })

            for _step in episode_step_indices(args.max_steps_per_episode):
                step_requests = []
                for env, state in zip(envs, env_states):
                    if state["done"]:
                        continue
                    if args.multi_agent:
                        action_payload = {}
                        selected_by_agent = {}
                        for agent_idx, agent_id in enumerate(env.agent_ids):
                            if state["done_mask"][agent_idx]:
                                action_payload[agent_id] = zero_env_action(action_meta)
                                continue
                            is_learner = bool(state["learner_mask"][agent_idx])
                            action_model = model if is_learner else opponent_match.model
                            action_log_std = log_std if is_learner else opponent_log_std
                            selected = select_action(
                                sample_fn_for(action_model),
                                action_log_std,
                                state["obs"][agent_idx],
                                action_meta,
                            )
                            action_payload[agent_id] = selected["env_action"]
                            if is_learner:
                                selected_by_agent[agent_id] = (
                                    agent_idx,
                                    selected,
                                    state["obs"][agent_idx].copy(),
                                )
                        state["selected_by_agent"] = selected_by_agent
                        step_requests.append((env, state, action_payload))
                    else:
                        selected = select_action(sample_fn_for(model), log_std, state["obs"], action_meta)
                        state["selected"] = selected
                        state["selected_obs"] = state["obs"].copy()
                        step_requests.append((env, state, selected["env_action"]))

                for env, state, _action, step_result in stepper.step(step_requests):
                    next_obs, reward, terminated, truncated, info = step_result
                    # `global_done` ends the rollout; only `terminated` cuts the GAE chain.
                    # A step-cap truncation leaves real future value behind, captured below
                    # as the trajectory's bootstrap value.
                    global_done = bool(terminated or truncated)
                    cut_short = bool(truncated and not terminated)
                    if args.multi_agent:
                        per_agent_rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
                        per_agent_done = np.asarray(info.get("per_agent_done"), dtype=np.bool_)
                        per_agent_terminated = np.asarray(
                            info.get("per_agent_terminated", per_agent_done), dtype=np.bool_
                        )
                        state["episode_reward"] += per_agent_rewards
                        for agent_id, (agent_idx, selected, agent_obs) in state["selected_by_agent"].items():
                            append_transition(
                                state["trajectories"][agent_id],
                                agent_obs,
                                selected,
                                float(per_agent_rewards[agent_idx]),
                                bool(terminated or per_agent_terminated[agent_idx]),
                            )
                            if cut_short and not per_agent_terminated[agent_idx]:
                                state["trajectories"][agent_id]["bootstrap_value"] = value_of(
                                    sample_fn_for(model), log_std, next_obs[agent_idx], action_meta
                                )
                        state["done_mask"] = np.logical_or(state["done_mask"], per_agent_done)
                        state["done"] = global_done or bool(np.all(state["done_mask"]))
                    else:
                        append_transition(
                            state["trajectory"],
                            state["selected_obs"],
                            state["selected"],
                            float(reward),
                            bool(terminated),
                        )
                        if cut_short:
                            state["trajectory"]["bootstrap_value"] = value_of(
                                sample_fn_for(model), log_std, next_obs, action_meta
                            )
                        state["episode_reward"] += float(reward)
                        state["done"] = global_done
                    state["obs"] = next_obs

                if all(state["done"] for state in env_states):
                    break

            for state in env_states:
                if args.multi_agent:
                    trajectories.extend(
                        trajectory
                        for trajectory in state["trajectories"].values()
                        if trajectory_has_samples(trajectory)
                    )
                    rewards_summary.append(state["episode_reward"].tolist())
                else:
                    if trajectory_has_samples(state["trajectory"]):
                        trajectories.append(state["trajectory"])
                    rewards_summary.append(state["episode_reward"])

            update_batch = build_update_batch(
                trajectories,
                action_meta,
                args.gamma,
                args.gae_lambda,
            )
            if update_batch is None:
                print_episode_metrics(episode, [
                    ("outcome", [("rewards", rewards_summary)]),
                    ("training", [("skipped", "no_samples")]),
                ], args.log_format)
                last_completed_episode = episode + 1
                continue

            metrics = ppo_update(model, log_std, optimizer, update_batch, action_meta, args)
            sample_count = len(update_batch["rewards"]) if "rewards" in update_batch else len(update_batch["obs"])
            print_episode_metrics(episode, [
                ("mode", [("opponent", opponent_match.label)]),
                ("outcome", [("rewards", rewards_summary)]),
                ("training", [
                    ("samples", sample_count),
                    ("loss", f"{metrics['loss']:.5f}"),
                    ("policy_loss", f"{metrics['policy_loss']:.5f}"),
                    ("value_loss", f"{metrics['value_loss']:.5f}"),
                    ("entropy", f"{metrics['entropy']:.5f}"),
                ]),
            ], args.log_format)
            last_completed_episode = episode + 1
            snapshot_path = opponent_pool.snapshot(model, episode + 1)
            if snapshot_path is not None:
                print(f"Saved opponent snapshot: {snapshot_path}", flush=True)

            apply_ready_best_checkpoint(best_tracker)
            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                saved_path = save_training_checkpoint(checkpoint, checkpoint_manager, episode + 1)
                last_saved_episode = episode + 1
            else:
                saved_path = None
            if best_tracker.should_evaluate(episode + 1):
                if saved_path is None:
                    saved_path = save_training_checkpoint(checkpoint, checkpoint_manager, episode + 1)
                    last_saved_episode = episode + 1
                request_best_checkpoint_evaluation(best_tracker, saved_path, episode + 1)

        if last_saved_episode != args.num_episodes:
            save_training_checkpoint(checkpoint, checkpoint_manager, args.num_episodes, final=True)
        # The evaluation requested on the last episode is still running; without this the
        # finally-block's close() cancels it and a final best can never be promoted.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
        model.save_weights(args.weights_path)
        print(f"Saved weights: {args.weights_path}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupt received: saving the last consistent PPO state...", flush=True)
        if checkpoint is not None and checkpoint_manager is not None and model is not None:
            interrupted_episode = last_completed_episode if last_completed_episode is not None else start_episode
            save_training_checkpoint(checkpoint, checkpoint_manager, interrupted_episode)
            model.save_weights(args.weights_path)
            print(f"Interrupted training saved at episode={interrupted_episode}", flush=True)
        else:
            print("Training state was not initialized; no checkpoint was written.", flush=True)
    finally:
        best_tracker.close()
        if stepper is not None:
            stepper.close()
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        manager.stop_all()


if __name__ == "__main__":
    main()
