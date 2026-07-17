import argparse
import os
import platform
import random
import sys
import sysconfig
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
        os.execv(sys.executable, [sys.executable] + sys.argv)


configure_tensorflow_runtime()

import numpy as np
import tensorflow as tf

from godot_process_manager import GodotProcessManager
from models import build_continuous_actor, build_continuous_critic
from opponent_pool import OpponentPool, add_opponent_pool_arguments, validate_team_layout
from replay_buffer import ReplayBuffer
from scenario_gym_env import ScenarioGymEnv
from training_support import (
    AsyncCollectorPool,
    AsyncEventScheduler,
    AsyncEpisodeEvent,
    AsyncStepEvent,
    AsyncWorkerDoneEvent,
    AsyncWorkerErrorEvent,
    ParallelEnvStepper,
    PolicySnapshot,
    BestCheckpointTracker,
    add_best_checkpoint_arguments,
    add_collector_arguments,
    add_lockstep_tuning_arguments,
    add_parallel_env_arguments,
    add_log_format_argument,
    add_tensorflow_runtime_arguments,
    build_async_worker,
    build_lockstep_user_args,
    configure_tensorflow_devices,
    episode_step_indices,
    print_episode_metrics,
    resolve_resume_checkpoint,
    restore_replay_buffer,
    save_replay_snapshot,
    validate_async_arguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generic DDPG trainer for continuous Godot scenarios.")
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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration-noise", type=float, default=0.2)
    parser.add_argument("--exploration-noise-min", type=float, default=0.02)
    parser.add_argument("--exploration-noise-decay", type=float, default=0.995)
    parser.add_argument("--exploration-noise-kind", choices=["ou", "gaussian"], default="ou")
    parser.add_argument("--ou-theta", type=float, default=0.15)
    parser.add_argument("--action-smoothing", type=float, default=0.2)
    parser.add_argument("--random-exploration-episodes", type=int, default=15)
    parser.add_argument("--random-drive-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--random-steering-abs-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--actor-drive-prior", type=float, default=0.75)
    parser.add_argument("--actor-steering-prior", type=float, default=0.0)
    parser.add_argument("--actor-drive-regularization", type=float, default=0.05)
    parser.add_argument("--actor-drive-target", type=float, default=0.65)
    parser.add_argument("--reset-progress-curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-progress-start-max", type=float, default=0.025)
    parser.add_argument("--reset-progress-end-max", type=float, default=0.35)
    parser.add_argument("--reset-progress-ramp-episodes", type=int, default=400)
    parser.add_argument("--replay-warmup", type=int, default=500)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--critic-warmup-updates", type=int, default=2000)
    parser.add_argument("--target-update-every", type=int, default=1)
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-weights-path", default="generic_ddpg_actor.weights.h5")
    parser.add_argument("--critic-weights-path", default="generic_ddpg_critic.weights.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_ddpg")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-replay-buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--demo-path", action="append", default=[])
    parser.add_argument("--demo-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-max-transitions", type=int, default=0)
    parser.add_argument("--demo-bc-epochs", type=int, default=0)
    parser.add_argument("--demo-bc-batch-size", type=int, default=128)
    parser.add_argument("--demo-bc-learning-rate", type=float, default=None)
    parser.add_argument("--demo-bc-on-resume", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--log-details", action=argparse.BooleanOptionalAction, default=False)
    add_collector_arguments(parser)
    add_parallel_env_arguments(parser)
    add_lockstep_tuning_arguments(parser)
    add_opponent_pool_arguments(parser)
    add_best_checkpoint_arguments(parser)
    add_log_format_argument(parser)
    add_tensorflow_runtime_arguments(parser)
    return parser.parse_args()


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


def scale_action_numpy(raw_action, low, high):
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def scale_action_tensor(raw_action, low, high):
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def select_action(actor, obs, low, high, noise_std, noise_kind="ou", noise_state=None, ou_theta=0.15):
    raw_action = actor(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    action = scale_action_numpy(raw_action, low, high)
    if noise_std > 0:
        if noise_kind == "ou" and noise_state is not None:
            noise_state += ou_theta * (0.0 - noise_state) + np.random.normal(0.0, noise_std, size=action.shape)
            action = action + noise_state
        else:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
    return np.clip(action, low, high).astype(np.float32)


def select_actions(actor, obs_batch, done_mask, low, high, noise_std, noise_kind="ou", noise_state=None, ou_theta=0.15):
    actions = np.zeros((obs_batch.shape[0], low.shape[0]), dtype=np.float32)
    for idx, obs in enumerate(obs_batch):
        if not done_mask[idx]:
            agent_noise_state = None if noise_state is None else noise_state[idx]
            actions[idx] = select_action(
                actor,
                obs,
                low,
                high,
                noise_std,
                noise_kind=noise_kind,
                noise_state=agent_noise_state,
                ou_theta=ou_theta,
            )
    return actions


def smooth_actions(actions, previous_actions, low, high, smoothing):
    smoothing = float(np.clip(smoothing, 0.0, 1.0))
    if smoothing <= 0.0:
        return np.clip(actions, low, high).astype(np.float32)

    smoothed = np.asarray(actions, dtype=np.float32).copy()
    previous = np.asarray(previous_actions, dtype=np.float32)
    if smoothed.ndim == 1:
        if np.all(np.isfinite(previous)):
            smoothed = previous + smoothing * (smoothed - previous)
    else:
        valid = np.all(np.isfinite(previous), axis=1)
        smoothed[valid] = previous[valid] + smoothing * (smoothed[valid] - previous[valid])

    return np.clip(smoothed, low, high).astype(np.float32)


def continuous_exploration_bounds(action_space_spec, low, high):
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    exploration_low = low.copy()
    exploration_high = high.copy()
    action_space_spec = action_space_spec or {}

    offset = 0
    for component in action_space_spec.values():
        if str(component.get("action_type", component.get("type", "discrete"))) != "continuous":
            continue

        size = int(component.get("size", 1))
        component_low = component_bound(component.get("exploration_low", None), size, None)
        component_high = component_bound(component.get("exploration_high", None), size, None)
        for idx in range(size):
            action_idx = offset + idx
            if action_idx >= low.shape[0]:
                break
            if component_low is not None:
                exploration_low[action_idx] = float(component_low[idx])
            if component_high is not None:
                exploration_high[action_idx] = float(component_high[idx])
        offset += size

    exploration_low = np.maximum(exploration_low, low)
    exploration_high = np.minimum(exploration_high, high)
    invalid = exploration_low > exploration_high
    exploration_low[invalid] = low[invalid]
    exploration_high[invalid] = high[invalid]
    return exploration_low.astype(np.float32), exploration_high.astype(np.float32)


def component_bound(value, size, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if not values:
        return default
    if len(values) == 1:
        return [float(values[0])] * size
    return [float(values[idx]) if idx < len(values) else float(values[-1]) for idx in range(size)]


def legacy_exploration_bounds(low, high, action_names, drive_min=None, steering_abs_max=None):
    exploration_low = np.asarray(low, dtype=np.float32).copy()
    exploration_high = np.asarray(high, dtype=np.float32).copy()
    if drive_min is None and steering_abs_max is None:
        return exploration_low, exploration_high

    names = [str(name).lower() for name in action_names]
    for idx, name in enumerate(names):
        if idx >= exploration_low.shape[0]:
            continue

        if drive_min is not None and any(token in name for token in ("throttle", "drive", "move", "accelerate")):
            exploration_low[idx] = max(float(exploration_low[idx]), min(float(exploration_high[idx]), float(drive_min)))
            continue

        if steering_abs_max is not None and any(token in name for token in ("rotation", "steer", "turn")):
            abs_max = abs(float(steering_abs_max))
            exploration_low[idx] = max(float(exploration_low[idx]), -abs_max)
            exploration_high[idx] = min(float(exploration_high[idx]), abs_max)

    return exploration_low, exploration_high


def sample_exploratory_action(
    low,
    high,
    action_names,
    rng,
    num_agents=None,
    drive_min=None,
    steering_abs_max=None,
    exploration_low=None,
    exploration_high=None,
):
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if exploration_low is None or exploration_high is None:
        exploration_low, exploration_high = legacy_exploration_bounds(
            low,
            high,
            action_names,
            drive_min=drive_min,
            steering_abs_max=steering_abs_max,
        )
    exploration_low = np.asarray(exploration_low, dtype=np.float32)
    exploration_high = np.asarray(exploration_high, dtype=np.float32)
    shape = (int(num_agents), low.shape[0]) if num_agents is not None else low.shape
    action = rng.uniform(exploration_low, exploration_high, size=shape).astype(np.float32)
    return np.clip(action, low, high).astype(np.float32)


def initialize_actor_action_prior(actor, action_low, action_high, action_names, drive_prior=0.75, steering_prior=0.0):
    final_layer = actor.layers[-1]
    weights = final_layer.get_weights()
    if len(weights) != 2:
        return

    kernel, bias = weights
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    names = [str(name).lower() for name in action_names]

    # Start DDPG from a sane driving policy: forward throttle and neutral steering.
    kernel[:] = 0.0
    for idx, name in enumerate(names):
        if idx >= bias.shape[0]:
            continue

        desired = None
        if any(token in name for token in ("throttle", "drive", "move", "accelerate")):
            desired = drive_prior
        elif any(token in name for token in ("rotation", "steer", "turn")):
            desired = steering_prior

        if desired is None:
            continue

        desired = float(np.clip(desired, low[idx], high[idx]))
        raw = 2.0 * (desired - low[idx]) / max(float(high[idx] - low[idx]), 1e-6) - 1.0
        raw = float(np.clip(raw, -0.95, 0.95))
        bias[idx] = np.arctanh(raw)

    final_layer.set_weights([kernel, bias])


def drive_action_indices(action_names):
    indices = []
    for idx, name in enumerate(action_names):
        normalized = str(name).lower()
        if any(token in normalized for token in ("throttle", "drive", "move", "accelerate")):
            indices.append(idx)
    return indices


def curriculum_reset_progress_max(episode, args):
    if not args.reset_progress_curriculum:
        return None
    ramp_episodes = max(int(args.reset_progress_ramp_episodes), 1)
    ratio = min(1.0, max(0.0, float(episode) / float(ramp_episodes)))
    start_value = float(args.reset_progress_start_max)
    end_value = float(args.reset_progress_end_max)
    return start_value + ratio * (end_value - start_value)


def update_episode_diagnostics(state, agent_info, agent_idx=None):
    progress = float(agent_info.get("track_progress", 0.0))
    finish_reached = bool(agent_info.get("finish_reached", agent_info.get("target_reached", False)))
    progress_stalled = bool(agent_info.get("progress_stalled", False))
    local_terms = agent_info.get("local_term_rewards", {}) or {}
    collision_term = float(local_terms.get("collision", 0.0))
    collision_seen = abs(collision_term) > 1e-9

    if agent_idx is None:
        state["max_track_progress"] = max(float(state["max_track_progress"]), progress)
        state["last_track_progress"] = progress
        state["finish_reached"] = bool(state["finish_reached"] or finish_reached)
        if collision_seen and not state["collision_seen"]:
            state["collision_count"] += 1
            state["collision_seen"] = True
        if progress_stalled and not state["stalled_seen"]:
            state["stalled_count"] += 1
            state["stalled_seen"] = True
        return

    state["max_track_progress"][agent_idx] = max(float(state["max_track_progress"][agent_idx]), progress)
    state["last_track_progress"][agent_idx] = progress
    state["finish_reached"][agent_idx] = bool(state["finish_reached"][agent_idx] or finish_reached)
    if collision_seen and not state["collision_seen"][agent_idx]:
        state["collision_count"][agent_idx] += 1
        state["collision_seen"][agent_idx] = True
    if progress_stalled and not state["stalled_seen"][agent_idx]:
        state["stalled_count"][agent_idx] += 1
        state["stalled_seen"][agent_idx] = True


def summarize_episode_diagnostics(env_states, multi_agent):
    if multi_agent:
        max_progress_arrays = [state["max_track_progress"] for state in env_states]
        last_progress_arrays = [state["last_track_progress"] for state in env_states]
        finish_arrays = [state["finish_reached"] for state in env_states]
        collision_arrays = [state["collision_count"] for state in env_states]
        stalled_arrays = [state["stalled_count"] for state in env_states]
        return {
            "progress_max": float(max(np.max(values) for values in max_progress_arrays)),
            "progress_mean": float(np.mean(np.concatenate(last_progress_arrays))),
            "finishes": int(sum(np.count_nonzero(values) for values in finish_arrays)),
            "collisions": int(sum(np.sum(values) for values in collision_arrays)),
            "stalls": int(sum(np.sum(values) for values in stalled_arrays)),
        }

    return {
        "progress_max": float(max(state["max_track_progress"] for state in env_states)),
        "progress_mean": float(np.mean([state["last_track_progress"] for state in env_states])),
        "finishes": int(sum(1 for state in env_states if state["finish_reached"])),
        "collisions": int(sum(state["collision_count"] for state in env_states)),
        "stalls": int(sum(state["stalled_count"] for state in env_states)),
    }


def summarize_rewards(env_states):
    values = []
    for state in env_states:
        reward_value = state["ep_reward"]
        if hasattr(reward_value, "reshape"):
            values.extend(np.asarray(reward_value, dtype=np.float32).reshape(-1).tolist())
        else:
            values.append(float(reward_value))

    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    reward_array = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(reward_array)),
        "min": float(np.min(reward_array)),
        "max": float(np.max(reward_array)),
    }


def summarize_actions(mean_actions):
    values = np.asarray(mean_actions, dtype=np.float32)
    if values.size == 0:
        return []
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return np.mean(values.reshape(-1, values.shape[-1]), axis=0).tolist()


def summarize_action_deltas(env_states, multi_agent):
    values = []
    for state in env_states:
        delta_sum = state["action_delta_sum"]
        delta_count = state["action_delta_count"]
        if multi_agent:
            values.append(delta_sum / np.maximum(delta_count, 1.0))
        else:
            values.append(delta_sum / max(float(delta_count), 1.0))

    if not values:
        return []

    return np.mean(np.asarray(values, dtype=np.float32).reshape(-1, values[0].shape[-1]), axis=0).tolist()


def format_float_list(values, precision=3):
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def create_continuous_async_worker(
    args,
    envs,
    local_models,
    policy_snapshot,
    action_size,
    action_selector,
):
    def begin_episode(worker_id, env, episode):
        reset_progress_max = curriculum_reset_progress_max(episode, args)
        scenario_config = {
            "training_episode": episode,
            "max_steps": args.max_steps_per_episode,
            "physics_frames_per_step": args.physics_frames_per_step,
        }
        if reset_progress_max is not None:
            scenario_config.update(reset_progress_min=0.0, reset_progress_max=reset_progress_max)
        env.configure(**scenario_config)
        obs, info = env.reset(seed=args.episode_seed_multiplier * episode + worker_id)
        agent_count = len(env.agent_ids) if args.multi_agent else 1
        state = {
            "obs": obs,
            "done": False,
            "reset_progress_max": reset_progress_max,
            "use_random_exploration": episode < max(0, args.random_exploration_episodes),
        }
        if args.multi_agent:
            state.update({
                "done_mask": np.asarray(
                    info.get("per_agent_done", np.zeros((agent_count,), dtype=np.bool_)),
                    dtype=np.bool_,
                ),
                "ep_reward": np.zeros((agent_count,), dtype=np.float32),
                "action_sum": np.zeros((agent_count, action_size), dtype=np.float32),
                "action_count": np.zeros((agent_count, 1), dtype=np.float32),
                "previous_action": np.full((agent_count, action_size), np.nan, dtype=np.float32),
                "action_delta_sum": np.zeros((agent_count, action_size), dtype=np.float32),
                "action_delta_count": np.zeros((agent_count, 1), dtype=np.float32),
                "max_track_progress": np.zeros((agent_count,), dtype=np.float32),
                "last_track_progress": np.zeros((agent_count,), dtype=np.float32),
                "finish_reached": np.zeros((agent_count,), dtype=np.bool_),
                "collision_seen": np.zeros((agent_count,), dtype=np.bool_),
                "collision_count": np.zeros((agent_count,), dtype=np.int32),
                "stalled_seen": np.zeros((agent_count,), dtype=np.bool_),
                "stalled_count": np.zeros((agent_count,), dtype=np.int32),
            })
        else:
            state.update({
                "done_mask": None,
                "ep_reward": 0.0,
                "action_sum": np.zeros((action_size,), dtype=np.float32),
                "action_count": 0.0,
                "previous_action": np.full((action_size,), np.nan, dtype=np.float32),
                "action_delta_sum": np.zeros((action_size,), dtype=np.float32),
                "action_delta_count": 0.0,
                "max_track_progress": 0.0,
                "last_track_progress": 0.0,
                "finish_reached": False,
                "collision_seen": False,
                "collision_count": 0,
                "stalled_seen": False,
                "stalled_count": 0,
            })
        return state

    def choose_action(worker_id, env, episode, step_idx, local_model, state):
        return action_selector(worker_id, env, episode, step_idx, local_model, state)

    def process_step(_worker_id, env, _episode, _step_idx, state, action, step_result):
        next_obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        transitions = []
        if args.multi_agent:
            rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
            done_mask = np.asarray(info.get("per_agent_done"), dtype=np.bool_)
            terminated_mask = np.asarray(
                info.get("per_agent_terminated", done_mask), dtype=np.bool_
            )
            infos = list(info.get("per_agent_infos", []))
            for agent_idx in range(len(env.agent_ids)):
                was_done = bool(state["done_mask"][agent_idx])
                if was_done and done_mask[agent_idx]:
                    continue
                agent_info = infos[agent_idx] if agent_idx < len(infos) else {}
                update_episode_diagnostics(state, agent_info, agent_idx=agent_idx)
                transitions.append((
                    state["obs"][agent_idx],
                    np.asarray(action[agent_idx], dtype=np.float32),
                    float(rewards[agent_idx]),
                    next_obs[agent_idx],
                    bool(terminated_mask[agent_idx] or terminated),
                ))
                state["ep_reward"][agent_idx] += rewards[agent_idx]
                state["action_sum"][agent_idx] += action[agent_idx]
                state["action_count"][agent_idx, 0] += 1.0
                previous = state["previous_action"][agent_idx]
                if np.all(np.isfinite(previous)):
                    state["action_delta_sum"][agent_idx] += np.abs(action[agent_idx] - previous)
                    state["action_delta_count"][agent_idx, 0] += 1.0
                state["previous_action"][agent_idx] = action[agent_idx]
            state["done_mask"] = done_mask
        else:
            update_episode_diagnostics(state, info.get("agent_info", {}))
            transitions.append((
                state["obs"],
                np.asarray(action, dtype=np.float32),
                float(reward),
                next_obs,
                bool(terminated),
            ))
            state["ep_reward"] += float(reward)
            state["action_sum"] += action
            state["action_count"] += 1.0
            if np.all(np.isfinite(state["previous_action"])):
                state["action_delta_sum"] += np.abs(action - state["previous_action"])
                state["action_delta_count"] += 1.0
            state["previous_action"] = action
        state["obs"] = next_obs
        state["done"] = done
        return transitions

    return build_async_worker(
        local_models,
        policy_snapshot,
        args.max_steps_per_episode,
        args.async_policy_sync_steps,
        begin_episode,
        choose_action,
        process_step,
        lambda _worker_id, _env, _episode, state: state,
    )


def soft_update(target_model, source_model, tau):
    target_weights = target_model.get_weights()
    source_weights = source_model.get_weights()
    updated = [
        (1.0 - tau) * target_weight + tau * source_weight
        for target_weight, source_weight in zip(target_weights, source_weights)
    ]
    target_model.set_weights(updated)


def train_step(
    actor,
    critic,
    target_actor,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    buffer,
    batch_size,
    gamma,
    action_low,
    action_high,
    drive_indices=None,
    actor_drive_regularization=0.0,
    actor_drive_target=0.65,
    update_actor=True,
):
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, action_dtype=np.float32)
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards.reshape(-1, 1), dtype=tf.float32)
    next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones.reshape(-1, 1), dtype=tf.float32)
    action_low = tf.convert_to_tensor(np.asarray(action_low, dtype=np.float32).reshape(1, -1), dtype=tf.float32)
    action_high = tf.convert_to_tensor(np.asarray(action_high, dtype=np.float32).reshape(1, -1), dtype=tf.float32)

    next_actions = scale_action_tensor(target_actor(next_obs, training=False), action_low, action_high)
    target_q = target_critic([next_obs, next_actions], training=False)
    y = rewards + (1.0 - dones) * gamma * target_q

    with tf.GradientTape() as tape:
        q = critic([obs, actions], training=True)
        critic_loss = tf.reduce_mean(tf.square(y - q))
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    actor_loss_value = None
    if update_actor:
        with tf.GradientTape() as tape:
            policy_actions = scale_action_tensor(actor(obs, training=True), action_low, action_high)
            actor_loss = -tf.reduce_mean(critic([obs, policy_actions], training=False))
            if drive_indices and actor_drive_regularization > 0.0:
                drive_values = tf.gather(policy_actions, drive_indices, axis=1)
                drive_deficit = tf.nn.relu(float(actor_drive_target) - drive_values)
                actor_loss = actor_loss + float(actor_drive_regularization) * tf.reduce_mean(tf.square(drive_deficit))
        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
        actor_loss_value = float(actor_loss.numpy())

    return actor_loss_value, float(critic_loss.numpy())


def save_training_checkpoint(
    checkpoint,
    checkpoint_manager,
    buffer,
    episode,
    noise_std,
    args,
    final=False,
    save_replay=True,
):
    checkpoint.episode.assign(episode)
    checkpoint.noise_std.assign(noise_std)
    saved_path = checkpoint_manager.save(checkpoint_number=episode)
    print(f"Saved {'final checkpoint' if final else 'checkpoint'}: {saved_path}", flush=True)
    if save_replay and args.save_replay_buffer and len(buffer) > 0:
        save_replay_snapshot(
            saved_path,
            checkpoint_manager,
            buffer,
            asynchronous=args.collector_mode == "async" and args.async_replay_save,
        )
    return saved_path


def request_best_checkpoint_evaluation(tracker, candidate_path, episode):
    tracker.evaluate_async(candidate_path, episode)


def apply_ready_best_checkpoint(tracker, best_checkpoint_manager, checkpoint, buffer, current_episode, noise_std, args):
    result = tracker.poll_ready()
    if result is None or not tracker.is_improvement(result):
        return None
    best_path = save_training_checkpoint(
        checkpoint,
        best_checkpoint_manager,
        buffer,
        current_episode,
        noise_std,
        args,
        save_replay=False,
    )
    if int(current_episode) != int(result.episode):
        print(
            f"Best checkpoint applied using live weights at episode={current_episode} "
            f"(background evaluation was requested for episode={result.episode})",
            flush=True,
        )
    tracker.record_best(result, best_path)
    return best_path


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


def run_async_ddpg(
    args,
    envs,
    actor,
    critic,
    target_actor,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    buffer,
    checkpoint,
    checkpoint_manager,
    best_checkpoint_manager,
    best_tracker,
    start_episode,
    start_noise_std,
    action_low,
    action_high,
    random_action_low,
    random_action_high,
    actor_drive_indices,
    critic_warmup_target,
):
    validate_async_arguments(args)
    obs_dim = envs[0].obs_dim
    action_size = envs[0].action_size
    with tf.device("/CPU:0"):
        local_models = [
            build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
            for _env in envs
        ]
    snapshot = PolicySnapshot(actor.get_weights())
    rngs = [np.random.default_rng(args.env_seed_base + 100_003 * idx) for idx in range(len(envs))]

    def noise_for_episode(episode):
        elapsed = max(0, int(episode) - int(start_episode))
        return max(
            args.exploration_noise_min,
            float(start_noise_std) * (args.exploration_noise_decay ** elapsed),
        )

    def action_selector(worker_id, env, episode, _step_idx, local_actor, state):
        rng = rngs[worker_id]
        if state["use_random_exploration"]:
            action = sample_exploratory_action(
                action_low,
                action_high,
                env.action_names,
                rng=rng,
                num_agents=len(env.agent_ids) if args.multi_agent else None,
                drive_min=args.random_drive_min,
                steering_abs_max=args.random_steering_abs_max,
                exploration_low=random_action_low,
                exploration_high=random_action_high,
            )
            if args.multi_agent:
                action[state["done_mask"]] = action_low
        else:
            observations = np.asarray(state["obs"], dtype=np.float32)
            is_batch = args.multi_agent
            if not is_batch:
                observations = np.expand_dims(observations, axis=0)
            raw_action = local_actor(observations, training=False).numpy()
            action = scale_action_numpy(raw_action, action_low, action_high)
            noise_std = noise_for_episode(episode)
            noise_state = state.get("exploration_noise_state")
            expected_shape = action.shape
            if noise_state is None:
                noise_state = np.zeros(expected_shape, dtype=np.float32)
            if args.exploration_noise_kind == "ou":
                noise_state += args.ou_theta * (0.0 - noise_state) + rng.normal(
                    0.0, noise_std, size=expected_shape
                )
                noise = noise_state
                state["exploration_noise_state"] = noise_state
            else:
                noise = rng.normal(0.0, noise_std, size=expected_shape)
            action = np.clip(action + noise, action_low, action_high).astype(np.float32)
            if not is_batch:
                action = action[0]
            elif np.any(state["done_mask"]):
                action[state["done_mask"]] = action_low
        return smooth_actions(
            action,
            state["previous_action"],
            action_low,
            action_high,
            args.action_smoothing,
        )

    worker = create_continuous_async_worker(
        args,
        envs,
        local_models,
        snapshot,
        action_size,
        action_selector,
    )
    pool = AsyncCollectorPool(
        envs,
        worker,
        start_episode,
        args.num_episodes,
        queue_capacity=args.async_queue_capacity,
    )
    scheduler = AsyncEventScheduler(args)
    print(
        f"Collector mode: async workers={len(envs)} queue={args.async_queue_capacity} "
        f"policy_sync_steps={args.async_policy_sync_steps} "
        f"update_basis={scheduler.update_basis} update_every={scheduler.update_every} "
        f"updates_per_interval={scheduler.updates_per_interval} "
        f"max_updates_per_env_step={scheduler.max_updates_per_env_step}",
        flush=True,
    )
    completed = int(start_episode)
    done_workers = 0
    learner_updates = 0
    critic_updates_since_resume = 0
    actor_losses = []
    critic_losses = []
    last_saved_episode = None
    interrupted = False
    pool.start()
    try:
        while done_workers < len(envs):
            apply_ready_best_checkpoint(
                best_tracker, best_checkpoint_manager, checkpoint, buffer, completed, noise_for_episode(completed), args
            )
            try:
                event = scheduler.next_event(pool, timeout=0.2)
            except Empty:
                continue
            if isinstance(event, AsyncWorkerErrorEvent):
                raise RuntimeError(f"Async collector {event.worker_id} failed") from event.error
            if isinstance(event, AsyncWorkerDoneEvent):
                done_workers += 1
                continue
            if isinstance(event, AsyncStepEvent):
                step_events = scheduler.drain_step_events(pool, event)
                for step_event in step_events:
                    for transition in step_event.transitions:
                        buffer.add(*transition)
                updates_due = scheduler.ingest(step_events)
                updates_performed = 0
                if len(buffer) >= max(args.replay_warmup, args.batch_size):
                    for _update in range(updates_due):
                        update_actor = critic_updates_since_resume >= critic_warmup_target
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
                            action_low,
                            action_high,
                            drive_indices=actor_drive_indices,
                            actor_drive_regularization=args.actor_drive_regularization,
                            actor_drive_target=args.actor_drive_target,
                            update_actor=update_actor,
                        )
                        critic_losses.append(critic_loss)
                        critic_updates_since_resume += 1
                        learner_updates += 1
                        updates_performed += 1
                        if actor_loss is not None:
                            actor_losses.append(actor_loss)
                        if learner_updates % args.target_update_every == 0:
                            soft_update(target_actor, actor, args.tau)
                            soft_update(target_critic, critic, args.tau)
                        if learner_updates % args.async_policy_publish_updates == 0:
                            snapshot.publish(actor.get_weights())
                scheduler.record_updates(updates_performed)
                continue

            if not isinstance(event, AsyncEpisodeEvent):
                continue
            completed += 1
            state = event.payload
            noise_std = noise_for_episode(completed)
            reward_stats = summarize_rewards([state])
            diagnostics = summarize_episode_diagnostics([state], args.multi_agent)
            if args.multi_agent:
                mean_actions = [(state["action_sum"] / np.maximum(state["action_count"], 1.0)).tolist()]
                controlled_agents = len(state["ep_reward"])
            else:
                mean_actions = [(state["action_sum"] / max(float(state["action_count"]), 1.0)).tolist()]
                controlled_agents = 1
            mean_action = summarize_actions(mean_actions)
            mean_delta = summarize_action_deltas([state], args.multi_agent)
            warmup_left = max(0, critic_warmup_target - critic_updates_since_resume)
            throughput = scheduler.throughput(pool)
            print_episode_metrics(event.episode, [
                ("mode", [
                    ("collector", "async"),
                    ("worker", event.worker_id),
                    ("exploration", "random" if state["use_random_exploration"] else "policy"),
                    ("noise", f"{noise_std:.3f}"),
                ]),
                ("outcome", [
                    ("reward", f"{reward_stats['mean']:.3f} [{reward_stats['min']:.3f}, {reward_stats['max']:.3f}]"),
                    ("progress", f"mean:{diagnostics['progress_mean']:.3f} max:{diagnostics['progress_max']:.3f}"),
                ]),
                ("agents", [
                    ("finish", f"{diagnostics['finishes']}/{controlled_agents}"),
                    ("collision", f"{diagnostics['collisions']}/{controlled_agents}"),
                    ("stall", f"{diagnostics['stalls']}/{controlled_agents}"),
                ]),
                ("actions", [("mean", format_float_list(mean_action)), ("delta", format_float_list(mean_delta))]),
                ("training", [
                    ("completed", f"{completed}/{args.num_episodes}"),
                    ("queue", f"{throughput['queue_size']}/{throughput['queue_capacity']} ({throughput['queue_saturation']:.0%})"),
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    ("critic_updates", len(critic_losses)),
                    ("policy_updates", len(actor_losses)),
                    ("warmup_left", warmup_left),
                    ("actor_loss", f"{float(np.mean(actor_losses)) if actor_losses else 0.0:.5f}"),
                    ("critic_loss", f"{float(np.mean(critic_losses)) if critic_losses else 0.0:.5f}"),
                    ("policy_version", snapshot.version),
                ]),
                ("throughput", [
                    ("env_steps_s", f"{throughput['env_steps_s']:.1f}"),
                    ("transitions_s", f"{throughput['transitions_s']:.1f}"),
                    ("transitions_step", f"{throughput['transitions_per_env_step']:.1f}"),
                    ("updates_s", f"{throughput['updates_s']:.1f}"),
                    ("updates_throttled", throughput["throttled_updates"]),
                ]),
            ], args.log_format)
            actor_losses.clear()
            critic_losses.clear()
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                saved_path = save_training_checkpoint(
                    checkpoint, checkpoint_manager, buffer, completed, noise_std, args
                )
                last_saved_episode = completed
            else:
                saved_path = None
            if best_tracker.should_evaluate(completed):
                if saved_path is None:
                    saved_path = save_training_checkpoint(
                        checkpoint, checkpoint_manager, buffer, completed, noise_std, args
                    )
                    last_saved_episode = completed
                request_best_checkpoint_evaluation(best_tracker, saved_path, completed)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupt received: stopping async DDPG collectors...", flush=True)
    finally:
        pool.close()

    noise_std = noise_for_episode(completed)
    if last_saved_episode != completed:
        save_training_checkpoint(
            checkpoint, checkpoint_manager, buffer, completed, noise_std, args, final=True
        )
    if interrupted:
        print(
            f"Interrupted async training saved at completed_episodes={completed}",
            flush=True,
        )
    return completed, noise_std


def pretrain_actor_behavior_cloning(actor, demo_data, epochs, batch_size, learning_rate, action_low, action_high):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    obs = demo_data["obs"]
    actions = demo_data["actions"]
    action_low_tensor = tf.convert_to_tensor(np.asarray(action_low, dtype=np.float32).reshape(1, -1), dtype=tf.float32)
    action_high_tensor = tf.convert_to_tensor(np.asarray(action_high, dtype=np.float32).reshape(1, -1), dtype=tf.float32)
    count = len(actions)

    for epoch in range(int(epochs)):
        order = np.random.permutation(count)
        losses = []

        for start in range(0, count, batch_size):
            idx = order[start:start + batch_size]
            batch_obs = tf.convert_to_tensor(obs[idx], dtype=tf.float32)
            batch_actions = tf.convert_to_tensor(actions[idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                predicted_actions = scale_action_tensor(
                    actor(batch_obs, training=True),
                    action_low_tensor,
                    action_high_tensor,
                )
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
    validate_async_arguments(args)
    best_tracker = BestCheckpointTracker(args, "ddpg")
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
    actor = None
    critic = None
    buffer = None
    checkpoint = None
    checkpoint_manager = None
    best_checkpoint_manager = None
    stepper = None
    start_episode = 0
    last_completed_episode = None
    noise_std = args.exploration_noise
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
        if env0.action_type != "continuous":
            raise RuntimeError(f"train_generic_ddpg.py requires action_type='continuous', got {env0.action_type!r}")

        obs_dim = env0.obs_dim
        action_size = env0.action_size
        action_low = env0.action_low
        action_high = env0.action_high
        random_action_low, random_action_high = continuous_exploration_bounds(
            env0.action_space_spec,
            action_low,
            action_high,
        )
        print(
            f"Scenario spec: agent_id={env0.agent_id} {env0.agent_summary()} multi_agent={args.multi_agent} "
            f"{env0.team_summary()} obs_dim={obs_dim} action_size={action_size} "
            f"action_names={env0.action_names}",
            flush=True,
        )
        print(
            f"Random exploration bounds: low={random_action_low.tolist()} high={random_action_high.tolist()}",
            flush=True,
        )
        actor_drive_indices = drive_action_indices(env0.action_names)

        for env in envs[1:]:
            if env.obs_dim != obs_dim or env.action_type != "continuous" or env.action_size != action_size:
                raise RuntimeError("All parallel environments must expose the same obs_dim and continuous action_size")

        actor = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
        critic = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        target_actor = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
        target_critic = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        initialize_actor_action_prior(
            actor,
            action_low,
            action_high,
            env0.action_names,
            drive_prior=args.actor_drive_prior,
            steering_prior=args.actor_steering_prior,
        )
        print(
            f"Initialized actor action prior: drive={args.actor_drive_prior:.3f} "
            f"steering={args.actor_steering_prior:.3f} "
            f"drive_regularization={args.actor_drive_regularization:.3f} "
            f"drive_indices={actor_drive_indices}",
            flush=True,
        )
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
        if best_tracker.enabled:
            best_checkpoint_manager = tf.train.CheckpointManager(
                checkpoint,
                directory=str(best_tracker.directory),
                max_to_keep=args.keep_best_checkpoints,
            )
        resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_manager)
        restored_replay_count = 0
        if resume_checkpoint:
            checkpoint.restore(resume_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            noise_std = float(checkpoint.noise_std.numpy())
            print(
                f"Resumed checkpoint {resume_checkpoint} from episode={start_episode} "
                f"noise_std={noise_std:.3f}",
                flush=True,
            )
            restored_replay_count = restore_replay_buffer(args, resume_checkpoint, buffer)

        actor_optimizer.learning_rate.assign(args.actor_learning_rate)
        critic_optimizer.learning_rate.assign(args.critic_learning_rate)

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

        if demo_data is not None and args.demo_prefill and restored_replay_count == 0:
            added = buffer.add_many(
                demo_data["obs"],
                demo_data["actions"],
                demo_data["rewards"],
                demo_data["next_obs"],
                demo_data["dones"],
            )
            print(f"Prefilled replay buffer with demonstration transitions={added}", flush=True)

        if demo_data is not None and args.demo_prefill and restored_replay_count > 0:
            print("Skipped demonstration prefill because the checkpoint replay was restored.", flush=True)

        if demo_data is not None and args.demo_bc_epochs > 0 and (not resume_checkpoint or args.demo_bc_on_resume):
            pretrain_actor_behavior_cloning(
                actor,
                demo_data,
                epochs=args.demo_bc_epochs,
                batch_size=args.demo_bc_batch_size,
                learning_rate=args.demo_bc_learning_rate or args.actor_learning_rate,
                action_low=action_low,
                action_high=action_high,
            )
            target_actor.set_weights(actor.get_weights())

        critic_warmup_target = max(0, int(args.critic_warmup_updates)) if resume_checkpoint else 0
        critic_updates_since_resume = 0
        if critic_warmup_target > 0:
            print(
                f"Resume critic warmup: updates={critic_warmup_target} actor=frozen",
                flush=True,
            )

        opponent_teams = validate_team_layout(envs, args.opponent_pool)
        opponent_pool = OpponentPool(
            args,
            algorithm="ddpg",
            model_factory=lambda: build_continuous_actor(obs_dim=obs_dim, action_size=action_size),
            metadata={"obs_dim": obs_dim, "action_size": action_size},
        )
        if opponent_pool.enabled:
            print(
                f"Opponent pool: dir={opponent_pool.directory} teams={opponent_teams} "
                f"snapshots={len(opponent_pool.entries)} sampling={opponent_pool.sampling}",
                flush=True,
            )

        if args.collector_mode == "async":
            last_completed_episode, noise_std = run_async_ddpg(
                args,
                envs,
                actor,
                critic,
                target_actor,
                target_critic,
                actor_optimizer,
                critic_optimizer,
                buffer,
                checkpoint,
                checkpoint_manager,
                best_checkpoint_manager,
                best_tracker,
                start_episode,
                noise_std,
                action_low,
                action_high,
                random_action_low,
                random_action_high,
                actor_drive_indices,
                critic_warmup_target,
            )
            actor.save_weights(args.actor_weights_path)
            critic.save_weights(args.critic_weights_path)
            print(f"Saved actor weights: {args.actor_weights_path}", flush=True)
            print(f"Saved critic weights: {args.critic_weights_path}", flush=True)
            return

        last_saved_episode = None
        for episode in range(start_episode, args.num_episodes):
            opponent_match = opponent_pool.start_episode(actor, episode)
            use_random_exploration = episode < max(0, args.random_exploration_episodes)
            reset_progress_max = curriculum_reset_progress_max(episode, args)
            env_states = []
            for env_idx, env in enumerate(envs):
                scenario_config = {
                    "training_episode": episode,
                    "max_steps": args.max_steps_per_episode,
                    "physics_frames_per_step": args.physics_frames_per_step,
                }
                if reset_progress_max is not None:
                    scenario_config.update(reset_progress_min=0.0, reset_progress_max=reset_progress_max)
                env.configure(**scenario_config)
                obs, info = env.reset(seed=args.episode_seed_multiplier * episode + env_idx)
                if args.multi_agent:
                    done_mask = np.asarray(info.get("per_agent_done", np.zeros((len(env.agent_ids),), dtype=np.bool_)), dtype=np.bool_)
                    ep_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)
                    action_sum = np.zeros((len(env.agent_ids), action_size), dtype=np.float32)
                    action_count = np.zeros((len(env.agent_ids), 1), dtype=np.float32)
                    previous_action = np.full((len(env.agent_ids), action_size), np.nan, dtype=np.float32)
                    exploration_noise_state = np.zeros((len(env.agent_ids), action_size), dtype=np.float32)
                    action_delta_sum = np.zeros((len(env.agent_ids), action_size), dtype=np.float32)
                    action_delta_count = np.zeros((len(env.agent_ids), 1), dtype=np.float32)
                    max_track_progress = np.zeros((len(env.agent_ids),), dtype=np.float32)
                    last_track_progress = np.zeros((len(env.agent_ids),), dtype=np.float32)
                    finish_reached = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    collision_seen = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    collision_count = np.zeros((len(env.agent_ids),), dtype=np.int32)
                    stalled_seen = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    stalled_count = np.zeros((len(env.agent_ids),), dtype=np.int32)
                    learner_mask = opponent_pool.learner_mask(
                        env.agent_team_ids,
                        opponent_teams,
                        episode,
                        env_idx,
                        use_current_policy=opponent_match.use_current_policy,
                    )
                else:
                    done_mask = None
                    ep_reward = 0.0
                    action_sum = np.zeros((action_size,), dtype=np.float32)
                    action_count = 0.0
                    previous_action = np.full((action_size,), np.nan, dtype=np.float32)
                    exploration_noise_state = np.zeros((action_size,), dtype=np.float32)
                    action_delta_sum = np.zeros((action_size,), dtype=np.float32)
                    action_delta_count = 0.0
                    max_track_progress = 0.0
                    last_track_progress = 0.0
                    finish_reached = False
                    collision_seen = False
                    collision_count = 0
                    stalled_seen = False
                    stalled_count = 0
                    learner_mask = None
                env_states.append({
                    "obs": obs,
                    "done": False,
                    "done_mask": done_mask,
                    "ep_reward": ep_reward,
                    "action_sum": action_sum,
                    "action_count": action_count,
                    "previous_action": previous_action,
                    "exploration_noise_state": exploration_noise_state,
                    "action_delta_sum": action_delta_sum,
                    "action_delta_count": action_delta_count,
                    "max_track_progress": max_track_progress,
                    "last_track_progress": last_track_progress,
                    "finish_reached": finish_reached,
                    "collision_seen": collision_seen,
                    "collision_count": collision_count,
                    "stalled_seen": stalled_seen,
                    "stalled_count": stalled_count,
                    "learner_mask": learner_mask,
                })

            actor_losses = []
            critic_losses = []
            for step_idx in episode_step_indices(args.max_steps_per_episode):
                if all(state["done"] for state in env_states):
                    break

                step_requests = []
                for env, state in zip(envs, env_states):
                    if state["done"]:
                        continue

                    if args.multi_agent:
                        if use_random_exploration:
                            action = sample_exploratory_action(
                                action_low,
                                action_high,
                                env.action_names,
                                rng=np.random,
                                num_agents=len(env.agent_ids),
                                drive_min=args.random_drive_min,
                                steering_abs_max=args.random_steering_abs_max,
                                exploration_low=random_action_low,
                                exploration_high=random_action_high,
                            )
                            action[state["done_mask"]] = action_low
                        else:
                            action = select_actions(
                                actor,
                                state["obs"],
                                state["done_mask"],
                                action_low,
                                action_high,
                                noise_std,
                                noise_kind=args.exploration_noise_kind,
                                noise_state=state["exploration_noise_state"],
                                ou_theta=args.ou_theta,
                            )
                            action = smooth_actions(
                                action,
                                state["previous_action"],
                                action_low,
                                action_high,
                                args.action_smoothing,
                            )
                    else:
                        if use_random_exploration:
                            action = sample_exploratory_action(
                                action_low,
                                action_high,
                                env.action_names,
                                rng=np.random,
                                drive_min=args.random_drive_min,
                                steering_abs_max=args.random_steering_abs_max,
                                exploration_low=random_action_low,
                                exploration_high=random_action_high,
                            )
                        else:
                            action = select_action(
                                actor,
                                state["obs"],
                                action_low,
                                action_high,
                                noise_std,
                                noise_kind=args.exploration_noise_kind,
                                noise_state=state["exploration_noise_state"],
                                ou_theta=args.ou_theta,
                            )
                            action = smooth_actions(
                                action,
                                state["previous_action"],
                                action_low,
                                action_high,
                                args.action_smoothing,
                            )

                    if args.multi_agent and opponent_match.model is not None:
                        opponent_action = select_actions(
                            opponent_match.model,
                            state["obs"],
                            state["done_mask"],
                            action_low,
                            action_high,
                            0.0,
                            noise_kind=args.exploration_noise_kind,
                        )
                        action = opponent_pool.merge_actions(
                            action,
                            opponent_action,
                            state["learner_mask"],
                        )

                    step_requests.append((env, state, action))

                for env, state, action, step_result in stepper.step(step_requests):
                    next_obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)

                    if args.multi_agent:
                        per_agent_rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
                        per_agent_done = np.asarray(info.get("per_agent_done"), dtype=np.bool_)
                        per_agent_terminated = np.asarray(
                            info.get("per_agent_terminated", per_agent_done), dtype=np.bool_
                        )
                        per_agent_infos = list(info.get("per_agent_infos", []))
                        for agent_idx in range(len(env.agent_ids)):
                            if state["done_mask"][agent_idx] and per_agent_done[agent_idx]:
                                continue
                            agent_info = per_agent_infos[agent_idx] if agent_idx < len(per_agent_infos) else {}
                            update_episode_diagnostics(state, agent_info, agent_idx=agent_idx)
                            if state["learner_mask"][agent_idx]:
                                buffer.add(
                                    state["obs"][agent_idx],
                                    action[agent_idx],
                                    float(per_agent_rewards[agent_idx]),
                                    next_obs[agent_idx],
                                    bool(per_agent_terminated[agent_idx] or terminated),
                                )
                            state["ep_reward"][agent_idx] += per_agent_rewards[agent_idx]
                            state["action_sum"][agent_idx] += action[agent_idx]
                            state["action_count"][agent_idx, 0] += 1.0
                            previous_action = state["previous_action"][agent_idx]
                            if np.all(np.isfinite(previous_action)):
                                state["action_delta_sum"][agent_idx] += np.abs(action[agent_idx] - previous_action)
                                state["action_delta_count"][agent_idx, 0] += 1.0
                            state["previous_action"][agent_idx] = action[agent_idx]
                        state["done_mask"] = per_agent_done
                    else:
                        update_episode_diagnostics(state, info.get("agent_info", {}))
                        buffer.add(state["obs"], action, float(reward), next_obs, bool(terminated))
                        state["ep_reward"] += float(reward)
                        state["action_sum"] += action
                        state["action_count"] += 1.0
                        if np.all(np.isfinite(state["previous_action"])):
                            state["action_delta_sum"] += np.abs(action - state["previous_action"])
                            state["action_delta_count"] += 1.0
                        state["previous_action"] = action

                    state["obs"] = next_obs
                    state["done"] = done

                    if len(buffer) >= max(args.replay_warmup, args.batch_size):
                        update_actor = critic_updates_since_resume >= critic_warmup_target
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
                            action_low,
                            action_high,
                            drive_indices=actor_drive_indices,
                            actor_drive_regularization=args.actor_drive_regularization,
                            actor_drive_target=args.actor_drive_target,
                            update_actor=update_actor,
                        )
                        critic_losses.append(critic_loss)
                        critic_updates_since_resume += 1
                        if update_actor:
                            actor_losses.append(actor_loss)
                        elif critic_updates_since_resume == critic_warmup_target:
                            print(
                                f"Resume critic warmup complete after updates={critic_updates_since_resume}; "
                                "actor will be unfrozen on the next update.",
                                flush=True,
                            )

                if len(buffer) >= max(args.replay_warmup, args.batch_size) and (step_idx + 1) % args.target_update_every == 0:
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
            diagnostics = summarize_episode_diagnostics(env_states, args.multi_agent)
            reward_stats = summarize_rewards(env_states)
            mean_action_summary = summarize_actions(mean_actions)
            mean_action_delta_summary = summarize_action_deltas(env_states, args.multi_agent)
            controlled_agents = sum(len(env.agent_ids) if args.multi_agent else 1 for env in envs)
            finish_rate = diagnostics["finishes"] / max(controlled_agents, 1)
            collision_rate = diagnostics["collisions"] / max(controlled_agents, 1)
            stall_rate = diagnostics["stalls"] / max(controlled_agents, 1)
            critic_warmup_remaining = max(0, critic_warmup_target - critic_updates_since_resume)
            print_episode_metrics(episode, [
                ("mode", [
                    ("exploration", "random" if use_random_exploration else "policy"),
                    ("noise", f"{noise_std:.3f}"),
                    ("opponent", opponent_match.label),
                    *(([("reset_progress_max", f"{reset_progress_max:.3f}")]) if reset_progress_max is not None else []),
                ]),
                ("outcome", [
                    ("reward", f"{reward_stats['mean']:.3f} [{reward_stats['min']:.3f}, {reward_stats['max']:.3f}]"),
                    ("progress", f"mean:{diagnostics['progress_mean']:.3f} max:{diagnostics['progress_max']:.3f}"),
                ]),
                ("agents", [
                    ("finish", f"{diagnostics['finishes']}/{controlled_agents} ({finish_rate:.2%})"),
                    ("collision", f"{diagnostics['collisions']}/{controlled_agents} ({collision_rate:.2%})"),
                    ("stall", f"{diagnostics['stalls']}/{controlled_agents} ({stall_rate:.2%})"),
                ]),
                ("actions", [
                    ("mean", format_float_list(mean_action_summary)),
                    ("delta", format_float_list(mean_action_delta_summary)),
                ]),
                ("training", [
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    ("critic_updates", len(critic_losses)),
                    ("policy_updates", len(actor_losses)),
                    ("warmup_left", critic_warmup_remaining),
                    ("actor_loss", f"{float(np.mean(actor_losses)) if actor_losses else 0.0:.5f}"),
                    ("critic_loss", f"{float(np.mean(critic_losses)) if critic_losses else 0.0:.5f}"),
                ]),
            ], args.log_format)
            if args.log_details:
                print(
                    f"episode={episode:04d} details rewards={rewards_summary} mean_actions={mean_actions}",
                    flush=True,
                )
            last_completed_episode = episode + 1
            snapshot_path = opponent_pool.snapshot(actor, episode + 1)
            if snapshot_path is not None:
                print(f"Saved opponent snapshot: {snapshot_path}", flush=True)

            apply_ready_best_checkpoint(
                best_tracker, best_checkpoint_manager, checkpoint, buffer, episode + 1, noise_std, args
            )
            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                saved_path = save_training_checkpoint(
                    checkpoint, checkpoint_manager, buffer, episode + 1, noise_std, args
                )
                last_saved_episode = episode + 1
            else:
                saved_path = None
            if best_tracker.should_evaluate(episode + 1):
                if saved_path is None:
                    saved_path = save_training_checkpoint(
                        checkpoint, checkpoint_manager, buffer, episode + 1, noise_std, args
                    )
                    last_saved_episode = episode + 1
                request_best_checkpoint_evaluation(best_tracker, saved_path, episode + 1)

        if last_saved_episode != args.num_episodes:
            save_training_checkpoint(checkpoint, checkpoint_manager, buffer, args.num_episodes, noise_std, args, final=True)
        actor.save_weights(args.actor_weights_path)
        critic.save_weights(args.critic_weights_path)
        print(f"Saved actor weights: {args.actor_weights_path}", flush=True)
        print(f"Saved critic weights: {args.critic_weights_path}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupt received: saving the last consistent DDPG state...", flush=True)
        if checkpoint is not None and checkpoint_manager is not None and buffer is not None and actor is not None:
            interrupted_episode = last_completed_episode if last_completed_episode is not None else start_episode
            save_training_checkpoint(
                checkpoint, checkpoint_manager, buffer, interrupted_episode, noise_std, args
            )
            actor.save_weights(args.actor_weights_path)
            if critic is not None:
                critic.save_weights(args.critic_weights_path)
            print(f"Interrupted training saved at episode={interrupted_episode}", flush=True)
        else:
            print("Training state was not initialized; no checkpoint was written.", flush=True)
    finally:
        best_tracker.close()
        if stepper is not None:
            stepper.close()
        if buffer is not None:
            try:
                buffer.wait_for_pending_saves()
            except Exception as exc:
                print(f"ERROR waiting for replay save: {exc}", flush=True)
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        manager.stop_all()


if __name__ == "__main__":
    main()
