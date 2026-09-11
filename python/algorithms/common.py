import argparse
import json
import os
import platform
import random
import sys
import sysconfig
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from threading import Event


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

from core.critic_audit import format_audit, q_ranking_audit
from core.curriculum import scenario_curriculum_config
from core.models import (
    build_actor_forward_fn,
    build_continuous_actor,
    build_continuous_critic,
    default_network_layers,
    set_network_layers,
)
from core.multi_policy import (
    MultiPolicySnapshot,
    add_multi_policy_arguments,
    build_policy_assignment,
    load_multi_policy_manifest,
    restore_policy_replays,
    save_policy_replays,
    validate_multi_policy_options,
    write_multi_policy_manifest,
)
from core.policy_artifact import PolicyArtifactSaver, build_policy_metadata, load_policy_into_model
from core.opponent_pool import OpponentPool, add_opponent_pool_arguments, validate_team_layout
from core.replay_buffer import ReplayBuffer
from core.training_health import OffPolicyRecoveryRuntime
from core.training import (
    AsyncCollectorPool,
    AsyncEventScheduler,
    SyncUpdateThrottle,
    AsyncEpisodeEvent,
    AsyncStepEvent,
    AsyncWorkerDoneEvent,
    AsyncWorkerErrorEvent,
    ParallelEnvStepper,
    PolicySnapshot,
    TrainingBudget,
    BestCheckpointTracker,
    add_training_budget_argument,
    add_best_checkpoint_arguments,
    add_training_health_arguments,
    apply_ready_best_checkpoint,
    add_collector_arguments,
    add_lockstep_tuning_arguments,
    add_parallel_env_arguments,
    add_log_format_argument,
    add_dashboard_arguments,
    maybe_start_dashboard,
    report_training_time,
    add_godot_render_argument,
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
    argument_group,
    GROUP_CHECKPOINTS,
    GROUP_CRITIC_AUDIT,
    GROUP_DEMOS,
    GROUP_DETERMINISTIC,
    GROUP_EXPLORATION,
    GROUP_GODOT,
    GROUP_LOGGING,
    GROUP_LOOP,
    GROUP_PROGRESS_CURRICULUM,
    GROUP_REPLAY,
)
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


DETERMINISTIC_ALGORITHMS = {"ddpg", "ddpg_bc", "ddpgfd", "td3", "td3_bc"}


def parse_args(trainer_variant):
    if trainer_variant not in DETERMINISTIC_ALGORITHMS:
        raise ValueError(f"Unknown deterministic trainer: {trainer_variant!r}")
    parser = argparse.ArgumentParser(
        description=(
            f"Generic {trainer_variant} trainer for continuous Godot scenarios. "
            "Collector, replay, checkpoint and multi-agent infrastructure is shared "
            "with the other deterministic algorithms."
        )
    )

    group = argument_group(parser, GROUP_GODOT)
    group.add_argument("--num-envs", type=int, default=1)
    group.add_argument("--base-port", type=int, default=6200)

    group = argument_group(parser, GROUP_LOOP)
    group.add_argument("--num-episodes", type=int, default=500)
    add_training_budget_argument(parser)
    group.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=500,
        help="Maximum episode steps shared with Godot; use 0 to rely only on terminal conditions.",
    )
    group.add_argument("--batch-size", type=int, default=128)
    group.add_argument("--gamma", type=float, default=0.99)
    group.add_argument("--tau", type=float, default=0.005)
    group.add_argument("--actor-learning-rate", type=float, default=1e-4)
    group.add_argument("--critic-learning-rate", type=float, default=1e-3)

    group = argument_group(parser, GROUP_EXPLORATION)
    group.add_argument("--exploration-noise", type=float, default=0.2)
    group.add_argument("--exploration-noise-min", type=float, default=0.02)
    group.add_argument("--exploration-noise-decay", type=float, default=0.995)
    group.add_argument("--exploration-noise-kind", choices=["ou", "gaussian"], default="ou")
    group.add_argument("--ou-theta", type=float, default=0.15)
    group.add_argument("--action-smoothing", type=float, default=0.2)
    group.add_argument("--random-exploration-episodes", type=int, default=15)
    group.add_argument("--random-drive-min", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--random-steering-abs-max", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--actor-drive-prior", type=float, default=0.75)
    group.add_argument("--actor-steering-prior", type=float, default=0.0)
    group.add_argument("--actor-drive-regularization", type=float, default=0.05)
    group.add_argument("--actor-drive-target", type=float, default=0.65)

    group = argument_group(parser, GROUP_PROGRESS_CURRICULUM)
    group.add_argument("--reset-progress-curriculum", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--reset-progress-start-max", type=float, default=0.025)
    group.add_argument("--reset-progress-end-max", type=float, default=0.35)
    group.add_argument("--reset-progress-ramp-episodes", type=int, default=400)

    group = argument_group(parser, GROUP_REPLAY)
    group.add_argument("--replay-warmup", type=int, default=500)
    group.add_argument("--replay-capacity", type=int, default=100000)

    group = argument_group(parser, GROUP_LOOP)
    group.add_argument("--critic-warmup-updates", type=int, default=2000)
    group.add_argument("--target-update-every", type=int, default=1)
    group.add_argument(
        "--network-layers",
        type=int,
        nargs="+",
        default=None,
        metavar="WIDTH",
        help="Hidden layer widths for actor/critic/Q networks, e.g. --network-layers 256 256. "
        "Omitted, each algorithm uses its reference architecture (SAC 256 256, "
        "TD3/DDPG 400 300, PPO and DQN 64 64). Checkpoints written before these "
        "defaults used 256 256 128 and need that value passed explicitly.",
    )

    group = argument_group(parser, GROUP_GODOT)
    group.add_argument("--env-seed-base", type=int, default=100)
    group.add_argument("--episode-seed-multiplier", type=int, default=1000)
    group.add_argument("--env-timeout", type=float, default=30.0)
    group.add_argument("--agent-id", default=None)
    group.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    add_multi_policy_arguments(parser)

    group = argument_group(parser, GROUP_CHECKPOINTS)
    group.add_argument("--actor-weights-path", default="generic_ddpg_actor.weights.h5")
    group.add_argument(
        "--policy-path",
        default=None,
        help="Warm-start the actor from a .keras model, full .h5 model, or .weights.h5 file.",
    )
    group.add_argument("--critic-weights-path", default="generic_ddpg_critic.weights.h5")
    group.add_argument("--checkpoint-dir", default="checkpoints/generic_ddpg")
    group.add_argument("--resume-checkpoint", default=None)
    group.add_argument("--checkpoint-every", type=int, default=25)
    group.add_argument("--keep-checkpoints", type=int, default=5)
    group.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    group = argument_group(parser, GROUP_REPLAY)
    group.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--require-replay-buffer", action=argparse.BooleanOptionalAction, default=False)

    group = argument_group(parser, GROUP_DEMOS)
    group.add_argument("--demo-path", action="append", default=[])
    group.add_argument("--demo-prefill", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--demo-max-transitions", type=int, default=0)
    group.add_argument("--demo-bc-epochs", type=int, default=0)
    group.add_argument("--demo-bc-batch-size", type=int, default=128)
    group.add_argument("--demo-bc-learning-rate", type=float, default=None)
    group.add_argument("--demo-bc-on-resume", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--demo-validation-path", action="append", default=[],
                        help="Held-out demo npz(s): BC keeps the epoch with the lowest validation "
                        "action MSE (best-checkpoint against overfitting).")
    group.add_argument("--bc-actor-weights-path", type=str, default=None,
                        help="Persist the BC-pretrained actor here BEFORE any RL update "
                        "(default: the actor weights path with a _bc suffix).")
    group.add_argument("--stop-after-bc", action=argparse.BooleanOptionalAction, default=False,
                        help="Run BC (+ save the BC actor) then ONE frozen evaluation at the "
                        "current curriculum level, and STOP before the RL loop. A deterministic "
                        "BC gate; errors if BC did not actually run.")

    group = argument_group(parser, GROUP_CRITIC_AUDIT)
    group.add_argument("--stop-after-critic-warmup", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Run the critic warmup with the actor FROZEN, then a Q-ranking audit on "
                        "the validation batch (min(Q1,Q2) for expert/clone/perturbed/random/saturated "
                        "actions), SAVE state and STOP before the first actor update. The audit gate "
                        "must pass (critic values expert+clone above random+saturated) before any "
                        "actor update is allowed.")
    group.add_argument("--gradient-telemetry-every", type=int, default=0,
                        help="Collect separate Q-term / BC-term gradient norms + cosine similarity + "
                        "actor-vs-BC deviation every N policy updates (0 = never). Adds two extra "
                        "backward passes on those steps.")
    group.add_argument("--critic-audit-win-rate", type=float, default=0.9,
                        help="stop-after-critic-warmup gate: each good-vs-bad action pair must beat "
                        "the bad family on at least this fraction of validation samples.")
    group.add_argument("--critic-audit-cell-margin-tol", type=float, default=0.0,
                        help="stop-after-critic-warmup gate: a cell fails if its MEAN good-vs-bad "
                        "margin is below -tol (0 = no clearly-negative cell allowed).")
    # Demo sources can feed BC, replay, or both without crossing those boundaries.

    group = argument_group(parser, GROUP_DEMOS)
    group.add_argument("--demo-replay-only-path", action="append", default=[],
                        help="Complete-transition npz(s) added ONLY to the replay buffer, never to "
                        "BC (e.g. DAgger recovery: expert-applied transitions valid for the critics "
                        "but which must not bias the imitation target).")
    group.add_argument("--demo-bc-path", action="append", default=[],
                        help="(obs, action) npz(s) added ONLY to the BC imitation loss, never to "
                        "replay (their next_obs was produced by a different policy).")
    group.add_argument("--demo-q-filter-start-policy-updates", type=int, default=0,
                        help="Enable the TD3+BC Q-filter only after this many policy updates have "
                        "run SINCE the critic warmup ended (0 = filter from the first update). "
                        "Before it, the demo BC term is unfiltered.")
    parser.set_defaults(
        critic2_weights_path="generic_td3_critic2.weights.h5",
        demo_bc_weight_start=1.0,
        demo_bc_weight_end=0.05,
        demo_bc_decay_updates=100000,
        demo_q_weight_start=0.0,
        demo_q_weight_end=1.0,
        demo_q_weight_ramp_updates=50000,
        demo_q_filter=False,
        td3_policy_delay=2,
        td3_target_policy_noise=0.2,
        td3_target_noise_clip=0.5,
        td3_bc_alpha=2.5,
        ddpgfd_pretrain_updates=1000,
        ddpgfd_priority_alpha=0.3,
        ddpgfd_priority_beta=1.0,
        ddpgfd_demo_priority_bonus=1.0,
        ddpgfd_actor_priority_weight=1e-3,
    )
    if variant_uses_joint_bc(trainer_variant) or str(trainer_variant) == "ddpgfd":
        group.add_argument("--demo-bc-weight-start", type=float, default=1.0)
        group.add_argument("--demo-bc-weight-end", type=float, default=0.05)
        group.add_argument("--demo-bc-decay-updates", type=int, default=100000)
        # Ramp Q maximization only after the actor leaves critic warmup.
        group.add_argument("--demo-q-weight-start", type=float, default=0.0)
        group.add_argument("--demo-q-weight-end", type=float, default=1.0)
        group.add_argument("--demo-q-weight-ramp-updates", type=int, default=50000)
        group.add_argument("--demo-q-filter", action=argparse.BooleanOptionalAction, default=False)

    group = argument_group(parser, GROUP_LOOP)
    group.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="Hard global gradient-norm cap for critic/actor updates (0 disables). Safety "
        "net against the deadly-triad Q-value divergence that otherwise blows critics up.",
    )
    group.add_argument(
        "--grad-clip-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip each network at --grad-clip-k * EMA(gradient-norm), bounded by "
        "--grad-clip-norm.",
    )
    group.add_argument(
        "--grad-clip-k",
        type=float,
        default=3.0,
        help="Multiplier on the running-mean gradient norm when --grad-clip-adaptive is set.",
    )
    if variant_uses_td3(trainer_variant):

        group = argument_group(parser, GROUP_CHECKPOINTS)
        group.add_argument("--critic2-weights-path", default="generic_td3_critic2.weights.h5")

        group = argument_group(parser, GROUP_DETERMINISTIC)
        group.add_argument("--td3-policy-delay", type=int, default=2)
        group.add_argument("--td3-target-policy-noise", type=float, default=0.2)
        group.add_argument("--td3-target-noise-clip", type=float, default=0.5)
    if trainer_variant == "td3_bc":
        group.add_argument("--td3-bc-alpha", type=float, default=2.5)
    if trainer_variant == "ddpgfd":
        group.add_argument("--ddpgfd-pretrain-updates", type=int, default=1000)
        group.add_argument("--ddpgfd-priority-alpha", type=float, default=0.3)
        group.add_argument("--ddpgfd-priority-beta", type=float, default=1.0)
        group.add_argument("--ddpgfd-demo-priority-bonus", type=float, default=1.0)
        group.add_argument("--ddpgfd-actor-priority-weight", type=float, default=1e-3)

    group = argument_group(parser, GROUP_GODOT)
    group.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    group.add_argument("--godot-project", default=None)
    group.add_argument("--godot-scene", default=None)
    group.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run Godot without a window. Rendering training instances contends with "
            "TensorFlow for the GPU, and only headless instances get --fixed-fps, without "
            "which physics stays gated to wall-clock 60Hz. Use --no-headless to watch."
        ),
    )
    group.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default=False)

    group = argument_group(parser, GROUP_LOGGING)
    group.add_argument("--log-details", action=argparse.BooleanOptionalAction, default=False)
    add_collector_arguments(parser)
    add_parallel_env_arguments(parser)
    add_lockstep_tuning_arguments(parser)
    add_opponent_pool_arguments(parser)
    add_best_checkpoint_arguments(parser)
    add_training_health_arguments(parser)
    add_log_format_argument(parser)
    add_dashboard_arguments(parser)
    add_tensorflow_runtime_arguments(parser, include_compile_learner=True)
    add_godot_render_argument(parser)
    args = parser.parse_args()
    args.trainer_variant = trainer_variant
    # Multi-policy runs reject audit options they cannot execute faithfully.
    if getattr(args, "multi_policy", False):
        if getattr(args, "stop_after_critic_warmup", False):
            parser.error("--stop-after-critic-warmup is not supported with --multi-policy "
                         "(single-policy only); remove one of them.")
        if int(getattr(args, "gradient_telemetry_every", 0) or 0) > 0:
            parser.error("--gradient-telemetry-every is not supported with --multi-policy "
                         "(single-policy only); set it to 0.")
    return args


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

    # Driving policies start with forward throttle and neutral steering.
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


def replay_warmup_threshold(args):
    return max(int(args.replay_warmup), int(args.batch_size))


def replay_warmup_remaining(buffer, args):
    return max(0, replay_warmup_threshold(args) - len(buffer))


def should_use_random_exploration(episode, args, replay_size):
    if (
        bool(getattr(args, "_replay_warmup_uses_restored_policy", False))
        and int(replay_size) < replay_warmup_threshold(args)
    ):
        return False
    return (
        int(episode) < max(0, int(args.random_exploration_episodes))
        or int(replay_size) < replay_warmup_threshold(args)
    )


def build_replay_ready_event(buffer, args):
    ready = Event()
    if (
        replay_warmup_remaining(buffer, args) == 0
        or bool(getattr(args, "_replay_warmup_uses_restored_policy", False))
    ):
        ready.set()
    return ready


def exploration_label(state):
    if not state.get("use_random_exploration", False):
        return "policy"
    if state.get("replay_warmup_exploration", False):
        return "warmup_random"
    return "scheduled_random"


def _record_collision_timing(state, finish_reached, agent_idx=None):
    if agent_idx is None:
        had_success = bool(state["finish_reached"])
        timing = (
            "after_success"
            if had_success
            else "at_success"
            if finish_reached
            else "before_success"
        )
        key = f"collision_{timing}_count"
        state[key] = int(state.get(key, 0)) + 1
        return

    had_success = bool(state["finish_reached"][agent_idx])
    timing = (
        "after_success"
        if had_success
        else "at_success"
        if finish_reached
        else "before_success"
    )
    key = f"collision_{timing}_count"
    values = state.setdefault(
        key,
        np.zeros(np.asarray(state["finish_reached"]).shape, dtype=np.int32),
    )
    values[agent_idx] += 1


def update_episode_diagnostics(state, agent_info, agent_idx=None):
    progress = float(agent_info.get("track_progress", 0.0))
    finish_reached = bool(agent_info.get("finish_reached", agent_info.get("target_reached", False)))
    progress_stalled = bool(agent_info.get("progress_stalled", False))
    local_terms = agent_info.get("local_term_rewards", {}) or {}
    scenario_terms = agent_info.get("scenario_terms", {}) or {}
    scenario_component_values = scenario_terms.get("component_values", {}) or {}
    events = agent_info.get("events", {}) or {}
    terminal_reason = str(agent_info.get("terminal_reason", "")).strip().lower()
    # Read both scalar reward components and top-level event diagnostics.
    collision_term = (
        float(local_terms.get("collision", 0.0))
        + float(scenario_terms.get("collision", 0.0))
        + float(scenario_component_values.get("collision", 0.0))
    )
    collision_seen = bool(
        abs(collision_term) > 1e-9
        or events.get("collision", False)
        or agent_info.get("collided", False)
        or terminal_reason == "collision"
    )

    if agent_idx is None:
        state["max_track_progress"] = max(float(state["max_track_progress"]), progress)
        state["last_track_progress"] = progress
        if collision_seen and not state["collision_seen"]:
            _record_collision_timing(state, finish_reached)
            state["collision_count"] += 1
            state["collision_seen"] = True
            collision_source = str(agent_info.get("collision_source", "")).strip()
            if collision_source:
                state["collision_source"] = collision_source
            collision_details = agent_info.get("collision_details")
            if isinstance(collision_details, dict) and collision_details:
                state["collision_details"] = dict(collision_details)
        state["finish_reached"] = bool(state["finish_reached"] or finish_reached)
        if progress_stalled and not state["stalled_seen"]:
            state["stalled_count"] += 1
            state["stalled_seen"] = True
        # Optional generic diagnostics surfaced by the agent body (e.g. reach-and-hold arm).
        # setdefault keeps this backward-compatible with states that never initialised the keys.
        max_joint_speed = agent_info.get("max_joint_speed")
        if max_joint_speed is not None:
            state["max_joint_speed_last"] = float(max_joint_speed)
            state["max_joint_speed_peak"] = max(
                float(state.get("max_joint_speed_peak", 0.0)), float(max_joint_speed))
        hold_frames = agent_info.get("hold_frames")
        if hold_frames is not None:
            state["hold_frames_peak"] = max(
                int(state.get("hold_frames_peak", 0)), int(hold_frames))
        position_error = agent_info.get("position_error_m")
        if position_error is not None:
            state["position_error_last"] = float(position_error)
        orientation_error = agent_info.get("orientation_error_deg")
        if orientation_error is not None:
            state["orientation_error_last"] = float(orientation_error)
        return

    state["max_track_progress"][agent_idx] = max(float(state["max_track_progress"][agent_idx]), progress)
    state["last_track_progress"][agent_idx] = progress
    if collision_seen and not state["collision_seen"][agent_idx]:
        _record_collision_timing(state, finish_reached, agent_idx=agent_idx)
        state["collision_count"][agent_idx] += 1
        state["collision_seen"][agent_idx] = True
        collision_source = str(agent_info.get("collision_source", "")).strip()
        if collision_source:
            sources = state.setdefault(
                "collision_sources",
                np.full(np.asarray(state["finish_reached"]).shape, "", dtype=object),
            )
            sources[agent_idx] = collision_source
        collision_details = agent_info.get("collision_details")
        if isinstance(collision_details, dict) and collision_details:
            details = state.setdefault(
                "collision_details",
                np.full(np.asarray(state["finish_reached"]).shape, None, dtype=object),
            )
            details[agent_idx] = dict(collision_details)
    state["finish_reached"][agent_idx] = bool(state["finish_reached"][agent_idx] or finish_reached)
    if progress_stalled and not state["stalled_seen"][agent_idx]:
        state["stalled_count"][agent_idx] += 1
        state["stalled_seen"][agent_idx] = True
    diagnostic_shape = np.asarray(state["max_track_progress"]).shape
    for info_key, state_key in (
        ("max_joint_speed", "max_joint_speed_last"),
        ("position_error_m", "position_error_last"),
        ("orientation_error_deg", "orientation_error_last"),
    ):
        value = agent_info.get(info_key)
        if value is None:
            continue
        values = state.setdefault(
            state_key, np.zeros(diagnostic_shape, dtype=np.float32))
        values[agent_idx] = float(value)
    hold_frames = agent_info.get("hold_frames")
    if hold_frames is not None:
        values = state.setdefault(
            "hold_frames_peak", np.zeros(diagnostic_shape, dtype=np.int32))
        values[agent_idx] = max(int(values[agent_idx]), int(hold_frames))


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
            "collisions_before_success": int(sum(
                np.sum(state.get("collision_before_success_count", 0))
                for state in env_states
            )),
            "collisions_at_success": int(sum(
                np.sum(state.get("collision_at_success_count", 0))
                for state in env_states
            )),
            "collisions_after_success": int(sum(
                np.sum(state.get("collision_after_success_count", 0))
                for state in env_states
            )),
            "collision_sources": sorted({
                str(source)
                for state in env_states
                for source in np.asarray(
                    state.get("collision_sources", []), dtype=object
                ).reshape(-1)
                if str(source)
            }),
            "collision_pairs": sorted({
                _collision_detail_label(details)
                for state in env_states
                for details in np.asarray(
                    state.get("collision_details", []), dtype=object
                ).reshape(-1)
                if isinstance(details, dict) and details
            }),
            "stalls": int(sum(np.sum(values) for values in stalled_arrays)),
            "max_joint_speed": float(np.mean([
                np.mean(state.get("max_joint_speed_last", 0.0))
                for state in env_states])),
            "hold_frames": int(max(
                (int(np.max(state.get("hold_frames_peak", 0)))
                 for state in env_states), default=0)),
            "position_error_m": float(np.mean([
                np.mean(state.get("position_error_last", 0.0))
                for state in env_states])),
            "orientation_error_deg": float(np.mean([
                np.mean(state.get("orientation_error_last", 0.0))
                for state in env_states])),
        }

    return {
        "progress_max": float(max(state["max_track_progress"] for state in env_states)),
        "progress_mean": float(np.mean([state["last_track_progress"] for state in env_states])),
        "finishes": int(sum(1 for state in env_states if state["finish_reached"])),
        "collisions": int(sum(state["collision_count"] for state in env_states)),
        "collisions_before_success": int(sum(
            state.get("collision_before_success_count", 0)
            for state in env_states
        )),
        "collisions_at_success": int(sum(
            state.get("collision_at_success_count", 0)
            for state in env_states
        )),
        "collisions_after_success": int(sum(
            state.get("collision_after_success_count", 0)
            for state in env_states
        )),
        "collision_sources": sorted({
            str(state.get("collision_source", ""))
            for state in env_states
            if str(state.get("collision_source", ""))
        }),
        "collision_pairs": sorted({
            _collision_detail_label(state["collision_details"])
            for state in env_states
            if isinstance(state.get("collision_details"), dict)
            and state["collision_details"]
        }),
        "stalls": int(sum(state["stalled_count"] for state in env_states)),
        "max_joint_speed": float(np.mean([
            state.get("max_joint_speed_last", 0.0) for state in env_states])),
        "hold_frames": int(max(
            (int(state.get("hold_frames_peak", 0)) for state in env_states), default=0)),
        "position_error_m": float(np.mean([
            state.get("position_error_last", 0.0) for state in env_states])),
        "orientation_error_deg": float(np.mean([
            state.get("orientation_error_last", 0.0) for state in env_states])),
}


def _collision_detail_label(details):
    source = str(details.get("source", "unknown"))
    checker = str(
        details.get("checker_link", details.get("checker_shape", "?"))
    )
    collider = str(
        details.get("collider", details.get("body_link", "?"))
    )
    return f"{source}:{checker}->{collider}"


def format_collision_diagnostics(diagnostics, controlled_agents):
    total = int(diagnostics["collisions"])
    rate = total / max(int(controlled_agents), 1)
    before = int(diagnostics.get("collisions_before_success", 0))
    at_success = int(diagnostics.get("collisions_at_success", 0))
    after = int(diagnostics.get("collisions_after_success", 0))
    sources = diagnostics.get("collision_sources", [])
    pairs = diagnostics.get("collision_pairs", [])
    result = f"{total}/{controlled_agents} ({rate:.2%})"
    if total:
        result += f" timing:{before}/{at_success}/{after}"
        if sources:
            result += f" source:{','.join(sources)}"
        if pairs:
            result += f" pair:{'|'.join(pairs)}"
    return result


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
    policy_assignment=None,
    replay_ready_event=None,
):
    def begin_episode(worker_id, env, episode):
        reset_progress_max = curriculum_reset_progress_max(episode, args)
        scenario_config = {
            "training_episode": episode,
            "max_steps": args.max_steps_per_episode,
            "physics_frames_per_step": args.physics_frames_per_step,
            "training_mode": True,
            **scenario_curriculum_config(args),
        }
        if reset_progress_max is not None:
            scenario_config.update(reset_progress_min=0.0, reset_progress_max=reset_progress_max)
        env.configure(**scenario_config)
        obs, info = env.reset(seed=args.episode_seed_multiplier * episode + worker_id)
        agent_count = len(env.agent_ids) if args.multi_agent else 1
        replay_warmup_exploration = bool(
            replay_ready_event is not None and not replay_ready_event.is_set()
        )
        scheduled_random_exploration = (
            episode < max(0, args.random_exploration_episodes)
        )
        state = {
            "obs": obs,
            "done": False,
            "reset_progress_max": reset_progress_max,
            "use_random_exploration": (
                replay_warmup_exploration or scheduled_random_exploration
            ),
            "replay_warmup_exploration": replay_warmup_exploration,
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
                transition = (
                    state["obs"][agent_idx],
                    np.asarray(action[agent_idx], dtype=np.float32),
                    float(rewards[agent_idx]),
                    next_obs[agent_idx],
                    bool(terminated_mask[agent_idx] or terminated),
                )
                if policy_assignment is not None:
                    transition = (
                        policy_assignment.policy_for_agent(
                            env.agent_ids[agent_idx]
                        ),
                        *transition,
                    )
                transitions.append(transition)
                state["ep_reward"][agent_idx] += rewards[agent_idx]
                state["action_sum"][agent_idx] += action[agent_idx]
                state["action_count"][agent_idx, 0] += 1.0
                previous = state["previous_action"][agent_idx]
                if np.all(np.isfinite(previous)):
                    state["action_delta_sum"][agent_idx] += np.abs(action[agent_idx] - previous)
                    state["action_delta_count"][agent_idx, 0] += 1.0
                state["previous_action"][agent_idx] = action[agent_idx]
            state["done_mask"] = done_mask
            done = done or bool(np.all(done_mask))
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


def gradient_clipper(model, role, grad_clip_norm, grad_clip_adaptive, grad_clip_k,
                     grad_clip_decay=0.99):
    """Per-network gradient clipper, cached on the model so its EMA survives across updates.

    Mirrors the SAC clipper: the fixed cap is a hard backstop against deadly-triad Q-divergence,
    while adaptive mode clips at ``grad_clip_k * EMA(grad_norm)`` bounded by that cap, so each
    network's own gradient scale sets the threshold instead of a hand-tuned constant. Non-finite
    norms are excluded from the EMA so a NaN cannot poison it.
    """
    cache = getattr(model, "_metis_grad_clippers", None)
    if cache is None:
        cache = {}
        setattr(model, "_metis_grad_clippers", cache)
    key = (role, float(grad_clip_norm), bool(grad_clip_adaptive), float(grad_clip_k))
    if key in cache:
        return cache[key]

    has_hard = bool(grad_clip_norm and grad_clip_norm > 0.0)
    enabled = has_hard or bool(grad_clip_adaptive)
    hard_c = tf.constant(float(grad_clip_norm) if has_hard else 0.0, dtype=tf.float32)
    k_c = tf.constant(float(grad_clip_k), dtype=tf.float32)
    decay_c = tf.constant(float(grad_clip_decay), dtype=tf.float32)
    warmup_c = tf.constant(25.0, dtype=tf.float32)
    ema = tf.Variable(0.0, dtype=tf.float32, trainable=False, name=f"clip_ema_{role}")
    seen = tf.Variable(0.0, dtype=tf.float32, trainable=False, name=f"clip_seen_{role}")

    def clip(grads):
        if not enabled:
            return grads
        if not grad_clip_adaptive:
            clipped, _ = tf.clip_by_global_norm(grads, hard_c)
            return clipped
        gnorm = tf.linalg.global_norm(grads)
        finite = tf.math.is_finite(gnorm)
        g_for_ema = tf.where(finite, gnorm, ema)
        new_ema = tf.where(
            seen > 0.0, decay_c * ema + (1.0 - decay_c) * g_for_ema, g_for_ema)
        ema.assign(new_ema)
        seen.assign_add(1.0)
        adaptive_cap = k_c * ema
        cap = tf.minimum(hard_c, adaptive_cap) if has_hard else adaptive_cap
        warmup_cap = hard_c if has_hard else adaptive_cap
        cap = tf.where(seen < warmup_c, warmup_cap, cap)  # cold EMA -> don't over-clip
        cap = tf.maximum(cap, 1e-3)
        clipped, _ = tf.clip_by_global_norm(grads, cap)
        return clipped

    cache[key] = clip
    return clip


def soft_update(target_model, source_model, tau):
    target_weights = target_model.weights
    source_weights = source_model.weights
    if len(target_weights) != len(source_weights):
        raise ValueError("Target and source models must expose the same number of weights")

    # Keep Polyak averaging on-device to avoid a NumPy round trip per update.
    for target_weight, source_weight in zip(target_weights, source_weights):
        if target_weight.shape != source_weight.shape:
            raise ValueError(
                "Target and source model weights must have matching shapes: "
                f"{target_weight.shape} != {source_weight.shape}"
            )
        tau_tensor = tf.cast(tau, target_weight.dtype)
        target_weight.assign_add(tau_tensor * (source_weight - target_weight))


def variant_uses_td3(variant):
    return str(variant) in {"td3", "td3_bc"}


def variant_uses_joint_bc(variant):
    return str(variant) in {"ddpg_bc", "td3_bc"}


def scheduled_bc_weight(update_step, start, end, decay_updates):
    decay_updates = max(1, int(decay_updates))
    ratio = min(1.0, max(0.0, float(update_step) / float(decay_updates)))
    return float(start) + ratio * (float(end) - float(start))


def _flatten_grads(grads):
    """Flatten a list of per-variable gradients into one 1-D tensor (drop the None entries)."""
    parts = [tf.reshape(g, [-1]) for g in grads if g is not None]
    if not parts:
        return None
    return tf.concat(parts, axis=0)


def scheduled_q_weight(update_step, start, end, ramp_updates):
    """Weight on the TD3 Q-maximisation term, ramped over REAL policy updates since the warmup.

    Starts at ``start`` (0.0 by default) and grows to ``end`` over ``ramp_updates`` policy updates.
    At policy_updates_since_warmup==0 the Q term is off, so the unfrozen actor is pulled ONLY by the
    BC anchor and cannot be yanked into a self-collision basin by a full-strength Q-max on a critic
    that has never scored the actor's own actions (the TD3+BC v4 collapse). Same clamped linear
    interpolation as scheduled_bc_weight; a separate name because the direction/intent is opposite
    (grow the Q term in, rather than decay the BC term out)."""
    ramp_updates = max(1, int(ramp_updates))
    ratio = min(1.0, max(0.0, float(update_step) / float(ramp_updates)))
    return float(start) + ratio * (float(end) - float(start))


def policy_update_count(counter):
    """Return a checkpointed policy-update counter as a Python integer."""
    if hasattr(counter, "numpy"):
        return int(counter.numpy())
    return int(counter)


def increment_policy_update_count(counter):
    """Advance either a TensorFlow checkpoint variable or a plain counter."""
    if hasattr(counter, "assign_add"):
        counter.assign_add(1)
        return counter
    return int(counter) + 1


def sample_demo_actions(demo_data, batch_size):
    if demo_data is None or len(demo_data["actions"]) == 0:
        return None
    count = min(int(batch_size), len(demo_data["actions"]))
    indices = np.random.randint(0, len(demo_data["actions"]), size=count)
    return demo_data["obs"][indices], demo_data["actions"][indices]


def train_deterministic_step(
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
    *,
    grad_clip_norm=10.0,
    grad_clip_adaptive=False,
    grad_clip_k=3.0,
    variant="ddpg",
    learner_update=1,
    critic2=None,
    target_critic2=None,
    critic2_optimizer=None,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    policy_delay=2,
    demo_data=None,
    demo_batch_size=128,
    bc_weight=0.0,
    demo_q_filter=False,
    q_filter_active=True,
    td3_bc_alpha=2.5,
    q_weight=1.0,
    gradient_telemetry=False,
    bc_reference_actor=None,
    ddpgfd_actor_priority_weight=1e-3,
):
    uses_td3 = variant_uses_td3(variant)

    def _clip(model, role):
        return gradient_clipper(
            model, role, grad_clip_norm, grad_clip_adaptive, grad_clip_k)

    if uses_td3 and (critic2 is None or target_critic2 is None or critic2_optimizer is None):
        raise ValueError("TD3 variants require a second critic, target critic and optimizer")
    if str(variant) == "ddpgfd":
        sample = buffer.sample_prioritized(batch_size, action_dtype=np.float32)
        obs, actions, rewards, next_obs, dones, replay_indices, importance_weights, _demo_mask = sample
    else:
        obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, action_dtype=np.float32)
        replay_indices = None
        importance_weights = np.ones((int(batch_size),), dtype=np.float32)
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards.reshape(-1, 1), dtype=tf.float32)
    next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones.reshape(-1, 1), dtype=tf.float32)
    action_low = tf.convert_to_tensor(np.asarray(action_low, dtype=np.float32).reshape(1, -1), dtype=tf.float32)
    action_high = tf.convert_to_tensor(np.asarray(action_high, dtype=np.float32).reshape(1, -1), dtype=tf.float32)
    importance_weights = tf.convert_to_tensor(importance_weights.reshape(-1, 1), dtype=tf.float32)

    next_actions = scale_action_tensor(target_actor(next_obs, training=False), action_low, action_high)
    if uses_td3:
        action_half_range = 0.5 * (action_high - action_low)
        noise = tf.random.normal(tf.shape(next_actions), dtype=tf.float32)
        noise *= float(target_policy_noise) * action_half_range
        noise_limit = float(target_noise_clip) * action_half_range
        noise = tf.clip_by_value(noise, -noise_limit, noise_limit)
        next_actions = tf.clip_by_value(next_actions + noise, action_low, action_high)
    target_q = target_critic([next_obs, next_actions], training=False)
    if uses_td3:
        target_q2 = target_critic2([next_obs, next_actions], training=False)
        target_q = tf.minimum(target_q, target_q2)
    y = rewards + (1.0 - dones) * gamma * target_q

    with tf.GradientTape() as tape:
        q = critic([obs, actions], training=True)
        td_error = y - q
        critic_loss = tf.reduce_mean(importance_weights * tf.square(td_error))
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_grads = _clip(critic, "critic")(critic_grads)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    critic2_loss = None
    if uses_td3:
        with tf.GradientTape() as tape:
            q2 = critic2([obs, actions], training=True)
            td_error2 = y - q2
            critic2_loss = tf.reduce_mean(importance_weights * tf.square(td_error2))
        critic2_grads = tape.gradient(critic2_loss, critic2.trainable_variables)
        critic2_grads = _clip(critic2, "critic2")(critic2_grads)
        critic2_optimizer.apply_gradients(zip(critic2_grads, critic2.trainable_variables))

    if replay_indices is not None:
        with tf.GradientTape() as action_tape:
            action_tape.watch(actions)
            sampled_q = critic([obs, actions], training=False)
        action_grads = action_tape.gradient(sampled_q, actions)
        actor_priority = tf.reduce_mean(tf.square(action_grads), axis=1)
        priority_values = tf.square(tf.squeeze(td_error, axis=1))
        priority_values += float(ddpgfd_actor_priority_weight) * actor_priority
        buffer.update_priorities(replay_indices, priority_values.numpy())

    actor_loss_value = None
    q_loss_value = None
    bc_loss_value = None
    q_scale_value = None
    scaled_q_loss_value = None
    q_grad_norm = None
    bc_grad_norm = None
    grad_cosine = None
    actor_bc_deviation_pre_update = None
    actor_bc_deviation_post_update = None
    _qf_used = False
    _qf_mask = None
    _qf_expert_q = None
    _qf_policy_q = None
    policy_due = not uses_td3 or int(learner_update) % max(1, int(policy_delay)) == 0
    if update_actor and policy_due:
        with tf.GradientTape(persistent=bool(gradient_telemetry)) as tape:
            policy_actions = scale_action_tensor(actor(obs, training=True), action_low, action_high)
            policy_q = critic([obs, policy_actions], training=False)
            q_loss = -tf.reduce_mean(policy_q)
            if str(variant) == "td3_bc":
                q_scale = float(td3_bc_alpha) / tf.maximum(
                    tf.reduce_mean(tf.abs(policy_q)), tf.constant(1e-6, dtype=tf.float32)
                )
                q_term = tf.stop_gradient(q_scale) * q_loss
            else:
                q_scale = tf.constant(1.0, dtype=tf.float32)
                q_term = q_loss
            # Ramp Q pressure after warmup so BC anchors the first actor updates.
            weighted_q_term = float(q_weight) * q_term

            demo_batch = sample_demo_actions(demo_data, demo_batch_size)
            bc_term = None
            if variant_uses_joint_bc(variant) and demo_batch is not None and bc_weight > 0.0:
                demo_obs_np, demo_actions_np = demo_batch
                demo_obs = tf.convert_to_tensor(demo_obs_np, dtype=tf.float32)
                demo_actions = tf.convert_to_tensor(demo_actions_np, dtype=tf.float32)
                demo_policy_actions = scale_action_tensor(
                    actor(demo_obs, training=True), action_low, action_high
                )
                per_sample_bc = tf.reduce_mean(tf.square(demo_policy_actions - demo_actions), axis=1)
                # Delay Q-filtering until the critic has observed post-warmup actor updates.
                # The filter affects BC selection; q_weight controls the Q objective itself.
                if demo_q_filter and q_filter_active:
                    expert_q = critic([demo_obs, demo_actions], training=False)
                    policy_demo_q = critic([demo_obs, demo_policy_actions], training=False)
                    if uses_td3:
                        expert_q = tf.minimum(expert_q, critic2([demo_obs, demo_actions], training=False))
                        policy_demo_q = tf.minimum(
                            policy_demo_q,
                            critic2([demo_obs, demo_policy_actions], training=False),
                        )
                    mask = tf.cast(tf.squeeze(expert_q > policy_demo_q, axis=1), tf.float32)
                    bc_loss = tf.reduce_sum(mask * per_sample_bc) / tf.maximum(tf.reduce_sum(mask), 1.0)
                    _qf_used = True
                    _qf_mask = mask
                    _qf_expert_q = expert_q
                    _qf_policy_q = policy_demo_q
                else:
                    bc_loss = tf.reduce_mean(per_sample_bc)
                bc_term = float(bc_weight) * bc_loss
            else:
                bc_loss = None
            drive_term = None
            if drive_indices and actor_drive_regularization > 0.0:
                drive_values = tf.gather(policy_actions, drive_indices, axis=1)
                drive_deficit = tf.nn.relu(float(actor_drive_target) - drive_values)
                drive_term = float(actor_drive_regularization) * tf.reduce_mean(tf.square(drive_deficit))

            actor_loss = weighted_q_term
            if bc_term is not None:
                actor_loss = actor_loss + bc_term
            if drive_term is not None:
                actor_loss = actor_loss + drive_term

        if gradient_telemetry:
            q_flat = _flatten_grads(tape.gradient(weighted_q_term, actor.trainable_variables))
            q_grad_norm = float(tf.norm(q_flat).numpy()) if q_flat is not None else 0.0
            if bc_term is not None:
                bc_flat = _flatten_grads(tape.gradient(bc_term, actor.trainable_variables))
                bc_grad_norm = float(tf.norm(bc_flat).numpy()) if bc_flat is not None else 0.0
                if q_flat is not None and bc_flat is not None:
                    denom = tf.maximum(tf.norm(q_flat) * tf.norm(bc_flat), tf.constant(1e-12))
                    grad_cosine = float((tf.reduce_sum(q_flat * bc_flat) / denom).numpy())
        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
        actor_grads = _clip(actor, "actor")(actor_grads)
        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
        if gradient_telemetry:
            del tape
        actor_loss_value = float(actor_loss.numpy())
        q_loss_value = float(q_loss.numpy())
        q_scale_value = float(q_scale.numpy())
        scaled_q_loss_value = float(weighted_q_term.numpy())
        if bc_loss is not None:
            bc_loss_value = float(bc_loss.numpy())
        if bc_reference_actor is not None:
            ref_actions = scale_action_tensor(
                bc_reference_actor(obs, training=False), action_low, action_high
            )
            # Recompute after apply_gradients to measure drift caused by this update.
            actor_bc_deviation_pre_update = float(
                tf.reduce_mean(tf.norm(policy_actions - ref_actions, axis=1)).numpy()
            )
            post_actions = scale_action_tensor(
                actor(obs, training=False), action_low, action_high
            )
            actor_bc_deviation_post_update = float(
                tf.reduce_mean(tf.norm(post_actions - ref_actions, axis=1)).numpy()
            )

    # None means the Q-filter did not run and should be displayed as N/A.
    if _qf_used:
        qf_total = int(_qf_mask.shape[0])
        qf_selected = int(tf.reduce_sum(_qf_mask).numpy())
        qf_fraction = (qf_selected / qf_total) if qf_total else None
        qf_expert_q_mean = float(tf.reduce_mean(_qf_expert_q).numpy())
        qf_policy_q_mean = float(tf.reduce_mean(_qf_policy_q).numpy())
    else:
        qf_total = 0
        qf_selected = 0
        qf_fraction = None
        qf_expert_q_mean = None
        qf_policy_q_mean = None

    target_update_due = (
        int(learner_update) % max(1, int(policy_delay)) == 0
        if uses_td3
        else False
    )
    return {
        "actor_loss": actor_loss_value,
        "q_loss": q_loss_value,
        "bc_loss": bc_loss_value,
        "critic_loss": float(critic_loss.numpy()),
        "critic2_loss": float(critic2_loss.numpy()) if critic2_loss is not None else None,
        "target_update_due": target_update_due,
        "policy_updated": actor_loss_value is not None,
        "q_filter_active": bool(_qf_used),
        "q_filter_selected_fraction": qf_fraction,
        "q_filter_selected_count": qf_selected,
        "q_filter_total_count": qf_total,
        "expert_q_mean": qf_expert_q_mean,
        "policy_q_mean": qf_policy_q_mean,
        # Separate Q/BC telemetry (None when the actor did not update this step).
        "raw_q_loss": q_loss_value,
        "td3_bc_q_scale": q_scale_value,
        "scaled_q_loss": scaled_q_loss_value,
        "q_weight": float(q_weight),
        "q_grad_norm": q_grad_norm,
        "bc_grad_norm": bc_grad_norm,
        "grad_cosine": grad_cosine,
        "actor_bc_deviation_pre_update": actor_bc_deviation_pre_update,
        "actor_bc_deviation_post_update": actor_bc_deviation_post_update,
    }


def train_step(*args, **kwargs):
    """Backward-compatible DDPG update used by older imports and focused tests."""
    result = train_deterministic_step(*args, **kwargs)
    return result["actor_loss"], result["critic_loss"]


def build_deterministic_learner_step(
    actor,
    critic,
    target_actor,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    gamma,
    action_low,
    action_high,
    *,
    grad_clip_norm=10.0,
    grad_clip_adaptive=False,
    grad_clip_k=3.0,
    variant="ddpg",
    drive_indices=None,
    actor_drive_regularization=0.0,
    actor_drive_target=0.65,
    critic2=None,
    target_critic2=None,
    critic2_optimizer=None,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    policy_delay=2,
    demo_data=None,
    demo_batch_size=128,
    demo_q_filter=False,
    td3_bc_alpha=2.5,
    ddpgfd_actor_priority_weight=1e-3,
    batch_size=128,
    compiled=True,
    xla=False,
):
    """Build the DDPG/TD3 gradient step, optionally compiled into a tf.function.

    Only the plain ``ddpg``/``td3`` variants compile: the demonstration variants
    (``ddpg_bc``/``ddpgfd``/``td3_bc``) run Python side-effects mid-step -- prioritized
    replay writeback (``buffer.update_priorities``) and demo sampling -- that do not graph
    cleanly, so they fall back to the eager ``train_deterministic_step``. Sampling stays in
    Python; only the tensor math is graphed. Mirrors build_sac_learner_step. Returns a
    callable ``learner_step(buffer, *, update_actor, learner_update, bc_weight)`` yielding the
    same dict as ``train_deterministic_step``.
    """
    uses_td3 = variant_uses_td3(variant)
    can_compile = bool(compiled) and str(variant) in ("ddpg", "td3")

    if not can_compile:
        print(
            f"Deterministic learner: eager (variant {variant} uses demo/prioritized replay)"
            if compiled
            else "Deterministic learner: eager",
            flush=True,
        )

        def learner_step(buffer, *, update_actor, learner_update, bc_weight=0.0,
                         q_filter_active=True, q_weight=1.0, gradient_telemetry=False,
                         bc_reference_actor=None):
            return train_deterministic_step(
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
                drive_indices=drive_indices,
                actor_drive_regularization=actor_drive_regularization,
                actor_drive_target=actor_drive_target,
                update_actor=update_actor,
                variant=variant,
                learner_update=learner_update,
                critic2=critic2,
                target_critic2=target_critic2,
                critic2_optimizer=critic2_optimizer,
                target_policy_noise=target_policy_noise,
                target_noise_clip=target_noise_clip,
                policy_delay=policy_delay,
                demo_data=demo_data,
                demo_batch_size=demo_batch_size,
                bc_weight=bc_weight,
                demo_q_filter=demo_q_filter,
                q_filter_active=q_filter_active,
                td3_bc_alpha=td3_bc_alpha,
                q_weight=q_weight,
                gradient_telemetry=gradient_telemetry,
                bc_reference_actor=bc_reference_actor,
                ddpgfd_actor_priority_weight=ddpgfd_actor_priority_weight,
                grad_clip_norm=grad_clip_norm,
                grad_clip_adaptive=grad_clip_adaptive,
                grad_clip_k=grad_clip_k,
            )

        return learner_step

    print(f"Deterministic learner: compiled graph{' + XLA' if xla else ''}", flush=True)
    gamma_c = tf.constant(float(gamma), dtype=tf.float32)
    low = tf.constant(np.asarray(action_low, dtype=np.float32).reshape(1, -1))
    high = tf.constant(np.asarray(action_high, dtype=np.float32).reshape(1, -1))
    drive_idx = list(drive_indices) if drive_indices else None
    drive_reg = float(actor_drive_regularization)
    drive_target = float(actor_drive_target)
    noise_scale = float(target_policy_noise)
    noise_clip = float(target_noise_clip)
    delay = max(1, int(policy_delay))

    critic_optimizer.build(critic.trainable_variables)
    actor_optimizer.build(actor.trainable_variables)
    if uses_td3:
        critic2_optimizer.build(critic2.trainable_variables)

    def _compute_targets(rewards, next_obs, dones):
        next_actions = scale_action_tensor(target_actor(next_obs, training=False), low, high)
        if uses_td3:
            half = 0.5 * (high - low)
            noise = tf.random.normal(tf.shape(next_actions), dtype=tf.float32) * noise_scale * half
            noise = tf.clip_by_value(noise, -noise_clip * half, noise_clip * half)
            next_actions = tf.clip_by_value(next_actions + noise, low, high)
        target_q = target_critic([next_obs, next_actions], training=False)
        if uses_td3:
            target_q = tf.minimum(target_q, target_critic2([next_obs, next_actions], training=False))
        return tf.stop_gradient(rewards + (1.0 - dones) * gamma_c * target_q)

    clip_critic = gradient_clipper(
        critic, "critic", grad_clip_norm, grad_clip_adaptive, grad_clip_k)
    clip_actor = gradient_clipper(
        actor, "actor", grad_clip_norm, grad_clip_adaptive, grad_clip_k)
    clip_critic2 = (
        gradient_clipper(
            critic2, "critic2", grad_clip_norm, grad_clip_adaptive, grad_clip_k)
        if critic2 is not None
        else clip_critic)

    def _update_critics_impl(obs, actions, rewards, next_obs, dones):
        y = _compute_targets(rewards, next_obs, dones)
        with tf.GradientTape() as tape:
            q = critic([obs, actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        grads = tape.gradient(critic_loss, critic.trainable_variables)
        grads = clip_critic(grads)
        critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))
        critic2_loss = tf.constant(0.0, dtype=tf.float32)
        if uses_td3:
            with tf.GradientTape() as tape:
                q2 = critic2([obs, actions], training=True)
                critic2_loss = tf.reduce_mean(tf.square(y - q2))
            grads2 = tape.gradient(critic2_loss, critic2.trainable_variables)
            grads2 = clip_critic2(grads2)
            critic2_optimizer.apply_gradients(zip(grads2, critic2.trainable_variables))
        return critic_loss, critic2_loss

    def _update_all_impl(obs, actions, rewards, next_obs, dones):
        critic_loss, critic2_loss = _update_critics_impl(obs, actions, rewards, next_obs, dones)
        with tf.GradientTape() as tape:
            policy_actions = scale_action_tensor(actor(obs, training=True), low, high)
            policy_q = critic([obs, policy_actions], training=False)
            q_loss = -tf.reduce_mean(policy_q)
            actor_loss = q_loss
            if drive_idx and drive_reg > 0.0:
                drive_values = tf.gather(policy_actions, drive_idx, axis=1)
                drive_deficit = tf.nn.relu(drive_target - drive_values)
                actor_loss = actor_loss + drive_reg * tf.reduce_mean(tf.square(drive_deficit))
        grads = tape.gradient(actor_loss, actor.trainable_variables)
        grads = clip_actor(grads)
        actor_optimizer.apply_gradients(zip(grads, actor.trainable_variables))
        return actor_loss, q_loss, critic_loss, critic2_loss

    if compiled:
        update_critics = tf.function(_update_critics_impl, reduce_retracing=True, jit_compile=xla)
        update_all = tf.function(_update_all_impl, reduce_retracing=True, jit_compile=xla)
    else:
        update_critics = _update_critics_impl
        update_all = _update_all_impl

    def learner_step(buffer, *, update_actor, learner_update, bc_weight=0.0,
                     q_filter_active=True, q_weight=1.0, gradient_telemetry=False,
                     bc_reference_actor=None):
        # Plain compiled variants accept the shared signature but do not use demo controls.
        obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, action_dtype=np.float32)
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards.reshape(-1, 1), dtype=tf.float32)
        next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones.reshape(-1, 1), dtype=tf.float32)
        policy_due = (not uses_td3) or (int(learner_update) % delay == 0)
        do_actor = bool(update_actor) and policy_due
        if do_actor:
            actor_loss, q_loss, critic_loss, critic2_loss = update_all(
                obs, actions, rewards, next_obs, dones
            )
            actor_loss_value = float(actor_loss.numpy())
            q_loss_value = float(q_loss.numpy())
        else:
            critic_loss, critic2_loss = update_critics(obs, actions, rewards, next_obs, dones)
            actor_loss_value = None
            q_loss_value = None
        target_update_due = (int(learner_update) % delay == 0) if uses_td3 else False
        return {
            "actor_loss": actor_loss_value,
            "q_loss": q_loss_value,
            "bc_loss": None,
            "critic_loss": float(critic_loss.numpy()),
            "critic2_loss": float(critic2_loss.numpy()) if uses_td3 else None,
            "target_update_due": target_update_due,
            "policy_updated": actor_loss_value is not None,
            "q_filter_active": False,
            "q_filter_selected_fraction": None,
            "q_filter_selected_count": 0,
            "q_filter_total_count": 0,
            "expert_q_mean": None,
            "policy_q_mean": None,
            "raw_q_loss": q_loss_value,
            "td3_bc_q_scale": None,
            "scaled_q_loss": q_loss_value,
            "q_weight": float(q_weight),
            "q_grad_norm": None,
            "bc_grad_norm": None,
            "grad_cosine": None,
            "actor_bc_deviation_pre_update": None,
            "actor_bc_deviation_post_update": None,
        }

    return learner_step


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
    policy_artifact = getattr(args, "policy_artifact", None)
    if policy_artifact is not None:
        policy_artifact.save(episode)
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




def load_demonstration_arrays(paths, obs_dim, action_size, max_transitions=0):
    obs_parts = []
    action_parts = []
    reward_parts = []
    next_obs_parts = []
    done_parts = []
    agent_id_parts = []
    group_parts = []
    group_offset = 0
    has_agent_ids = True
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
            agent_ids = (
                np.asarray(data["agent_ids"], dtype=str)
                if "agent_ids" in data
                else None
            )
            # Prefer explicit group ids, then episode ids, for balanced BC sampling.
            if "group_ids" in data:
                file_groups = np.asarray(data["group_ids"]).astype(np.int64)
            elif "episode_indices" in data:
                file_groups = np.asarray(data["episode_indices"]).astype(np.int64)
            else:
                file_groups = np.zeros(len(actions), dtype=np.int64)

        if action_type != "continuous":
            raise ValueError(f"Demo {path!r} has action_type={action_type!r}; DDPG requires continuous demos")
        if obs.ndim != 2 or obs.shape[1] != obs_dim:
            raise ValueError(f"Demo {path!r} obs shape {obs.shape} does not match obs_dim={obs_dim}")
        if next_obs.shape != obs.shape:
            raise ValueError(f"Demo {path!r} next_obs shape {next_obs.shape} does not match obs shape {obs.shape}")
        if actions.ndim != 2 or actions.shape[1] != action_size:
            raise ValueError(f"Demo {path!r} actions shape {actions.shape} does not match action_size={action_size}")
        if agent_ids is not None and len(agent_ids) != len(actions):
            raise ValueError(
                f"Demo {path!r} agent_ids length {len(agent_ids)} does not match "
                f"transitions={len(actions)}"
            )

        count = len(actions)
        if remaining > 0:
            count = min(count, remaining)
            remaining -= count

        obs_parts.append(obs[:count])
        action_parts.append(actions[:count])
        reward_parts.append(rewards[:count])
        next_obs_parts.append(next_obs[:count])
        done_parts.append(dones[:count])
        # Keep trajectory ids disjoint when combining files.
        file_g = file_groups[:count] + group_offset
        group_parts.append(file_g)
        group_offset = int(file_g.max()) + 1 if len(file_g) else group_offset
        if agent_ids is None:
            has_agent_ids = False
        else:
            agent_id_parts.append(agent_ids[:count])

        if remaining == 0 and max_transitions > 0:
            break

    if not obs_parts:
        return None

    result = {
        "obs": np.concatenate(obs_parts, axis=0),
        "actions": np.concatenate(action_parts, axis=0),
        "rewards": np.concatenate(reward_parts, axis=0),
        "next_obs": np.concatenate(next_obs_parts, axis=0),
        "dones": np.concatenate(done_parts, axis=0),
    }
    if has_agent_ids and len(agent_id_parts) == len(obs_parts):
        result["agent_ids"] = np.concatenate(agent_id_parts, axis=0)
    result["group_ids"] = np.concatenate(group_parts, axis=0)
    return result


def partition_demonstrations_by_policy(demo_data, assignment):
    if demo_data is None:
        return {policy_id: None for policy_id in assignment.policy_ids}
    if "agent_ids" not in demo_data:
        raise ValueError(
            "Independent multi-policy demonstration training requires recordings with "
            "agent_ids. Re-record with python/recorder.py (and --record-all-agents when "
            "the demonstration should cover every policy)."
        )

    agent_ids = np.asarray(demo_data["agent_ids"], dtype=str)
    unknown = sorted(set(agent_ids.tolist()) - set(assignment.agent_to_policy))
    if unknown:
        raise ValueError(
            "Demonstrations contain agents that are not present in the current policy "
            f"assignment: {unknown}"
        )

    partitioned = {}
    for policy_id in assignment.policy_ids:
        mask = np.asarray(
            [
                assignment.policy_for_agent(agent_id) == policy_id
                for agent_id in agent_ids
            ],
            dtype=np.bool_,
        )
        if not np.any(mask):
            partitioned[policy_id] = None
            continue
        partitioned[policy_id] = {
            key: np.asarray(value)[mask]
            for key, value in demo_data.items()
            if key != "agent_ids"
        }
        partitioned[policy_id]["agent_ids"] = agent_ids[mask]
    return partitioned


def _run_critic_warmup_audit(args, checkpoint, checkpoint_manager, buffer, episode,
                             actor, critic, critic2, demo_validation,
                             action_low, action_high):
    """Q-ranking audit at the end of the critic warmup (actor still frozen), then STOP.

    Prints ``CRITIC_WARMUP_AUDIT`` + a machine-readable ``CRITIC_WARMUP_AUDIT_JSON`` line, saves the
    checkpoint (critic warmed, actor = BC clone) and exits BEFORE any actor update. Fail-closed:
    exits non-zero if the validation batch is missing or the gate fails (critic does NOT value
    expert+clone above random+saturated), so no downstream step can mistake a bad critic for a
    green light to unfreeze the actor.
    """
    if demo_validation is None or "obs" not in demo_validation or len(demo_validation.get("actions", [])) == 0:
        raise SystemExit(
            "stop-after-critic-warmup: no validation batch (pass --demo-validation-path); "
            "cannot audit the critic. STOPPING fail-closed."
        )
    obs = np.asarray(demo_validation["obs"], dtype=np.float32)
    expert_actions = np.asarray(demo_validation["actions"], dtype=np.float32)
    # Cell ids enable worst-cell auditing; without them the audit falls back to win rate.
    group_ids = demo_validation.get("cells") if isinstance(demo_validation, dict) else None
    if group_ids is None:
        parts = []
        for path in (getattr(args, "demo_validation_path", None) or []):
            try:
                with np.load(path, allow_pickle=True) as z:
                    if "cells" in z:
                        parts.append(np.asarray(z["cells"]))
            except Exception:
                pass
        if parts:
            group_ids = np.concatenate(parts)
    if group_ids is not None and len(group_ids) != len(obs):
        group_ids = None  # length mismatch -> do not risk mislabelling; win-rate gate still applies

    def _actor_fn(o):
        return actor(o, training=False).numpy()

    def _critic_fn(oa):
        return critic(oa, training=False).numpy()

    _critic2_fn = None
    if critic2 is not None:
        def _critic2_fn(oa):
            return critic2(oa, training=False).numpy()

    report = q_ranking_audit(
        _critic_fn, _actor_fn, obs, expert_actions, action_low, action_high,
        critic2=_critic2_fn, group_ids=group_ids,
        win_rate_threshold=float(getattr(args, "critic_audit_win_rate", 0.9)),
        cell_margin_tol=float(getattr(args, "critic_audit_cell_margin_tol", 0.0)),
    )
    print(format_audit(report), flush=True)
    print("CRITIC_WARMUP_AUDIT_JSON " + json.dumps({
        "episode": int(episode),
        "critic_updates": int(getattr(args, "critic_warmup_updates", 0)),
        **{f"q_{k}": v for k, v in report["q_mean"].items()},
        "pairs": {k: {kk: vv for kk, vv in v.items() if kk != "cell_means"}
                  for k, v in report["pairs"].items()},
        "gate": report["gate"],
        "gate_passed": report["gate_passed"],
        "n": report["n"],
    }), flush=True)

    save_training_checkpoint(
        checkpoint, checkpoint_manager, buffer, episode,
        getattr(args, "exploration_noise", 0.0), args, final=True,
    )
    if report["gate_passed"]:
        print("stop-after-critic-warmup: GATE PASSED -- critic ranks expert+clone above "
              "random+saturated. Actor updates would be allowed. STOPPING as requested.", flush=True)
        raise SystemExit(0)
    raise SystemExit(
        "stop-after-critic-warmup: GATE FAILED -- critic does NOT value expert/clone above "
        "random/saturated actions. Actor updates are BLOCKED (they would collapse the clone). "
        "STOPPING fail-closed."
    )


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
    best_tracker,
    start_episode,
    start_noise_std,
    action_low,
    action_high,
    random_action_low,
    random_action_high,
    actor_drive_indices,
    critic_warmup_target,
    policy_update_counter,
    budget=None,
    critic2=None,
    target_critic2=None,
    critic2_optimizer=None,
    demo_data=None,
    demo_validation=None,
):
    validate_async_arguments(args)
    budget = budget or TrainingBudget(getattr(args, "total_timesteps", 0))
    obs_dim = envs[0].obs_dim
    action_size = envs[0].action_size
    with tf.device("/CPU:0"):
        local_models = [
            build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
            for _env in envs
        ]
    # One traced forward per collector: eager actor calls corrupt output under the
    # concurrent collector threads. Noise is added in numpy afterwards, so only the
    # forward needs tracing.
    actor_forwards = [build_actor_forward_fn(m, obs_dim) for m in local_models]
    snapshot = PolicySnapshot(actor.get_weights())
    health_monitor = getattr(best_tracker, "health_monitor", None)
    recovery_handler = getattr(health_monitor, "recovery_handler", None)
    if recovery_handler is not None:
        recovery_handler.set_policy_publisher(
            lambda: snapshot.publish(actor.get_weights())
        )
    rngs = [np.random.default_rng(args.env_seed_base + 100_003 * idx) for idx in range(len(envs))]
    replay_ready_event = build_replay_ready_event(buffer, args)

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
            raw_action = actor_forwards[worker_id](observations).numpy()
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
        replay_ready_event=replay_ready_event,
    )
    pool = AsyncCollectorPool(
        envs,
        worker,
        start_episode,
        args.num_episodes,
        queue_capacity=args.async_queue_capacity,
    )
    scheduler = AsyncEventScheduler(args)
    def refill_recovery_replay():
        if demo_data is None or not args.demo_prefill:
            return 0
        return buffer.add_many(
            demo_data["obs"],
            demo_data["actions"],
            demo_data["rewards"],
            demo_data["next_obs"],
            demo_data["dones"],
            is_demo=args.trainer_variant == "ddpgfd",
            protect=args.trainer_variant == "ddpgfd",
        )

    def sync_recovery_targets():
        target_actor.set_weights(actor.get_weights())
        target_critic.set_weights(critic.get_weights())
        if critic2 is not None:
            target_critic2.set_weights(critic2.get_weights())

    recovery_runtime = OffPolicyRecoveryRuntime(
        args,
        buffer,
        initial_critic_warmup=critic_warmup_target,
        replay_refill=refill_recovery_replay,
        target_sync=sync_recovery_targets,
    )
    recovery_runtime.attach_async(pool, scheduler, snapshot)
    if recovery_handler is not None:
        recovery_handler.set_post_restore(recovery_runtime.post_restore)
    print(
        f"Collector mode: async workers={len(envs)} queue={args.async_queue_capacity} "
        f"policy_sync_steps={args.async_policy_sync_steps} "
        f"update_basis={scheduler.update_basis} update_every={scheduler.update_every} "
        f"updates_per_interval={scheduler.updates_per_interval} "
        f"max_updates_per_env_step={scheduler.max_updates_per_env_step}",
        flush=True,
    )
    det_learner = build_deterministic_learner_step(
        actor,
        critic,
        target_actor,
        target_critic,
        actor_optimizer,
        critic_optimizer,
        args.gamma,
        action_low,
        action_high,
        variant=args.trainer_variant,
        drive_indices=actor_drive_indices,
        actor_drive_regularization=args.actor_drive_regularization,
        actor_drive_target=args.actor_drive_target,
        critic2=critic2,
        target_critic2=target_critic2,
        critic2_optimizer=critic2_optimizer,
        target_policy_noise=args.td3_target_policy_noise,
        target_noise_clip=args.td3_target_noise_clip,
        policy_delay=args.td3_policy_delay,
        demo_data=demo_data,
        demo_batch_size=args.demo_bc_batch_size,
        demo_q_filter=args.demo_q_filter,
        td3_bc_alpha=args.td3_bc_alpha,
        ddpgfd_actor_priority_weight=args.ddpgfd_actor_priority_weight,
        batch_size=args.batch_size,
        compiled=args.tf_compile_learner,
        xla=args.tf_xla,
        grad_clip_norm=args.grad_clip_norm,
        grad_clip_adaptive=args.grad_clip_adaptive,
        grad_clip_k=args.grad_clip_k,
    )
    completed = int(start_episode)
    done_workers = 0
    learner_updates = int(critic_optimizer.iterations.numpy())
    actor_losses = []
    critic_losses = []
    critic2_losses = []
    bc_losses = []
    # Count Q-filter delay in real actor updates after critic warmup.
    q_filter_start = max(0, int(getattr(args, "demo_q_filter_start_policy_updates", 0)))
    last_bc_weight = 0.0
    qf_last = {"active": False, "fraction": None, "selected": 0, "total": 0,
               "expert_q": None, "policy_q": None}
    # Keep a frozen BC reference for actor-deviation telemetry.
    bc_reference_actor = None
    if getattr(args, "gradient_telemetry_every", 0) > 0 and variant_uses_joint_bc(args.trainer_variant):
        bc_reference_actor = tf.keras.models.clone_model(actor)
        bc_reference_actor.set_weights(actor.get_weights())
    last_saved_episode = None
    interrupted = False
    pool.start()
    try:
        while done_workers < len(envs):
            apply_ready_best_checkpoint(best_tracker)
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
                step_events = [
                    item
                    for item in scheduler.drain_step_events(pool, event)
                    if recovery_runtime.accepts(item)
                ]
                if not step_events:
                    continue
                collected_transitions = 0
                for step_event in step_events:
                    for transition in step_event.transitions:
                        buffer.add(*transition)
                        collected_transitions += 1
                if len(buffer) >= replay_warmup_threshold(args):
                    replay_ready_event.set()
                budget.consume(collected_transitions)
                updates_due = scheduler.ingest(step_events)
                updates_performed = 0
                if len(buffer) >= max(args.replay_warmup, args.batch_size):
                    for _update in range(updates_due):
                        update_actor = recovery_runtime.should_update_policy()
                        next_update = int(critic_optimizer.iterations.numpy()) + 1
                        # Decay BC weight only when the actor actually updates.
                        policy_updates_since_warmup = policy_update_count(
                            policy_update_counter
                        )
                        bc_weight = scheduled_bc_weight(
                            policy_updates_since_warmup,
                            args.demo_bc_weight_start,
                            args.demo_bc_weight_end,
                            args.demo_bc_decay_updates,
                        )
                        # Keep Q-filtering off until enough actor updates have run.
                        q_filter_active = policy_updates_since_warmup >= q_filter_start
                        # Ramp Q pressure from its configured start after warmup.
                        q_weight = (
                            scheduled_q_weight(
                                policy_updates_since_warmup,
                                args.demo_q_weight_start,
                                args.demo_q_weight_end,
                                args.demo_q_weight_ramp_updates,
                            )
                            if variant_uses_joint_bc(args.trainer_variant)
                            else 1.0
                        )
                        grad_tele = (
                            args.gradient_telemetry_every > 0
                            and policy_updates_since_warmup % args.gradient_telemetry_every == 0
                        )
                        last_bc_weight = bc_weight
                        result = det_learner(
                            buffer,
                            update_actor=update_actor,
                            learner_update=next_update,
                            bc_weight=bc_weight,
                            q_filter_active=q_filter_active,
                            q_weight=q_weight,
                            gradient_telemetry=grad_tele,
                            bc_reference_actor=bc_reference_actor,
                        )
                        if result["policy_updated"]:
                            increment_policy_update_count(policy_update_counter)
                        if result.get("q_filter_active"):
                            qf_last = {
                                "active": True,
                                "fraction": result["q_filter_selected_fraction"],
                                "selected": result["q_filter_selected_count"],
                                "total": result["q_filter_total_count"],
                                "expert_q": result["expert_q_mean"],
                                "policy_q": result["policy_q_mean"],
                            }
                        elif result["policy_updated"]:
                            qf_last["active"] = False  # updating, filter not yet enabled -> N/A
                        critic_losses.append(result["critic_loss"])
                        if result["critic2_loss"] is not None:
                            critic2_losses.append(result["critic2_loss"])
                        if result["bc_loss"] is not None:
                            bc_losses.append(result["bc_loss"])
                        warmup_completed = recovery_runtime.record_critic_update()
                        if warmup_completed and getattr(args, "stop_after_critic_warmup", False):
                            # Audit at the end of warmup, before the first actor update.
                            _run_critic_warmup_audit(
                                args, checkpoint, checkpoint_manager, buffer, completed,
                                actor, critic, critic2, demo_validation,
                                action_low, action_high,
                            )
                        learner_updates = int(critic_optimizer.iterations.numpy())
                        updates_performed += 1
                        if result["actor_loss"] is not None:
                            actor_losses.append(result["actor_loss"])
                        target_due = (
                            result["target_update_due"]
                            if variant_uses_td3(args.trainer_variant)
                            else learner_updates % args.target_update_every == 0
                        )
                        if target_due:
                            soft_update(target_actor, actor, args.tau)
                            soft_update(target_critic, critic, args.tau)
                            if critic2 is not None:
                                soft_update(target_critic2, critic2, args.tau)
                        if learner_updates % args.async_policy_publish_updates == 0:
                            snapshot.publish(actor.get_weights())
                scheduler.record_updates(updates_performed)
                if budget.exhausted:
                    print(
                        f"Transition budget reached: {budget.collected}/{budget.limit}",
                        flush=True,
                    )
                    break
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
            replay_warmup_left = replay_warmup_remaining(buffer, args)
            critic_warmup_left = recovery_runtime.warmup_left
            throughput = scheduler.throughput(pool)
            print_episode_metrics(event.episode, [
                ("mode", [
                    ("collector", "async"),
                    ("algorithm", args.trainer_variant),
                    ("worker", event.worker_id),
                    ("exploration", exploration_label(state)),
                    ("noise", f"{noise_std:.3f}"),
                ]),
                ("outcome", [
                    ("reward", f"{reward_stats['mean']:.3f} [{reward_stats['min']:.3f}, {reward_stats['max']:.3f}]"),
                    ("progress", f"mean:{diagnostics['progress_mean']:.3f} max:{diagnostics['progress_max']:.3f}"),
                ]),
                ("agents", [
                    ("finish", f"{diagnostics['finishes']}/{controlled_agents}"),
                    ("collision", format_collision_diagnostics(
                        diagnostics, controlled_agents)),
                    ("stall", f"{diagnostics['stalls']}/{controlled_agents}"),
                ]),
                ("actions", [("mean", format_float_list(mean_action)), ("delta", format_float_list(mean_delta))]),
                ("training", [
                    ("completed", f"{completed}/{args.num_episodes}"),
                    ("total_timesteps", budget.collected),
                    ("queue", f"{throughput['queue_size']}/{throughput['queue_capacity']} ({throughput['queue_saturation']:.0%})"),
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    *(([("protected_demos", buffer.protected_demo_count)]) if args.trainer_variant == "ddpgfd" else []),
                    ("critic_updates", len(critic_losses)),
                    ("policy_updates", len(actor_losses)),
                    ("replay_warmup_left", replay_warmup_left),
                    ("critic_warmup_left", critic_warmup_left),
                    ("actor_loss", f"{float(np.mean(actor_losses)) if actor_losses else 0.0:.5f}"),
                    ("critic_loss", f"{float(np.mean(critic_losses)) if critic_losses else 0.0:.5f}"),
                    *(([("critic2_loss", f"{float(np.mean(critic2_losses)):.5f}")]) if critic2_losses else []),
                    *(([("bc_loss", f"{float(np.mean(bc_losses)):.5f}")]) if bc_losses else []),
                    ("policy_version", snapshot.version),
                ]),
                *(([("bc_qfilter", [
                    ("bc_weight", f"{last_bc_weight:.3f}"),
                    (
                        "policy_updates_since_warmup",
                        policy_update_count(policy_update_counter),
                    ),
                    ("q_filter_active", qf_last["active"]),
                    ("q_filter_selected_fraction",
                     "n/a" if qf_last["fraction"] is None else f"{qf_last['fraction']:.3f}"),
                    ("q_filter_selected_count", qf_last["selected"]),
                    ("q_filter_total_count", qf_last["total"]),
                    ("expert_q_mean",
                     "n/a" if qf_last["expert_q"] is None else f"{qf_last['expert_q']:.3f}"),
                    ("policy_q_mean",
                     "n/a" if qf_last["policy_q"] is None else f"{qf_last['policy_q']:.3f}"),
                    ("action_saturation",
                     f"{float(np.mean(np.abs(np.asarray(mean_action)) > 0.95)):.3f}"),
                ])]) if variant_uses_joint_bc(args.trainer_variant) else []),
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
            critic2_losses.clear()
            bc_losses.clear()
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
        print(
            f"\nInterrupt received: stopping async {args.trainer_variant} collectors...",
            flush=True,
        )
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
    else:
        # Collect the evaluation requested near the last episode. Skipped on interrupt:
        # Ctrl-C should exit, not wait.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
    return completed, noise_std


def _run_stop_after_bc_eval_common(args, checkpoint, checkpoint_manager, best_tracker,
                                   start_episode, buffer):
    """Save a BC checkpoint, run ONE frozen (deterministic) evaluation at the current curriculum
    level, then stop before RL. Task-agnostic (initial level means Stage A only for tasks that
    define it so). A failed/empty evaluation exits NON-ZERO instead of a passing gate; the summary
    includes whatever generic reset/cell breakdowns the evaluation produced."""
    print("stop-after-bc: saving BC checkpoint + a frozen evaluation at the current curriculum "
          "level (RL NOT started)...", flush=True)
    saved_path = save_training_checkpoint(
        checkpoint, checkpoint_manager, buffer, start_episode, 0.0, args,
        final=True, save_replay=False)
    result = None
    try:
        result = best_tracker.evaluate(str(saved_path), start_episode)
    except Exception as exc:
        print(f"stop-after-bc: frozen evaluation raised: {exc}", flush=True)
    if result is None or not getattr(result, "summary", None):
        raise SystemExit("stop-after-bc: frozen evaluation produced no summary (failed/timeout); "
                         "exiting non-zero without a gate result.")
    s = result.summary
    breakdown = {k: s[k] for k in ("reset_modes", "target_regions", "target_cells",
                 "region_success_floor", "regular_success_rate") if k in s}
    print("BC_FROZEN_EVAL " + json.dumps({
        "success_rate": float(s.get("success_rate", 0.0)),
        "selection_success_rate": float(s.get("selection_success_rate", s.get("success_rate", 0.0))),
        "position_error_mean": s.get("position_error_mean"),
        "orientation_error_mean": s.get("orientation_error_mean"),
        "hold_frames_max": s.get("hold_frames_max"),
        "episodes": int(getattr(args, "best_evaluation_episodes", 0)),
        "curriculum_level": s.get("curriculum_level"),
        "breakdown": breakdown}), flush=True)
    print("stop-after-bc: STOPPED before RL.", flush=True)


def _load_bc_pairs(paths, obs_dim, action_size):
    """Imitation-only (obs, actions) pairs (no rewards/next_obs/dones). None if paths empty."""
    obs_parts, act_parts = [], []
    for path in paths:
        with np.load(path) as d:
            if "obs" not in d or "actions" not in d:
                raise ValueError(f"BC-pairs file {path!r} must contain obs and actions arrays")
            o = np.asarray(d["obs"], dtype=np.float32)
            a = np.asarray(d["actions"], dtype=np.float32)
        if o.ndim != 2 or o.shape[1] != obs_dim:
            raise ValueError(f"BC-pairs {path!r} obs shape {o.shape} != obs_dim {obs_dim}")
        if a.ndim != 2 or a.shape[1] != action_size:
            raise ValueError(f"BC-pairs {path!r} actions shape {a.shape} != action_size {action_size}")
        obs_parts.append(o)
        act_parts.append(a)
    if not obs_parts:
        return None
    return {"obs": np.concatenate(obs_parts, axis=0), "actions": np.concatenate(act_parts, axis=0)}


def _combine_bc_sources(demo_data, bc_pairs):
    """Concatenate complete-transition demos (--demo-path) and imitation-only pairs (--demo-bc-path)
    into one BC training set. Replay-only demos are intentionally NOT included here."""
    obs_parts, act_parts = [], []
    for src in (demo_data, bc_pairs):
        if src is None or len(src.get("actions", [])) == 0:
            continue
        obs_parts.append(np.asarray(src["obs"], dtype=np.float32))
        act_parts.append(np.asarray(src["actions"], dtype=np.float32))
    if not obs_parts:
        return None
    return {"obs": np.concatenate(obs_parts, axis=0), "actions": np.concatenate(act_parts, axis=0)}


def _bc_actor_path_default(actor_weights_path):
    """…foo.weights.h5 -> …foo_bc.weights.h5 (persist the BC actor apart from RL weights)."""
    p = str(actor_weights_path or "bc_actor.weights.h5")
    if p.endswith(".weights.h5"):
        return p[: -len(".weights.h5")] + "_bc.weights.h5"
    root, dot, ext = p.rpartition(".")
    return f"{root}_bc.{ext}" if dot else f"{p}_bc"


def _bc_validation_mse(actor, obs, actions, low_t, high_t, batch_size):
    """Held-out action MSE for a deterministic (DDPG/TD3) actor, batched."""
    total = 0.0
    n = len(actions)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        pred = scale_action_tensor(
            actor(tf.convert_to_tensor(obs[start:stop], dtype=tf.float32), training=False),
            low_t, high_t)
        err = tf.reduce_sum(tf.square(
            tf.convert_to_tensor(actions[start:stop], dtype=tf.float32) - pred))
        total += float(err.numpy())
    return total / max(n * int(actions.shape[1]), 1)


def pretrain_actor_behavior_cloning(actor, demo_data, epochs, batch_size, learning_rate,
                                    action_low, action_high, demo_validation=None):
    """Behavior-clone a deterministic actor onto the demo actions. With a validation set, restore
    the epoch with the lowest validation MSE (best-checkpoint / early-stop against overfitting).
    Returns {"best_epoch", "best_val_mse"}."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    obs = demo_data["obs"]
    actions = demo_data["actions"]
    action_low_tensor = tf.convert_to_tensor(np.asarray(action_low, dtype=np.float32).reshape(1, -1), dtype=tf.float32)
    action_high_tensor = tf.convert_to_tensor(np.asarray(action_high, dtype=np.float32).reshape(1, -1), dtype=tf.float32)
    count = len(actions)

    val_obs = val_act = None
    if demo_validation is not None and len(demo_validation.get("actions", [])) > 0:
        val_obs = demo_validation["obs"]
        val_act = np.asarray(demo_validation["actions"], dtype=np.float32)

    best_val = None
    best_epoch = -1
    best_weights = None
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

        val_msg = ""
        if val_act is not None:
            val_mse = _bc_validation_mse(
                actor, val_obs, val_act, action_low_tensor, action_high_tensor, batch_size)
            val_msg = f" val_mse={val_mse:.6f}"
            if best_val is None or val_mse < best_val:
                best_val = val_mse
                best_epoch = epoch + 1
                best_weights = actor.get_weights()
        print(
            f"demo_bc_epoch={epoch + 1:04d}/{epochs:04d} "
            f"actor_mse={float(np.mean(losses)):.6f}{val_msg}",
            flush=True,
        )

    if best_weights is not None:
        actor.set_weights(best_weights)
        print(f"BC best epoch={best_epoch}/{epochs} val_mse={best_val:.6f} "
              "(restored best-validation weights)", flush=True)
    return {"best_epoch": best_epoch, "best_val_mse": best_val}


def apply_variant_defaults(args):
    variant = str(args.trainer_variant)
    if variant == "ddpg":
        return
    if args.actor_weights_path == "generic_ddpg_actor.weights.h5":
        args.actor_weights_path = f"generic_{variant}_actor.weights.h5"
    if args.critic_weights_path == "generic_ddpg_critic.weights.h5":
        args.critic_weights_path = f"generic_{variant}_critic1.weights.h5"
    if variant_uses_td3(variant) and args.critic2_weights_path == "generic_td3_critic2.weights.h5":
        args.critic2_weights_path = f"generic_{variant}_critic2.weights.h5"
    if args.checkpoint_dir == "checkpoints/generic_ddpg":
        args.checkpoint_dir = f"checkpoints/generic_{variant}"


def validate_variant_arguments(args):
    if variant_uses_joint_bc(args.trainer_variant) or args.trainer_variant == "ddpgfd":
        if not args.demo_path:
            raise ValueError(f"Algorithm {args.trainer_variant} requires at least one --demo-path")
    if args.trainer_variant == "ddpgfd" and not args.demo_prefill:
        raise ValueError("DDPGfD requires --demo-prefill so expert transitions remain in replay")
    if args.demo_bc_weight_start < 0.0 or args.demo_bc_weight_end < 0.0:
        raise ValueError("BC weights cannot be negative")
    if args.demo_bc_decay_updates < 1:
        raise ValueError("--demo-bc-decay-updates must be at least 1")
    if args.td3_policy_delay < 1:
        raise ValueError("--td3-policy-delay must be at least 1")
    if args.td3_target_policy_noise < 0.0 or args.td3_target_noise_clip < 0.0:
        raise ValueError("TD3 target noise values cannot be negative")
    if args.ddpgfd_pretrain_updates < 0:
        raise ValueError("--ddpgfd-pretrain-updates cannot be negative")
    if not 0.0 <= args.ddpgfd_priority_alpha <= 1.0:
        raise ValueError("--ddpgfd-priority-alpha must be in [0, 1]")
    if not 0.0 <= args.ddpgfd_priority_beta <= 1.0:
        raise ValueError("--ddpgfd-priority-beta must be in [0, 1]")


@dataclass
class DeterministicPolicyState:
    policy_id: str
    actor: object
    critic: object
    target_actor: object
    target_critic: object
    actor_optimizer: object
    critic_optimizer: object
    buffer: ReplayBuffer
    learner: object
    recovery: OffPolicyRecoveryRuntime
    artifact: PolicyArtifactSaver
    noise_std: object
    trainable: bool
    critic2: object = None
    target_critic2: object = None
    critic2_optimizer: object = None
    # Post-warmup actor updates drive BC decay and Q-filter delay.
    policy_updates_since_warmup: object = None


def _save_multi_policy_deterministic_checkpoint(
    checkpoint,
    checkpoint_manager,
    policy_states,
    assignment,
    episode,
    args,
    *,
    final=False,
):
    checkpoint.episode.assign(int(episode))
    saved_path = checkpoint_manager.save(checkpoint_number=int(episode))
    print(
        f"Saved {'final ' if final else ''}multi-policy checkpoint: {saved_path}",
        flush=True,
    )
    for state in policy_states.values():
        state.artifact.save(episode)
    write_multi_policy_manifest(
        args.checkpoint_dir,
        assignment,
        args.trainer_variant,
        episode=episode,
        policy_metadata={
            policy_id: {
                "noise_std": float(state.noise_std.numpy()),
                "replay_size": len(state.buffer),
                "policy_updates_since_warmup": policy_update_count(
                    state.policy_updates_since_warmup
                ),
            }
            for policy_id, state in policy_states.items()
        },
    )
    if args.save_replay_buffer:
        save_policy_replays(
            saved_path,
            checkpoint_manager,
            assignment,
            {
                policy_id: state.buffer
                for policy_id, state in policy_states.items()
                if state.trainable
            },
        )
    return saved_path


def _multi_policy_deterministic_recovery(policy_states):
    def post_restore(request):
        outcomes = {
            policy_id: state.recovery.post_restore(request)
            for policy_id, state in policy_states.items()
            if state.trainable
        }
        return {"multi_policy": outcomes}

    return post_restore


def run_async_multi_policy_deterministic(
    args,
    envs,
    policy_states,
    assignment,
    checkpoint,
    checkpoint_manager,
    best_tracker,
    start_episode,
    action_low,
    action_high,
    random_low,
    random_high,
    budget,
):
    env0 = envs[0]
    obs_dim = env0.obs_dim
    action_size = env0.action_size
    with tf.device("/CPU:0"):
        local_models = [
            {
                policy_id: build_continuous_actor(
                    obs_dim=obs_dim,
                    action_size=action_size,
                )
                for policy_id in assignment.policy_ids
            }
            for _env in envs
        ]
    actor_forwards = [
        {
            policy_id: build_actor_forward_fn(model, obs_dim)
            for policy_id, model in worker_models.items()
        }
        for worker_models in local_models
    ]

    def live_weights():
        return {
            policy_id: state.actor.get_weights()
            for policy_id, state in policy_states.items()
        }

    snapshot = MultiPolicySnapshot(live_weights())
    rngs = [
        np.random.default_rng(args.env_seed_base + 100_003 * worker_id)
        for worker_id in range(len(envs))
    ]
    initial_noise = {
        policy_id: float(state.noise_std.numpy())
        for policy_id, state in policy_states.items()
    }
    trainable_states = [
        state for state in policy_states.values() if state.trainable
    ]
    if not trainable_states:
        raise ValueError("Multi-policy training requires at least one trainable policy")
    replay_ready_event = build_replay_ready_event(
        trainable_states[0].buffer,
        args,
    )
    if any(
        replay_warmup_remaining(state.buffer, args) > 0
        for state in trainable_states
    ):
        replay_ready_event.clear()

    def noise_for(policy_id, episode):
        if not policy_states[policy_id].trainable:
            return 0.0
        elapsed = max(0, int(episode) - int(start_episode))
        return max(
            args.exploration_noise_min,
            initial_noise[policy_id]
            * (args.exploration_noise_decay ** elapsed),
        )

    def action_selector(
        worker_id,
        env,
        episode,
        _step_idx,
        worker_models,
        state,
    ):
        rng = rngs[worker_id]
        if state["use_random_exploration"]:
            actions = sample_exploratory_action(
                action_low,
                action_high,
                env.action_names,
                rng=rng,
                num_agents=len(env.agent_ids),
                drive_min=args.random_drive_min,
                steering_abs_max=args.random_steering_abs_max,
                exploration_low=random_low,
                exploration_high=random_high,
            )
            actions[state["done_mask"]] = action_low
            for agent_idx, agent_id in enumerate(env.agent_ids):
                policy_id = assignment.policy_for_agent(agent_id)
                if (
                    state["done_mask"][agent_idx]
                    or policy_states[policy_id].trainable
                ):
                    continue
                observation = np.expand_dims(
                    state["obs"][agent_idx],
                    axis=0,
                ).astype(np.float32)
                raw = actor_forwards[worker_id][policy_id](
                    observation
                ).numpy()[0]
                actions[agent_idx] = scale_action_numpy(
                    raw,
                    action_low,
                    action_high,
                )
        else:
            actions = np.zeros(
                (len(env.agent_ids), action_size),
                dtype=np.float32,
            )
            noise_state = state.get("exploration_noise_state")
            if noise_state is None:
                noise_state = np.zeros_like(actions)
            for agent_idx, agent_id in enumerate(env.agent_ids):
                if state["done_mask"][agent_idx]:
                    actions[agent_idx] = action_low
                    continue
                policy_id = assignment.policy_for_agent(agent_id)
                observation = np.expand_dims(
                    state["obs"][agent_idx],
                    axis=0,
                ).astype(np.float32)
                raw = actor_forwards[worker_id][policy_id](
                    observation
                ).numpy()[0]
                selected = scale_action_numpy(
                    raw,
                    action_low,
                    action_high,
                )
                std = noise_for(policy_id, episode)
                if args.exploration_noise_kind == "ou":
                    noise_state[agent_idx] += (
                        args.ou_theta * (0.0 - noise_state[agent_idx])
                        + rng.normal(
                            0.0,
                            std,
                            size=(action_size,),
                        )
                    )
                    noise = noise_state[agent_idx]
                else:
                    noise = rng.normal(
                        0.0,
                        std,
                        size=(action_size,),
                    )
                actions[agent_idx] = np.clip(
                    selected + noise,
                    action_low,
                    action_high,
                )
            state["exploration_noise_state"] = noise_state
        return smooth_actions(
            actions,
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
        policy_assignment=assignment,
        replay_ready_event=replay_ready_event,
    )
    pool = AsyncCollectorPool(
        envs,
        worker,
        start_episode,
        args.num_episodes,
        queue_capacity=args.async_queue_capacity,
    )
    scheduler = AsyncEventScheduler(args)
    minimum_policy_version = {"value": None}
    health_monitor = getattr(best_tracker, "health_monitor", None)
    recovery_handler = getattr(health_monitor, "recovery_handler", None)
    if recovery_handler is not None:
        recovery_handler.set_policy_publisher(
            lambda: snapshot.publish(live_weights())
        )

        base_post_restore = _multi_policy_deterministic_recovery(
            policy_states
        )

        def post_restore(request):
            outcome = base_post_restore(request)
            dropped = scheduler.reset_after_recovery(pool)
            minimum_policy_version["value"] = snapshot.version
            outcome["async_queue"] = (
                f"dropped {dropped['events']} events/"
                f"{dropped['transitions']} transitions"
            )
            return outcome

        recovery_handler.set_post_restore(post_restore)

    print(
        f"Collector mode: async multi-policy {args.trainer_variant} "
        f"workers={len(envs)} policies={list(assignment.policy_ids)} "
        f"queue={args.async_queue_capacity}",
        flush=True,
    )
    completed = int(start_episode)
    done_workers = 0
    publish_update_count = 0
    metrics = {
        policy_id: {
            "actor": [],
            "critic": [],
            "critic2": [],
        }
        for policy_id in assignment.policy_ids
    }
    last_saved_episode = None
    interrupted = False
    pool.start()
    try:
        while done_workers < len(envs):
            apply_ready_best_checkpoint(best_tracker)
            try:
                event = scheduler.next_event(pool, timeout=0.2)
            except Empty:
                continue
            if isinstance(event, AsyncWorkerErrorEvent):
                raise RuntimeError(
                    f"Async multi-policy {args.trainer_variant} collector "
                    f"{event.worker_id} failed"
                ) from event.error
            if isinstance(event, AsyncWorkerDoneEvent):
                done_workers += 1
                continue
            if isinstance(event, AsyncStepEvent):
                step_events = scheduler.drain_step_events(pool, event)
                minimum = minimum_policy_version["value"]
                if minimum is not None:
                    step_events = [
                        item
                        for item in step_events
                        if item.policy_version is None
                        or int(item.policy_version) >= int(minimum)
                    ]
                if not step_events:
                    continue
                collected = 0
                for step_event in step_events:
                    for transition in step_event.transitions:
                        policy_id, *payload = transition
                        policy_states[policy_id].buffer.add(*payload)
                        collected += 1
                if all(
                    replay_warmup_remaining(state.buffer, args) == 0
                    for state in trainable_states
                ):
                    replay_ready_event.set()
                budget.consume(collected)
                updates_due = scheduler.ingest_by_policy(
                    step_events,
                    lambda transition: transition[0],
                )
                performed = 0
                if not bool(
                    getattr(health_monitor, "verification_pending", False)
                ):
                    for policy_id, due in updates_due.items():
                        state = policy_states[policy_id]
                        if (
                            due <= 0
                            or not state.trainable
                            or len(state.buffer)
                            < max(args.replay_warmup, args.batch_size)
                        ):
                            continue
                        q_filter_start = max(
                            0,
                            int(getattr(
                                args,
                                "demo_q_filter_start_policy_updates",
                                0,
                            )),
                        )
                        for _update in range(due):
                            update_number = (
                                int(
                                    state.critic_optimizer.iterations.numpy()
                                )
                                + 1
                            )
                            policy_updates_since_warmup = policy_update_count(
                                state.policy_updates_since_warmup
                            )
                            q_filter_active = (
                                policy_updates_since_warmup >= q_filter_start
                            )
                            q_weight = (
                                scheduled_q_weight(
                                    policy_updates_since_warmup,
                                    args.demo_q_weight_start,
                                    args.demo_q_weight_end,
                                    args.demo_q_weight_ramp_updates,
                                )
                                if variant_uses_joint_bc(args.trainer_variant)
                                else 1.0
                            )
                            result = state.learner(
                                state.buffer,
                                update_actor=(
                                    state.recovery.should_update_policy()
                                ),
                                learner_update=update_number,
                                bc_weight=(
                                    scheduled_bc_weight(
                                        policy_updates_since_warmup,
                                        args.demo_bc_weight_start,
                                        args.demo_bc_weight_end,
                                        args.demo_bc_decay_updates,
                                    )
                                    if variant_uses_joint_bc(
                                        args.trainer_variant
                                    )
                                    else 0.0
                                ),
                                q_filter_active=q_filter_active,
                                q_weight=q_weight,
                            )
                            if result["policy_updated"]:
                                increment_policy_update_count(
                                    state.policy_updates_since_warmup
                                )
                            state.recovery.record_critic_update()
                            metrics[policy_id]["critic"].append(
                                result["critic_loss"]
                            )
                            if result["critic2_loss"] is not None:
                                metrics[policy_id]["critic2"].append(
                                    result["critic2_loss"]
                                )
                            if result["actor_loss"] is not None:
                                metrics[policy_id]["actor"].append(
                                    result["actor_loss"]
                                )
                            target_due = (
                                result["target_update_due"]
                                if variant_uses_td3(args.trainer_variant)
                                else update_number
                                % args.target_update_every
                                == 0
                            )
                            if target_due:
                                soft_update(
                                    state.target_actor,
                                    state.actor,
                                    args.tau,
                                )
                                soft_update(
                                    state.target_critic,
                                    state.critic,
                                    args.tau,
                                )
                                if state.critic2 is not None:
                                    soft_update(
                                        state.target_critic2,
                                        state.critic2,
                                        args.tau,
                                    )
                            performed += 1
                            publish_update_count += 1
                scheduler.record_updates(performed)
                if (
                    performed > 0
                    and publish_update_count
                    >= args.async_policy_publish_updates
                ):
                    snapshot.publish(live_weights())
                    publish_update_count %= (
                        args.async_policy_publish_updates
                    )
                if budget.exhausted:
                    break
                continue

            if not isinstance(event, AsyncEpisodeEvent):
                continue
            completed += 1
            payload = event.payload
            by_policy = {}
            for policy_id, state in policy_states.items():
                state.noise_std.assign(noise_for(policy_id, completed))
                indices = assignment.indices_for(
                    env0.agent_ids,
                    policy_id,
                )
                rewards = [
                    float(payload["ep_reward"][index])
                    for index in indices
                ]
                values = metrics[policy_id]
                by_policy[policy_id] = {
                    "reward_mean": (
                        float(np.mean(rewards)) if rewards else 0.0
                    ),
                    "noise": float(state.noise_std.numpy()),
                    "replay": len(state.buffer),
                    "policy_updates": len(values["actor"]),
                    "critic_updates": len(values["critic"]),
                    "actor_loss": (
                        float(np.mean(values["actor"]))
                        if values["actor"]
                        else 0.0
                    ),
                    "critic_loss": (
                        float(np.mean(values["critic"]))
                        if values["critic"]
                        else 0.0
                    ),
                }
            diagnostics = summarize_episode_diagnostics([payload], True)
            throughput = scheduler.throughput(pool)
            print_episode_metrics(event.episode, [
                ("mode", [
                    ("collector", "async"),
                    ("algorithm", args.trainer_variant),
                    ("worker", event.worker_id),
                    ("policies", len(policy_states)),
                    ("exploration", exploration_label(payload)),
                ]),
                ("outcome", [
                    (
                        "progress",
                        f"mean:{diagnostics['progress_mean']:.3f} "
                        f"max:{diagnostics['progress_max']:.3f}",
                    ),
                ]),
                ("training", [
                    ("completed", f"{completed}/{args.num_episodes}"),
                    ("total_timesteps", budget.collected),
                    ("by_policy", by_policy),
                    ("policy_version", snapshot.version),
                ]),
                ("throughput", [
                    ("env_steps_s", f"{throughput['env_steps_s']:.1f}"),
                    (
                        "transitions_s",
                        f"{throughput['transitions_s']:.1f}",
                    ),
                    ("updates_s", f"{throughput['updates_s']:.1f}"),
                ]),
            ], args.log_format)
            for values in metrics.values():
                for series in values.values():
                    series.clear()

            if (
                args.checkpoint_every > 0
                and completed % args.checkpoint_every == 0
            ):
                saved_path = _save_multi_policy_deterministic_checkpoint(
                    checkpoint,
                    checkpoint_manager,
                    policy_states,
                    assignment,
                    completed,
                    args,
                )
                last_saved_episode = completed
            else:
                saved_path = None
            if best_tracker.should_evaluate(completed):
                if saved_path is None:
                    saved_path = (
                        _save_multi_policy_deterministic_checkpoint(
                            checkpoint,
                            checkpoint_manager,
                            policy_states,
                            assignment,
                            completed,
                            args,
                        )
                    )
                    last_saved_episode = completed
                request_best_checkpoint_evaluation(
                    best_tracker,
                    saved_path,
                    completed,
                )
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\nInterrupt received: stopping async multi-policy "
            f"{args.trainer_variant} collectors...",
            flush=True,
        )
    finally:
        pool.close()

    if last_saved_episode != completed:
        _save_multi_policy_deterministic_checkpoint(
            checkpoint,
            checkpoint_manager,
            policy_states,
            assignment,
            completed,
            args,
            final=True,
        )
    if not interrupted:
        apply_ready_best_checkpoint(
            best_tracker,
            wait_timeout=args.best_final_drain_timeout,
        )
    for state in policy_states.values():
        state.buffer.wait_for_pending_saves()
    return completed


def run_sync_multi_policy_deterministic(
    args,
    envs,
    stepper,
    best_tracker,
    budget,
):
    assignment = build_policy_assignment(envs, args)
    validate_multi_policy_options(
        args,
        supports_async=True,
        supports_opponent_pool=False,
    )
    if args.policy_path:
        raise ValueError(
            "Independent multi-policy deterministic warm starts currently use "
            "--resume/--resume-checkpoint."
        )
    if len(assignment.trainable_policy_ids) < len(assignment.policy_ids) and not (
        args.resume or args.resume_checkpoint
    ):
        raise ValueError(
            "--train-policy requires --resume or --resume-checkpoint so frozen policies "
            "have trained weights."
        )

    env0 = envs[0]
    obs_dim = env0.obs_dim
    action_size = env0.action_size
    action_low = np.asarray(env0.action_low, dtype=np.float32)
    action_high = np.asarray(env0.action_high, dtype=np.float32)
    random_low, random_high = continuous_exploration_bounds(
        env0.action_space_spec,
        action_low,
        action_high,
    )
    demo_data = (
        load_demonstration_arrays(
            args.demo_path,
            obs_dim=obs_dim,
            action_size=action_size,
            max_transitions=args.demo_max_transitions,
        )
        if args.demo_path
        else None
    )
    demos_by_policy = partition_demonstrations_by_policy(
        demo_data,
        assignment,
    )
    if (
        variant_uses_joint_bc(args.trainer_variant)
        or args.trainer_variant == "ddpgfd"
    ):
        missing_demos = [
            policy_id
            for policy_id in assignment.trainable_policy_ids
            if demos_by_policy.get(policy_id) is None
        ]
        if missing_demos:
            raise ValueError(
                "Every trainable policy needs demonstration transitions for "
                f"{args.trainer_variant}; missing={missing_demos}"
            )
    drive_indices = drive_action_indices(env0.action_names)
    first_agent_for_policy = {
        policy_id: next(
            agent_id
            for agent_id in env0.agent_ids
            if assignment.policy_for_agent(agent_id) == policy_id
        )
        for policy_id in assignment.policy_ids
    }

    policy_states = OrderedDict()
    policy_trackables = {}
    for policy_id in assignment.policy_ids:
        policy_demo = demos_by_policy.get(policy_id)
        actor = build_continuous_actor(
            obs_dim=obs_dim,
            action_size=action_size,
        )
        critic = build_continuous_critic(
            obs_dim=obs_dim,
            action_size=action_size,
        )
        target_actor = build_continuous_actor(
            obs_dim=obs_dim,
            action_size=action_size,
        )
        target_critic = build_continuous_critic(
            obs_dim=obs_dim,
            action_size=action_size,
        )
        critic2 = (
            build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
            if variant_uses_td3(args.trainer_variant)
            else None
        )
        target_critic2 = (
            build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
            if variant_uses_td3(args.trainer_variant)
            else None
        )
        initialize_actor_action_prior(
            actor,
            action_low,
            action_high,
            env0.action_names,
            drive_prior=args.actor_drive_prior,
            steering_prior=args.actor_steering_prior,
        )
        target_actor.set_weights(actor.get_weights())
        target_critic.set_weights(critic.get_weights())
        if critic2 is not None:
            target_critic2.set_weights(critic2.get_weights())

        actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.actor_learning_rate
        )
        critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.critic_learning_rate
        )
        critic2_optimizer = (
            tf.keras.optimizers.Adam(
                learning_rate=args.critic_learning_rate
            )
            if critic2 is not None
            else None
        )
        buffer = ReplayBuffer(
            capacity=args.replay_capacity,
            prioritized=args.trainer_variant == "ddpgfd",
            priority_alpha=args.ddpgfd_priority_alpha,
            priority_beta=args.ddpgfd_priority_beta,
            demo_priority_bonus=args.ddpgfd_demo_priority_bonus,
        )
        learner = build_deterministic_learner_step(
            actor,
            critic,
            target_actor,
            target_critic,
            actor_optimizer,
            critic_optimizer,
            args.gamma,
            action_low,
            action_high,
            variant=args.trainer_variant,
            drive_indices=drive_indices,
            actor_drive_regularization=args.actor_drive_regularization,
            actor_drive_target=args.actor_drive_target,
            critic2=critic2,
            target_critic2=target_critic2,
            critic2_optimizer=critic2_optimizer,
            target_policy_noise=args.td3_target_policy_noise,
            target_noise_clip=args.td3_target_noise_clip,
            policy_delay=args.td3_policy_delay,
            demo_data=policy_demo,
            demo_batch_size=args.demo_bc_batch_size,
            demo_q_filter=args.demo_q_filter,
            td3_bc_alpha=args.td3_bc_alpha,
            ddpgfd_actor_priority_weight=args.ddpgfd_actor_priority_weight,
            batch_size=args.batch_size,
            compiled=args.tf_compile_learner,
            xla=args.tf_xla,
            grad_clip_norm=args.grad_clip_norm,
            grad_clip_adaptive=args.grad_clip_adaptive,
            grad_clip_k=args.grad_clip_k,
        )

        def refill_demo_replay(
            target_buffer=buffer,
            target_demo=policy_demo,
        ):
            if target_demo is None or not args.demo_prefill:
                return 0
            return target_buffer.add_many(
                target_demo["obs"],
                target_demo["actions"],
                target_demo["rewards"],
                target_demo["next_obs"],
                target_demo["dones"],
                is_demo=args.trainer_variant == "ddpgfd",
                protect=args.trainer_variant == "ddpgfd",
            )

        recovery = OffPolicyRecoveryRuntime(
            args,
            buffer,
            replay_refill=refill_demo_replay,
            target_sync=lambda a=actor, c=critic, ta=target_actor, tc=target_critic, c2=critic2, tc2=target_critic2: (
                ta.set_weights(a.get_weights()),
                tc.set_weights(c.get_weights()),
                tc2.set_weights(c2.get_weights())
                if c2 is not None
                else None,
            ),
        )
        noise_std = tf.Variable(
            args.exploration_noise
            if assignment.is_trainable(policy_id)
            else 0.0,
            dtype=tf.float32,
            name=f"noise_{assignment.key_for(policy_id)}",
        )
        policy_updates_since_warmup = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name=(
                "policy_updates_since_warmup_"
                f"{assignment.key_for(policy_id)}"
            ),
        )
        metadata = build_policy_metadata(
            args.trainer_variant,
            env0,
            agent_id=first_agent_for_policy[policy_id],
        )
        metadata.update({
            "policy_id": policy_id,
            "agent_ids": [
                agent_id
                for agent_id in env0.agent_ids
                if assignment.policy_for_agent(agent_id) == policy_id
            ],
            "parameter_sharing": len(
                assignment.indices_for(env0.agent_ids, policy_id)
            ) > 1,
        })
        state = DeterministicPolicyState(
            policy_id=policy_id,
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer=buffer,
            learner=learner,
            recovery=recovery,
            artifact=PolicyArtifactSaver(
                actor,
                Path(args.checkpoint_dir)
                / "policies"
                / assignment.key_for(policy_id),
                metadata,
            ),
            noise_std=noise_std,
            trainable=assignment.is_trainable(policy_id),
            critic2=critic2,
            target_critic2=target_critic2,
            critic2_optimizer=critic2_optimizer,
            policy_updates_since_warmup=policy_updates_since_warmup,
        )
        policy_states[policy_id] = state
        trackable = {
            "actor": actor,
            "critic": critic,
            "target_actor": target_actor,
            "target_critic": target_critic,
            "actor_optimizer": actor_optimizer,
            "critic_optimizer": critic_optimizer,
            "noise_std": noise_std,
            "policy_updates_since_warmup": policy_updates_since_warmup,
        }
        if critic2 is not None:
            trackable.update({
                "critic2": critic2,
                "target_critic2": target_critic2,
                "critic2_optimizer": critic2_optimizer,
            })
        policy_trackables[assignment.key_for(policy_id)] = (
            tf.train.Checkpoint(**trackable)
        )

    checkpoint = tf.train.Checkpoint(
        episode=tf.Variable(0, dtype=tf.int64),
        trainer_variant=tf.Variable(
            args.trainer_variant,
            dtype=tf.string,
            trainable=False,
        ),
        policies=tf.train.Checkpoint(**policy_trackables),
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=args.checkpoint_dir,
        max_to_keep=args.keep_checkpoints,
    )
    resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_manager)
    start_episode = 0
    restored_replays = {
        policy_id: 0 for policy_id in assignment.policy_ids
    }
    if resume_checkpoint:
        manifest_path = Path(args.checkpoint_dir) / "multi_policy.json"
        if manifest_path.is_file():
            stored = load_multi_policy_manifest(manifest_path)
            if stored.get("agent_to_policy", {}) != assignment.agent_to_policy:
                raise RuntimeError(
                    "The checkpoint policy assignment does not match the scenario"
                )
            if str(stored.get("algorithm")) != args.trainer_variant:
                raise RuntimeError(
                    f"Multi-policy checkpoint algorithm={stored.get('algorithm')!r}, "
                    f"expected {args.trainer_variant!r}"
                )
        checkpoint.restore(resume_checkpoint).expect_partial()
        start_episode = int(checkpoint.episode.numpy())
        restored_replays = restore_policy_replays(
            args,
            resume_checkpoint,
            assignment,
            {
                policy_id: state.buffer
                for policy_id, state in policy_states.items()
                if state.trainable
            },
        )
        print(
            f"Resumed multi-policy checkpoint {resume_checkpoint} "
            f"from episode={start_episode}",
            flush=True,
        )

    # Any actor warm start freezes policy updates while fresh critics catch up.
    bc_will_run = bool(args.demo_bc_epochs > 0 and (not resume_checkpoint or args.demo_bc_on_resume))
    warmup_source = (
        "checkpoint resume" if resume_checkpoint
        else "actor warm start" if getattr(args, "policy_path", None)
        else "behavior cloning" if bc_will_run
        else None
    )
    warmup = max(0, int(args.critic_warmup_updates)) if warmup_source is not None else 0
    if warmup > 0:
        print(f"Critic warmup after {warmup_source}: updates={warmup} actor=frozen", flush=True)
    for state in policy_states.values():
        state.actor_optimizer.learning_rate.assign(args.actor_learning_rate)
        state.critic_optimizer.learning_rate.assign(
            args.critic_learning_rate
        )
        if state.critic2_optimizer is not None:
            state.critic2_optimizer.learning_rate.assign(
                args.critic_learning_rate
            )
        state.recovery.critic_warmup_target = warmup if state.trainable else 0

    _bc_demo_validation = None
    if getattr(args, "demo_validation_path", None):
        _bc_demo_validation = load_demonstration_arrays(
            args.demo_validation_path, obs_dim, action_size, max_transitions=0)
        if _bc_demo_validation is not None:
            print(f"Loaded BC validation transitions={len(_bc_demo_validation['actions'])} "
                  f"paths={args.demo_validation_path}", flush=True)
    _bc_ran_policies = []
    _bc_gate_state = None

    for policy_id, state in policy_states.items():
        policy_demo = demos_by_policy.get(policy_id)
        if not state.trainable or policy_demo is None:
            continue
        if args.demo_prefill and restored_replays.get(policy_id, 0) == 0:
            added = state.buffer.add_many(
                policy_demo["obs"],
                policy_demo["actions"],
                policy_demo["rewards"],
                policy_demo["next_obs"],
                policy_demo["dones"],
                is_demo=args.trainer_variant == "ddpgfd",
                protect=args.trainer_variant == "ddpgfd",
            )
            print(
                f"Prefilled demonstration replay policy={policy_id} "
                f"transitions={added}",
                flush=True,
            )
        if (
            args.demo_bc_epochs > 0
            and (not resume_checkpoint or args.demo_bc_on_resume)
        ):
            pretrain_actor_behavior_cloning(
                state.actor,
                policy_demo,
                epochs=args.demo_bc_epochs,
                batch_size=args.demo_bc_batch_size,
                learning_rate=(
                    args.demo_bc_learning_rate
                    or args.actor_learning_rate
                ),
                action_low=action_low,
                action_high=action_high,
                demo_validation=_bc_demo_validation,
            )
            state.target_actor.set_weights(state.actor.get_weights())
            # Persist the BC-pretrained actor BEFORE any RL/warmup update touches it.
            _bc_ran_policies.append(policy_id)
            _bc_actor_path = (args.bc_actor_weights_path
                              or _bc_actor_path_default(args.actor_weights_path))
            if len(_bc_ran_policies) == 1:  # single-policy pilot: one BC actor artifact
                Path(_bc_actor_path).parent.mkdir(parents=True, exist_ok=True)
                state.actor.save_weights(_bc_actor_path)
                print(f"Saved BC actor weights (pre-RL): {_bc_actor_path}", flush=True)
                _bc_gate_state = state
        if (
            args.trainer_variant == "ddpgfd"
            and restored_replays.get(policy_id, 0) == 0
            and args.ddpgfd_pretrain_updates > 0
        ):
            if len(state.buffer) < args.batch_size:
                raise ValueError(
                    f"DDPGfD policy={policy_id!r} needs at least --batch-size "
                    "demonstration transitions for pretraining"
                )
            print(
                f"DDPGfD pretraining policy={policy_id} "
                f"updates={args.ddpgfd_pretrain_updates} "
                f"protected_demos={state.buffer.protected_demo_count}",
                flush=True,
            )
            for update in range(1, args.ddpgfd_pretrain_updates + 1):
                result = state.learner(
                    state.buffer,
                    update_actor=True,
                    learner_update=update,
                    bc_weight=0.0,
                )
                if update % args.target_update_every == 0:
                    soft_update(
                        state.target_actor,
                        state.actor,
                        args.tau,
                    )
                    soft_update(
                        state.target_critic,
                        state.critic,
                        args.tau,
                    )
                if (
                    update % 100 == 0
                    or update == args.ddpgfd_pretrain_updates
                ):
                    print(
                        f"ddpgfd_pretrain policy={policy_id} "
                        f"{update}/{args.ddpgfd_pretrain_updates} "
                        f"actor_loss={result['actor_loss']:.5f} "
                        f"critic_loss={result['critic_loss']:.5f}",
                        flush=True,
                    )

    write_multi_policy_manifest(
        args.checkpoint_dir,
        assignment,
        args.trainer_variant,
        episode=start_episode,
    )
    optimizers = []
    for policy_id, state in policy_states.items():
        if not state.trainable:
            continue
        key = assignment.key_for(policy_id)
        optimizers.extend([
            (f"actor_{key}", state.actor_optimizer),
            (f"critic_{key}", state.critic_optimizer),
            (f"critic2_{key}", state.critic2_optimizer),
        ])
    best_tracker.configure_recovery(
        checkpoint,
        optimizers,
        post_restore=_multi_policy_deterministic_recovery(policy_states),
    )
    print(
        f"Independent multi-policy {args.trainer_variant}: "
        f"assignment={assignment.mode} policies={list(assignment.policy_ids)} "
        f"trainable={list(assignment.trainable_policy_ids)} "
        f"agent_map={assignment.agent_to_policy}",
        flush=True,
    )

    # The BC-only gate runs one frozen evaluation before any RL update.
    if getattr(args, "stop_after_bc", False):
        if not _bc_ran_policies:
            raise SystemExit(
                "--stop-after-bc requires behavior cloning to have actually run: pass "
                "--demo-path and --demo-bc-epochs>0 (and, on resume, --demo-bc-on-resume).")
        _run_stop_after_bc_eval_common(
            args, checkpoint, checkpoint_manager, best_tracker, start_episode,
            getattr(_bc_gate_state, "buffer", None))
        return

    if args.collector_mode == "async":
        return run_async_multi_policy_deterministic(
            args,
            envs,
            policy_states,
            assignment,
            checkpoint,
            checkpoint_manager,
            best_tracker,
            start_episode,
            action_low,
            action_high,
            random_low,
            random_high,
            budget,
        )

    last_completed_episode = start_episode
    last_saved_episode = None
    interrupted = False
    sync_throttles = {
        policy_id: SyncUpdateThrottle(args)
        for policy_id, policy in policy_states.items()
        if policy.trainable
    }
    try:
        for episode in range(start_episode, args.num_episodes):
            trainable_replay_size = min(
                (
                    len(policy.buffer)
                    for policy in policy_states.values()
                    if policy.trainable
                ),
                default=replay_warmup_threshold(args),
            )
            use_random = should_use_random_exploration(
                episode,
                args,
                trainable_replay_size,
            )
            reset_progress_max = curriculum_reset_progress_max(episode, args)
            env_states = []
            for env_idx, env in enumerate(envs):
                config = {
                    "training_episode": episode,
                    "max_steps": args.max_steps_per_episode,
                    "physics_frames_per_step": args.physics_frames_per_step,
                    "training_mode": True,
                    **scenario_curriculum_config(args),
                }
                if reset_progress_max is not None:
                    config.update(
                        reset_progress_min=0.0,
                        reset_progress_max=reset_progress_max,
                    )
                env.configure(**config)
                obs, info = env.reset(
                    seed=args.episode_seed_multiplier * episode + env_idx
                )
                count = len(env.agent_ids)
                env_states.append({
                    "obs": obs,
                    "done": False,
                    "done_mask": np.asarray(
                        info.get(
                            "per_agent_done",
                            np.zeros((count,), dtype=np.bool_),
                        ),
                        dtype=np.bool_,
                    ),
                    "ep_reward": np.zeros((count,), dtype=np.float32),
                    "previous_action": np.full(
                        (count, action_size),
                        np.nan,
                        dtype=np.float32,
                    ),
                    "noise_state": np.zeros(
                        (count, action_size),
                        dtype=np.float32,
                    ),
                    "action_sum": np.zeros(
                        (count, action_size),
                        dtype=np.float32,
                    ),
                    "action_count": np.zeros(
                        (count, 1),
                        dtype=np.float32,
                    ),
                    "action_delta_sum": np.zeros(
                        (count, action_size),
                        dtype=np.float32,
                    ),
                    "action_delta_count": np.zeros(
                        (count, 1),
                        dtype=np.float32,
                    ),
                    "max_track_progress": np.zeros(
                        (count,),
                        dtype=np.float32,
                    ),
                    "last_track_progress": np.zeros(
                        (count,),
                        dtype=np.float32,
                    ),
                    "finish_reached": np.zeros((count,), dtype=np.bool_),
                    "collision_seen": np.zeros((count,), dtype=np.bool_),
                    "collision_count": np.zeros((count,), dtype=np.int32),
                    "stalled_seen": np.zeros((count,), dtype=np.bool_),
                    "stalled_count": np.zeros((count,), dtype=np.int32),
                })

            metrics = {
                policy_id: {"actor": [], "critic": [], "critic2": []}
                for policy_id in assignment.policy_ids
            }
            for step_idx in episode_step_indices(
                args.max_steps_per_episode
            ):
                requests = []
                for env, env_state in zip(envs, env_states):
                    if env_state["done"]:
                        continue
                    actions = np.zeros(
                        (len(env.agent_ids), action_size),
                        dtype=np.float32,
                    )
                    for agent_idx, agent_id in enumerate(env.agent_ids):
                        if env_state["done_mask"][agent_idx]:
                            actions[agent_idx] = action_low
                            continue
                        policy = policy_states[
                            assignment.policy_for_agent(agent_id)
                        ]
                        if use_random and policy.trainable:
                            selected = sample_exploratory_action(
                                action_low,
                                action_high,
                                env.action_names,
                                rng=np.random,
                                drive_min=args.random_drive_min,
                                steering_abs_max=args.random_steering_abs_max,
                                exploration_low=random_low,
                                exploration_high=random_high,
                            )
                        else:
                            selected = select_action(
                                policy.actor,
                                env_state["obs"][agent_idx],
                                action_low,
                                action_high,
                                float(policy.noise_std.numpy())
                                if policy.trainable
                                else 0.0,
                                noise_kind=args.exploration_noise_kind,
                                noise_state=env_state["noise_state"][agent_idx],
                                ou_theta=args.ou_theta,
                            )
                        actions[agent_idx] = smooth_actions(
                            selected,
                            env_state["previous_action"][agent_idx],
                            action_low,
                            action_high,
                            args.action_smoothing,
                        )
                    requests.append((env, env_state, actions))

                for env, env_state, actions, step_result in stepper.step(
                    requests
                ):
                    next_obs, _reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                    transitions_by_policy = {
                        policy_id: 0 for policy_id in sync_throttles
                    }
                    rewards = np.asarray(
                        info.get("per_agent_rewards"),
                        dtype=np.float32,
                    )
                    per_agent_done = np.asarray(
                        info.get("per_agent_done"),
                        dtype=np.bool_,
                    )
                    per_agent_terminated = np.asarray(
                        info.get("per_agent_terminated", per_agent_done),
                        dtype=np.bool_,
                    )
                    agent_infos = list(info.get("per_agent_infos", []))
                    for agent_idx, agent_id in enumerate(env.agent_ids):
                        if (
                            env_state["done_mask"][agent_idx]
                            and per_agent_done[agent_idx]
                        ):
                            continue
                        update_episode_diagnostics(
                            env_state,
                            agent_infos[agent_idx]
                            if agent_idx < len(agent_infos)
                            else {},
                            agent_idx=agent_idx,
                        )
                        policy_id = assignment.policy_for_agent(agent_id)
                        policy = policy_states[policy_id]
                        if policy.trainable:
                            policy.buffer.add(
                                env_state["obs"][agent_idx],
                                actions[agent_idx],
                                float(rewards[agent_idx]),
                                next_obs[agent_idx],
                                bool(
                                    per_agent_terminated[agent_idx]
                                    or terminated
                                ),
                            )
                            budget.consume(1)
                            transitions_by_policy[policy_id] += 1
                        env_state["ep_reward"][agent_idx] += rewards[agent_idx]
                        env_state["action_sum"][agent_idx] += actions[agent_idx]
                        env_state["action_count"][agent_idx, 0] += 1.0
                        previous = env_state["previous_action"][agent_idx]
                        if np.all(np.isfinite(previous)):
                            env_state["action_delta_sum"][
                                agent_idx
                            ] += np.abs(actions[agent_idx] - previous)
                            env_state["action_delta_count"][
                                agent_idx, 0
                            ] += 1.0
                        env_state["previous_action"][agent_idx] = (
                            actions[agent_idx]
                        )
                    env_state["obs"] = next_obs
                    env_state["done_mask"] = per_agent_done
                    env_state["done"] = done or bool(np.all(per_agent_done))

                    if not best_tracker.health_monitor.verification_pending:
                        for policy_id, policy in policy_states.items():
                            if (
                                not policy.trainable
                                or len(policy.buffer)
                                < max(args.replay_warmup, args.batch_size)
                            ):
                                continue
                            transitions_added = transitions_by_policy.get(
                                policy_id,
                                0,
                            )
                            updates_due = sync_throttles[
                                policy_id
                            ].updates_due(
                                transitions_added,
                                env_steps=1 if transitions_added > 0 else 0,
                            )
                            q_filter_start = max(
                                0,
                                int(getattr(
                                    args,
                                    "demo_q_filter_start_policy_updates",
                                    0,
                                )),
                            )
                            for _update in range(updates_due):
                                update_number = int(
                                    policy.critic_optimizer.iterations.numpy()
                                ) + 1
                                policy_updates_since_warmup = policy_update_count(
                                    policy.policy_updates_since_warmup
                                )
                                q_filter_active = (
                                    policy_updates_since_warmup >= q_filter_start
                                )
                                q_weight = (
                                    scheduled_q_weight(
                                        policy_updates_since_warmup,
                                        args.demo_q_weight_start,
                                        args.demo_q_weight_end,
                                        args.demo_q_weight_ramp_updates,
                                    )
                                    if variant_uses_joint_bc(args.trainer_variant)
                                    else 1.0
                                )
                                result = policy.learner(
                                    policy.buffer,
                                    update_actor=policy.recovery.should_update_policy(),
                                    learner_update=update_number,
                                    bc_weight=(
                                        scheduled_bc_weight(
                                            policy_updates_since_warmup,
                                            args.demo_bc_weight_start,
                                            args.demo_bc_weight_end,
                                            args.demo_bc_decay_updates,
                                        )
                                        if variant_uses_joint_bc(
                                            args.trainer_variant
                                        )
                                        else 0.0
                                    ),
                                    q_filter_active=q_filter_active,
                                    q_weight=q_weight,
                                )
                                if result["policy_updated"]:
                                    increment_policy_update_count(
                                        policy.policy_updates_since_warmup
                                    )
                                metrics[policy_id]["critic"].append(
                                    result["critic_loss"]
                                )
                                if result["critic2_loss"] is not None:
                                    metrics[policy_id]["critic2"].append(
                                        result["critic2_loss"]
                                    )
                                policy.recovery.record_critic_update()
                                if result["actor_loss"] is not None:
                                    metrics[policy_id]["actor"].append(
                                        result["actor_loss"]
                                    )
                                target_due = (
                                    result["target_update_due"]
                                    if variant_uses_td3(args.trainer_variant)
                                    else update_number
                                    % args.target_update_every
                                    == 0
                                )
                                if target_due:
                                    soft_update(
                                        policy.target_actor,
                                        policy.actor,
                                        args.tau,
                                    )
                                    soft_update(
                                        policy.target_critic,
                                        policy.critic,
                                        args.tau,
                                    )
                                    if policy.critic2 is not None:
                                        soft_update(
                                            policy.target_critic2,
                                            policy.critic2,
                                            args.tau,
                                        )
                if budget.exhausted:
                    break

            for policy in policy_states.values():
                if policy.trainable:
                    policy.noise_std.assign(
                        max(
                            args.exploration_noise_min,
                            float(policy.noise_std.numpy())
                            * args.exploration_noise_decay,
                        )
                    )
            diagnostics = summarize_episode_diagnostics(env_states, True)
            by_policy = {}
            for policy_id, policy in policy_states.items():
                indices = assignment.indices_for(
                    env0.agent_ids,
                    policy_id,
                )
                rewards = [
                    float(state["ep_reward"][index])
                    for state in env_states
                    for index in indices
                ]
                values = metrics[policy_id]
                by_policy[policy_id] = {
                    "reward_mean": (
                        float(np.mean(rewards)) if rewards else 0.0
                    ),
                    "noise": float(policy.noise_std.numpy()),
                    "replay": len(policy.buffer),
                    "policy_updates": len(values["actor"]),
                    "critic_updates": len(values["critic"]),
                    "actor_loss": (
                        float(np.mean(values["actor"]))
                        if values["actor"]
                        else 0.0
                    ),
                    "critic_loss": (
                        float(np.mean(values["critic"]))
                        if values["critic"]
                        else 0.0
                    ),
                }
            print_episode_metrics(episode, [
                ("mode", [
                    ("algorithm", args.trainer_variant),
                    ("policies", len(policy_states)),
                    ("assignment", assignment.mode),
                    ("exploration", "random" if use_random else "policy"),
                ]),
                ("outcome", [
                    (
                        "progress",
                        f"mean:{diagnostics['progress_mean']:.3f} "
                        f"max:{diagnostics['progress_max']:.3f}",
                    ),
                ]),
                ("training", [
                    ("total_timesteps", budget.collected),
                    ("by_policy", by_policy),
                ]),
            ], args.log_format)
            last_completed_episode = episode + 1

            apply_ready_best_checkpoint(best_tracker)
            if (
                args.checkpoint_every > 0
                and (episode + 1) % args.checkpoint_every == 0
            ):
                saved_path = _save_multi_policy_deterministic_checkpoint(
                    checkpoint,
                    checkpoint_manager,
                    policy_states,
                    assignment,
                    episode + 1,
                    args,
                )
                last_saved_episode = episode + 1
            else:
                saved_path = None
            if best_tracker.should_evaluate(episode + 1):
                if saved_path is None:
                    saved_path = _save_multi_policy_deterministic_checkpoint(
                        checkpoint,
                        checkpoint_manager,
                        policy_states,
                        assignment,
                        episode + 1,
                        args,
                    )
                    last_saved_episode = episode + 1
                request_best_checkpoint_evaluation(
                    best_tracker,
                    saved_path,
                    episode + 1,
                )
            if budget.exhausted:
                break
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\nInterrupt received: saving multi-policy "
            f"{args.trainer_variant} state...",
            flush=True,
        )

    if last_saved_episode != last_completed_episode:
        _save_multi_policy_deterministic_checkpoint(
            checkpoint,
            checkpoint_manager,
            policy_states,
            assignment,
            last_completed_episode,
            args,
            final=True,
        )
    if not interrupted:
        apply_ready_best_checkpoint(
            best_tracker,
            wait_timeout=args.best_final_drain_timeout,
        )
    for state in policy_states.values():
        state.buffer.wait_for_pending_saves()
    return last_completed_episode


def main(trainer_variant):
    args = parse_args(trainer_variant)
    budget = TrainingBudget(args.total_timesteps)
    apply_variant_defaults(args)
    validate_variant_arguments(args)
    validate_async_arguments(args)
    best_tracker = BestCheckpointTracker(args, args.trainer_variant)
    describe_tensorflow_backend(args)
    dashboard = maybe_start_dashboard(args)
    training_start_time = time.monotonic()
    print(f"Deterministic trainer variant: {args.trainer_variant}", flush=True)
    # Architecture is process-wide state, so it must be fixed before the first network is
    # built -- the seeding block below is the last point where nothing exists yet.
    set_network_layers(args.network_layers or default_network_layers(trainer_variant))
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
    critic2 = None
    buffer = None
    checkpoint = None
    checkpoint_manager = None
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
            render_mode=args.render_mode,
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
            raise RuntimeError(
                f"The deterministic trainer requires action_type='continuous', got {env0.action_type!r}"
            )

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

        if args.multi_policy:
            run_sync_multi_policy_deterministic(
                args,
                envs,
                stepper,
                best_tracker,
                budget,
            )
            return

        actor = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
        args.policy_artifact = PolicyArtifactSaver(
            actor,
            args.checkpoint_dir,
            build_policy_metadata(args.trainer_variant, env0),
        )
        critic = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        target_actor = build_continuous_actor(obs_dim=obs_dim, action_size=action_size)
        target_critic = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        if variant_uses_td3(args.trainer_variant):
            critic2 = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
            target_critic2 = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        else:
            critic2 = None
            target_critic2 = None
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
        if critic2 is not None:
            target_critic2.set_weights(critic2.get_weights())

        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_learning_rate)
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate)
        critic2_optimizer = (
            tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate)
            if critic2 is not None
            else None
        )
        buffer = ReplayBuffer(
            capacity=args.replay_capacity,
            prioritized=args.trainer_variant == "ddpgfd",
            priority_alpha=args.ddpgfd_priority_alpha,
            priority_beta=args.ddpgfd_priority_beta,
            demo_priority_bonus=args.ddpgfd_demo_priority_bonus,
        )
        noise_std = args.exploration_noise
        start_episode = 0

        checkpoint_items = dict(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            episode=tf.Variable(0, dtype=tf.int64),
            noise_std=tf.Variable(args.exploration_noise, dtype=tf.float32),
            policy_updates_since_warmup=tf.Variable(
                0,
                dtype=tf.int64,
                trainable=False,
            ),
            trainer_variant=tf.Variable(args.trainer_variant, dtype=tf.string, trainable=False),
        )
        if critic2 is not None:
            checkpoint_items.update(
                critic2=critic2,
                target_critic2=target_critic2,
                critic2_optimizer=critic2_optimizer,
            )
        checkpoint = tf.train.Checkpoint(**checkpoint_items)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.checkpoint_dir,
            max_to_keep=args.keep_checkpoints,
        )
        resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_manager)
        restored_replay_count = 0
        if resume_checkpoint and args.policy_path:
            raise RuntimeError("--policy-path cannot be combined with --resume or --resume-checkpoint")
        if resume_checkpoint:
            checkpoint_names = {name for name, _shape in tf.train.list_variables(resume_checkpoint)}
            if variant_uses_td3(args.trainer_variant) and not any(
                name.startswith(("critic2/", "target_critic2/", "critic2_optimizer/"))
                for name in checkpoint_names
            ):
                raise ValueError(
                    f"Checkpoint {resume_checkpoint} has no second critic and cannot resume "
                    f"{args.trainer_variant}. Start a new TD3 run or use a TD3 checkpoint."
                )
            checkpoint.restore(resume_checkpoint).expect_partial()
            restored_variant = checkpoint.trainer_variant.numpy().decode("utf-8")
            if restored_variant != args.trainer_variant:
                raise ValueError(
                    f"Checkpoint algorithm is {restored_variant!r}, but the requested trainer is "
                    f"{args.trainer_variant!r}. Use a matching checkpoint directory."
                )
            start_episode = int(checkpoint.episode.numpy())
            noise_std = float(checkpoint.noise_std.numpy())
            print(
                f"Resumed checkpoint {resume_checkpoint} from episode={start_episode} "
                f"noise_std={noise_std:.3f}",
                flush=True,
            )
            restored_replay_count = restore_replay_buffer(args, resume_checkpoint, buffer)
        elif args.policy_path:
            loaded_policy = load_policy_into_model(
                actor,
                args.policy_path,
                expected_algorithm=args.trainer_variant,
            )
            args._replay_warmup_uses_restored_policy = True
            target_actor.set_weights(actor.get_weights())
            print(
                f"Warm-started {args.trainer_variant} actor from {loaded_policy['source_kind']}: "
                f"{loaded_policy['path']} (fresh critics, optimizers and replay, episode=0)",
                flush=True,
            )

        actor_optimizer.learning_rate.assign(args.actor_learning_rate)
        critic_optimizer.learning_rate.assign(args.critic_learning_rate)
        if critic2_optimizer is not None:
            critic2_optimizer.learning_rate.assign(args.critic_learning_rate)
        best_tracker.configure_recovery(
            checkpoint,
            [
                ("actor", actor_optimizer),
                ("critic", critic_optimizer),
                ("critic2", critic2_optimizer),
            ],
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

        if demo_data is not None and args.demo_prefill and restored_replay_count == 0:
            added = buffer.add_many(
                demo_data["obs"],
                demo_data["actions"],
                demo_data["rewards"],
                demo_data["next_obs"],
                demo_data["dones"],
                is_demo=args.trainer_variant == "ddpgfd",
                protect=args.trainer_variant == "ddpgfd",
            )
            print(f"Prefilled replay buffer with demonstration transitions={added}", flush=True)

        # Replay-only demonstrations never enter the BC sampler.
        if getattr(args, "demo_replay_only_path", None) and args.demo_prefill and restored_replay_count == 0:
            replay_only = load_demonstration_arrays(
                args.demo_replay_only_path, obs_dim, action_size, max_transitions=0)
            if replay_only is not None:
                added_ro = buffer.add_many(
                    replay_only["obs"], replay_only["actions"], replay_only["rewards"],
                    replay_only["next_obs"], replay_only["dones"],
                    is_demo=args.trainer_variant == "ddpgfd",
                    protect=args.trainer_variant == "ddpgfd")
                print(f"Prefilled REPLAY-ONLY transitions (not in BC)={added_ro} "
                      f"paths={args.demo_replay_only_path}", flush=True)

        if demo_data is not None and args.demo_prefill and restored_replay_count > 0:
            print("Skipped demonstration prefill because the checkpoint replay was restored.", flush=True)

        # BC training set = --demo-path + --demo-bc-path (imitation-only pairs); replay-only excluded.
        _bc_pairs = _load_bc_pairs(args.demo_bc_path, obs_dim, action_size) if getattr(
            args, "demo_bc_path", None) else None
        if _bc_pairs is not None:
            print(f"Loaded imitation-only BC pairs (BC loss ONLY, not replayed) "
                  f"transitions={len(_bc_pairs['actions'])} paths={args.demo_bc_path}", flush=True)
        _bc_train = _combine_bc_sources(demo_data, _bc_pairs)
        _bc_ran = False
        _bc_demo_validation = None
        if getattr(args, "demo_validation_path", None):
            _bc_demo_validation = load_demonstration_arrays(
                args.demo_validation_path, obs_dim, action_size, max_transitions=0)
            if _bc_demo_validation is not None:
                print(f"Loaded BC validation transitions={len(_bc_demo_validation['actions'])} "
                      f"paths={args.demo_validation_path}", flush=True)
        if _bc_train is not None and args.demo_bc_epochs > 0 and (not resume_checkpoint or args.demo_bc_on_resume):
            pretrain_actor_behavior_cloning(
                actor,
                _bc_train,
                epochs=args.demo_bc_epochs,
                batch_size=args.demo_bc_batch_size,
                learning_rate=args.demo_bc_learning_rate or args.actor_learning_rate,
                action_low=action_low,
                action_high=action_high,
                demo_validation=_bc_demo_validation,
            )
            target_actor.set_weights(actor.get_weights())
            _bc_ran = True
            _bc_actor_path = (args.bc_actor_weights_path
                              or _bc_actor_path_default(args.actor_weights_path))
            Path(_bc_actor_path).parent.mkdir(parents=True, exist_ok=True)
            actor.save_weights(_bc_actor_path)
            print(f"Saved BC actor weights (pre-RL): {_bc_actor_path}", flush=True)

        if (
            args.trainer_variant == "ddpgfd"
            and restored_replay_count == 0
            and args.ddpgfd_pretrain_updates > 0
        ):
            if len(buffer) < args.batch_size:
                raise ValueError(
                    "DDPGfD needs at least --batch-size demonstration transitions for pretraining"
                )
            print(
                f"DDPGfD critic/actor pretraining updates={args.ddpgfd_pretrain_updates} "
                f"protected_demos={buffer.protected_demo_count}",
                flush=True,
            )
            for pretrain_update in range(1, args.ddpgfd_pretrain_updates + 1):
                result = train_deterministic_step(
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
                    variant="ddpgfd",
                    learner_update=pretrain_update,
                    ddpgfd_actor_priority_weight=args.ddpgfd_actor_priority_weight,
                    grad_clip_norm=args.grad_clip_norm,
                    grad_clip_adaptive=args.grad_clip_adaptive,
                    grad_clip_k=args.grad_clip_k,
                )
                if pretrain_update % args.target_update_every == 0:
                    soft_update(target_actor, actor, args.tau)
                    soft_update(target_critic, critic, args.tau)
                if pretrain_update % 100 == 0 or pretrain_update == args.ddpgfd_pretrain_updates:
                    print(
                        f"ddpgfd_pretrain={pretrain_update}/{args.ddpgfd_pretrain_updates} "
                        f"actor_loss={result['actor_loss']:.5f} critic_loss={result['critic_loss']:.5f}",
                        flush=True,
                    )

        critic_warmup_source = (
            "checkpoint resume"
            if resume_checkpoint
            else "actor warm start"
            if args.policy_path
            else "behavior cloning"
            if _bc_ran
            else None
        )
        critic_warmup_target = (
            max(0, int(args.critic_warmup_updates))
            if critic_warmup_source is not None
            else 0
        )

        # Deterministic BC gate: after BC (+ saved BC actor), ONE frozen evaluation, then STOP
        # before any RL update. Fails closed if BC did not actually run.
        if getattr(args, "stop_after_bc", False):
            if not _bc_ran:
                raise SystemExit(
                    "--stop-after-bc requires behavior cloning to have actually run: pass "
                    "--demo-path and --demo-bc-epochs>0 (and, on resume, --demo-bc-on-resume).")
            _run_stop_after_bc_eval_common(
                args, checkpoint, checkpoint_manager, best_tracker, start_episode, buffer)
            return

        if critic_warmup_target > 0:
            print(
                f"Critic warmup after {critic_warmup_source}: "
                f"updates={critic_warmup_target} actor=frozen",
                flush=True,
            )

        opponent_teams = validate_team_layout(envs, args.opponent_pool)
        opponent_pool = OpponentPool(
            args,
            algorithm=args.trainer_variant,
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
                best_tracker,
                start_episode,
                noise_std,
                action_low,
                action_high,
                random_action_low,
                random_action_high,
                actor_drive_indices,
                critic_warmup_target,
                checkpoint.policy_updates_since_warmup,
                budget,
                critic2=critic2,
                target_critic2=target_critic2,
                critic2_optimizer=critic2_optimizer,
                demo_data=demo_data,
                demo_validation=_bc_demo_validation,
            )
            actor.save_weights(args.actor_weights_path)
            critic.save_weights(args.critic_weights_path)
            print(f"Saved actor weights: {args.actor_weights_path}", flush=True)
            print(f"Saved critic weights: {args.critic_weights_path}", flush=True)
            if critic2 is not None:
                critic2.save_weights(args.critic2_weights_path)
                print(f"Saved critic2 weights: {args.critic2_weights_path}", flush=True)
            return

        det_learner = build_deterministic_learner_step(
            actor,
            critic,
            target_actor,
            target_critic,
            actor_optimizer,
            critic_optimizer,
            args.gamma,
            action_low,
            action_high,
            variant=args.trainer_variant,
            drive_indices=actor_drive_indices,
            actor_drive_regularization=args.actor_drive_regularization,
            actor_drive_target=args.actor_drive_target,
            critic2=critic2,
            target_critic2=target_critic2,
            critic2_optimizer=critic2_optimizer,
            target_policy_noise=args.td3_target_policy_noise,
            target_noise_clip=args.td3_target_noise_clip,
            policy_delay=args.td3_policy_delay,
            demo_data=demo_data,
            demo_batch_size=args.demo_bc_batch_size,
            demo_q_filter=args.demo_q_filter,
            td3_bc_alpha=args.td3_bc_alpha,
            ddpgfd_actor_priority_weight=args.ddpgfd_actor_priority_weight,
            batch_size=args.batch_size,
            compiled=args.tf_compile_learner,
            xla=args.tf_xla,
            grad_clip_norm=args.grad_clip_norm,
            grad_clip_adaptive=args.grad_clip_adaptive,
            grad_clip_k=args.grad_clip_k,
        )

        def refill_recovery_replay():
            if demo_data is None or not args.demo_prefill:
                return 0
            return buffer.add_many(
                demo_data["obs"],
                demo_data["actions"],
                demo_data["rewards"],
                demo_data["next_obs"],
                demo_data["dones"],
                is_demo=args.trainer_variant == "ddpgfd",
                protect=args.trainer_variant == "ddpgfd",
            )

        def sync_recovery_targets():
            target_actor.set_weights(actor.get_weights())
            target_critic.set_weights(critic.get_weights())
            if critic2 is not None:
                target_critic2.set_weights(critic2.get_weights())

        recovery_runtime = OffPolicyRecoveryRuntime(
            args,
            buffer,
            initial_critic_warmup=critic_warmup_target,
            replay_refill=refill_recovery_replay,
            target_sync=sync_recovery_targets,
        )
        recovery_handler = best_tracker.health_monitor.recovery_handler
        if recovery_handler is not None:
            recovery_handler.set_post_restore(recovery_runtime.post_restore)

        sync_throttle = SyncUpdateThrottle(args)
        # Enable Q-filtering after this many post-warmup actor updates.
        q_filter_start = max(0, int(getattr(args, "demo_q_filter_start_policy_updates", 0)))
        policy_update_counter = checkpoint.policy_updates_since_warmup
        last_bc_weight = 0.0
        qf_last = {"active": False, "fraction": None, "selected": 0, "total": 0,
                   "expert_q": None, "policy_q": None}
        bc_reference_actor = None
        if getattr(args, "gradient_telemetry_every", 0) > 0 and variant_uses_joint_bc(args.trainer_variant):
            bc_reference_actor = tf.keras.models.clone_model(actor)
            bc_reference_actor.set_weights(actor.get_weights())
        last_saved_episode = None
        for episode in range(start_episode, args.num_episodes):
            opponent_match = opponent_pool.start_episode(actor, episode)
            use_random_exploration = should_use_random_exploration(
                episode,
                args,
                len(buffer),
            )
            reset_progress_max = curriculum_reset_progress_max(episode, args)
            env_states = []
            for env_idx, env in enumerate(envs):
                scenario_config = {
                    "training_episode": episode,
                    "max_steps": args.max_steps_per_episode,
                    "physics_frames_per_step": args.physics_frames_per_step,
                    "training_mode": True,
                    **scenario_curriculum_config(args),
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
            critic2_losses = []
            bc_losses = []
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
                    transitions_added = 0

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
                                budget.consume(1)
                                transitions_added += 1
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
                        budget.consume(1)
                        transitions_added += 1
                        state["ep_reward"] += float(reward)
                        state["action_sum"] += action
                        state["action_count"] += 1.0
                        if np.all(np.isfinite(state["previous_action"])):
                            state["action_delta_sum"] += np.abs(action - state["previous_action"])
                            state["action_delta_count"] += 1.0
                        state["previous_action"] = action

                    state["obs"] = next_obs
                    state["done"] = done

                    if (
                        len(buffer) >= max(args.replay_warmup, args.batch_size)
                        and not best_tracker.health_monitor.verification_pending
                    ):
                        updates_due = sync_throttle.updates_due(
                            transitions_added,
                            env_steps=1,
                        )
                        for _update in range(updates_due):
                            update_actor = recovery_runtime.should_update_policy()
                            learner_update = int(critic_optimizer.iterations.numpy()) + 1
                            # BC weight decays over POLICY updates since the warmup ended -- NOT over
                            # critic updates -- so it stays at the start weight during the warmup.
                            policy_updates_since_warmup = policy_update_count(
                                policy_update_counter
                            )
                            bc_weight = scheduled_bc_weight(
                                policy_updates_since_warmup,
                                args.demo_bc_weight_start,
                                args.demo_bc_weight_end,
                                args.demo_bc_decay_updates,
                            )
                            # Delay the Q-filter until enough real policy updates have run.
                            q_filter_active = policy_updates_since_warmup >= q_filter_start
                            q_weight = (
                                scheduled_q_weight(
                                    policy_updates_since_warmup,
                                    args.demo_q_weight_start,
                                    args.demo_q_weight_end,
                                    args.demo_q_weight_ramp_updates,
                                )
                                if variant_uses_joint_bc(args.trainer_variant)
                                else 1.0
                            )
                            grad_tele = (
                                args.gradient_telemetry_every > 0
                                and policy_updates_since_warmup % args.gradient_telemetry_every == 0
                            )
                            last_bc_weight = bc_weight
                            result = det_learner(
                                buffer,
                                update_actor=update_actor,
                                learner_update=learner_update,
                                bc_weight=bc_weight,
                                q_filter_active=q_filter_active,
                                q_weight=q_weight,
                                gradient_telemetry=grad_tele,
                                bc_reference_actor=bc_reference_actor,
                            )
                            if result["policy_updated"]:
                                increment_policy_update_count(
                                    policy_update_counter
                                )
                            if result.get("q_filter_active"):
                                qf_last = {
                                    "active": True,
                                    "fraction": result["q_filter_selected_fraction"],
                                    "selected": result["q_filter_selected_count"],
                                    "total": result["q_filter_total_count"],
                                    "expert_q": result["expert_q_mean"],
                                    "policy_q": result["policy_q_mean"],
                                }
                            elif result["policy_updated"]:
                                qf_last["active"] = False
                            critic_losses.append(result["critic_loss"])
                            if result["critic2_loss"] is not None:
                                critic2_losses.append(result["critic2_loss"])
                            if result["bc_loss"] is not None:
                                bc_losses.append(result["bc_loss"])
                            warmup_completed = recovery_runtime.record_critic_update()
                            if warmup_completed and getattr(args, "stop_after_critic_warmup", False):
                                _run_critic_warmup_audit(
                                    args, checkpoint, checkpoint_manager, buffer, episode,
                                    actor, critic, critic2, _bc_demo_validation,
                                    action_low, action_high,
                                )
                            if result["actor_loss"] is not None:
                                actor_losses.append(result["actor_loss"])
                            elif warmup_completed:
                                print(
                                    f"Critic warmup complete after updates={recovery_runtime.critic_updates}; "
                                    "actor will be unfrozen on the next update.",
                                    flush=True,
                                )

                            target_due = (
                                result["target_update_due"]
                                if variant_uses_td3(args.trainer_variant)
                                else learner_update % args.target_update_every == 0
                            )
                            if target_due:
                                soft_update(target_actor, actor, args.tau)
                                soft_update(target_critic, critic, args.tau)
                                if critic2 is not None:
                                    soft_update(target_critic2, critic2, args.tau)
                if budget.exhausted:
                    break

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
            replay_warmup_left = replay_warmup_remaining(buffer, args)
            critic_warmup_left = recovery_runtime.warmup_left
            print_episode_metrics(episode, [
                ("mode", [
                    ("algorithm", args.trainer_variant),
                    (
                        "exploration",
                        "warmup_random"
                        if replay_warmup_left > 0
                        else "scheduled_random"
                        if use_random_exploration
                        else "policy",
                    ),
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
                    ("collision", format_collision_diagnostics(
                        diagnostics, controlled_agents)),
                    ("stall", f"{diagnostics['stalls']}/{controlled_agents} ({stall_rate:.2%})"),
                ]),
                ("actions", [
                    ("mean", format_float_list(mean_action_summary)),
                    ("delta", format_float_list(mean_action_delta_summary)),
                ]),
                ("training", [
                    ("total_timesteps", budget.collected),
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    *(([("protected_demos", buffer.protected_demo_count)]) if args.trainer_variant == "ddpgfd" else []),
                    ("critic_updates", len(critic_losses)),
                    ("policy_updates", len(actor_losses)),
                    ("replay_warmup_left", replay_warmup_left),
                    ("critic_warmup_left", critic_warmup_left),
                    ("actor_loss", f"{float(np.mean(actor_losses)) if actor_losses else 0.0:.5f}"),
                    ("critic_loss", f"{float(np.mean(critic_losses)) if critic_losses else 0.0:.5f}"),
                    *(([("critic2_loss", f"{float(np.mean(critic2_losses)):.5f}")]) if critic2_losses else []),
                    *(([("bc_loss", f"{float(np.mean(bc_losses)):.5f}")]) if bc_losses else []),
                    *(([
                        ("bc_weight", f"{last_bc_weight:.4f}"),
                        (
                            "policy_updates_since_warmup",
                            policy_update_count(policy_update_counter),
                        ),
                        ("q_filter_active", qf_last["active"]),
                        ("q_filter_selected_fraction", "n/a" if qf_last["fraction"] is None else f"{qf_last['fraction']:.3f}"),
                    ]) if variant_uses_joint_bc(args.trainer_variant) else []),
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

            apply_ready_best_checkpoint(best_tracker)
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
            if budget.exhausted:
                print(
                    f"Transition budget reached: {budget.collected}/{budget.limit}",
                    flush=True,
                )
                break

        completed_episode = last_completed_episode if last_completed_episode is not None else start_episode
        if last_saved_episode != completed_episode:
            save_training_checkpoint(
                checkpoint,
                checkpoint_manager,
                buffer,
                completed_episode,
                noise_std,
                args,
                final=True,
            )
        # The evaluation requested on the last episode is still running; without this the
        # finally-block's close() cancels it and a final best can never be promoted.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
        actor.save_weights(args.actor_weights_path)
        critic.save_weights(args.critic_weights_path)
        print(f"Saved actor weights: {args.actor_weights_path}", flush=True)
        print(f"Saved critic weights: {args.critic_weights_path}", flush=True)
        if critic2 is not None:
            critic2.save_weights(args.critic2_weights_path)
            print(f"Saved critic2 weights: {args.critic2_weights_path}", flush=True)
    except KeyboardInterrupt:
        print(
            f"\nInterrupt received: saving the last consistent {args.trainer_variant} state...",
            flush=True,
        )
        if checkpoint is not None and checkpoint_manager is not None and buffer is not None and actor is not None:
            interrupted_episode = last_completed_episode if last_completed_episode is not None else start_episode
            save_training_checkpoint(
                checkpoint, checkpoint_manager, buffer, interrupted_episode, noise_std, args
            )
            actor.save_weights(args.actor_weights_path)
            if critic is not None:
                critic.save_weights(args.critic_weights_path)
            if critic2 is not None:
                critic2.save_weights(args.critic2_weights_path)
            print(f"Interrupted training saved at episode={interrupted_episode}", flush=True)
        else:
            print("Training state was not initialized; no checkpoint was written.", flush=True)
    finally:
        report_training_time(dashboard, training_start_time)
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
    raise SystemExit("Use python/train.py and select a deterministic algorithm with --algorithm.")
