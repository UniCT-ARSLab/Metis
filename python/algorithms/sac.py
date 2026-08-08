import argparse
import itertools
import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

import numpy as np

from algorithms.common import (
    continuous_exploration_bounds,
    build_replay_ready_event,
    create_continuous_async_worker,
    curriculum_reset_progress_max,
    describe_tensorflow_backend,
    exploration_label,
    format_collision_diagnostics,
    format_float_list,
    load_demonstration_arrays,
    sample_exploratory_action,
    replay_warmup_remaining,
    replay_warmup_threshold,
    scenario_curriculum_config,
    should_use_random_exploration,
    smooth_actions,
    soft_update,
    summarize_action_deltas,
    summarize_actions,
    summarize_episode_diagnostics,
    summarize_rewards,
    update_episode_diagnostics,
)
import tensorflow as tf

from core.models import (
    build_continuous_critic,
    build_sac_actor,
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
    build_lockstep_user_args,
    episode_step_indices,
    print_episode_metrics,
    resolve_resume_checkpoint,
    restore_replay_buffer,
    save_replay_snapshot,
    validate_async_arguments,
    argument_group,
    GROUP_CHECKPOINTS,
    GROUP_DEMOS,
    GROUP_EXPLORATION,
    GROUP_GODOT,
    GROUP_LOGGING,
    GROUP_LOOP,
    GROUP_PROGRESS_CURRICULUM,
    GROUP_REPLAY,
    GROUP_SAC,
)
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


LOG_2PI = np.log(2.0 * np.pi).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic SAC trainer for continuous Godot scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    group.add_argument("--actor-learning-rate", type=float, default=3e-4)
    group.add_argument("--critic-learning-rate", type=float, default=3e-4)
    group.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="Hard global gradient-norm cap for critic/actor updates (0 disables). Safety net "
        "against the deadly-triad Q-value divergence that otherwise blows critics up to NaN.",
    )
    group.add_argument(
        "--grad-clip-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip each network at --grad-clip-k * EMA(gradient-norm), bounded by "
        "--grad-clip-norm. Tracks each network's gradient scale to reduce manual tuning; "
        "the hard cap remains the final safety bound.",
    )
    group.add_argument(
        "--grad-clip-k",
        type=float,
        default=3.0,
        help="Multiplier on the running-mean gradient norm when --grad-clip-adaptive is set.",
    )

    group = argument_group(parser, GROUP_SAC)
    group.add_argument("--alpha-learning-rate", type=float, default=3e-4)

    group = argument_group(parser, GROUP_CHECKPOINTS)
    group.add_argument(
        "--resume-actor-learning-rate",
        type=float,
        default=1e-5,
        help="Actor learning rate used after restoring a checkpoint; keeps a valid policy from drifting quickly.",
    )
    group.add_argument(
        "--resume-alpha-learning-rate",
        type=float,
        default=1e-5,
        help="Entropy-temperature learning rate used after restoring a checkpoint.",
    )

    group = argument_group(parser, GROUP_SAC)
    group.add_argument("--initial-alpha", type=float, default=0.2)
    group.add_argument(
        "--tune-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-tune the SAC entropy coefficient alpha toward --target-entropy. The tuner "
            "is unstable on some tasks (alpha runs away up or collapses); use --no-tune-alpha "
            "with --initial-alpha to hold it fixed."
        ),
    )
    group.add_argument("--target-entropy", type=float, default=None)
    group.add_argument(
        "--min-alpha",
        type=float,
        default=0.0,
        help="Floor for the auto-tuned entropy coefficient alpha (0 = no floor). Prevents the "
        "late-training entropy collapse where alpha decays too low and the critic drifts up.",
    )
    group.add_argument(
        "--caps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CAPS action-smoothness regularization on the actor: penalize different actions on "
        "temporally-adjacent and nearby states. Trains a jitter-free controller that holds steady "
        "at the target. Recommended for velocity-controlled continuous tasks.",
    )
    group.add_argument("--caps-lambda-temporal", type=float, default=1.0)
    group.add_argument("--caps-lambda-spatial", type=float, default=1.0)
    group.add_argument(
        "--caps-sigma",
        type=float,
        default=0.05,
        help="Std of the Gaussian perturbation for the CAPS spatial-smoothness term (obs are ~[-1,1]).",
    )
    group.add_argument(
        "--actor-anchor-coef",
        type=float,
        default=0.0,
        help=(
            "Trust-region penalty that keeps a resumed or warm-started actor close to the "
            "policy loaded at startup. Useful for conservative fine-tuning of an already "
            "competent policy; 0 disables it."
        ),
    )
    group.add_argument(
        "--actor-anchor-log-std-coef",
        type=float,
        default=0.1,
        help=(
            "Relative weight of the log-standard-deviation term inside the actor anchor. "
            "The deterministic action mean always has weight 1."
        ),
    )
    group.add_argument("--log-std-min", type=float, default=-20.0)
    group.add_argument("--log-std-max", type=float, default=2.0)

    group = argument_group(parser, GROUP_EXPLORATION)
    group.add_argument("--action-smoothing", type=float, default=0.0)
    group.add_argument("--random-exploration-episodes", type=int, default=15)
    group.add_argument("--random-drive-min", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--random-steering-abs-max", type=float, default=None, help=argparse.SUPPRESS)

    group = argument_group(parser, GROUP_PROGRESS_CURRICULUM)
    group.add_argument("--reset-progress-curriculum", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--reset-progress-start-max", type=float, default=0.025)
    group.add_argument("--reset-progress-end-max", type=float, default=0.35)
    group.add_argument("--reset-progress-ramp-episodes", type=int, default=400)

    group = argument_group(parser, GROUP_REPLAY)
    group.add_argument("--replay-warmup", type=int, default=10000)
    group.add_argument("--replay-capacity", type=int, default=200000)

    group = argument_group(parser, GROUP_LOOP)
    group.add_argument(
        "--critic-warmup-updates",
        type=int,
        default=2000,
        help=(
            "After checkpoint resume or an actor-only --policy-path warm start, update "
            "only critics and target critics for this many gradient steps before "
            "unfreezing actor and alpha. Use 0 to disable."
        ),
    )
    group.add_argument("--target-update-every", type=int, default=1)
    group.add_argument(
        "--policy-update-every",
        type=int,
        default=2,
        help="Update actor and entropy temperature once every N critic updates.",
    )
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
    group.add_argument("--actor-weights-path", default="generic_sac_actor.weights.h5")
    group.add_argument(
        "--policy-path",
        default=None,
        help="Warm-start the actor from a .keras model, full .h5 model, or .weights.h5 file.",
    )
    group.add_argument("--critic-weights-path", default=None)
    group.add_argument("--critic1-weights-path", default="generic_sac_critic1.weights.h5")
    group.add_argument("--critic2-weights-path", default="generic_sac_critic2.weights.h5")
    group.add_argument("--checkpoint-dir", default="checkpoints/generic_sac")
    group.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Specific TensorFlow checkpoint to restore (for example ckpt-1025 or "
            "checkpoints/run/ckpt-1025). When omitted, --resume restores the latest "
            "checkpoint in --checkpoint-dir."
        ),
    )
    group.add_argument("--checkpoint-every", type=int, default=25)
    group.add_argument("--keep-checkpoints", type=int, default=5)
    group.add_argument("--best-checkpoint-window", type=int, default=None, help=argparse.SUPPRESS)
    group.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    group = argument_group(parser, GROUP_REPLAY)
    group.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument(
        "--require-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of rebuilding an empty replay buffer when resuming an old checkpoint.",
    )

    group = argument_group(parser, GROUP_DEMOS)
    group.add_argument("--demo-path", action="append", default=[])
    group.add_argument("--demo-prefill", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--demo-max-transitions", type=int, default=0)
    group.add_argument("--demo-bc-epochs", type=int, default=0)
    group.add_argument("--demo-bc-batch-size", type=int, default=128)
    group.add_argument("--demo-bc-learning-rate", type=float, default=None)
    group.add_argument(
        "--demo-bc-on-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run behavior cloning again after restoring a checkpoint.",
    )
    group.add_argument(
        "--demo-validation-path",
        action="append",
        default=[],
        help="Held-out demo npz(s): BC measures action MSE on it each epoch and keeps the "
        "best-epoch weights (early best-checkpoint against overfitting the training demos).",
    )
    group.add_argument(
        "--demo-bc-path",
        action="append",
        default=[],
        help="Imitation-only npz(s) of (obs, actions) pairs: added to the BC imitation loss ONLY, "
        "never prefilled into the replay buffer. Use for DAgger labels (learner-visited states + "
        "expert action) whose next_obs was produced by the LEARNER, so they are not valid RL "
        "transitions. --demo-path stays the source of complete transitions (BC + replay).",
    )
    group.add_argument(
        "--bc-actor-weights-path",
        type=str,
        default=None,
        help="Persist the BC-pretrained actor here BEFORE any SAC/critic-warmup update "
        "(default: the actor weights path with a _bc suffix).",
    )
    group.add_argument(
        "--stop-after-bc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run BC (+ save the BC actor) then ONE frozen Stage-A evaluation, and STOP before "
        "the SAC training loop. A gate to inspect the cloned policy without committing to SAC.",
    )

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
    add_opponent_pool_arguments(parser)
    add_best_checkpoint_arguments(parser)
    add_training_health_arguments(parser)
    add_parallel_env_arguments(parser)
    add_lockstep_tuning_arguments(parser)
    add_log_format_argument(parser)
    add_dashboard_arguments(parser)
    add_tensorflow_runtime_arguments(parser, include_compile_learner=True)
    add_godot_render_argument(parser)

    # Accepted for command compatibility with DDPG runs; SAC exploration is entropy-based.

    group = argument_group(parser, GROUP_EXPLORATION)
    group.add_argument("--exploration-noise", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--exploration-noise-min", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--exploration-noise-decay", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--exploration-noise-kind", default=None, help=argparse.SUPPRESS)
    group.add_argument("--ou-theta", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--actor-drive-prior", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--actor-steering-prior", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--actor-drive-regularization", type=float, default=None, help=argparse.SUPPRESS)
    group.add_argument("--actor-drive-target", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def scale_action_tensor(raw_action, action_low, action_high):
    return action_low + 0.5 * (raw_action + 1.0) * (action_high - action_low)


def scale_action_numpy(raw_action, action_low, action_high):
    return action_low + 0.5 * (raw_action + 1.0) * (action_high - action_low)


def sample_actor(actor, obs, action_low, action_high, log_std_min, log_std_max, deterministic=False):
    mean, log_std = actor(obs, training=True)
    log_std = tf.clip_by_value(log_std, log_std_min, log_std_max)

    if deterministic:
        pre_tanh = mean
    else:
        std = tf.exp(log_std)
        pre_tanh = mean + std * tf.random.normal(tf.shape(mean))

    raw_action = tf.tanh(pre_tanh)
    action = scale_action_tensor(raw_action, action_low, action_high)

    std = tf.exp(log_std)
    log_prob = -0.5 * tf.square((pre_tanh - mean) / (std + 1e-6)) - log_std - 0.5 * LOG_2PI
    log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
    correction = tf.reduce_sum(tf.math.log(1.0 - tf.square(raw_action) + 1e-6), axis=1, keepdims=True)
    log_prob = log_prob - correction
    return action, log_prob, raw_action


def select_actions(
    actor,
    obs_batch,
    done_mask,
    action_low,
    action_high,
    log_std_min,
    log_std_max,
    deterministic=False,
):
    obs_batch = np.asarray(obs_batch, dtype=np.float32)
    action_low_tensor = tf.convert_to_tensor(action_low.reshape(1, -1), dtype=tf.float32)
    action_high_tensor = tf.convert_to_tensor(action_high.reshape(1, -1), dtype=tf.float32)
    actions, _, _ = sample_actor(
        actor,
        tf.convert_to_tensor(obs_batch, dtype=tf.float32),
        action_low_tensor,
        action_high_tensor,
        log_std_min,
        log_std_max,
        deterministic=deterministic,
    )
    actions = actions.numpy().astype(np.float32)
    if done_mask is not None:
        actions[np.asarray(done_mask, dtype=np.bool_)] = action_low
    return np.clip(actions, action_low, action_high).astype(np.float32)


def select_action(actor, obs, action_low, action_high, log_std_min, log_std_max):
    return select_actions(
        actor,
        np.expand_dims(obs, axis=0),
        None,
        action_low,
        action_high,
        log_std_min,
        log_std_max,
    )[0]


def build_sac_sample_fn(actor, obs_dim, action_low, action_high, log_std_min, log_std_max,
                        device="/CPU:0", deterministic=False):
    """Trace the squashed-Gaussian sample into one graph per collector.

    The collector calls the actor and tf.random.normal in eager mode. Run concurrently
    from several collector threads that is the exact hazard that crashed PPO -- shape
    corruption at best, silently wrong actions at worst. A traced concrete function is
    safe to call across threads; pinning to `device` keeps the ops with the CPU weights.
    Mirrors build_greedy_action_fn (DQN) and build_sample_action_fn (PPO).
    """
    low = tf.constant(np.asarray(action_low, dtype=np.float32).reshape(1, -1))
    high = tf.constant(np.asarray(action_high, dtype=np.float32).reshape(1, -1))

    @tf.function(input_signature=[tf.TensorSpec([None, obs_dim], tf.float32)])
    def sample(obs_batch):
        with tf.device(device):
            mean, log_std = actor(obs_batch, training=False)
            log_std = tf.clip_by_value(log_std, log_std_min, log_std_max)
            if deterministic:
                pre_tanh = mean
            else:
                std = tf.exp(log_std)
                pre_tanh = mean + std * tf.random.normal(tf.shape(mean))
            raw_action = tf.tanh(pre_tanh)
            return low + 0.5 * (raw_action + 1.0) * (high - low)

    return sample


def build_sac_log_prob_fn(actor, obs_dim, log_std_min, log_std_max, device="/CPU:0"):
    """Trace the mean log-prob of the current policy over a batch, for telemetry.

    Alpha moves on sign(log_prob + target_entropy) alone, and that quantity is otherwise
    invisible: a run whose alpha climbs to 14 and one whose alpha collapses to 0.003 look
    identical in the log. Traced (not eager) for the same reason the sampler is: an eager
    tf.random.normal here is a multi-device wrapped op that leaks ~74 kB per call.
    """
    @tf.function(input_signature=[tf.TensorSpec([None, obs_dim], tf.float32)])
    def mean_log_prob(obs_batch):
        with tf.device(device):
            mean, log_std = actor(obs_batch, training=False)
            log_std = tf.clip_by_value(log_std, log_std_min, log_std_max)
            std = tf.exp(log_std)
            pre_tanh = mean + std * tf.random.normal(tf.shape(mean))
            raw_action = tf.tanh(pre_tanh)
            per_dim = (
                -0.5 * tf.square((pre_tanh - mean) / (std + 1e-6))
                - log_std
                - 0.5 * LOG_2PI
            )
            log_prob = tf.reduce_sum(per_dim, axis=1)
            log_prob -= tf.reduce_sum(
                tf.math.log(1.0 - tf.square(raw_action) + 1e-6), axis=1)
            return tf.reduce_mean(log_prob)

    return mean_log_prob


def select_actions_with(sample_fn, obs_batch, done_mask, action_low, action_high):
    actions = sample_fn(np.asarray(obs_batch, dtype=np.float32)).numpy().astype(np.float32)
    if done_mask is not None:
        actions[np.asarray(done_mask, dtype=np.bool_)] = action_low
    return np.clip(actions, action_low, action_high).astype(np.float32)


def select_action_with(sample_fn, obs, action_low, action_high):
    return select_actions_with(
        sample_fn, np.expand_dims(obs, axis=0), None, action_low, action_high
    )[0]



def build_sac_learner_step(
    actor,
    critic1,
    critic2,
    target_critic1,
    target_critic2,
    actor_optimizer,
    critic1_optimizer,
    critic2_optimizer,
    alpha_optimizer,
    log_alpha,
    target_entropy,
    gamma,
    action_low,
    action_high,
    log_std_min,
    log_std_max,
    *,
    compiled=True,
    xla=False,
    tune_alpha=True,
    min_alpha=0.0,
    grad_clip_norm=0.0,
    grad_clip_adaptive=False,
    grad_clip_k=3.0,
    grad_clip_decay=0.99,
    caps=False,
    caps_lambda_temporal=1.0,
    caps_lambda_spatial=1.0,
    caps_sigma=0.05,
    actor_anchor_coef=0.0,
    actor_anchor_log_std_coef=0.1,
):
    """Build the SAC gradient step, optionally compiled into a tf.function.

    Eager updates dispatch each op individually and force a host sync via .numpy() every
    call, which makes the learner the async pipeline's bottleneck (queue saturates). A
    traced graph fuses the forward/backward/apply into one launch (typically 5-10x). Mirrors
    build_dqn_learner_step. The `update_policy` flag is a Python bool, so it selects between
    two concrete functions (critics-only vs full) to keep each graph's output structure
    fixed instead of returning None from inside a graph. Sampling stays in Python (the
    replay buffer is not a tensor); only the tensor math is graphed.
    """
    gamma_c = tf.constant(float(gamma), dtype=tf.float32)
    action_low_tensor = tf.constant(np.asarray(action_low, dtype=np.float32).reshape(1, -1))
    action_high_tensor = tf.constant(np.asarray(action_high, dtype=np.float32).reshape(1, -1))
    target_entropy_c = tf.constant(float(target_entropy), dtype=tf.float32)
    # Alpha floor: auto-tuned alpha can decay too low late in training, collapsing entropy and
    # letting Q drift upward. Clamp log_alpha so alpha never falls below min_alpha.
    alpha_floor_active = bool(min_alpha and min_alpha > 0.0)
    log_min_alpha_c = tf.constant(
        float(np.log(min_alpha)) if alpha_floor_active else 0.0, dtype=tf.float32)
    # CAPS (Conditioning for Action Policy Smoothness): penalize the deterministic policy for
    # producing different actions on temporally-adjacent states (temporal) and on nearby states
    # (spatial). This trains a smooth, jitter-free controller that holds steady at the target
    # (where consecutive states are ~identical, so the actions must be too). Uses the mean action
    # tanh(mu), not the sampled one.
    caps_enabled = bool(caps) and (caps_lambda_temporal > 0.0 or caps_lambda_spatial > 0.0)
    caps_lt_c = tf.constant(float(caps_lambda_temporal), dtype=tf.float32)
    caps_ls_c = tf.constant(float(caps_lambda_spatial), dtype=tf.float32)
    caps_sigma_c = tf.constant(float(caps_sigma), dtype=tf.float32)
    actor_anchor_enabled = bool(actor_anchor_coef and actor_anchor_coef > 0.0)
    actor_anchor_c = tf.constant(float(actor_anchor_coef), dtype=tf.float32)
    actor_anchor_log_std_c = tf.constant(
        float(actor_anchor_log_std_coef), dtype=tf.float32)
    anchor_actor = None
    if actor_anchor_enabled:
        # Snapshot the policy after checkpoint/policy loading. The frozen copy stays outside
        # the checkpoint so recovery restores the trainable actor without moving its trust
        # reference. A later process resume deliberately establishes a new reference.
        anchor_actor = tf.keras.models.clone_model(actor)
        anchor_actor.set_weights(actor.get_weights())
        anchor_actor.trainable = False

    # Build slots eagerly (outside any graph) so a deferred checkpoint restore populates them
    # deterministically and no variable is created inside the traced function after resume.
    critic1_optimizer.build(critic1.trainable_variables)
    critic2_optimizer.build(critic2.trainable_variables)
    actor_optimizer.build(actor.trainable_variables)
    alpha_optimizer.build([log_alpha])

    # -- gradient clipping: safety net against deadly-triad Q-divergence to NaN ------------
    # The fixed cap (grad_clip_norm) is a hard backstop. Adaptive mode instead clips at
    # grad_clip_k * EMA(grad_norm), bounded by that hard cap -- so the framework needs no
    # per-scenario tuning of the norm: each network's own gradient scale is tracked, while the
    # hard cap still bounds every finite update. The EMA ignores non-finite norms so NaN/Inf
    # cannot poison its state; finite spikes can affect the EMA but still cannot loosen the
    # threshold past the hard value.
    has_hard = bool(grad_clip_norm and grad_clip_norm > 0.0)
    clip_enabled = has_hard or bool(grad_clip_adaptive)
    hard_clip_c = tf.constant(float(grad_clip_norm) if has_hard else 0.0, dtype=tf.float32)
    clip_k_c = tf.constant(float(grad_clip_k), dtype=tf.float32)
    clip_decay_c = tf.constant(float(grad_clip_decay), dtype=tf.float32)
    clip_warmup_c = tf.constant(25.0, dtype=tf.float32)

    def _make_clipper(name):
        # EMA state persists across calls (created eagerly, captured by the traced graph).
        ema = tf.Variable(0.0, dtype=tf.float32, trainable=False, name=f"clip_ema_{name}")
        seen = tf.Variable(0.0, dtype=tf.float32, trainable=False, name=f"clip_seen_{name}")

        def clip(grads):
            if not clip_enabled:
                return grads
            gnorm = tf.linalg.global_norm(grads)
            if grad_clip_adaptive:
                finite = tf.math.is_finite(gnorm)
                g_for_ema = tf.where(finite, gnorm, ema)
                new_ema = tf.where(
                    seen > 0.0, clip_decay_c * ema + (1.0 - clip_decay_c) * g_for_ema, g_for_ema)
                ema.assign(new_ema)
                seen.assign_add(1.0)
                adaptive_cap = clip_k_c * ema
                cap = tf.minimum(hard_clip_c, adaptive_cap) if has_hard else adaptive_cap
                warmup_cap = hard_clip_c if has_hard else adaptive_cap
                cap = tf.where(seen < clip_warmup_c, warmup_cap, cap)  # cold EMA -> don't over-clip
                cap = tf.maximum(cap, 1e-3)
            else:
                cap = hard_clip_c
            clipped, _ = tf.clip_by_global_norm(grads, cap)
            return clipped

        return clip

    clip_critic1 = _make_clipper("critic1")
    clip_critic2 = _make_clipper("critic2")
    clip_actor = _make_clipper("actor")

    def _update_critics_impl(obs, actions, rewards, next_obs, dones):
        alpha = tf.exp(log_alpha)
        next_actions, next_log_prob, _ = sample_actor(
            actor, next_obs, action_low_tensor, action_high_tensor, log_std_min, log_std_max
        )
        target_q1 = target_critic1([next_obs, next_actions], training=False)
        target_q2 = target_critic2([next_obs, next_actions], training=False)
        target_q = tf.minimum(target_q1, target_q2) - alpha * next_log_prob
        y = tf.stop_gradient(rewards + (1.0 - dones) * gamma_c * target_q)

        with tf.GradientTape() as tape:
            q1 = critic1([obs, actions], training=True)
            critic1_loss = tf.reduce_mean(tf.square(y - q1))
        critic1_grads = tape.gradient(critic1_loss, critic1.trainable_variables)
        critic1_grads = clip_critic1(critic1_grads)
        critic1_optimizer.apply_gradients(zip(critic1_grads, critic1.trainable_variables))

        with tf.GradientTape() as tape:
            q2 = critic2([obs, actions], training=True)
            critic2_loss = tf.reduce_mean(tf.square(y - q2))
        critic2_grads = tape.gradient(critic2_loss, critic2.trainable_variables)
        critic2_grads = clip_critic2(critic2_grads)
        critic2_optimizer.apply_gradients(zip(critic2_grads, critic2.trainable_variables))
        return critic1_loss, critic2_loss

    def _update_all_impl(obs, actions, rewards, next_obs, dones):
        critic1_loss, critic2_loss = _update_critics_impl(obs, actions, rewards, next_obs, dones)
        alpha = tf.exp(log_alpha)

        with tf.GradientTape() as tape:
            policy_actions, log_prob, _ = sample_actor(
                actor, obs, action_low_tensor, action_high_tensor, log_std_min, log_std_max
            )
            q1_pi = critic1([obs, policy_actions], training=False)
            q2_pi = critic2([obs, policy_actions], training=False)
            q_pi = tf.minimum(q1_pi, q2_pi)
            actor_loss = tf.reduce_mean(alpha * log_prob - q_pi)
            if caps_enabled:
                mean_now, _ = actor(obs, training=True)
                action_now = tf.tanh(mean_now)
                mean_next, _ = actor(next_obs, training=True)
                temporal = tf.reduce_mean(
                    tf.reduce_sum(tf.square(action_now - tf.tanh(mean_next)), axis=1))
                noisy_obs = obs + tf.random.normal(tf.shape(obs)) * caps_sigma_c
                mean_noisy, _ = actor(noisy_obs, training=True)
                spatial = tf.reduce_mean(
                    tf.reduce_sum(tf.square(action_now - tf.tanh(mean_noisy)), axis=1))
                actor_loss = actor_loss + caps_lt_c * temporal + caps_ls_c * spatial
            if actor_anchor_enabled:
                mean_now, log_std_now = actor(obs, training=True)
                anchor_mean, anchor_log_std = anchor_actor(obs, training=False)
                mean_error = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.square(
                            tf.tanh(mean_now)
                            - tf.stop_gradient(tf.tanh(anchor_mean))
                        ),
                        axis=1,
                    )
                )
                log_std_error = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.square(
                            tf.clip_by_value(log_std_now, log_std_min, log_std_max)
                            - tf.stop_gradient(
                                tf.clip_by_value(
                                    anchor_log_std, log_std_min, log_std_max
                                )
                            )
                        ),
                        axis=1,
                    )
                )
                actor_loss = actor_loss + actor_anchor_c * (
                    mean_error + actor_anchor_log_std_c * log_std_error
                )
        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
        actor_grads = clip_actor(actor_grads)
        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

        if tune_alpha:
            with tf.GradientTape() as tape:
                _, log_prob, _ = sample_actor(
                    actor, obs, action_low_tensor, action_high_tensor, log_std_min, log_std_max
                )
                alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_prob + target_entropy_c))
            alpha_grads = tape.gradient(alpha_loss, [log_alpha])
            alpha_optimizer.apply_gradients(zip(alpha_grads, [log_alpha]))
            if alpha_floor_active:
                log_alpha.assign(tf.maximum(log_alpha, log_min_alpha_c))
        else:
            # Fixed alpha: skip the entropy-coefficient update entirely. The auto-tuner is
            # unstable on this task (alpha runs away up or collapses), so hold it constant.
            alpha_loss = tf.constant(0.0, dtype=tf.float32)
        return actor_loss, critic1_loss, critic2_loss, alpha_loss, tf.exp(log_alpha)

    if compiled:
        update_critics = tf.function(_update_critics_impl, reduce_retracing=True, jit_compile=xla)
        update_all = tf.function(_update_all_impl, reduce_retracing=True, jit_compile=xla)
    else:
        update_critics = _update_critics_impl
        update_all = _update_all_impl

    def learner_step(obs, actions, rewards, next_obs, dones, update_policy=True):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.asarray(rewards).reshape(-1, 1), dtype=tf.float32)
        next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        dones = tf.convert_to_tensor(np.asarray(dones).reshape(-1, 1), dtype=tf.float32)
        if update_policy:
            actor_loss, critic1_loss, critic2_loss, alpha_loss, alpha_value = update_all(
                obs, actions, rewards, next_obs, dones
            )
            return (
                float(actor_loss.numpy()),
                float(critic1_loss.numpy()),
                float(critic2_loss.numpy()),
                float(alpha_loss.numpy()),
                float(alpha_value.numpy()),
            )
        critic1_loss, critic2_loss = update_critics(obs, actions, rewards, next_obs, dones)
        return (
            None,
            float(critic1_loss.numpy()),
            float(critic2_loss.numpy()),
            None,
            float(tf.exp(log_alpha).numpy()),
        )

    return learner_step


def _bc_raw_targets(actions, action_low, action_high):
    action_low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    raw = 2.0 * (actions - action_low) / np.maximum(action_high - action_low, 1e-6) - 1.0
    return np.clip(raw, -0.995, 0.995).astype(np.float32)


def _bc_validation_mse(actor, obs, raw_targets, batch_size):
    """Deterministic action MSE (tanh(mean) vs raw targets) over a held-out set, batched."""
    total = 0.0
    n = len(raw_targets)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        mean, _ = actor(tf.convert_to_tensor(obs[start:stop], dtype=tf.float32), training=False)
        predicted = tf.tanh(mean)
        err = tf.reduce_sum(tf.square(
            tf.convert_to_tensor(raw_targets[start:stop], dtype=tf.float32) - predicted))
        total += float(err.numpy())
    return total / max(n * raw_targets.shape[1], 1)


def _bc_group_batches(group_ids, count, batch_size, rng):
    """Yield index batches balanced by group_id (trajectory), so long trajectories do not dominate
    the imitation loss. Each slot samples a group uniformly, then a transition from it. Task-
    agnostic: it only sees opaque group ids, never any cell/region concept. group_ids=None -> a
    plain shuffled pass."""
    n_batches = max(1, (count + batch_size - 1) // batch_size)
    if group_ids is None:
        order = rng.permutation(count)
        for start in range(0, count, batch_size):
            yield order[start:start + batch_size]
        return
    groups = np.unique(group_ids)
    by_group = {int(g): np.flatnonzero(group_ids == g) for g in groups}
    keys = np.array(list(by_group.keys()))
    for _ in range(n_batches):
        chosen = rng.choice(keys, size=batch_size, replace=True)
        yield np.array([rng.choice(by_group[int(g)]) for g in chosen])


def _load_bc_pairs(paths, obs_dim, action_size):
    """Load imitation-only (obs, actions) pairs (no rewards/next_obs/dones). Carries an opaque
    group_id per transition (group_ids/episode_indices key, else per-file constant), offset so
    trajectories stay disjoint across files. Returns None if paths is empty."""
    obs_parts, act_parts, grp_parts = [], [], []
    offset = 0
    for path in paths:
        with np.load(path) as d:
            if "obs" not in d or "actions" not in d:
                raise ValueError(f"BC-pairs file {path!r} must contain obs and actions arrays")
            o = np.asarray(d["obs"], dtype=np.float32)
            a = np.asarray(d["actions"], dtype=np.float32)
            if "group_ids" in d:
                g = np.asarray(d["group_ids"]).astype(np.int64)
            elif "episode_indices" in d:
                g = np.asarray(d["episode_indices"]).astype(np.int64)
            else:
                g = np.zeros(len(a), dtype=np.int64)
        if o.ndim != 2 or o.shape[1] != obs_dim:
            raise ValueError(f"BC-pairs {path!r} obs shape {o.shape} != obs_dim {obs_dim}")
        if a.ndim != 2 or a.shape[1] != action_size:
            raise ValueError(f"BC-pairs {path!r} actions shape {a.shape} != action_size {action_size}")
        g = g + offset
        obs_parts.append(o)
        act_parts.append(a)
        grp_parts.append(g)
        offset = int(g.max()) + 1 if len(g) else offset
    if not obs_parts:
        return None
    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "actions": np.concatenate(act_parts, axis=0),
        "group_ids": np.concatenate(grp_parts, axis=0),
    }


def _combine_bc_sources(demo_data, bc_pairs):
    """Concatenate the complete-transition demos (--demo-path) and imitation-only pairs
    (--demo-bc-path) into one BC training set with globally-disjoint group ids."""
    obs_parts, act_parts, grp_parts = [], [], []
    offset = 0
    for src in (demo_data, bc_pairs):
        if src is None or len(src.get("actions", [])) == 0:
            continue
        g = np.asarray(src.get("group_ids", np.zeros(len(src["actions"]), dtype=np.int64)))
        g = g.astype(np.int64) + offset
        obs_parts.append(np.asarray(src["obs"], dtype=np.float32))
        act_parts.append(np.asarray(src["actions"], dtype=np.float32))
        grp_parts.append(g)
        offset = int(g.max()) + 1 if len(g) else offset
    if not obs_parts:
        return None, None
    combined = {
        "obs": np.concatenate(obs_parts, axis=0),
        "actions": np.concatenate(act_parts, axis=0),
    }
    return combined, np.concatenate(grp_parts, axis=0)


def pretrain_actor_behavior_cloning(
    actor, demo_data, epochs, batch_size, learning_rate, action_low, action_high,
    demo_validation=None, group_ids=None,
):
    """Behavior-clone the actor onto the demo actions. Batches are balanced by group_id/trajectory
    when provided. With a validation set, keep the weights of the epoch with the lowest validation
    MSE (best-checkpoint / early-stop against overfitting)."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    rng = np.random.default_rng()
    obs = demo_data["obs"]
    raw_targets = _bc_raw_targets(demo_data["actions"], action_low, action_high)
    count = len(raw_targets)
    groups = None if group_ids is None else np.asarray(group_ids)

    val_obs = val_raw = None
    if demo_validation is not None and len(demo_validation.get("actions", [])) > 0:
        val_obs = demo_validation["obs"]
        val_raw = _bc_raw_targets(demo_validation["actions"], action_low, action_high)

    best_val = None
    best_epoch = -1
    best_weights = None
    for epoch in range(int(epochs)):
        losses = []
        for idx in _bc_group_batches(groups, count, batch_size, rng):
            batch_obs = tf.convert_to_tensor(obs[idx], dtype=tf.float32)
            batch_raw_targets = tf.convert_to_tensor(raw_targets[idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                mean, _ = actor(batch_obs, training=True)
                predicted = tf.tanh(mean)
                loss = tf.reduce_mean(tf.square(batch_raw_targets - predicted))
            grads = tape.gradient(loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            losses.append(float(loss.numpy()))

        val_msg = ""
        if val_raw is not None:
            val_mse = _bc_validation_mse(actor, val_obs, val_raw, batch_size)
            val_msg = f" val_mse={val_mse:.6f}"
            if best_val is None or val_mse < best_val:
                best_val = val_mse
                best_epoch = epoch + 1
                best_weights = actor.get_weights()
        print(
            f"demo_bc_epoch={epoch + 1:04d}/{epochs:04d} "
            f"actor_raw_mse={float(np.mean(losses)):.6f}{val_msg}",
            flush=True,
        )

    if best_weights is not None:
        actor.set_weights(best_weights)
        print(
            f"BC best epoch={best_epoch}/{epochs} val_mse={best_val:.6f} "
            "(restored best-validation weights)",
            flush=True,
        )
    return {"best_epoch": best_epoch, "best_val_mse": best_val}


def _bc_actor_path_default(actor_weights_path):
    """Derive a BC-actor path from the actor weights path (…foo.weights.h5 -> …foo_bc.weights.h5)."""
    p = str(actor_weights_path or "bc_actor.weights.h5")
    if p.endswith(".weights.h5"):
        return p[: -len(".weights.h5")] + "_bc.weights.h5"
    root, dot, ext = p.rpartition(".")
    return f"{root}_bc.{ext}" if dot else f"{p}_bc"


def _run_stop_after_bc_eval(
    args, actor, checkpoint, checkpoint_manager, buffer, best_tracker, start_episode
):
    """Save a BC checkpoint, run ONE frozen (deterministic) evaluation at the current curriculum
    level, then stop before SAC. The SAC backend is task-agnostic: the initial level means "Stage
    A" only for a task (e.g. OpenArm) that defines it that way, so nothing here assumes it.

    A failed/empty evaluation exits NON-ZERO (raises SystemExit) rather than reporting a passing
    gate. The summary includes whatever generic reset/cell breakdowns the evaluation produced.
    """
    print(
        "stop-after-bc: saving BC checkpoint + running a frozen evaluation at the current "
        "curriculum level (SAC NOT started)...",
        flush=True,
    )
    saved_path = save_training_checkpoint(
        checkpoint, checkpoint_manager, buffer, start_episode, args,
        final=True, save_replay=False,
    )
    result = None
    try:
        result = best_tracker.evaluate(str(saved_path), start_episode)
    except Exception as exc:
        print(f"stop-after-bc: frozen evaluation raised: {exc}", flush=True)
    if result is None or not getattr(result, "summary", None):
        # Fail closed: no summary means the gate did NOT pass -> non-zero exit.
        raise SystemExit(
            "stop-after-bc: frozen evaluation produced no summary (failed/timeout); "
            "exiting non-zero without a gate result."
        )
    s = result.summary
    # Generic reset/cell breakdowns as produced by the task's eval -- never hardcode cell ids.
    breakdown = {
        key: s[key]
        for key in (
            "reset_modes",
            "target_regions",
            "target_cells",
            "region_success_floor",
            "regular_success_rate",
        )
        if key in s
    }
    print(
        "BC_FROZEN_EVAL "
        + json.dumps(
            {
                "success_rate": float(s.get("success_rate", 0.0)),
                "selection_success_rate": float(
                    s.get("selection_success_rate", s.get("success_rate", 0.0))
                ),
                "position_error_mean": s.get("position_error_mean"),
                "orientation_error_mean": s.get("orientation_error_mean"),
                "hold_frames_max": s.get("hold_frames_max"),
                "episodes": int(getattr(args, "best_evaluation_episodes", 0)),
                "curriculum_level": s.get("curriculum_level"),
                "breakdown": breakdown,
            }
        ),
        flush=True,
    )
    print("stop-after-bc: STOPPED before SAC.", flush=True)


def save_critic_weights(critic1, critic2, args):
    critic1_path = args.critic1_weights_path
    if args.critic_weights_path:
        critic1_path = args.critic_weights_path
    critic1.save_weights(critic1_path)
    critic2.save_weights(args.critic2_weights_path)
    print(f"Saved critic1 weights: {critic1_path}", flush=True)
    print(f"Saved critic2 weights: {args.critic2_weights_path}", flush=True)


def save_training_checkpoint(
    checkpoint,
    checkpoint_manager,
    buffer,
    checkpoint_number_value,
    args,
    final=False,
    save_replay=True,
):
    checkpoint.episode.assign(checkpoint_number_value)
    saved_path = checkpoint_manager.save(checkpoint_number=checkpoint_number_value)
    label = "final checkpoint" if final else "checkpoint"
    print(f"Saved {label}: {saved_path}", flush=True)
    policy_artifact = getattr(args, "policy_artifact", None)
    if policy_artifact is not None:
        policy_artifact.save(checkpoint_number_value)
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




def apply_optimizer_learning_rates(
    actor_optimizer,
    critic1_optimizer,
    critic2_optimizer,
    alpha_optimizer,
    args,
    resumed=False,
):
    actor_learning_rate = args.resume_actor_learning_rate if resumed else args.actor_learning_rate
    alpha_learning_rate = args.resume_alpha_learning_rate if resumed else args.alpha_learning_rate
    actor_optimizer.learning_rate.assign(actor_learning_rate)
    critic1_optimizer.learning_rate.assign(args.critic_learning_rate)
    critic2_optimizer.learning_rate.assign(args.critic_learning_rate)
    alpha_optimizer.learning_rate.assign(alpha_learning_rate)
    print(
        "Optimizer learning rates: "
        f"actor={actor_learning_rate:g} critic={args.critic_learning_rate:g} "
        f"alpha={alpha_learning_rate:g} policy_update_every={args.policy_update_every}",
        flush=True,
    )


def describe_grad_clip(args):
    if getattr(args, "grad_clip_adaptive", False):
        return f"adaptive k={args.grad_clip_k:g} cap={args.grad_clip_norm:g}"
    if args.grad_clip_norm and args.grad_clip_norm > 0:
        return f"fixed norm={args.grad_clip_norm:g}"
    return "off"


def run_async_sac(
    args,
    envs,
    actor,
    critic1,
    critic2,
    target_critic1,
    target_critic2,
    actor_optimizer,
    critic1_optimizer,
    critic2_optimizer,
    alpha_optimizer,
    log_alpha,
    target_entropy,
    buffer,
    checkpoint,
    checkpoint_manager,
    best_tracker,
    start_episode,
    action_low,
    action_high,
    random_action_low,
    random_action_high,
    critic_warmup_target,
    demo_data=None,
    budget=None,
):
    validate_async_arguments(args)
    budget = budget or TrainingBudget(getattr(args, "total_timesteps", 0))
    obs_dim = envs[0].obs_dim
    action_size = envs[0].action_size
    sac_learner = build_sac_learner_step(
        actor,
        critic1,
        critic2,
        target_critic1,
        target_critic2,
        actor_optimizer,
        critic1_optimizer,
        critic2_optimizer,
        alpha_optimizer,
        log_alpha,
        target_entropy,
        args.gamma,
        action_low,
        action_high,
        args.log_std_min,
        args.log_std_max,
        compiled=args.tf_compile_learner,
        xla=args.tf_xla,
        tune_alpha=args.tune_alpha,
        min_alpha=args.min_alpha,
        grad_clip_norm=args.grad_clip_norm,
        grad_clip_adaptive=args.grad_clip_adaptive,
        grad_clip_k=args.grad_clip_k,
        caps=args.caps,
        caps_lambda_temporal=args.caps_lambda_temporal,
        caps_lambda_spatial=args.caps_lambda_spatial,
        caps_sigma=args.caps_sigma,
        actor_anchor_coef=args.actor_anchor_coef,
        actor_anchor_log_std_coef=args.actor_anchor_log_std_coef,
    )
    print(
        f"SAC learner: {'compiled graph' if args.tf_compile_learner else 'eager'}"
        f"{' + XLA' if args.tf_compile_learner and args.tf_xla else ''}"
        f" | grad-clip: {describe_grad_clip(args)}"
        f" | actor-anchor: {args.actor_anchor_coef:g}",
        flush=True,
    )
    with tf.device("/CPU:0"):
        local_models = [
            build_sac_actor(obs_dim=obs_dim, action_size=action_size)
            for _env in envs
        ]
    # One traced sampler per collector: they run concurrently and eager sampling is not
    # thread-safe. set_weights on the closed-over actor keeps each trace current.
    sample_fns = [
        build_sac_sample_fn(m, obs_dim, action_low, action_high, args.log_std_min, args.log_std_max)
        for m in local_models
    ]
    snapshot = PolicySnapshot(actor.get_weights())
    health_monitor = getattr(best_tracker, "health_monitor", None)
    recovery_handler = getattr(health_monitor, "recovery_handler", None)
    if recovery_handler is not None:
        recovery_handler.set_policy_publisher(
            lambda: snapshot.publish(actor.get_weights())
        )
    rngs = [np.random.default_rng(args.env_seed_base + 100_003 * idx) for idx in range(len(envs))]
    replay_ready_event = build_replay_ready_event(buffer, args)

    def action_selector(worker_id, env, _episode, _step_idx, local_actor, state):
        if state["use_random_exploration"]:
            action = sample_exploratory_action(
                action_low,
                action_high,
                env.action_names,
                rng=rngs[worker_id],
                num_agents=len(env.agent_ids) if args.multi_agent else None,
                drive_min=args.random_drive_min,
                steering_abs_max=args.random_steering_abs_max,
                exploration_low=random_action_low,
                exploration_high=random_action_high,
            )
            if args.multi_agent:
                action[state["done_mask"]] = action_low
        elif args.multi_agent:
            action = select_actions_with(
                sample_fns[worker_id],
                state["obs"],
                state["done_mask"],
                action_low,
                action_high,
            )
        else:
            action = select_action_with(
                sample_fns[worker_id],
                state["obs"],
                action_low,
                action_high,
            )
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
    recovery_runtime = OffPolicyRecoveryRuntime(
        args,
        buffer,
        initial_critic_warmup=critic_warmup_target,
        normal_policy_update_every=args.policy_update_every,
        replay_refill=(
            lambda: buffer.add_many(
                demo_data["obs"],
                demo_data["actions"],
                demo_data["rewards"],
                demo_data["next_obs"],
                demo_data["dones"],
            )
            if demo_data is not None and args.demo_prefill
            else 0
        ),
        target_sync=lambda: (
            target_critic1.set_weights(critic1.get_weights()),
            target_critic2.set_weights(critic2.get_weights()),
        ),
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
    completed = int(start_episode)
    done_workers = 0
    learner_updates = 0
    policy_updates_total = 0
    actor_losses = []
    critic1_losses = []
    critic2_losses = []
    alpha_losses = []
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
                        update_policy = recovery_runtime.should_update_policy()
                        batch = buffer.sample(args.batch_size, action_dtype=np.float32)
                        losses = sac_learner(*batch, update_policy=update_policy)
                        critic1_losses.append(losses[1])
                        critic2_losses.append(losses[2])
                        recovery_runtime.record_critic_update()
                        learner_updates += 1
                        updates_performed += 1
                        if losses[0] is not None:
                            actor_losses.append(losses[0])
                            alpha_losses.append(losses[3])
                            policy_updates_total += 1
                        if learner_updates % args.target_update_every == 0:
                            soft_update(target_critic1, critic1, args.tau)
                            soft_update(target_critic2, critic2, args.tau)
                        if (
                            update_policy
                            and policy_updates_total % args.async_policy_publish_updates == 0
                        ):
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
            finish_rate = diagnostics["finishes"] / max(controlled_agents, 1)
            collision_rate = diagnostics["collisions"] / max(controlled_agents, 1)
            stall_rate = diagnostics["stalls"] / max(controlled_agents, 1)
            replay_warmup_left = replay_warmup_remaining(buffer, args)
            critic_warmup_left = recovery_runtime.warmup_left
            throughput = scheduler.throughput(pool)
            print_episode_metrics(event.episode, [
                ("mode", [
                    ("collector", "async"),
                    ("worker", event.worker_id),
                    ("exploration", exploration_label(state)),
                    ("alpha", f"{float(tf.exp(log_alpha).numpy()):.4f}"),
                    *(([("reset_progress_max", f"{state['reset_progress_max']:.3f}")]) if state["reset_progress_max"] is not None else []),
                ]),
                ("outcome", [
                    ("reward", f"{reward_stats['mean']:.3f} [{reward_stats['min']:.3f}, {reward_stats['max']:.3f}]"),
                    ("progress", f"mean:{diagnostics['progress_mean']:.3f} max:{diagnostics['progress_max']:.3f}"),
                    ("pose_error", f"position:{diagnostics.get('position_error_m', 0.0):.4f}m orientation:{diagnostics.get('orientation_error_deg', 0.0):.1f}deg"),
                    ("hold", f"speed:{diagnostics.get('max_joint_speed', 0.0):.3f} frames:{diagnostics.get('hold_frames', 0)}"),
                ]),
                ("agents", [
                    ("finish", f"{diagnostics['finishes']}/{controlled_agents} ({finish_rate:.2%})"),
                    ("collision", format_collision_diagnostics(
                        diagnostics, controlled_agents)),
                    ("stall", f"{diagnostics['stalls']}/{controlled_agents} ({stall_rate:.2%})"),
                ]),
                ("actions", [("mean", format_float_list(mean_action)), ("delta", format_float_list(mean_delta))]),
                ("training", [
                    ("completed", f"{completed}/{args.num_episodes}"),
                    ("total_timesteps", budget.collected),
                    ("queue", f"{throughput['queue_size']}/{throughput['queue_capacity']} ({throughput['queue_saturation']:.0%})"),
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    ("critic_updates", len(critic1_losses)),
                    ("policy_updates", len(actor_losses)),
                    ("replay_warmup_left", replay_warmup_left),
                    ("critic_warmup_left", critic_warmup_left),
                    ("actor_loss", f"{float(np.mean(actor_losses)) if actor_losses else 0.0:.5f}"),
                    ("critic_loss", f"{float(np.mean(critic1_losses)) if critic1_losses else 0.0:.5f}/{float(np.mean(critic2_losses)) if critic2_losses else 0.0:.5f}"),
                    ("alpha_loss", f"{float(np.mean(alpha_losses)) if alpha_losses else 0.0:.5f}"),
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
            critic1_losses.clear()
            critic2_losses.clear()
            alpha_losses.clear()

            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                saved_path = save_training_checkpoint(
                    checkpoint, checkpoint_manager, buffer, completed, args
                )
                last_saved_episode = completed
            else:
                saved_path = None
            if best_tracker.should_evaluate(completed):
                if saved_path is None:
                    saved_path = save_training_checkpoint(
                        checkpoint, checkpoint_manager, buffer, completed, args
                    )
                    last_saved_episode = completed
                request_best_checkpoint_evaluation(best_tracker, saved_path, completed)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupt received: stopping async SAC collectors...", flush=True)
    finally:
        pool.close()

    if last_saved_episode != completed:
        save_training_checkpoint(
            checkpoint, checkpoint_manager, buffer, completed, args, final=True
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
    return completed


@dataclass
class SACPolicyState:
    policy_id: str
    actor: object
    critic1: object
    critic2: object
    target_critic1: object
    target_critic2: object
    actor_optimizer: object
    critic1_optimizer: object
    critic2_optimizer: object
    alpha_optimizer: object
    log_alpha: object
    buffer: ReplayBuffer
    learner: object
    recovery: OffPolicyRecoveryRuntime
    artifact: PolicyArtifactSaver
    trainable: bool


def _save_multi_policy_sac_checkpoint(
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
        "sac",
        episode=episode,
        policy_metadata={
            policy_id: {
                "alpha": float(tf.exp(state.log_alpha).numpy()),
                "replay_size": len(state.buffer),
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


def _multi_policy_sac_recovery_callback(policy_states):
    def post_restore(request):
        outcomes = {
            policy_id: state.recovery.post_restore(request)
            for policy_id, state in policy_states.items()
            if state.trainable
        }
        return {
            "multi_policy": outcomes,
            "target_networks": "all policy critic targets synchronized",
        }

    return post_restore


def run_async_multi_policy_sac(
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
    random_action_low,
    random_action_high,
    budget,
):
    env0 = envs[0]
    obs_dim = env0.obs_dim
    action_size = env0.action_size
    with tf.device("/CPU:0"):
        local_models = [
            {
                policy_id: build_sac_actor(
                    obs_dim=obs_dim,
                    action_size=action_size,
                )
                for policy_id in assignment.policy_ids
            }
            for _env in envs
        ]
    sample_fns = [
        {
            policy_id: build_sac_sample_fn(
                model,
                obs_dim,
                action_low,
                action_high,
                args.log_std_min,
                args.log_std_max,
            )
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
    trainable_states = [
        state for state in policy_states.values() if state.trainable
    ]
    if not trainable_states:
        raise ValueError("Multi-policy SAC requires at least one trainable policy")
    replay_ready_event = build_replay_ready_event(
        trainable_states[0].buffer,
        args,
    )
    if any(
        replay_warmup_remaining(state.buffer, args) > 0
        for state in trainable_states
    ):
        replay_ready_event.clear()

    def action_selector(
        worker_id,
        env,
        _episode,
        _step_idx,
        _worker_models,
        state,
    ):
        if state["use_random_exploration"]:
            actions = sample_exploratory_action(
                action_low,
                action_high,
                env.action_names,
                rng=rngs[worker_id],
                num_agents=len(env.agent_ids),
                drive_min=args.random_drive_min,
                steering_abs_max=args.random_steering_abs_max,
                exploration_low=random_action_low,
                exploration_high=random_action_high,
            )
            actions[state["done_mask"]] = action_low
            for agent_idx, agent_id in enumerate(env.agent_ids):
                policy_id = assignment.policy_for_agent(agent_id)
                if (
                    state["done_mask"][agent_idx]
                    or policy_states[policy_id].trainable
                ):
                    continue
                actions[agent_idx] = select_action_with(
                    sample_fns[worker_id][policy_id],
                    state["obs"][agent_idx],
                    action_low,
                    action_high,
                )
        else:
            actions = np.zeros(
                (len(env.agent_ids), action_size),
                dtype=np.float32,
            )
            for agent_idx, agent_id in enumerate(env.agent_ids):
                if state["done_mask"][agent_idx]:
                    actions[agent_idx] = action_low
                    continue
                policy_id = assignment.policy_for_agent(agent_id)
                actions[agent_idx] = select_action_with(
                    sample_fns[worker_id][policy_id],
                    state["obs"][agent_idx],
                    action_low,
                    action_high,
                )
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
        base_post_restore = _multi_policy_sac_recovery_callback(policy_states)

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
        "Collector mode: async multi-policy SAC "
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
            "critic1": [],
            "critic2": [],
            "alpha_loss": [],
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
                    f"Async multi-policy SAC collector {event.worker_id} failed"
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
                        state = policy_states[policy_id]
                        if state.trainable:
                            state.buffer.add(*payload)
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
                        for _update in range(due):
                            update_policy = (
                                state.recovery.should_update_policy()
                            )
                            result = state.learner(
                                *state.buffer.sample(
                                    args.batch_size,
                                    action_dtype=np.float32,
                                ),
                                update_policy=update_policy,
                            )
                            metrics[policy_id]["critic1"].append(result[1])
                            metrics[policy_id]["critic2"].append(result[2])
                            state.recovery.record_critic_update()
                            if result[0] is not None:
                                metrics[policy_id]["actor"].append(result[0])
                                metrics[policy_id]["alpha_loss"].append(
                                    result[3]
                                )
                            update_number = int(
                                state.critic1_optimizer.iterations.numpy()
                            )
                            if (
                                update_number > 0
                                and update_number
                                % args.target_update_every
                                == 0
                            ):
                                soft_update(
                                    state.target_critic1,
                                    state.critic1,
                                    args.tau,
                                )
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
                    publish_update_count %= args.async_policy_publish_updates
                if budget.exhausted:
                    break
                continue

            if not isinstance(event, AsyncEpisodeEvent):
                continue
            completed += 1
            payload = event.payload
            by_policy = {}
            for policy_id, state in policy_states.items():
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
                    "alpha": float(tf.exp(state.log_alpha).numpy()),
                    "replay": len(state.buffer),
                    "policy_updates": len(values["actor"]),
                    "critic_updates": len(values["critic1"]),
                    "actor_loss": (
                        float(np.mean(values["actor"]))
                        if values["actor"]
                        else 0.0
                    ),
                    "critic_loss": (
                        float(np.mean(values["critic1"]))
                        if values["critic1"]
                        else 0.0
                    ),
                }
            diagnostics = summarize_episode_diagnostics([payload], True)
            throughput = scheduler.throughput(pool)
            print_episode_metrics(event.episode, [
                ("mode", [
                    ("collector", "async"),
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
                saved_path = _save_multi_policy_sac_checkpoint(
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
                    saved_path = _save_multi_policy_sac_checkpoint(
                        checkpoint,
                        checkpoint_manager,
                        policy_states,
                        assignment,
                        completed,
                        args,
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
            "\nInterrupt received: stopping async multi-policy SAC collectors...",
            flush=True,
        )
    finally:
        pool.close()

    if last_saved_episode != completed:
        _save_multi_policy_sac_checkpoint(
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


def run_sync_multi_policy_sac(
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
    if args.demo_path:
        raise ValueError(
            "SAC demonstrations are not yet routed by policy. Remove --demo-path "
            "for independent multi-policy training."
        )
    if args.policy_path:
        raise ValueError(
            "Independent multi-policy SAC warm starts currently use "
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
    random_action_low, random_action_high = continuous_exploration_bounds(
        env0.action_space_spec,
        action_low,
        action_high,
    )
    target_entropy = (
        args.target_entropy
        if args.target_entropy is not None
        else -float(action_size)
    )
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
        actor = build_sac_actor(obs_dim=obs_dim, action_size=action_size)
        critic1 = build_continuous_critic(
            obs_dim=obs_dim,
            action_size=action_size,
        )
        critic2 = build_continuous_critic(
            obs_dim=obs_dim,
            action_size=action_size,
        )
        target_critic1 = build_continuous_critic(
            obs_dim=obs_dim,
            action_size=action_size,
        )
        target_critic2 = build_continuous_critic(
            obs_dim=obs_dim,
            action_size=action_size,
        )
        target_critic1.set_weights(critic1.get_weights())
        target_critic2.set_weights(critic2.get_weights())
        actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.actor_learning_rate
        )
        critic1_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.critic_learning_rate
        )
        critic2_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.critic_learning_rate
        )
        alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.alpha_learning_rate
        )
        log_alpha = tf.Variable(
            np.log(args.initial_alpha),
            dtype=tf.float32,
            name=f"log_alpha_{assignment.key_for(policy_id)}",
        )
        buffer = ReplayBuffer(capacity=args.replay_capacity)
        learner = build_sac_learner_step(
            actor,
            critic1,
            critic2,
            target_critic1,
            target_critic2,
            actor_optimizer,
            critic1_optimizer,
            critic2_optimizer,
            alpha_optimizer,
            log_alpha,
            target_entropy,
            args.gamma,
            action_low,
            action_high,
            args.log_std_min,
            args.log_std_max,
            compiled=args.tf_compile_learner,
            xla=args.tf_xla,
            tune_alpha=args.tune_alpha,
            min_alpha=args.min_alpha,
            grad_clip_norm=args.grad_clip_norm,
            grad_clip_adaptive=args.grad_clip_adaptive,
            grad_clip_k=args.grad_clip_k,
            caps=args.caps,
            caps_lambda_temporal=args.caps_lambda_temporal,
            caps_lambda_spatial=args.caps_lambda_spatial,
            caps_sigma=args.caps_sigma,
            actor_anchor_coef=args.actor_anchor_coef,
            actor_anchor_log_std_coef=args.actor_anchor_log_std_coef,
        )
        recovery = OffPolicyRecoveryRuntime(
            args,
            buffer,
            normal_policy_update_every=args.policy_update_every,
            target_sync=lambda c1=critic1, c2=critic2, t1=target_critic1, t2=target_critic2: (
                t1.set_weights(c1.get_weights()),
                t2.set_weights(c2.get_weights()),
            ),
        )
        metadata = build_policy_metadata(
            "sac",
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
        state = SACPolicyState(
            policy_id=policy_id,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target_critic1=target_critic1,
            target_critic2=target_critic2,
            actor_optimizer=actor_optimizer,
            critic1_optimizer=critic1_optimizer,
            critic2_optimizer=critic2_optimizer,
            alpha_optimizer=alpha_optimizer,
            log_alpha=log_alpha,
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
            trainable=assignment.is_trainable(policy_id),
        )
        policy_states[policy_id] = state
        policy_trackables[assignment.key_for(policy_id)] = tf.train.Checkpoint(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target_critic1=target_critic1,
            target_critic2=target_critic2,
            actor_optimizer=actor_optimizer,
            critic1_optimizer=critic1_optimizer,
            critic2_optimizer=critic2_optimizer,
            alpha_optimizer=alpha_optimizer,
            log_alpha=log_alpha,
        )

    checkpoint = tf.train.Checkpoint(
        episode=tf.Variable(0, dtype=tf.int64),
        policies=tf.train.Checkpoint(**policy_trackables),
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=args.checkpoint_dir,
        max_to_keep=args.keep_checkpoints,
    )
    resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_manager)
    start_episode = 0
    if resume_checkpoint:
        manifest_path = Path(args.checkpoint_dir) / "multi_policy.json"
        if manifest_path.is_file():
            stored = load_multi_policy_manifest(manifest_path)
            if stored.get("agent_to_policy", {}) != assignment.agent_to_policy:
                raise RuntimeError(
                    "The checkpoint policy assignment does not match the scenario"
                )
            if str(stored.get("algorithm", "sac")) != "sac":
                raise RuntimeError(
                    f"Multi-policy checkpoint algorithm={stored.get('algorithm')!r}, "
                    "expected 'sac'"
                )
        checkpoint.restore(resume_checkpoint).expect_partial()
        start_episode = int(checkpoint.episode.numpy())
        restore_policy_replays(
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
        if args.actor_anchor_coef > 0.0:
            # The first learner instances were built before checkpoint restoration. Rebuild
            # them now so every policy anchors to its restored actor, not to random startup
            # weights.
            for state in policy_states.values():
                state.learner = build_sac_learner_step(
                    state.actor,
                    state.critic1,
                    state.critic2,
                    state.target_critic1,
                    state.target_critic2,
                    state.actor_optimizer,
                    state.critic1_optimizer,
                    state.critic2_optimizer,
                    state.alpha_optimizer,
                    state.log_alpha,
                    target_entropy,
                    args.gamma,
                    action_low,
                    action_high,
                    args.log_std_min,
                    args.log_std_max,
                    compiled=args.tf_compile_learner,
                    xla=args.tf_xla,
                    tune_alpha=args.tune_alpha,
                    min_alpha=args.min_alpha,
                    grad_clip_norm=args.grad_clip_norm,
                    grad_clip_adaptive=args.grad_clip_adaptive,
                    grad_clip_k=args.grad_clip_k,
                    caps=args.caps,
                    caps_lambda_temporal=args.caps_lambda_temporal,
                    caps_lambda_spatial=args.caps_lambda_spatial,
                    caps_sigma=args.caps_sigma,
                    actor_anchor_coef=args.actor_anchor_coef,
                    actor_anchor_log_std_coef=args.actor_anchor_log_std_coef,
                )

    critic_warmup = (
        max(0, int(args.critic_warmup_updates))
        if resume_checkpoint
        else 0
    )
    for state in policy_states.values():
        apply_optimizer_learning_rates(
            state.actor_optimizer,
            state.critic1_optimizer,
            state.critic2_optimizer,
            state.alpha_optimizer,
            args,
            resumed=resume_checkpoint is not None,
        )
        state.recovery.critic_warmup_target = (
            critic_warmup if state.trainable else 0
        )

    write_multi_policy_manifest(
        args.checkpoint_dir,
        assignment,
        "sac",
        episode=start_episode,
    )
    best_tracker.configure_recovery(
        checkpoint,
        [
            (f"actor_{assignment.key_for(policy_id)}", state.actor_optimizer)
            for policy_id, state in policy_states.items()
            if state.trainable
        ]
        + [
            (f"critic1_{assignment.key_for(policy_id)}", state.critic1_optimizer)
            for policy_id, state in policy_states.items()
            if state.trainable
        ]
        + [
            (f"critic2_{assignment.key_for(policy_id)}", state.critic2_optimizer)
            for policy_id, state in policy_states.items()
            if state.trainable
        ]
        + [
            (f"alpha_{assignment.key_for(policy_id)}", state.alpha_optimizer)
            for policy_id, state in policy_states.items()
            if state.trainable
        ],
        post_restore=_multi_policy_sac_recovery_callback(policy_states),
    )
    print(
        "Independent multi-policy SAC: "
        f"assignment={assignment.mode} policies={list(assignment.policy_ids)} "
        f"trainable={list(assignment.trainable_policy_ids)} "
        f"agent_map={assignment.agent_to_policy}",
        flush=True,
    )
    print(
        f"SAC learner instances={len(policy_states)} "
        f"target_entropy={target_entropy:.3f} "
        f"grad_clip={describe_grad_clip(args)}",
        flush=True,
    )

    if args.collector_mode == "async":
        return run_async_multi_policy_sac(
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
            random_action_low,
            random_action_high,
            budget,
        )

    last_completed_episode = start_episode
    last_saved_episode = None
    interrupted = False
    # Action selection goes through the traced, CPU-pinned sample graph, like the async paths.
    # Calling sample_actor eagerly instead makes TF execute RandomStandardNormal as a wrapped
    # single-op function; with a GPU present that function is multi-device (the shape input is
    # host-side), and every call instantiates a partitioned function that carries a copy of the
    # whole function library -- ~74 kB leaked per action, ~5 MB/s at 70 actions/s.
    sample_fn_cache = {}

    def policy_sample_fn(policy_id, actor):
        cached = sample_fn_cache.get(policy_id)
        if cached is None or cached[0] is not actor:
            # Keyed on the model object so a checkpoint that swaps the actor retraces once.
            cached = (
                actor,
                build_sac_sample_fn(
                    actor,
                    obs_dim,
                    action_low,
                    action_high,
                    args.log_std_min,
                    args.log_std_max,
                ),
            )
            sample_fn_cache[policy_id] = cached
        return cached[1]

    sync_throttles = {
        policy_id: SyncUpdateThrottle(args)
        for policy_id, state in policy_states.items()
        if state.trainable
    }
    sync_learner_updates = {
        policy_id: int(state.critic1_optimizer.iterations.numpy())
        for policy_id, state in policy_states.items()
        if state.trainable
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
            reset_progress_max = curriculum_reset_progress_max(
                episode,
                args,
            )
            # `episode` counts sync BARRIERS here, not episodes -- see the note in the single-policy
            # sync loop. The scene curriculum needs the global episode count the async path sends.
            global_episode_base = episode * len(envs)
            env_states = []
            for env_idx, env in enumerate(envs):
                config = {
                    "training_episode": global_episode_base + env_idx,
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
                agent_count = len(env.agent_ids)
                env_states.append({
                    "obs": obs,
                    "done": False,
                    "done_mask": np.asarray(
                        info.get(
                            "per_agent_done",
                            np.zeros((agent_count,), dtype=np.bool_),
                        ),
                        dtype=np.bool_,
                    ),
                    "ep_reward": np.zeros(
                        (agent_count,),
                        dtype=np.float32,
                    ),
                    "action_sum": np.zeros(
                        (agent_count, action_size),
                        dtype=np.float32,
                    ),
                    "action_count": np.zeros(
                        (agent_count, 1),
                        dtype=np.float32,
                    ),
                    "previous_action": np.full(
                        (agent_count, action_size),
                        np.nan,
                        dtype=np.float32,
                    ),
                    "action_delta_sum": np.zeros(
                        (agent_count, action_size),
                        dtype=np.float32,
                    ),
                    "action_delta_count": np.zeros(
                        (agent_count, 1),
                        dtype=np.float32,
                    ),
                    "max_track_progress": np.zeros(
                        (agent_count,),
                        dtype=np.float32,
                    ),
                    "last_track_progress": np.zeros(
                        (agent_count,),
                        dtype=np.float32,
                    ),
                    "finish_reached": np.zeros(
                        (agent_count,),
                        dtype=np.bool_,
                    ),
                    "collision_seen": np.zeros(
                        (agent_count,),
                        dtype=np.bool_,
                    ),
                    "collision_count": np.zeros(
                        (agent_count,),
                        dtype=np.int32,
                    ),
                    "stalled_seen": np.zeros(
                        (agent_count,),
                        dtype=np.bool_,
                    ),
                    "stalled_count": np.zeros(
                        (agent_count,),
                        dtype=np.int32,
                    ),
                })

            metrics = {
                policy_id: {
                    "actor": [],
                    "critic1": [],
                    "critic2": [],
                    "alpha_loss": [],
                    "alpha": [],
                }
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
                        agent_policy_id = assignment.policy_for_agent(agent_id)
                        policy = policy_states[agent_policy_id]
                        if use_random and policy.trainable:
                            selected = sample_exploratory_action(
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
                            selected = select_action_with(
                                policy_sample_fn(
                                    agent_policy_id,
                                    policy.actor,
                                ),
                                env_state["obs"][agent_idx],
                                action_low,
                                action_high,
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
                        agent_info = (
                            agent_infos[agent_idx]
                            if agent_idx < len(agent_infos)
                            else {}
                        )
                        update_episode_diagnostics(
                            env_state,
                            agent_info,
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
                    env_state["done"] = done or bool(
                        np.all(per_agent_done)
                    )

                    if not best_tracker.health_monitor.verification_pending:
                        for policy_id, policy in policy_states.items():
                            if (
                                not policy.trainable
                                or len(policy.buffer)
                                < max(args.replay_warmup, args.batch_size)
                            ):
                                continue
                            updates_due = sync_throttles[
                                policy_id
                            ].updates_due(
                                transitions_by_policy.get(policy_id, 0),
                                env_steps=(
                                    1
                                    if transitions_by_policy.get(
                                        policy_id, 0
                                    ) > 0
                                    else 0
                                ),
                            )
                            for _update in range(updates_due):
                                update_policy = (
                                    policy.recovery.should_update_policy()
                                )
                                result = policy.learner(
                                    *policy.buffer.sample(
                                        args.batch_size,
                                        action_dtype=np.float32,
                                    ),
                                    update_policy=update_policy,
                                )
                                metrics[policy_id]["critic1"].append(result[1])
                                metrics[policy_id]["critic2"].append(result[2])
                                policy.recovery.record_critic_update()
                                if update_policy:
                                    metrics[policy_id]["actor"].append(
                                        result[0]
                                    )
                                    metrics[policy_id]["alpha_loss"].append(
                                        result[3]
                                    )
                                    metrics[policy_id]["alpha"].append(
                                        result[4]
                                    )
                                sync_learner_updates[policy_id] += 1
                                if (
                                    sync_learner_updates[policy_id]
                                    % args.target_update_every
                                    == 0
                                ):
                                    soft_update(
                                        policy.target_critic1,
                                        policy.critic1,
                                        args.tau,
                                    )
                                    soft_update(
                                        policy.target_critic2,
                                        policy.critic2,
                                        args.tau,
                                    )
                if budget.exhausted:
                    break

            rewards_summary = [
                state["ep_reward"].tolist()
                for state in env_states
            ]
            diagnostics = summarize_episode_diagnostics(
                env_states,
                True,
            )
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
                    "alpha": (
                        float(np.mean(values["alpha"]))
                        if values["alpha"]
                        else float(tf.exp(policy.log_alpha).numpy())
                    ),
                    "replay": len(policy.buffer),
                    "policy_updates": len(values["actor"]),
                    "critic_updates": len(values["critic1"]),
                    "actor_loss": (
                        float(np.mean(values["actor"]))
                        if values["actor"]
                        else 0.0
                    ),
                    "critic_loss": (
                        float(np.mean(values["critic1"]))
                        if values["critic1"]
                        else 0.0
                    ),
                }

            print_episode_metrics(episode, [
                ("mode", [
                    ("policies", len(policy_states)),
                    ("assignment", assignment.mode),
                    ("exploration", "random" if use_random else "policy"),
                    *(
                        [(
                            "reset_progress_max",
                            f"{reset_progress_max:.3f}",
                        )]
                        if reset_progress_max is not None
                        else []
                    ),
                ]),
                ("outcome", [
                    ("reward", rewards_summary),
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
                saved_path = _save_multi_policy_sac_checkpoint(
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
                    saved_path = _save_multi_policy_sac_checkpoint(
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
            "\nInterrupt received: saving the multi-policy SAC state...",
            flush=True,
        )

    if last_saved_episode != last_completed_episode:
        _save_multi_policy_sac_checkpoint(
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


def main():
    args = parse_args()
    budget = TrainingBudget(args.total_timesteps)
    validate_async_arguments(args)
    if args.policy_update_every < 1:
        raise ValueError("--policy-update-every must be at least 1")
    if args.actor_anchor_coef < 0.0:
        raise ValueError("--actor-anchor-coef cannot be negative")
    if args.actor_anchor_log_std_coef < 0.0:
        raise ValueError("--actor-anchor-log-std-coef cannot be negative")
    best_tracker = BestCheckpointTracker(args, "sac")
    describe_tensorflow_backend(args)
    dashboard = maybe_start_dashboard(args, algorithm="sac")
    training_start_time = time.monotonic()
    # Architecture is process-wide state, so it must be fixed before the first network is
    # built -- the seeding block below is the last point where nothing exists yet.
    set_network_layers(args.network_layers or default_network_layers("sac"))
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
    critic1 = None
    critic2 = None
    buffer = None
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
            raise RuntimeError(f"algorithms/sac.py requires action_type='continuous', got {env0.action_type!r}")

        obs_dim = env0.obs_dim
        action_size = env0.action_size
        if dashboard is not None:
            dashboard.set_meta(
                agents=len(getattr(env0, "agent_ids", []) or [env0.agent_id]),
                obs_dim=obs_dim,
                action_size=action_size,
            )
        action_low = np.asarray(env0.action_low, dtype=np.float32)
        action_high = np.asarray(env0.action_high, dtype=np.float32)
        random_action_low, random_action_high = continuous_exploration_bounds(
            env0.action_space_spec,
            action_low,
            action_high,
        )
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(action_size)
        print(
            f"Scenario spec: agent_id={env0.agent_id} {env0.agent_summary()} multi_agent={args.multi_agent} "
            f"{env0.team_summary()} obs_dim={obs_dim} action_size={action_size} action_names={env0.action_names} "
            f"target_entropy={target_entropy:.3f}",
            flush=True,
        )
        print(
            f"Random exploration bounds: low={random_action_low.tolist()} high={random_action_high.tolist()}",
            flush=True,
        )

        for env in envs[1:]:
            if env.obs_dim != obs_dim or env.action_type != "continuous" or env.action_size != action_size:
                raise RuntimeError("All parallel environments must expose the same obs_dim and continuous action_size")

        if args.multi_policy:
            run_sync_multi_policy_sac(
                args,
                envs,
                stepper,
                best_tracker,
                budget,
            )
            return

        actor = build_sac_actor(obs_dim=obs_dim, action_size=action_size)
        args.policy_artifact = PolicyArtifactSaver(
            actor,
            args.checkpoint_dir,
            build_policy_metadata("sac", env0),
        )
        critic1 = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        critic2 = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        target_critic1 = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        target_critic2 = build_continuous_critic(obs_dim=obs_dim, action_size=action_size)
        target_critic1.set_weights(critic1.get_weights())
        target_critic2.set_weights(critic2.get_weights())

        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_learning_rate)
        critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate)
        critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate)
        alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=args.alpha_learning_rate)
        log_alpha = tf.Variable(np.log(args.initial_alpha), dtype=tf.float32, name="log_alpha")
        buffer = ReplayBuffer(capacity=args.replay_capacity)
        start_episode = 0

        checkpoint = tf.train.Checkpoint(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target_critic1=target_critic1,
            target_critic2=target_critic2,
            actor_optimizer=actor_optimizer,
            critic1_optimizer=critic1_optimizer,
            critic2_optimizer=critic2_optimizer,
            alpha_optimizer=alpha_optimizer,
            log_alpha=log_alpha,
            episode=tf.Variable(0, dtype=tf.int64),
        )
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
            checkpoint.restore(resume_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            print(
                f"Resumed checkpoint {resume_checkpoint} from episode={start_episode} "
                f"alpha={float(tf.exp(log_alpha).numpy()):.4f}",
                flush=True,
            )
            restored_replay_count = restore_replay_buffer(args, resume_checkpoint, buffer)
        elif args.policy_path:
            loaded_policy = load_policy_into_model(actor, args.policy_path, expected_algorithm="sac")
            args._replay_warmup_uses_restored_policy = True
            print(
                f"Warm-started SAC actor from {loaded_policy['source_kind']}: "
                f"{loaded_policy['path']} (fresh critics, optimizers, replay and alpha, episode=0)",
                flush=True,
            )

        apply_optimizer_learning_rates(
            actor_optimizer,
            critic1_optimizer,
            critic2_optimizer,
            alpha_optimizer,
            args,
            resumed=resume_checkpoint is not None,
        )
        best_tracker.configure_recovery(
            checkpoint,
            [
                ("actor", actor_optimizer),
                ("critic1", critic1_optimizer),
                ("critic2", critic2_optimizer),
                ("alpha", alpha_optimizer),
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

        demo_validation = None
        if getattr(args, "demo_validation_path", None):
            demo_validation = load_demonstration_arrays(
                args.demo_validation_path,
                obs_dim=obs_dim,
                action_size=action_size,
                max_transitions=0,
            )
            if demo_validation is not None:
                print(
                    f"Loaded BC validation demonstrations transitions="
                    f"{len(demo_validation['actions'])} paths={args.demo_validation_path}",
                    flush=True,
                )

        bc_pairs = None
        if getattr(args, "demo_bc_path", None):
            bc_pairs = _load_bc_pairs(args.demo_bc_path, obs_dim, action_size)
            if bc_pairs is not None:
                print(
                    f"Loaded imitation-only BC pairs (BC loss ONLY, not replayed) "
                    f"transitions={len(bc_pairs['actions'])} paths={args.demo_bc_path}",
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
        elif demo_data is not None and args.demo_prefill:
            print(
                "Skipped demonstration prefill because the checkpoint replay buffer was restored.",
                flush=True,
            )

        bc_train, bc_groups = _combine_bc_sources(demo_data, bc_pairs)
        run_behavior_cloning = (
            bc_train is not None
            and args.demo_bc_epochs > 0
            and (resume_checkpoint is None or args.demo_bc_on_resume)
        )
        if run_behavior_cloning:
            pretrain_actor_behavior_cloning(
                actor,
                bc_train,
                epochs=args.demo_bc_epochs,
                batch_size=args.demo_bc_batch_size,
                learning_rate=args.demo_bc_learning_rate or args.actor_learning_rate,
                action_low=action_low,
                action_high=action_high,
                demo_validation=demo_validation,
                group_ids=bc_groups,
            )
        elif demo_data is not None and args.demo_bc_epochs > 0 and resume_checkpoint is not None:
            print(
                "Skipped behavior cloning on resume; use --demo-bc-on-resume to request it explicitly.",
                flush=True,
            )

        # A freshly-run behavior cloning warm-starts the actor from the demos, so the critics (which
        # start random) must catch up on the prefilled replay with the actor + alpha FROZEN before
        # SAC begins -- otherwise the first policy-gradient steps chase a garbage Q and wreck the BC
        # actor. Resume / actor warm-start trigger the same warmup.
        critic_warmup_source = (
            "checkpoint resume"
            if resume_checkpoint
            else "actor warm start"
            if args.policy_path
            else "behavior cloning"
            if run_behavior_cloning
            else None
        )
        critic_warmup_target = (
            max(0, int(args.critic_warmup_updates))
            if critic_warmup_source is not None
            else 0
        )
        if critic_warmup_target > 0:
            print(
                f"Critic warmup after {critic_warmup_source}: "
                f"updates={critic_warmup_target} actor=frozen alpha=frozen",
                flush=True,
            )

        # Persist the BC-pretrained actor BEFORE any SAC/warmup update touches it.
        if run_behavior_cloning:
            bc_actor_path = (
                args.bc_actor_weights_path
                or _bc_actor_path_default(args.actor_weights_path)
            )
            Path(bc_actor_path).parent.mkdir(parents=True, exist_ok=True)
            actor.save_weights(bc_actor_path)
            print(f"Saved BC actor weights (pre-SAC): {bc_actor_path}", flush=True)

        # Gate: BC + frozen evaluation, then STOP before the SAC loop.
        if args.stop_after_bc:
            if not run_behavior_cloning:
                raise SystemExit(
                    "--stop-after-bc requires behavior cloning to have actually run: pass "
                    "--demo-path and --demo-bc-epochs>0 (and, on resume, --demo-bc-on-resume). "
                    "Refusing to report a BC gate when no BC was performed."
                )
            _run_stop_after_bc_eval(
                args, actor, checkpoint, checkpoint_manager, buffer, best_tracker,
                start_episode,
            )
            return

        opponent_teams = validate_team_layout(envs, args.opponent_pool)
        opponent_pool = OpponentPool(
            args,
            algorithm="sac",
            model_factory=lambda: build_sac_actor(obs_dim=obs_dim, action_size=action_size),
            metadata={"obs_dim": obs_dim, "action_size": action_size},
        )
        if opponent_pool.enabled:
            print(
                f"Opponent pool: dir={opponent_pool.directory} teams={opponent_teams} "
                f"snapshots={len(opponent_pool.entries)} sampling={opponent_pool.sampling}",
                flush=True,
            )

        if args.collector_mode == "async":
            last_completed_episode = run_async_sac(
                args,
                envs,
                actor,
                critic1,
                critic2,
                target_critic1,
                target_critic2,
                actor_optimizer,
                critic1_optimizer,
                critic2_optimizer,
                alpha_optimizer,
                log_alpha,
                target_entropy,
                buffer,
                checkpoint,
                checkpoint_manager,
                best_tracker,
                start_episode,
                action_low,
                action_high,
                random_action_low,
                random_action_high,
                critic_warmup_target,
                demo_data,
                budget,
            )
            actor.save_weights(args.actor_weights_path)
            save_critic_weights(critic1, critic2, args)
            print(f"Saved actor weights: {args.actor_weights_path}", flush=True)
            return

        sac_learner = build_sac_learner_step(
            actor,
            critic1,
            critic2,
            target_critic1,
            target_critic2,
            actor_optimizer,
            critic1_optimizer,
            critic2_optimizer,
            alpha_optimizer,
            log_alpha,
            target_entropy,
            args.gamma,
            action_low,
            action_high,
            args.log_std_min,
            args.log_std_max,
            compiled=args.tf_compile_learner,
            xla=args.tf_xla,
            tune_alpha=args.tune_alpha,
            min_alpha=args.min_alpha,
            grad_clip_norm=args.grad_clip_norm,
            grad_clip_adaptive=args.grad_clip_adaptive,
            grad_clip_k=args.grad_clip_k,
            caps=args.caps,
            caps_lambda_temporal=args.caps_lambda_temporal,
            caps_lambda_spatial=args.caps_lambda_spatial,
            caps_sigma=args.caps_sigma,
            actor_anchor_coef=args.actor_anchor_coef,
            actor_anchor_log_std_coef=args.actor_anchor_log_std_coef,
        )
        print(
            f"SAC learner: {'compiled graph' if args.tf_compile_learner else 'eager'}"
            f"{' + XLA' if args.tf_compile_learner and args.tf_xla else ''}"
            f" | grad-clip: {describe_grad_clip(args)}"
            f" | actor-anchor: {args.actor_anchor_coef:g}",
            flush=True,
        )

        recovery_runtime = OffPolicyRecoveryRuntime(
            args,
            buffer,
            initial_critic_warmup=critic_warmup_target,
            normal_policy_update_every=args.policy_update_every,
            replay_refill=(
                lambda: buffer.add_many(
                    demo_data["obs"],
                    demo_data["actions"],
                    demo_data["rewards"],
                    demo_data["next_obs"],
                    demo_data["dones"],
                )
                if demo_data is not None and args.demo_prefill
                else 0
            ),
            target_sync=lambda: (
                target_critic1.set_weights(critic1.get_weights()),
                target_critic2.set_weights(critic2.get_weights()),
            ),
        )
        recovery_handler = best_tracker.health_monitor.recovery_handler
        if recovery_handler is not None:
            recovery_handler.set_post_restore(recovery_runtime.post_restore)

        # Transition-based update throttle so the sync path runs the SAME updates-per-transition (UTD)
        # as async (update_every); otherwise sync did 1 update/transition and diverged on high rewards.
        sync_throttle = SyncUpdateThrottle(args)
        sync_learner_updates = 0
        last_saved_episode = None
        # Action selection goes through the traced, CPU-pinned sample graph, like the async path.
        # Calling sample_actor eagerly instead makes TF execute RandomStandardNormal as a wrapped
        # single-op function; with a GPU present that function is multi-device (the shape input is
        # host-side), and every call instantiates a partitioned function carrying a copy of the
        # whole function library -- ~74 kB leaked per action, ~5 MB/s at 70 actions/s.
        sync_sample_fns = {}

        def sync_sample_fn(model, deterministic=False):
            # Keyed on the model object, so a swapped opponent or restored actor retraces once.
            key = (id(model), bool(deterministic))
            cached = sync_sample_fns.get(key)
            if cached is None or cached[0] is not model:
                cached = (
                    model,
                    build_sac_sample_fn(
                        model,
                        obs_dim,
                        action_low,
                        action_high,
                        args.log_std_min,
                        args.log_std_max,
                        deterministic=deterministic,
                    ),
                )
                sync_sample_fns[key] = cached
            return cached[1]

        # Envs auto-reset the moment they terminate, exactly like an SB3 VecEnv, instead of idling
        # until the slowest env in the batch finishes. With an episode-wide barrier a collision on
        # step 10 parked that env for the remaining 290 steps: measured barrier utilisation was 40%,
        # and the tail of every barrier was fed by a single trajectory. The batch is therefore only a
        # step budget now -- episodes are never aligned across envs, and `sync_completed_episodes`
        # (not the batch index) is what counts toward --num-episodes.
        # One sync iteration runs every env at once, so `episode` counts BARRIERS, not
        # episodes. The Godot curriculum keys on a global episode count (openarm_scenario.gd:
        # "the REAL global episode count"), which is what the async path sends; handing it the
        # barrier index paces every episode-driven schedule in the scene len(envs) times slower.
        # Envs auto-reset the moment they terminate, like an SB3 VecEnv, instead of idling until
        # the slowest env in the batch finishes. With an episode-wide barrier a collision on step
        # 10 parked that env for the remaining 290 steps: measured batch utilisation was 40%, and
        # the tail of every batch was fed by a single trajectory. A batch is only a step budget
        # now -- episodes never align across envs, and `sync_completed_episodes` (not the batch
        # index) is what counts toward --num-episodes.
        sync_next_episode = start_episode
        sync_completed_episodes = start_episode
        env_states = []
        # A batch is only a REPORTING window: episodes span batches, so shortening it costs no
        # telemetry. These accumulators therefore live across batches and are cleared when a
        # report actually goes out.
        batch_step_budget = (
            args.sync_batch_steps
            if getattr(args, "sync_batch_steps", 0) > 0
            else args.max_steps_per_episode
        )
        sync_log_prob_fn = build_sac_log_prob_fn(
            actor, obs_dim, args.log_std_min, args.log_std_max)
        finished_states = []
        actor_losses = []
        critic1_losses = []
        critic2_losses = []
        alpha_losses = []
        alphas = []
        batch_started = time.monotonic()
        batch_transitions_start = budget.collected
        for episode in range(start_episode, args.num_episodes):
            opponent_match = opponent_pool.start_episode(actor, sync_completed_episodes)
            use_random_exploration = sync_completed_episodes < max(
                0, args.random_exploration_episodes)
            reset_progress_max = curriculum_reset_progress_max(sync_completed_episodes, args)
            def start_env_episode(env, env_idx, episode_number):
                """Configure and reset one env for a NEW episode, returning its state dict.

                Called to prime the pool and, mid-batch, the moment an env terminates -- so the
                scene always gets a distinct global episode number and the env never idles.
                """
                scenario_config = {
                    "training_episode": episode_number,
                    "max_steps": args.max_steps_per_episode,
                    "physics_frames_per_step": args.physics_frames_per_step,
                    "training_mode": True,
                }
                if reset_progress_max is not None:
                    scenario_config.update(reset_progress_min=0.0, reset_progress_max=reset_progress_max)
                env.configure(**scenario_config)
                obs, info = env.reset(
                    seed=args.episode_seed_multiplier * episode_number + env_idx)
                if args.multi_agent:
                    agent_count = len(env.agent_ids)
                    done_mask = np.asarray(info.get("per_agent_done", np.zeros((agent_count,), dtype=np.bool_)), dtype=np.bool_)
                    ep_reward = np.zeros((agent_count,), dtype=np.float32)
                    action_sum = np.zeros((agent_count, action_size), dtype=np.float32)
                    action_count = np.zeros((agent_count, 1), dtype=np.float32)
                    previous_action = np.full((agent_count, action_size), np.nan, dtype=np.float32)
                    action_delta_sum = np.zeros((agent_count, action_size), dtype=np.float32)
                    action_delta_count = np.zeros((agent_count, 1), dtype=np.float32)
                    max_track_progress = np.zeros((agent_count,), dtype=np.float32)
                    last_track_progress = np.zeros((agent_count,), dtype=np.float32)
                    finish_reached = np.zeros((agent_count,), dtype=np.bool_)
                    collision_seen = np.zeros((agent_count,), dtype=np.bool_)
                    collision_count = np.zeros((agent_count,), dtype=np.int32)
                    stalled_seen = np.zeros((agent_count,), dtype=np.bool_)
                    stalled_count = np.zeros((agent_count,), dtype=np.int32)
                    learner_mask = opponent_pool.learner_mask(
                        env.agent_team_ids,
                        opponent_teams,
                        episode_number,
                        env_idx,
                        use_current_policy=opponent_match.use_current_policy,
                    )
                else:
                    done_mask = None
                    ep_reward = 0.0
                    action_sum = np.zeros((action_size,), dtype=np.float32)
                    action_count = 0.0
                    previous_action = np.full((action_size,), np.nan, dtype=np.float32)
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
                return {
                    "env_idx": env_idx,
                    "obs": obs,
                    "done": False,
                    "done_mask": done_mask,
                    "ep_reward": ep_reward,
                    "action_sum": action_sum,
                    "action_count": action_count,
                    "previous_action": previous_action,
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
                }

            if not env_states:
                for env_idx, env in enumerate(envs):
                    env_states.append(start_env_episode(env, env_idx, sync_next_episode))
                    sync_next_episode += 1
            # Episodes closed during THIS batch; diagnostics summarise these, not a snapshot of
            # envs caught mid-episode.

            pending_resets = []
            for step_idx in episode_step_indices(batch_step_budget):
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
                            action = select_actions_with(
                                sync_sample_fn(actor),
                                state["obs"],
                                state["done_mask"],
                                action_low,
                                action_high,
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
                            action = select_action_with(
                                sync_sample_fn(actor),
                                state["obs"],
                                action_low,
                                action_high,
                            )
                            action = smooth_actions(
                                action,
                                state["previous_action"],
                                action_low,
                                action_high,
                                args.action_smoothing,
                            )

                    if args.multi_agent and opponent_match.model is not None:
                        opponent_action = select_actions_with(
                            sync_sample_fn(
                                opponent_match.model,
                                deterministic=True,
                            ),
                            state["obs"],
                            state["done_mask"],
                            action_low,
                            action_high,
                        )
                        action = opponent_pool.merge_actions(
                            action,
                            opponent_action,
                            state["learner_mask"],
                        )

                    step_requests.append((env, state, action))

                for env, state, action, step_result in stepper.step(step_requests):
                    next_obs, reward, terminated, truncated, info = step_result
                    # `done` ends the episode loop; the replay flag must carry `terminated` only,
                    # so a time-limit truncation still bootstraps instead of teaching
                    # the agent that the world ends at the step cap.
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
                                budget.consume(1)
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
                        state["ep_reward"] += float(reward)
                        state["action_sum"] += action
                        state["action_count"] += 1.0
                        if np.all(np.isfinite(state["previous_action"])):
                            state["action_delta_sum"] += np.abs(action - state["previous_action"])
                            state["action_delta_count"] += 1.0
                        state["previous_action"] = action

                    state["obs"] = next_obs
                    state["done"] = done
                    if done:
                        # Harvest now, reset after the whole step batch has been received:
                        # `stepper` still has in-flight replies for the other envs, and injecting a
                        # reset mid-drain would interleave it with them on the lockstep connection.
                        finished_states.append(dict(state))
                        sync_completed_episodes += 1
                        pending_resets.append(state["env_idx"])

                    if len(buffer) >= max(args.replay_warmup, args.batch_size):
                        update_policy = recovery_runtime.should_update_policy()
                        batch = buffer.sample(args.batch_size, action_dtype=np.float32)
                        losses = sac_learner(*batch, update_policy=update_policy)
                        critic1_losses.append(losses[1])
                        critic2_losses.append(losses[2])
                        warmup_completed = recovery_runtime.record_critic_update()
                        if update_policy:
                            actor_losses.append(losses[0])
                            alpha_losses.append(losses[3])
                            alphas.append(losses[4])
                        elif warmup_completed:
                            print(
                                f"Critic warmup complete after updates={recovery_runtime.critic_updates}; "
                                "actor and alpha will be unfrozen on the next update.",
                                flush=True,
                            )

                if len(buffer) >= max(args.replay_warmup, args.batch_size) and (step_idx + 1) % args.target_update_every == 0:
                    soft_update(target_critic1, critic1, args.tau)
                    soft_update(target_critic2, critic2, args.tau)

                for env_idx in pending_resets:
                    env_states[env_idx] = start_env_episode(
                        envs[env_idx], env_idx, sync_next_episode)
                    sync_next_episode += 1
                pending_resets.clear()
                if budget.exhausted:
                    break

            batch_elapsed = max(time.monotonic() - batch_started, 1e-6)
            batch_transitions = budget.collected - batch_transitions_start
            batch_env_steps = max(step_idx + 1, 1)
            if not finished_states:
                # No episode closed in this window: keep the accumulators running -- they are only
                # cleared once a report goes out -- and try again after the next window.
                continue
            entropy_gap = None
            mean_log_prob = None
            if len(buffer) >= args.batch_size:
                probe_obs = buffer.sample(args.batch_size, action_dtype=np.float32)[0]
                mean_log_prob = float(sync_log_prob_fn(
                    np.asarray(probe_obs, dtype=np.float32)).numpy())
                # Positive gap pushes alpha UP (policy sharper than the entropy target).
                entropy_gap = mean_log_prob + float(target_entropy)
            rewards_summary = [
                state["ep_reward"].tolist() if hasattr(state["ep_reward"], "tolist") else state["ep_reward"]
                for state in finished_states
            ]
            if args.multi_agent:
                mean_actions = [
                    (state["action_sum"] / np.maximum(state["action_count"], 1.0)).tolist()
                    for state in finished_states
                ]
            else:
                mean_actions = [
                    (state["action_sum"] / max(float(state["action_count"]), 1.0)).tolist()
                    for state in finished_states
                ]
            diagnostics = summarize_episode_diagnostics(finished_states, args.multi_agent)
            reward_stats = summarize_rewards(finished_states)
            mean_action_summary = summarize_actions(mean_actions)
            mean_action_delta_summary = summarize_action_deltas(finished_states, args.multi_agent)
            controlled_agents = sum(len(envs[state['env_idx']].agent_ids) if args.multi_agent else 1
                for state in finished_states)
            finish_rate = diagnostics["finishes"] / max(controlled_agents, 1)
            collision_rate = diagnostics["collisions"] / max(controlled_agents, 1)
            stall_rate = diagnostics["stalls"] / max(controlled_agents, 1)
            critic_warmup_remaining = recovery_runtime.warmup_left
            print_episode_metrics(sync_completed_episodes, [
                ("mode", [
                    ("exploration", "random" if use_random_exploration else "policy"),
                    ("alpha", f"{float(np.mean(alphas)) if alphas else float(tf.exp(log_alpha).numpy()):.4f}"),
                    ("opponent", opponent_match.label),
                    *(([("reset_progress_max", f"{reset_progress_max:.3f}")]) if reset_progress_max is not None else []),
                ]),
                ("outcome", [
                    ("reward", f"{reward_stats['mean']:.3f} [{reward_stats['min']:.3f}, {reward_stats['max']:.3f}]"),
                    ("progress", f"mean:{diagnostics['progress_mean']:.3f} max:{diagnostics['progress_max']:.3f}"),
                    ("pose_error", f"position:{diagnostics.get('position_error_m', 0.0):.4f}m orientation:{diagnostics.get('orientation_error_deg', 0.0):.1f}deg"),
                    ("hold", f"speed:{diagnostics.get('max_joint_speed', 0.0):.3f} frames:{diagnostics.get('hold_frames', 0)}"),
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
                    ("total_timesteps", budget.collected),
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    ("critic_updates", len(critic1_losses)),
                    ("policy_updates", len(actor_losses)),
                    ("warmup_left", critic_warmup_remaining),
                    ("actor_loss", f"{float(np.mean(actor_losses)) if actor_losses else 0.0:.5f}"),
                    ("critic_loss", f"{float(np.mean(critic1_losses)) if critic1_losses else 0.0:.5f}/{float(np.mean(critic2_losses)) if critic2_losses else 0.0:.5f}"),
                    ("alpha_loss", f"{float(np.mean(alpha_losses)) if alpha_losses else 0.0:.5f}"),
                ]),
                ("entropy", [
                    *(([("log_prob", f"{mean_log_prob:.3f}")]) if mean_log_prob is not None else []),
                    *(([("entropy_gap", f"{entropy_gap:.3f}")]) if entropy_gap is not None else []),
                ]),
                ("throughput", [
                    ("env_steps_s", f"{batch_transitions / batch_elapsed:.1f}"),
                    ("transitions_s", f"{batch_transitions / batch_elapsed:.1f}"),
                    ("transitions_step", f"{batch_transitions / batch_env_steps:.1f}"),
                    ("updates_s", f"{len(critic1_losses) / batch_elapsed:.1f}"),
                    ("episodes_batch", len(finished_states)),
                ]),
            ], args.log_format)
            if args.log_details:
                print(
                    f"episode={episode:04d} details rewards={rewards_summary} mean_actions={mean_actions}",
                    flush=True,
                )
            last_completed_episode = sync_completed_episodes
            snapshot_path = opponent_pool.snapshot(actor, sync_completed_episodes)
            if snapshot_path is not None:
                print(f"Saved opponent snapshot: {snapshot_path}", flush=True)

            apply_ready_best_checkpoint(best_tracker)
            if args.checkpoint_every > 0 and sync_completed_episodes - (last_saved_episode or start_episode) >= args.checkpoint_every:
                saved_path = save_training_checkpoint(
                    checkpoint,
                    checkpoint_manager,
                    buffer,
                    sync_completed_episodes,
                    args,
                )
                last_saved_episode = sync_completed_episodes
            else:
                saved_path = None
            if best_tracker.should_evaluate(sync_completed_episodes):
                if saved_path is None:
                    saved_path = save_training_checkpoint(
                        checkpoint, checkpoint_manager, buffer, sync_completed_episodes, args
                    )
                    last_saved_episode = sync_completed_episodes
                request_best_checkpoint_evaluation(best_tracker, saved_path, sync_completed_episodes)
            # A report just went out: start fresh accumulators for the next one.
            finished_states = []
            actor_losses = []
            critic1_losses = []
            critic2_losses = []
            alpha_losses = []
            alphas = []
            batch_started = time.monotonic()
            batch_transitions_start = budget.collected
            if budget.exhausted:
                print(
                    f"Transition budget reached: {budget.collected}/{budget.limit}",
                    flush=True,
                )
                break
            if sync_completed_episodes >= args.num_episodes:
                break

        completed_episode = last_completed_episode if last_completed_episode is not None else start_episode
        if last_saved_episode != completed_episode:
            save_training_checkpoint(
                checkpoint,
                checkpoint_manager,
                buffer,
                completed_episode,
                args,
                final=True,
            )
        # The evaluation requested on the last episode is still running; without this the
        # finally-block's close() cancels it and a final best can never be promoted.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
        actor.save_weights(args.actor_weights_path)
        save_critic_weights(critic1, critic2, args)
        print(f"Saved actor weights: {args.actor_weights_path}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupt received: saving the last consistent SAC state...", flush=True)
        if checkpoint is not None and checkpoint_manager is not None and buffer is not None and actor is not None:
            interrupted_episode = last_completed_episode if last_completed_episode is not None else start_episode
            save_training_checkpoint(
                checkpoint,
                checkpoint_manager,
                buffer,
                interrupted_episode,
                args,
            )
            actor.save_weights(args.actor_weights_path)
            if critic1 is not None and critic2 is not None:
                save_critic_weights(critic1, critic2, args)
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
    main()
