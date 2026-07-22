import argparse
import os
import random
from queue import Empty

import numpy as np

from algorithms.common import (
    continuous_exploration_bounds,
    create_continuous_async_worker,
    curriculum_reset_progress_max,
    describe_tensorflow_backend,
    format_float_list,
    load_demonstration_arrays,
    sample_exploratory_action,
    smooth_actions,
    soft_update,
    summarize_action_deltas,
    summarize_actions,
    summarize_episode_diagnostics,
    summarize_rewards,
    update_episode_diagnostics,
)
import tensorflow as tf

from core.models import build_continuous_critic, build_sac_actor
from core.policy_artifact import PolicyArtifactSaver, build_policy_metadata, load_policy_into_model
from core.opponent_pool import OpponentPool, add_opponent_pool_arguments, validate_team_layout
from core.replay_buffer import ReplayBuffer
from core.training import (
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
    apply_ready_best_checkpoint,
    add_collector_arguments,
    add_lockstep_tuning_arguments,
    add_parallel_env_arguments,
    add_log_format_argument,
    add_dashboard_arguments,
    maybe_start_dashboard,
    add_godot_render_argument,
    add_tensorflow_runtime_arguments,
    build_lockstep_user_args,
    episode_step_indices,
    print_episode_metrics,
    resolve_resume_checkpoint,
    restore_replay_buffer,
    save_replay_snapshot,
    validate_async_arguments,
)
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


LOG_2PI = np.log(2.0 * np.pi).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic SAC trainer for continuous Godot scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="Hard global gradient-norm cap for critic/actor updates (0 disables). Safety net "
        "against the deadly-triad Q-value divergence that otherwise blows critics up to NaN.",
    )
    parser.add_argument(
        "--grad-clip-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip each network at --grad-clip-k * EMA(gradient-norm), bounded by "
        "--grad-clip-norm. Tracks each network's gradient scale to reduce manual tuning; "
        "the hard cap remains the final safety bound.",
    )
    parser.add_argument(
        "--grad-clip-k",
        type=float,
        default=3.0,
        help="Multiplier on the running-mean gradient norm when --grad-clip-adaptive is set.",
    )
    parser.add_argument("--alpha-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--resume-actor-learning-rate",
        type=float,
        default=1e-5,
        help="Actor learning rate used after restoring a checkpoint; keeps a valid policy from drifting quickly.",
    )
    parser.add_argument(
        "--resume-alpha-learning-rate",
        type=float,
        default=1e-5,
        help="Entropy-temperature learning rate used after restoring a checkpoint.",
    )
    parser.add_argument("--initial-alpha", type=float, default=0.2)
    parser.add_argument(
        "--tune-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-tune the SAC entropy coefficient alpha toward --target-entropy. The tuner "
            "is unstable on some tasks (alpha runs away up or collapses); use --no-tune-alpha "
            "with --initial-alpha to hold it fixed."
        ),
    )
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--log-std-min", type=float, default=-20.0)
    parser.add_argument("--log-std-max", type=float, default=2.0)
    parser.add_argument("--action-smoothing", type=float, default=0.0)
    parser.add_argument("--random-exploration-episodes", type=int, default=15)
    parser.add_argument("--random-drive-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--random-steering-abs-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reset-progress-curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-progress-start-max", type=float, default=0.025)
    parser.add_argument("--reset-progress-end-max", type=float, default=0.35)
    parser.add_argument("--reset-progress-ramp-episodes", type=int, default=400)
    parser.add_argument("--replay-warmup", type=int, default=10000)
    parser.add_argument("--replay-capacity", type=int, default=200000)
    parser.add_argument(
        "--critic-warmup-updates",
        type=int,
        default=2000,
        help=(
            "On checkpoint resume, update only critics and target critics for this many "
            "gradient steps before unfreezing actor and alpha. Use 0 to disable."
        ),
    )
    parser.add_argument("--target-update-every", type=int, default=1)
    parser.add_argument(
        "--policy-update-every",
        type=int,
        default=2,
        help="Update actor and entropy temperature once every N critic updates.",
    )
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-weights-path", default="generic_sac_actor.weights.h5")
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Warm-start the actor from a .keras model, full .h5 model, or .weights.h5 file.",
    )
    parser.add_argument("--critic-weights-path", default=None)
    parser.add_argument("--critic1-weights-path", default="generic_sac_critic1.weights.h5")
    parser.add_argument("--critic2-weights-path", default="generic_sac_critic2.weights.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_sac")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Specific TensorFlow checkpoint to restore (for example ckpt-1025 or "
            "checkpoints/run/ckpt-1025). When omitted, --resume restores the latest "
            "checkpoint in --checkpoint-dir."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--best-checkpoint-window", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of rebuilding an empty replay buffer when resuming an old checkpoint.",
    )
    parser.add_argument("--demo-path", action="append", default=[])
    parser.add_argument("--demo-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-max-transitions", type=int, default=0)
    parser.add_argument("--demo-bc-epochs", type=int, default=0)
    parser.add_argument("--demo-bc-batch-size", type=int, default=128)
    parser.add_argument("--demo-bc-learning-rate", type=float, default=None)
    parser.add_argument(
        "--demo-bc-on-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run behavior cloning again after restoring a checkpoint.",
    )
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
    add_opponent_pool_arguments(parser)
    add_best_checkpoint_arguments(parser)
    add_parallel_env_arguments(parser)
    add_lockstep_tuning_arguments(parser)
    add_log_format_argument(parser)
    add_dashboard_arguments(parser)
    add_tensorflow_runtime_arguments(parser, include_compile_learner=True)
    add_godot_render_argument(parser)

    # Accepted for command compatibility with DDPG runs; SAC exploration is entropy-based.
    parser.add_argument("--exploration-noise", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--exploration-noise-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--exploration-noise-decay", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--exploration-noise-kind", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ou-theta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--actor-drive-prior", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--actor-steering-prior", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--actor-drive-regularization", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--actor-drive-target", type=float, default=None, help=argparse.SUPPRESS)
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


def build_sac_sample_fn(actor, obs_dim, action_low, action_high, log_std_min, log_std_max, device="/CPU:0"):
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
            std = tf.exp(log_std)
            pre_tanh = mean + std * tf.random.normal(tf.shape(mean))
            raw_action = tf.tanh(pre_tanh)
            return low + 0.5 * (raw_action + 1.0) * (high - low)

    return sample


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
    grad_clip_norm=0.0,
    grad_clip_adaptive=False,
    grad_clip_k=3.0,
    grad_clip_decay=0.99,
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


def pretrain_actor_behavior_cloning(actor, demo_data, epochs, batch_size, learning_rate, action_low, action_high):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    obs = demo_data["obs"]
    actions = demo_data["actions"]
    action_low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    raw_targets = 2.0 * (actions - action_low) / np.maximum(action_high - action_low, 1e-6) - 1.0
    raw_targets = np.clip(raw_targets, -0.995, 0.995).astype(np.float32)
    count = len(actions)

    for epoch in range(int(epochs)):
        order = np.random.permutation(count)
        losses = []
        for start in range(0, count, batch_size):
            idx = order[start:start + batch_size]
            batch_obs = tf.convert_to_tensor(obs[idx], dtype=tf.float32)
            batch_raw_targets = tf.convert_to_tensor(raw_targets[idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                mean, _ = actor(batch_obs, training=True)
                predicted = tf.tanh(mean)
                loss = tf.reduce_mean(tf.square(batch_raw_targets - predicted))
            grads = tape.gradient(loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            losses.append(float(loss.numpy()))

        print(
            f"demo_bc_epoch={epoch + 1:04d}/{epochs:04d} "
            f"actor_raw_mse={float(np.mean(losses)):.6f}",
            flush=True,
        )


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
):
    validate_async_arguments(args)
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
        grad_clip_norm=args.grad_clip_norm,
        grad_clip_adaptive=args.grad_clip_adaptive,
        grad_clip_k=args.grad_clip_k,
    )
    print(
        f"SAC learner: {'compiled graph' if args.tf_compile_learner else 'eager'}"
        f"{' + XLA' if args.tf_compile_learner and args.tf_xla else ''}"
        f" | grad-clip: {describe_grad_clip(args)}",
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
    rngs = [np.random.default_rng(args.env_seed_base + 100_003 * idx) for idx in range(len(envs))]

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
    policy_updates_total = 0
    critic_updates_since_resume = 0
    policy_update_candidates = 0
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
                step_events = scheduler.drain_step_events(pool, event)
                for step_event in step_events:
                    for transition in step_event.transitions:
                        buffer.add(*transition)
                updates_due = scheduler.ingest(step_events)
                updates_performed = 0
                if len(buffer) >= max(args.replay_warmup, args.batch_size):
                    for _update in range(updates_due):
                        warmup_complete = critic_updates_since_resume >= critic_warmup_target
                        update_policy = warmup_complete and policy_update_candidates % args.policy_update_every == 0
                        if warmup_complete:
                            policy_update_candidates += 1
                        batch = buffer.sample(args.batch_size, action_dtype=np.float32)
                        losses = sac_learner(*batch, update_policy=update_policy)
                        critic1_losses.append(losses[1])
                        critic2_losses.append(losses[2])
                        critic_updates_since_resume += 1
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
            warmup_left = max(0, critic_warmup_target - critic_updates_since_resume)
            throughput = scheduler.throughput(pool)
            print_episode_metrics(event.episode, [
                ("mode", [
                    ("collector", "async"),
                    ("worker", event.worker_id),
                    ("exploration", "random" if state["use_random_exploration"] else "policy"),
                    ("alpha", f"{float(tf.exp(log_alpha).numpy()):.4f}"),
                    *(([("reset_progress_max", f"{state['reset_progress_max']:.3f}")]) if state["reset_progress_max"] is not None else []),
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
                ("actions", [("mean", format_float_list(mean_action)), ("delta", format_float_list(mean_delta))]),
                ("training", [
                    ("completed", f"{completed}/{args.num_episodes}"),
                    ("queue", f"{throughput['queue_size']}/{throughput['queue_capacity']} ({throughput['queue_saturation']:.0%})"),
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    ("critic_updates", len(critic1_losses)),
                    ("policy_updates", len(actor_losses)),
                    ("warmup_left", warmup_left),
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


def main():
    args = parse_args()
    validate_async_arguments(args)
    if args.policy_update_every < 1:
        raise ValueError("--policy-update-every must be at least 1")
    best_tracker = BestCheckpointTracker(args, "sac")
    describe_tensorflow_backend(args)
    dashboard = maybe_start_dashboard(args, algorithm="sac")
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
        elif demo_data is not None and args.demo_prefill:
            print(
                "Skipped demonstration prefill because the checkpoint replay buffer was restored.",
                flush=True,
            )

        run_behavior_cloning = (
            demo_data is not None
            and args.demo_bc_epochs > 0
            and (resume_checkpoint is None or args.demo_bc_on_resume)
        )
        if run_behavior_cloning:
            pretrain_actor_behavior_cloning(
                actor,
                demo_data,
                epochs=args.demo_bc_epochs,
                batch_size=args.demo_bc_batch_size,
                learning_rate=args.demo_bc_learning_rate or args.actor_learning_rate,
                action_low=action_low,
                action_high=action_high,
            )
        elif demo_data is not None and args.demo_bc_epochs > 0 and resume_checkpoint is not None:
            print(
                "Skipped behavior cloning on resume; use --demo-bc-on-resume to request it explicitly.",
                flush=True,
            )

        critic_warmup_target = max(0, int(args.critic_warmup_updates)) if resume_checkpoint else 0
        critic_updates_since_resume = 0
        policy_update_candidates = 0
        if critic_warmup_target > 0:
            print(
                f"Resume critic warmup: updates={critic_warmup_target} actor=frozen alpha=frozen",
                flush=True,
            )

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
            grad_clip_norm=args.grad_clip_norm,
            grad_clip_adaptive=args.grad_clip_adaptive,
            grad_clip_k=args.grad_clip_k,
        )
        print(
            f"SAC learner: {'compiled graph' if args.tf_compile_learner else 'eager'}"
            f"{' + XLA' if args.tf_compile_learner and args.tf_xla else ''}"
            f" | grad-clip: {describe_grad_clip(args)}",
            flush=True,
        )

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
            critic1_losses = []
            critic2_losses = []
            alpha_losses = []
            alphas = []

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
                                args.log_std_min,
                                args.log_std_max,
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
                                args.log_std_min,
                                args.log_std_max,
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
                            args.log_std_min,
                            args.log_std_max,
                            deterministic=True,
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
                        warmup_complete = critic_updates_since_resume >= critic_warmup_target
                        update_policy = (
                            warmup_complete
                            and policy_update_candidates % args.policy_update_every == 0
                        )
                        if warmup_complete:
                            policy_update_candidates += 1
                        batch = buffer.sample(args.batch_size, action_dtype=np.float32)
                        losses = sac_learner(*batch, update_policy=update_policy)
                        critic1_losses.append(losses[1])
                        critic2_losses.append(losses[2])
                        critic_updates_since_resume += 1
                        if update_policy:
                            actor_losses.append(losses[0])
                            alpha_losses.append(losses[3])
                            alphas.append(losses[4])
                        elif critic_updates_since_resume == critic_warmup_target:
                            print(
                                f"Resume critic warmup complete after updates={critic_updates_since_resume}; "
                                "actor and alpha will be unfrozen on the next update.",
                                flush=True,
                            )

                if len(buffer) >= max(args.replay_warmup, args.batch_size) and (step_idx + 1) % args.target_update_every == 0:
                    soft_update(target_critic1, critic1, args.tau)
                    soft_update(target_critic2, critic2, args.tau)

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
                    ("alpha", f"{float(np.mean(alphas)) if alphas else float(tf.exp(log_alpha).numpy()):.4f}"),
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
                    ("critic_updates", len(critic1_losses)),
                    ("policy_updates", len(actor_losses)),
                    ("warmup_left", critic_warmup_remaining),
                    ("actor_loss", f"{float(np.mean(actor_losses)) if actor_losses else 0.0:.5f}"),
                    ("critic_loss", f"{float(np.mean(critic1_losses)) if critic1_losses else 0.0:.5f}/{float(np.mean(critic2_losses)) if critic2_losses else 0.0:.5f}"),
                    ("alpha_loss", f"{float(np.mean(alpha_losses)) if alpha_losses else 0.0:.5f}"),
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
                    checkpoint,
                    checkpoint_manager,
                    buffer,
                    episode + 1,
                    args,
                )
                last_saved_episode = episode + 1
            else:
                saved_path = None
            if best_tracker.should_evaluate(episode + 1):
                if saved_path is None:
                    saved_path = save_training_checkpoint(
                        checkpoint, checkpoint_manager, buffer, episode + 1, args
                    )
                    last_saved_episode = episode + 1
                request_best_checkpoint_evaluation(best_tracker, saved_path, episode + 1)

        if last_saved_episode != args.num_episodes:
            save_training_checkpoint(
                checkpoint,
                checkpoint_manager,
                buffer,
                args.num_episodes,
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
