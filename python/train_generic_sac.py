import argparse
import os
import random

import numpy as np

from train_generic_ddpg import (
    continuous_exploration_bounds,
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

from godot_process_manager import GodotProcessManager
from models import build_continuous_critic, build_sac_actor
from replay_buffer import ReplayBuffer
from scenario_gym_env import ScenarioGymEnv
from training_support import (
    add_log_format_argument,
    print_episode_metrics,
    resolve_resume_checkpoint,
    restore_replay_buffer,
    save_replay_snapshot,
)


LOG_2PI = np.log(2.0 * np.pi).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic SAC trainer for continuous Godot scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha-learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-alpha", type=float, default=0.2)
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
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-weights-path", default="generic_sac_actor.weights.h5")
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
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-details", action=argparse.BooleanOptionalAction, default=False)
    add_log_format_argument(parser)

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


def select_actions(actor, obs_batch, done_mask, action_low, action_high, log_std_min, log_std_max):
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
        deterministic=False,
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


def train_step(
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
    batch_size,
    gamma,
    action_low,
    action_high,
    log_std_min,
    log_std_max,
    update_policy=True,
):
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, action_dtype=np.float32)
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards.reshape(-1, 1), dtype=tf.float32)
    next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones.reshape(-1, 1), dtype=tf.float32)
    action_low_tensor = tf.convert_to_tensor(action_low.reshape(1, -1), dtype=tf.float32)
    action_high_tensor = tf.convert_to_tensor(action_high.reshape(1, -1), dtype=tf.float32)

    alpha = tf.exp(log_alpha)
    next_actions, next_log_prob, _ = sample_actor(
        actor,
        next_obs,
        action_low_tensor,
        action_high_tensor,
        log_std_min,
        log_std_max,
    )
    target_q1 = target_critic1([next_obs, next_actions], training=False)
    target_q2 = target_critic2([next_obs, next_actions], training=False)
    target_q = tf.minimum(target_q1, target_q2) - alpha * next_log_prob
    y = rewards + (1.0 - dones) * gamma * target_q

    with tf.GradientTape() as tape:
        q1 = critic1([obs, actions], training=True)
        critic1_loss = tf.reduce_mean(tf.square(tf.stop_gradient(y) - q1))
    critic1_grads = tape.gradient(critic1_loss, critic1.trainable_variables)
    critic1_optimizer.apply_gradients(zip(critic1_grads, critic1.trainable_variables))

    with tf.GradientTape() as tape:
        q2 = critic2([obs, actions], training=True)
        critic2_loss = tf.reduce_mean(tf.square(tf.stop_gradient(y) - q2))
    critic2_grads = tape.gradient(critic2_loss, critic2.trainable_variables)
    critic2_optimizer.apply_gradients(zip(critic2_grads, critic2.trainable_variables))

    actor_loss_value = None
    alpha_loss_value = None
    if update_policy:
        with tf.GradientTape() as tape:
            policy_actions, log_prob, _ = sample_actor(
                actor,
                obs,
                action_low_tensor,
                action_high_tensor,
                log_std_min,
                log_std_max,
            )
            q1_pi = critic1([obs, policy_actions], training=False)
            q2_pi = critic2([obs, policy_actions], training=False)
            q_pi = tf.minimum(q1_pi, q2_pi)
            actor_loss = tf.reduce_mean(alpha * log_prob - q_pi)
        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
        actor_loss_value = float(actor_loss.numpy())

        with tf.GradientTape() as tape:
            _, log_prob, _ = sample_actor(
                actor,
                obs,
                action_low_tensor,
                action_high_tensor,
                log_std_min,
                log_std_max,
            )
            alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_prob + target_entropy))
        alpha_grads = tape.gradient(alpha_loss, [log_alpha])
        alpha_optimizer.apply_gradients(zip(alpha_grads, [log_alpha]))
        alpha_loss_value = float(alpha_loss.numpy())

    return (
        actor_loss_value,
        float(critic1_loss.numpy()),
        float(critic2_loss.numpy()),
        alpha_loss_value,
        float(tf.exp(log_alpha).numpy()),
    )


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


def save_training_checkpoint(checkpoint, checkpoint_manager, buffer, checkpoint_number_value, args, final=False):
    checkpoint.episode.assign(checkpoint_number_value)
    saved_path = checkpoint_manager.save(checkpoint_number=checkpoint_number_value)
    label = "final checkpoint" if final else "checkpoint"
    print(f"Saved {label}: {saved_path}", flush=True)
    if args.save_replay_buffer and len(buffer) > 0:
        save_replay_snapshot(saved_path, checkpoint_manager, buffer)
    return saved_path


def apply_optimizer_learning_rates(actor_optimizer, critic1_optimizer, critic2_optimizer, alpha_optimizer, args):
    actor_optimizer.learning_rate.assign(args.actor_learning_rate)
    critic1_optimizer.learning_rate.assign(args.critic_learning_rate)
    critic2_optimizer.learning_rate.assign(args.critic_learning_rate)
    alpha_optimizer.learning_rate.assign(args.alpha_learning_rate)
    print(
        "Optimizer learning rates: "
        f"actor={args.actor_learning_rate:g} critic={args.critic_learning_rate:g} "
        f"alpha={args.alpha_learning_rate:g}",
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
    envs = []
    actor = None
    critic1 = None
    critic2 = None
    buffer = None
    checkpoint = None
    checkpoint_manager = None
    start_episode = 0
    last_completed_episode = None
    try:
        manager.start_many(ports, headless=args.headless, debug=args.godot_debug)
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

        env0 = envs[0]
        if env0.action_type != "continuous":
            raise RuntimeError(f"train_generic_sac.py requires action_type='continuous', got {env0.action_type!r}")

        obs_dim = env0.obs_dim
        action_size = env0.action_size
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
            f"obs_dim={obs_dim} action_size={action_size} action_names={env0.action_names} "
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
        if resume_checkpoint:
            checkpoint.restore(resume_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            print(
                f"Resumed checkpoint {resume_checkpoint} from episode={start_episode} "
                f"alpha={float(tf.exp(log_alpha).numpy()):.4f}",
                flush=True,
            )
            restored_replay_count = restore_replay_buffer(args, resume_checkpoint, buffer)

        apply_optimizer_learning_rates(
            actor_optimizer,
            critic1_optimizer,
            critic2_optimizer,
            alpha_optimizer,
            args,
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
        if critic_warmup_target > 0:
            print(
                f"Resume critic warmup: updates={critic_warmup_target} actor=frozen alpha=frozen",
                flush=True,
            )

        last_saved_episode = None
        for episode in range(start_episode, args.num_episodes):
            use_random_exploration = episode < max(0, args.random_exploration_episodes)
            reset_progress_max = curriculum_reset_progress_max(episode, args)
            env_states = []
            for env_idx, env in enumerate(envs):
                if reset_progress_max is not None:
                    env.configure(reset_progress_min=0.0, reset_progress_max=reset_progress_max)
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
                })

            actor_losses = []
            critic1_losses = []
            critic2_losses = []
            alpha_losses = []
            alphas = []

            for step_idx in range(args.max_steps_per_episode):
                if all(state["done"] for state in env_states):
                    break

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

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)

                    if args.multi_agent:
                        per_agent_rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
                        per_agent_done = np.asarray(info.get("per_agent_done"), dtype=np.bool_)
                        per_agent_infos = list(info.get("per_agent_infos", []))
                        for agent_idx in range(len(env.agent_ids)):
                            if state["done_mask"][agent_idx] and per_agent_done[agent_idx]:
                                continue
                            agent_info = per_agent_infos[agent_idx] if agent_idx < len(per_agent_infos) else {}
                            update_episode_diagnostics(state, agent_info, agent_idx=agent_idx)
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
                            previous_action = state["previous_action"][agent_idx]
                            if np.all(np.isfinite(previous_action)):
                                state["action_delta_sum"][agent_idx] += np.abs(action[agent_idx] - previous_action)
                                state["action_delta_count"][agent_idx, 0] += 1.0
                            state["previous_action"][agent_idx] = action[agent_idx]
                        state["done_mask"] = per_agent_done
                    else:
                        update_episode_diagnostics(state, info.get("agent_info", {}))
                        buffer.add(state["obs"], action, float(reward), next_obs, done)
                        state["ep_reward"] += float(reward)
                        state["action_sum"] += action
                        state["action_count"] += 1.0
                        if np.all(np.isfinite(state["previous_action"])):
                            state["action_delta_sum"] += np.abs(action - state["previous_action"])
                            state["action_delta_count"] += 1.0
                        state["previous_action"] = action

                    state["obs"] = next_obs
                    state["done"] = done

                    if len(buffer) >= args.replay_warmup:
                        update_policy = critic_updates_since_resume >= critic_warmup_target
                        losses = train_step(
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
                            args.batch_size,
                            args.gamma,
                            action_low,
                            action_high,
                            args.log_std_min,
                            args.log_std_max,
                            update_policy=update_policy,
                        )
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

                if len(buffer) >= args.replay_warmup and (step_idx + 1) % args.target_update_every == 0:
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

            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                save_training_checkpoint(
                    checkpoint,
                    checkpoint_manager,
                    buffer,
                    episode + 1,
                    args,
                )
                last_saved_episode = episode + 1

        if last_saved_episode != args.num_episodes:
            save_training_checkpoint(
                checkpoint,
                checkpoint_manager,
                buffer,
                args.num_episodes,
                args,
                final=True,
            )
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
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        manager.stop_all()


if __name__ == "__main__":
    main()
