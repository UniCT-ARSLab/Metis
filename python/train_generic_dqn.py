import argparse
import math
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
from models import build_greedy_action_fn, build_shared_q_network
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


SUCCESS_EVENTS = {"level_cleared", "target_reached", "finish_reached", "goal_scored", "success"}
SUCCESS_TERMINAL_REASONS = {"level_cleared", "target_reached", "finish_reached", "goal_scored", "success"}

# Share of the remaining episode budget a derived --epsilon-decay spends annealing down to
# epsilon-min; the rest of the run then exploits at the floor.
EPSILON_DECAY_HORIZON_FRACTION = 0.5
# Effective target for a derived decay when --epsilon-min is 0: multiplicative decay is
# asymptotic, so annealing to exactly 0 has no finite horizon.
EPSILON_DECAY_FLOOR = 0.01


def update_episode_diagnostics(state, agent_info, *, agent_idx=None, newly_done=False):
    events = agent_info.get("events", {})
    if isinstance(events, dict):
        for name, value in events.items():
            if isinstance(value, (bool, int, float, np.number)):
                state["event_counts"][str(name)] = state["event_counts"].get(str(name), 0.0) + float(value)

    terminal_reason = str(agent_info.get("terminal_reason", ""))
    if newly_done and terminal_reason:
        state["terminal_counts"][terminal_reason] = state["terminal_counts"].get(terminal_reason, 0) + 1

    successful_event = any(
        isinstance(events.get(name, 0.0), (bool, int, float, np.number))
        and float(events.get(name, 0.0)) > 0.0
        for name in SUCCESS_EVENTS
    ) if isinstance(events, dict) else False
    succeeded = (
        bool(agent_info.get("target_reached", False))
        or bool(agent_info.get("finish_reached", False))
        or terminal_reason in SUCCESS_TERMINAL_REASONS
        or successful_event
    )
    if agent_idx is None:
        state["success"] = bool(state["success"] or succeeded)
    elif succeeded:
        state["success"][agent_idx] = True


def merge_count_dicts(states, key, *, integer=False):
    merged = {}
    for state in states:
        for name, value in state[key].items():
            merged[name] = merged.get(name, 0) + value
    if integer:
        return {name: int(value) for name, value in sorted(merged.items()) if value}
    return {
        name: int(value) if float(value).is_integer() else round(float(value), 3)
        for name, value in sorted(merged.items())
        if value
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generic DQN trainer for Godot scenarios using BridgeServer.")
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-update-every", type=int, default=20)
    parser.add_argument("--replay-warmup", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=None,
        help=(
            "Per-episode multiplicative epsilon decay. When omitted it is derived from the "
            "episode budget so exploration reaches epsilon-min at "
            f"{int(EPSILON_DECAY_HORIZON_FRACTION * 100)}%% of the remaining run, which keeps "
            "the schedule anchored to --num-episodes instead of to a hand-tuned constant. "
            "Pass a value to pin it."
        ),
    )
    parser.add_argument(
        "--epsilon-decay-horizon-fraction",
        type=float,
        default=EPSILON_DECAY_HORIZON_FRACTION,
        help=(
            "Fraction of the remaining episodes over which a derived --epsilon-decay anneals "
            "epsilon down to epsilon-min. Ignored when --epsilon-decay is given."
        ),
    )
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-action-every", type=int, default=1)
    parser.add_argument("--weights-path", default="generic_dqn_weights.weights.h5")
    parser.add_argument("--initial-weights-path", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_dqn")
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
    parser.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default= False)
    add_collector_arguments(parser)
    add_parallel_env_arguments(parser)
    add_lockstep_tuning_arguments(parser)
    add_opponent_pool_arguments(parser)
    add_best_checkpoint_arguments(parser)
    add_log_format_argument(parser)
    add_tensorflow_runtime_arguments(parser, include_compile_learner=True)
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


def resolve_epsilon_decay(args, start_epsilon, start_episode):
    """Return the per-episode epsilon decay, deriving it when not pinned on the CLI.

    A hand-picked constant silently means different things as the episode budget or the
    episode length changes. Anchoring the schedule to the remaining episode budget keeps
    "epsilon is spent by X% of the run" true regardless of either.
    """
    if args.epsilon_decay is not None:
        return float(args.epsilon_decay)

    remaining = max(1, int(args.num_episodes) - int(start_episode))
    horizon = max(1.0, remaining * float(args.epsilon_decay_horizon_fraction))
    start = float(start_epsilon)
    # A multiplicative decay approaches zero asymptotically and never reaches it, so an
    # epsilon-min of 0 has no finite horizon. Anneal to a small floor instead; the caller
    # still clamps to the real epsilon-min.
    target = max(float(args.epsilon_min), EPSILON_DECAY_FLOOR)
    if start <= target:
        return 1.0
    return (target / start) ** (1.0 / horizon)


def episodes_to_reach_epsilon(start_epsilon, target, decay):
    """Episodes for start_epsilon * decay**n to fall to target, or None if it never does."""
    if start_epsilon <= target:
        return 0
    if decay >= 1.0 or decay <= 0.0:
        return None
    exact = math.log(target / start_epsilon) / math.log(decay)
    # A derived decay is built as (target/start)**(1/horizon), so `exact` is horizon up to
    # float error. Absorb that error before rounding up, or the reported episode overshoots
    # the horizon it was derived from by one.
    return int(math.ceil(exact - 1e-9))


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


def build_dqn_learner_step(model, target_model, optimizer, gamma, *, compiled=True):
    gamma = tf.constant(float(gamma), dtype=tf.float32)
    optimizer.build(model.trainable_variables)

    def update_one(obs, actions, rewards, next_obs, dones):
        next_q = target_model(next_obs, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets = tf.stop_gradient(rewards + (1.0 - dones) * gamma * max_next_q)

        with tf.GradientTape() as tape:
            q_values = model(obs, training=True)
            action_mask = tf.one_hot(actions, tf.shape(q_values)[-1], dtype=q_values.dtype)
            q_selected = tf.reduce_sum(q_values * action_mask, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_selected))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    if compiled:
        @tf.function(reduce_retracing=True)
        def update_batches(obs, actions, rewards, next_obs, dones):
            batch_count = tf.shape(obs)[0]
            losses = tf.TensorArray(tf.float32, size=batch_count)
            for batch_idx in tf.range(batch_count):
                losses = losses.write(
                    batch_idx,
                    update_one(
                        obs[batch_idx],
                        actions[batch_idx],
                        rewards[batch_idx],
                        next_obs[batch_idx],
                        dones[batch_idx],
                    ),
                )
            return losses.stack()
    else:
        def update_batches(obs, actions, rewards, next_obs, dones):
            losses = [
                update_one(obs[idx], actions[idx], rewards[idx], next_obs[idx], dones[idx])
                for idx in range(int(obs.shape[0]))
            ]
            return tf.stack(losses)

    def learner_step(buffer, batch_size, update_count=1):
        batches = buffer.sample_batches(update_count, batch_size)
        tensors = tuple(tf.convert_to_tensor(batch) for batch in batches)
        return np.asarray(update_batches(*tensors).numpy(), dtype=np.float32)

    return learner_step


def run_async_dqn(
    args,
    envs,
    model,
    target_model,
    optimizer,
    buffer,
    checkpoint,
    checkpoint_manager,
    best_checkpoint_manager,
    best_tracker,
    start_episode,
    start_epsilon,
    learner_step,
):
    validate_async_arguments(args)
    obs_dim = envs[0].obs_dim
    num_actions = envs[0].num_actions
    with tf.device("/CPU:0"):
        local_models = [
            build_shared_q_network(obs_dim=obs_dim, num_actions=num_actions)
            for _env in envs
        ]
    # One traced graph per collector, built once. sync_model only calls set_weights,
    # which mutates the variables these graphs already close over, so the traces stay
    # valid across policy syncs.
    greedy_actions = [
        build_greedy_action_fn(local_model, obs_dim)
        for local_model in local_models
    ]
    snapshot = PolicySnapshot(model.get_weights())
    rngs = [np.random.default_rng(args.env_seed_base + 100_003 * idx) for idx in range(len(envs))]

    def epsilon_for_episode(episode):
        elapsed = max(0, int(episode) - int(start_episode))
        return max(args.epsilon_min, float(start_epsilon) * (args.epsilon_decay ** elapsed))

    def begin_episode(worker_id, env, episode):
        env.configure(
            training_episode=episode,
            max_steps=args.max_steps_per_episode,
            physics_frames_per_step=args.physics_frames_per_step,
        )
        obs, info = env.reset(seed=args.episode_seed_multiplier * episode + worker_id)
        if args.multi_agent:
            agent_count = len(env.agent_ids)
            done_mask = np.asarray(
                info.get("per_agent_done", np.zeros((agent_count,), dtype=np.bool_)),
                dtype=np.bool_,
            )
            return {
                "obs": obs,
                "done": False,
                "done_mask": done_mask,
                "ep_reward": np.zeros((agent_count,), dtype=np.float32),
                "target_reached": np.zeros((agent_count,), dtype=np.bool_),
                "target_seen": np.zeros((agent_count,), dtype=np.bool_),
                "success": np.zeros((agent_count,), dtype=np.bool_),
                "episode_steps": np.zeros((agent_count,), dtype=np.int32),
                "event_counts": {},
                "terminal_counts": {},
                "action_counts": np.zeros((agent_count, num_actions), dtype=np.int32),
                "last_action": np.zeros((agent_count,), dtype=np.int32),
            }
        return {
            "obs": obs,
            "done": False,
            "done_mask": None,
            "ep_reward": 0.0,
            "target_reached": False,
            "target_seen": False,
            "success": False,
            "episode_steps": 0,
            "event_counts": {},
            "terminal_counts": {},
            "action_counts": np.zeros((num_actions,), dtype=np.int32),
            "last_action": 0,
        }

    def choose_action(worker_id, _env, episode, _step_idx, local_model, state):
        rng = rngs[worker_id]
        epsilon = epsilon_for_episode(episode)
        if args.multi_agent:
            observations = np.asarray(state["obs"], dtype=np.float32)
            q_values = local_model(observations, training=False).numpy()
            actions = np.argmax(q_values, axis=1).astype(np.int32)
            for agent_idx in range(len(actions)):
                if state["done_mask"][agent_idx]:
                    actions[agent_idx] = 0
                elif rng.random() < epsilon:
                    actions[agent_idx] = int(rng.integers(num_actions))
                state["action_counts"][agent_idx, actions[agent_idx]] += 1
                state["last_action"][agent_idx] = actions[agent_idx]
            return actions

        if rng.random() < epsilon:
            action = int(rng.integers(num_actions))
        else:
            obs_batch = np.expand_dims(state["obs"], axis=0).astype(np.float32)
            action = int(greedy_actions[worker_id](obs_batch).numpy()[0])
        state["action_counts"][action] += 1
        state["last_action"] = action
        return action

    def process_step(_worker_id, env, _episode, _step_idx, state, action, step_result):
        next_obs, reward, terminated, truncated, info = step_result
        # `done` ends the episode loop; `terminated` is what the TD target keys off. A
        # time-limit truncation cuts an episode that was still going, so bootstrapping
        # must continue through it -- treating it as terminal teaches the agent that the
        # world ends at the step cap.
        done = bool(terminated or truncated)
        transitions = []
        if args.multi_agent:
            rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
            done_mask = np.asarray(info.get("per_agent_done"), dtype=np.bool_)
            terminated_mask = np.asarray(
                info.get("per_agent_terminated", done_mask), dtype=np.bool_
            )
            infos = info.get("per_agent_infos", [{} for _agent in env.agent_ids])
            for agent_idx in range(len(env.agent_ids)):
                agent_info = infos[agent_idx] if agent_idx < len(infos) else {}
                was_done = bool(state["done_mask"][agent_idx])
                if not was_done:
                    state["episode_steps"][agent_idx] += 1
                update_episode_diagnostics(
                    state,
                    agent_info,
                    agent_idx=agent_idx,
                    newly_done=bool(done_mask[agent_idx] and not was_done),
                )
                state["target_reached"][agent_idx] |= bool(agent_info.get("target_reached", False))
                state["target_seen"][agent_idx] |= bool(agent_info.get("target_first_seen", False))
                if not (was_done and done_mask[agent_idx]):
                    transitions.append((
                        state["obs"][agent_idx],
                        int(action[agent_idx]),
                        float(rewards[agent_idx]),
                        next_obs[agent_idx],
                        bool(terminated_mask[agent_idx] or terminated),
                    ))
                    state["ep_reward"][agent_idx] += rewards[agent_idx]
            state["done_mask"] = done_mask
        else:
            agent_info = info.get("agent_info", {})
            state["episode_steps"] += 1
            update_episode_diagnostics(state, agent_info, newly_done=done)
            state["target_reached"] |= bool(agent_info.get("target_reached", False))
            state["target_seen"] |= bool(agent_info.get("target_first_seen", False))
            transitions.append((state["obs"], int(action), float(reward), next_obs, bool(terminated)))
            state["ep_reward"] += float(reward)
            # Scenario progress (0..1). Logged because it is the only scale-invariant
            # measure of how well the agent plays: episode reward cannot be compared
            # across reward-tuning experiments, progress can.
            state["progress"] = float(agent_info.get("progress", state.get("progress", 0.0)))
        state["obs"] = next_obs
        state["done"] = done
        return transitions

    worker = build_async_worker(
        local_models,
        snapshot,
        args.max_steps_per_episode,
        args.async_policy_sync_steps,
        begin_episode,
        choose_action,
        process_step,
        lambda _worker_id, _env, _episode, state: state,
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
    update_count = 0
    losses_since_log = []
    last_saved_episode = None
    interrupted = False
    pool.start()
    try:
        while done_workers < len(envs):
            apply_ready_best_checkpoint(
                best_tracker, best_checkpoint_manager, checkpoint, buffer, completed, epsilon_for_episode(completed), args
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
                    remaining_updates = updates_due
                    while remaining_updates > 0:
                        until_publish = (
                            args.async_policy_publish_updates
                            - update_count % args.async_policy_publish_updates
                        )
                        chunk_size = min(remaining_updates, until_publish)
                        chunk_losses = learner_step(buffer, args.batch_size, chunk_size)
                        losses_since_log.extend(chunk_losses.tolist())
                        update_count += chunk_size
                        updates_performed += chunk_size
                        remaining_updates -= chunk_size
                        if update_count % args.async_policy_publish_updates == 0:
                            snapshot.publish(model.get_weights())
                scheduler.record_updates(updates_performed)
                continue

            if not isinstance(event, AsyncEpisodeEvent):
                continue
            completed += 1
            state = event.payload
            if completed % args.target_update_every == 0:
                target_model.set_weights(model.get_weights())
            epsilon = epsilon_for_episode(completed)
            rewards = state["ep_reward"].tolist() if hasattr(state["ep_reward"], "tolist") else state["ep_reward"]
            if args.multi_agent:
                metric_count = len(state["success"])
                success_total = int(np.sum(state["success"]))
                episode_steps = state["episode_steps"]
            else:
                metric_count = 1
                success_total = int(state["success"])
                episode_steps = np.asarray([state["episode_steps"]], dtype=np.int32)
            mean_loss = float(np.mean(losses_since_log)) if losses_since_log else 0.0
            throughput = scheduler.throughput(pool)
            print_episode_metrics(event.episode, [
                ("mode", [("collector", "async"), ("worker", event.worker_id), ("epsilon", f"{epsilon:.3f}")]),
                ("outcome", [
                    ("reward", rewards),
                    ("progress", f"{float(state.get('progress', 0.0)):.4f}"),
                    ("success", f"{success_total}/{metric_count} ({success_total / max(metric_count, 1):.2%})"),
                    ("steps", f"mean:{float(np.mean(episode_steps)):.1f} range:[{int(np.min(episode_steps))},{int(np.max(episode_steps))}]"),
                ]),
                ("training", [
                    ("completed", f"{completed}/{args.num_episodes}"),
                    ("queue", f"{throughput['queue_size']}/{throughput['queue_capacity']} ({throughput['queue_saturation']:.0%})"),
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    ("updates", len(losses_since_log)),
                    ("loss", f"{mean_loss:.5f}"),
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
            losses_since_log.clear()

            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                saved_path = save_training_checkpoint(
                    checkpoint, checkpoint_manager, buffer, completed, epsilon, args
                )
                last_saved_episode = completed
            else:
                saved_path = None
            if best_tracker.should_evaluate(completed):
                if saved_path is None:
                    saved_path = save_training_checkpoint(
                        checkpoint, checkpoint_manager, buffer, completed, epsilon, args
                    )
                    last_saved_episode = completed
                request_best_checkpoint_evaluation(best_tracker, saved_path, completed)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupt received: stopping async DQN collectors...", flush=True)
    finally:
        pool.close()

    epsilon = epsilon_for_episode(completed)
    snapshot.publish(model.get_weights())
    if last_saved_episode != completed:
        save_training_checkpoint(
            checkpoint, checkpoint_manager, buffer, completed, epsilon, args, final=True
        )
    if interrupted:
        print(
            f"Interrupted async training saved at completed_episodes={completed}",
            flush=True,
        )
    return completed, epsilon


def save_training_checkpoint(
    checkpoint,
    checkpoint_manager,
    buffer,
    episode,
    epsilon,
    args,
    final=False,
    save_replay=True,
):
    checkpoint.episode.assign(episode)
    checkpoint.epsilon.assign(epsilon)
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


def apply_ready_best_checkpoint(tracker, best_checkpoint_manager, checkpoint, buffer, current_episode, epsilon, args):
    result = tracker.poll_ready()
    if result is None or not tracker.is_improvement(result):
        return None
    best_path = save_training_checkpoint(
        checkpoint,
        best_checkpoint_manager,
        buffer,
        current_episode,
        epsilon,
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
    validate_async_arguments(args)
    best_tracker = BestCheckpointTracker(args, "dqn")
    describe_tensorflow_backend(args)
    random.seed(args.env_seed_base)
    np.random.seed(args.env_seed_base)
    tf.random.set_seed(args.env_seed_base)

    ports = [args.base_port + i for i in range(args.num_envs)]
    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
        scene_path=args.godot_scene
    )
    envs = []
    model = None
    buffer = None
    checkpoint = None
    checkpoint_manager = None
    best_checkpoint_manager = None
    stepper = None
    start_episode = 0
    last_completed_episode = None
    epsilon = args.epsilon_start
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
            f"Scenario spec: agent_id={agent_id} {envs[0].agent_summary()} multi_agent={args.multi_agent} "
            f"{envs[0].team_summary()} obs_dim={obs_dim} num_actions={num_actions} "
            f"actions={envs[0].action_names}",
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
        if best_tracker.enabled:
            best_checkpoint_manager = tf.train.CheckpointManager(
                checkpoint,
                directory=str(best_tracker.directory),
                max_to_keep=args.keep_best_checkpoints,
            )
        resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_manager)
        restored_replay_count = 0
        if resume_checkpoint and args.initial_weights_path:
            raise RuntimeError("--initial-weights-path cannot be combined with --resume or --resume-checkpoint")
        if resume_checkpoint:
            checkpoint.restore(resume_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            epsilon = float(checkpoint.epsilon.numpy())
            restored_checkpoint = True
            print(
                f"Resumed checkpoint {resume_checkpoint} from episode={start_episode} epsilon={epsilon:.3f}",
                flush=True,
            )
            restored_replay_count = restore_replay_buffer(args, resume_checkpoint, buffer)
        elif args.initial_weights_path:
            initial_weights_path = Path(args.initial_weights_path)
            if not initial_weights_path.is_file():
                raise RuntimeError(f"Initial DQN weights not found: {initial_weights_path}")
            try:
                model.load_weights(str(initial_weights_path))
            except ValueError as exc:
                raise RuntimeError(
                    f"Initial DQN weights are incompatible with the current scenario "
                    f"(obs_dim={obs_dim}, num_actions={num_actions}): {initial_weights_path}"
                ) from exc
            target_model.set_weights(model.get_weights())
            print(
                f"Warm-started DQN policy from weights: {initial_weights_path} "
                f"(fresh replay, optimizer, epsilon={epsilon:.3f}, episode=0)",
                flush=True,
            )
        optimizer.learning_rate.assign(args.learning_rate)

        # Resolved here, not at parse time: on --resume the horizon must span the episodes
        # that are actually left, starting from the epsilon the checkpoint carried.
        derived_epsilon_decay = args.epsilon_decay is None
        args.epsilon_decay = resolve_epsilon_decay(args, epsilon, start_episode)
        origin = "derived" if derived_epsilon_decay else "pinned via --epsilon-decay"
        floor = max(float(args.epsilon_min), EPSILON_DECAY_FLOOR)
        reached = episodes_to_reach_epsilon(epsilon, floor, args.epsilon_decay)
        horizon = "never" if reached is None else f"episode {start_episode + reached}/{args.num_episodes}"
        print(
            f"Epsilon schedule: start={epsilon:.3f} min={args.epsilon_min:.3f} "
            f"decay={args.epsilon_decay:.6f} ({origin}); reaches {floor:.3f} at {horizon}",
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

        opponent_teams = validate_team_layout(envs, args.opponent_pool)
        opponent_pool = OpponentPool(
            args,
            algorithm="dqn",
            model_factory=lambda: build_shared_q_network(obs_dim=obs_dim, num_actions=num_actions),
            metadata={"obs_dim": obs_dim, "num_actions": num_actions},
        )
        if opponent_pool.enabled:
            print(
                f"Opponent pool: dir={opponent_pool.directory} teams={opponent_teams} "
                f"snapshots={len(opponent_pool.entries)} sampling={opponent_pool.sampling}",
                flush=True,
            )

        learner_step = build_dqn_learner_step(
            model,
            target_model,
            optimizer,
            args.gamma,
            compiled=args.tf_compile_learner,
        )
        print(
            f"DQN learner: {'compiled batched graph' if args.tf_compile_learner else 'eager'}",
            flush=True,
        )

        if args.collector_mode == "async":
            last_completed_episode, epsilon = run_async_dqn(
                args,
                envs,
                model,
                target_model,
                optimizer,
                buffer,
                checkpoint,
                checkpoint_manager,
                best_checkpoint_manager,
                best_tracker,
                start_episode,
                epsilon,
                learner_step,
            )
            model.save_weights(args.weights_path)
            print(f"Saved weights: {args.weights_path}", flush=True)
            return

        last_saved_episode = None
        for episode in range(start_episode, args.num_episodes):
            opponent_match = opponent_pool.start_episode(model, episode)
            env_states = []
            for env_idx, env in enumerate(envs):
                env.configure(
                    training_episode=episode,
                    max_steps=args.max_steps_per_episode,
                    physics_frames_per_step=args.physics_frames_per_step,
                )
                obs, info = env.reset(seed=args.episode_seed_multiplier * episode + env_idx)
                if args.multi_agent:
                    done_mask = np.asarray(info.get("per_agent_done", np.zeros((len(env.agent_ids),), dtype=np.bool_)), dtype=np.bool_)
                    ep_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)
                    target_reached = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    target_seen = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    success = np.zeros((len(env.agent_ids),), dtype=np.bool_)
                    episode_steps = np.zeros((len(env.agent_ids),), dtype=np.int32)
                    action_counts = np.zeros((len(env.agent_ids), num_actions), dtype=np.int32)
                    last_action = np.zeros((len(env.agent_ids),), dtype=np.int32)
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
                    target_reached = False
                    target_seen = False
                    success = False
                    episode_steps = 0
                    action_counts = np.zeros((num_actions,), dtype=np.int32)
                    last_action = 0
                    learner_mask = None
                env_states.append({
                    "obs": obs,
                    "done": False,
                    "done_mask": done_mask,
                    "ep_reward": ep_reward,
                    "target_reached": target_reached,
                    "target_seen": target_seen,
                    "success": success,
                    "episode_steps": episode_steps,
                    "event_counts": {},
                    "terminal_counts": {},
                    "action_counts": action_counts,
                    "last_action": last_action,
                    "learner_mask": learner_mask,
                })

            losses = []
            for step_idx in episode_step_indices(args.max_steps_per_episode):
                if all(state["done"] for state in env_states):
                    break

                step_requests = []
                for env, state in zip(envs, env_states):
                    if state["done"]:
                        continue

                    if args.multi_agent:
                        action = select_actions(model, state["obs"], state["done_mask"], epsilon, num_actions)
                        if opponent_match.model is not None:
                            opponent_action = select_actions(
                                opponent_match.model,
                                state["obs"],
                                state["done_mask"],
                                0.0,
                                num_actions,
                            )
                            action = opponent_pool.merge_actions(
                                action,
                                opponent_action,
                                state["learner_mask"],
                            )
                        for agent_idx, action_id in enumerate(action):
                            state["action_counts"][agent_idx, int(action_id)] += 1
                            state["last_action"][agent_idx] = int(action_id)
                    else:
                        action = select_action(model, state["obs"], epsilon, num_actions)
                        state["action_counts"][int(action)] += 1
                        state["last_action"] = int(action)

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
                        per_agent_infos = info.get("per_agent_infos", [{} for _ in env.agent_ids])

                        for agent_idx in range(len(env.agent_ids)):
                            agent_info = per_agent_infos[agent_idx] if agent_idx < len(per_agent_infos) else {}
                            was_done = bool(state["done_mask"][agent_idx])
                            if not was_done:
                                state["episode_steps"][agent_idx] += 1
                            update_episode_diagnostics(
                                state,
                                agent_info,
                                agent_idx=agent_idx,
                                newly_done=bool(per_agent_done[agent_idx] and not was_done),
                            )
                            if bool(agent_info.get("target_reached", False)):
                                state["target_reached"][agent_idx] = True
                            if bool(agent_info.get("target_first_seen", False)):
                                state["target_seen"][agent_idx] = True

                            if state["done_mask"][agent_idx] and per_agent_done[agent_idx]:
                                continue
                            if state["learner_mask"][agent_idx]:
                                buffer.add(
                                    state["obs"][agent_idx],
                                    int(action[agent_idx]),
                                    float(per_agent_rewards[agent_idx]),
                                    next_obs[agent_idx],
                                    bool(per_agent_terminated[agent_idx] or terminated),
                                )
                            state["ep_reward"][agent_idx] += per_agent_rewards[agent_idx]

                        state["done_mask"] = per_agent_done
                    else:
                        agent_info = info.get("agent_info", {})
                        state["episode_steps"] += 1
                        update_episode_diagnostics(state, agent_info, newly_done=done)
                        if bool(agent_info.get("target_reached", False)):
                            state["target_reached"] = True
                        if bool(agent_info.get("target_first_seen", False)):
                            state["target_seen"] = True

                        buffer.add(
                            state["obs"],
                            int(action),
                            float(reward),
                            next_obs,
                            bool(terminated),
                        )
                        state["ep_reward"] += float(reward)

                    state["obs"] = next_obs
                    state["done"] = done

                    if len(buffer) >= max(args.replay_warmup, args.batch_size):
                        loss = float(learner_step(buffer, args.batch_size, 1)[0])
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
                success_total = sum(int(np.sum(state["success"])) for state in env_states)
                episode_steps = np.concatenate([state["episode_steps"] for state in env_states])
            else:
                reached_summary = [int(state["target_reached"]) for state in env_states]
                seen_summary = [int(state["target_seen"]) for state in env_states]
                reached_total = sum(int(state["target_reached"]) for state in env_states)
                seen_total = sum(int(state["target_seen"]) for state in env_states)
                metric_count = len(env_states)
                success_total = sum(int(state["success"]) for state in env_states)
                episode_steps = np.asarray([state["episode_steps"] for state in env_states], dtype=np.int32)
            success_rate = success_total / max(metric_count, 1)
            seen_rate = seen_total / max(metric_count, 1)
            event_counts = merge_count_dicts(env_states, "event_counts")
            terminal_counts = merge_count_dicts(env_states, "terminal_counts", integer=True)
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
            outcome_metrics = [
                ("reward", rewards_summary),
                ("success", f"{success_total}/{metric_count} ({success_rate:.2%})"),
                (
                    "steps",
                    f"mean:{float(np.mean(episode_steps)):.1f} "
                    f"range:[{int(np.min(episode_steps))},{int(np.max(episode_steps))}]",
                ),
            ]
            if event_counts:
                outcome_metrics.append(("events", event_counts))
            if terminal_counts:
                outcome_metrics.append(("terminal", terminal_counts))
            if seen_total > 0:
                outcome_metrics.append(("seen_rate", f"{seen_rate:.3f}"))
            print_episode_metrics(episode, [
                ("mode", [
                    ("epsilon", f"{epsilon:.3f}"),
                    ("opponent", opponent_match.label),
                ]),
                ("outcome", outcome_metrics),
                ("training", [
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    ("updates", len(losses)),
                    ("loss", f"{mean_loss:.5f}"),
                ]),
            ], args.log_format)
            if args.log_action_every > 0 and episode % args.log_action_every == 0:
                print(
                    f"  actions   counts={action_summary}  last={last_action_summary}",
                    flush=True,
                )
            last_completed_episode = episode + 1
            snapshot_path = opponent_pool.snapshot(model, episode + 1)
            if snapshot_path is not None:
                print(f"Saved opponent snapshot: {snapshot_path}", flush=True)

            apply_ready_best_checkpoint(
                best_tracker, best_checkpoint_manager, checkpoint, buffer, episode + 1, epsilon, args
            )
            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                saved_path = save_training_checkpoint(
                    checkpoint, checkpoint_manager, buffer, episode + 1, epsilon, args
                )
                last_saved_episode = episode + 1
            else:
                saved_path = None
            if best_tracker.should_evaluate(episode + 1):
                if saved_path is None:
                    saved_path = save_training_checkpoint(
                        checkpoint, checkpoint_manager, buffer, episode + 1, epsilon, args
                    )
                    last_saved_episode = episode + 1
                request_best_checkpoint_evaluation(best_tracker, saved_path, episode + 1)

        if last_saved_episode != args.num_episodes:
            save_training_checkpoint(checkpoint, checkpoint_manager, buffer, args.num_episodes, epsilon, args, final=True)
        model.save_weights(args.weights_path)
        print(f"Saved weights: {args.weights_path}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupt received: saving the last consistent DQN state...", flush=True)
        if checkpoint is not None and checkpoint_manager is not None and buffer is not None and model is not None:
            interrupted_episode = last_completed_episode if last_completed_episode is not None else start_episode
            save_training_checkpoint(
                checkpoint, checkpoint_manager, buffer, interrupted_episode, epsilon, args
            )
            model.save_weights(args.weights_path)
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
