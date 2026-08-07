import argparse
import math
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


def replay_warmup_threshold(args):
    return max(int(args.replay_warmup), int(args.batch_size))

from core.curriculum import scenario_curriculum_config
from core.models import (
    build_greedy_action_fn,
    build_shared_q_network,
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
from core.opponent_pool import (
    OpponentMatch,
    OpponentPool,
    add_opponent_pool_arguments,
    validate_team_layout,
)
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
)
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


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
    add_training_budget_argument(parser)
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
    parser.add_argument(
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
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    add_multi_policy_arguments(parser)
    parser.add_argument("--log-action-every", type=int, default=1)
    parser.add_argument("--weights-path", default="generic_dqn_weights.weights.h5")
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Warm-start the policy from a .keras model, full .h5 model, or .weights.h5 file.",
    )
    parser.add_argument("--initial-weights-path", default=None, help=argparse.SUPPRESS)
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
    add_training_health_arguments(parser)
    add_log_format_argument(parser)
    add_dashboard_arguments(parser)
    add_tensorflow_runtime_arguments(parser, include_compile_learner=True)
    add_godot_render_argument(parser)
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
    best_tracker,
    start_episode,
    start_epsilon,
    learner_step,
    demo_data=None,
    budget=None,
    opponent_pool=None,
    opponent_teams=None,
):
    validate_async_arguments(args, supports_opponent_pool=True)
    budget = budget or TrainingBudget(getattr(args, "total_timesteps", 0))
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
    # Opponent snapshots are reused objects (the pool keeps one per worker), so a traced
    # greedy fn cached by model identity stays valid across their load_weights calls --
    # same trick as PPO's sync sample_fn cache. Keeps opponent inference off eager too.
    opponent_greedy_fns = {}

    def opponent_greedy_fn(opponent_model):
        fn = opponent_greedy_fns.get(id(opponent_model))
        if fn is None:
            fn = build_greedy_action_fn(opponent_model, obs_dim)
            opponent_greedy_fns[id(opponent_model)] = fn
        return fn

    snapshot = PolicySnapshot(model.get_weights())
    health_monitor = getattr(best_tracker, "health_monitor", None)
    recovery_handler = getattr(health_monitor, "recovery_handler", None)
    if recovery_handler is not None:
        recovery_handler.set_policy_publisher(
            lambda: snapshot.publish(model.get_weights())
        )
    rngs = [np.random.default_rng(args.env_seed_base + 100_003 * idx) for idx in range(len(envs))]
    replay_ready_event = Event()
    warmup_completed_episode = {
        "value": int(start_episode)
        if len(buffer) >= replay_warmup_threshold(args)
        else None
    }
    if warmup_completed_episode["value"] is not None:
        replay_ready_event.set()

    def epsilon_for_episode(episode):
        warmup_episode = warmup_completed_episode["value"]
        if warmup_episode is None:
            return 1.0
        elapsed = max(0, int(episode) - int(warmup_episode))
        return max(args.epsilon_min, float(start_epsilon) * (args.epsilon_decay ** elapsed))

    def begin_episode(worker_id, env, episode):
        env.configure(
            training_episode=episode,
            max_steps=args.max_steps_per_episode,
            physics_frames_per_step=args.physics_frames_per_step,
            training_mode=True,
            **scenario_curriculum_config(args),
        )
        obs, info = env.reset(seed=args.episode_seed_multiplier * episode + worker_id)
        # Each worker draws its own opponent into a model private to it, so concurrent
        # episodes cannot load weights over one another.
        opponent_match = (
            opponent_pool.start_episode(model, episode, worker_id=worker_id)
            if opponent_pool is not None and opponent_pool.enabled
            else OpponentMatch(None, "disabled", True)
        )
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
                "opponent_match": opponent_match,
                # Which agents the learner drives, and therefore which transitions may
                # enter the replay buffer. The opponent's experience is not ours to learn.
                "learner_mask": (
                    opponent_pool.learner_mask(
                        env.agent_team_ids,
                        opponent_teams,
                        episode,
                        worker_id,
                        use_current_policy=opponent_match.use_current_policy,
                    )
                    if opponent_pool is not None and opponent_pool.enabled
                    else np.ones((agent_count,), dtype=np.bool_)
                ),
                "replay_warmup_exploration": not replay_ready_event.is_set(),
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
            "opponent_match": opponent_match,
            "learner_mask": None,
            "replay_warmup_exploration": not replay_ready_event.is_set(),
        }

    def choose_action(worker_id, _env, episode, _step_idx, local_model, state):
        rng = rngs[worker_id]
        epsilon = epsilon_for_episode(episode)
        if args.multi_agent:
            observations = np.asarray(state["obs"], dtype=np.float32)
            # Traced (not eager): several collector threads call this at once, and eager
            # inference corrupts output under concurrency. The graph folds in the argmax.
            actions = greedy_actions[worker_id](observations).numpy().astype(np.int32)
            for agent_idx in range(len(actions)):
                if state["done_mask"][agent_idx]:
                    actions[agent_idx] = 0
                elif rng.random() < epsilon:
                    actions[agent_idx] = int(rng.integers(num_actions))
            opponent_model = state["opponent_match"].model
            if opponent_model is not None:
                # The frozen opponent plays greedily off its own snapshot: exploration
                # noise belongs to the learner, and the snapshot is not being trained.
                opponent_actions = opponent_greedy_fn(opponent_model)(observations).numpy().astype(np.int32)
                opponent_actions[state["done_mask"]] = 0
                actions = opponent_pool.merge_actions(
                    actions, opponent_actions, state["learner_mask"]
                )
            for agent_idx in range(len(actions)):
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
                learner_mask = state["learner_mask"]
                is_learner = learner_mask is None or bool(learner_mask[agent_idx])
                # The opponent's transitions come from a frozen snapshot's policy, not
                # ours: learning from them would poison the buffer with off-policy data
                # the learner never chose.
                if is_learner and not (was_done and done_mask[agent_idx]):
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
    recovery_runtime = OffPolicyRecoveryRuntime(
        args,
        buffer,
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
        target_sync=lambda: target_model.set_weights(model.get_weights()),
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
    update_count = 0
    losses_since_log = []
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
                if (
                    not replay_ready_event.is_set()
                    and len(buffer) >= replay_warmup_threshold(args)
                ):
                    warmup_completed_episode["value"] = int(completed)
                    replay_ready_event.set()
                budget.consume(collected_transitions)
                updates_due = scheduler.ingest(step_events)
                updates_performed = 0
                if (
                    len(buffer) >= max(args.replay_warmup, args.batch_size)
                    and not bool(getattr(health_monitor, "verification_pending", False))
                ):
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
            if completed % args.target_update_every == 0:
                target_model.set_weights(model.get_weights())
            if opponent_pool is not None:
                # Snapshot from the learner, not the workers: a worker's local model is
                # only synced every --async-policy-sync-steps and would freeze a policy
                # that is already stale.
                opponent_pool.snapshot(model, completed)
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
                ("mode", [
                    ("collector", "async"),
                    ("worker", event.worker_id),
                    (
                        "exploration",
                        "warmup_random"
                        if state.get("replay_warmup_exploration", False)
                        else "epsilon_greedy",
                    ),
                    ("epsilon", f"{epsilon:.3f}"),
                ]),
                ("outcome", [
                    ("reward", rewards),
                    ("progress", f"{float(state.get('progress', 0.0)):.4f}"),
                    ("success", f"{success_total}/{metric_count} ({success_total / max(metric_count, 1):.2%})"),
                    ("steps", f"mean:{float(np.mean(episode_steps)):.1f} range:[{int(np.min(episode_steps))},{int(np.max(episode_steps))}]"),
                ]),
                ("training", [
                    ("completed", f"{completed}/{args.num_episodes}"),
                    ("total_timesteps", budget.collected),
                    ("queue", f"{throughput['queue_size']}/{throughput['queue_capacity']} ({throughput['queue_saturation']:.0%})"),
                    ("replay", f"{len(buffer)}/{args.replay_capacity}"),
                    ("replay_warmup_left", max(
                        0, replay_warmup_threshold(args) - len(buffer))),
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
    else:
        # An evaluation requested near the last episode is still running here; without
        # this drain its result is dropped and a genuine final best is lost. Skipped on
        # interrupt: Ctrl-C should exit, not wait.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
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


@dataclass
class DQNPolicyState:
    policy_id: str
    model: object
    target_model: object
    optimizer: object
    buffer: ReplayBuffer
    epsilon: object
    learner_step: object
    artifact: PolicyArtifactSaver
    trainable: bool


def _validate_restored_assignment(args, assignment):
    manifest_path = Path(args.checkpoint_dir) / "multi_policy.json"
    if not manifest_path.is_file():
        return
    payload = load_multi_policy_manifest(manifest_path)
    stored = payload.get("agent_to_policy", {})
    if stored != assignment.agent_to_policy:
        raise RuntimeError(
            "The checkpoint multi-policy assignment does not match the current scenario: "
            f"stored={stored}, current={assignment.agent_to_policy}"
        )
    saved_algorithm = str(payload.get("algorithm", "dqn"))
    if saved_algorithm != "dqn":
        raise RuntimeError(
            f"Multi-policy checkpoint algorithm={saved_algorithm!r}, expected 'dqn'"
        )


def _save_multi_policy_dqn_checkpoint(
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
        "dqn",
        episode=episode,
        policy_metadata={
            policy_id: {
                "epsilon": float(state.epsilon.numpy()),
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


def _multi_policy_recovery_callback(policy_states):
    def post_restore(request):
        replay = {}
        for policy_id, state in policy_states.items():
            state.target_model.set_weights(state.model.get_weights())
            if request.mode == "hard":
                state.buffer.clear_online(preserve_protected_demos=True)
                replay[policy_id] = "cleared"
            else:
                replay[policy_id] = "preserved"
        return {
            "replay_buffer": replay,
            "target_networks": "all policy targets synchronized",
        }

    return post_restore


def run_async_multi_policy_dqn(
    args,
    envs,
    policy_states,
    assignment,
    checkpoint,
    checkpoint_manager,
    best_tracker,
    start_episode,
    budget,
):
    env0 = envs[0]
    obs_dim = env0.obs_dim
    num_actions = env0.num_actions
    with tf.device("/CPU:0"):
        local_models = [
            {
                policy_id: build_shared_q_network(
                    obs_dim=obs_dim,
                    num_actions=num_actions,
                )
                for policy_id in assignment.policy_ids
            }
            for _env in envs
        ]
    greedy_actions = [
        {
            policy_id: build_greedy_action_fn(model, obs_dim)
            for policy_id, model in worker_models.items()
        }
        for worker_models in local_models
    ]

    def live_weights():
        return {
            policy_id: state.model.get_weights()
            for policy_id, state in policy_states.items()
        }

    snapshot = MultiPolicySnapshot(live_weights())
    rngs = [
        np.random.default_rng(args.env_seed_base + 100_003 * worker_id)
        for worker_id in range(len(envs))
    ]
    initial_epsilons = {
        policy_id: float(state.epsilon.numpy())
        for policy_id, state in policy_states.items()
    }
    trainable_states = [
        state for state in policy_states.values() if state.trainable
    ]
    if not trainable_states:
        raise ValueError("Multi-policy DQN requires at least one trainable policy")
    replay_ready_event = Event()
    warmup_completed_episode = {
        "value": int(start_episode)
        if all(
            len(state.buffer) >= replay_warmup_threshold(args)
            for state in trainable_states
        )
        else None
    }
    if warmup_completed_episode["value"] is not None:
        replay_ready_event.set()

    def epsilon_for(policy_id, episode):
        if not policy_states[policy_id].trainable:
            return 0.0
        warmup_episode = warmup_completed_episode["value"]
        if warmup_episode is None:
            return 1.0
        elapsed = max(0, int(episode) - int(warmup_episode))
        return max(
            args.epsilon_min,
            initial_epsilons[policy_id] * (args.epsilon_decay ** elapsed),
        )

    def begin_episode(worker_id, env, episode):
        env.configure(
            training_episode=episode,
            max_steps=args.max_steps_per_episode,
            physics_frames_per_step=args.physics_frames_per_step,
            training_mode=True,
            **scenario_curriculum_config(args),
        )
        obs, info = env.reset(
            seed=args.episode_seed_multiplier * episode + worker_id
        )
        agent_count = len(env.agent_ids)
        return {
            "obs": obs,
            "done": False,
            "done_mask": np.asarray(
                info.get(
                    "per_agent_done",
                    np.zeros((agent_count,), dtype=np.bool_),
                ),
                dtype=np.bool_,
            ),
            "ep_reward": np.zeros((agent_count,), dtype=np.float32),
            "success": np.zeros((agent_count,), dtype=np.bool_),
            "episode_steps": np.zeros((agent_count,), dtype=np.int32),
            "event_counts": {},
            "terminal_counts": {},
            "replay_warmup_exploration": not replay_ready_event.is_set(),
        }

    def choose_action(
        worker_id,
        env,
        episode,
        _step_idx,
        worker_models,
        state,
    ):
        rng = rngs[worker_id]
        actions = np.zeros((len(env.agent_ids),), dtype=np.int32)
        for agent_idx, agent_id in enumerate(env.agent_ids):
            if state["done_mask"][agent_idx]:
                continue
            policy_id = assignment.policy_for_agent(agent_id)
            epsilon = epsilon_for(policy_id, episode)
            if rng.random() < epsilon:
                actions[agent_idx] = int(rng.integers(num_actions))
            else:
                observation = np.expand_dims(
                    state["obs"][agent_idx],
                    axis=0,
                ).astype(np.float32)
                actions[agent_idx] = int(
                    greedy_actions[worker_id][policy_id](
                        observation
                    ).numpy()[0]
                )
        return actions

    def process_step(
        _worker_id,
        env,
        _episode,
        _step_idx,
        state,
        actions,
        step_result,
    ):
        next_obs, _reward, terminated, truncated, info = step_result
        global_done = bool(terminated or truncated)
        rewards = np.asarray(info.get("per_agent_rewards"), dtype=np.float32)
        done_mask = np.asarray(info.get("per_agent_done"), dtype=np.bool_)
        terminated_mask = np.asarray(
            info.get("per_agent_terminated", done_mask),
            dtype=np.bool_,
        )
        agent_infos = list(info.get("per_agent_infos", []))
        transitions = []
        for agent_idx, agent_id in enumerate(env.agent_ids):
            was_done = bool(state["done_mask"][agent_idx])
            if was_done and done_mask[agent_idx]:
                continue
            agent_info = (
                agent_infos[agent_idx]
                if agent_idx < len(agent_infos)
                else {}
            )
            state["episode_steps"][agent_idx] += 1
            update_episode_diagnostics(
                state,
                agent_info,
                agent_idx=agent_idx,
                newly_done=bool(done_mask[agent_idx] and not was_done),
            )
            policy_id = assignment.policy_for_agent(agent_id)
            if policy_states[policy_id].trainable:
                transitions.append((
                    policy_id,
                    state["obs"][agent_idx],
                    int(actions[agent_idx]),
                    float(rewards[agent_idx]),
                    next_obs[agent_idx],
                    bool(terminated_mask[agent_idx] or terminated),
                ))
            state["ep_reward"][agent_idx] += rewards[agent_idx]
        state["obs"] = next_obs
        state["done_mask"] = done_mask
        state["done"] = global_done or bool(np.all(done_mask))
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
    minimum_policy_version = {"value": None}
    health_monitor = getattr(best_tracker, "health_monitor", None)
    recovery_handler = getattr(health_monitor, "recovery_handler", None)
    if recovery_handler is not None:
        recovery_handler.set_policy_publisher(
            lambda: snapshot.publish(live_weights())
        )

        base_post_restore = _multi_policy_recovery_callback(policy_states)

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
        "Collector mode: async multi-policy DQN "
        f"workers={len(envs)} policies={list(assignment.policy_ids)} "
        f"queue={args.async_queue_capacity}",
        flush=True,
    )
    completed = int(start_episode)
    done_workers = 0
    update_count = 0
    losses = {policy_id: [] for policy_id in assignment.policy_ids}
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
                    f"Async multi-policy collector {event.worker_id} failed"
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
                if (
                    not replay_ready_event.is_set()
                    and all(
                        len(state.buffer) >= replay_warmup_threshold(args)
                        for state in trainable_states
                    )
                ):
                    warmup_completed_episode["value"] = int(completed)
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
                        policy_losses = state.learner_step(
                            state.buffer,
                            args.batch_size,
                            due,
                        )
                        losses[policy_id].extend(policy_losses.tolist())
                        update_count += int(due)
                        performed += int(due)
                scheduler.record_updates(performed)
                if (
                    performed > 0
                    and update_count >= args.async_policy_publish_updates
                ):
                    snapshot.publish(live_weights())
                    update_count %= args.async_policy_publish_updates
                if budget.exhausted:
                    break
                continue

            if not isinstance(event, AsyncEpisodeEvent):
                continue
            completed += 1
            if completed % args.target_update_every == 0:
                for state in policy_states.values():
                    if state.trainable:
                        state.target_model.set_weights(
                            state.model.get_weights()
                        )
            for policy_id, state in policy_states.items():
                if state.trainable:
                    state.epsilon.assign(epsilon_for(policy_id, completed))

            payload = event.payload
            by_policy = {}
            for policy_id, state in policy_states.items():
                indices = assignment.indices_for(env0.agent_ids, policy_id)
                rewards = [
                    float(payload["ep_reward"][index])
                    for index in indices
                ]
                by_policy[policy_id] = {
                    "reward_mean": (
                        float(np.mean(rewards)) if rewards else 0.0
                    ),
                    "epsilon": float(state.epsilon.numpy()),
                    "replay": len(state.buffer),
                    "updates": len(losses[policy_id]),
                    "loss": (
                        float(np.mean(losses[policy_id]))
                        if losses[policy_id]
                        else 0.0
                    ),
                }
            throughput = scheduler.throughput(pool)
            print_episode_metrics(event.episode, [
                ("mode", [
                    ("collector", "async"),
                    ("worker", event.worker_id),
                    ("policies", len(policy_states)),
                    (
                        "exploration",
                        "warmup_random"
                        if payload.get("replay_warmup_exploration", False)
                        else "epsilon_greedy",
                    ),
                ]),
                ("outcome", [("reward", payload["ep_reward"].tolist())]),
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
            for values in losses.values():
                values.clear()

            if (
                args.checkpoint_every > 0
                and completed % args.checkpoint_every == 0
            ):
                saved_path = _save_multi_policy_dqn_checkpoint(
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
                    saved_path = _save_multi_policy_dqn_checkpoint(
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
            "\nInterrupt received: stopping async multi-policy DQN collectors...",
            flush=True,
        )
    finally:
        pool.close()

    if last_saved_episode != completed:
        _save_multi_policy_dqn_checkpoint(
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


def run_sync_multi_policy_dqn(
    args,
    envs,
    stepper,
    best_tracker,
    budget,
):
    assignment = build_policy_assignment(envs, args)
    validate_multi_policy_options(args, supports_async=True, supports_opponent_pool=False)
    if args.demo_path:
        raise ValueError(
            "DQN demonstrations are not yet routed by policy. Remove --demo-path for "
            "independent multi-policy training."
        )
    if args.policy_path or args.initial_weights_path:
        raise ValueError(
            "Independent multi-policy warm starts use --resume/--resume-checkpoint. "
            "Per-policy Keras warm starts are not implemented yet."
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
    num_actions = env0.num_actions
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
        model = build_shared_q_network(obs_dim=obs_dim, num_actions=num_actions)
        target_model = build_shared_q_network(obs_dim=obs_dim, num_actions=num_actions)
        target_model.set_weights(model.get_weights())
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        buffer = ReplayBuffer(capacity=args.replay_capacity)
        epsilon = tf.Variable(
            args.epsilon_start if assignment.is_trainable(policy_id) else 0.0,
            dtype=tf.float32,
            name=f"epsilon_{assignment.key_for(policy_id)}",
        )
        learner_step = build_dqn_learner_step(
            model,
            target_model,
            optimizer,
            args.gamma,
            compiled=args.tf_compile_learner,
        )
        artifact_dir = (
            Path(args.checkpoint_dir)
            / "policies"
            / assignment.key_for(policy_id)
        )
        metadata = build_policy_metadata(
            "dqn",
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
            "parameter_sharing": len(assignment.indices_for(env0.agent_ids, policy_id)) > 1,
        })
        state = DQNPolicyState(
            policy_id=policy_id,
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            buffer=buffer,
            epsilon=epsilon,
            learner_step=learner_step,
            artifact=PolicyArtifactSaver(model, artifact_dir, metadata),
            trainable=assignment.is_trainable(policy_id),
        )
        policy_states[policy_id] = state
        policy_trackables[assignment.key_for(policy_id)] = tf.train.Checkpoint(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            epsilon=epsilon,
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
        _validate_restored_assignment(args, assignment)
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

    for state in policy_states.values():
        state.optimizer.learning_rate.assign(args.learning_rate)

    epsilon_start = max(
        (
            float(state.epsilon.numpy())
            for state in policy_states.values()
            if state.trainable
        ),
        default=args.epsilon_start,
    )
    args.epsilon_decay = resolve_epsilon_decay(args, epsilon_start, start_episode)
    write_multi_policy_manifest(
        args.checkpoint_dir,
        assignment,
        "dqn",
        episode=start_episode,
    )

    best_tracker.configure_recovery(
        checkpoint,
        [
            (f"policy_{assignment.key_for(policy_id)}", state.optimizer)
            for policy_id, state in policy_states.items()
            if state.trainable
        ],
        post_restore=_multi_policy_recovery_callback(policy_states),
    )

    print(
        "Independent multi-policy DQN: "
        f"assignment={assignment.mode} policies={list(assignment.policy_ids)} "
        f"trainable={list(assignment.trainable_policy_ids)} "
        f"agent_map={assignment.agent_to_policy}",
        flush=True,
    )
    print(
        f"DQN learner: {'compiled batched graph' if args.tf_compile_learner else 'eager'} "
        f"instances={len(policy_states)}",
        flush=True,
    )

    if args.collector_mode == "async":
        return run_async_multi_policy_dqn(
            args,
            envs,
            policy_states,
            assignment,
            checkpoint,
            checkpoint_manager,
            best_tracker,
            start_episode,
            budget,
        )

    last_completed_episode = start_episode
    last_saved_episode = None
    interrupted = False
    sync_throttles = {
        policy_id: SyncUpdateThrottle(args)
        for policy_id, state in policy_states.items()
        if state.trainable
    }
    try:
        for episode in range(start_episode, args.num_episodes):
            env_states = []
            for env_idx, env in enumerate(envs):
                env.configure(
                    training_episode=episode,
                    max_steps=args.max_steps_per_episode,
                    physics_frames_per_step=args.physics_frames_per_step,
                    training_mode=True,
                    **scenario_curriculum_config(args),
                )
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
                    "ep_reward": np.zeros((agent_count,), dtype=np.float32),
                    "success": np.zeros((agent_count,), dtype=np.bool_),
                    "target_reached": np.zeros((agent_count,), dtype=np.bool_),
                    "target_seen": np.zeros((agent_count,), dtype=np.bool_),
                    "episode_steps": np.zeros((agent_count,), dtype=np.int32),
                    "event_counts": {},
                    "terminal_counts": {},
                    "action_counts": np.zeros(
                        (agent_count, num_actions),
                        dtype=np.int32,
                    ),
                    "last_action": np.zeros((agent_count,), dtype=np.int32),
                })

            losses = {policy_id: [] for policy_id in assignment.policy_ids}
            for _step in episode_step_indices(args.max_steps_per_episode):
                if all(state["done"] for state in env_states):
                    break

                requests = []
                for env, env_state in zip(envs, env_states):
                    if env_state["done"]:
                        continue
                    actions = np.zeros((len(env.agent_ids),), dtype=np.int32)
                    for agent_idx, agent_id in enumerate(env.agent_ids):
                        if env_state["done_mask"][agent_idx]:
                            continue
                        policy_id = assignment.policy_for_agent(agent_id)
                        policy = policy_states[policy_id]
                        epsilon = (
                            1.0
                            if (
                                policy.trainable
                                and len(policy.buffer)
                                < replay_warmup_threshold(args)
                            )
                            else float(policy.epsilon.numpy())
                            if policy.trainable
                            else 0.0
                        )
                        actions[agent_idx] = select_action(
                            policy.model,
                            env_state["obs"][agent_idx],
                            epsilon,
                            num_actions,
                        )
                        env_state["action_counts"][
                            agent_idx, actions[agent_idx]
                        ] += 1
                        env_state["last_action"][agent_idx] = actions[agent_idx]
                    requests.append((env, env_state, actions))

                for env, env_state, actions, step_result in stepper.step(requests):
                    next_obs, _reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                    transitions_by_policy = {
                        policy_id: 0 for policy_id in sync_throttles
                    }
                    per_agent_rewards = np.asarray(
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
                    per_agent_infos = info.get(
                        "per_agent_infos",
                        [{} for _ in env.agent_ids],
                    )

                    for agent_idx, agent_id in enumerate(env.agent_ids):
                        was_done = bool(env_state["done_mask"][agent_idx])
                        agent_info = (
                            per_agent_infos[agent_idx]
                            if agent_idx < len(per_agent_infos)
                            else {}
                        )
                        if not was_done:
                            env_state["episode_steps"][agent_idx] += 1
                        update_episode_diagnostics(
                            env_state,
                            agent_info,
                            agent_idx=agent_idx,
                            newly_done=bool(
                                per_agent_done[agent_idx] and not was_done
                            ),
                        )
                        if bool(agent_info.get("target_reached", False)):
                            env_state["target_reached"][agent_idx] = True
                        if bool(agent_info.get("target_first_seen", False)):
                            env_state["target_seen"][agent_idx] = True
                        if was_done and per_agent_done[agent_idx]:
                            continue

                        policy_id = assignment.policy_for_agent(agent_id)
                        policy = policy_states[policy_id]
                        if policy.trainable:
                            policy.buffer.add(
                                env_state["obs"][agent_idx],
                                int(actions[agent_idx]),
                                float(per_agent_rewards[agent_idx]),
                                next_obs[agent_idx],
                                bool(
                                    per_agent_terminated[agent_idx]
                                    or terminated
                                ),
                            )
                            budget.consume(1)
                            transitions_by_policy[policy_id] += 1
                        env_state["ep_reward"][agent_idx] += (
                            per_agent_rewards[agent_idx]
                        )

                    env_state["obs"] = next_obs
                    env_state["done_mask"] = per_agent_done
                    env_state["done"] = done or bool(np.all(per_agent_done))

                    if not best_tracker.health_monitor.verification_pending:
                        for policy_id, policy in policy_states.items():
                            if not policy.trainable:
                                continue
                            if len(policy.buffer) < max(
                                args.replay_warmup,
                                args.batch_size,
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
                            if updates_due <= 0:
                                continue
                            losses[policy_id].extend(
                                policy.learner_step(
                                    policy.buffer,
                                    args.batch_size,
                                    updates_due,
                                ).tolist()
                            )
                if budget.exhausted:
                    break

            if (episode + 1) % args.target_update_every == 0:
                for policy in policy_states.values():
                    if policy.trainable:
                        policy.target_model.set_weights(
                            policy.model.get_weights()
                        )

            for policy in policy_states.values():
                if (
                    policy.trainable
                    and len(policy.buffer) >= replay_warmup_threshold(args)
                ):
                    policy.epsilon.assign(
                        max(
                            args.epsilon_min,
                            float(policy.epsilon.numpy())
                            * args.epsilon_decay,
                        )
                    )

            rewards_summary = [
                state["ep_reward"].tolist()
                for state in env_states
            ]
            controlled_agents = len(envs) * len(env0.agent_ids)
            success_total = sum(
                int(np.sum(state["success"]))
                for state in env_states
            )
            episode_steps = np.concatenate(
                [state["episode_steps"] for state in env_states]
            )
            policy_metrics = {}
            for policy_id, policy in policy_states.items():
                indices = assignment.indices_for(env0.agent_ids, policy_id)
                policy_rewards = [
                    float(state["ep_reward"][index])
                    for state in env_states
                    for index in indices
                ]
                policy_metrics[policy_id] = {
                    "reward_mean": (
                        float(np.mean(policy_rewards))
                        if policy_rewards
                        else 0.0
                    ),
                    "epsilon": float(policy.epsilon.numpy()),
                    "replay": len(policy.buffer),
                    "updates": len(losses[policy_id]),
                    "loss": (
                        float(np.mean(losses[policy_id]))
                        if losses[policy_id]
                        else 0.0
                    ),
                }

            print_episode_metrics(episode, [
                ("mode", [
                    ("policies", len(policy_states)),
                    ("assignment", assignment.mode),
                ]),
                ("outcome", [
                    ("reward", rewards_summary),
                    (
                        "success",
                        f"{success_total}/{controlled_agents} "
                        f"({success_total / max(controlled_agents, 1):.2%})",
                    ),
                    (
                        "steps",
                        f"mean:{float(np.mean(episode_steps)):.1f} "
                        f"range:[{int(np.min(episode_steps))},"
                        f"{int(np.max(episode_steps))}]",
                    ),
                ]),
                ("training", [
                    ("total_timesteps", budget.collected),
                    ("by_policy", policy_metrics),
                ]),
            ], args.log_format)

            last_completed_episode = episode + 1
            apply_ready_best_checkpoint(best_tracker)
            if (
                args.checkpoint_every > 0
                and (episode + 1) % args.checkpoint_every == 0
            ):
                saved_path = _save_multi_policy_dqn_checkpoint(
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
                    saved_path = _save_multi_policy_dqn_checkpoint(
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
            "\nInterrupt received: saving the multi-policy DQN state...",
            flush=True,
        )

    if last_saved_episode != last_completed_episode:
        _save_multi_policy_dqn_checkpoint(
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
    validate_async_arguments(args, supports_opponent_pool=True)
    best_tracker = BestCheckpointTracker(args, "dqn")
    describe_tensorflow_backend(args)
    dashboard = maybe_start_dashboard(args, algorithm="dqn")
    training_start_time = time.monotonic()
    # Architecture is process-wide state, so it must be fixed before the first network is
    # built -- the seeding block below is the last point where nothing exists yet.
    set_network_layers(args.network_layers or default_network_layers("dqn"))
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

        obs_dim = envs[0].obs_dim
        num_actions = envs[0].num_actions
        agent_id = envs[0].agent_id
        agent_ids = envs[0].agent_ids
        if envs[0].action_type != "discrete":
            raise RuntimeError(
                f"algorithms/dqn.py supports only discrete action spaces, got action_type={envs[0].action_type!r}. "
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

        if args.multi_policy:
            run_sync_multi_policy_dqn(
                args,
                envs,
                stepper,
                best_tracker,
                budget,
            )
            return

        model = build_shared_q_network(obs_dim=obs_dim, num_actions=num_actions)
        target_model = build_shared_q_network(obs_dim=obs_dim, num_actions=num_actions)
        args.policy_artifact = PolicyArtifactSaver(
            model,
            args.checkpoint_dir,
            build_policy_metadata("dqn", envs[0]),
        )
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
        resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_manager)
        restored_replay_count = 0
        if args.policy_path and args.initial_weights_path:
            raise RuntimeError("Use either --policy-path or the legacy --initial-weights-path, not both")
        initial_policy_path = args.policy_path or args.initial_weights_path
        if resume_checkpoint and initial_policy_path:
            raise RuntimeError("--policy-path cannot be combined with --resume or --resume-checkpoint")
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
        elif initial_policy_path:
            loaded_policy = load_policy_into_model(model, initial_policy_path, expected_algorithm="dqn")
            target_model.set_weights(model.get_weights())
            print(
                f"Warm-started DQN policy from {loaded_policy['source_kind']}: "
                f"{loaded_policy['path']} "
                f"(fresh replay, optimizer, epsilon={epsilon:.3f}, episode=0)",
                flush=True,
            )
        optimizer.learning_rate.assign(args.learning_rate)
        best_tracker.configure_recovery(
            checkpoint,
            [("q_network", optimizer)],
        )

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
                best_tracker,
                start_episode,
                epsilon,
                learner_step,
                demo_data,
                budget,
                opponent_pool=opponent_pool,
                opponent_teams=opponent_teams,
            )
            model.save_weights(args.weights_path)
            print(f"Saved weights: {args.weights_path}", flush=True)
            return

        recovery_runtime = OffPolicyRecoveryRuntime(
            args,
            buffer,
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
            target_sync=lambda: target_model.set_weights(model.get_weights()),
        )
        recovery_handler = best_tracker.health_monitor.recovery_handler
        if recovery_handler is not None:
            recovery_handler.set_post_restore(recovery_runtime.post_restore)

        sync_throttle = SyncUpdateThrottle(args)
        last_saved_episode = None
        for episode in range(start_episode, args.num_episodes):
            opponent_match = opponent_pool.start_episode(model, episode)
            warmup_exploration = len(buffer) < replay_warmup_threshold(args)
            action_epsilon = (
                1.0
                if warmup_exploration
                else epsilon
            )
            env_states = []
            for env_idx, env in enumerate(envs):
                env.configure(
                    training_episode=episode,
                    max_steps=args.max_steps_per_episode,
                    physics_frames_per_step=args.physics_frames_per_step,
                    training_mode=True,
                    **scenario_curriculum_config(args),
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
                        action = select_actions(
                            model,
                            state["obs"],
                            state["done_mask"],
                            action_epsilon,
                            num_actions,
                        )
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
                        action = select_action(
                            model,
                            state["obs"],
                            action_epsilon,
                            num_actions,
                        )
                        state["action_counts"][int(action)] += 1
                        state["last_action"] = int(action)

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
                                budget.consume(1)
                                transitions_added += 1
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
                        budget.consume(1)
                        transitions_added += 1
                        state["ep_reward"] += float(reward)

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
                        if updates_due > 0:
                            losses.extend(
                                learner_step(
                                    buffer,
                                    args.batch_size,
                                    updates_due,
                                ).tolist()
                            )
                if budget.exhausted:
                    break

            if (episode + 1) % args.target_update_every == 0:
                target_model.set_weights(model.get_weights())

            if len(buffer) >= replay_warmup_threshold(args):
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
                    (
                        "exploration",
                        "warmup_random"
                        if warmup_exploration
                        else "epsilon_greedy",
                    ),
                    ("epsilon", f"{action_epsilon:.3f}"),
                    ("opponent", opponent_match.label),
                ]),
                ("outcome", outcome_metrics),
                ("training", [
                    ("total_timesteps", budget.collected),
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

            apply_ready_best_checkpoint(best_tracker)
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
                epsilon,
                args,
                final=True,
            )
        # The evaluation requested on the last episode is still running; collect it
        # instead of dropping a possibly-best result on the floor.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
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
