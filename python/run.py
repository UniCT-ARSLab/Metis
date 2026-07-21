import argparse
import json
import os
import platform
import random
import re
import sys
import sysconfig
import time
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
        original_args = list(getattr(sys, "orig_argv", sys.argv))
        os.execv(sys.executable, [sys.executable] + original_args[1:])


configure_tensorflow_runtime()

import numpy as np
import tensorflow as tf

from algorithms.ppo import build_action_metadata, pack_action, split_model_outputs
from core.models import build_continuous_actor, build_hybrid_actor_critic, build_sac_actor, build_shared_q_network
from core.policy_artifact import (
    POLICY_MODEL_FILENAME,
    load_policy_into_model,
    load_policy_model,
    policy_algorithms_are_compatible,
    resolve_policy_path,
)
from core.training import (
    add_godot_render_argument,
    add_tensorflow_runtime_arguments,
    configure_tensorflow_devices,
    episode_step_indices,
)
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


SUCCESS_EVENTS = {"level_cleared", "target_reached", "finish_reached", "goal_scored", "success"}
SUCCESS_TERMINAL_REASONS = SUCCESS_EVENTS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a trained policy on any Godot BridgeServer scenario.")
    parser.add_argument(
        "--algorithm",
        choices=["auto", "dqn", "ddpg", "ddpg_bc", "ddpgfd", "td3", "td3_bc", "sac", "ppo"],
        default="auto",
    )
    parser.add_argument(
        "--load-from",
        choices=["auto", "policy", "keras", "weights", "checkpoint"],
        default="auto",
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help=(
            "Policy bundle/directory, complete .keras/.h5 model, or legacy .weights.h5 file. "
            "Defaults to CHECKPOINT_DIR/policy.keras."
        ),
    )
    parser.add_argument("--weights-path", default="generic_dqn_weights.weights.h5")
    parser.add_argument("--actor-weights-path", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/generic")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Exact TensorFlow checkpoint prefix (for example checkpoints/run/ckpt-2000).",
    )
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6200)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum episode steps; use 0 or --no-time-limit to rely on terminal conditions.",
    )
    parser.add_argument("--infinite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Physically reset the scenario after a terminal state. With --no-reset, "
            "Metis starts the next logical episode from the current Godot state."
        ),
    )
    parser.add_argument(
        "--initial-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Physically reset the scenario before the first inference. With "
            "--no-initial-reset Metis initializes observations and episode bookkeeping "
            "from the current Godot state without moving agents or targets."
        ),
    )
    parser.add_argument(
        "--continue-after-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ask the scenario to report successful goals without terminating the episode, "
            "so inference can continue when a target moves. Scenario support is required."
        ),
    )
    parser.add_argument("--time-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--training-episode",
        type=int,
        default=None,
        help="Scenario curriculum episode. Defaults to the episode encoded in a checkpoint name, or 0.",
    )
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--execution-mode",
        choices=["auto", "lockstep", "realtime"],
        default="auto",
        help=(
            "Lockstep is deterministic but pauses the scene between policy updates, which "
            "reads as stuttering to a human watching. Realtime lets Godot run continuously "
            "but paces the sim to wall-clock. 'auto' picks lockstep when --headless (nobody "
            "is watching; keep it fast and reproducible) and realtime otherwise."
        ),
    )
    parser.add_argument(
        "--realtime-action-hz",
        type=float,
        default=60.0,
        help="Policy action-update frequency in realtime mode; 0 disables wall-clock pacing.",
    )
    parser.add_argument(
        "--realtime-simulation-fps",
        type=int,
        default=60,
        help="Godot render/process frame cap while realtime mode is active.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    add_tensorflow_runtime_arguments(parser)
    add_godot_render_argument(parser, default="project")
    parser.add_argument("--connect-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path where the aggregate evaluation summary is written as JSON.",
    )
    return parser.parse_args(argv)


def iter_episode_numbers(args):
    episode = 0
    while args.infinite or episode < args.episodes:
        yield episode
        episode += 1


def configured_max_steps(args):
    if args.time_limit:
        return args.max_steps
    return 0


def should_preserve_state(args, episode):
    if episode == 0:
        return not args.initial_reset
    return not args.reset


def wait_for_realtime_tick(previous_deadline, frequency_hz, clock=time.monotonic, sleeper=time.sleep):
    now = clock()
    if frequency_hz <= 0.0:
        return now
    period = 1.0 / float(frequency_hz)
    deadline = float(previous_deadline) + period
    remaining = deadline - now
    if remaining > 0.0:
        sleeper(remaining)
        return deadline
    if -remaining > period:
        return now
    return deadline


def greedy_action(model, obs):
    q_values = model(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    return int(np.argmax(q_values)), q_values


def select_action(model, obs, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1), None
    return greedy_action(model, obs)


def select_multi_actions(model, obs_batch, epsilon, num_actions):
    actions = []
    q_values = []
    for obs in obs_batch:
        action, q = select_action(model, obs, epsilon, num_actions)
        actions.append(action)
        q_values.append(q)
    return np.asarray(actions, dtype=np.int32), q_values


def scale_action_numpy(raw_action, low, high):
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def select_continuous_action(model, obs, env, algorithm):
    output = model(np.expand_dims(obs, axis=0), training=False)
    if algorithm == "sac":
        raw_action = tf.tanh(output[0]).numpy()[0]
    else:
        raw_action = output.numpy()[0]
    return np.clip(scale_action_numpy(raw_action, env.action_low, env.action_high), env.action_low, env.action_high).astype(np.float32)


def select_multi_continuous_actions(model, obs_batch, env, algorithm):
    actions = [
        select_continuous_action(model, obs, env, algorithm)
        for obs in np.asarray(obs_batch, dtype=np.float32)
    ]
    return np.asarray(actions, dtype=np.float32)


def select_hybrid_action(model, obs, action_meta):
    outputs = model(np.expand_dims(obs, axis=0), training=False)
    logits, continuous_mean, _value = split_model_outputs(outputs, action_meta)
    discrete_actions = [int(tf.argmax(component_logits[0]).numpy()) for component_logits in logits]
    if action_meta["continuous_size"] > 0:
        continuous_action = np.clip(
            continuous_mean.numpy()[0],
            action_meta["continuous_low"],
            action_meta["continuous_high"],
        ).astype(np.float32)
    else:
        continuous_action = np.zeros((0,), dtype=np.float32)
    return pack_action(discrete_actions, continuous_action, action_meta)


def select_multi_hybrid_actions(model, obs_batch, agent_ids, action_meta):
    return {
        agent_id: select_hybrid_action(model, obs, action_meta)
        for agent_id, obs in zip(agent_ids, np.asarray(obs_batch, dtype=np.float32))
    }


def resolve_algorithm(requested, env, weights_path, manifest=None):
    if requested != "auto":
        if manifest is not None and not policy_algorithms_are_compatible(
            requested,
            str(manifest.get("algorithm", "")),
        ):
            raise RuntimeError(
                f"Requested algorithm={requested!r} does not match policy manifest "
                f"algorithm={manifest.get('algorithm')!r}"
            )
        return requested
    if manifest is not None and manifest.get("algorithm"):
        return str(manifest["algorithm"])
    if env.action_type == "discrete":
        return "dqn"
    if env.action_type == "continuous":
        weight_name = Path(weights_path).name.lower()
        if "sac" in weight_name:
            return "sac"
        for variant in ("td3_bc", "ddpg_bc", "ddpgfd", "td3"):
            if variant in weight_name:
                return variant
        return "ddpg"
    if env.action_type == "hybrid":
        return "ppo"
    raise RuntimeError(f"run.py does not support action_type={env.action_type!r}")


def policy_weights_path(args, algorithm):
    if algorithm in {"ddpg", "ddpg_bc", "ddpgfd", "td3", "td3_bc", "sac"} and args.actor_weights_path:
        return args.actor_weights_path
    return args.weights_path


def keras_policy_path(args):
    requested = args.policy_path or str(Path(args.checkpoint_dir) / POLICY_MODEL_FILENAME)
    return resolve_policy_path(requested)


def load_direct_policy(args):
    policy_path = keras_policy_path(args)
    if args.policy_path and (args.load_from == "checkpoint" or args.checkpoint_path):
        raise RuntimeError("--policy-path cannot be combined with checkpoint loading options")
    should_load = (
        args.policy_path is not None
        or args.load_from in {"policy", "keras"}
        or (args.load_from == "auto" and policy_path.is_file())
    )
    if not should_load:
        return None, None, None, None
    if not policy_path.is_file():
        raise RuntimeError(f"Policy not found: {policy_path}")
    model, manifest, source_kind, resolved_path = load_policy_model(policy_path)
    if model is not None:
        print(f"Loaded policy {source_kind}: {resolved_path}", flush=True)
    return model, manifest, source_kind, str(resolved_path)


def validate_keras_policy(model, manifest, env):
    if manifest is not None:
        observation_manifest = manifest.get("observation", {})
        action_manifest = manifest.get("action", {})
        expected_obs_dim = int(observation_manifest.get("size", env.obs_dim))
        if expected_obs_dim != env.obs_dim:
            raise RuntimeError(
                f"Policy obs_dim={expected_obs_dim} does not match scenario obs_dim={env.obs_dim}"
            )
        expected_observation_names = list(observation_manifest.get("names", []))
        current_observation_names = list(env._spec_for_agent(env.agent_id).get("observation_names", []))
        if expected_observation_names and current_observation_names != expected_observation_names:
            raise RuntimeError(
                "Policy observation order does not match the scenario: "
                f"policy={expected_observation_names}, scenario={current_observation_names}"
            )
        expected_action_type = str(action_manifest.get("type", env.action_type))
        if expected_action_type != env.action_type:
            raise RuntimeError(
                f"Policy action_type={expected_action_type!r} does not match scenario "
                f"action_type={env.action_type!r}"
            )
        expected_action_size = int(action_manifest.get("size", env.action_size))
        if expected_action_size != env.action_size:
            raise RuntimeError(
                f"Policy action_size={expected_action_size} does not match scenario action_size={env.action_size}"
            )
        expected_action_names = list(action_manifest.get("names", []))
        if expected_action_names and expected_action_names != env.action_names:
            raise RuntimeError(
                "Policy action order does not match the scenario: "
                f"policy={expected_action_names}, scenario={env.action_names}"
            )
    if isinstance(model.input_shape, list) or int(model.input_shape[-1]) != env.obs_dim:
        raise RuntimeError(
            f"Keras policy input_shape={model.input_shape} is incompatible with obs_dim={env.obs_dim}"
        )


def normalize_checkpoint_path(checkpoint_path):
    path = str(checkpoint_path)
    if path.endswith(".index"):
        path = path[:-len(".index")]
    if not Path(f"{path}.index").is_file():
        raise RuntimeError(f"Checkpoint index not found: {path}.index")
    return path


def checkpoint_episode(checkpoint_path):
    match = re.search(r"ckpt-(\d+)$", str(checkpoint_path))
    return int(match.group(1)) if match else None


def build_policy_checkpoint(model, algorithm):
    if algorithm == "dqn":
        return tf.train.Checkpoint(model=model)
    if algorithm in {"ddpg", "ddpg_bc", "ddpgfd", "td3", "td3_bc", "sac"}:
        return tf.train.Checkpoint(actor=model)
    if algorithm == "ppo":
        return tf.train.Checkpoint(model=model)
    raise RuntimeError(f"Unsupported checkpoint load for algorithm={algorithm!r}")


def warm_policy_inference(model, obs_dim):
    outputs = model(np.zeros((1, int(obs_dim)), dtype=np.float32), training=False)
    for tensor in tf.nest.flatten(outputs):
        if hasattr(tensor, "numpy"):
            tensor.numpy()


def load_policy(model, args, algorithm):
    weights_path = Path(policy_weights_path(args, algorithm))
    checkpoint_dir = Path(args.checkpoint_dir)
    latest_checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))

    if args.checkpoint_path:
        if args.load_from == "weights":
            raise RuntimeError("--checkpoint-path cannot be used with --load-from weights")
        selected_checkpoint = normalize_checkpoint_path(args.checkpoint_path)
        build_policy_checkpoint(model, algorithm).restore(selected_checkpoint).expect_partial()
        print(f"Loaded exact policy checkpoint: {selected_checkpoint}", flush=True)
        return "checkpoint", selected_checkpoint

    if args.load_from == "weights" or (args.load_from == "auto" and weights_path.exists()):
        model.load_weights(str(weights_path))
        print(f"Loaded policy weights: {weights_path}", flush=True)
        return "weights", str(weights_path)

    if args.load_from == "checkpoint" or (args.load_from == "auto" and latest_checkpoint):
        if not latest_checkpoint:
            raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")
        build_policy_checkpoint(model, algorithm).restore(latest_checkpoint).expect_partial()
        print(f"Loaded policy checkpoint: {latest_checkpoint}", flush=True)
        return "checkpoint", latest_checkpoint

    raise RuntimeError(
        f"No policy found. Checked weights={weights_path} and checkpoint_dir={checkpoint_dir}"
    )


def episode_agent_infos(info, multi_agent):
    if multi_agent:
        return [item for item in info.get("per_agent_infos", []) if isinstance(item, dict)]
    agent_info = info.get("agent_info", {})
    return [agent_info] if isinstance(agent_info, dict) else []


def agent_succeeded(agent_info):
    events = agent_info.get("events", {})
    successful_event = isinstance(events, dict) and any(
        isinstance(events.get(name, 0.0), (bool, int, float, np.number))
        and float(events.get(name, 0.0)) > 0.0
        for name in SUCCESS_EVENTS
    )
    return (
        bool(agent_info.get("target_reached", False))
        or bool(agent_info.get("finish_reached", False))
        or str(agent_info.get("terminal_reason", "")) in SUCCESS_TERMINAL_REASONS
        or successful_event
    )


def summarize_episode_outcome(info, multi_agent):
    agent_infos = episode_agent_infos(info, multi_agent)
    successes = sum(int(agent_succeeded(agent_info)) for agent_info in agent_infos)
    reasons = [
        str(agent_info.get("terminal_reason"))
        for agent_info in agent_infos
        if agent_info.get("terminal_reason")
    ]
    return successes, len(agent_infos), reasons


def build_evaluation_summary(rewards, steps, successes, trials):
    if not rewards:
        return None
    reward_values = np.asarray(rewards, dtype=np.float32)
    step_values = np.asarray(steps, dtype=np.float32)
    return {
        "episodes": len(rewards),
        "successes": int(successes),
        "trials": int(trials),
        "success_rate": float(successes / max(trials, 1)),
        "reward_mean": float(np.mean(reward_values)),
        "reward_min": float(np.min(reward_values)),
        "reward_max": float(np.max(reward_values)),
        "steps_mean": float(np.mean(step_values)),
        "steps_min": int(np.min(step_values)),
        "steps_max": int(np.max(step_values)),
    }


def emit_evaluation_summary(args, rewards, steps, successes, trials):
    summary = build_evaluation_summary(rewards, steps, successes, trials)
    if summary is None:
        return None
    print(
        "Evaluation summary\n"
        f"  episodes  {summary['episodes']}\n"
        f"  success   {summary['successes']}/{summary['trials']} ({summary['success_rate']:.2%})\n"
        f"  reward    mean:{summary['reward_mean']:.4f} "
        f"range:[{summary['reward_min']:.4f},{summary['reward_max']:.4f}]\n"
        f"  steps     mean:{summary['steps_mean']:.1f} "
        f"range:[{summary['steps_min']},{summary['steps_max']}]",
        flush=True,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    args = parse_args()
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative; use --no-time-limit for no limit")
    if args.delay < 0.0:
        raise ValueError("--delay cannot be negative")
    if args.realtime_action_hz < 0.0:
        raise ValueError("--realtime-action-hz cannot be negative")
    if args.realtime_simulation_fps < 1:
        raise ValueError("--realtime-simulation-fps must be at least 1")
    if args.training_episode is not None and args.training_episode < 0:
        raise ValueError("--training-episode cannot be negative")
    if args.execution_mode == "auto":
        # Lockstep freezes the scene between policy updates, so a human watching sees the
        # sim stutter. Nobody is watching a headless run, and automated checkpoint
        # evaluation spawns this script headless without pinning the mode -- there,
        # lockstep is what keeps eval fast and reproducible.
        args.execution_mode = "lockstep" if args.headless else "realtime"
    # Match training: let TensorFlow grow GPU memory on demand instead of grabbing most of
    # the VRAM at context init. Must run before the first TF op (tf.random.set_seed below).
    configure_tensorflow_devices(tf, memory_growth=args.gpu_memory_growth, system_name=platform.system())
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    manager = None
    if not args.connect_only:
        manager = GodotProcessManager(
            godot_bin=args.godot_bin,
            project_dir=args.godot_project,
            scene_path=args.godot_scene,
        )
        manager.start_many([args.port], headless=args.headless, render_mode=args.render_mode)
        print(f"Started Godot on port {args.port}", flush=True)

    env = None
    evaluation_rewards = []
    evaluation_steps = []
    evaluation_successes = 0
    evaluation_trials = 0
    try:
        env = ScenarioGymEnv(
            host=args.host,
            port=args.port,
            seed=args.seed,
            agent_id=args.agent_id,
            multi_agent=args.multi_agent,
        )
        model, policy_manifest, direct_policy_kind, direct_policy_path = load_direct_policy(args)
        algorithm = resolve_algorithm(
            args.algorithm,
            env,
            args.actor_weights_path or args.weights_path,
            manifest=policy_manifest,
        )
        action_meta = None
        if algorithm == "ppo":
            action_meta = build_action_metadata(env.action_space_spec)
        if model is not None:
            validate_keras_policy(model, policy_manifest, env)
            loaded_kind, loaded_path = direct_policy_kind, direct_policy_path
        else:
            if algorithm == "dqn":
                model = build_shared_q_network(obs_dim=env.obs_dim, num_actions=env.num_actions)
            elif algorithm in {"ddpg", "ddpg_bc", "ddpgfd", "td3", "td3_bc"}:
                model = build_continuous_actor(obs_dim=env.obs_dim, action_size=env.action_size)
            elif algorithm == "sac":
                model = build_sac_actor(obs_dim=env.obs_dim, action_size=env.action_size)
            elif algorithm == "ppo":
                model = build_hybrid_actor_critic(
                    obs_dim=env.obs_dim,
                    discrete_sizes=action_meta["discrete_sizes"],
                    continuous_size=action_meta["continuous_size"],
                )
                model(np.zeros((1, env.obs_dim), dtype=np.float32), training=False)
            else:
                raise RuntimeError(f"Unsupported algorithm={algorithm!r}")
            if direct_policy_path is not None:
                loaded_policy = load_policy_into_model(
                    model,
                    direct_policy_path,
                    expected_algorithm=algorithm,
                )
                loaded_kind = loaded_policy["source_kind"]
                loaded_path = str(loaded_policy["path"])
                validate_keras_policy(model, policy_manifest, env)
                print(f"Loaded policy weights: {loaded_path}", flush=True)
            else:
                loaded_kind, loaded_path = load_policy(model, args, algorithm)
        warm_policy_inference(model, env.obs_dim)

        env_max_steps = configured_max_steps(args)
        inferred_episode = checkpoint_episode(loaded_path) if loaded_kind == "checkpoint" else None
        if policy_manifest is not None and direct_policy_path is not None:
            inferred_episode = int(policy_manifest.get("episode", 0))
        training_episode = args.training_episode
        if training_episode is None:
            training_episode = inferred_episode if inferred_episode is not None else 0
        scenario_config = {
            "max_steps": env_max_steps,
            "training_episode": training_episode,
        }
        if args.continue_after_success:
            scenario_config.update(
                continue_after_success=True,
                terminate_on_stalled_progress=False,
            )
        config_reply = env.configure(**scenario_config)
        if args.continue_after_success and not bool(
            config_reply.get("continue_after_success", False)
        ):
            print(
                "WARNING: the scenario did not confirm continue_after_success; "
                "successful states may remain terminal.",
                flush=True,
            )
        env.set_execution_mode(
            args.execution_mode,
            simulation_fps=args.realtime_simulation_fps,
        )

        print(
            f"Running policy algorithm={algorithm} obs_dim={env.obs_dim} "
            f"action_type={env.action_type} action_size={env.action_size} agents={env.agent_ids} "
            f"training_episode={training_episode} execution_mode={args.execution_mode}",
            flush=True,
        )
        if args.execution_mode == "realtime":
            pacing = "uncapped" if args.realtime_action_hz == 0.0 else f"{args.realtime_action_hz:g} Hz"
            print(
                f"Realtime policy pacing: {pacing}; Godot frame cap: {args.realtime_simulation_fps} FPS",
                flush=True,
            )
        if args.infinite:
            print("Running indefinitely. Stop with Ctrl+C.", flush=True)
        if not args.reset:
            print(
                "Physical resets after terminal states disabled; inference resumes from the "
                "current state in the next logical episode.",
                flush=True,
            )
        if not args.initial_reset:
            print(
                "Initial physical reset disabled; initializing from the current Godot state.",
                flush=True,
            )
        if not args.time_limit:
            print("Episode time limit disabled; episodes end only on terminal scenario state.", flush=True)

        for episode in iter_episode_numbers(args):
            reset_options = {
                "preserve_state": should_preserve_state(args, episode),
            }
            obs, info = env.reset(seed=args.seed + episode, options=reset_options)
            realtime_deadline = time.monotonic()
            total_reward = 0.0
            steps_taken = 0
            if args.multi_agent:
                total_reward = np.zeros((len(env.agent_ids),), dtype=np.float32)

            for step in episode_step_indices(env_max_steps):
                if env.action_type == "hybrid":
                    if args.multi_agent:
                        action = select_multi_hybrid_actions(model, obs, env.agent_ids, action_meta)
                    else:
                        action = select_hybrid_action(model, obs, action_meta)
                elif env.action_type == "continuous":
                    if args.multi_agent:
                        action = select_multi_continuous_actions(model, obs, env, algorithm)
                    else:
                        action = select_continuous_action(model, obs, env, algorithm)
                elif args.multi_agent:
                    action, _ = select_multi_actions(model, obs, args.epsilon, env.num_actions)
                else:
                    action, _ = select_action(model, obs, args.epsilon, env.num_actions)

                obs, reward, terminated, truncated, info = env.step(action)
                steps_taken = step + 1
                if args.multi_agent:
                    total_reward += np.asarray(info.get("per_agent_rewards", []), dtype=np.float32)
                else:
                    total_reward += float(reward)

                if args.print_every > 0 and step % args.print_every == 0:
                    if isinstance(action, dict):
                        action_label = action
                    else:
                        action_label = action.tolist() if hasattr(action, "tolist") else int(action)
                    total_label = total_reward.tolist() if hasattr(total_reward, "tolist") else f"{total_reward:.4f}"
                    print(
                        f"episode={episode:04d} step={step:04d} action={action_label} "
                        f"reward={reward:.4f} total={total_label} "
                        f"terminated={terminated} truncated={truncated}",
                        flush=True,
                    )

                if args.execution_mode == "realtime":
                    realtime_deadline = wait_for_realtime_tick(
                        realtime_deadline,
                        args.realtime_action_hz,
                    )
                elif args.delay > 0:
                    time.sleep(args.delay)

                if terminated or truncated:
                    break

            success_count, trial_count, terminal_reasons = summarize_episode_outcome(info, args.multi_agent)
            reward_score = float(np.mean(total_reward)) if args.multi_agent else float(total_reward)
            evaluation_rewards.append(reward_score)
            evaluation_steps.append(steps_taken)
            evaluation_successes += success_count
            evaluation_trials += trial_count
            total_label = total_reward.tolist() if hasattr(total_reward, "tolist") else f"{total_reward:.4f}"
            terminal_label = terminal_reasons if terminal_reasons else ["unknown"]
            print(
                f"episode={episode:04d} finished total_reward={total_label} steps={steps_taken} "
                f"success={success_count}/{trial_count} terminal={terminal_label}",
                flush=True,
            )
            if not args.reset and (args.infinite or episode + 1 < args.episodes):
                print(
                    "Terminal boundary cleared logically; continuing from the current "
                    "physical state.",
                    flush=True,
                )
        emit_evaluation_summary(
            args,
            evaluation_rewards,
            evaluation_steps,
            evaluation_successes,
            evaluation_trials,
        )
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
        emit_evaluation_summary(
            args,
            evaluation_rewards,
            evaluation_steps,
            evaluation_successes,
            evaluation_trials,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if manager is not None:
            manager.stop_all()


if __name__ == "__main__":
    main()
