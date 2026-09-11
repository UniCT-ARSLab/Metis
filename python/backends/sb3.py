import argparse
import json
import os
import time
from itertools import count
from pathlib import Path

import numpy as np
from gymnasium import spaces

try:
    from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The SB3 backend is optional. Install it with: "
        "python -m pip install -r python/requirements-sb3.txt"
    ) from exc

from backends.sb3_actions import SB3ActionCodec
from backends.sb3_multi_vec_env import MetisSB3MultiAgentVecEnv
from backends.sb3_vec_env import MetisSB3VecEnv
from backends.sb3_state import (
    checkpoint_number as _checkpoint_number,
    latest_checkpoint,
    load_training_state,
    replay_path_for_model as _replay_path_for_model,
    state_path_for_model as _state_path_for_model,
)
from core.evaluation import (
    build_evaluation_summary,
    print_evaluation_summary,
    summarize_episode_outcome,
    write_evaluation_summary,
)
from core.multi_policy import add_multi_policy_arguments
from core.training import (
    add_dashboard_arguments,
    add_godot_render_argument,
    add_lockstep_tuning_arguments,
    add_parallel_env_arguments,
    add_training_health_arguments,
    add_transition_snapshot_arguments,
    build_lockstep_user_args,
    maybe_start_dashboard,
    print_episode_metrics,
    report_training_time,
    validate_transition_snapshot_arguments,
)
from core.transition_snapshots import TransitionSnapshotPublisher
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


ALGORITHM_CLASSES = {
    "dqn": DQN,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "ppo": PPO,
}
OFF_POLICY_ALGORITHMS = {"dqn", "ddpg", "td3", "sac"}


def training_episode_cursor(env, fallback):
    current = env
    while current is not None:
        cursor = getattr(current, "next_training_episode", None)
        if cursor is not None:
            return int(cursor)
        current = getattr(current, "venv", None)
    return int(fallback)


def current_training_episode(env, fallback):
    current = env
    while current is not None:
        episode = getattr(current, "current_training_episode", None)
        if episode is not None:
            return int(episode)
        current = getattr(current, "venv", None)
    return int(fallback)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stable-Baselines3 backend for Metis Godot scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algorithm", choices=sorted(ALGORITHM_CLASSES), required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Optional completed-episode limit in addition to --total-timesteps; zero disables it.",
    )
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--network", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--env-seed-base", type=int, default=100)
    parser.add_argument("--episode-seed-multiplier", type=int, default=1000)
    parser.add_argument("--env-timeout", type=float, default=30.0)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--multi-agent", action=argparse.BooleanOptionalAction, default=False)
    add_multi_policy_arguments(parser)
    parser.add_argument(
        "--sb3-multi-agent-partial-done",
        choices=["error", "reset-all"],
        default="error",
        help=(
            "How grouped SB3 lanes handle one agent ending before the whole Godot "
            "world. 'error' preserves task semantics; 'reset-all' truncates the "
            "remaining agents and resets the complete world."
        ),
    )

    parser.add_argument("--buffer-size", "--replay-capacity", dest="buffer_size", type=int, default=100_000)
    parser.add_argument("--learning-starts", "--replay-warmup", dest="learning_starts", type=int, default=500)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=-1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-update-interval", "--target-update-every", dest="target_update_interval", type=int, default=10_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--action-noise", type=float, default=0.1)

    parser.add_argument("--ppo-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--reset-progress-curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-progress-start-max", type=float, default=0.025)
    parser.add_argument("--reset-progress-end-max", type=float, default=0.35)
    parser.add_argument("--reset-progress-ramp-episodes", type=int, default=400)

    parser.add_argument("--checkpoint-dir", default="checkpoints/generic_sb3")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Warm-start from an SB3 .zip model without restoring replay or episode counters.",
    )
    parser.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-replay-buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    add_transition_snapshot_arguments(parser)

    parser.add_argument("--evaluation-episodes", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--evaluation-training-episode", type=int, default=None)
    parser.add_argument("--evaluation-max-steps", type=int, default=None)
    parser.add_argument("--evaluation-json", default=None)

    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument("--godot-project", default=None)
    parser.add_argument("--godot-scene", default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--godot-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--collector-mode",
        choices=["sync", "async"],
        default="sync",
        help="SB3 currently uses synchronized vector rollouts; async is rejected explicitly.",
    )
    add_parallel_env_arguments(parser)
    add_lockstep_tuning_arguments(parser)
    add_godot_render_argument(parser)
    add_dashboard_arguments(parser)
    add_training_health_arguments(parser)
    parser.add_argument("--log-format", choices=["pretty", "compact"], default="pretty")
    return parser.parse_args(argv)


def validate_args(args):
    validate_transition_snapshot_arguments(args)
    if args.multi_policy:
        raise ValueError(
            "The SB3 comparison backend does not support independent --multi-policy "
            "training. Use --backend metis --multi-agent --multi-policy "
            "--collector-mode sync."
        )
    if args.collector_mode != "sync":
        raise ValueError(
            "Stable-Baselines3 collects synchronized VecEnv batches. "
            "Use --collector-mode sync or the Metis backend for asynchronous collectors."
        )
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.total_timesteps < 1:
        raise ValueError("--total-timesteps must be at least 1 for the SB3 backend")
    if args.num_episodes < 0:
        raise ValueError("--num-episodes cannot be negative")
    if args.max_steps_per_episode < 0:
        raise ValueError("--max-steps-per-episode cannot be negative")
    if args.physics_frames_per_step < 1:
        raise ValueError("--physics-frames-per-step must be at least 1")
    if args.policy_path and (args.resume or args.resume_checkpoint):
        raise ValueError("--policy-path cannot be combined with --resume or --resume-checkpoint")
    if args.auto_recovery:
        raise ValueError(
            "--auto-recovery requires the native Metis backend. The SB3 comparison "
            "adapter exposes telemetry health but not isolated frozen-checkpoint rollback."
        )


def validate_scenario(args, envs):
    first = envs[0]
    expected_type = {
        "dqn": {"discrete"},
        "ppo": {"discrete", "continuous", "hybrid"},
        "ddpg": {"continuous"},
        "td3": {"continuous"},
        "sac": {"continuous"},
    }[args.algorithm]
    if first.action_type not in expected_type:
        raise ValueError(
            f"SB3 {args.algorithm.upper()} is incompatible with action_type={first.action_type!r}; "
            f"expected one of {sorted(expected_type)}"
        )
    for env in envs[1:]:
        if env.observation_space != first.observation_space or env.action_space != first.action_space:
            raise ValueError("All Godot environments must expose identical observation and action spaces")
    if args.multi_agent:
        expected_contract = [
            (
                int(spec.get("obs_dim", -1)),
                str(spec.get("action_type", "")),
                json.dumps(spec.get("action_space", {}), sort_keys=True),
            )
            for spec in first.agent_specs
        ]
        if len(set(expected_contract)) != 1:
            raise ValueError(
                "SB3 parameter sharing requires every agent to expose the same "
                "observation and action contract"
            )


def build_action_codec(env):
    if not env.multi_agent:
        native_space = env.action_space
    elif env.action_type == "discrete":
        native_space = spaces.Discrete(env.num_actions)
    elif env.action_type == "continuous":
        native_space = spaces.Box(
            low=np.asarray(env.action_low, dtype=np.float32),
            high=np.asarray(env.action_high, dtype=np.float32),
            dtype=np.float32,
        )
    else:
        native_space = spaces.Dict({})
    return SB3ActionCodec(
        env.action_type,
        native_space,
        env.action_space_spec,
    )


def algorithm_kwargs(args, env):
    common = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "policy_kwargs": {"net_arch": list(args.network)},
        "tensorboard_log": args.tensorboard_log,
        "seed": args.env_seed_base,
        "device": args.device,
        "verbose": args.verbose,
    }
    if args.algorithm == "ppo":
        common.update(
            n_steps=args.ppo_steps,
            n_epochs=args.ppo_epochs,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_ratio,
            ent_coef=args.entropy_coef,
            vf_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm,
        )
        return common

    common.update(
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
    )
    if args.algorithm == "dqn":
        common.update(
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.epsilon_start,
            exploration_final_eps=args.epsilon_min,
        )
        return common

    common["tau"] = args.tau
    if args.algorithm in {"ddpg", "td3"} and args.action_noise > 0.0:
        action_size = int(np.prod(env.action_space.shape))
        common["action_noise"] = NormalActionNoise(
            mean=np.zeros(action_size, dtype=np.float32),
            sigma=np.full(action_size, args.action_noise, dtype=np.float32),
        )
    return common


def resolve_load_path(args):
    if args.resume_checkpoint:
        path = Path(args.resume_checkpoint)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if not path.is_file():
            raise FileNotFoundError(f"SB3 checkpoint not found: {path}")
        return path, True
    if args.resume:
        path = latest_checkpoint(args.checkpoint_dir)
        if path is None:
            raise FileNotFoundError(f"No SB3 checkpoint found in {args.checkpoint_dir}")
        return path, True
    if args.policy_path:
        path = Path(args.policy_path)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if not path.is_file():
            raise FileNotFoundError(f"SB3 policy not found: {path}")
        return path, False
    return None, False


## Scene telemetry the native backend logs from the Godot step info but the SB3 callback used to
## drop, leaving the dashboard with reward and success only -- too little to compare a run against
## the native backend, where the pose errors are the metrics that actually move first.
_POSE_FIELDS = (
    ("position_error_m", "position_error_mean", "{:.5f}"),
    ("orientation_error_deg", "orientation_error_deg", "{:.3f}"),
    ("max_joint_speed", "max_joint_speed", "{:.4f}"),
    ("hold_frames", "hold_frames", "{:.1f}"),
    ("progress", "progress", "{:.5f}"),
)


## Optimizer-side numbers SB3 keeps in its own logger. Reading them here is what puts alpha and
## the losses next to the scene telemetry, the way the native backend reports them.
_TRAINER_FIELDS = (
    ("train/ent_coef", "alpha", "{:.5f}"),
    ("train/actor_loss", "actor_loss", "{:.5f}"),
    ("train/critic_loss", "critic_loss", "{:.5f}"),
    ("train/learning_rate", "learning_rate", "{:.7f}"),
    ("train/n_updates", "policy_updates", "{:.0f}"),
)


def _trainer_telemetry(model):
    fields = []
    values = getattr(getattr(model, "logger", None), "name_to_value", None) or {}
    for source, name, fmt in _TRAINER_FIELDS:
        value = values.get(source)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        fields.append((name, fmt.format(float(value))))
    buffer = getattr(model, "replay_buffer", None)
    if buffer is not None:
        try:
            # SB3 stores buffer_size // n_envs slots, each holding one transition per env, so both
            # size() and buffer_size count SLOTS. Scaling by n_envs reports transitions, which is
            # what the native backend logs and the only figure comparable between the two.
            envs = max(int(getattr(buffer, "n_envs", 1)), 1)
            fields.append(("replay", int(buffer.size()) * envs))
            fields.append(("replay_capacity", int(buffer.buffer_size) * envs))
        except Exception:
            pass
    return fields


def _pose_telemetry(info):
    agent_info = info.get("agent_info", info)
    if not isinstance(agent_info, dict):
        return []
    fields = []
    for source, name, fmt in _POSE_FIELDS:
        value = agent_info.get(source)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        fields.append((name, fmt.format(float(value))))
    collided = agent_info.get("collided")
    if collided is not None:
        fields.append(("collision_rate", f"{float(bool(collided)):.5f}"))
    stalled = agent_info.get("progress_stalled")
    if stalled is not None:
        fields.append(("stall_rate", f"{float(bool(stalled)):.5f}"))
    return fields


class MetisSB3Callback(BaseCallback):
    def __init__(self, args, *, start_episode=0):
        super().__init__(verbose=0)
        self.args = args
        self.completed_episodes = int(start_episode)
        self.started = time.monotonic()
        self.initial_num_timesteps = 0
        self.last_checkpoint_episode = None
        # Peak progress reached inside each running episode: the final value alone hides an
        # arm that got close and then drifted away, which is exactly the failure to watch.
        self._progress_peak = {}
        self._initial_updates = 0.0
        self.transition_publisher = None

    def _updates_per_second(self):
        """Gradient steps per second, so the dashboard can show the update rate the way the
        native backend does. SB3 only exposes a cumulative counter, hence the delta."""
        values = getattr(getattr(self.model, "logger", None), "name_to_value", None) or {}
        updates = values.get("train/n_updates")
        if not isinstance(updates, (int, float)) or isinstance(updates, bool):
            return 0.0
        elapsed = max(time.monotonic() - self.started, 1e-6)
        return max(0.0, float(updates) - self._initial_updates) / elapsed

    def _on_training_start(self):
        self.started = time.monotonic()
        self.initial_num_timesteps = int(self.num_timesteps)
        values = getattr(getattr(self.model, "logger", None), "name_to_value", None) or {}
        updates = values.get("train/n_updates")
        self._initial_updates = float(updates) if isinstance(updates, (int, float)) else 0.0
        if int(self.args.checkpoint_every_transitions or 0) > 0:
            self.transition_publisher = TransitionSnapshotPublisher(
                self.args.transition_snapshot_dir,
                interval=self.args.checkpoint_every_transitions,
                budget=self.args.total_timesteps,
                backend="sb3",
                algorithm=self.args.algorithm,
            )

    def _publish_transition_snapshots(self):
        if self.transition_publisher is None:
            return

        def save_policy(directory, _threshold):
            model_path = directory / "model.zip"
            self.model.save(model_path)
            state_path = directory / "training_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "backend": "sb3",
                        "algorithm": self.args.algorithm,
                        "completed_episodes": self.completed_episodes,
                        "num_timesteps": int(self.num_timesteps),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return model_path

        self.transition_publisher.publish_due(
            int(self.num_timesteps), self.completed_episodes, save_policy
        )

    def _checkpoint(self):
        directory = Path(self.args.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"ckpt-{self.completed_episodes}.zip"
        self.model.save(path)
        state = {
            "backend": "sb3",
            "algorithm": self.args.algorithm,
            "completed_episodes": self.completed_episodes,
            "training_episode": training_episode_cursor(
                self.training_env,
                self.completed_episodes,
            ),
            "num_timesteps": int(self.num_timesteps),
        }
        _state_path_for_model(path).write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.args.save_replay_buffer and self.args.algorithm in OFF_POLICY_ALGORITHMS:
            self.model.save_replay_buffer(_replay_path_for_model(path))
        self.last_checkpoint_episode = self.completed_episodes
        self._prune_checkpoints()
        print(f"Saved SB3 checkpoint: {path}", flush=True)
        return path

    def _prune_checkpoints(self):
        keep = int(self.args.keep_checkpoints)
        if keep <= 0:
            return
        directory = Path(self.args.checkpoint_dir)
        checkpoints = sorted(directory.glob("ckpt-*.zip"), key=_checkpoint_number)
        for path in checkpoints[:-keep]:
            path.unlink(missing_ok=True)
            _state_path_for_model(path).unlink(missing_ok=True)
            _replay_path_for_model(path).unlink(missing_ok=True)

    def _on_step(self):
        # SB3 invokes callbacks after collection and before the corresponding learner update.
        self._publish_transition_snapshots()
        dones = np.asarray(self.locals.get("dones", []), dtype=np.bool_)
        infos = list(self.locals.get("infos", []))
        for index, info in enumerate(infos):
            agent_info = info.get("agent_info", info)
            progress = agent_info.get("progress") if isinstance(agent_info, dict) else None
            if isinstance(progress, (int, float)) and not isinstance(progress, bool):
                self._progress_peak[index] = max(
                    self._progress_peak.get(index, float(progress)), float(progress))
        for index, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            progress_peak = self._progress_peak.pop(index, None)
            self.completed_episodes += 1
            episode = info.get("episode", {})
            reward = float(episode.get("r", 0.0))
            length = int(episode.get("l", 0))
            success = int(bool(info.get("is_success", False)))
            elapsed = max(time.monotonic() - self.started, 1e-6)
            invocation_timesteps = max(
                0,
                int(self.num_timesteps) - self.initial_num_timesteps,
            )
            sections = [
                ("mode", [("backend", "sb3"), ("algorithm", self.args.algorithm)]),
                (
                    "outcome",
                    [
                        ("reward", f"{reward:.5f}"),
                        ("reward_mean", f"{reward:.5f}"),
                        ("success_rate", f"{float(success):.5f}"),
                        ("steps_mean", f"{float(length):.1f}"),
                    ],
                ),
            ]
            pose = _pose_telemetry(info)
            if progress_peak is not None:
                pose.append(("progress_max", f"{progress_peak:.5f}"))
            if pose:
                sections.append(("pose", pose))
            trainer = _trainer_telemetry(self.model)
            if trainer:
                sections.append(("trainer", trainer))
            print_episode_metrics(
                self.completed_episodes - 1,
                sections + [
                    (
                        "training",
                        [
                            ("completed", self.completed_episodes),
                            ("total_timesteps", int(self.num_timesteps)),
                            ("transitions_s", f"{invocation_timesteps / elapsed:.1f}"),
                            ("env_steps_s", f"{invocation_timesteps / elapsed:.1f}"),
                            ("updates_s", f"{self._updates_per_second():.1f}"),
                        ],
                    ),
                ],
                self.args.log_format,
            )
            if (
                self.args.checkpoint_every > 0
                and self.completed_episodes % self.args.checkpoint_every == 0
            ):
                self._checkpoint()
        return not (
            self.args.num_episodes > 0
            and self.completed_episodes >= self.args.num_episodes
        )


def save_final_model(model, callback, args):
    directory = Path(args.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "final_model.zip"
    model.save(path)
    state = {
        "backend": "sb3",
        "algorithm": args.algorithm,
        "completed_episodes": callback.completed_episodes,
        "training_episode": training_episode_cursor(
            model.get_env(),
            callback.completed_episodes,
        ),
        "num_timesteps": int(model.num_timesteps),
    }
    _state_path_for_model(path).write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.save_replay_buffer and args.algorithm in OFF_POLICY_ALGORITHMS:
        model.save_replay_buffer(_replay_path_for_model(path))
    print(f"Saved final SB3 model: {path}", flush=True)
    return path


def evaluate_model(model, env, action_codec, args, training_episode):
    rewards = []
    steps = []
    successes = 0
    trials = 0
    max_steps = args.evaluation_max_steps
    if max_steps is None:
        max_steps = args.max_steps_per_episode or 10_000

    for episode in range(args.evaluation_episodes):
        env.configure(
            training_episode=training_episode,
            max_steps=max_steps,
            physics_frames_per_step=args.physics_frames_per_step,
            training_mode=False,
        )
        obs, _ = env.reset(seed=args.evaluation_seed + episode)
        total_reward = 0.0
        steps_taken = 0
        info = {}
        step_range = range(max_steps) if max_steps > 0 else count()
        for step in step_range:
            policy_action, _state = model.predict(obs, deterministic=True)
            if args.multi_agent:
                action = {
                    agent_id: action_codec.decode(policy_action[agent_index])
                    for agent_index, agent_id in enumerate(env.agent_ids)
                }
            else:
                action = action_codec.decode(policy_action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps_taken = step + 1
            if terminated or truncated:
                break
        success_count, trial_count, _reasons = summarize_episode_outcome(
            info,
            args.multi_agent,
        )
        rewards.append(total_reward)
        steps.append(steps_taken)
        successes += success_count
        trials += trial_count

    summary = build_evaluation_summary(rewards, steps, successes, trials)
    if summary is not None:
        summary.update(
            backend="sb3",
            algorithm=args.algorithm,
            total_timesteps=int(model.num_timesteps),
        )
    print_evaluation_summary(summary)
    write_evaluation_summary(args.evaluation_json, summary)
    return summary


def main():
    args = parse_args()
    args.backend = "sb3"
    validate_args(args)
    dashboard = maybe_start_dashboard(args, algorithm=args.algorithm)
    training_started = time.monotonic()
    np.random.seed(args.env_seed_base)

    ports = [args.base_port + index for index in range(args.num_envs)]
    manager = GodotProcessManager(
        godot_bin=args.godot_bin,
        project_dir=args.godot_project,
        scene_path=args.godot_scene,
    )
    raw_envs = []
    vec_env = None
    model = None
    callback = None
    interrupted = False
    training_ready = False
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
        raw_envs = [
            ScenarioGymEnv(
                port=port,
                seed=args.env_seed_base + index,
                timeout=args.env_timeout,
                agent_id=args.agent_id,
                multi_agent=args.multi_agent,
            )
            for index, port in enumerate(ports)
        ]
        validate_scenario(args, raw_envs)
        first = raw_envs[0]
        action_codec = build_action_codec(first)
        print(
            f"Scenario spec: agent_id={first.agent_id} {first.agent_summary()} "
            f"obs_dim={first.obs_dim} action_type={first.action_type} "
            f"action_size={first.action_size} actions={first.action_names} "
            f"sb3_action={action_codec.describe()}",
            flush=True,
        )

        load_path, is_resume = resolve_load_path(args)
        training_state = load_training_state(load_path) if is_resume else {}
        start_episode = int(training_state.get("completed_episodes", 0))
        training_episode_start = int(training_state.get("training_episode", start_episode))
        vec_kwargs = {
            "action_codec": action_codec,
            "max_steps": args.max_steps_per_episode,
            "physics_frames_per_step": args.physics_frames_per_step,
            "episode_seed_multiplier": args.episode_seed_multiplier,
            "training_episode_start": training_episode_start,
            "parallel_steps": args.parallel_env_steps,
            "reset_progress_curriculum": args.reset_progress_curriculum,
            "reset_progress_start_max": args.reset_progress_start_max,
            "reset_progress_end_max": args.reset_progress_end_max,
            "reset_progress_ramp_episodes": args.reset_progress_ramp_episodes,
        }
        if args.multi_agent:
            vec_env = MetisSB3MultiAgentVecEnv(
                raw_envs,
                partial_done_mode=args.sb3_multi_agent_partial_done,
                **vec_kwargs,
            )
        else:
            vec_env = MetisSB3VecEnv(raw_envs, **vec_kwargs)
        callback = MetisSB3Callback(args, start_episode=start_episode)

        model_class = ALGORITHM_CLASSES[args.algorithm]
        if load_path is not None:
            model = model_class.load(
                load_path,
                env=vec_env,
                device=args.device,
                tensorboard_log=args.tensorboard_log,
            )
            print(
                f"Loaded SB3 {'checkpoint' if is_resume else 'policy'}: {load_path} "
                f"timesteps={model.num_timesteps} episodes={start_episode}",
                flush=True,
            )
            if is_resume and args.algorithm in OFF_POLICY_ALGORITHMS:
                replay_path = _replay_path_for_model(load_path)
                if replay_path.is_file():
                    model.load_replay_buffer(replay_path)
                    print(f"Restored SB3 replay buffer: {replay_path}", flush=True)
                elif args.require_replay_buffer:
                    raise FileNotFoundError(f"Required SB3 replay buffer not found: {replay_path}")
                else:
                    print(f"SB3 replay buffer not found; resuming policy only: {replay_path}", flush=True)
        else:
            model = model_class(**algorithm_kwargs(args, vec_env))

        if dashboard is not None:
            dashboard.set_meta(
                backend="sb3",
                agents=len(first.agent_ids) if args.multi_agent else 1,
                action_type=first.action_type,
                obs_dim=first.obs_dim,
                collector_mode="sync",
            )
        print(
            f"SB3 learner: algorithm={args.algorithm} device={model.device} "
            f"godot_envs={args.num_envs} policy_lanes={vec_env.num_envs} "
            f"total_timesteps={args.total_timesteps}",
            flush=True,
        )
        training_ready = True
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=not is_resume,
            progress_bar=False,
        )
        if (
            callback.transition_publisher is not None
            and int(model.num_timesteps) >= int(args.total_timesteps)
            and not callback.transition_publisher.complete
        ):
            raise RuntimeError(
                "Transition budget ended before every SB3 policy snapshot was published"
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupt received: saving the current SB3 state...", flush=True)
    finally:
        if training_ready and model is not None and callback is not None:
            final_path = save_final_model(model, callback, args)
            if (
                not interrupted
                and args.evaluation_episodes > 0
                and raw_envs
            ):
                evaluation_episode = args.evaluation_training_episode
                if evaluation_episode is None:
                    evaluation_episode = current_training_episode(
                        vec_env,
                        callback.completed_episodes,
                    )
                evaluate_model(
                    model,
                    raw_envs[0],
                    action_codec,
                    args,
                    evaluation_episode,
                )
        report_training_time(dashboard, training_started)
        if vec_env is not None:
            vec_env.close()
        else:
            for env in raw_envs:
                try:
                    env.close()
                except Exception:
                    pass
        manager.stop_all()


if __name__ == "__main__":
    main()
