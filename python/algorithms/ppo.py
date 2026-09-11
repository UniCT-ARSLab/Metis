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

from core.curriculum import scenario_curriculum_config
from core.models import (
    build_hybrid_actor_critic,
    default_network_layers,
    set_network_layers,
)
from core.policy_action_adapter import (
    ResidualActionAdapter, StandardActionAdapter, load_gate_config, sha_file as _adapter_sha_file)
from core.multi_policy import (
    MultiPolicySnapshot,
    add_multi_policy_arguments,
    build_policy_assignment,
    load_multi_policy_manifest,
    validate_multi_policy_options,
    write_multi_policy_manifest,
)
from core.policy_artifact import PolicyArtifactSaver, build_policy_metadata, load_policy_into_model
from core.opponent_pool import OpponentPool, add_opponent_pool_arguments, validate_team_layout
from core.training import (
    AsyncCollectorPool,
    AsyncEpisodeEvent,
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
    add_log_format_argument,
    add_dashboard_arguments,
    maybe_start_dashboard,
    report_training_time,
    add_lockstep_tuning_arguments,
    add_parallel_env_arguments,
    add_godot_render_argument,
    add_tensorflow_runtime_arguments,
    build_lockstep_user_args,
    configure_tensorflow_devices,
    episode_step_indices,
    print_episode_metrics,
    resolve_resume_checkpoint,
    validate_async_arguments,
    argument_group,
    GROUP_CHECKPOINTS,
    GROUP_GODOT,
    GROUP_LOOP,
    GROUP_PPO,
    GROUP_RESIDUAL,
)
from envs.process_manager import GodotProcessManager
from envs.scenario import ScenarioGymEnv


LOG_2PI = np.float32(np.log(2.0 * np.pi))

# Model heads follow the components declared by the action-space specification.
SUPPORTED_ACTION_TYPES = {"hybrid", "discrete", "continuous"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generic PPO trainer for hybrid Godot action spaces.")

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

    group = argument_group(parser, GROUP_PPO)
    group.add_argument("--ppo-epochs", type=int, default=4)
    group.add_argument(
        "--ppo-rollout-steps",
        type=int,
        default=0,
        help=(
            "Rollout length per generation. 0 (default) = DYNAMIC: each worker collects one full "
            "episode and the update waits for all workers (episode barrier). N>0 = FIXED: each worker "
            "collects exactly N steps across episode boundaries (resetting mid-rollout) and bootstraps "
            "the tail value via GAE, so no worker idles waiting for the slowest episode - the standard "
            "PPO n-steps rollout, higher/steadier throughput. Single-agent only."
        ),
    )

    group = argument_group(parser, GROUP_LOOP)
    group.add_argument("--gamma", type=float, default=0.99)

    group = argument_group(parser, GROUP_PPO)
    group.add_argument("--gae-lambda", type=float, default=0.95)
    group.add_argument("--clip-ratio", type=float, default=0.2)

    group = argument_group(parser, GROUP_LOOP)
    group.add_argument("--learning-rate", type=float, default=3e-4)

    group = argument_group(parser, GROUP_PPO)
    group.add_argument("--value-loss-coef", type=float, default=0.5)
    group.add_argument("--entropy-coef", type=float, default=0.01)
    group.add_argument("--ppo-grad-clip", type=float, default=0.0,
                        help="Global-norm gradient clip for the PPO update. 0 (default) = OFF = unchanged.")
    group.add_argument("--ppo-target-kl", type=float, default=0.0,
                        help="Approx-KL early stop: skip the rest of the epochs once mean KL exceeds "
                             "1.5x this. 0 (default) = OFF = unchanged.")
    group.add_argument("--initial-log-std", type=float, default=-0.5)
    group.add_argument(
        "--ppo-log-std-max",
        type=float,
        default=2.0,
        help=(
            "Upper bound on the (state-independent) policy log-std, clipped after every optimizer step. "
            "The entropy bonus pushes log-std up; when the reward gradient is weak early on it can run "
            "away (std -> huge, policy uselessly random). Cap it near the initial value (e.g. -0.5) to "
            "keep exploration bounded. Default 2.0 is effectively no cap."
        ),
    )
    group.add_argument(
        "--ppo-log-std-min",
        type=float,
        default=-20.0,
        help="Lower bound on the policy log-std (prevents a premature collapse to a deterministic policy).",
    )
    # Residual mode learns a bounded correction around a frozen base policy.

    group = argument_group(parser, GROUP_RESIDUAL)
    group.add_argument("--policy-mode", choices=["standard", "residual"], default="standard")
    group.add_argument("--base-policy", default=None,
                        help="Frozen base policy (.keras/.h5/.weights.h5) for --policy-mode residual; NOT trained/checkpointed.")
    group.add_argument("--residual-delta-max", type=float, default=0.005,
                        help="Max |residual correction| per joint when the gate is fully open.")
    group.add_argument("--residual-gate-config", default=None,
                        help="Gate spec: JSON path or 'outer,inner,err_start[,err_size]' (obs-space near-target gate).")
    group.add_argument("--residual-update-mask", choices=["gate", "all"], default="gate",
                        help="'gate': train the policy only where the residual gate is open; 'all': everywhere.")
    group.add_argument("--freeze-base-policy", action=argparse.BooleanOptionalAction, default=True,
                        help="Residual mode requires this True: the base (and its embedded residual) is a "
                             "separate frozen model, never in the trainable variables. --no-freeze-base-policy "
                             "is refused (base training is not implemented and would break the design).")

    group = argument_group(parser, GROUP_LOOP)
    group.add_argument("--value-learning-rate", type=float, default=0.0,
                        help="Residual mode only: LR of a SEPARATE optimizer for the value tower (needs "
                             "separate_value_tower). 0 (default) = single optimizer (standard, unchanged).")
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
    group.add_argument("--weights-path", default="generic_ppo_hybrid.weights.h5")
    group.add_argument(
        "--policy-path",
        default=None,
        help="Warm-start the policy from a .keras model, full .h5 model, or .weights.h5 file.",
    )
    group.add_argument("--checkpoint-dir", default="checkpoints/generic_ppo_hybrid")
    group.add_argument("--resume-checkpoint", default=None)
    group.add_argument("--checkpoint-every", type=int, default=25)
    group.add_argument("--keep-checkpoints", type=int, default=5)
    group.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

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
    return parser.parse_args()


def save_training_checkpoint(checkpoint, checkpoint_manager, episode, args, final=False):
    checkpoint.episode.assign(episode)
    saved_path = checkpoint_manager.save(checkpoint_number=episode)
    print(f"Saved {'final checkpoint' if final else 'checkpoint'}: {saved_path}", flush=True)
    policy_artifact = getattr(args, "policy_artifact", None)
    if policy_artifact is not None:
        policy_artifact.save(episode)
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
        # A pure discrete space expects a scalar rather than a hybrid mapping.
        "single_discrete": len(discrete) == 1 and not continuous,
        # A pure continuous space expects a flat array rather than a hybrid mapping.
        "single_continuous": bool(continuous) and not discrete,
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

    # Box decoding expects the bare continuous array.
    if action_meta["single_continuous"]:
        return np.asarray(continuous_action, dtype=np.float32).tolist()

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


def build_sample_action_fn(model, obs_dim, action_meta, device="/CPU:0", rng_gen=None):
    """Trace the stochastic action sample into one graph per collector.

    The eager op-by-op version corrupts tensor shapes when several collector threads
    run it concurrently -- a 0-D tensor where a 1-D one is expected -- crashing roughly
    two runs in five at --num-envs 4. A traced concrete function is safe to call from
    multiple threads where eager execution is not, the same fix build_greedy_action_fn
    applies to DQN. Pinning to `device` keeps the ops with the collector's CPU weights.

    rng_gen: an explicit tf.random.Generator (CHECKPOINTABLE) for the continuous Gaussian draw. Used in
    residual mode so a resumed run reproduces the exact action sampling; None keeps the global
    tf.random.normal (standard, unchanged).
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

            # This constant branch disappears from discrete-only traces.
            if continuous_size > 0:
                mean = continuous_mean[0]
                std = tf.exp(log_std)
                noise = rng_gen.normal(tf.shape(mean)) if rng_gen is not None else tf.random.normal(tf.shape(mean))
                raw_action = mean + noise * std
                # Score log-probability before the adapter maps the raw sample to the environment.
                continuous_out = raw_action
                log_prob += gaussian_log_prob(raw_action[None, :], continuous_mean, log_std)[0]
            else:
                continuous_out = tf.zeros((0,), dtype=tf.float32)

            return discrete_out, continuous_out, log_prob, value[0]

    return sample


def _standard_adapter_for(action_meta):
    # Reuse adapters because this path runs once per environment step.
    key = id(action_meta)
    ad = _STANDARD_ADAPTERS.get(key)
    if ad is None:
        ad = StandardActionAdapter(action_meta["continuous_low"], action_meta["continuous_high"])
        _STANDARD_ADAPTERS[key] = ad
    return ad


def select_action(sample_fn, log_std, obs, action_meta, adapter=None):
    obs_batch = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)
    log_std_tensor = tf.convert_to_tensor(log_std, dtype=tf.float32)
    discrete_t, continuous_t, log_prob_t, value_t = sample_fn(obs_batch, log_std_tensor)
    discrete_actions = discrete_t.numpy().astype(np.int32)
    raw_continuous = continuous_t.numpy().astype(np.float32)

    adapter = adapter or _standard_adapter_for(action_meta)
    env_continuous, stored_continuous, diag = adapter.transform(obs, raw_continuous)
    return {
        "env_action": pack_action(discrete_actions, env_continuous, action_meta),
        "discrete_actions": discrete_actions,
        "continuous_action": stored_continuous.astype(np.float32),
        "log_prob": float(log_prob_t.numpy()),
        "value": float(value_t.numpy()),
        "train_mask": float(adapter.train_mask(obs)),
        "action_diag": diag,
    }


def residual_should_zero_init(is_resuming, policy_path):
    """Zero-init the residual mean head ONLY for a truly FRESH policy: NOT on a resume (an explicit boolean
    -- a valid resume from ckpt-0 has start_episode==0 yet must NOT be re-zeroed) and NOT when warm-starting
    from --policy-path (which loaded a trained residual to keep)."""
    return (not bool(is_resuming)) and policy_path is None


def validate_residual_mode(args):
    """Fail-closed entry guard. Residual PPO is implemented ONLY for the sync single-policy path; the
    multi-policy branch returns before the adapter is built and the multi-agent loop calls select_action
    without an adapter, so both would SILENTLY train standard (unprotected) actions. Refuse explicitly."""
    if getattr(args, "policy_mode", "standard") != "residual":
        return
    if getattr(args, "multi_policy", False):
        raise RuntimeError("--policy-mode residual is not implemented for --multi-policy "
                           "(the multi-policy path has no residual adapter). Refusing fail-closed.")
    if getattr(args, "multi_agent", False):
        raise RuntimeError("--policy-mode residual is not implemented for --multi-agent "
                           "(the multi-agent loop applies no residual adapter). Refusing fail-closed.")
    if getattr(args, "collector_mode", "sync") == "async":
        raise RuntimeError("--policy-mode residual is validated for SYNC collectors only; "
                           "use --collector-mode sync.")
    if getattr(args, "auto_recovery", False):
        # Generic TensorFlow recovery cannot restore the residual policy bundle yet.
        raise RuntimeError("--policy-mode residual does not support --auto-recovery yet (recovery restores a "
                           "TensorFlow checkpoint; the residual best is a policy.keras and its weights live in "
                           "a versioned .weights.h5). Run without --auto-recovery.")
    if not getattr(args, "freeze_base_policy", True):
        # The base model is deliberately outside the residual optimizer and checkpoint.
        raise RuntimeError("--policy-mode residual requires a frozen base; --no-freeze-base-policy is not "
                           "supported (base training would break the protected-residual design).")


def build_action_adapter(args, model, action_meta, *, zero_init_head=True):
    """Return the PPO action adapter. Standard -> None (select_action uses the cached StandardAdapter,
    bit-identical to before). Residual -> a ResidualActionAdapter around a FROZEN base policy: zero-inits
    the residual mean head so the initial policy == the base, loads + hashes the frozen base, keeps it out
    of the trainable variables, and (for now) requires SYNC collectors."""
    if getattr(args, "policy_mode", "standard") != "residual":
        return None
    if int(action_meta["continuous_size"]) <= 0:
        raise RuntimeError("--policy-mode residual requires a continuous action space")
    if getattr(args, "collector_mode", "sync") == "async":
        raise RuntimeError("--policy-mode residual is validated for SYNC collectors only; async residual "
                           "with versioned snapshots is a separate step. Use --collector-mode sync.")
    if not args.base_policy:
        raise RuntimeError("--policy-mode residual requires --base-policy")
    from core.policy_artifact import load_policy_model
    base_model, _mani, _kind, _bpath = load_policy_model(args.base_policy)
    if base_model is None:
        # A frozen base needs a complete model graph, not only weights.
        raise RuntimeError(
            f"--policy-mode residual requires a COMPLETE base model (.keras or a full .h5 with architecture); "
            f"got a weights-only file with no architecture: {args.base_policy}. Re-export the base as policy.keras.")
    _low = np.asarray(action_meta["continuous_low"], np.float32)
    _high = np.asarray(action_meta["continuous_high"], np.float32)

    def base_act_fn(obs):
        o = tf.convert_to_tensor(np.asarray(obs, np.float32)[None, :], tf.float32)
        return np.clip(base_model(o, training=False).numpy()[0], _low, _high)   # clip to the ENV bounds, not [-1,1]

    if zero_init_head:                                        # deterministic residual 0 -> policy == base
        mean_layer = model.get_layer("continuous_mean")
        mean_layer.set_weights([np.zeros_like(w) for w in mean_layer.get_weights()])
    gc = load_gate_config(args.residual_gate_config)
    base_sha = _adapter_sha_file(args.base_policy) if Path(args.base_policy).exists() else None
    adapter = ResidualActionAdapter(
        base_act_fn, low=action_meta["continuous_low"], high=action_meta["continuous_high"],
        err_start=gc["err_start"], err_size=gc["err_size"], gate_outer=gc["outer"], gate_inner=gc["inner"],
        delta_max=float(args.residual_delta_max), update_mask=args.residual_update_mask, base_sha=base_sha)
    adapter.base_export_model = base_model   # kept so main() can fuse a standalone effective policy.keras for evaluation/export
    print(f"PPO residual policy-mode: base={args.base_policy} delta_max={args.residual_delta_max} "
          f"gate(outer={gc['outer']},inner={gc['inner']},err[{gc['err_start']}:{gc['err_start']+gc['err_size']}]) "
          f"update_mask={args.residual_update_mask} base_sha={str(base_sha)[:12]} zero_init={zero_init_head}", flush=True)
    return adapter


# Shared checkpoint helpers for residual training and protected evaluations.
def _atomic_write_json(path, obj):
    p = Path(path); tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    os.replace(tmp, p)


def residual_manifest(args, adapter, obs_dim, action_size, task_config_path=None):
    """Versioned manifest saved next to a residual checkpoint. Everything a resume must re-validate. Built
    from the adapter when available, else derived from args (the checkpoint is built before the adapter)."""
    import hashlib
    tcsum = None
    if task_config_path and Path(task_config_path).exists():
        tcsum = hashlib.sha256(Path(task_config_path).read_bytes()).hexdigest()
    if adapter is not None:
        d = adapter.describe()
        base_sha = d.get("base_sha"); delta_max = d.get("delta_max")
        gate_o = d.get("gate_outer"); gate_i = d.get("gate_inner")
        err_s = d.get("err_start"); err_z = d.get("err_size"); umask = d.get("update_mask")
    else:
        gc = load_gate_config(args.residual_gate_config)
        bp = getattr(args, "base_policy", None)
        base_sha = _adapter_sha_file(bp) if (bp and Path(bp).exists()) else None
        delta_max = float(args.residual_delta_max); gate_o = gc["outer"]; gate_i = gc["inner"]
        err_s = gc["err_start"]; err_z = gc["err_size"]; umask = args.residual_update_mask
    return {"schema": "residual-ppo-checkpoint/1", "policy_mode": getattr(args, "policy_mode", "standard"),
            "base_sha": base_sha, "obs_dim": int(obs_dim), "action_size": int(action_size),
            "residual_delta_max": delta_max, "gate_outer": gate_o, "gate_inner": gate_i,
            "err_start": err_s, "err_size": err_z, "update_mask": umask,
            "actor_lr": float(getattr(args, "learning_rate", getattr(args, "actor_learning_rate", 0.0))),
            "value_lr": float(getattr(args, "value_learning_rate", 0.0)),
            "actor_arch": "hybrid_actor_critic:linear_head+separate_value_tower", "task_config_sha": tcsum}


def validate_residual_manifest(saved, expected):
    """Fail-closed: any mismatch in the frozen-contract fields aborts the resume with a clear error."""
    keys = ["policy_mode", "base_sha", "obs_dim", "action_size", "residual_delta_max",
            "gate_outer", "gate_inner", "err_start", "err_size", "update_mask"]
    mism = [(k, saved.get(k), expected.get(k)) for k in keys if saved.get(k) != expected.get(k)]
    if mism:
        raise RuntimeError("residual checkpoint manifest MISMATCH (fail-closed): "
                           + "; ".join(f"{k}: ckpt={a!r} vs run={b!r}" for k, a, b in mism))


def materialize_optimizer_slots(model, log_std, optimizer, value_optimizer, obs_dim, action_size, action_meta):
    """Create the optimizer slot variables (Keras-3 optimizer.build) so a checkpoint restore fills them
    immediately. build() does NOT apply a step (no iterations bump, no weight change) -- unlike a dummy
    apply -- so the slots exist as fresh zeros ready to receive the restored momentum/velocity."""
    if value_optimizer is not None:
        actor_vars, value_vars = _split_actor_value_vars(model, log_std, action_size)
        optimizer.build(actor_vars); value_optimizer.build(value_vars)
    else:
        tv = model.trainable_variables + ([log_std] if action_size > 0 else [])
        optimizer.build(tv)


def save_residual_checkpoint(ckdir, ck_manager, manifest, np_rng, counters, model):
    """Atomic + crash-safe. Keras-3 keras.Model weights are NOT reliably tracked by tf.train.Checkpoint,
    so the MODEL is persisted with model.save_weights (versioned) and the rest (log_std, BOTH optimizers,
    tf RNG, counters) via the tf.train.Checkpoint. Order: tf save -> versioned model weights -> sidecar
    (manifest+NumPy RNG+counters) -> publish latest_complete.json LAST. A crash between steps leaves
    latest_complete pointing at the previous fully-written checkpoint."""
    ckdir = Path(ckdir)
    save_path = ck_manager.save()                              # log_std / optimizers / tf_gen / counters
    n = int(str(save_path).split("-")[-1])
    tmp = ckdir / f"model-{n}.tmp.weights.h5"; model.save_weights(tmp); os.replace(tmp, ckdir / f"model-{n}.weights.h5")
    _atomic_write_json(ckdir / f"sidecar-{n}.json",
                       {"manifest": manifest, "np_rng_state": (np_rng.bit_generator.state if np_rng is not None else None),
                        "counters": counters, "tf_ckpt": str(save_path), "model_weights": f"model-{n}.weights.h5"})
    _atomic_write_json(ckdir / "latest_complete.json", {"n": n, "tf_ckpt": str(save_path)})   # publish AFTER all artefacts
    return str(save_path)


def load_residual_checkpoint(ckdir, tf_checkpoint, expected_manifest, np_rng, model, resume_path=None):
    """Restore a COMPLETE residual checkpoint, validate the manifest fail-closed, restore model weights +
    the tf.train.Checkpoint + the NumPy RNG, and EXPLICITLY require the mandatory components (no
    expect_partial masking). resume_path=None -> the last COMPLETE checkpoint (partial saves ignored via
    latest_complete); resume_path='.../ckpt-N' -> that SPECIFIC checkpoint (requires its sidecar + versioned
    model weights + a matching manifest). Returns (counters, saved_manifest, tf_ckpt_path)."""
    ckdir = Path(ckdir)
    if resume_path is not None:
        n = int(str(resume_path).split("-")[-1]); tf_path = str(resume_path)
    else:
        lc = ckdir / "latest_complete.json"
        if not lc.exists():
            raise RuntimeError(f"no complete residual checkpoint to resume in {ckdir}")
        meta = json.loads(lc.read_text()); n = int(meta["n"]); tf_path = meta["tf_ckpt"]
    sc = ckdir / f"sidecar-{n}.json"
    if not sc.exists():
        raise RuntimeError(f"sidecar-{n}.json missing (partial checkpoint) -- refusing to resume")
    side = json.loads(sc.read_text())
    validate_residual_manifest(side["manifest"], expected_manifest)
    mw = ckdir / f"model-{n}.weights.h5"
    if not mw.exists():
        raise RuntimeError(f"model-{n}.weights.h5 missing (partial checkpoint) -- refusing to resume")
    # Missing optimizer or RNG state makes a residual resume invalid.
    present = set(tf.train.load_checkpoint(tf_path).get_variable_to_shape_map().keys())
    required = ["log_std", "optimizer", "value_optimizer", "tf_gen", "generation", "policy_updates"]
    missing = [r for r in required if not any(k == r or k.startswith(r + "/") or ("/" + r + "/") in k for k in present)]
    if missing:
        raise RuntimeError(f"residual checkpoint missing required components {missing} -- refusing to resume")
    model.load_weights(mw)
    tf_checkpoint.restore(tf_path)
    if np_rng is not None and side.get("np_rng_state") is not None:
        np_rng.bit_generator.state = side["np_rng_state"]
    return side["counters"], side["manifest"], tf_path


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
        # Adapters may mask states where their policy must remain frozen.
        "train_masks": [],
        # Non-terminal tails carry V(s_T) for GAE bootstrapping.
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
    trajectory["train_masks"].append(float(selected.get("train_mask", 1.0)))


def trajectory_has_samples(trajectory):
    return len(trajectory["rewards"]) > 0


def build_update_batch(trajectories, action_meta, gamma, gae_lambda):
    obs_parts = []
    discrete_parts = []
    continuous_parts = []
    log_prob_parts = []
    return_parts = []
    advantage_parts = []
    train_mask_parts = []

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
        # NumPy cannot infer rows when the second dimension is zero.
        discrete_parts.append(
            np.asarray(trajectory["discrete_actions"], dtype=np.int32).reshape(
                len(trajectory["rewards"]),
                len(action_meta["discrete"]),
            )
        )
        # State the row count explicitly for a zero-width continuous block.
        continuous_parts.append(
            np.asarray(trajectory["continuous_actions"], dtype=np.float32).reshape(
                len(trajectory["rewards"]),
                action_meta["continuous_size"],
            )
        )
        log_prob_parts.append(np.asarray(trajectory["log_probs"], dtype=np.float32))
        return_parts.append(returns)
        advantage_parts.append(advantages)
        # Default to all-ones when a trajectory predates the adapter (keeps the loss == the old mean).
        masks = trajectory.get("train_masks") or [1.0] * len(trajectory["rewards"])
        train_mask_parts.append(np.asarray(masks, dtype=np.float32))

    if not obs_parts:
        return None

    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "discrete_actions": np.concatenate(discrete_parts, axis=0),
        "continuous_actions": np.concatenate(continuous_parts, axis=0),
        "log_probs": np.concatenate(log_prob_parts, axis=0),
        "returns": np.concatenate(return_parts, axis=0),
        "advantages": np.concatenate(advantage_parts, axis=0),
        "train_masks": np.concatenate(train_mask_parts, axis=0),
    }


_PPO_MINIBATCH_STEPS = {}
_STANDARD_ADAPTERS = {}   # action_meta id -> cached StandardActionAdapter (adapter=None call sites)


def _split_actor_value_vars(model, log_std, continuous_size):
    """Split trainable variables into (actor, value) by the "value_*" tower name tags (only present when
    the model was built with separate_value_tower=True). log_std is an actor variable."""
    value_vars = [v for v in model.trainable_variables if "value" in getattr(v, "path", v.name).lower()]
    value_ids = {id(v) for v in value_vars}
    actor_vars = [v for v in model.trainable_variables if id(v) not in value_ids]
    if continuous_size > 0:
        actor_vars = actor_vars + [log_std]
    return actor_vars, value_vars


def build_ppo_minibatch_step(model, log_std, optimizer, action_meta, args, *, value_optimizer=None,
                             compiled=True, xla=False):
    """PPO minibatch update, optionally compiled into a tf.function.

    Eager runs the forward/backward/apply as individual ops with a host sync per minibatch;
    a traced graph fuses them (typically 5-10x). The epoch/shuffle/gather loop stays in
    Python. Mirrors build_sac_learner_step / build_dqn_learner_step.

    value_optimizer: when provided (residual PPO), the value loss is applied to the value-tower variables
    by THIS optimizer (its own LR) and the policy/entropy loss to the actor variables by `optimizer` --
    so the value network can use a higher LR without touching the residual policy. None (default) keeps
    the single-optimizer path bit-identical to standard PPO.
    """
    continuous_size = int(action_meta["continuous_size"])
    clip_ratio = float(args.clip_ratio)
    value_loss_coef = float(args.value_loss_coef)
    entropy_coef = float(args.entropy_coef)
    log_std_min = float(getattr(args, "ppo_log_std_min", -20.0))
    log_std_max = float(getattr(args, "ppo_log_std_max", 2.0))
    grad_clip = float(getattr(args, "ppo_grad_clip", 0.0))
    split = value_optimizer is not None
    if split:
        actor_vars, value_vars = _split_actor_value_vars(model, log_std, continuous_size)
    else:
        train_vars = model.trainable_variables + ([log_std] if continuous_size > 0 else [])

    def _clip(grads):
        return tf.clip_by_global_norm(grads, grad_clip)[0] if grad_clip > 0.0 else grads

    def minibatch_step(obs, discrete_actions, continuous_actions, old_log_probs, advantages, returns, train_masks):
        with tf.GradientTape(persistent=split) as tape:
            new_log_probs, entropy, values = evaluate_actions(
                model, log_std, obs, discrete_actions, continuous_actions, action_meta
            )
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            surrogate = tf.minimum(ratio * advantages, clipped_ratio * advantages)
            mask_sum = tf.reduce_sum(train_masks) + 1e-8
            policy_loss = -tf.reduce_sum(train_masks * surrogate) / mask_sum
            value_loss = tf.reduce_mean(tf.square(returns - values))
            entropy_bonus = tf.reduce_sum(train_masks * entropy) / mask_sum
            actor_loss = policy_loss - entropy_coef * entropy_bonus
            value_term = value_loss_coef * value_loss
            loss = actor_loss + value_term
        if split:                                            # residual: separate actor/value optimizers
            optimizer.apply_gradients(zip(_clip(tape.gradient(actor_loss, actor_vars)), actor_vars))
            value_optimizer.apply_gradients(zip(_clip(tape.gradient(value_term, value_vars)), value_vars))
            del tape
        else:                                                # standard: single optimizer over all vars
            optimizer.apply_gradients(zip(_clip(tape.gradient(loss, train_vars)), train_vars))
        if continuous_size > 0:
            log_std.assign(tf.clip_by_value(log_std, log_std_min, log_std_max))
        return loss, policy_loss, value_loss, entropy_bonus

    if compiled:
        return tf.function(minibatch_step, reduce_retracing=True, jit_compile=xla)
    return minibatch_step


def _get_ppo_minibatch_step(model, log_std, optimizer, action_meta, args, value_optimizer=None):
    # Build once per (model, log_std, optimizer[, value_optimizer]): rebuilding per rollout would
    # re-trace the graph every update. One learner model per run, so this holds a single entry.
    compiled = bool(getattr(args, "tf_compile_learner", True))
    xla = bool(getattr(args, "tf_xla", False))
    key = (id(model), id(log_std), id(optimizer), id(value_optimizer), compiled, xla)
    step = _PPO_MINIBATCH_STEPS.get(key)
    if step is None:
        step = build_ppo_minibatch_step(
            model, log_std, optimizer, action_meta, args, value_optimizer=value_optimizer, compiled=compiled, xla=xla
        )
        _PPO_MINIBATCH_STEPS[key] = step
        mode = "compiled graph" if compiled else "eager"
        print(f"PPO learner: {mode}{' + XLA' if compiled and xla else ''}"
              f"{' + separate value optimizer' if value_optimizer is not None else ''}", flush=True)
    return step


def ppo_update(model, log_std, optimizer, batch, action_meta, args, value_optimizer=None, shuffle_rng=None):
    obs = tf.convert_to_tensor(batch["obs"], dtype=tf.float32)
    discrete_actions = tf.convert_to_tensor(batch["discrete_actions"], dtype=tf.int32)
    continuous_actions = tf.convert_to_tensor(batch["continuous_actions"], dtype=tf.float32)
    old_log_probs = tf.convert_to_tensor(batch["log_probs"], dtype=tf.float32)
    returns = tf.convert_to_tensor(batch["returns"], dtype=tf.float32)
    advantages = tf.convert_to_tensor(batch["advantages"], dtype=tf.float32)
    mask_np = np.asarray(batch.get("train_masks", np.ones(batch["obs"].shape[0], np.float32)), np.float64)
    train_masks = tf.convert_to_tensor(mask_np, dtype=tf.float32)
    active = mask_np > 0.5

    # Normalize over trainable states; a standard policy uses the full batch.
    adv_np = np.asarray(batch["advantages"], np.float64)
    sub = adv_np[active] if active.sum() > 1 else adv_np
    advantages = tf.convert_to_tensor((adv_np - sub.mean()) / (sub.std() + 1e-8), dtype=tf.float32)
    count = obs.shape[0]
    indices = np.arange(count)
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []

    minibatch_step = _get_ppo_minibatch_step(model, log_std, optimizer, action_meta, args, value_optimizer)
    target_kl = float(getattr(args, "ppo_target_kl", 0.0))
    epochs_run = 0; approx_kl = 0.0; approx_kl_global = 0.0; approx_kl_signed = 0.0
    _shuffle = shuffle_rng.shuffle if shuffle_rng is not None else np.random.shuffle   # explicit rng in residual
    for _ in range(args.ppo_epochs):
        _shuffle(indices)
        for start in range(0, count, args.batch_size):
            idx = indices[start:start + args.batch_size]
            loss, policy_loss, value_loss, entropy_bonus = minibatch_step(
                tf.gather(obs, idx),
                tf.gather(discrete_actions, idx),
                tf.gather(continuous_actions, idx),
                tf.gather(old_log_probs, idx),
                tf.gather(advantages, idx),
                tf.gather(returns, idx),
                tf.gather(train_masks, idx),
            )
            losses.append(float(loss.numpy()))
            policy_losses.append(float(policy_loss.numpy()))
            value_losses.append(float(value_loss.numpy()))
            entropies.append(float(entropy_bonus.numpy()))
        epochs_run += 1
        if target_kl > 0.0:                                  # approx-KL early stop (0 = OFF = unchanged)
            new_lp, _, _ = evaluate_actions(model, log_std, obs, discrete_actions, continuous_actions, action_meta)
            log_ratio = (new_lp - old_log_probs).numpy()
            # Schulman's estimator stays non-negative when signed terms cancel.
            schulman = np.expm1(log_ratio) - log_ratio
            approx_kl = float(np.mean(schulman[active])) if active.sum() > 0 else float(np.mean(schulman))  # gate-active, non-neg -> early-stop/rollback
            approx_kl_global = float(np.mean(schulman))
            approx_kl_signed = float(np.mean(-log_ratio))    # signed diagnostic ONLY (may be negative)
            if approx_kl > 1.5 * target_kl:
                break

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "approx_kl": approx_kl, "approx_kl_global": approx_kl_global, "approx_kl_signed": approx_kl_signed,
        "epochs_run": epochs_run, "n_active": int(active.sum()), "n_total": int(len(mask_np)),
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
    budget=None,
):
    validate_async_arguments(args)
    budget = budget or TrainingBudget(getattr(args, "total_timesteps", 0))
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
    health_monitor = getattr(best_tracker, "health_monitor", None)
    recovery_handler = getattr(health_monitor, "recovery_handler", None)
    if recovery_handler is not None:
        recovery_handler.set_policy_publisher(
            lambda: snapshot.publish(model.get_weights(), state=log_std.numpy())
        )

    def worker(worker_id, env, _allocator, put, stop_event):
        local_model = local_models[worker_id]
        sample_fn = sample_fns[worker_id]
        policy_version = -1
        rollout_steps = int(getattr(args, "ppo_rollout_steps", 0))
        if rollout_steps > 0 and not args.multi_agent:
            # Fixed chunks reset completed episodes immediately and bootstrap non-terminal tails.
            episode_counter = start_episode
            need_reset = True
            obs = None
            episode_reward = 0.0
            for generation in range(start_episode, args.num_episodes):
                if stop_event.is_set():
                    return
                policy_version, local_log_std = snapshot.sync_model_with_state(
                    local_model, policy_version
                )
                local_log_std = tf.convert_to_tensor(local_log_std, dtype=tf.float32)
                trajectory = new_trajectory()
                chunk_rewards = []
                last_terminated = False
                for _ in range(rollout_steps):
                    if stop_event.is_set():
                        return
                    if need_reset:
                        env.configure(
                            training_episode=episode_counter,
                            max_steps=args.max_steps_per_episode,
                            physics_frames_per_step=args.physics_frames_per_step,
                            training_mode=True,
                            **scenario_curriculum_config(args),
                        )
                        obs, _info = env.reset(
                            seed=args.episode_seed_multiplier * episode_counter + worker_id
                        )
                        need_reset = False
                        episode_reward = 0.0
                    selected = select_action(sample_fn, local_log_std, obs, action_meta)
                    selected_obs = obs.copy()
                    next_obs, reward, terminated, truncated, _info = env.step(
                        selected["env_action"]
                    )
                    append_transition(
                        trajectory, selected_obs, selected, float(reward), bool(terminated)
                    )
                    episode_reward += float(reward)
                    obs = next_obs
                    last_terminated = bool(terminated)
                    if terminated or truncated:
                        chunk_rewards.append(episode_reward)
                        episode_counter += 1
                        need_reset = True
                if stop_event.is_set():
                    return
                # Bootstrap the tail value when the chunk ends mid-episode (no real terminal on the last
                # step): GAE needs V(s_next) to account for the unseen future return.
                if not last_terminated and obs is not None:
                    trajectory["bootstrap_value"] = value_of(
                        sample_fn, local_log_std, obs, action_meta
                    )
                rollout_trajectories = (
                    [trajectory] if trajectory_has_samples(trajectory) else []
                )
                rewards = (
                    float(np.mean(chunk_rewards))
                    if chunk_rewards
                    else float(episode_reward)
                )
                payload = {
                    "policy_version": policy_version,
                    "trajectories": rollout_trajectories,
                    "rewards": rewards,
                }
                if not put(AsyncEpisodeEvent(worker_id, generation, payload)):
                    return
                # Keep every worker on the same policy version for a generation.
                if generation + 1 < args.num_episodes:
                    if not snapshot.wait_for_newer(policy_version, stop_event):
                        return
            return
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
                training_mode=True,
                **scenario_curriculum_config(args),
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
    if recovery_handler is not None:
        recovery_handler.set_post_restore(
            lambda request: {
                "replay_buffer": "not applicable to on-policy PPO",
                "stabilization": (
                    "pre-recovery rollout generations are rejected and policy updates "
                    "remain frozen until immediate verification completes"
                ),
                "async_queue": "kept for generation accounting; stale policy versions are discarded",
            }
        )
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
            if len(versions) != 1:
                raise RuntimeError(
                    f"PPO generation {completed} mixed collector policy versions: "
                    f"{sorted(versions)}"
                )
            stale_after_recovery = next(iter(versions)) != snapshot.version
            verification_freeze = bool(
                getattr(health_monitor, "verification_pending", False)
            )
            trajectories = [
                trajectory
                for payload in payloads
                for trajectory in payload["trajectories"]
            ]
            rewards_summary = [payload["rewards"] for payload in payloads]
            update_batch = None if stale_after_recovery or verification_freeze else build_update_batch(
                trajectories, action_meta, args.gamma, args.gae_lambda
            )
            if update_batch is None:
                metrics = None
            else:
                budget.consume(len(update_batch["obs"]))
                metrics = ppo_update(model, log_std, optimizer, update_batch, action_meta, args)

            completed += 1
            policy_version = (
                snapshot.version
                if stale_after_recovery
                else snapshot.publish(model.get_weights(), state=log_std.numpy())
            )
            training_metrics = [
                ("completed", f"{completed}/{args.num_episodes}"),
                ("total_timesteps", budget.collected),
                ("queue", pool.events.qsize()),
                ("policy_version", policy_version),
            ]
            if stale_after_recovery:
                training_metrics.append(("skipped", "pre_recovery_policy"))
            elif verification_freeze:
                training_metrics.append(("skipped", "recovery_verification"))
            elif metrics is None:
                training_metrics.append(("skipped", "no_samples"))
            else:
                training_metrics.extend([
                    ("samples", len(update_batch["obs"])),
                    ("loss", f"{metrics['loss']:.5f}"),
                    ("policy_loss", f"{metrics['policy_loss']:.5f}"),
                    ("value_loss", f"{metrics['value_loss']:.5f}"),
                    ("entropy", f"{metrics['entropy']:.5f}"),
                ])
            # Flat numeric mean return, so the dashboard's reward chart (and any metric sink) has a
            # plottable value; rewards_summary itself is a list of per-agent episode returns.
            _reward_values = []
            for _r in rewards_summary:
                if isinstance(_r, (list, tuple)):
                    _reward_values.extend(float(_x) for _x in _r)
                else:
                    try:
                        _reward_values.append(float(_r))
                    except (TypeError, ValueError):
                        pass
            _reward_mean = float(np.mean(_reward_values)) if _reward_values else 0.0
            print_episode_metrics(event.episode, [
                ("mode", [("collector", "async_on_policy"), ("workers", len(envs))]),
                ("outcome", [
                    ("reward", f"{_reward_mean:.3f}"),
                    ("rewards", rewards_summary),
                ]),
                ("training", training_metrics),
            ], args.log_format)
            pending.pop(event.episode, None)

            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                saved_path = save_training_checkpoint(checkpoint, checkpoint_manager, completed, args)
                last_saved_episode = completed
            else:
                saved_path = None
            if best_tracker.should_evaluate(completed):
                if saved_path is None:
                    saved_path = save_training_checkpoint(checkpoint, checkpoint_manager, completed, args)
                    last_saved_episode = completed
                request_best_checkpoint_evaluation(best_tracker, saved_path, completed)
            if budget.exhausted:
                print(
                    f"Transition budget reached: {budget.collected}/{budget.limit}",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupt received: stopping async PPO collectors...", flush=True)
    finally:
        pool.close()

    if last_saved_episode != completed:
        save_training_checkpoint(checkpoint, checkpoint_manager, completed, args, final=True)
    if interrupted:
        print(f"Interrupted async PPO training saved at episode={completed}", flush=True)
    else:
        # Collect the evaluation requested near the last episode. Skipped on interrupt:
        # Ctrl-C should exit, not wait.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
    return completed


@dataclass
class PPOPolicyState:
    policy_id: str
    model: object
    log_std: object
    optimizer: object
    artifact: PolicyArtifactSaver
    sample_fn: object
    trainable: bool


def _save_multi_policy_ppo_checkpoint(
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
        "ppo",
        episode=episode,
        policy_metadata={
            policy_id: {
                "log_std": np.asarray(state.log_std.numpy()).tolist(),
            }
            for policy_id, state in policy_states.items()
        },
    )
    return saved_path


def run_async_multi_policy_ppo(
    args,
    envs,
    policy_states,
    assignment,
    action_meta,
    checkpoint,
    checkpoint_manager,
    best_tracker,
    start_episode,
    budget,
):
    env0 = envs[0]
    with tf.device("/CPU:0"):
        local_models = [
            {
                policy_id: build_hybrid_actor_critic(
                    obs_dim=env0.obs_dim,
                    discrete_sizes=action_meta["discrete_sizes"],
                    continuous_size=action_meta["continuous_size"],
                )
                for policy_id in assignment.policy_ids
            }
            for _env in envs
        ]
    sample_fns = [
        {
            policy_id: build_sample_action_fn(
                model,
                env0.obs_dim,
                action_meta,
            )
            for policy_id, model in worker_models.items()
        }
        for worker_models in local_models
    ]

    def live_weights():
        return {
            policy_id: state.model.get_weights()
            for policy_id, state in policy_states.items()
        }

    def live_log_stds():
        return {
            policy_id: state.log_std.numpy()
            for policy_id, state in policy_states.items()
        }

    snapshot = MultiPolicySnapshot(
        live_weights(),
        state_by_policy=live_log_stds(),
    )
    health_monitor = getattr(best_tracker, "health_monitor", None)
    recovery_handler = getattr(health_monitor, "recovery_handler", None)
    if recovery_handler is not None:
        recovery_handler.set_policy_publisher(
            lambda: snapshot.publish(
                live_weights(),
                state_by_policy=live_log_stds(),
            )
        )
        recovery_handler.set_post_restore(
            lambda _request: {
                "replay_buffer": "not applicable to on-policy PPO",
                "stabilization": (
                    "pre-recovery multi-policy rollout generations are rejected"
                ),
                "async_queue": (
                    "kept for generation accounting; stale group versions are discarded"
                ),
            }
        )

    def worker(worker_id, env, _allocator, put, stop_event):
        worker_models = local_models[worker_id]
        worker_sample_fns = sample_fns[worker_id]
        policy_version = -1
        for episode in range(start_episode, args.num_episodes):
            if stop_event.is_set():
                return
            policy_version, log_std_values = snapshot.sync_model_with_state(
                worker_models,
                policy_version,
            )
            local_log_stds = {
                policy_id: tf.convert_to_tensor(
                    log_std_values[policy_id],
                    dtype=tf.float32,
                )
                for policy_id in assignment.policy_ids
            }
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
            done_mask = np.asarray(
                info.get(
                    "per_agent_done",
                    np.zeros((len(env.agent_ids),), dtype=np.bool_),
                ),
                dtype=np.bool_,
            )
            trajectories = {
                agent_id: new_trajectory()
                for agent_id in env.agent_ids
                if policy_states[
                    assignment.policy_for_agent(agent_id)
                ].trainable
            }
            episode_reward = np.zeros(
                (len(env.agent_ids),),
                dtype=np.float32,
            )
            global_done = False
            for _step_idx in episode_step_indices(
                args.max_steps_per_episode
            ):
                if stop_event.is_set() or global_done:
                    break
                action_payload = {}
                selected_by_agent = {}
                for agent_idx, agent_id in enumerate(env.agent_ids):
                    if done_mask[agent_idx]:
                        action_payload[agent_id] = zero_env_action(
                            action_meta
                        )
                        continue
                    policy_id = assignment.policy_for_agent(agent_id)
                    selected = select_action(
                        worker_sample_fns[policy_id],
                        local_log_stds[policy_id],
                        obs[agent_idx],
                        action_meta,
                    )
                    action_payload[agent_id] = selected["env_action"]
                    if policy_states[policy_id].trainable:
                        selected_by_agent[agent_id] = (
                            agent_idx,
                            policy_id,
                            selected,
                            obs[agent_idx].copy(),
                        )

                next_obs, _reward, terminated, truncated, step_info = (
                    env.step(action_payload)
                )
                global_done = bool(terminated or truncated)
                cut_short = bool(truncated and not terminated)
                rewards = np.asarray(
                    step_info.get("per_agent_rewards"),
                    dtype=np.float32,
                )
                per_agent_done = np.asarray(
                    step_info.get("per_agent_done"),
                    dtype=np.bool_,
                )
                per_agent_terminated = np.asarray(
                    step_info.get(
                        "per_agent_terminated",
                        per_agent_done,
                    ),
                    dtype=np.bool_,
                )
                episode_reward += rewards
                for (
                    agent_id,
                    (
                        agent_idx,
                        policy_id,
                        selected,
                        selected_obs,
                    ),
                ) in selected_by_agent.items():
                    trajectory = trajectories[agent_id]
                    append_transition(
                        trajectory,
                        selected_obs,
                        selected,
                        float(rewards[agent_idx]),
                        bool(
                            terminated
                            or per_agent_terminated[agent_idx]
                        ),
                    )
                    if (
                        cut_short
                        and not per_agent_terminated[agent_idx]
                    ):
                        trajectory["bootstrap_value"] = value_of(
                            worker_sample_fns[policy_id],
                            local_log_stds[policy_id],
                            next_obs[agent_idx],
                            action_meta,
                        )
                obs = next_obs
                done_mask = np.logical_or(done_mask, per_agent_done)
                global_done = global_done or bool(np.all(done_mask))

            if stop_event.is_set():
                return
            by_policy = {
                policy_id: []
                for policy_id in assignment.policy_ids
            }
            for agent_id, trajectory in trajectories.items():
                if trajectory_has_samples(trajectory):
                    by_policy[
                        assignment.policy_for_agent(agent_id)
                    ].append(trajectory)
            payload = {
                "policy_version": policy_version,
                "trajectories": by_policy,
                "rewards": episode_reward.tolist(),
            }
            if not put(AsyncEpisodeEvent(worker_id, episode, payload)):
                return
            if episode + 1 < args.num_episodes:
                if not snapshot.wait_for_newer(
                    policy_version,
                    stop_event,
                ):
                    return

    pool = AsyncCollectorPool(
        envs,
        worker,
        start_episode,
        args.num_episodes,
        queue_capacity=args.async_queue_capacity,
    )
    print(
        "Collector mode: async on-policy multi-policy "
        f"workers={len(envs)} policies={list(assignment.policy_ids)} "
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
                    raise RuntimeError(
                        "All multi-policy PPO collectors stopped before completion"
                    )
                continue
            if isinstance(event, AsyncWorkerErrorEvent):
                raise RuntimeError(
                    f"Async multi-policy PPO collector {event.worker_id} failed"
                ) from event.error
            if isinstance(event, AsyncWorkerDoneEvent):
                done_workers += 1
                continue
            if not isinstance(event, AsyncEpisodeEvent):
                continue

            generation = pending.setdefault(event.episode, {})
            generation[event.worker_id] = event.payload
            if event.episode != completed or len(generation) < len(envs):
                continue
            payloads = [
                generation[worker_id]
                for worker_id in range(len(envs))
            ]
            versions = {
                payload["policy_version"]
                for payload in payloads
            }
            if len(versions) != 1:
                raise RuntimeError(
                    f"PPO multi-policy generation {completed} mixed group "
                    f"versions: {sorted(versions)}"
                )
            stale = next(iter(versions)) != snapshot.version
            verification_freeze = bool(
                getattr(health_monitor, "verification_pending", False)
            )
            by_policy_metrics = {}
            for policy_id, state in policy_states.items():
                trajectories = [
                    trajectory
                    for payload in payloads
                    for trajectory in payload["trajectories"][policy_id]
                ]
                batch = (
                    None
                    if (
                        stale
                        or verification_freeze
                        or not state.trainable
                        or not trajectories
                    )
                    else build_update_batch(
                        trajectories,
                        action_meta,
                        args.gamma,
                        args.gae_lambda,
                    )
                )
                metrics = None
                if state.trainable and batch is not None:
                    budget.consume(len(batch["obs"]))
                    metrics = ppo_update(
                        state.model,
                        state.log_std,
                        state.optimizer,
                        batch,
                        action_meta,
                        args,
                    )
                by_policy_metrics[policy_id] = {
                    "samples": len(batch["obs"]) if batch is not None else 0,
                    "loss": (
                        metrics["loss"] if metrics is not None else 0.0
                    ),
                    "policy_loss": (
                        metrics["policy_loss"]
                        if metrics is not None
                        else 0.0
                    ),
                    "value_loss": (
                        metrics["value_loss"]
                        if metrics is not None
                        else 0.0
                    ),
                }

            completed += 1
            policy_version = (
                snapshot.version
                if stale
                else snapshot.publish(
                    live_weights(),
                    state_by_policy=live_log_stds(),
                )
            )
            training_metrics = [
                ("completed", f"{completed}/{args.num_episodes}"),
                ("total_timesteps", budget.collected),
                ("queue", pool.events.qsize()),
                ("by_policy", by_policy_metrics),
                ("policy_version", policy_version),
            ]
            if stale:
                training_metrics.append(
                    ("skipped", "pre_recovery_policy")
                )
            elif verification_freeze:
                training_metrics.append(
                    ("skipped", "recovery_verification")
                )
            print_episode_metrics(event.episode, [
                ("mode", [
                    ("collector", "async_on_policy"),
                    ("workers", len(envs)),
                    ("policies", len(policy_states)),
                ]),
                (
                    "outcome",
                    [(
                        "rewards",
                        [payload["rewards"] for payload in payloads],
                    )],
                ),
                ("training", training_metrics),
            ], args.log_format)
            pending.pop(event.episode, None)

            if (
                args.checkpoint_every > 0
                and completed % args.checkpoint_every == 0
            ):
                saved_path = _save_multi_policy_ppo_checkpoint(
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
                    saved_path = _save_multi_policy_ppo_checkpoint(
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
            if budget.exhausted:
                break
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\nInterrupt received: stopping async multi-policy PPO collectors...",
            flush=True,
        )
    finally:
        pool.close()

    if last_saved_episode != completed:
        _save_multi_policy_ppo_checkpoint(
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
    return completed


def run_sync_multi_policy_ppo(
    args,
    envs,
    stepper,
    action_meta,
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
            "Independent multi-policy PPO warm starts currently use "
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
        model = build_hybrid_actor_critic(
            obs_dim=env0.obs_dim,
            discrete_sizes=action_meta["discrete_sizes"],
            continuous_size=action_meta["continuous_size"],
        )
        model(np.zeros((1, env0.obs_dim), dtype=np.float32), training=False)
        log_std = tf.Variable(
            np.full(
                (action_meta["continuous_size"],),
                args.initial_log_std,
                dtype=np.float32,
            ),
            name=f"log_std_{assignment.key_for(policy_id)}",
            trainable=True,
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate
        )
        metadata = build_policy_metadata(
            "ppo",
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
        state = PPOPolicyState(
            policy_id=policy_id,
            model=model,
            log_std=log_std,
            optimizer=optimizer,
            artifact=PolicyArtifactSaver(
                model,
                Path(args.checkpoint_dir)
                / "policies"
                / assignment.key_for(policy_id),
                metadata,
            ),
            sample_fn=build_sample_action_fn(
                model,
                env0.obs_dim,
                action_meta,
            ),
            trainable=assignment.is_trainable(policy_id),
        )
        policy_states[policy_id] = state
        policy_trackables[assignment.key_for(policy_id)] = tf.train.Checkpoint(
            model=model,
            log_std=log_std,
            optimizer=optimizer,
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
            if str(stored.get("algorithm", "ppo")) != "ppo":
                raise RuntimeError(
                    f"Multi-policy checkpoint algorithm={stored.get('algorithm')!r}, "
                    "expected 'ppo'"
                )
        checkpoint.restore(resume_checkpoint).expect_partial()
        start_episode = int(checkpoint.episode.numpy())
        print(
            f"Resumed multi-policy checkpoint {resume_checkpoint} "
            f"from episode={start_episode}",
            flush=True,
        )
    for state in policy_states.values():
        state.optimizer.learning_rate.assign(args.learning_rate)

    write_multi_policy_manifest(
        args.checkpoint_dir,
        assignment,
        "ppo",
        episode=start_episode,
    )
    best_tracker.configure_recovery(
        checkpoint,
        [
            (f"policy_{assignment.key_for(policy_id)}", state.optimizer)
            for policy_id, state in policy_states.items()
            if state.trainable
        ],
    )
    print(
        "Independent multi-policy PPO: "
        f"assignment={assignment.mode} policies={list(assignment.policy_ids)} "
        f"trainable={list(assignment.trainable_policy_ids)} "
        f"agent_map={assignment.agent_to_policy}",
        flush=True,
    )

    if args.collector_mode == "async":
        return run_async_multi_policy_ppo(
            args,
            envs,
            policy_states,
            assignment,
            action_meta,
            checkpoint,
            checkpoint_manager,
            best_tracker,
            start_episode,
            budget,
        )

    last_completed_episode = start_episode
    last_saved_episode = None
    interrupted = False
    try:
        for episode in range(start_episode, args.num_episodes):
            trajectories = {
                policy_id: []
                for policy_id in assignment.policy_ids
            }
            rewards_summary = []
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
                done_mask = np.asarray(
                    info.get(
                        "per_agent_done",
                        np.zeros((len(env.agent_ids),), dtype=np.bool_),
                    ),
                    dtype=np.bool_,
                )
                env_states.append({
                    "obs": obs,
                    "done": False,
                    "done_mask": done_mask,
                    "episode_reward": np.zeros(
                        (len(env.agent_ids),),
                        dtype=np.float32,
                    ),
                    "trajectories": {
                        agent_id: new_trajectory()
                        for agent_id in env.agent_ids
                        if policy_states[
                            assignment.policy_for_agent(agent_id)
                        ].trainable
                    },
                })

            for _step in episode_step_indices(args.max_steps_per_episode):
                requests = []
                for env, env_state in zip(envs, env_states):
                    if env_state["done"]:
                        continue
                    action_payload = {}
                    selected_by_agent = {}
                    for agent_idx, agent_id in enumerate(env.agent_ids):
                        if env_state["done_mask"][agent_idx]:
                            action_payload[agent_id] = zero_env_action(
                                action_meta
                            )
                            continue
                        policy_id = assignment.policy_for_agent(agent_id)
                        policy = policy_states[policy_id]
                        selected = select_action(
                            policy.sample_fn,
                            policy.log_std,
                            env_state["obs"][agent_idx],
                            action_meta,
                        )
                        action_payload[agent_id] = selected["env_action"]
                        if policy.trainable:
                            selected_by_agent[agent_id] = (
                                agent_idx,
                                policy_id,
                                selected,
                                env_state["obs"][agent_idx].copy(),
                            )
                    env_state["selected_by_agent"] = selected_by_agent
                    requests.append((env, env_state, action_payload))

                for env, env_state, _action, step_result in stepper.step(
                    requests
                ):
                    next_obs, _reward, terminated, truncated, info = step_result
                    global_done = bool(terminated or truncated)
                    cut_short = bool(truncated and not terminated)
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
                    env_state["episode_reward"] += rewards
                    for (
                        agent_id,
                        (
                            agent_idx,
                            policy_id,
                            selected,
                            agent_obs,
                        ),
                    ) in env_state["selected_by_agent"].items():
                        trajectory = env_state["trajectories"][agent_id]
                        append_transition(
                            trajectory,
                            agent_obs,
                            selected,
                            float(rewards[agent_idx]),
                            bool(
                                terminated
                                or per_agent_terminated[agent_idx]
                            ),
                        )
                        if (
                            cut_short
                            and not per_agent_terminated[agent_idx]
                        ):
                            policy = policy_states[policy_id]
                            trajectory["bootstrap_value"] = value_of(
                                policy.sample_fn,
                                policy.log_std,
                                next_obs[agent_idx],
                                action_meta,
                            )
                    env_state["obs"] = next_obs
                    env_state["done_mask"] = np.logical_or(
                        env_state["done_mask"],
                        per_agent_done,
                    )
                    env_state["done"] = (
                        global_done
                        or bool(np.all(env_state["done_mask"]))
                    )
                if all(state["done"] for state in env_states):
                    break

            for env, env_state in zip(envs, env_states):
                rewards_summary.append(
                    env_state["episode_reward"].tolist()
                )
                for agent_id, trajectory in env_state["trajectories"].items():
                    if trajectory_has_samples(trajectory):
                        trajectories[
                            assignment.policy_for_agent(agent_id)
                        ].append(trajectory)

            verification_freeze = bool(
                best_tracker.health_monitor.verification_pending
            )
            policy_metrics = {}
            for policy_id, policy in policy_states.items():
                batch = build_update_batch(
                    trajectories[policy_id],
                    action_meta,
                    args.gamma,
                    args.gae_lambda,
                )
                sample_count = (
                    len(batch["obs"])
                    if batch is not None
                    else 0
                )
                if policy.trainable and sample_count:
                    budget.consume(sample_count)
                metrics = None
                if (
                    policy.trainable
                    and batch is not None
                    and not verification_freeze
                ):
                    metrics = ppo_update(
                        policy.model,
                        policy.log_std,
                        policy.optimizer,
                        batch,
                        action_meta,
                        args,
                    )
                policy_metrics[policy_id] = {
                    "trainable": policy.trainable,
                    "samples": sample_count,
                    "loss": (
                        f"{metrics['loss']:.5f}"
                        if metrics is not None
                        else "0.00000"
                    ),
                    "policy_loss": (
                        f"{metrics['policy_loss']:.5f}"
                        if metrics is not None
                        else "0.00000"
                    ),
                    "value_loss": (
                        f"{metrics['value_loss']:.5f}"
                        if metrics is not None
                        else "0.00000"
                    ),
                }

            print_episode_metrics(episode, [
                ("mode", [
                    ("policies", len(policy_states)),
                    ("assignment", assignment.mode),
                ]),
                ("outcome", [("rewards", rewards_summary)]),
                ("training", [
                    ("total_timesteps", budget.collected),
                    ("by_policy", policy_metrics),
                    *(
                        [("skipped", "recovery_verification")]
                        if verification_freeze
                        else []
                    ),
                ]),
            ], args.log_format)
            last_completed_episode = episode + 1

            apply_ready_best_checkpoint(best_tracker)
            if (
                args.checkpoint_every > 0
                and (episode + 1) % args.checkpoint_every == 0
            ):
                saved_path = _save_multi_policy_ppo_checkpoint(
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
                    saved_path = _save_multi_policy_ppo_checkpoint(
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
            "\nInterrupt received: saving the multi-policy PPO state...",
            flush=True,
        )

    if last_saved_episode != last_completed_episode:
        _save_multi_policy_ppo_checkpoint(
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
    return last_completed_episode


def main():
    args = parse_args()
    validate_residual_mode(args)          # fail-closed BEFORE the multi-policy branch can return unprotected
    budget = TrainingBudget(args.total_timesteps)
    validate_async_arguments(args)
    best_tracker = BestCheckpointTracker(args, "ppo")
    describe_tensorflow_backend(args)
    dashboard = maybe_start_dashboard(args, algorithm="ppo")
    training_start_time = time.monotonic()
    # Architecture is process-wide state, so it must be fixed before the first network is
    # built -- the seeding block below is the last point where nothing exists yet.
    set_network_layers(args.network_layers or default_network_layers("ppo"))
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

        if args.multi_policy:
            run_sync_multi_policy_ppo(
                args,
                envs,
                stepper,
                action_meta,
                best_tracker,
                budget,
            )
            return

        # The adapter owns tanh in residual mode, and its value tower must not update the actor.
        _residual = getattr(args, "policy_mode", "standard") == "residual"
        model = build_hybrid_actor_critic(
            obs_dim=obs_dim,
            discrete_sizes=action_meta["discrete_sizes"],
            continuous_size=action_meta["continuous_size"],
            continuous_activation="linear" if _residual else "tanh",
            separate_value_tower=_residual,
        )
        model(np.zeros((1, obs_dim), dtype=np.float32), training=False)
        args.policy_artifact = PolicyArtifactSaver(
            model,
            args.checkpoint_dir,
            build_policy_metadata("ppo", env0),
        )
        log_std = tf.Variable(
            np.full((action_meta["continuous_size"],), args.initial_log_std, dtype=np.float32),
            name="continuous_log_std",
            trainable=True,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        # Residual checkpoints always include the separate value optimizer.
        if _residual and float(getattr(args, "value_learning_rate", 0.0)) <= 0.0:
            args.value_learning_rate = 3e-4
        value_optimizer = (tf.keras.optimizers.Adam(learning_rate=args.value_learning_rate)
                           if (_residual and float(getattr(args, "value_learning_rate", 0.0)) > 0.0) else None)
        start_episode = 0
        # Residual runs persist both TensorFlow and NumPy random streams.
        episode_var = tf.Variable(0, dtype=tf.int64)
        if _residual:
            residual_tf_gen = tf.random.Generator.from_seed(int(args.env_seed_base))
            residual_shuffle_rng = np.random.default_rng(int(args.env_seed_base))
            generation_var = tf.Variable(0, dtype=tf.int64)
            policy_updates_var = tf.Variable(0, dtype=tf.int64)
            residual_manifest_obj = residual_manifest(args, None, obs_dim, action_meta["continuous_size"],
                                                      getattr(args, "task_config", None))
        else:
            residual_tf_gen = residual_shuffle_rng = None
            generation_var = policy_updates_var = None
            residual_manifest_obj = None

        _ckpt_kw = {"model": model, "log_std": log_std, "optimizer": optimizer, "episode": episode_var}
        if value_optimizer is not None:
            _ckpt_kw["value_optimizer"] = value_optimizer
        if _residual:
            _ckpt_kw.update(tf_gen=residual_tf_gen, generation=generation_var, policy_updates=policy_updates_var)
        checkpoint = tf.train.Checkpoint(**_ckpt_kw)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.checkpoint_dir,
            max_to_keep=args.keep_checkpoints,
        )
        resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_manager)
        if resume_checkpoint and args.policy_path:
            raise RuntimeError("--policy-path cannot be combined with --resume or --resume-checkpoint")
        if resume_checkpoint and _residual:
            # Materialize optimizer slots before restoring the complete residual state.
            materialize_optimizer_slots(model, log_std, optimizer, value_optimizer, obs_dim,
                                        action_meta["continuous_size"], action_meta)
            counters, _saved, _p = load_residual_checkpoint(args.checkpoint_dir, checkpoint, residual_manifest_obj,
                                                            residual_shuffle_rng, model, resume_path=resume_checkpoint)
            start_episode = int(counters.get("episode", checkpoint.episode.numpy()))
            print(f"Resumed RESIDUAL checkpoint from episode={start_episode} gen={counters.get('generation')}", flush=True)
        elif resume_checkpoint:
            checkpoint.restore(resume_checkpoint).expect_partial()
            start_episode = int(checkpoint.episode.numpy())
            print(f"Resumed checkpoint {resume_checkpoint} from episode={start_episode}", flush=True)
        elif args.policy_path:
            loaded_policy = load_policy_into_model(model, args.policy_path, expected_algorithm="ppo")
            print(
                f"Warm-started PPO policy from {loaded_policy['source_kind']}: "
                f"{loaded_policy['path']} (fresh optimizer and rollout state, episode=0)",
                flush=True,
            )
        optimizer.learning_rate.assign(args.learning_rate)
        # Zero-initialize only a fresh residual policy; resumes and warm starts keep their weights.
        _fresh_residual = residual_should_zero_init(resume_checkpoint is not None, args.policy_path)
        action_adapter = build_action_adapter(args, model, action_meta, zero_init_head=_fresh_residual)

        def _save_ckpt(ep, final=False):
            # Residual saves publish their model, optimizer, RNG and sidecar atomically.
            if _residual:
                episode_var.assign(int(ep)); generation_var.assign(int(ep))
                saved = save_residual_checkpoint(
                    args.checkpoint_dir, checkpoint_manager, residual_manifest_obj, residual_shuffle_rng,
                    {"episode": int(ep), "generation": int(ep), "policy_updates": int(policy_updates_var.numpy())}, model)
                # Export the fused policy so deployment needs no residual-aware runtime.
                _artifact = getattr(args, "policy_artifact", None)
                if _artifact is not None:
                    _artifact.save(int(ep))
                return saved
            return save_training_checkpoint(checkpoint, checkpoint_manager, ep, args, final=final)

        best_tracker.configure_recovery(
            checkpoint,
            [("policy", optimizer)],
        )

        if _residual:
            # Evaluation uses the live fused policy, including the environment action bounds.
            from core.composite_policy import build_residual_export_policy

            _effective_manifest = build_policy_metadata("ppo", env0)

            def _build_effective_policy():
                d = action_adapter.describe()
                return build_residual_export_policy(
                    action_adapter.base_export_model, model,
                    delta_max=d["delta_max"], gate_outer=d["gate_outer"], gate_inner=d["gate_inner"],
                    err_start=d["err_start"], err_size=d["err_size"],
                    action_low=action_meta["continuous_low"], action_high=action_meta["continuous_high"])

            def _export_effective_bundle(dest):
                # Keep the exported policy and its manifest together.
                _build_effective_policy().save(str(dest))
                Path(dest).with_suffix(".json").write_text(json.dumps(_effective_manifest, indent=2))

            # Each residual checkpoint includes a standalone effective policy bundle.
            args.policy_artifact = PolicyArtifactSaver(
                _build_effective_policy(), args.checkpoint_dir, _effective_manifest)
            # Best-checkpoint evaluation receives a frozen fused snapshot.
            best_tracker.set_policy_artifact_exporter(_export_effective_bundle)

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
                budget,
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
                # Residual sampling uses its checkpointed RNG for reproducible resumes.
                _rng = residual_tf_gen if (_residual and sampled_model is model) else None
                fn = build_sample_action_fn(sampled_model, obs_dim, action_meta, rng_gen=_rng)
                sample_fns[id(sampled_model)] = fn
            return fn

        recovery_handler = best_tracker.health_monitor.recovery_handler
        if recovery_handler is not None:
            recovery_handler.set_post_restore(
                lambda request: {
                    "replay_buffer": "not applicable to on-policy PPO",
                    "stabilization": (
                        "the next rollout is collected with restored weights and policy "
                        "updates remain frozen until immediate verification completes"
                    ),
                }
            )

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
                    training_mode=True,
                    **scenario_curriculum_config(args),
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
                        selected = select_action(sample_fn_for(model), log_std, state["obs"], action_meta,
                                                 adapter=action_adapter)
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

            sample_count = len(update_batch["rewards"]) if "rewards" in update_batch else len(update_batch["obs"])
            budget.consume(sample_count)
            verification_freeze = bool(
                best_tracker.health_monitor.verification_pending
            )
            metrics = (
                None
                if verification_freeze
                else ppo_update(model, log_std, optimizer, update_batch, action_meta, args, value_optimizer,
                                shuffle_rng=residual_shuffle_rng)
            )
            # Count only updates that contained at least one trainable residual state.
            if _residual and metrics is not None and int(metrics.get("n_active", 0)) > 0:
                policy_updates_var.assign_add(1)
            training_metrics = [
                ("samples", sample_count),
                ("total_timesteps", budget.collected),
            ]
            if verification_freeze:
                training_metrics.append(("skipped", "recovery_verification"))
            else:
                training_metrics.extend([
                    ("loss", f"{metrics['loss']:.5f}"),
                    ("policy_loss", f"{metrics['policy_loss']:.5f}"),
                    ("value_loss", f"{metrics['value_loss']:.5f}"),
                    ("entropy", f"{metrics['entropy']:.5f}"),
                ])
            print_episode_metrics(episode, [
                ("mode", [("opponent", opponent_match.label)]),
                ("outcome", [("rewards", rewards_summary)]),
                ("training", training_metrics),
            ], args.log_format)
            last_completed_episode = episode + 1
            snapshot_path = opponent_pool.snapshot(model, episode + 1)
            if snapshot_path is not None:
                print(f"Saved opponent snapshot: {snapshot_path}", flush=True)

            apply_ready_best_checkpoint(best_tracker)
            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                saved_path = _save_ckpt(episode + 1)
                last_saved_episode = episode + 1
            else:
                saved_path = None
            if best_tracker.should_evaluate(episode + 1):
                # The tracker evaluates the staged fused policy through the standard path.
                if saved_path is None:
                    saved_path = _save_ckpt(episode + 1)
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
            _save_ckpt(completed_episode, final=True)
        # The evaluation requested on the last episode is still running; without this the
        # finally-block's close() cancels it and a final best can never be promoted.
        apply_ready_best_checkpoint(best_tracker, wait_timeout=args.best_final_drain_timeout)
        model.save_weights(args.weights_path)
        print(f"Saved weights: {args.weights_path}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupt received: saving the last consistent PPO state...", flush=True)
        if checkpoint is not None and checkpoint_manager is not None and model is not None:
            interrupted_episode = last_completed_episode if last_completed_episode is not None else start_episode
            _save_ckpt(interrupted_episode)
            model.save_weights(args.weights_path)
            print(f"Interrupted training saved at episode={interrupted_episode}", flush=True)
        else:
            print("Training state was not initialized; no checkpoint was written.", flush=True)
    finally:
        report_training_time(dashboard, training_start_time)
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
