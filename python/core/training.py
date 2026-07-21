import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from itertools import count
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np


def episode_step_indices(max_steps):
    """Iterate episode step indices; zero means no Python-side time limit."""
    max_steps = int(max_steps)
    if max_steps < 0:
        raise ValueError("Episode max steps cannot be negative")
    return count() if max_steps == 0 else range(max_steps)


def add_tensorflow_runtime_arguments(parser, *, include_compile_learner=False):
    parser.add_argument(
        "--gpu-memory-growth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Let TensorFlow grow CUDA GPU memory usage on demand instead of reserving most "
            "available VRAM at startup. This setting is ignored by non-CUDA backends."
        ),
    )
    if include_compile_learner:
        parser.add_argument(
            "--tf-compile-learner",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Compile learner updates into a TensorFlow graph and batch consecutive updates.",
        )
        parser.add_argument(
            "--tf-xla",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Compile the learner graph with XLA (jit_compile). Measured SLOWER for the "
                "small actor-critic MLPs used here (fusion overhead exceeds the gain); off by "
                "default. Enable with --tf-xla only for large networks where XLA pays off."
            ),
        )


RENDER_MODES = ("project", "cpu", "light-gpu", "gpu")


def add_godot_render_argument(parser, *, default="light-gpu"):
    parser.add_argument(
        "--render-mode",
        choices=RENDER_MODES,
        default=default,
        help=(
            "Renderer for non-headless Godot instances (ignored when headless). 'project' "
            "leaves the Godot project setting untouched; 'light-gpu' uses the OpenGL "
            "compatibility renderer and on Linux falls back to a switcherooctl discrete "
            "GPU when default OpenGL is software-rendered (light GPU, low TensorFlow "
            "contention); 'cpu' forces "
            "software rendering via Mesa llvmpipe (no GPU, lower FPS); 'gpu' forces the full "
            "Vulkan forward+ renderer."
        ),
    )


def add_best_checkpoint_arguments(parser):
    parser.add_argument(
        "--best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Periodically evaluate a frozen policy and retain independently selected best checkpoints.",
    )
    parser.add_argument(
        "--best-checkpoint-dir",
        default=None,
        help="Best-checkpoint destination; defaults to CHECKPOINT_DIR/best.",
    )
    parser.add_argument("--keep-best-checkpoints", type=int, default=3)
    parser.add_argument(
        "--best-evaluation-every",
        type=int,
        default=100,
        help="Completed training episodes between deterministic best-policy evaluations.",
    )
    parser.add_argument("--best-evaluation-episodes", type=int, default=20)
    parser.add_argument("--best-evaluation-seed", type=int, default=10_000)
    parser.add_argument(
        "--best-evaluation-training-episode",
        type=int,
        default=None,
        help="Fixed scenario curriculum episode used for evaluation; defaults to --num-episodes.",
    )
    parser.add_argument(
        "--best-evaluation-max-steps",
        type=int,
        default=None,
        help=(
            "Evaluation time limit; defaults to --max-steps-per-episode, or 10000 when training "
            "episodes are unlimited. Use 0 explicitly for an unlimited evaluation."
        ),
    )
    parser.add_argument(
        "--best-evaluation-port",
        type=int,
        default=None,
        help="Port for the isolated evaluation environment; defaults after the training environment ports.",
    )
    parser.add_argument("--best-evaluation-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--best-final-drain-timeout",
        type=float,
        default=120.0,
        help=(
            "Seconds to wait, when training finishes normally, for an evaluation that is "
            "still running so its result is not thrown away. An evaluation requested on "
            "the last episode would otherwise never be collected. Zero exits immediately; "
            "interrupting with Ctrl-C never waits."
        ),
    )
    parser.add_argument(
        "--best-evaluation-device",
        choices=["cpu", "auto"],
        default="cpu",
        help="Device exposed to the isolated policy evaluator.",
    )
    parser.add_argument(
        "--best-evaluation-cpu-threads",
        type=int,
        default=1,
        help=(
            "CPU threads available to a CPU best-checkpoint evaluator. Small policy "
            "networks are usually faster with one thread and do not starve training."
        ),
    )
    parser.add_argument(
        "--best-metric",
        choices=["auto", "success_rate", "reward_mean"],
        default="auto",
        help="Primary metric used to select the best frozen policy.",
    )


@dataclass(frozen=True)
class PolicyEvaluationResult:
    episode: int
    summary: dict


class BestCheckpointTracker:
    """Evaluate exact checkpoints in an isolated Godot process and rank them consistently."""

    # Grace between asking the evaluator to stop and killing it outright.
    EVALUATION_TERMINATE_GRACE = 5.0

    def __init__(self, args, algorithm):
        self.args = args
        self.algorithm = str(algorithm)
        self.enabled = bool(getattr(args, "best_checkpoint", False))
        configured_dir = getattr(args, "best_checkpoint_dir", None)
        self.directory = Path(configured_dir or (Path(args.checkpoint_dir) / "best"))
        self.metadata_path = self.directory / "best_metrics.json"
        self.staging_directory = self.directory / ".staging"
        self.best_key = None
        self.best_summary = None
        self._executor = None
        self._pending = None  # (future, episode, staged_prefix)
        self._ready = None  # (result, staged_prefix) awaiting promote or discard
        self._retained = []  # [(episode, prefix)] oldest first
        self._closing = threading.Event()
        self._process = None
        self._process_lock = threading.Lock()
        if self.enabled:
            self._validate_arguments()
            self.directory.mkdir(parents=True, exist_ok=True)
            self.staging_directory.mkdir(parents=True, exist_ok=True)
            self._clear_staging()  # drop leftovers from a killed run
            self._load_retained()
            self._load_metadata()

    def _validate_arguments(self):
        # The tracker owns and prunes its directory, so sharing it with the training
        # checkpoints would let it delete live training state.
        if self.directory.resolve() == Path(self.args.checkpoint_dir).resolve():
            raise ValueError("--best-checkpoint-dir must differ from --checkpoint-dir")
        if int(self.args.best_evaluation_every) < 1:
            raise ValueError("--best-evaluation-every must be at least 1 when best checkpoints are enabled")
        if int(self.args.best_evaluation_episodes) < 1:
            raise ValueError("--best-evaluation-episodes must be at least 1")
        if int(self.args.keep_best_checkpoints) < 1:
            raise ValueError("--keep-best-checkpoints must be at least 1")
        if float(self.args.best_evaluation_timeout) <= 0.0:
            raise ValueError("--best-evaluation-timeout must be greater than zero")
        # Zero is meaningful (exit without waiting); negative is not.
        if float(getattr(self.args, "best_final_drain_timeout", 0.0)) < 0.0:
            raise ValueError("--best-final-drain-timeout cannot be negative")
        if int(getattr(self.args, "best_evaluation_cpu_threads", 1)) < 1:
            raise ValueError("--best-evaluation-cpu-threads must be at least 1")
        if self.args.best_evaluation_training_episode is not None and int(
            self.args.best_evaluation_training_episode
        ) < 0:
            raise ValueError("--best-evaluation-training-episode cannot be negative")
        if self.args.best_evaluation_max_steps is not None and int(
            self.args.best_evaluation_max_steps
        ) < 0:
            raise ValueError("--best-evaluation-max-steps cannot be negative")

    def _load_metadata(self):
        if not self.metadata_path.is_file():
            return
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"WARNING: could not read best-checkpoint metadata: {exc}", flush=True)
            return
        if metadata.get("metric") != self.args.best_metric:
            print(
                f"Best-checkpoint metric changed from {metadata.get('metric')!r} "
                f"to {self.args.best_metric!r}; starting a new comparison baseline.",
                flush=True,
            )
            return
        comparison_key = metadata.get("comparison_key")
        if isinstance(comparison_key, list) and len(comparison_key) == 2:
            self.best_key = tuple(float(value) for value in comparison_key)
            self.best_summary = metadata

    def should_evaluate(self, episode):
        every = int(getattr(self.args, "best_evaluation_every", 0))
        return self.enabled and every > 0 and int(episode) > 0 and int(episode) % every == 0

    def comparison_key(self, summary):
        success_rate = float(summary.get("success_rate", 0.0))
        reward_mean = float(summary.get("reward_mean", -np.inf))
        if self.args.best_metric == "reward_mean":
            return reward_mean, success_rate
        return success_rate, reward_mean

    def is_improvement(self, result):
        candidate_key = self.comparison_key(result.summary)
        return self.best_key is None or candidate_key > self.best_key

    def _evaluation_command(self, checkpoint_path, summary_path):
        evaluation_port = self.args.best_evaluation_port
        if evaluation_port is None:
            evaluation_port = int(self.args.base_port) + max(int(self.args.num_envs), 1)
        training_episode = self.args.best_evaluation_training_episode
        if training_episode is None:
            training_episode = int(self.args.num_episodes)
        max_steps = self.args.best_evaluation_max_steps
        if max_steps is None:
            max_steps = int(self.args.max_steps_per_episode)
            if max_steps == 0:
                max_steps = 10_000

        runner = Path(__file__).resolve().parents[1] / "run.py"
        command = [
            sys.executable,
            str(runner),
            "--algorithm", self.algorithm,
            "--load-from", "checkpoint",
            "--checkpoint-path", str(checkpoint_path),
            "--episodes", str(self.args.best_evaluation_episodes),
            "--epsilon", "0.0",
            "--training-episode", str(training_episode),
            "--seed", str(self.args.best_evaluation_seed),
            "--max-steps", str(max_steps),
            "--port", str(evaluation_port),
            "--print-every", "0",
            "--summary-json", str(summary_path),
            "--headless",
            "--multi-agent" if self.args.multi_agent else "--no-multi-agent",
        ]
        optional_values = (
            ("--godot-bin", self.args.godot_bin),
            ("--godot-project", self.args.godot_project),
            ("--godot-scene", self.args.godot_scene),
            ("--agent-id", self.args.agent_id),
        )
        for option, value in optional_values:
            if value is not None:
                command.extend((option, str(value)))
        return command

    def evaluate(self, checkpoint_path, episode):
        with tempfile.TemporaryDirectory(prefix="godot-policy-eval-") as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            command = self._evaluation_command(checkpoint_path, summary_path)
            child_env = os.environ.copy()
            if self.args.best_evaluation_device == "cpu":
                child_env["CUDA_VISIBLE_DEVICES"] = ""
                cpu_threads = str(getattr(self.args, "best_evaluation_cpu_threads", 1))
                for variable in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "TF_NUM_INTRAOP_THREADS",
                    "TF_NUM_INTEROP_THREADS",
                ):
                    child_env[variable] = cpu_threads
            print(
                f"Evaluating frozen checkpoint episode={episode} "
                f"episodes={self.args.best_evaluation_episodes} metric={self.args.best_metric} "
                f"device={self.args.best_evaluation_device}"
                + (
                    f" cpu_threads={getattr(self.args, 'best_evaluation_cpu_threads', 1)}"
                    if self.args.best_evaluation_device == "cpu"
                    else ""
                )
                + "...",
                flush=True,
            )
            try:
                returncode, stdout, stderr = self._run_evaluation_process(
                    command, child_env, float(self.args.best_evaluation_timeout)
                )
            except subprocess.TimeoutExpired:
                print(
                    f"WARNING: best-policy evaluation exceeded {self.args.best_evaluation_timeout:.0f}s; "
                    "training will continue without updating best.",
                    flush=True,
                )
                return None
            if self._closing.is_set():
                # We killed it on the way out; an exit code from that is not a failure.
                print("Best-policy evaluation cancelled during shutdown.", flush=True)
                return None
            if returncode != 0 or not summary_path.is_file():
                details = (stderr or stdout or "no evaluator output").strip()
                print(
                    f"WARNING: best-policy evaluation failed with exit={returncode}:\n{details}",
                    flush=True,
                )
                return None
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(
                f"Frozen evaluation episode={episode}: success={summary['successes']}/{summary['trials']} "
                f"({summary['success_rate']:.2%}) reward_mean={summary['reward_mean']:.4f} "
                f"steps_mean={summary['steps_mean']:.1f}",
                flush=True,
            )
            return PolicyEvaluationResult(int(episode), summary)

    def _run_evaluation_process(self, command, child_env, timeout):
        """Run the evaluator, keeping a handle so close() can actually stop it.

        subprocess.run() gives no way to reach the child, so a shutdown had to wait out
        --best-evaluation-timeout (30 min by default) -- and ThreadPoolExecutor's atexit
        hook joins its non-daemon threads, so that wait blocked interpreter exit.

        start_new_session isolates the evaluator from the terminal's Ctrl-C so that
        _terminate_evaluation is the single path that stops it. That is only safe
        together with the killpg in _signal_evaluation: on its own it would strand the
        evaluator on Ctrl-C, which is the very hang this removes.
        """
        popen_kwargs = {"start_new_session": True} if os.name == "posix" else {}
        process = subprocess.Popen(
            command,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **popen_kwargs,
        )
        with self._process_lock:
            self._process = process
        try:
            if self._closing.is_set():
                # close() can run between the enabled-check and Popen; without this the
                # child would outlive the shutdown that was meant to take it down.
                self._terminate_evaluation()
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            self._terminate_evaluation()
            process.communicate()
            raise
        finally:
            with self._process_lock:
                self._process = None

    def _terminate_evaluation(self):
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        print("Stopping the in-flight best-policy evaluation...", flush=True)
        self._signal_evaluation(process, signal.SIGTERM)
        try:
            process.wait(timeout=self.EVALUATION_TERMINATE_GRACE)
        except subprocess.TimeoutExpired:
            self._signal_evaluation(process, signal.SIGKILL)

    @staticmethod
    def _signal_evaluation(process, signal_number):
        """Signal the evaluator's whole session, so its Godot child dies with it.

        run.py only reaps Godot from a `finally`, which a bare kill()
        skips entirely. The orphaned instance keeps holding --best-evaluation-port and
        would make every later run's evaluation fail to bind.
        """
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal_number)
                return
        except (ProcessLookupError, PermissionError, OSError):
            pass
        if signal_number == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()

    def evaluate_async(self, checkpoint_path, episode):
        """Schedule evaluate() on a background thread so it never blocks the training loop.

        Returns True if an evaluation was scheduled, False if one was already running
        (in which case this evaluation request is skipped) or best-checkpoint tracking
        is disabled.
        """
        if not self.enabled or self._closing.is_set():
            return False
        if self._pending is not None:
            print(
                f"Skipping best-checkpoint evaluation for episode={episode}: "
                "a previous evaluation is still running.",
                flush=True,
            )
            return False
        episode = int(episode)
        # Copy now, not on promotion. The training directory prunes to
        # --keep-checkpoints while the evaluation runs, so by the time a result comes
        # back the evaluated bytes may already be gone -- and the evaluator itself has
        # been reading a file the trainer was free to delete underneath it.
        try:
            staged_prefix = copy_checkpoint_files(
                checkpoint_path, self.staging_directory / f"ckpt-{episode}"
            )
        except OSError as exc:
            print(
                f"WARNING: could not stage checkpoint {checkpoint_path} for evaluation: {exc}",
                flush=True,
            )
            return False
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="best-ckpt-eval")
        future = self._executor.submit(self.evaluate, staged_prefix, episode)
        self._pending = (future, episode, staged_prefix)
        return True

    def poll_ready(self, timeout=0.0):
        """Collect a completed background evaluation.

        timeout=0.0 is the non-blocking check the training loop uses every iteration.
        A positive timeout waits, so a final evaluation requested on the last episode
        can still be collected instead of being silently thrown away at exit.
        """
        if self._pending is None:
            return None
        future, episode, staged_prefix = self._pending
        if timeout > 0.0:
            print(
                f"Waiting up to {timeout:g}s for the best-policy evaluation of episode={episode}...",
                flush=True,
            )
            try:
                future.result(timeout=timeout)
            except FutureTimeoutError:
                print(
                    f"WARNING: the evaluation of episode={episode} did not finish within "
                    f"{timeout:g}s; its result is discarded.",
                    flush=True,
                )
                return None
            except Exception:
                pass  # surfaced by the future.result() below
        elif not future.done():
            return None
        self._pending = None
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive: evaluate() already catches its own errors
            print(f"WARNING: best-policy evaluation raised an exception: {exc}", flush=True)
            result = None
        if result is None:
            remove_checkpoint_files(staged_prefix)
            return None
        # Hand the result over still attached to the bytes that produced it, so a caller
        # cannot promote anything else by accident.
        self._ready = (result, staged_prefix)
        return result

    def discard_ready(self):
        """Drop a polled result that did not improve, and its staged copy."""
        if self._ready is None:
            return
        _result, staged_prefix = self._ready
        self._ready = None
        remove_checkpoint_files(staged_prefix)

    def promote_ready(self, result):
        """Publish the exact weights that produced `result`.

        Previously the caller re-saved the *live* model here, so the metadata paired
        episode-N metrics with weights from whatever episode training had since reached.
        Moving the staged copy means the file and the metrics always agree.
        """
        if self._ready is None or self._ready[0] is not result:
            raise RuntimeError("promote_ready() requires the result returned by the last poll_ready()")
        _result, staged_prefix = self._ready
        self._ready = None
        episode = int(result.episode)
        best_prefix = self.directory / f"ckpt-{episode}"
        remove_checkpoint_files(best_prefix)
        for shard in checkpoint_shard_paths(staged_prefix):
            suffix = shard.name[len(staged_prefix.name):]
            os.replace(shard, best_prefix.with_name(best_prefix.name + suffix))
        self._retained = [entry for entry in self._retained if entry[0] != episode]
        self._retained.append((episode, best_prefix))
        self._retained.sort()
        self._prune_retained()
        self._write_checkpoint_state()
        self.record_best(result, best_prefix)
        return best_prefix

    def _load_retained(self):
        entries = []
        for index_path in self.directory.glob("ckpt-*.index"):
            prefix = index_path.with_suffix("")
            number = checkpoint_number(prefix)
            if number is not None:
                entries.append((number, prefix))
        self._retained = sorted(entries)

    def _prune_retained(self):
        while len(self._retained) > int(self.args.keep_best_checkpoints):
            _episode, prefix = self._retained.pop(0)
            remove_checkpoint_files(prefix)

    def _write_checkpoint_state(self):
        """Keep tf.train.latest_checkpoint() working on the best directory.

        run.py falls back to latest_checkpoint() when --checkpoint-path is omitted, so this
        metafile is a user-facing contract even though a CheckpointManager no longer writes
        it. Written by hand as a text-format CheckpointState proto -- the exact format
        tf.train.latest_checkpoint() parses -- to avoid the deprecated
        tf.compat.v1.train.update_checkpoint_state. Paths stay absolute: the best directory
        is not relocatable.
        """
        if not self._retained:
            return

        def _escape(path):
            return str(path).replace("\\", "\\\\").replace('"', '\\"')

        lines = [f'model_checkpoint_path: "{_escape(self._retained[-1][1])}"']
        lines += [
            f'all_model_checkpoint_paths: "{_escape(prefix)}"'
            for _episode, prefix in self._retained
        ]
        (self.directory / "checkpoint").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _clear_staging(self):
        for index_path in self.staging_directory.glob("ckpt-*.index"):
            remove_checkpoint_files(index_path.with_suffix(""))

    def close(self):
        """Stop the evaluator and its Godot child, then wait for the worker to unwind.

        wait=True is only cheap because _terminate_evaluation ran first: once the child
        is gone communicate() returns in milliseconds. Waiting is worth it -- the
        executor's threads are non-daemon and CPython joins them at exit regardless, so
        the choice is between waiting here with the child dead or waiting at exit with
        it alive.
        """
        self._closing.set()
        self._terminate_evaluation()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._pending = None
        self.discard_ready()
        if self.enabled:
            self._clear_staging()

    def record_best(self, result, checkpoint_path):
        self.best_key = self.comparison_key(result.summary)
        metadata = {
            "algorithm": self.algorithm,
            "metric": self.args.best_metric,
            "episode": int(result.episode),
            "checkpoint": str(checkpoint_path),
            "comparison_key": list(self.best_key),
            **result.summary,
        }
        temporary_path = self.metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_path.replace(self.metadata_path)
        self.best_summary = metadata
        print(
            f"New best checkpoint: {checkpoint_path} metric={self.args.best_metric} "
            f"success={result.summary['success_rate']:.2%} "
            f"reward_mean={result.summary['reward_mean']:.4f}",
            flush=True,
        )


def configure_tensorflow_devices(tf_module, *, memory_growth=True, system_name=None):
    """Configure CUDA devices before TensorFlow creates its runtime context."""
    system_name = system_name or __import__("platform").system()
    devices = list(tf_module.config.list_physical_devices("GPU"))
    memory_growth_enabled = False
    if memory_growth and system_name == "Linux":
        try:
            for device in devices:
                tf_module.config.experimental.set_memory_growth(device, True)
            memory_growth_enabled = bool(devices)
        except (RuntimeError, ValueError) as exc:
            print(f"Warning: could not enable TensorFlow GPU memory growth: {exc}", flush=True)
    return devices, memory_growth_enabled


def add_parallel_env_arguments(parser):
    parser.add_argument(
        "--parallel-env-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dispatch Godot env.step calls concurrently across environment instances.",
    )
    parser.add_argument(
        "--render-env-count",
        type=int,
        default=None,
        help=(
            "When training without --headless, render only this many envs and run the rest "
            "headless. When omitted, render all --num-envs instances."
        ),
    )
    parser.add_argument(
        "--physics-frames-per-step",
        type=int,
        default=1,
        help=(
            "Number of Godot physics ticks advanced per env.step() call (frame skip / "
            "control-rate downsampling, e.g. 4 ticks at 60Hz physics = a 15Hz control rate). "
            "Values above 1 reduce the number of TCP round-trips per simulated second, which "
            "speeds up lockstep training. Forwarded to ScenarioController.configure()."
        ),
    )


def add_lockstep_tuning_arguments(parser):
    parser.add_argument(
        "--lockstep-idle-sleep-usec",
        type=int,
        default=None,
        help=(
            "Override BridgeServer.lockstep_idle_sleep_usec via a Godot cmdline user arg. "
            "This is the backoff the bridge sleeps for once a client has gone quiet for "
            "--lockstep-spin-polls consecutive empty polls. In headless mode Godot clamps it to "
            "lockstep_headless_idle_sleep_usec (2000us by default). Left unset to keep that "
            "headless-safe default. See python/tools/bench_lockstep.py to measure the effect."
        ),
    )
    parser.add_argument(
        "--lockstep-spin-polls",
        type=int,
        default=None,
        help=(
            "Empty TCP polls the BridgeServer tolerates at zero delay before backing off to "
            "--lockstep-idle-sleep-usec. The bridge round-trip is ~0.3ms, so a short spin covers "
            "the client's turnaround and skips the idle sleep entirely on the hot path. Higher "
            "values cut latency at the cost of CPU while idle; 0 disables spinning and restores "
            "pure sleep-based backoff. Measured on 20 cores with tools/bench_lockstep.py: sleep=2000 "
            "gives ~330 steps/s per instance at 0.4 cores each, sleep=0 gives ~2700 but burns "
            "~3.2 cores per instance and stops scaling past 4 instances."
        ),
    )


def build_lockstep_user_args(args):
    """Build the list of Godot cmdline user args controlling lockstep pacing."""
    user_args = []
    lockstep_idle_sleep_usec = getattr(args, "lockstep_idle_sleep_usec", None)
    if lockstep_idle_sleep_usec is not None:
        user_args.append(f"--lockstep-idle-sleep-usec={int(lockstep_idle_sleep_usec)}")
    lockstep_spin_polls = getattr(args, "lockstep_spin_polls", None)
    if lockstep_spin_polls is not None:
        user_args.append(f"--lockstep-spin-polls={int(lockstep_spin_polls)}")
    return user_args


def add_collector_arguments(parser):
    parser.add_argument(
        "--collector-mode",
        choices=["sync", "async"],
        default="async",
        help=(
            "sync advances all environments in batches; async runs one independent collector "
            "per Godot instance while the learner trains concurrently."
        ),
    )
    parser.add_argument(
        "--async-queue-capacity",
        type=int,
        default=256,
        help="Maximum pending collector events before collectors apply backpressure.",
    )
    parser.add_argument(
        "--async-policy-sync-steps",
        type=int,
        default=100,
        help="Collector control steps between checks for a newer learner policy.",
    )
    parser.add_argument(
        "--async-policy-publish-updates",
        type=int,
        default=100,
        help="Learner updates between policy snapshots published to collectors.",
    )
    parser.add_argument(
        "--async-updates-per-step",
        type=int,
        default=1,
        help="Gradient updates performed whenever the configured collection interval is reached.",
    )
    parser.add_argument(
        "--async-update-basis",
        choices=["transitions", "env_steps"],
        default="transitions",
        help=(
            "Schedule updates from individual agent transitions (multi-agent adaptive) or "
            "from Godot environment step events (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--async-update-every",
        "--async-update-every-steps",
        dest="async_update_every",
        type=int,
        default=4,
        help=(
            "Collected units required before scheduling learner updates. The unit is selected "
            "by --async-update-basis; --async-update-every-steps is retained as a legacy alias."
        ),
    )
    parser.add_argument(
        "--async-max-updates-per-env-step",
        type=int,
        default=1,
        help=(
            "Safety cap for scheduled updates per collected Godot step. Zero disables the cap; "
            "the default lets multi-agent collection increase learning without unbounded bursts."
        ),
    )
    parser.add_argument(
        "--async-drain-max-events",
        type=int,
        default=64,
        help="Maximum queued step events ingested into replay before running scheduled updates.",
    )
    parser.add_argument(
        "--async-replay-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress replay snapshots in a background thread during async training.",
    )


class ParallelEnvStepper:
    def __init__(self, max_workers, enabled=True):
        self.enabled = bool(enabled) and int(max_workers) > 1
        self._executor = (
            ThreadPoolExecutor(max_workers=int(max_workers), thread_name_prefix="godot-env")
            if self.enabled
            else None
        )

    def step(self, requests):
        """Return (env, state, action, step_result) tuples in request order."""
        requests = list(requests)
        if not requests:
            return []
        if self._executor is None or len(requests) == 1:
            return [(env, state, action, env.step(action)) for env, state, action in requests]

        futures = [self._executor.submit(env.step, action) for env, _state, action in requests]
        return [
            (env, state, action, future.result())
            for (env, state, action), future in zip(requests, futures)
        ]

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


@dataclass(frozen=True)
class AsyncStepEvent:
    worker_id: int
    transitions: tuple


@dataclass(frozen=True)
class AsyncEpisodeEvent:
    worker_id: int
    episode: int
    payload: object


@dataclass(frozen=True)
class AsyncWorkerDoneEvent:
    worker_id: int


@dataclass(frozen=True)
class AsyncWorkerErrorEvent:
    worker_id: int
    error: BaseException


class PolicySnapshot:
    """Thread-safe immutable weight snapshots for collector-local inference models."""

    def __init__(self, weights, state=None):
        self._condition = threading.Condition()
        self._version = 0
        self._weights = self._copy_weights(weights)
        self._state = copy.deepcopy(state)

    @staticmethod
    def _copy_weights(weights):
        return tuple(np.array(weight, copy=True) for weight in weights)

    @property
    def version(self):
        with self._condition:
            return self._version

    def publish(self, weights, state=None):
        copied = self._copy_weights(weights)
        copied_state = copy.deepcopy(state)
        with self._condition:
            self._weights = copied
            self._state = copied_state
            self._version += 1
            self._condition.notify_all()
            return self._version

    def sync_model(self, model, current_version=-1):
        with self._condition:
            if current_version == self._version:
                return current_version
            version = self._version
            weights = self._copy_weights(self._weights)
        model.set_weights(weights)
        return version

    def sync_model_with_state(self, model, current_version=-1):
        with self._condition:
            if current_version == self._version:
                return current_version, copy.deepcopy(self._state)
            version = self._version
            weights = self._copy_weights(self._weights)
            state = copy.deepcopy(self._state)
        model.set_weights(weights)
        return version, state

    def wait_for_newer(self, current_version, stop_event, timeout=0.2):
        with self._condition:
            while self._version <= current_version and not stop_event.is_set():
                self._condition.wait(timeout=timeout)
            return self._version > current_version


class EpisodeAllocator:
    def __init__(self, start_episode, end_episode):
        self._next_episode = int(start_episode)
        self._end_episode = int(end_episode)
        self._lock = threading.Lock()

    def claim(self):
        with self._lock:
            if self._next_episode >= self._end_episode:
                return None
            episode = self._next_episode
            self._next_episode += 1
            return episode

class AsyncCollectorPool:
    """Runs one user-supplied collector loop per environment."""

    def __init__(self, envs, worker_fn, start_episode, end_episode, queue_capacity=256):
        if int(queue_capacity) <= 0:
            raise ValueError("--async-queue-capacity must be greater than zero")
        self.events = Queue(maxsize=int(queue_capacity))
        self.stop_event = threading.Event()
        self.allocator = EpisodeAllocator(start_episode, end_episode)
        self._threads = []
        for worker_id, env in enumerate(envs):
            thread = threading.Thread(
                target=self._run_worker,
                args=(worker_id, env, worker_fn),
                name=f"godot-collector-{worker_id}",
                daemon=True,
            )
            self._threads.append(thread)

    def _put(self, event):
        while not self.stop_event.is_set():
            try:
                self.events.put(event, timeout=0.2)
                return True
            except Full:
                continue
        return False

    def _run_worker(self, worker_id, env, worker_fn):
        try:
            worker_fn(worker_id, env, self.allocator, self._put, self.stop_event)
        except BaseException as exc:
            self._put(AsyncWorkerErrorEvent(worker_id, exc))
        finally:
            self._put(AsyncWorkerDoneEvent(worker_id))

    def start(self):
        for thread in self._threads:
            thread.start()

    def get(self, timeout=0.2):
        return self.events.get(timeout=timeout)

    def stop(self):
        self.stop_event.set()

    def close(self, timeout=5.0):
        self.stop()
        for thread in self._threads:
            thread.join(timeout=timeout)

    @property
    def alive_count(self):
        return sum(thread.is_alive() for thread in self._threads)


class AsyncEventScheduler:
    """Drains collector bursts and converts collected experience into learner updates."""

    def __init__(self, args):
        self.update_basis = str(getattr(args, "async_update_basis", "transitions"))
        self.update_every = int(
            getattr(args, "async_update_every", getattr(args, "async_update_every_steps", 4))
        )
        self.updates_per_interval = int(args.async_updates_per_step)
        self.max_updates_per_env_step = int(
            getattr(args, "async_max_updates_per_env_step", 1)
        )
        self.drain_max_events = int(args.async_drain_max_events)
        self._collection_credit = 0
        self._deferred = deque()
        self._interval_started = time.monotonic()
        self._interval_steps = 0
        self._interval_transitions = 0
        self._interval_updates = 0
        self._interval_requested_updates = 0
        self._interval_throttled_updates = 0
        self._last_throughput = None

    def next_event(self, pool, timeout=0.2):
        if self._deferred:
            return self._deferred.popleft()
        return pool.get(timeout=timeout)

    def drain_step_events(self, pool, first_event):
        events = [first_event]
        while len(events) < self.drain_max_events:
            try:
                event = pool.events.get_nowait()
            except Empty:
                break
            if isinstance(event, AsyncStepEvent):
                events.append(event)
            else:
                self._deferred.append(event)
        return events

    def ingest(self, events):
        step_count = len(events)
        transition_count = sum(len(event.transitions) for event in events)
        self._interval_steps += step_count
        self._interval_transitions += transition_count
        collected_units = transition_count if self.update_basis == "transitions" else step_count
        self._collection_credit += collected_units
        intervals, self._collection_credit = divmod(self._collection_credit, self.update_every)
        requested_updates = intervals * self.updates_per_interval

        updates_due = requested_updates
        if self.max_updates_per_env_step > 0:
            max_updates = step_count * self.max_updates_per_env_step
            updates_due = min(requested_updates, max_updates)

        self._interval_requested_updates += requested_updates
        self._interval_throttled_updates += requested_updates - updates_due
        return updates_due

    def record_updates(self, count):
        self._interval_updates += int(count)

    def throughput(self, pool):
        queue_size = pool.events.qsize()
        queue_capacity = pool.events.maxsize
        if self._interval_steps == 0 and self._last_throughput is not None:
            metrics = dict(self._last_throughput)
            metrics.update(
                queue_size=queue_size,
                queue_capacity=queue_capacity,
                queue_saturation=queue_size / max(queue_capacity, 1),
            )
            return metrics

        elapsed = max(time.monotonic() - self._interval_started, 1e-6)
        metrics = {
            "env_steps_s": self._interval_steps / elapsed,
            "transitions_s": self._interval_transitions / elapsed,
            "transitions_per_env_step": (
                self._interval_transitions / max(self._interval_steps, 1)
            ),
            "updates_s": self._interval_updates / elapsed,
            "requested_updates": self._interval_requested_updates,
            "throttled_updates": self._interval_throttled_updates,
            "queue_size": queue_size,
            "queue_capacity": queue_capacity,
            "queue_saturation": queue_size / max(queue_capacity, 1),
        }
        self._interval_started = time.monotonic()
        self._interval_steps = 0
        self._interval_transitions = 0
        self._interval_updates = 0
        self._interval_requested_updates = 0
        self._interval_throttled_updates = 0
        self._last_throughput = dict(metrics)
        return metrics


def validate_async_arguments(args, supports_opponent_pool=False):
    """Validate async collector settings.

    `supports_opponent_pool` is per-trainer and defaults to False: a trainer whose async
    path does not thread the pool through must keep failing loudly here rather than
    accept --opponent-pool and quietly ignore it.
    """
    if int(getattr(args, "max_steps_per_episode", 500)) < 0:
        raise ValueError("--max-steps-per-episode cannot be negative; use 0 for no limit")
    if args.collector_mode != "async":
        return
    if args.async_policy_sync_steps <= 0:
        raise ValueError("--async-policy-sync-steps must be greater than zero")
    if args.async_policy_publish_updates <= 0:
        raise ValueError("--async-policy-publish-updates must be greater than zero")
    if args.async_updates_per_step < 0:
        raise ValueError("--async-updates-per-step cannot be negative")
    update_every = int(
        getattr(args, "async_update_every", getattr(args, "async_update_every_steps", 4))
    )
    if update_every <= 0:
        raise ValueError("--async-update-every must be greater than zero")
    update_basis = str(getattr(args, "async_update_basis", "transitions"))
    if update_basis not in {"transitions", "env_steps"}:
        raise ValueError("--async-update-basis must be 'transitions' or 'env_steps'")
    if int(getattr(args, "async_max_updates_per_env_step", 1)) < 0:
        raise ValueError("--async-max-updates-per-env-step cannot be negative")
    if args.async_drain_max_events <= 0:
        raise ValueError("--async-drain-max-events must be greater than zero")
    if getattr(args, "opponent_pool", False) and not supports_opponent_pool:
        raise ValueError(
            "This trainer's --collector-mode async path does not support historical "
            "--opponent-pool sampling; use --collector-mode sync for that mode."
        )


def build_async_worker(
    local_models,
    policy_snapshot,
    max_steps_per_episode,
    policy_sync_steps,
    begin_episode,
    select_action,
    process_step,
    finish_episode,
):
    """Compose an algorithm-specific collector from small lifecycle callbacks."""

    max_steps = int(max_steps_per_episode)
    sync_steps = int(policy_sync_steps)

    def worker(worker_id, env, allocator, put, stop_event):
        local_model = local_models[worker_id]
        policy_version = policy_snapshot.sync_model(local_model)
        control_steps = 0
        while not stop_event.is_set():
            episode = allocator.claim()
            if episode is None:
                return
            state = begin_episode(worker_id, env, episode)
            for step_idx in episode_step_indices(max_steps):
                if stop_event.is_set() or state.get("done", False):
                    break
                if control_steps % sync_steps == 0:
                    policy_version = policy_snapshot.sync_model(local_model, policy_version)
                action = select_action(worker_id, env, episode, step_idx, local_model, state)
                step_result = env.step(action)
                transitions = process_step(
                    worker_id,
                    env,
                    episode,
                    step_idx,
                    state,
                    action,
                    step_result,
                )
                control_steps += 1
                if transitions and not put(AsyncStepEvent(worker_id, tuple(transitions))):
                    return
            payload = finish_episode(worker_id, env, episode, state)
            if not put(AsyncEpisodeEvent(worker_id, episode, payload)):
                return

    return worker


def add_log_format_argument(parser):
    parser.add_argument(
        "--log-format",
        choices=["pretty", "compact"],
        default="pretty",
        help="Pretty prints one readable block per episode; compact keeps one machine-friendly line.",
    )


def print_episode_metrics(episode, sections, log_format="pretty"):
    normalized = [
        (name, [(str(key), str(value)) for key, value in metrics])
        for name, metrics in sections
        if metrics
    ]
    if log_format == "compact":
        fields = [f"episode={episode:04d}"]
        for _name, metrics in normalized:
            for key, value in metrics:
                compact_value = json.dumps(value) if any(char.isspace() for char in value) else value
                fields.append(f"{key}={compact_value}")
        print(" ".join(fields), flush=True)
        return

    print(f"episode={episode:04d}", flush=True)
    for name, metrics in normalized:
        values = "  ".join(f"{key}={value}" for key, value in metrics)
        print(f"  {name:<9} {values}", flush=True)


def normalize_checkpoint_path(path):
    path = str(path)
    if path.endswith(".index"):
        return path[:-len(".index")]
    if ".data-" in path:
        return path.split(".data-", 1)[0]
    return path


def resolve_resume_checkpoint(args, checkpoint_manager):
    requested_value = getattr(args, "resume_checkpoint", None)
    if requested_value:
        requested = Path(requested_value).expanduser()
        if requested.name.isdigit():
            requested = Path(args.checkpoint_dir) / f"ckpt-{requested.name}"
        elif requested.parent == Path("."):
            requested = Path(args.checkpoint_dir) / requested.name
        checkpoint_path = normalize_checkpoint_path(requested)
        if not Path(checkpoint_path + ".index").is_file():
            raise FileNotFoundError(
                f"Checkpoint {checkpoint_path!r} does not exist (missing {checkpoint_path + '.index'!r})"
            )
        return checkpoint_path

    if getattr(args, "resume", False):
        if not checkpoint_manager.latest_checkpoint:
            raise FileNotFoundError(f"No checkpoint found in --checkpoint-dir {args.checkpoint_dir!r}")
        return checkpoint_manager.latest_checkpoint
    return None


def checkpoint_number(checkpoint_path):
    match = re.search(r"ckpt-(\d+)$", str(checkpoint_path))
    return int(match.group(1)) if match else None


def apply_ready_best_checkpoint(tracker, wait_timeout=0.0):
    """Promote the evaluated checkpoint itself, never the live model.

    Replaces four near-identical per-trainer copies that each re-saved the live
    checkpoint at the current episode, attributing the evaluated episode's metrics to
    weights that were never evaluated.

    wait_timeout > 0 drains an in-flight evaluation, for use when training ends normally.
    """
    result = tracker.poll_ready(timeout=wait_timeout)
    if result is None:
        return None
    if not tracker.is_improvement(result):
        tracker.discard_ready()
        return None
    return tracker.promote_ready(result)


def checkpoint_shard_paths(prefix):
    """Every file that makes up the TensorFlow checkpoint written at `prefix`.

    The '.' in the glob keeps ckpt-1 from also matching ckpt-10's shards.
    """
    prefix = Path(prefix)
    return sorted(
        path
        for path in prefix.parent.glob(prefix.name + ".*")
        if path.name == prefix.name + ".index" or ".data-" in path.name
    )


def copy_checkpoint_files(source_prefix, target_prefix):
    """Copy a checkpoint's shards so it survives --keep-checkpoints pruning.

    Each shard lands via a temporary name and an atomic replace: a reader that opens the
    destination concurrently sees either nothing or a complete file, never a partial one.
    """
    source_prefix = Path(normalize_checkpoint_path(source_prefix))
    target_prefix = Path(target_prefix)
    shards = checkpoint_shard_paths(source_prefix)
    if not any(shard.name.endswith(".index") for shard in shards):
        raise FileNotFoundError(f"Checkpoint {str(source_prefix)!r} has no .index file")
    target_prefix.parent.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        suffix = shard.name[len(source_prefix.name):]
        destination = target_prefix.with_name(target_prefix.name + suffix)
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(shard, temporary)
        temporary.replace(destination)
    return target_prefix


def remove_checkpoint_files(prefix):
    for shard in checkpoint_shard_paths(prefix):
        try:
            shard.unlink()
        except OSError:
            pass


def replay_path_for_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    number = checkpoint_number(checkpoint_path)
    filename = f"replay-{number}.npz" if number is not None else checkpoint_path.name + ".replay.npz"
    return checkpoint_path.parent / filename


def save_replay_snapshot(checkpoint_path, checkpoint_manager, buffer, asynchronous=False):
    replay_path = replay_path_for_checkpoint(checkpoint_path)
    if asynchronous:
        transitions, future = buffer.save_async(replay_path)
        print(
            f"Scheduled replay buffer save: {replay_path} transitions={transitions}",
            flush=True,
        )

        def on_complete(completed):
            try:
                saved_transitions = completed.result()
                print(
                    f"Saved replay buffer: {replay_path} transitions={saved_transitions}",
                    flush=True,
                )
                cleanup_stale_replay_buffers(checkpoint_manager)
            except Exception as exc:
                print(f"ERROR saving replay buffer {replay_path}: {exc}", flush=True)

        future.add_done_callback(on_complete)
        return replay_path

    transitions = buffer.save(replay_path)
    print(f"Saved replay buffer: {replay_path} transitions={transitions}", flush=True)
    cleanup_stale_replay_buffers(checkpoint_manager)
    return replay_path


def cleanup_stale_replay_buffers(checkpoint_manager):
    retained_numbers = {
        checkpoint_number(path)
        for path in checkpoint_manager.checkpoints
        if checkpoint_number(path) is not None
    }
    for replay_path in Path(checkpoint_manager.directory).glob("replay-*.npz"):
        match = re.fullmatch(r"replay-(\d+)\.npz", replay_path.name)
        if match and int(match.group(1)) not in retained_numbers:
            replay_path.unlink()


def restore_replay_buffer(args, checkpoint_path, buffer):
    replay_path = replay_path_for_checkpoint(checkpoint_path)
    if replay_path.is_file():
        restored = buffer.load(replay_path)
        print(
            f"Restored replay buffer: {replay_path} transitions={restored}/{args.replay_capacity}",
            flush=True,
        )
        return restored
    if getattr(args, "require_replay_buffer", False):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path!r} has no replay buffer at {str(replay_path)!r}")
    print(
        f"WARNING: no replay buffer found for {checkpoint_path}; updates remain disabled until "
        f"replay_size reaches replay_warmup={args.replay_warmup}.",
        flush=True,
    )
    return 0
