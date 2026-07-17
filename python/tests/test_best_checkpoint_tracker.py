import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


from core.training import (
    BestCheckpointTracker,
    PolicyEvaluationResult,
    apply_ready_best_checkpoint,
    remove_checkpoint_files,
)


def tracker_args(checkpoint_dir, **overrides):
    values = {
        "best_checkpoint": True,
        "best_checkpoint_dir": None,
        "keep_best_checkpoints": 3,
        "best_evaluation_every": 100,
        "best_evaluation_episodes": 20,
        "best_evaluation_seed": 10_000,
        "best_evaluation_training_episode": None,
        "best_evaluation_max_steps": None,
        "best_evaluation_port": None,
        "best_evaluation_timeout": 30.0,
        "best_final_drain_timeout": 120.0,
        "best_evaluation_device": "cpu",
        "best_metric": "auto",
        "checkpoint_dir": str(checkpoint_dir),
        "base_port": 6200,
        "num_envs": 4,
        "num_episodes": 500,
        "max_steps_per_episode": 0,
        "multi_agent": False,
        "godot_bin": "/godot",
        "godot_project": "godot",
        "godot_scene": "res://scenario.tscn",
        "agent_id": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def evaluation(episode, success_rate, reward_mean):
    return PolicyEvaluationResult(
        episode,
        {
            "episodes": 20,
            "successes": int(success_rate * 20),
            "trials": 20,
            "success_rate": success_rate,
            "reward_mean": reward_mean,
            "reward_min": reward_mean,
            "reward_max": reward_mean,
            "steps_mean": 100.0,
            "steps_min": 50,
            "steps_max": 150,
        },
    )


class BestCheckpointTrackerTests(unittest.TestCase):
    def test_evaluation_schedule(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(tracker_args(temp_dir), "dqn")

            self.assertFalse(tracker.should_evaluate(99))
            self.assertTrue(tracker.should_evaluate(100))
            self.assertFalse(tracker.should_evaluate(101))

    def test_auto_metric_prioritizes_success_then_reward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(tracker_args(temp_dir), "dqn")
            baseline = evaluation(100, success_rate=0.10, reward_mean=50.0)
            tracker.record_best(baseline, "best/ckpt-100")

            self.assertFalse(tracker.is_improvement(evaluation(200, 0.05, 100.0)))
            self.assertTrue(tracker.is_improvement(evaluation(200, 0.10, 51.0)))
            self.assertTrue(tracker.is_improvement(evaluation(200, 0.15, -10.0)))

    def test_reward_metric_prioritizes_reward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = tracker_args(temp_dir, best_metric="reward_mean")
            tracker = BestCheckpointTracker(args, "sac")
            tracker.record_best(evaluation(100, 0.50, 10.0), "best/ckpt-100")

            self.assertTrue(tracker.is_improvement(evaluation(200, 0.10, 11.0)))

    def test_metadata_is_restored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = tracker_args(temp_dir)
            tracker = BestCheckpointTracker(args, "ppo")
            tracker.record_best(evaluation(300, 0.25, 7.0), "best/ckpt-300")

            restored = BestCheckpointTracker(args, "ppo")

            self.assertEqual(restored.best_key, (0.25, 7.0))
            self.assertEqual(restored.best_summary["episode"], 300)
            self.assertTrue((Path(temp_dir) / "best" / "best_metrics.json").is_file())

    def test_unlimited_training_gets_a_finite_evaluation_watchdog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(
                tracker_args(temp_dir, max_steps_per_episode=0),
                "dqn",
            )

            command = tracker._evaluation_command("ckpt-100", Path(temp_dir) / "summary.json")

            max_steps_index = command.index("--max-steps") + 1
            self.assertEqual(command[max_steps_index], "10000")


def write_fake_checkpoint(prefix, marker):
    """A checkpoint's shards with recognisable bytes, so we can prove which ones landed."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_name(prefix.name + ".index")).write_text(f"index:{marker}")
    (prefix.with_name(prefix.name + ".data-00000-of-00001")).write_text(f"data:{marker}")
    return prefix


class PromoteEvaluatedCheckpointTests(unittest.TestCase):
    """The promoted file must be the one that was evaluated.

    The old code re-saved the live model at the current episode, so best_metrics.json
    recorded episode 100's metrics next to whatever weights training had reached by the
    time the evaluation came back.
    """

    def staged_tracker(self, temp_dir, result, **overrides):
        tracker = BestCheckpointTracker(tracker_args(temp_dir, **overrides), "dqn")
        tracker.evaluate = lambda checkpoint_path, episode: result
        return tracker

    def test_promoted_checkpoint_keeps_the_evaluated_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = write_fake_checkpoint(Path(temp_dir) / "ckpt-100", "evaluated")
            tracker = self.staged_tracker(temp_dir, evaluation(100, 0.5, 1.0))

            tracker.evaluate_async(source, 100)
            # --keep-checkpoints deletes the evaluated checkpoint while the evaluation
            # is still running. Staging at request time is what survives this.
            remove_checkpoint_files(source)
            write_fake_checkpoint(Path(temp_dir) / "ckpt-150", "live-and-never-evaluated")

            best_path = apply_ready_best_checkpoint(tracker, wait_timeout=10.0)

            self.assertIsNotNone(best_path, "the improvement was not promoted")
            promoted = Path(best_path).with_name(Path(best_path).name + ".index")
            self.assertEqual(promoted.read_text(), "index:evaluated")
            self.assertTrue(str(best_path).endswith("ckpt-100"))
            tracker.close()

    def test_metadata_episode_matches_the_promoted_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = write_fake_checkpoint(Path(temp_dir) / "ckpt-100", "evaluated")
            tracker = self.staged_tracker(temp_dir, evaluation(100, 0.5, 1.0))
            tracker.evaluate_async(source, 100)
            apply_ready_best_checkpoint(tracker, wait_timeout=10.0)

            metadata = json.loads(tracker.metadata_path.read_text())
            self.assertEqual(metadata["episode"], 100)
            self.assertTrue(metadata["checkpoint"].endswith("ckpt-100"))
            tracker.close()

    def test_a_non_improvement_leaves_the_best_directory_untouched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(tracker_args(temp_dir), "dqn")
            tracker.record_best(evaluation(100, 0.9, 10.0), "best/ckpt-100")
            tracker.evaluate = lambda checkpoint_path, episode: evaluation(200, 0.1, 1.0)

            source = write_fake_checkpoint(Path(temp_dir) / "ckpt-200", "worse")
            tracker.evaluate_async(source, 200)
            self.assertIsNone(apply_ready_best_checkpoint(tracker, wait_timeout=10.0))

            self.assertEqual(list(tracker.directory.glob("ckpt-*.index")), [])
            self.assertEqual(list(tracker.staging_directory.glob("*")), [],
                             "the rejected staging copy was left behind")
            tracker.close()

    def test_pruning_keeps_only_the_newest_best_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(tracker_args(temp_dir, keep_best_checkpoints=2), "dqn")
            for episode, rate in ((100, 0.1), (200, 0.2), (300, 0.3)):
                source = write_fake_checkpoint(Path(temp_dir) / f"ckpt-{episode}", str(episode))
                tracker.evaluate = lambda _p, _e, rate=rate, episode=episode: evaluation(episode, rate, 1.0)
                tracker.evaluate_async(source, episode)
                self.assertIsNotNone(apply_ready_best_checkpoint(tracker, wait_timeout=10.0))

            kept = sorted(p.name for p in tracker.directory.glob("ckpt-*.index"))
            self.assertEqual(kept, ["ckpt-200.index", "ckpt-300.index"])
            tracker.close()

    def test_staging_survives_the_source_being_pruned_mid_evaluation(self):
        # The evaluator subprocess reads the staged copy, so the trainer is free to
        # prune the training directory while an evaluation is in flight.
        with tempfile.TemporaryDirectory() as temp_dir:
            source = write_fake_checkpoint(Path(temp_dir) / "ckpt-100", "evaluated")
            tracker = BestCheckpointTracker(tracker_args(temp_dir), "dqn")
            seen = {}

            def record_path(checkpoint_path, episode):
                seen["path"] = Path(checkpoint_path)
                return evaluation(int(episode), 0.5, 1.0)

            tracker.evaluate = record_path
            tracker.evaluate_async(source, 100)
            apply_ready_best_checkpoint(tracker, wait_timeout=10.0)

            self.assertEqual(seen["path"].parent, tracker.staging_directory,
                             "the evaluator was pointed at the prunable training directory")
            tracker.close()


class EveryTrainerDrainsBeforeExitTests(unittest.TestCase):
    """Every trainer must drain, not just the one that was worked on.

    Twice now a fix landed in one collector and not its three siblings. `finally:
    best_tracker.close()` cancels whatever evaluation is in flight, so a trainer that
    exits without draining can never promote a best found on its last episode.
    """

    TRAINERS = ["dqn", "ppo", "sac", "ddpg"]
    TRAINER_DIR = Path(__file__).resolve().parents[1] / "algorithms"

    def trainer_source(self, trainer):
        filename = "common.py" if trainer == "ddpg" else f"{trainer}.py"
        return (self.TRAINER_DIR / filename).read_text()

    def test_every_trainer_drains_in_both_collector_modes(self):
        for trainer in self.TRAINERS:
            source = self.trainer_source(trainer)
            drains = source.count("wait_timeout=args.best_final_drain_timeout")
            # One for the async tail, one before the sync final save_weights.
            self.assertEqual(
                drains, 2,
                msg=f"algorithms/{trainer}.py has {drains} drain call(s), expected 2",
            )

    def test_no_trainer_kept_a_private_copy_of_the_helper(self):
        # Four near-identical copies is how the live-weights bug survived in all of them.
        for trainer in self.TRAINERS:
            source = self.trainer_source(trainer)
            self.assertNotIn(
                "def apply_ready_best_checkpoint", source,
                msg=f"algorithms/{trainer}.py redefines the shared helper",
            )
            self.assertNotIn(
                "best_checkpoint_manager", source,
                msg=f"algorithms/{trainer}.py still drives a CheckpointManager over the best dir",
            )


class DrainTimeoutValidationTests(unittest.TestCase):
    def test_negative_drain_timeout_is_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                BestCheckpointTracker(tracker_args(temp_dir, best_final_drain_timeout=-1.0), "dqn")

    def test_zero_drain_timeout_is_allowed(self):
        # Zero is meaningful: exit without waiting.
        with tempfile.TemporaryDirectory() as temp_dir:
            BestCheckpointTracker(tracker_args(temp_dir, best_final_drain_timeout=0.0), "dqn")

    def test_best_directory_may_not_be_the_training_directory(self):
        # The tracker prunes its directory; sharing it would delete live training state.
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                BestCheckpointTracker(
                    tracker_args(temp_dir, best_checkpoint_dir=temp_dir), "dqn"
                )


class CheckpointStateMetafileTests(unittest.TestCase):
    """The tracker now writes the CheckpointState metafile that CheckpointManager used to.

    run.py falls back to tf.train.latest_checkpoint() when --checkpoint-path
    is omitted, so `best/` must stay readable that way. This is the gate on hand-editing
    the proto -- TF deprecates doing so, and if it ever stops working the fallback is a
    CheckpointManager for the best directory.
    """

    def test_latest_checkpoint_finds_the_promoted_best(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = BestCheckpointTracker(tracker_args(temp_dir, keep_best_checkpoints=2), "dqn")
            for episode, rate in ((100, 0.1), (200, 0.2), (300, 0.3)):
                # Real checkpoints, so the round-trip proves they are restorable.
                source = Path(temp_dir) / f"ckpt-{episode}"
                tf.train.Checkpoint(v=tf.Variable(float(episode))).write(str(source))
                tracker.evaluate = lambda _p, _e, rate=rate, episode=episode: evaluation(episode, rate, 1.0)
                tracker.evaluate_async(source, episode)
                apply_ready_best_checkpoint(tracker, wait_timeout=10.0)

            latest = tf.train.latest_checkpoint(str(tracker.directory))
            self.assertIsNotNone(latest, "latest_checkpoint() cannot see the promoted best")
            self.assertTrue(latest.endswith("ckpt-300"), latest)

            state = tf.train.get_checkpoint_state(str(tracker.directory))
            self.assertEqual(len(state.all_model_checkpoint_paths), 2, "pruned entries linger in the metafile")

            restored = tf.train.Checkpoint(v=tf.Variable(0.0))
            restored.read(latest).expect_partial()
            self.assertEqual(float(restored.v.numpy()), 300.0, "the promoted checkpoint is not restorable")
            tracker.close()


class SleepingEvaluatorTracker(BestCheckpointTracker):
    """Swaps Godot for a child that only sleeps, so shutdown can be tested anywhere."""

    def _evaluation_command(self, checkpoint_path, summary_path):
        return [sys.executable, "-c", "import time; time.sleep(600)"]


class OrphanSpawningTracker(BestCheckpointTracker):
    """Child that spawns a grandchild, mirroring the evaluator spawning Godot."""

    def __init__(self, args, algorithm, pid_path):
        self.pid_path = pid_path
        super().__init__(args, algorithm)

    def _evaluation_command(self, checkpoint_path, summary_path):
        script = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
            f"open({str(self.pid_path)!r}, 'w').write(str(child.pid))\n"
            "time.sleep(600)\n"
        )
        return [sys.executable, "-c", script]


def wait_for_process(test, tracker, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with tracker._process_lock:
            process = tracker._process
        if process is not None:
            return process
        time.sleep(0.02)
    test.fail("the evaluation subprocess never started")


class CloseStopsTheEvaluatorTests(unittest.TestCase):
    """close() used to abandon the child, and ThreadPoolExecutor's non-daemon threads are
    joined by CPython at exit -- so the interpreter hung until --best-evaluation-timeout.
    """

    def test_close_kills_the_child_instead_of_waiting_out_the_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = SleepingEvaluatorTracker(
                tracker_args(temp_dir, best_evaluation_timeout=600.0), "dqn"
            )
            write_fake_checkpoint(Path(temp_dir) / "ckpt-100", "staged")
            tracker.evaluate_async(Path(temp_dir) / "ckpt-100", 100)
            process = wait_for_process(self, tracker)

            started = time.monotonic()
            tracker.close()
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 10.0, f"close() took {elapsed:.1f}s")
            self.assertIsNotNone(process.poll(), "the evaluator survived close()")

    def test_close_is_safe_with_no_evaluation_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            SleepingEvaluatorTracker(tracker_args(temp_dir), "dqn").close()

    def test_evaluate_async_is_refused_once_closing(self):
        # A request racing close() would otherwise spawn a child nothing would reap.
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = SleepingEvaluatorTracker(tracker_args(temp_dir), "dqn")
            tracker.close()
            self.assertFalse(tracker.evaluate_async("ckpt-100", 100))

    @unittest.skipUnless(os.name == "posix", "process groups are posix-only")
    def test_close_reaps_the_grandchild_so_no_orphan_holds_the_port(self):
        # run.py only stops Godot from a `finally`, which a bare kill() skips.
        # An orphan keeps --best-evaluation-port bound and breaks every later run.
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_path = Path(temp_dir) / "grandchild.pid"
            tracker = OrphanSpawningTracker(
                tracker_args(temp_dir, best_evaluation_timeout=600.0), "dqn", pid_path
            )
            write_fake_checkpoint(Path(temp_dir) / "ckpt-100", "staged")
            tracker.evaluate_async(Path(temp_dir) / "ckpt-100", 100)
            wait_for_process(self, tracker)

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and not pid_path.is_file():
                time.sleep(0.02)
            self.assertTrue(pid_path.is_file(), "the grandchild never reported its pid")
            grandchild_pid = int(pid_path.read_text())

            tracker.close()

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    os.kill(grandchild_pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.05)
            os.kill(grandchild_pid, signal.SIGKILL)  # don't leak it out of the test
            self.fail("the grandchild outlived close()")


class FinalDrainTests(unittest.TestCase):
    """An evaluation requested on the last episode was never collected."""

    def setUp(self):
        self.released = threading.Event()
        self.addCleanup(self.released.set)

    def blocking_tracker(self, temp_dir):
        tracker = BestCheckpointTracker(tracker_args(temp_dir), "dqn")
        released = self.released

        def blocking_evaluate(checkpoint_path, episode):
            released.wait(timeout=30.0)
            return evaluation(int(episode), 0.5, 1.0)

        tracker.evaluate = blocking_evaluate
        # evaluate_async stages the checkpoint before scheduling, so the shards must
        # exist -- a request for a checkpoint that isn't there is refused, by design.
        write_fake_checkpoint(Path(temp_dir) / "ckpt-100", "staged")
        return tracker

    def test_zero_timeout_stays_non_blocking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = self.blocking_tracker(temp_dir)
            tracker.evaluate_async(Path(temp_dir) / "ckpt-100", 100)
            started = time.monotonic()
            self.assertIsNone(tracker.poll_ready())
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertIsNotNone(tracker._pending)
            self.released.set()
            tracker.close()

    def test_positive_timeout_collects_a_result_that_arrives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = self.blocking_tracker(temp_dir)
            tracker.evaluate_async(Path(temp_dir) / "ckpt-100", 100)
            self.released.set()
            result = tracker.poll_ready(timeout=10.0)
            self.assertIsNotNone(result, "the drain discarded a finished evaluation")
            self.assertEqual(result.episode, 100)
            tracker.close()

    def test_drain_gives_up_without_hanging_and_keeps_the_evaluation_pending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = self.blocking_tracker(temp_dir)
            tracker.evaluate_async(Path(temp_dir) / "ckpt-100", 100)
            started = time.monotonic()
            self.assertIsNone(tracker.poll_ready(timeout=0.2))
            self.assertLess(time.monotonic() - started, 3.0)
            # Giving up on the wait must not lose the handle.
            self.assertIsNotNone(tracker._pending)
            self.released.set()
            tracker.close()


if __name__ == "__main__":
    unittest.main()
