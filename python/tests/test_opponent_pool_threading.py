"""The opponent pool must survive concurrent async collectors.

It was written for the sync loop, which touches it from one thread. Three things break
under collectors: a single shared opponent model every worker load_weights() over, an
unlocked manifest/entries list, and pruning that deletes files a worker may be loading.
"""

import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opponent_pool import OpponentPool  # noqa: E402


class FakeModel:
    """Stands in for Keras: records which snapshot it currently holds."""

    _counter = 0

    def __init__(self):
        FakeModel._counter += 1
        self.identity = FakeModel._counter
        self.loaded = None

    def save_weights(self, path):
        Path(path).write_text(f"weights:{self.identity}")

    def load_weights(self, path):
        self.loaded = Path(path).name


def pool_args(directory, **overrides):
    values = {
        "opponent_pool": True,
        "opponent_pool_dir": str(directory),
        "opponent_pool_size": 10,
        "opponent_snapshot_every": 100,
        "opponent_current_probability": 0.0,  # always sample a snapshot
        "opponent_sampling": "uniform",
        "learner_team": None,
        "env_seed_base": 100,
        "checkpoint_dir": str(directory),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PerWorkerOpponentModelTests(unittest.TestCase):
    def test_each_worker_gets_its_own_opponent_model(self):
        # The shared instance was the core race: worker A would find B's weights loaded.
        with TemporaryDirectory() as temp_dir:
            pool = OpponentPool(pool_args(temp_dir), "dqn", FakeModel)
            pool.snapshot(FakeModel(), 100, force=True)
            pool.snapshot(FakeModel(), 200, force=True)

            match_a = pool.start_episode(FakeModel(), 300, worker_id=0)
            match_b = pool.start_episode(FakeModel(), 300, worker_id=1)

            self.assertIsNot(match_a.model, match_b.model,
                             "two workers share one opponent model")

    def test_the_same_worker_reuses_its_model(self):
        # Otherwise every episode would allocate a fresh network.
        with TemporaryDirectory() as temp_dir:
            pool = OpponentPool(pool_args(temp_dir), "dqn", FakeModel)
            pool.snapshot(FakeModel(), 100, force=True)
            first = pool.start_episode(FakeModel(), 200, worker_id=0)
            second = pool.start_episode(FakeModel(), 300, worker_id=0)
            self.assertIs(first.model, second.model)

    def test_a_workers_choice_is_not_overwritten_by_another_worker(self):
        with TemporaryDirectory() as temp_dir:
            pool = OpponentPool(pool_args(temp_dir, opponent_sampling="latest"), "dqn", FakeModel)
            pool.snapshot(FakeModel(), 100, force=True)
            match_a = pool.start_episode(FakeModel(), 150, worker_id=0)
            loaded_by_a = match_a.model.loaded

            pool.snapshot(FakeModel(), 200, force=True)
            pool.start_episode(FakeModel(), 250, worker_id=1)

            self.assertEqual(match_a.model.loaded, loaded_by_a,
                             "worker 1 reloaded worker 0's opponent model")


class ConcurrentAccessTests(unittest.TestCase):
    def test_collectors_and_learner_can_hammer_the_pool_together(self):
        # Workers sampling while the learner snapshots and prunes: this deadlocks or
        # corrupts entries/the manifest without the lock.
        with TemporaryDirectory() as temp_dir:
            pool = OpponentPool(pool_args(temp_dir, opponent_pool_size=3), "dqn", FakeModel)
            pool.snapshot(FakeModel(), 100, force=True)
            errors = []
            stop = threading.Event()

            def collector(worker_id):
                try:
                    for episode in range(200, 260):
                        if stop.is_set():
                            return
                        match = pool.start_episode(FakeModel(), episode, worker_id=worker_id)
                        # A pruned-away file would surface as a load error here.
                        self.assertTrue(match.use_current_policy or match.model.loaded)
                except Exception as exc:  # noqa: BLE001 - reported via `errors`
                    errors.append(exc)
                    stop.set()

            def learner():
                try:
                    for episode in range(200, 260):
                        if stop.is_set():
                            return
                        pool.snapshot(FakeModel(), episode, force=True)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    stop.set()

            threads = [threading.Thread(target=collector, args=(i,)) for i in range(4)]
            threads.append(threading.Thread(target=learner))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30.0)
                self.assertFalse(thread.is_alive(), "the pool deadlocked")

            self.assertEqual(errors, [])
            # Pruning kept its bound despite the concurrency.
            self.assertLessEqual(len(pool.entries), 3)

    def test_manifest_stays_readable_after_concurrent_snapshots(self):
        with TemporaryDirectory() as temp_dir:
            args = pool_args(temp_dir, opponent_pool_size=5)
            pool = OpponentPool(args, "dqn", FakeModel)
            errors = []

            def snapshotter(base):
                try:
                    for episode in range(base, base + 20):
                        pool.snapshot(FakeModel(), episode, force=True)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=snapshotter, args=(b,)) for b in (100, 500)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30.0)

            self.assertEqual(errors, [])
            # A torn manifest would fail to parse when reloaded.
            reloaded = OpponentPool(args, "dqn", FakeModel)
            self.assertLessEqual(len(reloaded.entries), 5)
            for entry in reloaded.entries:
                self.assertTrue((Path(temp_dir) / entry["file"]).is_file())


if __name__ == "__main__":
    unittest.main()
