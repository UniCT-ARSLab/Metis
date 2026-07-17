import argparse
import time
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itertools import islice

import numpy as np

from core.training import (
    AsyncCollectorPool,
    AsyncEventScheduler,
    AsyncEpisodeEvent,
    AsyncStepEvent,
    AsyncWorkerDoneEvent,
    EpisodeAllocator,
    PolicySnapshot,
    add_collector_arguments,
    add_parallel_env_arguments,
    episode_step_indices,
)


class FakeModel:
    def __init__(self):
        self.weights = []

    def set_weights(self, weights):
        self.weights = [np.array(weight, copy=True) for weight in weights]


class AsyncCollectorTests(unittest.TestCase):
    def test_episode_step_indices_supports_finite_and_unlimited_episodes(self):
        self.assertEqual(list(episode_step_indices(3)), [0, 1, 2])
        self.assertEqual(list(islice(episode_step_indices(0), 4)), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            episode_step_indices(-1)

    def test_episode_allocator_claims_each_episode_once(self):
        allocator = EpisodeAllocator(3, 6)
        self.assertEqual([allocator.claim(), allocator.claim(), allocator.claim()], [3, 4, 5])
        self.assertIsNone(allocator.claim())

    def test_policy_snapshot_only_syncs_new_versions(self):
        snapshot = PolicySnapshot(
            [np.asarray([1.0], dtype=np.float32)],
            state=np.asarray([-0.5], dtype=np.float32),
        )
        model = FakeModel()
        version, state = snapshot.sync_model_with_state(model)
        self.assertEqual(version, 0)
        self.assertEqual(model.weights[0].tolist(), [1.0])
        self.assertEqual(state.tolist(), [-0.5])

        self.assertEqual(snapshot.sync_model(model, version), version)
        snapshot.publish(
            [np.asarray([2.0], dtype=np.float32)],
            state=np.asarray([-0.25], dtype=np.float32),
        )
        version, state = snapshot.sync_model_with_state(model, version)
        self.assertEqual(version, 1)
        self.assertEqual(model.weights[0].tolist(), [2.0])
        self.assertEqual(state.tolist(), [-0.25])

    def test_async_is_the_default_collector_mode(self):
        parser = argparse.ArgumentParser()
        add_collector_arguments(parser)
        defaults = parser.parse_args([])
        self.assertEqual(defaults.collector_mode, "async")
        self.assertEqual(defaults.async_update_basis, "transitions")
        self.assertEqual(defaults.async_update_every, 4)
        self.assertEqual(defaults.async_max_updates_per_env_step, 1)
        self.assertEqual(parser.parse_args(["--collector-mode", "sync"]).collector_mode, "sync")
        self.assertEqual(
            parser.parse_args(["--async-update-every-steps", "8"]).async_update_every,
            8,
        )

    def test_all_environments_render_when_count_is_omitted(self):
        parser = argparse.ArgumentParser()
        add_parallel_env_arguments(parser)
        self.assertIsNone(parser.parse_args([]).render_env_count)
        self.assertEqual(parser.parse_args(["--render-env-count", "1"]).render_env_count, 1)

    def test_pool_runs_independent_workers_and_finishes(self):
        def worker(worker_id, _env, allocator, put, stop_event):
            while not stop_event.is_set():
                episode = allocator.claim()
                if episode is None:
                    return
                put(AsyncStepEvent(worker_id, ((episode,),)))
                put(AsyncEpisodeEvent(worker_id, episode, {"episode": episode}))

        pool = AsyncCollectorPool([object(), object()], worker, 0, 5, queue_capacity=16)
        pool.start()
        episodes = []
        done = 0
        deadline = time.monotonic() + 2.0
        while done < 2 and time.monotonic() < deadline:
            event = pool.get(timeout=0.2)
            if isinstance(event, AsyncEpisodeEvent):
                episodes.append(event.episode)
            elif isinstance(event, AsyncWorkerDoneEvent):
                done += 1
        pool.close()

        self.assertEqual(sorted(episodes), list(range(5)))
        self.assertEqual(done, 2)

    def test_scheduler_converts_step_bursts_to_update_credits(self):
        args = argparse.Namespace(
            async_update_basis="transitions",
            async_update_every=4,
            async_updates_per_step=1,
            async_max_updates_per_env_step=1,
            async_drain_max_events=8,
        )
        scheduler = AsyncEventScheduler(args)
        events = [AsyncStepEvent(0, ((idx,),)) for idx in range(8)]
        self.assertEqual(scheduler.ingest(events), 2)
        scheduler.record_updates(2)

        class Pool:
            class Events:
                maxsize = 16

                @staticmethod
                def qsize():
                    return 4

            events = Events()

        metrics = scheduler.throughput(Pool())
        self.assertEqual(metrics["queue_saturation"], 0.25)
        self.assertEqual(metrics["transitions_per_env_step"], 1.0)
        self.assertEqual(metrics["throttled_updates"], 0)
        self.assertGreater(metrics["env_steps_s"], 0.0)

    def test_scheduler_scales_with_multi_agent_transitions_and_caps_updates(self):
        args = argparse.Namespace(
            async_update_basis="transitions",
            async_update_every=4,
            async_updates_per_step=1,
            async_max_updates_per_env_step=1,
            async_drain_max_events=8,
        )
        scheduler = AsyncEventScheduler(args)
        transition = (np.zeros((2,), dtype=np.float32), 0, 0.0, np.zeros((2,)), False)
        events = [AsyncStepEvent(0, tuple(transition for _ in range(10))) for _ in range(4)]

        # Forty agent transitions request ten updates, limited to four Godot steps.
        self.assertEqual(scheduler.ingest(events), 4)

        class Pool:
            class Events:
                maxsize = 16

                @staticmethod
                def qsize():
                    return 0

            events = Events()

        metrics = scheduler.throughput(Pool())
        self.assertEqual(metrics["transitions_per_env_step"], 10.0)
        self.assertEqual(metrics["requested_updates"], 10)
        self.assertEqual(metrics["throttled_updates"], 6)

    def test_scheduler_can_restore_legacy_env_step_schedule(self):
        args = argparse.Namespace(
            async_update_basis="env_steps",
            async_update_every=4,
            async_updates_per_step=1,
            async_max_updates_per_env_step=1,
            async_drain_max_events=8,
        )
        scheduler = AsyncEventScheduler(args)
        events = [AsyncStepEvent(0, tuple((idx,) for idx in range(10))) for _ in range(4)]
        self.assertEqual(scheduler.ingest(events), 1)


if __name__ == "__main__":
    unittest.main()
