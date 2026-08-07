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
    SyncUpdateThrottle,
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

    def test_sync_scheduler_matches_transition_credit_and_cap(self):
        args = argparse.Namespace(
            async_update_basis="transitions",
            async_update_every=4,
            async_updates_per_step=1,
            async_max_updates_per_env_step=1,
        )
        scheduler = SyncUpdateThrottle(args)

        self.assertEqual(scheduler.updates_due(3, env_steps=1), 0)
        self.assertEqual(scheduler.updates_due(1, env_steps=1), 1)
        # Ten transitions request two updates, but one synchronous environment
        # result grants at most one with the default safety cap.
        self.assertEqual(scheduler.updates_due(10, env_steps=1), 1)

    def test_sync_scheduler_supports_env_steps_and_disabled_updates(self):
        env_step_args = argparse.Namespace(
            async_update_basis="env_steps",
            async_update_every=4,
            async_updates_per_step=1,
            async_max_updates_per_env_step=1,
        )
        scheduler = SyncUpdateThrottle(env_step_args)
        self.assertEqual(scheduler.updates_due(40, env_steps=3), 0)
        self.assertEqual(scheduler.updates_due(1, env_steps=1), 1)

        disabled_args = argparse.Namespace(
            async_update_basis="transitions",
            async_update_every=1,
            async_updates_per_step=0,
            async_max_updates_per_env_step=0,
        )
        self.assertEqual(
            SyncUpdateThrottle(disabled_args).updates_due(100, env_steps=1),
            0,
        )

    def test_scheduler_keeps_multi_policy_update_credits_separate(self):
        args = argparse.Namespace(
            async_update_basis="transitions",
            async_update_every=4,
            async_updates_per_step=1,
            async_max_updates_per_env_step=1,
            async_drain_max_events=8,
        )
        scheduler = AsyncEventScheduler(args)
        events = [
            AsyncStepEvent(
                0,
                tuple(
                    [("red", index) for index in range(3)]
                    + [("blue", index) for index in range(2)]
                ),
            ),
            AsyncStepEvent(
                1,
                tuple(
                    [("red", index) for index in range(3, 5)]
                    + [("blue", index) for index in range(2, 4)]
                ),
            ),
        ]

        updates = scheduler.ingest_by_policy(events, lambda transition: transition[0])

        self.assertEqual(updates, {"blue": 1, "red": 1})
        self.assertEqual(scheduler._policy_collection_credit, {"blue": 0, "red": 1})

    def test_recovery_reset_discards_experience_but_preserves_lifecycle_events(self):
        args = argparse.Namespace(
            async_update_basis="transitions",
            async_update_every=4,
            async_updates_per_step=1,
            async_max_updates_per_env_step=1,
            async_drain_max_events=8,
        )
        scheduler = AsyncEventScheduler(args)
        pool = AsyncCollectorPool([], lambda *_args: None, 0, 0, queue_capacity=8)
        scheduler.ingest([
            AsyncStepEvent(0, ((1,), (2,), (3,))),
        ])
        pool.events.put(AsyncEpisodeEvent(0, 7, {"episode": 7}))
        scheduler.drain_step_events(pool, AsyncStepEvent(0, ((0,),)))
        pool.events.put(AsyncStepEvent(0, ((4,), (5,)), policy_version=2))
        pool.events.put(AsyncWorkerDoneEvent(0))

        dropped = scheduler.reset_after_recovery(pool)

        self.assertEqual(dropped, {"events": 1, "transitions": 2})
        self.assertIsInstance(scheduler.next_event(pool), AsyncEpisodeEvent)
        self.assertIsInstance(pool.get(), AsyncWorkerDoneEvent)
        # The three pre-reset units must not combine with one new unit to grant an update.
        self.assertEqual(scheduler.ingest([AsyncStepEvent(0, ((6,),))]), 0)


if __name__ == "__main__":
    unittest.main()
