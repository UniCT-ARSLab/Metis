"""Continuous-action collectors must be thread-safe too.

DDPG and SAC async collectors do actor inference from one thread per env. Eager
inference corrupts output under concurrency -- the same hazard that crashed PPO. Both
now go through a per-worker traced function; these check the traced samplers stay
well-formed when hammered from several threads at once.
"""

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OBS_DIM = 14
ACTION_SIZE = 2
LOW = np.array([-1.0, -1.0], dtype=np.float32)
HIGH = np.array([1.0, 1.0], dtype=np.float32)


def hammer(call, checker, threads=4, iterations=200):
    errors = []

    def run():
        try:
            for _ in range(iterations):
                checker(call())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    workers = [threading.Thread(target=run) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30.0)
    return errors


class SacSamplerTests(unittest.TestCase):
    def build(self):
        from models import build_sac_actor
        from train_generic_sac import build_sac_sample_fn, select_action_with

        actor = build_sac_actor(obs_dim=OBS_DIM, action_size=ACTION_SIZE)
        fn = build_sac_sample_fn(actor, OBS_DIM, LOW, HIGH, log_std_min=-20.0, log_std_max=2.0)
        return fn, select_action_with

    def test_single_action_shape_and_bounds(self):
        fn, select_action_with = self.build()
        action = select_action_with(fn, np.zeros((OBS_DIM,), np.float32), LOW, HIGH)
        self.assertEqual(action.shape, (ACTION_SIZE,))
        self.assertTrue(np.all(action >= LOW) and np.all(action <= HIGH))

    def test_concurrent_calls_stay_well_formed(self):
        fn, select_action_with = self.build()

        def call():
            return select_action_with(fn, np.zeros((OBS_DIM,), np.float32), LOW, HIGH)

        def checker(action):
            assert action.shape == (ACTION_SIZE,), action.shape

        self.assertEqual(hammer(call, checker), [])


class DdpgActorForwardTests(unittest.TestCase):
    def build(self):
        from models import build_actor_forward_fn, build_continuous_actor

        actor = build_continuous_actor(obs_dim=OBS_DIM, action_size=ACTION_SIZE)
        return build_actor_forward_fn(actor, OBS_DIM)

    def test_forward_shape_for_single_and_batched(self):
        forward = self.build()
        single = forward(np.zeros((1, OBS_DIM), np.float32)).numpy()
        self.assertEqual(single.shape, (1, ACTION_SIZE))
        batched = forward(np.zeros((3, OBS_DIM), np.float32)).numpy()
        self.assertEqual(batched.shape, (3, ACTION_SIZE))

    def test_concurrent_calls_stay_well_formed(self):
        forward = self.build()

        def call():
            return forward(np.zeros((1, OBS_DIM), np.float32)).numpy()

        def checker(raw):
            assert raw.shape == (1, ACTION_SIZE), raw.shape

        self.assertEqual(hammer(call, checker), [])


if __name__ == "__main__":
    unittest.main()
