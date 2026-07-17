"""PPO must drive plain discrete scenarios, not only hybrid ones.

Breakout exposes Discrete(3) and its env does `int(action)`, so the hybrid dict PPO
emits would raise TypeError. The model already supports the shape -- build_action_metadata
handles discrete components and build_hybrid_actor_critic accepts continuous_size=0 --
so the only real difference is the action format crossing the bridge.
"""

import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_generic_ppo import (  # noqa: E402
    SUPPORTED_ACTION_TYPES,
    build_action_metadata,
    pack_action,
    zero_env_action,
)

# Exactly what Godot reports for breakout (verified against a live bridge).
BREAKOUT_SPEC = OrderedDict({
    "action": {"action_type": "discrete", "names": ["idle", "move_left", "move_right"], "size": 3}
})

HYBRID_SPEC = OrderedDict({
    "gear": {"action_type": "discrete", "names": ["down", "up"], "size": 2},
    "steer": {"action_type": "continuous", "size": 1, "low": -1.0, "high": 1.0},
})


class DiscreteActionMetadataTests(unittest.TestCase):
    def test_discrete_is_a_supported_action_type(self):
        self.assertIn("discrete", SUPPORTED_ACTION_TYPES)
        self.assertIn("hybrid", SUPPORTED_ACTION_TYPES)

    def test_breakout_spec_produces_a_single_discrete_head_and_no_continuous(self):
        meta = build_action_metadata(BREAKOUT_SPEC)
        self.assertEqual(meta["discrete_sizes"], [3])
        self.assertEqual(meta["continuous_size"], 0)
        self.assertEqual(meta["continuous"], [])
        self.assertTrue(meta["single_discrete"])

    def test_hybrid_spec_is_not_flagged_single_discrete(self):
        meta = build_action_metadata(HYBRID_SPEC)
        self.assertFalse(meta["single_discrete"])
        self.assertEqual(meta["discrete_sizes"], [2])
        self.assertEqual(meta["continuous_size"], 1)

    def test_two_discrete_components_are_not_single_discrete(self):
        # Two heads still need the dict: there is no bare int that carries both.
        spec = OrderedDict({
            "throttle": {"action_type": "discrete", "size": 2},
            "gear": {"action_type": "discrete", "size": 3},
        })
        self.assertFalse(build_action_metadata(spec)["single_discrete"])


class PackActionFormatTests(unittest.TestCase):
    """The env's step() decides the shape; pack_action must match it per scenario."""

    def test_single_discrete_packs_a_bare_int_the_env_can_cast(self):
        meta = build_action_metadata(BREAKOUT_SPEC)
        action = pack_action(np.asarray([2], dtype=np.int32), np.zeros((0,), np.float32), meta)
        self.assertIsInstance(action, int)
        self.assertEqual(action, 2)
        # This is the operation ScenarioGymEnv.step performs on a discrete action.
        self.assertEqual(int(action), 2)

    def test_hybrid_still_packs_the_component_dict(self):
        meta = build_action_metadata(HYBRID_SPEC)
        action = pack_action(np.asarray([1], dtype=np.int32), np.asarray([0.5], np.float32), meta)
        self.assertEqual(action, {"gear": 1, "steer": 0.5})

    def test_zero_env_action_matches_the_packed_format(self):
        # Used for already-done agents, so it must be the same shape the env accepts.
        self.assertEqual(zero_env_action(build_action_metadata(BREAKOUT_SPEC)), 0)
        self.assertEqual(
            zero_env_action(build_action_metadata(HYBRID_SPEC)), {"gear": 0, "steer": 0.0}
        )

    def test_a_dict_would_not_survive_the_discrete_env(self):
        # Guards why single_discrete exists: this is the failure it prevents.
        with self.assertRaises(TypeError):
            int({"action": 2})


class DiscretePolicyRoundTripTests(unittest.TestCase):
    def test_the_shared_network_builds_and_splits_with_no_continuous_head(self):
        import tensorflow as tf
        from models import build_hybrid_actor_critic
        from train_generic_ppo import split_model_outputs

        meta = build_action_metadata(BREAKOUT_SPEC)
        model = build_hybrid_actor_critic(
            obs_dim=5, discrete_sizes=meta["discrete_sizes"], continuous_size=meta["continuous_size"]
        )
        outputs = model(tf.zeros((1, 5)), training=False)
        logits, continuous_mean, value = split_model_outputs(outputs, meta)

        self.assertEqual(len(logits), 1)
        self.assertEqual(logits[0].shape, (1, 3))
        self.assertIsNone(continuous_mean, "a discrete scenario must not grow a continuous head")
        self.assertEqual(value.shape, (1,))


class TracedSamplerTests(unittest.TestCase):
    """The sampler is traced per collector because eager sampling corrupts tensor shapes
    under concurrent collector threads. These check the shapes the crash was about, and
    that concurrent calls stay well-formed.
    """

    def build(self, spec, obs_dim=5):
        import tensorflow as tf
        from models import build_hybrid_actor_critic
        from train_generic_ppo import build_action_metadata, build_sample_action_fn, select_action

        meta = build_action_metadata(spec)
        model = build_hybrid_actor_critic(
            obs_dim=obs_dim,
            discrete_sizes=meta["discrete_sizes"],
            continuous_size=meta["continuous_size"],
        )
        fn = build_sample_action_fn(model, obs_dim, action_meta=meta)
        return meta, fn, select_action

    def test_discrete_sampler_returns_a_scalar_action_and_bare_int(self):
        import numpy as np

        meta, fn, select_action = self.build(BREAKOUT_SPEC)
        log_std = np.zeros((0,), dtype=np.float32)
        selected = select_action(fn, log_std, np.zeros((5,), np.float32), meta)

        self.assertIsInstance(selected["env_action"], int)
        self.assertEqual(selected["discrete_actions"].shape, (1,))
        self.assertIn(selected["env_action"], (0, 1, 2))
        self.assertEqual(selected["continuous_action"].shape, (0,))

    def test_hybrid_sampler_returns_the_component_dict(self):
        import numpy as np

        meta, fn, select_action = self.build(HYBRID_SPEC)
        log_std = np.zeros((meta["continuous_size"],), dtype=np.float32)
        selected = select_action(fn, log_std, np.zeros((5,), np.float32), meta)

        self.assertIsInstance(selected["env_action"], dict)
        self.assertIn("gear", selected["env_action"])
        self.assertIn("steer", selected["env_action"])

    def test_concurrent_calls_stay_well_formed(self):
        # The reason this exists: eager sampling produced a 0-D action under 4 threads.
        # A traced function is safe; this asserts every concurrent result is usable.
        import threading

        import numpy as np

        meta, fn, select_action = self.build(BREAKOUT_SPEC)
        log_std = np.zeros((0,), dtype=np.float32)
        errors = []

        def hammer():
            try:
                for _ in range(200):
                    selected = select_action(fn, log_std, np.zeros((5,), np.float32), meta)
                    assert isinstance(selected["env_action"], int)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
