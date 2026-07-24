import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gymnasium import spaces


class FakeAlgorithm:
    pass


class FakeCallback:
    def __init__(self, verbose=0):
        self.verbose = verbose


class FakeNoise:
    def __init__(self, **_kwargs):
        pass


class FakeVecEnv:
    pass


def load_sb3_backend_module():
    sb3_module = types.ModuleType("stable_baselines3")
    for name in ("DDPG", "DQN", "PPO", "SAC", "TD3"):
        setattr(sb3_module, name, FakeAlgorithm)

    callbacks_module = types.ModuleType("stable_baselines3.common.callbacks")
    callbacks_module.BaseCallback = FakeCallback
    noise_module = types.ModuleType("stable_baselines3.common.noise")
    noise_module.NormalActionNoise = FakeNoise
    vec_env_module = types.ModuleType("stable_baselines3.common.vec_env")
    vec_env_module.VecEnv = FakeVecEnv
    common_module = types.ModuleType("stable_baselines3.common")
    common_module.callbacks = callbacks_module
    common_module.noise = noise_module
    common_module.vec_env = vec_env_module

    modules = {
        "stable_baselines3": sb3_module,
        "stable_baselines3.common": common_module,
        "stable_baselines3.common.callbacks": callbacks_module,
        "stable_baselines3.common.noise": noise_module,
        "stable_baselines3.common.vec_env": vec_env_module,
    }
    path = Path(__file__).resolve().parents[1] / "backends" / "sb3.py"
    spec = importlib.util.spec_from_file_location("_metis_test_sb3_backend", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        sys.modules.pop("backends.sb3_vec_env", None)
        spec.loader.exec_module(module)
    return module


class SB3BackendCliTests(unittest.TestCase):
    def test_shared_physics_argument_is_registered_once(self):
        module = load_sb3_backend_module()

        args = module.parse_args(
            [
                "--algorithm",
                "ppo",
                "--physics-frames-per-step",
                "4",
            ]
        )

        self.assertEqual(args.algorithm, "ppo")
        self.assertEqual(args.physics_frames_per_step, 4)

    def test_ppo_accepts_hybrid_contract_through_action_codec(self):
        module = load_sb3_backend_module()
        action_spec = {
            "movement": {
                "action_type": "continuous",
                "size": 2,
                "low": -1.0,
                "high": 1.0,
            },
            "weapon": {
                "action_type": "discrete",
                "size": 2,
            },
        }
        env = SimpleNamespace(
            action_type="hybrid",
            action_space=spaces.Dict({}),
            action_space_spec=action_spec,
            multi_agent=False,
        )
        args = SimpleNamespace(algorithm="ppo", multi_agent=False)

        module.validate_scenario(args, [env])
        codec = module.build_action_codec(env)

        self.assertTrue(codec.uses_hybrid_encoding)
        self.assertEqual(codec.policy_space.shape, (4,))


if __name__ == "__main__":
    unittest.main()
