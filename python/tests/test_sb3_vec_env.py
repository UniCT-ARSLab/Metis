import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class FakeVecEnv:
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.reset_infos = [{} for _ in range(num_envs)]
        self._seeds = [None for _ in range(num_envs)]
        self._options = [{} for _ in range(num_envs)]
        self.render_mode = self.get_attr("render_mode")[0]

    def _get_indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    def _reset_seeds(self):
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self):
        self._options = [{} for _ in range(self.num_envs)]


class FakeGodotEnv:
    observation_space = "observation-space"
    action_space = "action-space"
    render_mode = None

    def __init__(self, step_result):
        self.step_result = step_result
        self.configs = []
        self.reset_count = 0
        self.actions = []
        self.closed = False

    def configure(self, **config):
        self.configs.append(config)

    def reset(self, seed=None, options=None):
        self.reset_count += 1
        return np.asarray([100 + self.reset_count], dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        return self.step_result

    def close(self):
        self.closed = True


class FakeActionCodec:
    policy_space = "encoded-action-space"

    def decode(self, action):
        return f"decoded:{int(action)}"


def load_vec_env_module():
    vec_module = types.ModuleType("stable_baselines3.common.vec_env")
    vec_module.VecEnv = FakeVecEnv
    common_module = types.ModuleType("stable_baselines3.common")
    common_module.vec_env = vec_module
    sb3_module = types.ModuleType("stable_baselines3")
    sb3_module.common = common_module
    modules = {
        "stable_baselines3": sb3_module,
        "stable_baselines3.common": common_module,
        "stable_baselines3.common.vec_env": vec_module,
    }
    path = Path(__file__).resolve().parents[1] / "backends" / "sb3_vec_env.py"
    spec = importlib.util.spec_from_file_location("_metis_test_sb3_vec_env", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class MetisSB3VecEnvTests(unittest.TestCase):
    def test_terminal_observation_is_preserved_before_automatic_reset(self):
        module = load_vec_env_module()
        terminal_info = {"agent_info": {"terminal_reason": "level_cleared"}}
        env0 = FakeGodotEnv((
            np.asarray([9.0], dtype=np.float32),
            3.0,
            True,
            False,
            terminal_info,
        ))
        env1 = FakeGodotEnv((
            np.asarray([4.0], dtype=np.float32),
            -1.0,
            False,
            False,
            {"agent_info": {}},
        ))
        vec = module.MetisSB3VecEnv(
            [env0, env1],
            max_steps=50,
            training_episode_start=10,
            parallel_steps=False,
        )

        first_obs = vec.reset()
        vec.step_async(np.asarray([2, 1]))
        obs, rewards, dones, infos = vec.step_wait()

        self.assertEqual(first_obs.shape, (2, 1))
        self.assertEqual(rewards.tolist(), [3.0, -1.0])
        self.assertEqual(dones.tolist(), [True, False])
        self.assertEqual(infos[0]["terminal_observation"].tolist(), [9.0])
        self.assertEqual(infos[0]["episode"]["r"], 3.0)
        self.assertTrue(infos[0]["is_success"])
        self.assertEqual(obs[0].tolist(), [102.0])
        self.assertEqual(obs[1].tolist(), [4.0])
        self.assertEqual(env0.configs[0]["training_episode"], 10)
        self.assertEqual(env1.configs[0]["training_episode"], 10)
        self.assertEqual(env0.configs[1]["training_episode"], 11)
        self.assertEqual(vec.reset_infos[1]["seed"], 10_001)
        self.assertEqual(vec.current_training_episode, 11)
        self.assertEqual(vec.next_training_episode, 12)
        vec.close()
        self.assertTrue(env0.closed)
        self.assertTrue(env1.closed)

    def test_action_codec_is_exposed_to_sb3_and_decoded_for_godot(self):
        module = load_vec_env_module()
        env = FakeGodotEnv((
            np.asarray([4.0], dtype=np.float32),
            0.0,
            False,
            False,
            {"agent_info": {}},
        ))
        vec = module.MetisSB3VecEnv(
            [env],
            max_steps=50,
            parallel_steps=False,
            action_codec=FakeActionCodec(),
        )

        vec.reset()
        vec.step_async(np.asarray([2]))
        vec.step_wait()

        self.assertEqual(vec.action_space, "encoded-action-space")
        self.assertEqual(env.actions, ["decoded:2"])
        vec.close()


if __name__ == "__main__":
    unittest.main()
