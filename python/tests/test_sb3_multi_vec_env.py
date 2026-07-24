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


class FakeBox:
    def __init__(self, low, high, shape=None, dtype=None):
        self.low = low
        self.high = high
        self.shape = tuple(shape) if shape is not None else np.asarray(low).shape
        self.dtype = dtype


class FakeActionCodec:
    policy_space = "encoded-action-space"

    def decode(self, action):
        return {"decoded": int(action)}


class FakeMultiAgentEnv:
    multi_agent = True
    render_mode = None
    obs_dim = 1
    action_type = "hybrid"
    action_space_spec = {"movement": {"action_type": "continuous", "size": 1}}

    def __init__(self, step_result):
        self.agent_ids = ["Left", "Right"]
        self.step_result = step_result
        self.configs = []
        self.actions = []
        self.reset_count = 0
        self.closed = False

    def configure(self, **config):
        self.configs.append(config)

    def reset(self, seed=None, options=None):
        self.reset_count += 1
        obs = np.asarray(
            [[100 + self.reset_count], [200 + self.reset_count]],
            dtype=np.float32,
        )
        return obs, {
            "per_agent_infos": [
                {"reset_seed": seed},
                {"reset_seed": seed},
            ]
        }

    def step(self, action):
        self.actions.append(action)
        return self.step_result

    def close(self):
        self.closed = True


def load_multi_vec_env_module():
    vec_module = types.ModuleType("stable_baselines3.common.vec_env")
    vec_module.VecEnv = FakeVecEnv
    common_module = types.ModuleType("stable_baselines3.common")
    common_module.vec_env = vec_module
    sb3_module = types.ModuleType("stable_baselines3")
    sb3_module.common = common_module
    gym_spaces = types.ModuleType("gymnasium.spaces")
    gym_spaces.Box = FakeBox
    gym_module = types.ModuleType("gymnasium")
    gym_module.spaces = gym_spaces
    modules = {
        "stable_baselines3": sb3_module,
        "stable_baselines3.common": common_module,
        "stable_baselines3.common.vec_env": vec_module,
        "gymnasium": gym_module,
        "gymnasium.spaces": gym_spaces,
    }
    path = Path(__file__).resolve().parents[1] / "backends" / "sb3_multi_vec_env.py"
    spec = importlib.util.spec_from_file_location("_metis_test_sb3_multi_vec_env", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class MetisSB3MultiAgentVecEnvTests(unittest.TestCase):
    def test_coordinated_agents_become_shared_policy_vector_lanes(self):
        module = load_multi_vec_env_module()
        env = FakeMultiAgentEnv((
            np.asarray([[9.0], [8.0]], dtype=np.float32),
            1.5,
            True,
            False,
            {
                "per_agent_rewards": np.asarray([1.0, 2.0], dtype=np.float32),
                "per_agent_done": np.asarray([True, True]),
                "per_agent_terminated": np.asarray([True, True]),
                "per_agent_infos": [
                    {"terminal_reason": "point_won"},
                    {"terminal_reason": "point_lost"},
                ],
            },
        ))
        vec = module.MetisSB3MultiAgentVecEnv(
            [env],
            action_codec=FakeActionCodec(),
            max_steps=50,
            training_episode_start=7,
            parallel_steps=False,
        )

        first_obs = vec.reset()
        vec.step_async(np.asarray([3, 4]))
        obs, rewards, dones, infos = vec.step_wait()

        self.assertEqual(vec.num_envs, 2)
        self.assertEqual(vec.action_space, "encoded-action-space")
        self.assertEqual(first_obs.tolist(), [[101.0], [201.0]])
        self.assertEqual(
            env.actions,
            [{"Left": {"decoded": 3}, "Right": {"decoded": 4}}],
        )
        self.assertEqual(rewards.tolist(), [1.0, 2.0])
        self.assertEqual(dones.tolist(), [True, True])
        self.assertEqual(infos[0]["terminal_observation"].tolist(), [9.0])
        self.assertEqual(infos[1]["terminal_observation"].tolist(), [8.0])
        self.assertTrue(infos[0]["is_success"])
        self.assertFalse(infos[1]["is_success"])
        self.assertEqual(obs.tolist(), [[102.0], [202.0]])
        self.assertEqual(env.configs[0]["training_episode"], 7)
        self.assertEqual(env.configs[1]["training_episode"], 8)
        vec.close()
        self.assertTrue(env.closed)

    def test_partial_done_is_rejected_by_default(self):
        module = load_multi_vec_env_module()
        env = FakeMultiAgentEnv((
            np.asarray([[9.0], [8.0]], dtype=np.float32),
            0.5,
            False,
            False,
            {
                "per_agent_rewards": np.asarray([-1.0, 0.5], dtype=np.float32),
                "per_agent_done": np.asarray([True, False]),
                "per_agent_terminated": np.asarray([True, False]),
                "per_agent_infos": [{}, {}],
            },
        ))
        vec = module.MetisSB3MultiAgentVecEnv(
            [env],
            action_codec=FakeActionCodec(),
            max_steps=50,
            parallel_steps=False,
        )
        vec.reset()
        vec.step_async(np.asarray([0, 1]))

        with self.assertRaisesRegex(RuntimeError, "reset individual agents"):
            vec.step_wait()
        vec.close()

    def test_reset_all_turns_active_agents_into_truncations(self):
        module = load_multi_vec_env_module()
        env = FakeMultiAgentEnv((
            np.asarray([[9.0], [8.0]], dtype=np.float32),
            0.5,
            False,
            False,
            {
                "per_agent_rewards": np.asarray([-1.0, 0.5], dtype=np.float32),
                "per_agent_done": np.asarray([True, False]),
                "per_agent_terminated": np.asarray([True, False]),
                "per_agent_infos": [{}, {}],
            },
        ))
        vec = module.MetisSB3MultiAgentVecEnv(
            [env],
            action_codec=FakeActionCodec(),
            max_steps=50,
            parallel_steps=False,
            partial_done_mode="reset-all",
        )
        vec.reset()
        vec.step_async(np.asarray([0, 1]))
        _obs, _rewards, dones, infos = vec.step_wait()

        self.assertEqual(dones.tolist(), [True, True])
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertTrue(infos[1]["TimeLimit.truncated"])
        self.assertEqual(env.reset_count, 2)
        vec.close()


if __name__ == "__main__":
    unittest.main()
