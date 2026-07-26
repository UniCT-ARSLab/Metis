import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from core.evaluation import summarize_episode_outcome


def _is_wrapped(env, wrapper_class):
    current = env
    while current is not None:
        if isinstance(current, wrapper_class):
            return True
        current = getattr(current, "env", None)
    return False


class MetisSB3VecEnv(VecEnv):
    """SB3 VecEnv over already-running Godot socket environments.

    Godot instances are independent external processes, so threads only overlap socket
    waits; they do not run Python simulation code under the GIL. The class also performs
    SB3's required automatic reset while preserving the real terminal observation.
    """

    def __init__(
        self,
        envs,
        *,
        max_steps,
        physics_frames_per_step=1,
        episode_seed_multiplier=1000,
        training_episode_start=0,
        parallel_steps=True,
        reset_progress_curriculum=False,
        reset_progress_start_max=0.025,
        reset_progress_end_max=0.35,
        reset_progress_ramp_episodes=400,
        action_codec=None,
    ):
        if not envs:
            raise ValueError("MetisSB3VecEnv requires at least one environment")
        self.envs = list(envs)
        self.raw_envs = list(envs)
        self.action_codec = action_codec
        self.max_steps = int(max_steps)
        self.physics_frames_per_step = int(physics_frames_per_step)
        self.episode_seed_multiplier = int(episode_seed_multiplier)
        self._next_training_episodes = [
            int(training_episode_start) for _env in self.envs
        ]
        self._episode_numbers = [-1] * len(self.envs)
        self._episode_returns = np.zeros((len(self.envs),), dtype=np.float64)
        self._episode_lengths = np.zeros((len(self.envs),), dtype=np.int64)
        self._episode_started = np.full((len(self.envs),), time.monotonic(), dtype=np.float64)
        self._actions = None
        self._closed = False
        self.reset_progress_curriculum = bool(reset_progress_curriculum)
        self.reset_progress_start_max = float(reset_progress_start_max)
        self.reset_progress_end_max = float(reset_progress_end_max)
        self.reset_progress_ramp_episodes = max(1, int(reset_progress_ramp_episodes))
        self._executor = (
            ThreadPoolExecutor(max_workers=len(self.envs), thread_name_prefix="sb3-godot-env")
            if parallel_steps and len(self.envs) > 1
            else None
        )

        first = self.envs[0]
        for env in self.envs[1:]:
            if env.observation_space != first.observation_space:
                raise ValueError("All Godot environments must expose the same observation space")
            if env.action_space != first.action_space:
                raise ValueError("All Godot environments must expose the same action space")
        policy_action_space = (
            self.action_codec.policy_space
            if self.action_codec is not None
            else first.action_space
        )
        super().__init__(len(self.envs), first.observation_space, policy_action_space)

    @property
    def next_training_episode(self):
        return max(self._next_training_episodes)

    @property
    def current_training_episode(self):
        active = [episode for episode in self._episode_numbers if episode >= 0]
        return max(active) if active else self.next_training_episode

    def _curriculum_progress_max(self, episode):
        if not self.reset_progress_curriculum:
            return None
        ratio = min(1.0, max(0.0, float(episode) / float(self.reset_progress_ramp_episodes)))
        return self.reset_progress_start_max + ratio * (
            self.reset_progress_end_max - self.reset_progress_start_max
        )

    def _reset_one(self, index, *, explicit_seed=None, options=None):
        episode = self._next_training_episodes[index]
        self._next_training_episodes[index] += 1
        self._episode_numbers[index] = episode

        config = {
            "training_episode": episode,
            "max_steps": self.max_steps,
            "physics_frames_per_step": self.physics_frames_per_step,
            "training_mode": True,
        }
        progress_max = self._curriculum_progress_max(episode)
        if progress_max is not None:
            config.update(reset_progress_min=0.0, reset_progress_max=progress_max)
        self.envs[index].configure(**config)

        seed = (
            int(explicit_seed)
            if explicit_seed is not None
            else self.episode_seed_multiplier * episode + index
        )
        obs, reset_info = self.envs[index].reset(seed=seed, options=options)
        self._episode_returns[index] = 0.0
        self._episode_lengths[index] = 0
        self._episode_started[index] = time.monotonic()
        return np.asarray(obs, dtype=np.float32), dict(reset_info)

    def reset(self):
        observations = []
        reset_infos = []
        for index in range(self.num_envs):
            options = self._options[index] if self._options[index] else None
            # Metis derives scene seeds from the training episode in every native
            # trainer. SB3 also calls VecEnv.seed() while seeding its policy, but using
            # those values here would make the Godot curriculum differ by backend.
            obs, info = self._reset_one(index, options=options)
            observations.append(obs)
            reset_infos.append(info)
        self.reset_infos = reset_infos
        self._reset_seeds()
        self._reset_options()
        return np.stack(observations)

    def step_async(self, actions):
        if self._actions is not None:
            raise RuntimeError("step_async called while a previous SB3 Godot step is pending")
        self._actions = actions

    def _step_all(self, actions):
        requests = [
            (
                env,
                self.action_codec.decode(actions[index])
                if self.action_codec is not None
                else actions[index],
            )
            for index, env in enumerate(self.envs)
        ]
        if self._executor is None:
            return [env.step(action) for env, action in requests]
        futures = [self._executor.submit(env.step, action) for env, action in requests]
        return [future.result() for future in futures]

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait called before step_async")
        actions = self._actions
        self._actions = None
        results = self._step_all(actions)

        observations = []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos = []

        for index, (obs, reward, terminated, truncated, info) in enumerate(results):
            reward = float(reward)
            done = bool(terminated or truncated)
            info = dict(info)
            terminal_obs = np.asarray(obs, dtype=np.float32)
            self._episode_returns[index] += reward
            self._episode_lengths[index] += 1

            rewards[index] = reward
            dones[index] = done
            info["TimeLimit.truncated"] = bool(truncated and not terminated)
            info["training_episode"] = self._episode_numbers[index]

            if done:
                successes, trials, reasons = summarize_episode_outcome(info, multi_agent=False)
                info["is_success"] = bool(successes)
                info["successes"] = int(successes)
                info["trials"] = int(trials)
                info["terminal_reasons"] = reasons
                info["terminal_observation"] = terminal_obs.copy()
                info["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_lengths[index]),
                    "t": float(time.monotonic() - self._episode_started[index]),
                }
                obs, reset_info = self._reset_one(index)
                self.reset_infos[index] = reset_info

            observations.append(np.asarray(obs, dtype=np.float32))
            infos.append(info)

        return np.stack(observations), rewards, dones, infos

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass

    def get_attr(self, attr_name, indices=None):
        return [
            getattr(self.envs[index], attr_name, None)
            for index in self._get_indices(indices)
        ]

    def set_attr(self, attr_name, value, indices=None):
        for index in self._get_indices(indices):
            setattr(self.envs[index], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [
            getattr(self.envs[index], method_name)(*method_args, **method_kwargs)
            for index in self._get_indices(indices)
        ]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [
            _is_wrapped(self.envs[index], wrapper_class)
            for index in self._get_indices(indices)
        ]
