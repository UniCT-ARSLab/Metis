import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from core.evaluation import agent_succeeded


def _is_wrapped(env, wrapper_class):
    current = env
    while current is not None:
        if isinstance(current, wrapper_class):
            return True
        current = getattr(current, "env", None)
    return False


class MetisSB3MultiAgentVecEnv(VecEnv):
    """Expose shared-policy agents as SB3 vector lanes.

    A Godot process is still one physical world. Its agent actions are collected from
    adjacent SB3 lanes and sent in one bridge step, and the world is reset once for the
    whole group. By default, partial per-agent termination is rejected because SB3
    expects each vector lane to be independently resettable. `reset-all` converts the
    still-active lanes to truncations and resets the complete Godot world.
    """

    def __init__(
        self,
        envs,
        *,
        action_codec,
        max_steps,
        physics_frames_per_step=1,
        episode_seed_multiplier=1000,
        training_episode_start=0,
        parallel_steps=True,
        partial_done_mode="error",
        reset_progress_curriculum=False,
        reset_progress_start_max=0.025,
        reset_progress_end_max=0.35,
        reset_progress_ramp_episodes=400,
    ):
        if not envs:
            raise ValueError("MetisSB3MultiAgentVecEnv requires at least one environment")
        if partial_done_mode not in {"error", "reset-all"}:
            raise ValueError("partial_done_mode must be 'error' or 'reset-all'")

        self.envs = list(envs)
        self.raw_envs = list(envs)
        self.action_codec = action_codec
        self.partial_done_mode = partial_done_mode
        self.agent_count = len(self.envs[0].agent_ids)
        if self.agent_count < 1:
            raise ValueError("A multi-agent Godot environment returned no agents")
        self.max_steps = int(max_steps)
        self.physics_frames_per_step = int(physics_frames_per_step)
        self.episode_seed_multiplier = int(episode_seed_multiplier)
        self._next_training_episodes = [
            int(training_episode_start) for _env in self.envs
        ]
        self._episode_numbers = [-1] * len(self.envs)
        shape = (len(self.envs), self.agent_count)
        self._episode_returns = np.zeros(shape, dtype=np.float64)
        self._episode_lengths = np.zeros(shape, dtype=np.int64)
        self._episode_started = np.full(shape, time.monotonic(), dtype=np.float64)
        self._actions = None
        self._closed = False
        self.reset_progress_curriculum = bool(reset_progress_curriculum)
        self.reset_progress_start_max = float(reset_progress_start_max)
        self.reset_progress_end_max = float(reset_progress_end_max)
        self.reset_progress_ramp_episodes = max(1, int(reset_progress_ramp_episodes))
        self._executor = (
            ThreadPoolExecutor(
                max_workers=len(self.envs),
                thread_name_prefix="sb3-godot-multi-env",
            )
            if parallel_steps and len(self.envs) > 1
            else None
        )

        first = self.envs[0]
        for env in self.envs:
            if not env.multi_agent:
                raise ValueError("Every grouped SB3 environment must use multi_agent=True")
            if len(env.agent_ids) != self.agent_count:
                raise ValueError("All Godot environments must expose the same agent count")
            if env.obs_dim != first.obs_dim:
                raise ValueError("All Godot agents must expose the same observation size")
            if env.action_type != first.action_type:
                raise ValueError("All Godot agents must expose the same action type")
            if env.action_space_spec != first.action_space_spec:
                raise ValueError("All Godot agents must expose the same action components")

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(first.obs_dim,),
            dtype=np.float32,
        )
        super().__init__(
            len(self.envs) * self.agent_count,
            observation_space,
            self.action_codec.policy_space,
        )

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

    def _lane(self, env_index, agent_index):
        return env_index * self.agent_count + agent_index

    def _lane_reset_infos(self, env_index, reset_info):
        per_agent = list(reset_info.get("per_agent_infos", []))
        result = []
        for agent_index, agent_id in enumerate(self.envs[env_index].agent_ids):
            agent_info = (
                dict(per_agent[agent_index])
                if agent_index < len(per_agent) and isinstance(per_agent[agent_index], dict)
                else {}
            )
            agent_info["agent_id"] = agent_id
            agent_info["godot_env_index"] = env_index
            result.append(agent_info)
        return result

    def _reset_one(self, env_index, *, options=None):
        episode = self._next_training_episodes[env_index]
        self._next_training_episodes[env_index] += 1
        self._episode_numbers[env_index] = episode

        config = {
            "training_episode": episode,
            "max_steps": self.max_steps,
            "physics_frames_per_step": self.physics_frames_per_step,
        }
        progress_max = self._curriculum_progress_max(episode)
        if progress_max is not None:
            config.update(reset_progress_min=0.0, reset_progress_max=progress_max)
        self.envs[env_index].configure(**config)

        seed = self.episode_seed_multiplier * episode + env_index
        obs, reset_info = self.envs[env_index].reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32).reshape(
            self.agent_count,
            self.observation_space.shape[0],
        )
        self._episode_returns[env_index, :] = 0.0
        self._episode_lengths[env_index, :] = 0
        self._episode_started[env_index, :] = time.monotonic()
        return obs, self._lane_reset_infos(env_index, dict(reset_info))

    def reset(self):
        observations = []
        reset_infos = [{} for _lane in range(self.num_envs)]
        for env_index in range(len(self.envs)):
            first_lane = self._lane(env_index, 0)
            options = self._options[first_lane] if self._options[first_lane] else None
            obs, infos = self._reset_one(env_index, options=options)
            observations.extend(obs)
            for agent_index, info in enumerate(infos):
                reset_infos[self._lane(env_index, agent_index)] = info
        self.reset_infos = reset_infos
        self._reset_seeds()
        self._reset_options()
        return np.stack(observations)

    def step_async(self, actions):
        if self._actions is not None:
            raise RuntimeError("step_async called while a previous Godot step is pending")
        self._actions = actions

    def _group_actions(self, actions):
        requests = []
        for env_index, env in enumerate(self.envs):
            payload = {}
            for agent_index, agent_id in enumerate(env.agent_ids):
                lane = self._lane(env_index, agent_index)
                payload[agent_id] = self.action_codec.decode(actions[lane])
            requests.append((env, payload))
        return requests

    def _step_all(self, actions):
        requests = self._group_actions(actions)
        if self._executor is None:
            return [env.step(action) for env, action in requests]
        futures = [self._executor.submit(env.step, action) for env, action in requests]
        return [future.result() for future in futures]

    def _partial_done_error(self, env_index, done_mask):
        done_ids = [
            agent_id
            for agent_id, done in zip(self.envs[env_index].agent_ids, done_mask)
            if done
        ]
        raise RuntimeError(
            "SB3 parameter sharing cannot reset individual agents inside one Godot "
            f"world. Environment {env_index} ended only agents {done_ids}. Configure "
            "coordinated scenario termination or pass "
            "--sb3-multi-agent-partial-done reset-all."
        )

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_wait called before step_async")
        actions = self._actions
        self._actions = None
        results = self._step_all(actions)

        observations = np.zeros(
            (self.num_envs, self.observation_space.shape[0]),
            dtype=np.float32,
        )
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos = [{} for _lane in range(self.num_envs)]

        for env_index, (obs, _mean_reward, terminated, truncated, info) in enumerate(results):
            obs = np.asarray(obs, dtype=np.float32).reshape(
                self.agent_count,
                self.observation_space.shape[0],
            )
            per_rewards = np.asarray(
                info.get("per_agent_rewards", np.zeros(self.agent_count)),
                dtype=np.float32,
            ).reshape(self.agent_count)
            per_done = np.asarray(
                info.get("per_agent_done", np.zeros(self.agent_count)),
                dtype=np.bool_,
            ).reshape(self.agent_count)
            per_terminated = np.asarray(
                info.get("per_agent_terminated", per_done),
                dtype=np.bool_,
            ).reshape(self.agent_count)
            per_infos = list(info.get("per_agent_infos", []))
            global_done = bool(terminated or truncated)
            partial_done = bool(np.any(per_done) and not global_done)
            if partial_done and self.partial_done_mode == "error":
                self._partial_done_error(env_index, per_done)
            group_done = global_done or (
                partial_done and self.partial_done_mode == "reset-all"
            )

            self._episode_returns[env_index] += per_rewards
            self._episode_lengths[env_index] += 1
            reset_obs = None
            reset_infos = None

            for agent_index, agent_id in enumerate(self.envs[env_index].agent_ids):
                lane = self._lane(env_index, agent_index)
                agent_info = (
                    dict(per_infos[agent_index])
                    if agent_index < len(per_infos) and isinstance(per_infos[agent_index], dict)
                    else {}
                )
                lane_info = dict(agent_info)
                lane_info.update(
                    agent_info=agent_info,
                    agent_id=agent_id,
                    godot_env_index=env_index,
                    training_episode=self._episode_numbers[env_index],
                )
                rewards[lane] = per_rewards[agent_index]
                dones[lane] = group_done

                if group_done:
                    lane_terminated = bool(per_terminated[agent_index])
                    lane_truncated = not lane_terminated
                    lane_info["TimeLimit.truncated"] = lane_truncated
                    lane_info["terminal_observation"] = obs[agent_index].copy()
                    lane_info["is_success"] = bool(agent_succeeded(agent_info))
                    lane_info["successes"] = int(lane_info["is_success"])
                    lane_info["trials"] = 1
                    reason = agent_info.get("terminal_reason")
                    lane_info["terminal_reasons"] = [str(reason)] if reason else []
                    lane_info["episode"] = {
                        "r": float(self._episode_returns[env_index, agent_index]),
                        "l": int(self._episode_lengths[env_index, agent_index]),
                        "t": float(
                            time.monotonic()
                            - self._episode_started[env_index, agent_index]
                        ),
                    }
                observations[lane] = obs[agent_index]
                infos[lane] = lane_info

            if group_done:
                reset_obs, reset_infos = self._reset_one(env_index)
                for agent_index in range(self.agent_count):
                    lane = self._lane(env_index, agent_index)
                    observations[lane] = reset_obs[agent_index]
                    self.reset_infos[lane] = reset_infos[agent_index]

        return observations, rewards, dones, infos

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

    def _lane_pairs(self, indices):
        return [
            (lane, lane // self.agent_count)
            for lane in self._get_indices(indices)
        ]

    def get_attr(self, attr_name, indices=None):
        return [
            getattr(self.envs[env_index], attr_name, None)
            for _lane, env_index in self._lane_pairs(indices)
        ]

    def set_attr(self, attr_name, value, indices=None):
        for env_index in {env_index for _lane, env_index in self._lane_pairs(indices)}:
            setattr(self.envs[env_index], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [
            getattr(self.envs[env_index], method_name)(*method_args, **method_kwargs)
            for _lane, env_index in self._lane_pairs(indices)
        ]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [
            _is_wrapped(self.envs[env_index], wrapper_class)
            for _lane, env_index in self._lane_pairs(indices)
        ]
