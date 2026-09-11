import json
import socket
from pathlib import Path
from collections import OrderedDict

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'gymnasium'. Install project requirements with: "
        "python -m pip install -r python/requirements.txt"
    ) from exc


class ScenarioGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host="127.0.0.1",
        port=5555,
        seed=0,
        timeout=30.0,
        agent_id=None,
        multi_agent=False,
    ):
        super().__init__()
        self.host = host
        self.port = int(port)
        self.seed_value = int(seed)
        self.timeout = float(timeout)
        self.multi_agent = bool(multi_agent)

        self.sock = socket.create_connection((host, self.port), timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.file = self.sock.makefile("rwb")

        self._send({"cmd": "hello", "version": 1})
        hello = self._recv()
        if not hello.get("ok", False):
            raise RuntimeError(f"Handshake failed: {hello}")
        self.execution_mode = str(
            hello.get("execution_mode", "lockstep" if hello.get("lockstep", True) else "realtime")
        )

        self.spec = self._request_spec()
        self.agent_specs = list(self.spec.get("agents", []))
        if not self.agent_specs:
            raise RuntimeError(f"Godot scenario returned no agents in spec: {self.spec}")

        self.agent_ids = [str(item["id"]) for item in self.agent_specs]
        self.agent_team_ids = [item.get("team_id") for item in self.agent_specs]
        self.agent_team_by_id = dict(zip(self.agent_ids, self.agent_team_ids))
        self.agent_policy_ids = [
            str(item.get("policy_id", "shared") or "shared")
            for item in self.agent_specs
        ]
        self.agent_policy_by_id = dict(zip(self.agent_ids, self.agent_policy_ids))
        if agent_id is None:
            self.agent_id = self.agent_ids[0]
        else:
            self.agent_id = str(agent_id)
            if self.agent_id not in self.agent_ids:
                raise RuntimeError(f"agent_id={self.agent_id!r} not in Godot spec: {self.agent_ids}")

        first_spec = self._spec_for_agent(self.agent_id)
        self.obs_dim = int(first_spec["obs_dim"])
        self.action_type = str(first_spec.get("action_type", "discrete"))
        self.action_size = int(first_spec.get("action_size", first_spec.get("num_actions", 0)))
        self.num_actions = int(first_spec.get("num_actions", self.action_size))
        self.action_names = list(first_spec.get("action_names", []))
        self.action_low = np.asarray(first_spec.get("action_low", [-1.0] * self.action_size), dtype=np.float32)
        self.action_high = np.asarray(first_spec.get("action_high", [1.0] * self.action_size), dtype=np.float32)
        self.action_space_spec = OrderedDict(first_spec.get("action_space", {}))
        if self.action_space_spec and not self.action_names:
            self.action_names = list(self.action_space_spec.keys())

        if self.multi_agent:
            dims = {int(item["obs_dim"]) for item in self.agent_specs}
            action_types = {str(item.get("action_type", "discrete")) for item in self.agent_specs}
            action_sizes = {int(item.get("action_size", item.get("num_actions", 0))) for item in self.agent_specs}
            if len(dims) != 1 or len(action_types) != 1 or len(action_sizes) != 1:
                raise RuntimeError("multi_agent=True currently requires equal obs_dim, action_type and action_size")
            self.obs_dim = dims.pop()
            self.action_type = action_types.pop()
            self.action_size = action_sizes.pop()
            self.num_actions = self.action_size
            if self.action_type == "continuous":
                lows = np.tile(self.action_low, (len(self.agent_specs), 1))
                highs = np.tile(self.action_high, (len(self.agent_specs), 1))
                self.action_space = spaces.Box(low=lows, high=highs, dtype=np.float32)
            elif self.action_type == "hybrid":
                self.action_space = spaces.Dict({
                    agent_id: self._gym_space_from_action_spec(self._spec_for_agent(agent_id).get("action_space", {}))
                    for agent_id in self.agent_ids
                })
            else:
                self.action_space = spaces.MultiDiscrete([self.num_actions] * len(self.agent_specs))
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.agent_specs), self.obs_dim),
                dtype=np.float32,
            )
        else:
            if self.action_type == "continuous":
                self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
            elif self.action_type == "hybrid":
                self.action_space = self._gym_space_from_action_spec(self.action_space_spec)
            else:
                self.action_space = spaces.Discrete(self.num_actions)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )

    def _request_spec(self):
        self._send({"cmd": "spec"})
        msg = self._recv()
        if not msg.get("ok", False):
            raise RuntimeError(f"Godot spec failed: {msg}")
        return msg

    def request_spec(self):
        """Read the current live contract after any runtime configuration changes."""
        return self._request_spec()

    def _spec_for_agent(self, agent_id):
        for item in self.agent_specs:
            if str(item["id"]) == agent_id:
                return item
        raise KeyError(agent_id)

    def _gym_space_from_action_spec(self, action_space_spec):
        components = OrderedDict(action_space_spec or {})
        if not components:
            raise RuntimeError("Cannot build hybrid action space from an empty action_space spec")

        result = OrderedDict()
        for name, component in components.items():
            component = dict(component)
            action_type = str(component.get("action_type", component.get("type", "discrete")))
            size = int(component.get("size", 1))
            if action_type == "continuous":
                low = self._component_bound(component.get("low", -1.0), size, -1.0)
                high = self._component_bound(component.get("high", 1.0), size, 1.0)
                result[str(name)] = spaces.Box(low=low, high=high, dtype=np.float32)
            elif action_type == "discrete":
                result[str(name)] = spaces.Discrete(size)
            else:
                raise RuntimeError(f"Unsupported action component type {action_type!r} for {name!r}")
        return spaces.Dict(result)

    def _component_bound(self, value, size, default):
        array = np.asarray(value if isinstance(value, (list, tuple)) else [value], dtype=np.float32)
        if array.size == 0:
            array = np.asarray([default], dtype=np.float32)
        if array.size == 1:
            return np.full((size,), float(array[0]), dtype=np.float32)
        if array.size != size:
            raise RuntimeError(f"Action bound has size {array.size}, expected {size}")
        return array.astype(np.float32)

    def _jsonable_action(self, action):
        if isinstance(action, np.ndarray):
            return action.astype(np.float32).tolist()
        if isinstance(action, np.generic):
            return action.item()
        if isinstance(action, dict):
            return {str(key): self._jsonable_action(value) for key, value in action.items()}
        if isinstance(action, (list, tuple)):
            return [self._jsonable_action(value) for value in action]
        return action

    def _send(self, payload):
        self.file.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.file.flush()

    def _recv(self):
        try:
            line = self.file.readline()
        except socket.timeout as exc:
            raise RuntimeError(f"Timed out waiting for Godot response on {self.host}:{self.port}") from exc
        if not line:
            raise RuntimeError(f"Godot disconnected on {self.host}:{self.port}\n{self._godot_log_tail()}")
        return json.loads(line.decode("utf-8"))

    def _godot_log_tail(self, lines=30):
        log_path = Path(__file__).resolve().parent / "logs" / f"godot_{self.port}.log"
        if not log_path.exists():
            return f"Log not found: {log_path}"
        text = log_path.read_text(errors="ignore").splitlines()
        tail = "\n".join(text[-lines:]) if text else "<empty log file>"
        return f"Last lines from {log_path}:\n{tail}"

    def _agent_items_by_id(self, msg):
        return {str(item["id"]): item for item in msg.get("agents", [])}

    def _single_agent_item_from_msg(self, msg):
        return self._agent_items_by_id(msg).get(self.agent_id)

    def _single_obs_from_msg(self, msg):
        if "obs" in msg:
            return np.asarray(msg["obs"], dtype=np.float32)

        item = self._single_agent_item_from_msg(msg)
        if item is None:
            raise RuntimeError(f"Missing agent {self.agent_id!r} in message: {msg}")
        return np.asarray(item.get("obs", []), dtype=np.float32)

    def _multi_obs_from_msg(self, msg):
        by_id = self._agent_items_by_id(msg)
        obs = np.zeros((len(self.agent_specs), self.obs_dim), dtype=np.float32)
        rewards = np.zeros((len(self.agent_specs),), dtype=np.float32)
        done = np.zeros((len(self.agent_specs),), dtype=np.bool_)
        # Kept apart from `done`: a value estimator must bootstrap through a time-limit
        # truncation but not through a real terminal, so the two cannot be fused.
        terminated = np.zeros((len(self.agent_specs),), dtype=np.bool_)
        infos = []

        for idx, agent_id in enumerate(self.agent_ids):
            item = by_id.get(agent_id)
            if item is None:
                infos.append({})
                continue
            obs[idx] = np.asarray(item.get("obs", [0.0] * self.obs_dim), dtype=np.float32)
            rewards[idx] = np.float32(item.get("reward", 0.0))
            done[idx] = bool(item.get("done", False))
            # Older bridges only sent "done"; fall back to it so they degrade to the
            # previous behaviour instead of silently reporting nothing as terminal.
            terminated[idx] = bool(item.get("terminated", item.get("done", False)))
            infos.append(dict(item.get("info", {})))
        return obs, rewards, done, terminated, infos

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed_value = int(seed)
        request = {"cmd": "reset", "seed": self.seed_value}
        if options:
            request["preserve_state"] = bool(options.get("preserve_state", False))
        self._send(request)
        msg = self._recv()

        if self.multi_agent:
            obs, _, done, terminated, agent_infos = self._multi_obs_from_msg(msg)
            info = msg.get("info", {})
            info["agent_ids"] = list(self.agent_ids)
            info["per_agent_done"] = done
            info["per_agent_terminated"] = terminated
            info["per_agent_infos"] = agent_infos
            return obs, info

        obs = self._single_obs_from_msg(msg)
        info = msg.get("info", {})
        item = self._single_agent_item_from_msg(msg)
        info["agent_info"] = (
            dict(item.get("info", {}))
            if isinstance(item, dict)
            else {}
        )
        info["agent_id"] = self.agent_id
        info["agent_ids"] = list(self.agent_ids)
        return obs, info

    def configure(self, **config):
        clean_config = {key: value for key, value in config.items() if value is not None}
        if not clean_config:
            return {}
        self._send({"cmd": "config", "config": clean_config})
        msg = self._recv()
        if not msg.get("ok", False):
            raise RuntimeError(f"Godot config failed: {msg}")
        return msg

    def set_execution_mode(self, mode, simulation_fps=60):
        mode = str(mode).lower()
        if mode not in {"lockstep", "realtime"}:
            raise ValueError("execution mode must be 'lockstep' or 'realtime'")
        simulation_fps = int(simulation_fps)
        if simulation_fps < 1:
            raise ValueError("simulation_fps must be at least 1")
        self._send({
            "cmd": "execution_mode",
            "mode": mode,
            "simulation_fps": simulation_fps,
        })
        msg = self._recv()
        if not msg.get("ok", False):
            raise RuntimeError(f"Godot execution-mode change failed: {msg}")
        self.execution_mode = str(msg.get("mode", mode))
        return msg

    def step(self, action):
        if self.multi_agent:
            if isinstance(action, dict):
                action_payload = {
                    str(agent_id): self._jsonable_action(value)
                    for agent_id, value in action.items()
                }
            elif self.action_type == "continuous":
                action_array = np.asarray(action, dtype=np.float32).reshape(len(self.agent_specs), self.action_size)
                action_payload = {
                    agent_id: action_array[idx].tolist()
                    for idx, agent_id in enumerate(self.agent_ids)
                }
            elif self.action_type == "hybrid":
                if isinstance(action, dict) and all(agent_id in action for agent_id in self.agent_ids):
                    action_payload = {
                        agent_id: self._jsonable_action(action[agent_id])
                        for agent_id in self.agent_ids
                    }
                else:
                    action_payload = {
                        agent_id: self._jsonable_action(action[idx])
                        for idx, agent_id in enumerate(self.agent_ids)
                    }
            else:
                action_array = np.asarray(action, dtype=np.int32).reshape(len(self.agent_specs))
                action_payload = {
                    agent_id: int(action_array[idx])
                    for idx, agent_id in enumerate(self.agent_ids)
                }
        else:
            if isinstance(action, str):
                action_payload = {self.agent_id: action}
            elif self.action_type == "continuous":
                action_payload = {self.agent_id: np.asarray(action, dtype=np.float32).reshape(self.action_size).tolist()}
            elif self.action_type == "hybrid":
                action_payload = {self.agent_id: self._jsonable_action(action)}
            else:
                action_payload = {self.agent_id: int(action)}

        self._send({"cmd": "step", "actions": action_payload})
        msg = self._recv()

        info = msg.get("info", {})

        if self.multi_agent:
            terminated = bool(msg.get("terminated", msg.get("done", False)))
            truncated = bool(msg.get("truncated", False))
            obs, rewards, done, per_agent_terminated, agent_infos = self._multi_obs_from_msg(msg)
            info["agent_ids"] = list(self.agent_ids)
            info["per_agent_rewards"] = rewards
            info["per_agent_done"] = done
            info["per_agent_terminated"] = per_agent_terminated
            info["per_agent_infos"] = agent_infos
            return obs, float(np.mean(rewards)), terminated, truncated, info

        item = self._single_agent_item_from_msg(msg)
        obs = self._single_obs_from_msg(msg)
        if item is None:
            reward = float(msg.get("reward", 0.0))
            terminated = bool(msg.get("terminated", msg.get("done", False)))
            truncated = bool(msg.get("truncated", False))
            agent_info = {}
        else:
            reward = float(item.get("reward", msg.get("reward", 0.0)))
            terminated = bool(item.get("terminated", msg.get("terminated", item.get("done", False))))
            truncated = bool(item.get("truncated", msg.get("truncated", False)))
            agent_info = item.get("info", {})

        info["agent_id"] = self.agent_id
        info["agent_ids"] = list(self.agent_ids)
        info["agent_info"] = agent_info
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._send({"cmd": "close"})
        except Exception:
            pass
        try:
            self.file.close()
        finally:
            self.sock.close()

    def agent_summary(self, sample_size=4):
        count = len(self.agent_ids)
        if count <= sample_size * 2:
            return f"agent_count={count} agents={self.agent_ids}"
        sample = self.agent_ids[:sample_size] + ["..."] + self.agent_ids[-sample_size:]
        return f"agent_count={count} agents_sample={sample}"

    def team_summary(self):
        teams = sorted({team_id for team_id in self.agent_team_ids if team_id is not None}, key=str)
        missing = sum(team_id is None for team_id in self.agent_team_ids)
        return f"teams={teams} unassigned={missing}"

    def policy_summary(self):
        policies = sorted(set(self.agent_policy_ids))
        return f"declared_policies={policies}"
