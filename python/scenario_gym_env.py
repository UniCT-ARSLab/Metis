import json
import socket
from pathlib import Path

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
        self.file = self.sock.makefile("rwb")

        self._send({"cmd": "hello", "version": 1})
        hello = self._recv()
        if not hello.get("ok", False):
            raise RuntimeError(f"Handshake failed: {hello}")

        self.spec = self._request_spec()
        self.agent_specs = list(self.spec.get("agents", []))
        if not self.agent_specs:
            raise RuntimeError(f"Godot scenario returned no agents in spec: {self.spec}")

        self.agent_ids = [str(item["id"]) for item in self.agent_specs]
        if agent_id is None:
            self.agent_id = self.agent_ids[0]
        else:
            self.agent_id = str(agent_id)
            if self.agent_id not in self.agent_ids:
                raise RuntimeError(f"agent_id={self.agent_id!r} not in Godot spec: {self.agent_ids}")

        first_spec = self._spec_for_agent(self.agent_id)
        self.obs_dim = int(first_spec["obs_dim"])
        self.num_actions = int(first_spec["num_actions"])
        self.action_names = list(first_spec.get("action_names", []))

        if self.multi_agent:
            dims = {int(item["obs_dim"]) for item in self.agent_specs}
            actions = {int(item["num_actions"]) for item in self.agent_specs}
            if len(dims) != 1 or len(actions) != 1:
                raise RuntimeError("multi_agent=True currently requires equal obs_dim and num_actions for all agents")
            self.obs_dim = dims.pop()
            self.num_actions = actions.pop()
            self.action_space = spaces.MultiDiscrete([self.num_actions] * len(self.agent_specs))
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.agent_specs), self.obs_dim),
                dtype=np.float32,
            )
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

    def _spec_for_agent(self, agent_id):
        for item in self.agent_specs:
            if str(item["id"]) == agent_id:
                return item
        raise KeyError(agent_id)

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

    def _single_obs_from_msg(self, msg):
        if "obs" in msg:
            return np.asarray(msg["obs"], dtype=np.float32)

        item = self._agent_items_by_id(msg).get(self.agent_id)
        if item is None:
            raise RuntimeError(f"Missing agent {self.agent_id!r} in message: {msg}")
        return np.asarray(item.get("obs", []), dtype=np.float32)

    def _multi_obs_from_msg(self, msg):
        by_id = self._agent_items_by_id(msg)
        obs = np.zeros((len(self.agent_specs), self.obs_dim), dtype=np.float32)
        rewards = np.zeros((len(self.agent_specs),), dtype=np.float32)
        done = np.zeros((len(self.agent_specs),), dtype=np.bool_)

        for idx, agent_id in enumerate(self.agent_ids):
            item = by_id.get(agent_id)
            if item is None:
                continue
            obs[idx] = np.asarray(item.get("obs", [0.0] * self.obs_dim), dtype=np.float32)
            rewards[idx] = np.float32(item.get("reward", 0.0))
            done[idx] = bool(item.get("done", False))
        return obs, rewards, done

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed_value = int(seed)
        self._send({"cmd": "reset", "seed": self.seed_value})
        msg = self._recv()

        if self.multi_agent:
            obs, _, done = self._multi_obs_from_msg(msg)
            info = msg.get("info", {})
            info["agent_ids"] = list(self.agent_ids)
            info["per_agent_done"] = done
            return obs, info

        obs = self._single_obs_from_msg(msg)
        info = msg.get("info", {})
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

    def step(self, action):
        if self.multi_agent:
            action_array = np.asarray(action, dtype=np.int32).reshape(len(self.agent_specs))
            action_payload = {
                agent_id: int(action_array[idx])
                for idx, agent_id in enumerate(self.agent_ids)
            }
        else:
            action_payload = int(action)

        self._send({"cmd": "step", "actions": action_payload})
        msg = self._recv()

        terminated = bool(msg.get("terminated", msg.get("done", False)))
        truncated = bool(msg.get("truncated", False))
        info = msg.get("info", {})

        if self.multi_agent:
            obs, rewards, done = self._multi_obs_from_msg(msg)
            info["agent_ids"] = list(self.agent_ids)
            info["per_agent_rewards"] = rewards
            info["per_agent_done"] = done
            return obs, float(np.mean(rewards)), terminated, truncated, info

        obs = self._single_obs_from_msg(msg)
        reward = float(msg.get("reward", 0.0))
        info["agent_id"] = self.agent_id
        info["agent_ids"] = list(self.agent_ids)
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
