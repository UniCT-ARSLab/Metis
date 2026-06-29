import json
import socket
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TeamBattleGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host="127.0.0.1",
        port=5555,
        seed=0,
        num_agents=2,
        obs_dim=18,
        num_actions=8,
        timeout=30.0,
        agent_ids=None,
        teams=None,
        controlled_teams=None,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.seed_value = int(seed)
        self.agent_ids = list(agent_ids) if agent_ids is not None else [f"A{i}" for i in range(num_agents)]
        self.num_agents = len(self.agent_ids)
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.timeout = float(timeout)
        self.teams = list(teams) if teams is not None else [0]
        self.controlled_teams = list(controlled_teams) if controlled_teams is not None else list(self.teams)

        self.action_space = spaces.MultiDiscrete([num_actions] * num_agents)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_agents, obs_dim), dtype=np.float32
        )

        self.sock = socket.create_connection((host, port), timeout=self.timeout)
        self.file = self.sock.makefile("rwb")
        self._send({"cmd": "hello", "version": 1})
        reply = self._recv()
        if not reply.get("ok", False):
            raise RuntimeError(f"Handshake failed: {reply}")

    def _send(self, payload):
        self.file.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.file.flush()

    def _recv(self):
        try:
            line = self.file.readline()
        except socket.timeout as exc:
            raise RuntimeError(
                f"Timed out waiting for Godot response on {self.host}:{self.port}"
            ) from exc
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

    def _channels_to_obs(self, agents):
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        rewards = np.zeros((self.num_agents,), dtype=np.float32)
        dones = np.zeros((self.num_agents,), dtype=np.bool_)
        alive = np.zeros((self.num_agents,), dtype=np.bool_)

        by_id = {agent["id"]: agent for agent in agents}
        for i, agent_id in enumerate(self.agent_ids):
            item = by_id.get(agent_id)
            if item is None:
                continue
            obs[i] = np.asarray(item.get("obs", [0.0] * self.obs_dim), dtype=np.float32)
            rewards[i] = np.float32(item.get("reward", 0.0))
            dones[i] = bool(item.get("done", False))
            alive[i] = bool(item.get("alive", True))
        return obs, rewards, dones, alive

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed_value = int(seed)
        self._send({"cmd": "reset", "seed": self.seed_value, "teams": self.teams})
        msg = self._recv()
        obs, _, _, alive = self._channels_to_obs(msg["agents"])
        info = msg.get("info", {})
        info["agent_ids"] = list(self.agent_ids)
        info["alive_mask"] = alive
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
        action = np.asarray(action, dtype=np.int32).reshape(self.num_agents)
        action_map = {agent_id: int(action[i]) for i, agent_id in enumerate(self.agent_ids)}
        self._send({
            "cmd": "step",
            "actions": action_map,
            "teams": self.teams,
            "controlled_teams": self.controlled_teams,
        })
        msg = self._recv()

        obs, per_agent_rewards, per_agent_done, alive = self._channels_to_obs(msg["agents"])
        terminated = bool(msg.get("terminated", False))
        truncated = bool(msg.get("truncated", False))
        scalar_reward = float(np.mean(per_agent_rewards))

        info = msg.get("info", {})
        info["agent_ids"] = list(self.agent_ids)
        info["per_agent_rewards"] = per_agent_rewards
        info["per_agent_done"] = per_agent_done
        info["alive_mask"] = alive

        return obs, scalar_reward, terminated, truncated, info

    def close(self):
        try:
            self._send({"cmd": "close"})
        except Exception:
            pass
        try:
            self.file.close()
        finally:
            self.sock.close()
