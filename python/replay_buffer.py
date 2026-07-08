import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def add_many(self, obs, actions, rewards, next_obs, dones, max_items=None):
        count = len(actions)
        if max_items is not None and max_items > 0:
            count = min(count, int(max_items))
        for idx in range(count):
            self.add(obs[idx], actions[idx], rewards[idx], next_obs[idx], dones[idx])
        return count

    def load_demonstrations(self, path, max_items=None):
        with np.load(path) as data:
            required = ("obs", "actions", "rewards", "next_obs", "dones")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"Demonstration file {path!r} is missing arrays: {missing}")
            return self.add_many(
                data["obs"],
                data["actions"],
                data["rewards"],
                data["next_obs"],
                data["dones"],
                max_items=max_items,
            )

    def sample(self, batch_size, action_dtype=np.int32):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=action_dtype),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
