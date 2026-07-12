import random
from collections import deque
import os
from pathlib import Path

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

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        transitions = list(self.buffer)
        if not transitions:
            raise ValueError("Cannot save an empty replay buffer")

        obs, actions, rewards, next_obs, dones = zip(*transitions)
        temporary_path = path.with_name(path.name + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                version=np.asarray([1], dtype=np.int32),
                capacity=np.asarray([self.buffer.maxlen], dtype=np.int64),
                obs=np.asarray(obs, dtype=np.float32),
                actions=np.asarray(actions),
                rewards=np.asarray(rewards, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                dones=np.asarray(dones, dtype=np.bool_),
            )
        os.replace(temporary_path, path)
        return len(transitions)

    def load(self, path, max_items=None, clear=True):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            required = ("obs", "actions", "rewards", "next_obs", "dones")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"Replay buffer {str(path)!r} is missing arrays: {missing}")

            lengths = {key: len(data[key]) for key in required}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Replay buffer {str(path)!r} has inconsistent array lengths: {lengths}")

            count = lengths["obs"]
            limit = self.buffer.maxlen
            if max_items is not None and max_items > 0:
                limit = min(limit, int(max_items))
            start = max(0, count - limit)
            arrays = {key: np.asarray(data[key][start:count]) for key in required}

        if clear:
            self.buffer.clear()
        return self.add_many(
            arrays["obs"],
            arrays["actions"],
            arrays["rewards"],
            arrays["next_obs"],
            arrays["dones"],
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
