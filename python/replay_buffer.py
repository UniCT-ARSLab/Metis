import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Replay buffer capacity must be greater than zero")
        self._size = 0
        self._position = 0
        self._obs = None
        self._actions = None
        self._rewards = np.empty((self.capacity,), dtype=np.float32)
        self._next_obs = None
        self._dones = np.empty((self.capacity,), dtype=np.bool_)
        self._save_executor = None
        self._save_futures = []

    def _initialize(self, obs, action, next_obs):
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        action_array = np.asarray(action)
        action_dtype = np.int32 if np.issubdtype(action_array.dtype, np.integer) else np.float32
        self._obs = np.empty((self.capacity,) + obs.shape, dtype=np.float32)
        self._next_obs = np.empty((self.capacity,) + next_obs.shape, dtype=np.float32)
        self._actions = np.empty((self.capacity,) + action_array.shape, dtype=action_dtype)

    def add(self, obs, action, reward, next_obs, done):
        if self._obs is None:
            self._initialize(obs, action, next_obs)
        idx = self._position
        self._obs[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_obs[idx] = next_obs
        self._dones[idx] = done
        self._position = (idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_many(self, obs, actions, rewards, next_obs, dones, max_items=None):
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool_)
        count = len(actions)
        if max_items is not None and max_items > 0:
            count = min(count, int(max_items))
        if count <= 0:
            return 0

        obs = obs[:count]
        actions = actions[:count]
        rewards = rewards[:count]
        next_obs = next_obs[:count]
        dones = dones[:count]
        if count >= self.capacity:
            obs = obs[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_obs = next_obs[-self.capacity:]
            dones = dones[-self.capacity:]
            count = self.capacity

        if self._obs is None:
            self._initialize(obs[0], actions[0], next_obs[0])

        first_count = min(count, self.capacity - self._position)
        first = slice(self._position, self._position + first_count)
        self._obs[first] = obs[:first_count]
        self._actions[first] = actions[:first_count]
        self._rewards[first] = rewards[:first_count]
        self._next_obs[first] = next_obs[:first_count]
        self._dones[first] = dones[:first_count]

        remaining = count - first_count
        if remaining > 0:
            second = slice(0, remaining)
            self._obs[second] = obs[first_count:]
            self._actions[second] = actions[first_count:]
            self._rewards[second] = rewards[first_count:]
            self._next_obs[second] = next_obs[first_count:]
            self._dones[second] = dones[first_count:]

        self._position = (self._position + count) % self.capacity
        self._size = min(self._size + count, self.capacity)
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

    def clear(self):
        self._size = 0
        self._position = 0

    def export_snapshot(self):
        if self._size <= 0:
            raise ValueError("Cannot snapshot an empty replay buffer")
        if self._size < self.capacity:
            indices = np.arange(self._size)
        else:
            indices = np.concatenate((
                np.arange(self._position, self.capacity),
                np.arange(0, self._position),
            ))
        return {
            "version": np.asarray([2], dtype=np.int32),
            "capacity": np.asarray([self.capacity], dtype=np.int64),
            "obs": self._obs[indices].copy(),
            "actions": self._actions[indices].copy(),
            "rewards": self._rewards[indices].copy(),
            "next_obs": self._next_obs[indices].copy(),
            "dones": self._dones[indices].copy(),
        }

    @staticmethod
    def write_snapshot(path, snapshot):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(path.name + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **snapshot)
        os.replace(temporary_path, path)
        return len(snapshot["obs"])

    def save(self, path):
        return self.write_snapshot(path, self.export_snapshot())

    def save_async(self, path):
        snapshot = self.export_snapshot()
        if self._save_executor is None:
            self._save_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="replay-save",
            )
        future = self._save_executor.submit(self.write_snapshot, path, snapshot)
        self._save_futures.append(future)
        return len(snapshot["obs"]), future

    def wait_for_pending_saves(self):
        futures = self._save_futures
        self._save_futures = []
        for future in futures:
            future.result()
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=True)
            self._save_executor = None

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
            limit = self.capacity
            if max_items is not None and max_items > 0:
                limit = min(limit, int(max_items))
            start = max(0, count - limit)
            arrays = {key: np.asarray(data[key][start:count]) for key in required}

        if clear:
            self.clear()
        return self.add_many(
            arrays["obs"],
            arrays["actions"],
            arrays["rewards"],
            arrays["next_obs"],
            arrays["dones"],
        )

    def sample(self, batch_size, action_dtype=np.int32):
        if int(batch_size) > self._size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from replay_size={self._size}")
        indices = np.random.randint(0, self._size, size=int(batch_size))
        return (
            self._obs[indices],
            self._actions[indices].astype(action_dtype, copy=False),
            self._rewards[indices],
            self._next_obs[indices],
            self._dones[indices].astype(np.float32, copy=False),
        )

    def sample_batches(self, num_batches, batch_size, action_dtype=np.int32):
        num_batches = int(num_batches)
        batch_size = int(batch_size)
        if num_batches <= 0:
            raise ValueError("num_batches must be greater than zero")
        if batch_size > self._size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from replay_size={self._size}")
        indices = np.random.randint(0, self._size, size=(num_batches, batch_size))
        return (
            self._obs[indices],
            self._actions[indices].astype(action_dtype, copy=False),
            self._rewards[indices],
            self._next_obs[indices],
            self._dones[indices].astype(np.float32, copy=False),
        )

    def __len__(self):
        return self._size
