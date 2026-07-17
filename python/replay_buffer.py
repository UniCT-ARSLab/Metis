import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity=100000,
        *,
        prioritized=False,
        priority_alpha=0.3,
        priority_beta=1.0,
        demo_priority_bonus=1.0,
    ):
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
        self.prioritized = bool(prioritized)
        self.priority_alpha = float(priority_alpha)
        self.priority_beta = float(priority_beta)
        self.demo_priority_bonus = float(demo_priority_bonus)
        self._priorities = np.ones((self.capacity,), dtype=np.float32)
        self._is_demo = np.zeros((self.capacity,), dtype=np.bool_)
        self._protected_demo_count = 0
        self._max_priority = 1.0
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

    def add(self, obs, action, reward, next_obs, done, *, is_demo=False):
        if self._obs is None:
            self._initialize(obs, action, next_obs)
        idx = self._position
        if self._protected_demo_count and idx < self._protected_demo_count:
            idx = self._protected_demo_count
        self._obs[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_obs[idx] = next_obs
        self._dones[idx] = done
        self._is_demo[idx] = bool(is_demo)
        self._priorities[idx] = self._max_priority + (
            self.demo_priority_bonus if is_demo else 0.0
        )
        self._position = idx + 1
        if self._position >= self.capacity:
            self._position = self._protected_demo_count
        self._size = min(self._size + 1, self.capacity)

    def add_many(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        max_items=None,
        *,
        is_demo=False,
        protect=False,
    ):
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

        if protect:
            if self._size != 0:
                raise ValueError("Protected demonstrations must be loaded into an empty replay buffer")
            count = min(count, self.capacity - 1)
            if count <= 0:
                raise ValueError("Replay capacity must leave room for at least one online transition")
            obs = obs[:count]
            actions = actions[:count]
            rewards = rewards[:count]
            next_obs = next_obs[:count]
            dones = dones[:count]
            if self._obs is None:
                self._initialize(obs[0], actions[0], next_obs[0])
            self._obs[:count] = obs
            self._actions[:count] = actions
            self._rewards[:count] = rewards
            self._next_obs[:count] = next_obs
            self._dones[:count] = dones
            self._is_demo[:count] = True
            self._priorities[:count] = self._max_priority + self.demo_priority_bonus
            self._max_priority = max(self._max_priority, float(np.max(self._priorities[:count])))
            self._protected_demo_count = count
            self._size = count
            self._position = count
            return count

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
        self._is_demo[first] = bool(is_demo)
        self._priorities[first] = self._max_priority + (
            self.demo_priority_bonus if is_demo else 0.0
        )

        remaining = count - first_count
        if remaining > 0:
            second = slice(0, remaining)
            self._obs[second] = obs[first_count:]
            self._actions[second] = actions[first_count:]
            self._rewards[second] = rewards[first_count:]
            self._next_obs[second] = next_obs[first_count:]
            self._dones[second] = dones[first_count:]
            self._is_demo[second] = bool(is_demo)
            self._priorities[second] = self._max_priority + (
                self.demo_priority_bonus if is_demo else 0.0
            )

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
        self._protected_demo_count = 0
        self._max_priority = 1.0

    def export_snapshot(self):
        if self._size <= 0:
            raise ValueError("Cannot snapshot an empty replay buffer")
        if self._size < self.capacity:
            indices = np.arange(self._size)
        elif self._protected_demo_count:
            indices = np.concatenate((
                np.arange(0, self._protected_demo_count),
                np.arange(self._position, self.capacity),
                np.arange(self._protected_demo_count, self._position),
            ))
        else:
            indices = np.concatenate((
                np.arange(self._position, self.capacity),
                np.arange(0, self._position),
            ))
        return {
            "version": np.asarray([3], dtype=np.int32),
            "capacity": np.asarray([self.capacity], dtype=np.int64),
            "obs": self._obs[indices].copy(),
            "actions": self._actions[indices].copy(),
            "rewards": self._rewards[indices].copy(),
            "next_obs": self._next_obs[indices].copy(),
            "dones": self._dones[indices].copy(),
            "priorities": self._priorities[indices].copy(),
            "is_demo": self._is_demo[indices].copy(),
            "protected_demo_count": np.asarray([self._protected_demo_count], dtype=np.int64),
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
            protected = int(data["protected_demo_count"][0]) if "protected_demo_count" in data else 0
            if protected > 0 and count > limit:
                protected = min(protected, max(0, limit - 1))
                online_count = limit - protected
                indices = np.concatenate((
                    np.arange(protected, dtype=np.int64),
                    np.arange(count - online_count, count, dtype=np.int64),
                ))
            else:
                start = max(0, count - limit)
                indices = np.arange(start, count, dtype=np.int64)
                if start > 0:
                    protected = 0
            arrays = {key: np.asarray(data[key][indices]) for key in required}
            priorities = np.asarray(data["priorities"][indices]) if "priorities" in data else None
            is_demo = np.asarray(data["is_demo"][indices]) if "is_demo" in data else None

        if clear:
            self.clear()
        loaded = self.add_many(
            arrays["obs"],
            arrays["actions"],
            arrays["rewards"],
            arrays["next_obs"],
            arrays["dones"],
        )
        if priorities is not None:
            self._priorities[:loaded] = priorities[:loaded]
            self._max_priority = max(float(np.max(self._priorities[:loaded])), 1.0)
        if is_demo is not None:
            self._is_demo[:loaded] = is_demo[:loaded]
        if protected > 0 and loaded > 1:
            self._protected_demo_count = min(protected, loaded, self.capacity - 1)
            self._is_demo[:self._protected_demo_count] = True
            if self._position < self._protected_demo_count:
                self._position = self._protected_demo_count
        return loaded

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

    def sample_prioritized(self, batch_size, action_dtype=np.int32, beta=None):
        if int(batch_size) > self._size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from replay_size={self._size}")
        alpha = max(0.0, self.priority_alpha)
        scaled = np.power(np.maximum(self._priorities[:self._size], 1e-8), alpha)
        probabilities = scaled / np.sum(scaled)
        indices = np.random.choice(self._size, size=int(batch_size), replace=True, p=probabilities)
        importance_beta = self.priority_beta if beta is None else float(beta)
        weights = np.power(self._size * probabilities[indices], -importance_beta)
        weights /= max(float(np.max(weights)), 1e-8)
        return (
            self._obs[indices],
            self._actions[indices].astype(action_dtype, copy=False),
            self._rewards[indices],
            self._next_obs[indices],
            self._dones[indices].astype(np.float32, copy=False),
            indices.astype(np.int64),
            weights.astype(np.float32),
            self._is_demo[indices].copy(),
        )

    def update_priorities(self, indices, td_errors):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        values = np.abs(np.asarray(td_errors, dtype=np.float32).reshape(-1)) + 1e-6
        if len(indices) != len(values):
            raise ValueError("Priority indices and TD errors must have the same length")
        demo_bonus = self._is_demo[indices].astype(np.float32) * self.demo_priority_bonus
        updated = values + demo_bonus
        self._priorities[indices] = updated
        if len(updated):
            self._max_priority = max(self._max_priority, float(np.max(updated)))

    @property
    def protected_demo_count(self):
        return self._protected_demo_count

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
