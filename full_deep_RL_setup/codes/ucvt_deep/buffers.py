"""Per-task replay buffers + context sampling for meta-RL."""

from __future__ import annotations

import numpy as np


class TaskReplayBuffer:
    """Fixed-capacity ring buffer of (s, a, r, s', done) for a single task."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, s, a, r, sp, done):
        self.obs[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_obs[self.ptr] = sp
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=min(batch_size, self.size))
        return (
            self.obs[idx], self.actions[idx], self.rewards[idx],
            self.next_obs[idx], self.dones[idx],
        )

    def sample_context(self, n: int, rng: np.random.Generator):
        """Sample a context batch of raw transitions, for the task encoder."""
        idx = rng.integers(0, self.size, size=min(n, self.size))
        return (
            self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx],
        )


class MultiTaskBuffer:
    """Dict of TaskReplayBuffer keyed by task id (index into the task list)."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 100_000):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.buffers: dict[int, TaskReplayBuffer] = {}

    def get(self, task_id: int) -> TaskReplayBuffer:
        if task_id not in self.buffers:
            self.buffers[task_id] = TaskReplayBuffer(self.obs_dim, self.action_dim, self.capacity)
        return self.buffers[task_id]

    def ready_tasks(self, min_size: int) -> list[int]:
        return [tid for tid, buf in self.buffers.items() if buf.size >= min_size]
