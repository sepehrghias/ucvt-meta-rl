"""Meta-RL OOD task families on modern Gymnasium + native MuJoCo bindings.

Two task families, matching the spirit of the TAVT paper's cheetah-vel-ood
(reward shift) and walker-mass-ood (transition shift) environments, but
reimplemented on top of Gymnasium's MuJoCo v5 envs since the official repo's
legacy `mujoco-py` / license-key stack cannot run on modern infrastructure
(see project README for details on this decision).

Task parameter `z` is a single scalar per family:
  - cheetah_vel: target forward velocity. Reward = -|actual_vel - z| - ctrl_cost.
  - walker_mass: multiplicative body-mass scale factor (1.0 = nominal).
    Reward is Walker2d's standard forward-progress reward; the shift comes
    purely from changed dynamics (heavier/lighter body).

Both expose `set_task(z)` / `sample_task(...)` and a `TASK_RANGES` dict
describing train / near / mid / far OOD bins, used by train.py.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


TASK_RANGES = {
    "cheetah_vel": {
        "train": (0.0, 1.5),
        "near": (1.5, 2.0),
        "mid": (2.0, 2.5),
        "far": (2.5, 3.0),
    },
    "walker_mass": {
        # z is a mass-scale factor around 1.0; OOD bins are symmetric
        # distances |z - 1.0| beyond the training half-width.
        "train": (0.8, 1.2),      # |z-1| in [0, 0.2]
        "near": (0.2, 0.35),      # |z-1| in [0.2, 0.35]
        "mid": (0.35, 0.5),
        "far": (0.5, 0.7),
    },
}


def sample_task(family: str, bin_name: str, rng: np.random.Generator) -> float:
    lo, hi = TASK_RANGES[family][bin_name]
    if family == "cheetah_vel":
        return float(rng.uniform(lo, hi))
    # walker_mass: bins are distances from 1.0, sample a signed offset then apply
    dist = float(rng.uniform(lo, hi))
    sign = rng.choice([-1.0, 1.0])
    return float(1.0 + sign * dist)


class CheetahVelOOD(gym.Wrapper):
    """HalfCheetah with a task-specific target forward velocity."""

    def __init__(self, ctrl_cost_weight: float = 0.1):
        env = gym.make("HalfCheetah-v5", exclude_current_positions_from_observation=False)
        super().__init__(env)
        self.ctrl_cost_weight = ctrl_cost_weight
        self._task_z = 0.0
        obs_dim = self.observation_space.shape[0] + 1  # + task scalar appended
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def set_task(self, z: float):
        self._task_z = float(z)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, [self._task_z]]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["task"] = self._task_z
        return self._augment(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        x_velocity = float(info.get("x_velocity", 0.0))
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        reward = -abs(x_velocity - self._task_z) - ctrl_cost
        info["task"] = self._task_z
        info["x_velocity"] = x_velocity
        return self._augment(obs), reward, terminated, truncated, info


class WalkerMassOOD(gym.Wrapper):
    """Walker2d with a task-specific global body-mass scale factor."""

    def __init__(self):
        env = gym.make("Walker2d-v5", exclude_current_positions_from_observation=False)
        super().__init__(env)
        model = self.env.unwrapped.model
        self._base_mass = model.body_mass.copy()
        self._task_z = 1.0
        obs_dim = self.observation_space.shape[0] + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def set_task(self, z: float):
        self._task_z = float(z)
        model = self.env.unwrapped.model
        model.body_mass[:] = self._base_mass * self._task_z

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, [self._task_z]]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["task"] = self._task_z
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["task"] = self._task_z
        return self._augment(obs), reward, terminated, truncated, info


def make_env(family: str) -> gym.Wrapper:
    if family == "cheetah_vel":
        return CheetahVelOOD()
    if family == "walker_mass":
        return WalkerMassOOD()
    raise ValueError(f"Unknown task family: {family}")
