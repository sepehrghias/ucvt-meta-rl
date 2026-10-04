"""Robust checkpointing for UC-VT training.

Design goals:
- standard checkpoints are atomic and rotated (default: keep 3),
- ``best_model.pth`` is never touched by rotation,
- ``emergency_checkpoint.pth`` is lightweight (no replay buffer),
- every load is staged through CPU via ``map_location='cpu'`` before model
  states are copied to the agent's device,
- checkpoints include optimizer/RNG/training-loop state needed for resume.

Only load checkpoints you trust: they intentionally contain Python/NumPy RNG
objects and therefore require ``weights_only=False``.
"""

from __future__ import annotations

import glob
import os
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_VERSION = 2


def _to_cpu(obj: Any) -> Any:
    """Recursively detach tensors and move them to CPU for portable saving."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer tensor state to the parameter device after CPU loading."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def capture_rng_state(rng: np.random.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if rng is not None:
        state["numpy_generator"] = rng.bit_generator.state
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return _to_cpu(state)


def restore_rng_state(state: dict[str, Any] | None, rng: np.random.Generator | None = None) -> None:
    if not state:
        return

    # v2 names
    if "python" in state:
        random.setstate(state["python"])
    if "numpy_legacy" in state:
        np.random.set_state(state["numpy_legacy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu())
    if rng is not None and "numpy_generator" in state:
        rng.bit_generator.state = state["numpy_generator"]
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [s.cpu() for s in state["torch_cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)

    # Backward compatibility with the original checkpoint format.
    if "numpy" in state and "numpy_legacy" not in state:
        np.random.set_state(state["numpy"])
    if "torch" in state and "torch_cpu" not in state:
        torch.set_rng_state(state["torch"].cpu())


def serialize_replay_buffer(buffer) -> dict[str, Any]:
    """Serialize only populated replay-buffer rows, not empty capacity."""
    tasks: dict[int, dict[str, Any]] = {}
    for tid, task_buffer in buffer.buffers.items():
        size = int(task_buffer.size)
        if size == 0:
            continue
        n_rows = task_buffer.capacity if size == task_buffer.capacity else size
        tasks[int(tid)] = {
            "size": size,
            "ptr": int(task_buffer.ptr),
            "capacity": int(task_buffer.capacity),
            "obs": task_buffer.obs[:n_rows].copy(),
            "actions": task_buffer.actions[:n_rows].copy(),
            "rewards": task_buffer.rewards[:n_rows].copy(),
            "next_obs": task_buffer.next_obs[:n_rows].copy(),
            "dones": task_buffer.dones[:n_rows].copy(),
        }
    return {
        "obs_dim": int(buffer.obs_dim),
        "action_dim": int(buffer.action_dim),
        "capacity": int(buffer.capacity),
        "tasks": tasks,
    }


def restore_replay_buffer(buffer, state: dict[str, Any] | None) -> None:
    if not state:
        return
    if int(state.get("obs_dim", buffer.obs_dim)) != buffer.obs_dim:
        raise ValueError("Checkpoint replay-buffer obs_dim does not match current environment.")
    if int(state.get("action_dim", buffer.action_dim)) != buffer.action_dim:
        raise ValueError("Checkpoint replay-buffer action_dim does not match current environment.")

    buffer.buffers.clear()
    for raw_tid, saved in state.get("tasks", {}).items():
        tid = int(raw_tid)
        task_buffer = buffer.get(tid)
        saved_size = int(saved["size"])
        rows = int(saved["obs"].shape[0])
        if rows > task_buffer.capacity or saved_size > task_buffer.capacity:
            raise ValueError(
                f"Checkpoint buffer for task {tid} exceeds current capacity "
                f"({saved_size} > {task_buffer.capacity})."
            )
        task_buffer.obs[:rows] = saved["obs"]
        task_buffer.actions[:rows] = saved["actions"]
        task_buffer.rewards[:rows] = saved["rewards"]
        task_buffer.next_obs[:rows] = saved["next_obs"]
        task_buffer.dones[:rows] = saved["dones"]
        task_buffer.size = saved_size
        task_buffer.ptr = int(saved["ptr"]) % task_buffer.capacity


class CheckpointManager:
    """Save, rotate, discover, and restore UC-VT checkpoints."""

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3):
        self.checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
        self.max_to_keep = max(1, int(max_to_keep))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.emergency_path = os.path.join(self.checkpoint_dir, "emergency_checkpoint.pth")
        self.last_standard_path: str | None = None

    # ------------------------------- save helpers ------------------------------- #

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], path: str) -> None:
        tmp_path = f"{path}.tmp-{os.getpid()}"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _agent_payload(agent) -> tuple[dict[str, Any], dict[str, Any]]:
        agent_state = _to_cpu(agent.state_dict())
        aux = {
            "grad_scaler": _to_cpu(agent.scaler.state_dict()) if hasattr(agent, "scaler") else None,
            "uncertainty_history": [
                _to_cpu(x) for x in list(getattr(agent, "uncertainty_history", []))
            ],
            "train_task_z_cache": _to_cpu(getattr(agent, "train_task_z_cache", {})),
        }
        return agent_state, aux

    def save(
        self,
        agent,
        *,
        iteration: int,
        next_iteration: int,
        episode: int,
        cfg,
        rng: np.random.Generator,
        training_state: dict[str, Any] | None = None,
        replay_buffer=None,
        final: bool = False,
    ) -> str:
        """Write a full standard checkpoint and rotate older standard files."""
        agent_state, agent_aux = self._agent_payload(agent)
        payload: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "kind": "standard",
            "iteration": int(iteration),
            "next_iteration": int(next_iteration),
            "episode": int(episode),
            "final": bool(final),
            "agent": agent_state,
            "agent_aux": agent_aux,
            "config": dict(vars(cfg)),
            "rng_state": capture_rng_state(rng),
            "training_state": _to_cpu(training_state or {}),
        }
        if replay_buffer is not None:
            payload["replay_buffer"] = serialize_replay_buffer(replay_buffer)

        path = os.path.join(
            self.checkpoint_dir,
            f"ckpt_it{max(iteration, 0):06d}_ep{max(episode, 0):07d}.pth",
        )
        self._atomic_torch_save(payload, path)
        self.last_standard_path = path
        self.rotate()

        # A successfully written full checkpoint supersedes an old emergency file.
        if os.path.exists(self.emergency_path):
            try:
                os.remove(self.emergency_path)
            except OSError:
                pass
        return path

    def save_best(
        self,
        agent,
        *,
        score: float,
        iteration: int,
        episode: int,
        cfg,
        metric_name: str = "mean_ood_discounted_return",
    ) -> str:
        """Write/replace best_model.pth. Rotation never considers this file."""
        agent_state, agent_aux = self._agent_payload(agent)
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "kind": "best",
            "iteration": int(iteration),
            "next_iteration": int(iteration) + 1,
            "episode": int(episode),
            "best_score": float(score),
            "best_metric": metric_name,
            "agent": agent_state,
            "agent_aux": agent_aux,
            "config": dict(vars(cfg)),
        }
        self._atomic_torch_save(payload, self.best_path)
        return self.best_path

    def emergency_save(
        self,
        agent,
        *,
        iteration: int,
        next_iteration: int,
        episode: int,
        cfg,
        rng: np.random.Generator,
        training_state: dict[str, Any] | None = None,
    ) -> str:
        """Write a lightweight emergency checkpoint.

        Deliberately excludes the replay buffer to maximize the probability that
        SIGTERM/SIGINT saving completes before the process is terminated. The file
        records the newest standard checkpoint, allowing resume code to recover its
        replay buffer as a fallback while restoring the newer model/RNG state here.
        """
        agent_state, agent_aux = self._agent_payload(agent)
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "kind": "emergency",
            "iteration": int(iteration),
            "next_iteration": int(next_iteration),
            "episode": int(episode),
            "agent": agent_state,
            "agent_aux": agent_aux,
            "config": dict(vars(cfg)),
            "rng_state": capture_rng_state(rng),
            "training_state": _to_cpu(training_state or {}),
            "parent_standard_checkpoint": self.last_standard_path,
            "parent_standard_basename": (
                os.path.basename(self.last_standard_path) if self.last_standard_path else None
            ),
        }
        self._atomic_torch_save(payload, self.emergency_path)
        return self.emergency_path

    # ----------------------------- discovery/rotation ---------------------------- #

    def _standard_files(self) -> list[str]:
        # New format + backward-compatible original format.
        files = glob.glob(os.path.join(self.checkpoint_dir, "ckpt_*.pth"))
        files += glob.glob(os.path.join(self.checkpoint_dir, "ckpt_*.pt"))
        return sorted(set(files), key=os.path.getmtime)

    def rotate(self) -> None:
        files = self._standard_files()
        while len(files) > self.max_to_keep:
            old = files.pop(0)
            try:
                os.remove(old)
            except FileNotFoundError:
                pass

    def checkpoint_candidates(self, include_emergency: bool = True) -> list[str]:
        files = self._standard_files()
        if include_emergency and os.path.exists(self.emergency_path):
            files.append(self.emergency_path)
        return sorted(set(files), key=os.path.getmtime, reverse=True)

    def latest_checkpoint(self) -> str | None:
        candidates = self.checkpoint_candidates(include_emergency=True)
        return candidates[0] if candidates else None

    # ----------------------------------- load ----------------------------------- #

    def _resolve_parent_standard(self, payload: dict[str, Any]) -> str | None:
        parent = payload.get("parent_standard_checkpoint")
        if parent and os.path.exists(parent):
            return os.path.abspath(parent)

        basename = payload.get("parent_standard_basename")
        if basename:
            relocated = os.path.join(self.checkpoint_dir, basename)
            if os.path.exists(relocated):
                return relocated

        # Handles checkpoint folders copied between Kaggle sessions/machines where
        # the original absolute parent path no longer exists.
        standards = self._standard_files()
        return standards[-1] if standards else None

    @staticmethod
    def load_file(path: str) -> dict[str, Any]:
        """Always deserialize onto CPU first to avoid CUDA allocation spikes."""
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_latest(self) -> tuple[dict[str, Any], str]:
        errors: list[str] = []
        for path in self.checkpoint_candidates(include_emergency=True):
            try:
                payload = self.load_file(path)
                if os.path.basename(path).startswith("ckpt_"):
                    self.last_standard_path = path
                else:
                    parent = self._resolve_parent_standard(payload)
                    if parent is not None:
                        self.last_standard_path = parent
                return payload, path
            except Exception as exc:  # corrupt/incomplete file: fall back to previous
                errors.append(f"{path}: {exc}")
        if errors:
            raise RuntimeError("No readable checkpoint found. Errors:\n" + "\n".join(errors))
        raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

    def load_path(self, path: str) -> tuple[dict[str, Any], str]:
        resolved = os.path.abspath(os.path.expanduser(path))
        payload = self.load_file(resolved)
        if os.path.basename(resolved).startswith("ckpt_"):
            self.last_standard_path = resolved
        return payload, resolved

    @staticmethod
    def restore_agent(agent, payload: dict[str, Any]) -> None:
        if "agent" not in payload:
            raise KeyError("Checkpoint does not contain an 'agent' state.")

        # payload was loaded on CPU; load_state_dict performs the controlled copy
        # into modules already resident on agent.device.
        agent.load_state_dict(payload["agent"])

        for optimizer_name in (
            "encoder_opt", "decoder_opt", "actor_opt", "critic_opt", "alpha_opt"
        ):
            optimizer = getattr(agent, optimizer_name, None)
            if optimizer is not None:
                _optimizer_to_device(optimizer, agent.device)

        aux = payload.get("agent_aux", {})
        scaler_state = aux.get("grad_scaler")
        if scaler_state is not None and hasattr(agent, "scaler"):
            agent.scaler.load_state_dict(scaler_state)

        if hasattr(agent, "uncertainty_history") and "uncertainty_history" in aux:
            maxlen = agent.uncertainty_history.maxlen
            agent.uncertainty_history = deque(maxlen=maxlen)
            for x in aux["uncertainty_history"]:
                agent.uncertainty_history.append(x.to(agent.device))

        if hasattr(agent, "train_task_z_cache") and "train_task_z_cache" in aux:
            agent.train_task_z_cache = {
                int(k): v.to(agent.device) if torch.is_tensor(v) else v
                for k, v in aux["train_task_z_cache"].items()
            }

    def restore_replay_buffer_with_emergency_fallback(self, buffer, payload: dict[str, Any]) -> str | None:
        """Restore replay state from this checkpoint or its last full parent.

        Returns the source path when a parent standard checkpoint was used.
        """
        if "replay_buffer" in payload:
            restore_replay_buffer(buffer, payload["replay_buffer"])
            return None

        parent = self._resolve_parent_standard(payload)
        if parent is not None:
            parent_payload = self.load_file(parent)
            if "replay_buffer" in parent_payload:
                restore_replay_buffer(buffer, parent_payload["replay_buffer"])
                self.last_standard_path = parent
                return parent
        return None
