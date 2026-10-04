"""OOD evaluation utilities and standalone checkpoint evaluator for UC-VT."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from algorithm import UCVTAgent
from buffers import MultiTaskBuffer
from config import Config
from envs import make_env, sample_task
from utils.checkpoint import CheckpointManager


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_context_row(s, a, r, sp):
    return np.concatenate([s, a, [r], sp]).astype(np.float32)


def get_context_batch(buffer, task_id, n, rng):
    buf = buffer.get(task_id)
    if buf.size == 0:
        return None
    s, a, r, sp = buf.sample_context(n, rng)
    rows = np.stack([build_context_row(s[i], a[i], r[i], sp[i]) for i in range(s.shape[0])])
    return rows[None, :, :]


def sample_prior_z(latent_dim, device):
    return torch.randn(latent_dim, device=device)


def rollout_episode(env, agent, z, cfg, greedy: bool = False):
    obs, _ = env.reset()
    transitions = []
    ep_return, ep_disc_return, disc = 0.0, 0.0, 1.0
    for _ in range(cfg.episode_len):
        action = agent.act(obs, z, deterministic=greedy)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated
        transitions.append((obs, action, reward, next_obs, done))
        ep_return += reward
        ep_disc_return += disc * reward
        disc *= cfg.gamma
        obs = next_obs
        if terminated or truncated:
            break
    return transitions, ep_return, ep_disc_return


def evaluate_agent(agent, task_family, cfg, device, rng, iteration=None, log_writer=None):
    """Evaluate train/near/mid/far bins using context adaptation.

    This is the same evaluation procedure previously embedded in train.py; it is
    separated here so both training and standalone checkpoint evaluation use one
    implementation.
    """
    results = {}
    for bin_name in ("train", "near", "mid", "far"):
        returns = []
        for _ in range(cfg.eval_episodes_per_bin):
            z_task = sample_task(task_family, bin_name, rng)
            env = make_env(task_family)
            env.set_task(z_task)

            eval_buffer = MultiTaskBuffer(env.observation_space.shape[0], env.action_space.shape[0])
            z_latent = sample_prior_z(cfg.latent_dim, device)
            for _ in range(cfg.eval_context_episodes):
                transitions, _, _ = rollout_episode(env, agent, z_latent, cfg, greedy=False)
                for transition in transitions:
                    eval_buffer.get(0).add(*transition)
                ctx = get_context_batch(eval_buffer, 0, cfg.context_size, rng)
                if ctx is not None:
                    with torch.no_grad():
                        z_latent, _, _ = agent.encode(ctx, sample=False)
                        z_latent = z_latent.squeeze(0)

            _, _, ep_disc_return = rollout_episode(env, agent, z_latent, cfg, greedy=True)
            returns.append(ep_disc_return)
            env.close()

        mean_ret = float(np.mean(returns))
        results[bin_name] = mean_ret
        if log_writer is not None:
            log_writer.writerow([iteration, bin_name, mean_ret])
    return results


def _apply_saved_config(cfg: Config, saved: dict) -> Config:
    for key, value in saved.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate a UC-VT checkpoint on OOD bins.")
    parser.add_argument("--task_family", choices=["cheetah_vel", "walker_mass"], default=None)
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best', 'latest', or an explicit .pth/.pt checkpoint path",
    )
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes_per_bin", type=int, default=None)
    args = parser.parse_args()

    bootstrap_cfg = Config()
    if args.checkpoint_dir is not None:
        bootstrap_cfg.checkpoint_dir = args.checkpoint_dir
    manager = CheckpointManager(bootstrap_cfg.checkpoint_dir, bootstrap_cfg.max_checkpoints_to_keep)

    if args.checkpoint == "best":
        if not os.path.exists(manager.best_path):
            raise FileNotFoundError(f"best_model.pth not found in {manager.checkpoint_dir}")
        payload, checkpoint_path = manager.load_path(manager.best_path)
    elif args.checkpoint == "latest":
        payload, checkpoint_path = manager.load_latest()
    else:
        payload, checkpoint_path = manager.load_path(args.checkpoint)

    saved_cfg = payload.get("config", {})
    task_family = args.task_family or saved_cfg.get("task_family", "cheetah_vel")
    cfg = _apply_saved_config(Config(task_family=task_family), saved_cfg)
    cfg.task_family = task_family
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.eval_episodes_per_bin is not None:
        cfg.eval_episodes_per_bin = args.eval_episodes_per_bin

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_env = make_env(cfg.task_family)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()

    agent = UCVTAgent(obs_dim, action_dim, cfg, device)
    manager.restore_agent(agent, payload)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    results = evaluate_agent(agent, cfg.task_family, cfg, device, rng)
    ood_mean = float(np.mean([results[name] for name in ("near", "mid", "far")]))
    print(f"Loaded: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Results: {results}")
    print(f"Mean OOD discounted return (near/mid/far): {ood_mean:.6f}")


if __name__ == "__main__":
    main()
