"""Main training entry point for UC-VT (TAVT + UC-VT) meta-RL.

Kaggle examples:
    python train.py --task_family cheetah_vel --total_iterations 2000
    python train.py --task_family cheetah_vel --total_iterations 2000 --resume

Resume checkpoints restore the agent (encoder/decoder/actor/critics/targets),
all optimizers, AMP scaler, uncertainty history, RNG states, counters, scalar
histories, task list/latents, and the replay buffer for full checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import signal
import time
from typing import Any

import numpy as np
import torch

from algorithm import UCVTAgent
from buffers import MultiTaskBuffer
from config import Config
from envs import make_env, sample_task
from evaluate import evaluate_agent
from utils.checkpoint import CheckpointManager, restore_rng_state


TRAIN_LOG_HEADER = [
    "iteration", "episode", "task_id", "ep_return", "ep_disc_return",
    "real_critic_loss", "virtual_critic_loss", "actor_loss", "kl_loss",
    "task_preserving_loss", "state_reg_loss", "decoder_sup_loss", "alpha",
]
EVAL_LOG_HEADER = ["iteration", "bin", "mean_discounted_return"]
STAT_KEYS = (
    "real_critic_loss", "virtual_critic_loss", "actor_loss", "kl_loss",
    "task_preserving_loss", "state_reg_loss", "decoder_sup_loss", "alpha",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_context_row(s, a, r, sp):
    return np.concatenate([s, a, [r], sp]).astype(np.float32)


def sample_prior_z(latent_dim, device):
    return torch.randn(latent_dim, device=device)


def get_context_batch(buffer, task_id, n, rng):
    buf = buffer.get(task_id)
    if buf.size == 0:
        return None
    s, a, r, sp = buf.sample_context(n, rng)
    rows = np.stack([build_context_row(s[i], a[i], r[i], sp[i]) for i in range(s.shape[0])])
    return rows[None, :, :]  # [1, n, in_dim]


def rollout_episode(env, agent, z, cfg, rng, greedy: bool = False):
    # rng is kept in the signature for backwards compatibility with the original
    # training helper. Environment randomness is owned by Gymnasium's env RNG.
    del rng
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


def _open_csv(path: str, header: list[str], append: bool):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    handle = open(path, "a" if append else "w", newline="")
    writer = csv.writer(handle)
    if not append or not file_exists:
        writer.writerow(header)
        handle.flush()
    return handle, writer


def _tensor_dict_to_device(values: dict[Any, Any], device: torch.device) -> dict[int, Any]:
    result = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            value = value.to(device)
        result[int(key)] = value
    return result


def _capture_env_rng_states(task_envs: dict[int, Any], pending_states: dict[int, Any]) -> dict[int, Any]:
    states = dict(pending_states)
    for tid, env in task_envs.items():
        try:
            states[int(tid)] = env.unwrapped.np_random.bit_generator.state
        except Exception:
            pass
    return states


def _restore_env_rng_state(env, state: Any) -> None:
    if state is None:
        return
    try:
        env.unwrapped.np_random.bit_generator.state = state
    except Exception:
        # Environment RNG restoration is best-effort; model/training RNG states are
        # still restored exactly from the checkpoint manager.
        pass


def _validate_resume_config(cfg: Config, saved_config: dict[str, Any]) -> None:
    if not saved_config:
        return
    critical = (
        "task_family", "latent_dim", "encoder_hidden", "decoder_hidden",
        "actor_hidden", "critic_hidden", "ensemble_size",
    )
    mismatches = []
    for key in critical:
        if key in saved_config and getattr(cfg, key) != saved_config[key]:
            mismatches.append(f"{key}: current={getattr(cfg, key)!r}, checkpoint={saved_config[key]!r}")
    if mismatches:
        raise ValueError(
            "Resume configuration is incompatible with the checkpoint:\n  - "
            + "\n  - ".join(mismatches)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_family", type=str, default="cheetah_vel", choices=["cheetah_vel", "walker_mass"])
    parser.add_argument("--total_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="resume from the newest standard/emergency checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="resume from an explicit .pth/.pt file")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true", help="tiny run to sanity-check the pipeline")
    args = parser.parse_args()

    cfg = Config(task_family=args.task_family, seed=args.seed)
    if args.total_iterations is not None:
        cfg.total_iterations = args.total_iterations
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir

    if args.smoke_test:
        cfg.n_train_tasks = 2
        cfg.meta_batch_size = 2
        cfg.total_iterations = 3 if args.total_iterations is None else args.total_iterations
        cfg.rollout_episodes_per_iter = 1
        cfg.updates_per_iter = 1
        cfg.episode_len = 20
        cfg.min_buffer_before_update = 20
        cfg.eval_every = 2
        cfg.eval_episodes_per_bin = 1
        cfg.eval_context_episodes = 1
        cfg.batch_size = 16
        cfg.context_size = 8
        cfg.bisim_probe_size = 16
        cfg.virtual_batch_size = 8
        cfg.n_virtual_per_step = 2
        cfg.state_reg_batch = 8
        cfg.checkpoint_every_episodes = 2
        if args.checkpoint_dir is None:
            cfg.checkpoint_dir = "./checkpoints"
        if args.log_dir is None:
            cfg.log_dir = "./logs"

    resume_requested = bool(args.resume or args.resume_from or cfg.resume)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, task family: {cfg.task_family}")
    print(f"Checkpoint dir: {os.path.abspath(cfg.checkpoint_dir)}")

    rng = np.random.default_rng(cfg.seed)
    probe_env = make_env(cfg.task_family)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()

    # Build the objects first. Checkpoint loading is explicitly CPU-staged inside
    # CheckpointManager before these state_dicts are copied to the selected device.
    train_tasks = [sample_task(cfg.task_family, "train", rng) for _ in range(cfg.n_train_tasks)]
    agent = UCVTAgent(obs_dim, action_dim, cfg, device)
    buffer = MultiTaskBuffer(obs_dim, action_dim)
    task_latents = {tid: sample_prior_z(cfg.latent_dim, device) for tid in range(cfg.n_train_tasks)}
    task_envs: dict[int, Any] = {}
    pending_env_rng_states: dict[int, Any] = {}

    history: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    best_score: float | None = None
    episode_count = 0
    start_iteration = 0
    last_completed_iteration = -1

    checkpoint_manager = CheckpointManager(cfg.checkpoint_dir, max_to_keep=cfg.max_checkpoints_to_keep)

    if resume_requested:
        if args.resume_from:
            checkpoint, checkpoint_path = checkpoint_manager.load_path(args.resume_from)
        else:
            checkpoint, checkpoint_path = checkpoint_manager.load_latest()

        _validate_resume_config(cfg, checkpoint.get("config", {}))
        checkpoint_manager.restore_agent(agent, checkpoint)
        replay_parent = checkpoint_manager.restore_replay_buffer_with_emergency_fallback(buffer, checkpoint)

        # Important: restore RNG *after* object construction/loading, because model
        # initialization itself consumes PyTorch random numbers.
        restore_rng_state(checkpoint.get("rng_state"), rng)

        episode_count = int(checkpoint.get("episode", 0))
        start_iteration = int(checkpoint.get(
            "next_iteration", int(checkpoint.get("iteration", -1)) + 1
        ))
        last_completed_iteration = start_iteration - 1

        saved_training = checkpoint.get("training_state", {})
        if "train_tasks" in saved_training:
            train_tasks = [float(x) for x in saved_training["train_tasks"]]
        if len(train_tasks) != cfg.n_train_tasks:
            raise ValueError(
                f"Checkpoint has {len(train_tasks)} training tasks but cfg.n_train_tasks={cfg.n_train_tasks}."
            )
        if "task_latents" in saved_training:
            task_latents = _tensor_dict_to_device(saved_training["task_latents"], device)
        history = saved_training.get("history", history)
        history.setdefault("train", [])
        history.setdefault("eval", [])
        best_score = saved_training.get("best_score", checkpoint.get("best_score"))
        pending_env_rng_states = {
            int(k): v for k, v in saved_training.get("env_rng_states", {}).items()
        }

        print(
            f"Resumed from: {checkpoint_path}\n"
            f"  kind={checkpoint.get('kind', 'legacy')} episode={episode_count} "
            f"next_iteration={start_iteration}"
        )
        if checkpoint.get("kind") == "emergency":
            if replay_parent:
                print(
                    "  Emergency checkpoint is lightweight; replay buffer was restored "
                    f"from its last full checkpoint: {replay_parent}"
                )
            else:
                print(
                    "  WARNING: emergency checkpoint had no recoverable full replay buffer; "
                    "training will continue with an empty replay buffer."
                )

    os.makedirs(cfg.log_dir, exist_ok=True)
    train_log, train_writer = _open_csv(
        os.path.join(cfg.log_dir, "train_log.csv"), TRAIN_LOG_HEADER, append=resume_requested
    )
    eval_log, eval_writer = _open_csv(
        os.path.join(cfg.log_dir, "eval_log.csv"), EVAL_LOG_HEADER, append=resume_requested
    )

    runtime = {
        "iteration": last_completed_iteration,
        "next_iteration": start_iteration,
        "phase": "idle",
    }
    handling_signal = False

    def build_training_state() -> dict[str, Any]:
        return {
            "train_tasks": list(train_tasks),
            "task_latents": {
                int(k): v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in task_latents.items()
            },
            "history": history,
            "best_score": best_score,
            "env_rng_states": _capture_env_rng_states(task_envs, pending_env_rng_states),
            "runtime_phase": runtime["phase"],
        }

    def handle_termination(signum, _frame):
        nonlocal handling_signal
        if handling_signal:
            return
        handling_signal = True
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name}; writing emergency checkpoint...", flush=True)
        try:
            path = checkpoint_manager.emergency_save(
                agent,
                iteration=int(runtime["iteration"]),
                next_iteration=int(runtime["next_iteration"]),
                episode=episode_count,
                cfg=cfg,
                rng=rng,
                training_state=build_training_state(),
            )
            print(f"Emergency checkpoint saved: {path}", flush=True)
        except Exception as exc:
            print(f"Emergency checkpoint FAILED: {exc}", flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)

    t0 = time.time()
    last_standard_episode = episode_count

    try:
        for it in range(start_iteration, cfg.total_iterations):
            runtime["next_iteration"] = it
            runtime["phase"] = "rollout"
            ep_return = float("nan")
            ep_disc_return = float("nan")

            # ---- collect rollouts across a batch of tasks ----
            for _ in range(cfg.rollout_episodes_per_iter):
                tid = int(rng.integers(0, cfg.n_train_tasks))
                if tid not in task_envs:
                    task_envs[tid] = make_env(cfg.task_family)
                    task_envs[tid].set_task(train_tasks[tid])
                    _restore_env_rng_state(task_envs[tid], pending_env_rng_states.pop(tid, None))
                env = task_envs[tid]

                ctx = get_context_batch(buffer, tid, cfg.context_size, rng)
                if ctx is not None:
                    with torch.no_grad():
                        z, _, _ = agent.encode(ctx, sample=True)
                        z = z.squeeze(0)
                else:
                    z = sample_prior_z(cfg.latent_dim, device)
                task_latents[tid] = z

                transitions, ep_return, ep_disc_return = rollout_episode(env, agent, z, cfg, rng)
                for transition in transitions:
                    buffer.get(tid).add(*transition)
                episode_count += 1

            # ---- gradient updates ----
            runtime["phase"] = "update"
            ready = buffer.ready_tasks(cfg.min_buffer_before_update)
            stats = None
            if len(ready) >= 2:
                for _ in range(cfg.updates_per_iter):
                    batch_tids = list(rng.choice(
                        ready, size=min(cfg.meta_batch_size, len(ready)), replace=False
                    ))
                    task_batches, task_contexts, task_context_next = {}, {}, {}
                    for tid in batch_tids:
                        task_batches[tid] = buffer.get(tid).sample(cfg.batch_size, rng)
                        ctx = get_context_batch(buffer, tid, cfg.context_size, rng)
                        task_contexts[tid] = ctx[0]
                        ctx2 = get_context_batch(buffer, tid, cfg.context_size, rng)
                        task_context_next[tid] = ctx2[0]
                    stats = agent.update(task_batches, task_contexts, task_context_next)

            if stats is not None and it % 10 == 0:
                row = [it, episode_count, "-", ep_return, ep_disc_return, *[stats[k] for k in STAT_KEYS]]
                train_writer.writerow(row)
                train_log.flush()
                history["train"].append({
                    "iteration": it,
                    "episode": episode_count,
                    "ep_return": float(ep_return),
                    "ep_disc_return": float(ep_disc_return),
                    **{k: float(stats[k]) for k in STAT_KEYS},
                })
                elapsed = time.time() - t0
                print(
                    f"[it {it:5d}] ep {episode_count:5d} ep_return {ep_return:8.2f} "
                    f"critic {stats['real_critic_loss']:.3f} virt {stats['virtual_critic_loss']:.3f} "
                    f"actor {stats['actor_loss']:.3f} alpha {stats['alpha']:.3f} ({elapsed:.1f}s)"
                )

            # ---- evaluation / best-model tracking ----
            runtime["phase"] = "evaluation"
            if it % cfg.eval_every == 0 and it > 0:
                results = evaluate_agent(agent, cfg.task_family, cfg, device, rng, it, eval_writer)
                eval_log.flush()
                for bin_name, mean_ret in results.items():
                    history["eval"].append({
                        "iteration": it,
                        "bin": bin_name,
                        "mean_discounted_return": float(mean_ret),
                    })

                # Proposal-facing model selection: aggregate OOD performance only,
                # excluding the in-distribution "train" bin.
                ood_score = float(np.mean([results[name] for name in ("near", "mid", "far")]))
                if best_score is None or ood_score > best_score:
                    best_score = ood_score
                    best_path = checkpoint_manager.save_best(
                        agent,
                        score=best_score,
                        iteration=it,
                        episode=episode_count,
                        cfg=cfg,
                    )
                    print(f"  new best OOD score {best_score:.4f} -> {best_path}")
                print(f"  eval @ it {it}: {results}")

            # Mark iteration complete only after rollout, updates, evaluation, and
            # best-model bookkeeping have all succeeded.
            last_completed_iteration = it
            runtime["iteration"] = it
            runtime["next_iteration"] = it + 1
            runtime["phase"] = "checkpoint"

            # Save full checkpoints at iteration boundaries. This avoids the old
            # mid-iteration ambiguity where a checkpoint could skip or duplicate an
            # update on resume.
            if episode_count - last_standard_episode >= cfg.checkpoint_every_episodes:
                path = checkpoint_manager.save(
                    agent,
                    iteration=it,
                    next_iteration=it + 1,
                    episode=episode_count,
                    cfg=cfg,
                    rng=rng,
                    training_state=build_training_state(),
                    replay_buffer=buffer,
                )
                last_standard_episode = episode_count
                print(f"  checkpoint -> {path}")

            runtime["phase"] = "idle"

        # Always leave a full final checkpoint, even if the periodic cadence did not
        # land exactly on the last iteration.
        final_iteration = max(last_completed_iteration, start_iteration - 1)
        final_path = checkpoint_manager.save(
            agent,
            iteration=final_iteration,
            next_iteration=max(final_iteration + 1, start_iteration),
            episode=episode_count,
            cfg=cfg,
            rng=rng,
            training_state=build_training_state(),
            replay_buffer=buffer,
            final=True,
        )
        print(f"Final checkpoint: {final_path}")
        print("Training complete.")

    finally:
        train_log.close()
        eval_log.close()
        for env in task_envs.values():
            env.close()


if __name__ == "__main__":
    main()
