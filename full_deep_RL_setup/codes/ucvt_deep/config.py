from dataclasses import dataclass, field
import os


def _runtime_dir(name: str) -> str:
    """Use Kaggle working storage when available; otherwise stay project-local."""
    kaggle_working = "/kaggle/working"
    if os.path.isdir(kaggle_working):
        return os.path.join(kaggle_working, name)
    return os.path.join(".", name)


@dataclass
class Config:
    # --- task setup ---
    task_family: str = "cheetah_vel"     # "cheetah_vel" or "walker_mass"
    n_train_tasks: int = 20
    seed: int = 0

    # --- network sizes ---
    latent_dim: int = 64
    encoder_hidden: tuple = (256, 256)
    decoder_hidden: tuple = (256, 256)
    actor_hidden: tuple = (256, 256)
    critic_hidden: tuple = (256, 256)
    ensemble_size: int = 5

    # --- RL ---
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    grad_clip: float = 10.0

    # --- learning rates ---
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    encoder_lr: float = 3e-4
    decoder_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # --- UC-VT specific ---
    lambda_s: float = 1.0                  # weight of next-state variance in uncertainty score
    beta: float = 16.0                     # soft-weighting temperature
    uncertainty_quantile: float = 0.65     # running c_t percentile
    uncertainty_history_len: int = 200
    lambda_v: float = 0.5                  # virtual loss weight in total critic loss
    lambda_task: float = 1.0               # task-preserving (bisimulation) loss weight
    lambda_reg: float = 0.1                # state-regularization loss weight
    kl_weight: float = 1e-3
    virtual_interp_range: tuple = (-0.5, 1.5)   # <0 or >1 = extrapolation beyond the two tasks
    n_virtual_per_step: int = 4
    virtual_batch_size: int = 64
    bisim_probe_size: int = 128
    state_reg_batch: int = 32

    # --- meta-training loop ---
    meta_batch_size: int = 8               # tasks per gradient step
    context_size: int = 64                 # transitions used to encode z
    batch_size: int = 128                  # real transitions per task per grad step
    episode_len: int = 200
    rollout_episodes_per_iter: int = 4     # env episodes collected before each update batch
    updates_per_iter: int = 4
    total_iterations: int = 2000
    min_buffer_before_update: int = 256

    # --- evaluation ---
    eval_every: int = 50
    eval_episodes_per_bin: int = 3
    eval_context_episodes: int = 2         # exploration episodes to build context for OOD adaptation

    # --- infra ---
    use_amp: bool = True
    resume: bool = False                   # can be overridden by train.py --resume
    checkpoint_every_episodes: int = 10
    max_checkpoints_to_keep: int = 3       # standard checkpoints only; best_model.pth is never rotated
    checkpoint_dir: str = field(default_factory=lambda: _runtime_dir("checkpoints"))
    log_dir: str = field(default_factory=lambda: _runtime_dir("logs"))
