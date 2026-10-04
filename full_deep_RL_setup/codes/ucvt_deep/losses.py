"""Loss functions for SAC + TAVT-style task-preserving/state-reg losses + UC-VT weighting.

The task-preserving (bisimulation) loss below mirrors the official TAVT
implementation's `compute_bisim_loss` (rlkit/torch/sac/sac_revised.py):
for a permuted pairing of tasks in a meta-batch,

    r_dist          = || r_hat(task_i) - r_hat(task_j) ||_2
    transition_dist = || s'_hat(task_i) - s'_hat(task_j) ||_2
    sample_dist     = r_dist + gamma * transition_dist
    z_dist          = || z_i - z_j ||_1
    L_task          = MSE(z_dist, sample_dist)

i.e. the latent-space distance between two task embeddings is regressed
towards how differently their decoders behave on the same (s, a) inputs.
We omit the official code's WGAN discriminator/generator refinement step
and its auxiliary context-distribution VAE (out of scope here; see README).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sac_critic_loss(critic, target_critic, actor, s, a, r, sp, done, z, z_next, gamma, alpha):
    with torch.no_grad():
        a_next, log_prob_next, _ = actor.sample(sp, z_next)
        q1_t, q2_t = target_critic(sp, a_next, z_next)
        q_t = torch.min(q1_t, q2_t) - alpha * log_prob_next.squeeze(-1)
        target = r + (1.0 - done) * gamma * q_t
    q1, q2 = critic(s, a, z)
    loss = F.mse_loss(q1, target, reduction="none") + F.mse_loss(q2, target, reduction="none")
    return loss, q1, q2


def sac_actor_loss(critic, actor, s, z, alpha):
    a, log_prob, _ = actor.sample(s, z)
    q1, q2 = critic(s, a, z)
    q = torch.min(q1, q2)
    loss = (alpha * log_prob.squeeze(-1) - q)
    return loss.mean(), log_prob


def task_preserving_loss(z_batch: torch.Tensor, r_hat: torch.Tensor, ds_hat: torch.Tensor, gamma: float) -> torch.Tensor:
    """Bisimulation-style task-preserving loss (see module docstring).

    z_batch: [n_tasks, latent_dim] task embeddings (one per task in the meta-batch)
    r_hat, ds_hat: decoder predictions [n_tasks, shared_batch, ...] evaluated on the
        SAME shared (s, a) probe batch for every task, so cross-task differences
        reflect task identity rather than input differences.
    """
    n_tasks = z_batch.shape[0]
    # A pure random permutation can be the identity (self-pairing -> zero distance),
    # and d/dx sqrt(x) diverges at x=0, which would NaN the gradient. A cyclic shift
    # guarantees every task is paired with a DIFFERENT task, and we still add a small
    # epsilon under sqrt() as a second line of defense.
    perm = torch.roll(torch.arange(n_tasks, device=z_batch.device), shifts=1)

    r_dist = torch.sqrt((r_hat - r_hat[perm]).pow(2).mean(dim=1) + 1e-8)
    transition_dist = torch.sqrt((ds_hat - ds_hat[perm]).pow(2).mean(dim=-1).mean(dim=1) + 1e-8)
    sample_dist = r_dist + gamma * transition_dist

    # NOTE: official TAVT uses .sum(dim=-1) here, calibrated for its small latent_dim (~10).
    # We use .mean(dim=-1) so the loss scale doesn't grow with latent_dim (64 by default here).
    z_dist = (z_batch - z_batch[perm]).abs().mean(dim=-1)
    return F.mse_loss(z_dist, sample_dist)


def state_regularization_loss(decoder, s_probe, a_probe, z_probe, eps: float = 0.05) -> torch.Tensor:
    """Lipschitz-style smoothness penalty on the decoder's next-state prediction
    w.r.t. small perturbations of the task latent z. Encourages the virtual-task
    manifold to vary smoothly, which is the practical goal of TAVT's state
    regularization (mitigating overestimation from erratic extrapolated states),
    implemented here without the official code's WGAN machinery (see README).
    """
    with torch.no_grad():
        direction = torch.randn_like(z_probe)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        z_perturbed = z_probe + eps * direction

    out_a = decoder(s_probe, a_probe, z_probe)
    out_b = decoder(s_probe, a_probe, z_perturbed)
    delta_state_change = (out_a["ds_mean"] - out_b["ds_mean"]).pow(2).mean(dim=-1)
    return delta_state_change.mean() / (eps ** 2)


def ucvt_confidence_weights(uncertainty: torch.Tensor, running_c_t: float, beta: float) -> torch.Tensor:
    """Soft weighting rule w(u) = exp(-beta * u / c_t), c_t detached (running stat)."""
    c_t = max(running_c_t, 1e-8)
    return torch.exp(-beta * uncertainty.detach() / c_t)
