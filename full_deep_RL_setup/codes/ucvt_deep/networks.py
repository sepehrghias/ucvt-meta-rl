"""Networks for UC-VT: task encoder, ensemble decoder, SAC actor, twin critics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(in_dim: int, hidden: list[int], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class TaskEncoder(nn.Module):
    """Permutation-invariant context encoder q(z | c).

    Context c is a set of (s, a, r, s') transitions from the current task.
    Each transition is embedded by a shared MLP [256, 256], embeddings are
    mean-pooled (order-invariant, as in PEARL), then a linear head produces
    (mu, logvar) for a diagonal Gaussian latent z (reparameterization trick).
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64, hidden=(256, 256)):
        super().__init__()
        in_dim = 2 * obs_dim + action_dim + 1  # s, a, r, s'
        self.latent_dim = latent_dim
        self.trunk = mlp(in_dim, list(hidden), hidden[-1])
        self.head = nn.Linear(hidden[-1], 2 * latent_dim)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """context: [batch, n_transitions, in_dim] -> mu, logvar: [batch, latent_dim]."""
        h = self.trunk(context)                # [batch, n_transitions, hidden]
        h = h.mean(dim=1)                       # permutation-invariant aggregation
        out = self.head(h)
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -8.0, 8.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


class TaskDecoder(nn.Module):
    """Single decoder D(s, a, z) -> (r_hat, delta_s_hat)."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden=(256, 256)):
        super().__init__()
        in_dim = obs_dim + action_dim + latent_dim
        self.body = mlp(in_dim, list(hidden), hidden[-1])
        self.reward_head = nn.Linear(hidden[-1], 1)
        self.delta_state_head = nn.Linear(hidden[-1], obs_dim)

    def forward(self, s, a, z):
        h = self.body(torch.cat([s, a, z], dim=-1))
        r_hat = self.reward_head(h).squeeze(-1)
        delta_s_hat = self.delta_state_head(h)
        return r_hat, delta_s_hat


class EnsembleTaskDecoder(nn.Module):
    """K independently initialized task decoders (bootstrap ensemble).

    forward() returns per-member predictions stacked on a new leading dim,
    plus the ensemble mean and the epistemic-uncertainty score `u` combining
    reward and next-state disagreement, as specified for UC-VT:
        u = Var_k(r_hat_k) + lambda_s * mean_dims Var_k(delta_s_hat_k)
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, k: int = 5,
                 hidden=(256, 256), lambda_s: float = 1.0):
        super().__init__()
        self.k = k
        self.lambda_s = lambda_s
        self.members = nn.ModuleList(
            [TaskDecoder(obs_dim, action_dim, latent_dim, hidden) for _ in range(k)]
        )

    def forward(self, s, a, z):
        r_preds, ds_preds = [], []
        for m in self.members:
            r_hat, ds_hat = m(s, a, z)
            r_preds.append(r_hat)
            ds_preds.append(ds_hat)
        r_preds = torch.stack(r_preds, dim=0)     # [K, batch]
        ds_preds = torch.stack(ds_preds, dim=0)   # [K, batch, obs_dim]

        r_mean = r_preds.mean(dim=0)
        ds_mean = ds_preds.mean(dim=0)
        r_var = r_preds.var(dim=0, unbiased=False)
        ds_var = ds_preds.var(dim=0, unbiased=False).mean(dim=-1)
        uncertainty = r_var + self.lambda_s * ds_var
        return {
            "r_mean": r_mean, "ds_mean": ds_mean,
            "r_preds": r_preds, "ds_preds": ds_preds,
            "uncertainty": uncertainty,
        }


class TanhGaussianActor(nn.Module):
    """SAC policy pi(a | s, z), squashed Gaussian with tanh reparameterization."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden=(256, 256)):
        super().__init__()
        in_dim = obs_dim + latent_dim
        self.trunk = mlp(in_dim, list(hidden), hidden[-1])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)

    def forward(self, s, z):
        h = self.trunk(torch.cat([s, z], dim=-1))
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, s, z):
        mu, log_std = self.forward(s, z)
        std = log_std.exp()
        dist = Normal(mu, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        # Clamp BEFORE log: float32 rounding can push (1 - tanh(x)^2) fractionally
        # negative when actions saturate near +/-1, which would otherwise NaN.
        log_det_jacobian = torch.log(torch.clamp(1 - action.pow(2), min=1e-6))
        log_prob = dist.log_prob(pre_tanh) - log_det_jacobian
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class QCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden=(256, 256)):
        super().__init__()
        in_dim = obs_dim + action_dim + latent_dim
        self.net = mlp(in_dim, list(hidden), 1)

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1)).squeeze(-1)


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = QCritic(obs_dim, action_dim, latent_dim, hidden)
        self.q2 = QCritic(obs_dim, action_dim, latent_dim, hidden)

    def forward(self, s, a, z):
        return self.q1(s, a, z), self.q2(s, a, z)
