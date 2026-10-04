"""UC-VT / TAVT-SAC agent: encoder + ensemble decoder + actor + twin critics."""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from networks import TaskEncoder, EnsembleTaskDecoder, TanhGaussianActor, TwinQCritic
from losses import (
    sac_critic_loss, sac_actor_loss, task_preserving_loss,
    state_regularization_loss, ucvt_confidence_weights,
)


class UCVTAgent:
    def __init__(self, obs_dim, action_dim, cfg, device):
        self.cfg = cfg
        self.device = device
        L = cfg.latent_dim

        self.encoder = TaskEncoder(obs_dim, action_dim, L, cfg.encoder_hidden).to(device)
        self.decoder = EnsembleTaskDecoder(
            obs_dim, action_dim, L, k=cfg.ensemble_size,
            hidden=cfg.decoder_hidden, lambda_s=cfg.lambda_s,
        ).to(device)
        self.actor = TanhGaussianActor(obs_dim, action_dim, L, cfg.actor_hidden).to(device)
        self.critic = TwinQCritic(obs_dim, action_dim, L, cfg.critic_hidden).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), requires_grad=True, device=device)
        self.target_entropy = -float(action_dim)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.encoder_lr)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=cfg.decoder_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.scaler = torch.amp.GradScaler(enabled=cfg.use_amp)
        self.uncertainty_history = deque(maxlen=cfg.uncertainty_history_len)

        self.train_task_z_cache: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def encode(self, context_np, sample: bool = True):
        """context_np: [batch, n_transitions, in_dim] numpy -> z, mu, logvar (torch)."""
        context = torch.as_tensor(context_np, dtype=torch.float32, device=self.device)
        mu, logvar = self.encoder(context)
        z = self.encoder.reparameterize(mu, logvar) if sample else mu
        return z, mu, logvar

    def act(self, obs_np: np.ndarray, z: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = z.reshape(1, -1)
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(obs, z)
            else:
                action, _, _ = self.actor.sample(obs, z)
        return action.squeeze(0).cpu().numpy()

    def soft_update_target(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    # ------------------------------------------------------------------ #

    def update(self, task_batches: dict, task_contexts: dict, task_context_next: dict) -> dict:
        """One gradient step across a meta-batch of real training tasks.

        task_batches[tid]       = (s, a, r, sp, done) numpy arrays, real transitions
        task_contexts[tid]      = [n_ctx, in_dim] numpy, context for encoding z (this task)
        task_context_next[tid]  = same, a *different* context sample (for SAC's target z,
                                   decorrelated from the current-step context per PEARL convention)
        """
        cfg, device = self.cfg, self.device
        task_ids = list(task_batches.keys())
        n_tasks = len(task_ids)
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp, dtype=amp_dtype):
            # ---- encode task latents (gradients flow into encoder via critic loss) ----
            z_list, mu_list, logvar_list = [], [], []
            for tid in task_ids:
                ctx = torch.as_tensor(task_contexts[tid], dtype=torch.float32, device=device).unsqueeze(0)
                mu, logvar = self.encoder(ctx)
                z = self.encoder.reparameterize(mu, logvar)
                z_list.append(z.squeeze(0))
                mu_list.append(mu.squeeze(0))
                logvar_list.append(logvar.squeeze(0))
            z_batch = torch.stack(z_list, dim=0)         # [n_tasks, L]
            mu_batch = torch.stack(mu_list, dim=0)
            logvar_batch = torch.stack(logvar_list, dim=0)
            kl_loss = self.encoder.kl_to_standard_normal(mu_batch, logvar_batch)

            # ---- real-data critic + decoder + task-preserving losses, per task ----
            real_critic_loss = 0.0
            decoder_sup_loss = 0.0
            all_s, all_a, all_r, all_sp = [], [], [], []
            for i, tid in enumerate(task_ids):
                s, a, r, sp, done = task_batches[tid]
                s = torch.as_tensor(s, dtype=torch.float32, device=device)
                a = torch.as_tensor(a, dtype=torch.float32, device=device)
                r = torch.as_tensor(r, dtype=torch.float32, device=device)
                sp = torch.as_tensor(sp, dtype=torch.float32, device=device)
                done = torch.as_tensor(done, dtype=torch.float32, device=device)
                z_i = z_batch[i].unsqueeze(0).expand(s.shape[0], -1)

                c_loss, q1, q2 = sac_critic_loss(
                    self.critic, self.target_critic, self.actor,
                    s, a, r, sp, done, z_i, z_i, cfg.gamma, self.alpha.detach(),
                )
                real_critic_loss = real_critic_loss + c_loss.mean()

                out = self.decoder(s, a, z_i.detach())
                decoder_sup_loss = decoder_sup_loss + (
                    (out["r_mean"] - r).pow(2).mean() + (out["ds_mean"] - (sp - s)).pow(2).mean(-1).mean()
                )
                all_s.append(s); all_a.append(a); all_r.append(r); all_sp.append(sp)
            real_critic_loss = real_critic_loss / n_tasks
            decoder_sup_loss = decoder_sup_loss / n_tasks

            # ---- task-preserving loss: evaluate every task's decoder on a SHARED probe batch ----
            probe_s = torch.cat(all_s, dim=0)
            probe_a = torch.cat(all_a, dim=0)
            probe_idx = torch.randperm(probe_s.shape[0], device=device)[: cfg.bisim_probe_size]
            probe_s, probe_a = probe_s[probe_idx], probe_a[probe_idx]

            r_hat_per_task, ds_hat_per_task = [], []
            for i in range(n_tasks):
                z_i = z_batch[i].unsqueeze(0).expand(probe_s.shape[0], -1)
                out = self.decoder(probe_s, probe_a, z_i)
                r_hat_per_task.append(out["r_mean"])
                ds_hat_per_task.append(out["ds_mean"])
            r_hat_per_task = torch.stack(r_hat_per_task, dim=0)
            ds_hat_per_task = torch.stack(ds_hat_per_task, dim=0)
            tp_loss = task_preserving_loss(z_batch, r_hat_per_task, ds_hat_per_task, cfg.gamma)

            # ---- virtual transitions: interpolate/extrapolate task latents ----
            virt_loss_terms = []
            uncertainties_all = []
            if n_tasks >= 2 and cfg.lambda_v > 0:
                for _ in range(cfg.n_virtual_per_step):
                    i, j = np.random.choice(n_tasks, size=2, replace=False)
                    alpha_mix = float(np.random.uniform(*cfg.virtual_interp_range))
                    z_virtual = (alpha_mix * z_batch[i] + (1 - alpha_mix) * z_batch[j]).unsqueeze(0)

                    idx = torch.randperm(probe_s.shape[0], device=device)[: cfg.virtual_batch_size]
                    s_v, a_v = probe_s[idx], probe_a[idx]
                    z_v = z_virtual.expand(s_v.shape[0], -1)

                    out_v = self.decoder(s_v, a_v, z_v)
                    r_hat_v = out_v["r_mean"]
                    sp_hat_v = s_v + out_v["ds_mean"]
                    u_v = out_v["uncertainty"]
                    uncertainties_all.append(u_v.detach())

                    running_c_t = (
                        float(np.quantile(torch.cat(list(self.uncertainty_history)).cpu().numpy(),
                                           cfg.uncertainty_quantile))
                        if len(self.uncertainty_history) > 0 else float(u_v.mean().item()) + 1e-6
                    )
                    w = ucvt_confidence_weights(u_v, running_c_t, cfg.beta)

                    v_critic_loss, _, _ = sac_critic_loss(
                        self.critic, self.target_critic, self.actor,
                        s_v, a_v, r_hat_v.detach(), sp_hat_v.detach(),
                        torch.zeros_like(r_hat_v), z_v, z_v, cfg.gamma, self.alpha.detach(),
                    )
                    virt_loss_terms.append((w.detach() * v_critic_loss).mean())

                if uncertainties_all:
                    self.uncertainty_history.append(torch.cat(uncertainties_all))

            virtual_critic_loss = torch.stack(virt_loss_terms).mean() if virt_loss_terms else torch.zeros((), device=device)

            state_reg_loss = state_regularization_loss(
                self.decoder, probe_s[: cfg.state_reg_batch], probe_a[: cfg.state_reg_batch],
                z_batch[torch.randint(0, n_tasks, (min(cfg.state_reg_batch, probe_s.shape[0]),), device=device)],
            )

            critic_total = real_critic_loss + cfg.lambda_v * virtual_critic_loss
            # Single combined loss: critic_total and state_reg_loss both depend on the
            # encoder's z_batch graph, and decoder_sup_loss/tp_loss depend on the decoder's
            # graph, so critic/encoder/decoder must share ONE backward() call rather than
            # being split (splitting frees shared graph nodes after the first backward).
            total_loss = (
                critic_total
                + cfg.kl_weight * kl_loss
                + cfg.lambda_task * tp_loss
                + decoder_sup_loss
                + cfg.lambda_reg * state_reg_loss
            )

        # ---- critic + encoder + decoder step (one combined backward) ----
        self.critic_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        self.decoder_opt.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.critic_opt)
        self.scaler.unscale_(self.encoder_opt)
        self.scaler.unscale_(self.decoder_opt)
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), cfg.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), cfg.grad_clip)
        self.scaler.step(self.critic_opt)
        self.scaler.step(self.encoder_opt)
        self.scaler.step(self.decoder_opt)
        self.scaler.update()

        # ---- actor step (detached z, real data only) ----
        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp, dtype=amp_dtype):
            actor_loss_total = 0.0
            log_probs_all = []
            for i, tid in enumerate(task_ids):
                s = torch.as_tensor(task_batches[tid][0], dtype=torch.float32, device=device)
                z_i = z_batch[i].detach().unsqueeze(0).expand(s.shape[0], -1)
                a_loss, log_prob = sac_actor_loss(self.critic, self.actor, s, z_i, self.alpha.detach())
                actor_loss_total = actor_loss_total + a_loss
                log_probs_all.append(log_prob.detach())
            actor_loss_total = actor_loss_total / n_tasks

        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss_total).backward()
        self.scaler.unscale_(self.actor_opt)
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.scaler.step(self.actor_opt)
        self.scaler.update()

        # ---- alpha (entropy temperature) step ----
        log_probs_cat = torch.cat(log_probs_all, dim=0)
        alpha_loss = -(self.log_alpha * (log_probs_cat + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update_target()

        return {
            "real_critic_loss": float(real_critic_loss.detach().cpu()),
            "virtual_critic_loss": float(virtual_critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss_total.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "task_preserving_loss": float(tp_loss.detach().cpu()),
            "state_reg_loss": float(state_reg_loss.detach().cpu()),
            "decoder_sup_loss": float(decoder_sup_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }

    # ------------------------------------------------------------------ #

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "encoder_opt": self.encoder_opt.state_dict(),
            "decoder_opt": self.decoder_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, sd):
        self.encoder.load_state_dict(sd["encoder"])
        self.decoder.load_state_dict(sd["decoder"])
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.target_critic.load_state_dict(sd["target_critic"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.encoder_opt.load_state_dict(sd["encoder_opt"])
        self.decoder_opt.load_state_dict(sd["decoder_opt"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
