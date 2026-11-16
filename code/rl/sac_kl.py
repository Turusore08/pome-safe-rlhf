"""
Soft Actor-Critic with KL Penalty + Undiscounted Failure Critic Ensemble (Safe Offline-to-Online RLHF).

Combines:
1. Off-policy REDQ / SAC-KL actor-critic updates (Chen & Luo, 2026) anchored to pi_sft reference
   with adaptive KL coefficient beta and optional Q_r initialization from Cal-QL.
2. Undiscounted Safety / Failure Critic Ensemble C_{psi_m}(s, a) estimating within-episode
   violation probability via Bharadhwaj et al. (ICLR 2021) / Thananjeyan et al. (2021) Bellman backup:
       y_{C, t} = c_t + (1 - c_t) * (1 - d_t) * max_{m} C_{bar_psi_m}(s_{t+1}, a')
   trained with Binary Cross-Entropy loss and target Polyak updates.
3. Lagrangian dual penalty integration during actor optimization using multiplier lambda_dual
   updated end-of-episode by PrimalDualSafetyController.
"""

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.critics import FailureCriticEnsemble, RewardCritic
from models.policy_sft import HurdlePolicy


class SACKLAgent(nn.Module):
    """Soft Actor-Critic with SFT-anchored KL penalty and Failure Critic Ensemble."""

    def __init__(
        self,
        state_dim: int = 224,
        action_dim: int = 5,
        pi_sft: Optional[HurdlePolicy] = None,
        pretrained_q_critic: Optional[RewardCritic] = None,
        failure_ensemble: Optional[FailureCriticEnsemble] = None,
        num_q: int = 4,
        ensemble_size_m: int = 4,
        gamma: float = 0.99,
        tau: float = 0.005,
        kl_beta_init: float = 1.0,
        target_kl: float = 0.5,
        adaptive_kl: bool = True,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        failure_lr: float = 1e-3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.num_q = num_q
        self.kl_beta = kl_beta_init
        self.target_kl = target_kl
        self.adaptive_kl = adaptive_kl
        self.has_sft = pi_sft is not None
        self.device = torch.device("cpu")

        # Active policy pi_theta warm-started from SFT
        self.actor = HurdlePolicy(state_dim=state_dim, action_dim=action_dim)
        if pi_sft is not None:
            self.pi_sft = copy.deepcopy(pi_sft)
            self.actor.load_state_dict(pi_sft.state_dict())
        else:
            self.pi_sft = HurdlePolicy(state_dim=state_dim, action_dim=action_dim)

        for p in self.pi_sft.parameters():
            p.requires_grad = False

        # Reward Critic ensemble warm-started from Cal-QL when provided
        self.critics = nn.ModuleList([
            copy.deepcopy(pretrained_q_critic)
            if pretrained_q_critic is not None
            else RewardCritic(state_dim=state_dim, action_dim=action_dim)
            for _ in range(num_q)
        ])
        self.target_critics = copy.deepcopy(self.critics)
        for target_net in self.target_critics:
            for p in target_net.parameters():
                p.requires_grad = False

        # Failure Critic Ensemble + Polyak target network
        if failure_ensemble is not None:
            self.failure_ensemble = failure_ensemble
        else:
            self.failure_ensemble = FailureCriticEnsemble(
                state_dim=state_dim,
                action_dim=action_dim,
                ensemble_size=ensemble_size_m,
            )
        self.target_failure_ensemble = copy.deepcopy(self.failure_ensemble)
        for p in self.target_failure_ensemble.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic_optimizers = [
            torch.optim.AdamW(c.parameters(), lr=critic_lr, weight_decay=1e-4)
            for c in self.critics
        ]
        self.failure_optimizer = torch.optim.AdamW(
            self.failure_ensemble.parameters(), lr=failure_lr, weight_decay=1e-4
        )

    def to(self, device: torch.device):
        """Migrate actor, reference, reward critics, and failure ensemble to target device."""
        self.device = device
        self.actor.to(device)
        self.pi_sft.to(device)
        self.critics.to(device)
        self.target_critics.to(device)
        self.failure_ensemble.to(device)
        self.target_failure_ensemble.to(device)
        return self

    def soft_update(self):
        """Polyak soft update for both reward critics and failure critic ensemble."""
        for net, target_net in zip(self.critics, self.target_critics):
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        for param, target_param in zip(
            self.failure_ensemble.parameters(), self.target_failure_ensemble.parameters()
        ):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Update REDQ Q_r ensemble with stabilized KL-regularized soft Bellman target."""
        with torch.no_grad():
            next_actions, _ = self.actor.sample(next_states)
            if self.has_sft:
                kl_next = self.actor.kl_divergence(self.pi_sft, next_states)
                reg_term = -torch.clamp(self.kl_beta * kl_next, min=0.0, max=2.0)
            else:
                reg_term = 0.05 * torch.clamp(self.actor.entropy(next_states), max=10.0)

            subset_indices = np.random.choice(self.num_q, size=min(2, self.num_q), replace=False)
            q_targets = torch.stack(
                [self.target_critics[i](next_states, next_actions) for i in subset_indices],
                dim=-1,
            )
            min_q_target = q_targets.min(dim=-1).values
            y_t = rewards + (self.gamma * (1.0 - dones) * (min_q_target + reg_term))

        total_critic_loss = 0.0
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            current_q = critic(states, actions)
            loss = F.smooth_l1_loss(current_q, y_t, beta=5.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            total_critic_loss += float(loss.item())

        return total_critic_loss / max(1, self.num_q)

    def update_failure_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        violations: torch.Tensor,
    ) -> float:
        """
        Update FailureCriticEnsemble with discounted within-episode Bellman backup:
            y_{C, t} = c_t + (1 - c_t) * (1 - d_t) * 0.96 * C_hat_{bar_psi}(s_{t+1}, a')
        trained via Binary Cross-Entropy across all M ensemble members.
        """
        with torch.no_grad():
            next_actions, _ = self.actor.sample(next_states)
            next_conservative_risk, _ = self.target_failure_ensemble(next_states, next_actions)
            c_t = torch.clamp(violations.float(), 0.0, 1.0)
            d_t = torch.clamp(dones.float(), 0.0, 1.0)
            y_c = torch.clamp(
                c_t + (1.0 - c_t) * (1.0 - d_t) * 0.96 * next_conservative_risk,
                1e-6,
                1.0 - 1e-6,
            )

        _, all_risks = self.failure_ensemble(states, actions)
        all_risks_clamped = torch.clamp(all_risks.float(), 1e-6, 1.0 - 1e-6)
        targets_expanded = y_c.unsqueeze(-1).expand_as(all_risks_clamped)
        bce_loss = F.binary_cross_entropy(all_risks_clamped, targets_expanded)

        self.failure_optimizer.zero_grad()
        bce_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.failure_ensemble.parameters(), 1.0)
        self.failure_optimizer.step()
        return float(bce_loss.item())

    def update_actor(
        self,
        states: torch.Tensor,
        lambda_dual: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Actor update minimizing analytical Cragg-Hurdle KL divergence to pi_sft and
        Lagrangian failure risk while maximizing Q_r:
            L_pi(theta) = E [ beta * KL(pi || pi_sft) - mean Q_r(s, a) + lambda * hat_C(s, a) ]
        """
        actions, _ = self.actor.sample(states)
        if self.has_sft:
            kl_div = self.actor.kl_divergence(self.pi_sft, states).mean()
            kl_loss_term = self.kl_beta * kl_div
        else:
            ent = self.actor.entropy(states).mean()
            kl_div = -ent
            kl_loss_term = -0.05 * ent

        q_values = torch.stack([c(states, actions) for c in self.critics], dim=-1)
        mean_q = q_values.mean(dim=-1).mean()

        if lambda_dual > 0.0:
            conservative_risk, _ = self.failure_ensemble(states, actions)
            risk_penalty = float(lambda_dual) * conservative_risk.mean()
        else:
            risk_penalty = torch.tensor(0.0, device=states.device)

        actor_loss = kl_loss_term - mean_q + risk_penalty

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        if self.adaptive_kl and self.has_sft:
            self._adapt_beta(float(kl_div.item()))

        return float(actor_loss.item()), float(kl_div.item())

    def _adapt_beta(self, current_kl: float):
        """Proportional adaptive controller for KL penalty coefficient beta."""
        err = float(np.clip((current_kl / max(1e-4, self.target_kl)) - 1.0, -0.2, 0.2))
        k_beta = 0.05
        self.kl_beta = float(np.clip(self.kl_beta * (1.0 + k_beta * err), 0.05, 5.0))

    @property
    def primary_reward_critic(self) -> RewardCritic:
        """Expose primary Q_r critic for RiskAwareSelector deployment."""
        return self.critics[0]


def warmstart_failure_ensemble(
    failure_ensemble: FailureCriticEnsemble,
    dataset,
    policy: Optional[HurdlePolicy] = None,
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 1e-3,
    gamma_failure: float = 0.98,
    tau: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Offline calibration of FailureCriticEnsemble using exact episodic backward failure
    return-to-go targets Y_{C,t} = max(c_t, (1 - d_t) * gamma_failure * Y_{C,t+1}).
    Avoids the degenerate C(s, a) = 0.50 fixed point of undiscounted bootstrap warmup on
    long (300+ step) zero-violation trajectories.
    """
    if len(dataset) == 0:
        return 0.0
    failure_ensemble.to(device).train()
    opt = torch.optim.AdamW(failure_ensemble.parameters(), lr=lr, weight_decay=1e-4)
    st_all = torch.as_tensor(dataset.states, dtype=torch.float32, device=device)
    ac_all = torch.as_tensor(dataset.actions, dtype=torch.float32, device=device)

    vl_np = np.asarray(dataset.violations, dtype=np.float32)
    dn_np = np.asarray(dataset.dones, dtype=np.float32)
    y_mc_np = np.zeros_like(vl_np, dtype=np.float32)
    running = 0.0
    for i in range(len(vl_np) - 1, -1, -1):
        c_i = float(np.clip(vl_np[i], 0.0, 1.0))
        if float(dn_np[i]) > 0.5:
            running = c_i
        else:
            running = max(c_i, gamma_failure * running)
        y_mc_np[i] = float(np.clip(running, 1e-6, 1.0 - 1e-6))

    y_all = torch.as_tensor(y_mc_np, dtype=torch.float32, device=device)

    n = len(st_all)
    last_loss = 0.0
    for _ in range(max(1, epochs)):
        perm = torch.randperm(n, device=device)
        ep_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            bs, ba, by = st_all[idx], ac_all[idx], y_all[idx]
            _, all_risks = failure_ensemble(bs, ba)
            probs = torch.clamp(all_risks, 1e-6, 1.0 - 1e-6)
            loss = F.binary_cross_entropy(probs, by.unsqueeze(-1).expand_as(probs))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(failure_ensemble.parameters(), 1.0)
            opt.step()
            ep_losses.append(float(loss.item()))
        last_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
    failure_ensemble.eval()
    return last_loss
