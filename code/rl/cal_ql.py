"""
Calibrated Offline Q-Learning (Cal-QL) Pre-training for Reward Critic Q_r.

Implements calibrated conservative value regularization with Log-Sum-Exp backup
(Nakamoto et al., NeurIPS 2023, Eqs. 3-5; Kumar et al., NeurIPS 2020):
    L(Q_r) = L_TD(Q_r) + alpha_cql * E_{s ~ D} [
        logsumexp_{a in C(s)} max(Q_r(s, a), V^mu(s)) - log|C(s)| - Q_r(s, a_data)
    ]

Prevents over-conservatism and value underflow below empirical behavior policy
return V^mu(s), stabilizing offline-to-online handoff.
"""

import copy
import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import ACTION_BOUNDS, ACTION_COLUMNS
from models.critics import RewardCritic

_DEFAULT_ACTION_HIGHS = [float(ACTION_BOUNDS[c][1]) for c in ACTION_COLUMNS]


def compute_cal_ql_loss(
    q_critic: RewardCritic,
    states: torch.Tensor,
    actions: torch.Tensor,
    target_q: torch.Tensor,
    candidate_actions: torch.Tensor,
    v_mu_monte_carlo: torch.Tensor,
    alpha_cql: float = 1.0,
    temperature: float = 1.0,
    use_logsumexp: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Cal-QL loss combining MSE Bellman backup and calibrated Log-Sum-Exp CQL penalty.
    """
    # 1. Temporal Difference (TD) loss
    current_q = q_critic(states, actions)
    td_loss = F.mse_loss(current_q, target_q)

    # 2. Conservative Log-Sum-Exp penalty over OOD candidate actions lower-bounded by V^mu(s)
    batch_size = states.shape[0]
    num_cands = candidate_actions.shape[1]

    expanded_states = states.unsqueeze(1).expand(batch_size, num_cands, states.shape[-1]).reshape(-1, states.shape[-1])
    flat_cands = candidate_actions.reshape(-1, candidate_actions.shape[-1])
    cand_q = q_critic(expanded_states, flat_cands).view(batch_size, num_cands)

    # Calibration lower bound: max(Q_r(s, a), V^mu(s))
    v_mu_expanded = v_mu_monte_carlo.unsqueeze(1).expand(batch_size, num_cands)
    calibrated_q = torch.maximum(cand_q, v_mu_expanded)

    if use_logsumexp:
        tau = max(1e-3, float(temperature))
        cql_ood = tau * (torch.logsumexp(calibrated_q / tau, dim=1) - math.log(max(1, num_cands)))
    else:
        cql_ood = calibrated_q.mean(dim=1)

    cql_loss = alpha_cql * (cql_ood - current_q).mean()
    total_loss = td_loss + cql_loss

    info = {
        "cal_ql_total_loss": float(total_loss.item()),
        "td_loss": float(td_loss.item()),
        "cql_loss": float(cql_loss.item()),
        "q_dataset_mean": float(current_q.mean().item()),
        "q_candidate_mean": float(cand_q.mean().item()),
    }
    return total_loss, info


def compute_empirical_mc_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Compute exact trajectory-wise Monte Carlo return-to-go V^mu(s_t) respecting done flags."""
    v_mu = torch.zeros_like(rewards)
    running_ret = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    for i in reversed(range(len(rewards))):
        if float(dones[i].item()) >= 0.5:
            running_ret = rewards[i]
        else:
            running_ret = rewards[i] + gamma * running_ret
        v_mu[i] = running_ret
    return v_mu


def _sample_calql_candidates(
    states: torch.Tensor,
    action_high: torch.Tensor,
    policy: Optional[nn.Module] = None,
    n_unif: int = 4,
    n_pol: int = 4,
) -> torch.Tensor:
    """Sample diverse candidate actions from Uniform[0, action_high]^D and current policy."""
    batch_size = states.shape[0]
    action_dim = action_high.shape[-1]
    unif = torch.rand(batch_size, n_unif, action_dim, device=states.device) * action_high.view(1, 1, action_dim)

    if policy is not None:
        exp_states = states.unsqueeze(1).expand(batch_size, n_pol, states.shape[-1]).reshape(-1, states.shape[-1])
        with torch.no_grad():
            pol_acts, _ = policy.sample(exp_states, deterministic=False)
        pol_acts = pol_acts.view(batch_size, n_pol, action_dim)
    else:
        pol_acts = torch.rand(batch_size, n_pol, action_dim, device=states.device) * action_high.view(1, 1, action_dim)

    return torch.cat([unif, pol_acts], dim=1)


def pretrain_calibrated_critic(
    q_critic: RewardCritic,
    dataset,
    policy: Optional[nn.Module] = None,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha_cql: float = 1.0,
    device: Optional[torch.device] = None,
) -> RewardCritic:
    """
    Full-epoch offline pre-training for calibrated critic Q_r using target network Q_target
    and calibrated Log-Sum-Exp penalty over uniform + policy action candidates.
    """
    dev = device if device is not None else torch.device("cpu")
    q_critic.to(dev).train()
    q_target = copy.deepcopy(q_critic).to(dev).eval()

    action_high = torch.tensor(_DEFAULT_ACTION_HIGHS[: q_critic.action_dim], dtype=torch.float32, device=dev)
    optimizer = torch.optim.AdamW(q_critic.parameters(), lr=lr, weight_decay=1e-4)

    states = dataset.states.to(dev)
    actions = dataset.actions.to(dev)
    rewards = dataset.rewards.to(dev)
    next_states = dataset.next_states.to(dev)
    dones = dataset.dones.to(dev)

    v_mu = compute_empirical_mc_returns(rewards, dones, gamma=gamma)
    n_samples = len(states)
    print(f"Pre-training Cal-QL Calibrated Critic on {n_samples:,} historical transitions ({epochs} epochs)...")

    last_loss = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=dev)
        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            idxs = perm[start : start + batch_size]
            b_s = states[idxs]
            b_a = actions[idxs]
            b_r = rewards[idxs]
            b_s_next = next_states[idxs]
            b_d = dones[idxs]
            b_vmu = v_mu[idxs]

            cand_curr = _sample_calql_candidates(b_s, action_high, policy=policy, n_unif=4, n_pol=4)
            cand_next = _sample_calql_candidates(b_s_next, action_high, policy=policy, n_unif=4, n_pol=4)

            with torch.no_grad():
                flat_s_next = b_s_next.unsqueeze(1).expand(len(idxs), cand_next.shape[1], b_s_next.shape[-1]).reshape(-1, b_s_next.shape[-1])
                flat_next_cands = cand_next.reshape(-1, cand_next.shape[-1])
                q_next_cands = q_target(flat_s_next, flat_next_cands).view(len(idxs), cand_next.shape[1])
                max_q_next = q_next_cands.max(dim=1).values
                target_q = b_r + gamma * (1.0 - b_d) * max_q_next

            loss, _ = compute_cal_ql_loss(
                q_critic, b_s, b_a, target_q, cand_curr, b_vmu, alpha_cql=alpha_cql, use_logsumexp=True
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_critic.parameters(), 1.0)
            optimizer.step()

            # Soft Polyak target update
            with torch.no_grad():
                for p, p_targ in zip(q_critic.parameters(), q_target.parameters()):
                    p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

            epoch_losses.append(float(loss.item()))

        last_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))

    q_critic.eval()
    print(f"Cal-QL critic pre-training finished (Final epoch loss: {last_loss:.4f})")
    return q_critic
