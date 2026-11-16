"""
Direct Preference-based Policy Optimization (DPPO).
Adapted from An et al. (2023) and Gerold, Brandner, & Lucia (ESCAPE 36 / LAPSE 2026).

Directly optimizes policy parameters theta on preference dataset D_pref
without training an intermediate neural reward function:
1. Aggregated distance d(pi_theta, sigma) between policy predictions and labeled segment actions:
   d(pi_theta, sigma) = (1 / L_s) sum_{t=1}^{L_s} || (pi_theta(s_t) - a_t) / scale ||_2^2
2. Preference likelihood under policy (Equation 6 in LAPSE):
   s_tilde(pi_theta, sigma_1 > sigma_2, lambda_reg) = exp(-d(pi_theta, sigma_1)) / (exp(-d(pi_theta, sigma_1)) + exp(-lambda_reg * d(pi_theta, sigma_2)))
3. Maximum Likelihood Estimation (MLE) Training Objective (Equation 5 in LAPSE):
   min_theta - sum_{(sigma_1, sigma_2, mu) in D} [ mu(1) * log s_tilde(pi_theta, sigma_1 > sigma_2) + mu(2) * log s_tilde(pi_theta, sigma_2 > sigma_1) ]
"""

import os
import sys
import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.policy_sft import HurdlePolicy


def compute_segment_distance(
    policy: HurdlePolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    action_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes normalized squared Euclidean distance between policy actions and segment actions:
    d(pi_theta, sigma) = (1 / L_s) sum_{t=1}^{L_s} || (pi_theta(s_t) - a_t) / scale ||_2^2

    Args:
        policy: HurdlePolicy network
        states: (B, L_s, state_dim) or (L_s, state_dim)
        actions: (B, L_s, action_dim) or (L_s, action_dim)
        action_scale: (action_dim,) normalization bounds
    Returns:
        dist: (B,) scalar distances per segment
    """
    if states.ndim == 2:
        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)

    B, L, S_dim = states.shape
    flat_states = states.view(B * L, S_dim)
    gate_logits, mu, _ = policy(flat_states)

    # Differentiable continuous expected action: p_gate * mu
    gate_probs = torch.sigmoid(gate_logits)
    pred_actions = (gate_probs * mu).view(B, L, -1)

    if action_scale is None:
        action_scale = getattr(policy, "action_high", None)
        if action_scale is None:
            action_scale = torch.tensor([6.0, 5.0, 100.0, 20.0, 5000.0], device=states.device)
        else:
            action_scale = action_scale.to(states.device)

    # Normalized squared error across actions and time steps
    norm_diff = (pred_actions - actions) / (action_scale + 1e-6)
    step_dists = torch.sum(norm_diff ** 2, dim=-1)  # (B, L)
    dist = torch.mean(step_dists, dim=-1)  # (B,)
    return dist


def compute_dppo_loss(
    policy: HurdlePolicy,
    states_1: torch.Tensor,
    actions_1: torch.Tensor,
    states_2: torch.Tensor,
    actions_2: torch.Tensor,
    preferred_idx: torch.Tensor,
    lambda_reg: float = 0.1,
    drift_penalty_weight: float = 0.01,
    ref_policy: Optional[HurdlePolicy] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes DPPO training loss following Equations (5) & (6) in LAPSE paper.

    Args:
        policy: Policy network being optimized pi_theta
        states_1, actions_1: Trajectory segment 1 (B, L_s, ...)
        states_2, actions_2: Trajectory segment 2 (B, L_s, ...)
        preferred_idx: (B,) 0 if tau_1 > tau_2, 1 if tau_2 > tau_1, 2 for tie
        lambda_reg: Regularization factor in [0.01, 1.0] (default 0.1 per LAPSE)
        drift_penalty_weight: L2 penalty against reference policy to prevent divergence
        ref_policy: Optional frozen baseline policy
    """
    d1 = compute_segment_distance(policy, states_1, actions_1)  # (B,)
    d2 = compute_segment_distance(policy, states_2, actions_2)  # (B,)

    # Stable log-sum-exp computation for Equation 6:
    # s_tilde(pi, 1 > 2) = exp(-d1) / [exp(-d1) + exp(-lambda_reg * d2)]
    # log s_tilde(pi, 1 > 2) = -d1 - log(exp(-d1) + exp(-lambda_reg * d2))
    logits_1 = torch.stack([-d1, -lambda_reg * d2], dim=1)  # (B, 2)
    log_s1 = -d1 - torch.logsumexp(logits_1, dim=1)  # (B,)

    logits_2 = torch.stack([-d2, -lambda_reg * d1], dim=1)  # (B, 2)
    log_s2 = -d2 - torch.logsumexp(logits_2, dim=1)  # (B,)

    # Preferences weights mu(1) and mu(2)
    # preferred_idx: 0 -> tau_1 won, 1 -> tau_2 won, other -> tie (0.5, 0.5)
    mu1 = torch.where(preferred_idx == 0, 1.0, torch.where(preferred_idx == 1, 0.0, 0.5))
    mu2 = 1.0 - mu1

    dppo_mle_loss = - torch.mean(mu1 * log_s1 + mu2 * log_s2)

    # Policy drift regularization (normalized by action_scale across heterogeneous actuators)
    drift_loss = torch.tensor(0.0, device=d1.device)
    if ref_policy is not None and drift_penalty_weight > 0.0:
        a_scale = getattr(policy, "action_high", None)
        if a_scale is None:
            a_scale = torch.tensor([6.0, 5.0, 100.0, 20.0, 5000.0], device=states_1.device)
        else:
            a_scale = a_scale.to(states_1.device)
        with torch.no_grad():
            ref_logits, ref_mu, _ = ref_policy(states_1.view(-1, states_1.shape[-1]))
            ref_a = (torch.sigmoid(ref_logits) * ref_mu) / (a_scale + 1e-6)
        cur_logits, cur_mu, _ = policy(states_1.view(-1, states_1.shape[-1]))
        cur_a = (torch.sigmoid(cur_logits) * cur_mu) / (a_scale + 1e-6)
        drift_loss = F.mse_loss(cur_a, ref_a) * drift_penalty_weight

    total_loss = dppo_mle_loss + drift_loss

    # Accuracy: does the policy predict closer distance to the preferred segment?
    with torch.no_grad():
        pred_closer = (d1 < d2).long()  # 1 if tau_1 closer, 0 if tau_2 closer
        target_closer = torch.where(preferred_idx == 0, 1, torch.where(preferred_idx == 1, 0, -1))
        valid_mask = target_closer >= 0
        acc = (pred_closer[valid_mask] == target_closer[valid_mask]).float().mean().item() if valid_mask.any() else 0.5

    info = {
        "dppo_loss": float(dppo_mle_loss.item()),
        "total_loss": float(total_loss.item()),
        "drift_loss": float(drift_loss.item()),
        "mean_d1": float(d1.mean().item()),
        "mean_d2": float(d2.mean().item()),
        "accuracy": float(acc),
    }
    return total_loss, info


class DPPOAgent(nn.Module):
    """
    Direct Preference-based Policy Optimization Agent.
    Wraps HurdlePolicy and manages DPPO training and deployment.
    """

    def __init__(
        self,
        state_dim: int = 224,
        action_dim: int = 5,
        hidden_dim: int = 128,
        lambda_reg: float = 0.1,
        lr: float = 5e-4,
        init_policy: Optional[HurdlePolicy] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lambda_reg = lambda_reg

        self.policy = HurdlePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        if init_policy is not None:
            self.policy.load_state_dict(init_policy.state_dict())

        self.ref_policy = copy.deepcopy(self.policy).to(self.device)
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
        gate_threshold: float = 0.50,
    ) -> np.ndarray:
        """Selects action given environmental observation."""
        self.policy.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            else:
                st = state.to(self.device).float()

            if st.ndim == 1:
                st = st.unsqueeze(0)

            act, _ = self.policy.sample(st, deterministic=deterministic, gate_threshold=gate_threshold)
            return act.squeeze(0).cpu().numpy()

    def train_epoch(
        self,
        dataloader,
        clip_grad: float = 1.0,
    ) -> Dict[str, float]:
        """Runs one training epoch over preference pairs."""
        self.policy.train()
        epoch_metrics = []

        for batch in dataloader:
            s1_key = "tau_1_obs_224" if (self.state_dim == 224 and "tau_1_obs_224" in batch) else "tau_1_states"
            s2_key = "tau_2_obs_224" if (self.state_dim == 224 and "tau_2_obs_224" in batch) else "tau_2_states"
            s1 = batch[s1_key].to(self.device)
            a1 = batch["tau_1_actions"].to(self.device)
            s2 = batch[s2_key].to(self.device)
            a2 = batch["tau_2_actions"].to(self.device)
            y = batch["preferred_idx"].to(self.device)

            if s1.shape[-1] != self.state_dim or s2.shape[-1] != self.state_dim:
                raise ValueError(
                    f"[DPPO] State dimension mismatch: policy expects {self.state_dim}-dim observations "
                    f"(identical to POMEReactorEnv), but batch provided {s1.shape[-1]}-dim vectors."
                )

            loss, info = compute_dppo_loss(
                policy=self.policy,
                states_1=s1,
                actions_1=a1,
                states_2=s2,
                actions_2=a2,
                preferred_idx=y,
                lambda_reg=self.lambda_reg,
                ref_policy=self.ref_policy,
            )

            self.optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), clip_grad)
            self.optimizer.step()

            epoch_metrics.append(info)

        avg_metrics = {k: float(np.mean([m[k] for m in epoch_metrics])) for k in epoch_metrics[0]}
        return avg_metrics

    def _adapt_state_dim(self, s: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pads or truncates states to match policy target state dimension."""
        curr_dim = s.shape[-1]
        if curr_dim > target_dim:
            return s[..., :target_dim]
        elif curr_dim < target_dim:
            pad = torch.zeros(*s.shape[:-1], target_dim - curr_dim, device=s.device)
            return torch.cat([s, pad], dim=-1)
        return s

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lambda_reg": self.lambda_reg,
        }, path)
        print(f"[DPPO] Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()
        print(f"[DPPO] Loaded checkpoint from: {path}")


def train_dppo_model(
    preference_dataset,
    state_dim: int = 224,
    action_dim: int = 5,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 5e-4,
    lambda_reg: float = 0.1,
    init_policy: Optional[HurdlePolicy] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> DPPOAgent:
    """
    High-level routine to train DPPO policy on preference dataset.
    """
    from torch.utils.data import DataLoader
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lambda_reg=lambda_reg,
        lr=lr,
        init_policy=init_policy,
        device=dev,
    )

    dataloader = DataLoader(
        preference_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"[DPPO] Training on {len(preference_dataset)} preference pairs ({epochs} epochs, lambda_reg={lambda_reg})...")
    for epoch in range(epochs):
        metrics = agent.train_epoch(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch+1:02d}/{epochs:02d} | "
                f"Loss: {metrics['dppo_loss']:.4f} | "
                f"Acc: {metrics['accuracy']*100:.1f}% | "
                f"d1: {metrics['mean_d1']:.3f} | d2: {metrics['mean_d2']:.3f}"
            )

    if save_path:
        agent.save_checkpoint(save_path)

    return agent
