"""
T-REX Preference-Based Reward Model Network r_phi.
Learns continuous scalar reward signals from human pairwise preferences (Bradley-Terry)
without relying on manually hand-crafted heuristic weights (penjelasan.md §4, Liu et al., 2023).
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class RewardNet(nn.Module):
    """
    Continuous scalar reward model r_phi(s, a) using T-REX preference extrapolation.
    Consumes state vector and continuous dosing action vector to evaluate bioreactor quality.
    """

    def __init__(
        self,
        state_dim: int = 224,
        action_dim: int = 5,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input is state-action pair (s, a)
        in_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate scalar reward r_phi(s, a).
        Shape: (B, 1) or (B,)
        """
        sa = torch.cat([state, action], dim=-1)
        r = self.net(sa).squeeze(-1)
        return r


class RewardEnsemble(nn.Module):
    """
    Ensemble of 3 RewardNet models to estimate uncertainty and prevent reward hacking.
    """

    def __init__(self, state_dim: int = 224, action_dim: int = 5, ensemble_size: int = 3):
        super().__init__()
        self.models = nn.ModuleList([
            RewardNet(state_dim=state_dim, action_dim=action_dim)
            for _ in range(ensemble_size)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_reward: Average reward across the ensemble
            std_reward: Epistemic disagreement between reward models
        """
        rewards = torch.stack([model(state, action) for model in self.models], dim=-1)
        mean_reward = rewards.mean(dim=-1)
        std_reward = rewards.std(dim=-1)
        return mean_reward, std_reward
