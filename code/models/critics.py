"""
Critic Architectures for Safe Reinforcement Learning.
1. Reward Critic Q_r(s, a): Estimates expected discounted cumulative return.
2. Failure Critic Ensemble C_{psi_m}(s, a): Ensemble of M critics estimating
   the cumulative probability of triggering a safety violation within the episode.
LayerNorm is applied to all hidden layers to stabilize high UTD updates (penjelasan.md §3.2, Chen & Luo 2026).
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


try:
    from models.policy_sft import encode_state_symlog, get_action_high_tensor
except ImportError:
    from .policy_sft import encode_state_symlog, get_action_high_tensor


def _build_normalized_sa(
    state: torch.Tensor,
    action: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """Compress multi-scale telemetry via symlog and scale actions to [0, 1]^D."""
    s_norm = encode_state_symlog(state)
    clean_action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    a_norm = clean_action / action_high
    return torch.cat([s_norm, a_norm], dim=-1)


class RewardCritic(nn.Module):
    """
    Reward Critic Q_r(s, a) mapping state-action pair to expected return.
    Uses symlog state compression + [0, 1]^D action scaling so heterogeneous control
    levers (0-5 g NaHCO3 vs 0-5000 lux LED) have balanced LayerNorm gradient sensitivity.
    """

    def __init__(
        self,
        state_dim: int = 224,
        action_dim: int = 5,
        hidden_dim: int = 256,
        action_bounds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.register_buffer("action_high", get_action_high_tensor(action_dim, action_bounds))

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = _build_normalized_sa(state, action, self.action_high)
        q_reward = self.net(sa).squeeze(-1)
        return q_reward


class FailureCritic(nn.Module):
    """
    Failure / Safety Critic C_psi(s, a) estimating within-episode violation probability.
    Output is bounded in [0, 1] via Sigmoid activation and trained with BCE loss (no discounting).
    """

    def __init__(
        self,
        state_dim: int = 224,
        action_dim: int = 5,
        hidden_dim: int = 256,
        action_bounds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("action_high", get_action_high_tensor(action_dim, action_bounds))
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = _build_normalized_sa(state, action, self.action_high)
        logits = self.net(sa).squeeze(-1)
        prob_violation = torch.sigmoid(logits)
        return prob_violation


class FailureCriticEnsemble(nn.Module):
    """
    Ensemble of M failure critics for conservative risk estimation:
    hat_C(s, a) = max_{m=1..M} C_{psi_m}(s, a)
    Pessimism under disagreement protects against untested out-of-distribution actions.
    """

    def __init__(
        self,
        state_dim: int = 224,
        action_dim: int = 5,
        ensemble_size: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_critics = ensemble_size
        self.critics = nn.ModuleList([
            FailureCritic(state_dim, action_dim, hidden_dim)
            for _ in range(ensemble_size)
        ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            conservative_risk: Conservative risk estimate across ensemble (max or upper quantile)
            all_risks: (B, ensemble_size) tensor of individual critic probabilities
        """
        all_risks = torch.stack([c(state, action) for c in self.critics], dim=-1)
        # Conservative risk: maximum probability across ensemble
        conservative_risk = all_risks.max(dim=-1).values
        return conservative_risk, all_risks
