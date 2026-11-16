"""
Hurdle Actor Policy Network (pi_sft and pi_theta).
Models Zero-Inflated Continuous Actuation via Cragg's (1971) Two-Part Hurdle Model:
1. Discrete gate z_d ~ Bernoulli(p_d) controlling whether pump/lamp lever d activates.
2. Normalized Logit-Normal continuous magnitude a_d = sigmoid(eta_d) * H_d on (0, H_d)
   when z_d = 1, with exact Haarnoja et al. (2018) / Atchison & Shen (1980) Jacobian correction
   to ensure scale-invariant gradients across heterogeneously bounded actuators
   (0-6 h aeration, 0-5 g NaHCO3, 0-100 mL inoculum, 0-20 mL molasses, 0-5000 lux LED).
"""

import os
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

try:
    from config import ACTION_COLUMNS, ACTION_BOUNDS
    _DEFAULT_ACTION_HIGHS = [float(ACTION_BOUNDS[c][1]) for c in ACTION_COLUMNS]
except Exception:
    _DEFAULT_ACTION_HIGHS = [6.0, 5.0, 100.0, 20.0, 5000.0]


def encode_state_symlog(state: torch.Tensor) -> torch.Tensor:
    """
    Scale-invariant signed logarithmic state compression (Hafner et al., 2023):
        phi(s) = 0.1 * sign(s) * ln(1 + |s|)
    Keeps multi-magnitude bioreactor telemetry (from pH ~7 to ppm ~4e5) in O(1)
    while using purely opset-14 ONNX-compatible primitives.
    """
    clean = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.1 * torch.sign(clean) * torch.log(1.0 + torch.abs(clean))


def get_action_high_tensor(
    action_dim: int = 5,
    action_bounds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resolve physical upper-bound vector for action normalization to [0, 1]^D."""
    if action_bounds is not None:
        high = action_bounds.to(dtype=torch.float32).flatten()
    else:
        base = list(_DEFAULT_ACTION_HIGHS)
        if len(base) < action_dim:
            base.extend([1.0] * (action_dim - len(base)))
        high = torch.tensor(base[:action_dim], dtype=torch.float32)
    return torch.clamp(high, min=1e-6)


class HurdlePolicy(nn.Module):
    """
    Cragg Hurdle Policy Network with scale-invariant Logit-Normal magnitude distribution.

    Each control channel d in {0..4} combines:
      - Gate probability p_d = sigmoid(g_d(s)) with exact log-sigmoid stability.
      - Unconstrained Gaussian latent eta_d ~ N(mu_eta,d(s), sigma_eta,d^2)
        mapped via u_d = sigmoid(eta_d) in (0, 1) and scaled by action_high[d].
    """

    def __init__(
        self,
        state_dim: int = 224,
        action_dim: int = 5,
        hidden_dim: int = 128,
        action_bounds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared multimodal feature extraction trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 1. Discrete gating logits (Bernoulli parameter pre-sigmoid)
        self.dose_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # 2. Continuous magnitude parameters in unconstrained logit space
        self.amount_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        # Initialized at exp(-0.5) ~= 0.61 in logit space (~0.15 std around sigmoid midpoint)
        self.amount_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        if action_bounds is None:
            self.register_buffer(
                "action_high",
                get_action_high_tensor(action_dim),
            )
        else:
            self.register_buffer("action_high", get_action_high_tensor(action_dim, action_bounds))

        self.register_buffer("_zero_eps", torch.tensor(1e-4, dtype=torch.float32))

    def forward_raw(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (gate_logits, mu_logit, std_logit) in unconstrained distribution space."""
        norm_state = encode_state_symlog(state)
        feats = self.trunk(norm_state)
        gate_logits = torch.clamp(self.dose_gate(feats), min=-15.0, max=15.0)
        mu_logit = torch.clamp(self.amount_mean(feats), min=-7.5, max=7.5)
        std_logit = torch.exp(torch.clamp(self.amount_log_std, min=-5.0, max=1.5))
        std_logit = std_logit.unsqueeze(0).expand_as(mu_logit)
        return gate_logits, mu_logit, std_logit

    def kl_divergence(self, other: "HurdlePolicy", state: torch.Tensor) -> torch.Tensor:
        """
        Exact closed-form KL divergence D_KL(pi_self(.|s) || pi_other(.|s)) per sample:
            sum_d [ KL(Bern(p1_d) || Bern(p2_d)) + p1_d * KL(N(mu1_d, std1_d^2) || N(mu2_d, std2_d^2)) ]
        Analytically invariant to the Logit-Normal action-bound Jacobian (Atchison & Shen, 1980),
        non-negative, and free of Monte Carlo clamp dead-zones.
        """
        g1, mu1, std1 = self.forward_raw(state)
        with torch.no_grad():
            g2, mu2, std2 = other.forward_raw(state)

        p1 = torch.sigmoid(g1)
        kl_gate = p1 * (F.logsigmoid(g1) - F.logsigmoid(g2)) + (1.0 - p1) * (
            F.logsigmoid(-g1) - F.logsigmoid(-g2)
        )
        kl_amt = torch.log(std2 / std1) + (std1.pow(2) + (mu1 - mu2).pow(2)) / (2.0 * std2.pow(2)) - 0.5
        return torch.clamp((kl_gate + p1 * kl_amt).sum(dim=-1), min=0.0)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Closed-form two-part Bernoulli + active latent Gaussian entropy per sample."""
        g, _, std = self.forward_raw(state)
        p = torch.sigmoid(g)
        h_gate = -(p * F.logsigmoid(g) + (1.0 - p) * F.logsigmoid(-g))
        h_amt = 0.5 * torch.log(2.0 * torch.pi * torch.e * std.pow(2))
        return (h_gate + p * h_amt).sum(dim=-1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward-compatible forward pass returning physical units for ONNX export:
        Returns (gate_logits, mu_phys, std_phys_approx).
        """
        gate_logits, mu_logit, std_logit = self.forward_raw(state)
        u_mean = torch.sigmoid(mu_logit)
        mu_phys = u_mean * self.action_high
        std_phys = u_mean * (1.0 - u_mean) * std_logit * self.action_high
        return gate_logits, mu_phys, std_phys

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute exact two-part Cragg Hurdle log probability log pi_theta(a | s):
        - Zero action (a_d <= eps): log(1 - p_d) via numerically stable logsigmoid
        - Active dose (a_d > eps):  log(p_d) + log N(logit(u_d); mu_logit, std_logit)
                                    - log(u_d * (1 - u_d) * action_high_d)
        """
        gate_logits, mu_logit, std_logit = self.forward_raw(state)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalized zero detection threshold ensures exact separation for all actuator scales
        is_zero = (action <= 1e-4 * self.action_high).float()
        is_active = 1.0 - is_zero

        # Numerically stable log Bernoulli probabilities
        log_p_active = F.logsigmoid(gate_logits)
        log_p_zero = F.logsigmoid(-gate_logits)

        # Map physical action to interior of (0, 1) to evaluate Logit-Normal density + Jacobian
        u = torch.clamp(action / self.action_high, min=1e-5, max=1.0 - 1e-5)
        eta = torch.log(u) - torch.log1p(-u)

        normal_dist = dist.Normal(mu_logit, std_logit)
        log_jacobian = torch.log(u) + torch.log1p(-u) + torch.log(self.action_high)
        log_prob_amount = normal_dist.log_prob(eta) - log_jacobian

        # Mask inactive terms before summation to keep graph cleanly conditioned
        total_log_prob = (is_zero * log_p_zero) + (is_active * (log_p_active + log_prob_amount))
        return total_log_prob.sum(dim=-1)

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        gate_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample physical bioreactor actions in [0, action_high]^D.
        Uses straight-through Bernoulli gate estimator + pathwise Logit-Normal reparameterization
        so gradients flow cleanly from critics back through both gate_probs and continuous magnitudes.
        """
        gate_logits, mu_logit, std_logit = self.forward_raw(state)
        gate_probs = torch.sigmoid(gate_logits)

        if deterministic:
            is_active = (gate_probs >= gate_threshold).float()
            eta = mu_logit
        else:
            b_sample = torch.bernoulli(gate_probs)
            is_active = gate_probs + (b_sample - gate_probs).detach()
            normal_dist = dist.Normal(mu_logit, std_logit)
            eta = torch.clamp(normal_dist.rsample(), min=-9.0, max=9.0)

        u = torch.sigmoid(eta)
        # Keep active magnitudes strictly above the zero threshold
        u_bounded = torch.clamp(u, min=2e-4, max=1.0 - 1e-5)
        amount = u_bounded + (u - u_bounded).detach() if False else (2e-4 + (1.0 - 2e-4 - 1e-5) * u)

        action = is_active * amount * self.action_high
        action = torch.clamp(action, min=0.0)
        action = torch.minimum(action, self.action_high)
        log_p = self.log_prob(state, action)
        return action, log_p
