"""
Execution-Time Risk-Aware Selection (RAS).
Provides execution-time safety without requiring an analytical bioprocess model:
1. Generates candidate_actions C(s) = {a^(i) ~ pi_theta(s)}_{i=1}^N
2. Evaluates reward critic Q_r and failure critic ensemble
3. Aggregates conservative risk via ensemble_max (or logsumexp)
4. Selects a^exec = argmax_{a in C(s)} [ Q_r(s, a) - lambda * hat_C(s, a) ]
(penjelasan.md §3.2, Chen & Luo, 2026).
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Tuple, Dict, Optional, List
from models.critics import RewardCritic, FailureCriticEnsemble


class RiskAwareSelector:
    """
    Risk-Aware Action Selection (RAS) at execution time.
    Vetoes dangerous candidate actions and maximizes Lagrangian utility.
    """

    def __init__(
        self,
        reward_critic: RewardCritic,
        failure_critic_ensemble: FailureCriticEnsemble,
        num_candidates: int = 16,
        safety_budget: float = 0.15,
        default_lambda: float = 2.0,
        gate_threshold: float = 0.50,
    ):
        self.reward_critic = reward_critic
        self.failure_critics = failure_critic_ensemble
        self.num_candidates = num_candidates
        self.safety_budget = safety_budget
        self.lambda_dual = default_lambda
        self.gate_threshold = gate_threshold

    def select_action(
        self,
        state: torch.Tensor,
        policy,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Executes the Risk-Aware Selection rule:
        a_exec = argmax [ Q_r(s, a) - lambda * hat_C(s, a) ]
        Applies Stage 0 Hurdle Gate thresholding to suppress zero-inflation false dosing.
        """
        device = state.device
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Stage 0 Hurdle Gate evaluation: if P(active | s) < gate_threshold, enforce zero dosing
        with torch.no_grad():
            if hasattr(policy, "forward_raw"):
                gate_logits, mu_logit, std_logit = policy.forward_raw(state)
                gate_probs = torch.sigmoid(gate_logits)  # (1, action_dim)
                gate_mask = (gate_probs >= self.gate_threshold).float()
            elif hasattr(policy, "dose_gate"):
                gate_logits, _, _ = policy(state)
                gate_probs = torch.sigmoid(gate_logits)
                gate_mask = (gate_probs >= self.gate_threshold).float()
                mu_logit, std_logit = None, None
            else:
                gate_probs = None
                gate_mask = None
                mu_logit, std_logit = None, None

        # 1. Sample N candidate actions in a single vectorized batch call
        expanded_state = state.repeat(self.num_candidates, 1)
        with torch.no_grad():
            candidates_tensor, _ = policy.sample(expanded_state, deterministic=False)

            # Contingency hurdle candidates keep rare actuators & zero shutoffs testable by Q_r and hat_C
            if hasattr(policy, "forward_raw") and self.num_candidates >= 4 and gate_probs is not None:
                n_cont = max(2, self.num_candidates // 3)
                p_soft = torch.clamp(0.70 * gate_probs + 0.30 * 0.50, 0.15, 0.92).repeat(n_cont, 1)
                z_soft = torch.bernoulli(p_soft)
                eta_soft = dist.Normal(
                    mu_logit.repeat(n_cont, 1),
                    torch.clamp(std_logit.repeat(n_cont, 1), min=1.0),
                ).rsample()
                u_soft = 2e-4 + (1.0 - 2e-4 - 1e-5) * torch.sigmoid(torch.clamp(eta_soft, -9.0, 9.0))
                cont_cands = torch.clamp(z_soft * u_soft * policy.action_high, min=0.0)
                candidates_tensor[-n_cont:] = torch.minimum(cont_cands, policy.action_high)

            if deterministic:
                det_cand, _ = policy.sample(state, deterministic=True)
                if gate_mask is not None:
                    det_cand = det_cand * gate_mask
                candidates_tensor[0:1] = det_cand

            # 2. Evaluate reward critic Q_r and failure critic ensemble
            q_values = self.reward_critic(expanded_state, candidates_tensor)
            conservative_risk, all_risks = self.failure_critics(expanded_state, candidates_tensor)
            safe_mask = conservative_risk <= self.safety_budget

            max_risk = all_risks.max(dim=-1).values

            # 3. Formulate Lagrangian utility: Q_r(s, a) - lambda * hat_C(s, a)
            lagrangian_utility = q_values - (self.lambda_dual * conservative_risk)

            if safe_mask.any():
                lagrangian_utility[~safe_mask] = -1e9
                best_idx = torch.argmax(lagrangian_utility)
                was_vetoed = False
            else:
                best_idx = torch.argmin(conservative_risk)
                was_vetoed = True

            exec_action = candidates_tensor[best_idx : best_idx + 1]

        info = {
            "selected_q_value": float(q_values[best_idx].item()),
            "selected_risk": float(conservative_risk[best_idx].item()),
            "max_risk_in_pool": float(max_risk.max().item()),
            "min_risk_in_pool": float(max_risk.min().item()),
            "lambda_multiplier": float(self.lambda_dual),
            "was_vetoed": was_vetoed,
        }

        return exec_action.squeeze(0), info
