"""
Primal-Dual Multiplier Update for Lagrangian Safety.
Adapts Lagrange multiplier lambda at the end of each cultivation batch:
lambda <- clip(lambda + alpha_lambda * (p_bar_k - delta), 0, lambda_max)
Guarantees that long-term empirical violation probability stays within constraint_level delta.
(penjelasan.md §3.2, Chen & Luo 2026).
"""

import numpy as np
from typing import List, Dict


class PrimalDualSafetyController:
    """
    Manages dual multiplier lambda for Risk-Aware Selection using primal-dual ascent.
    """

    def __init__(
        self,
        constraint_level: float = 0.05,     # Allowed violation probability delta <= 5%
        dual_step_size: float = 0.5,        # alpha_lambda
        lambda_init: float = 2.0,
        lambda_max: float = 25.0,
        window_size: int = 10,              # Episodes in moving-average window W
    ):
        self.constraint_level = constraint_level
        self.delta = constraint_level
        self.alpha_lambda = dual_step_size
        self.dual_step = dual_step_size
        self.lambda_dual = lambda_init
        self.lambda_max = lambda_max
        self.window_size = window_size
        self.episode_violations: List[int] = []

    def record_episode_result(self, had_violation: bool):
        """Record whether recent episode finished safely (0) or suffered a violation (1)."""
        self.episode_violations.append(1 if had_violation else 0)

    def dual_update(self) -> float:
        """
        Execute primal-dual update step at episode end:
        p_bar = mean of recent violations
        lambda <- clip(lambda + alpha * (p_bar - delta), 0, lambda_max)
        """
        if len(self.episode_violations) == 0:
            return self.lambda_dual

        recent = self.episode_violations[-self.window_size :]
        p_bar = float(np.mean(recent))

        # Dual ascent step
        delta_p = p_bar - self.delta
        self.lambda_dual = float(np.clip(
            self.lambda_dual + (self.alpha_lambda * delta_p),
            0.0,
            self.lambda_max,
        ))

        return self.lambda_dual

    def get_violation_rate(self) -> float:
        """Calculate overall historical violation rate."""
        if len(self.episode_violations) == 0:
            return 0.0
        return float(np.mean(self.episode_violations))
