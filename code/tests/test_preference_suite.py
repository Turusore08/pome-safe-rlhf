"""
Unit tests for Stage 3: Preference Learning Suite (Tri-Reward, Segment Length L_s Ablation,
Rubric Evaluation, and Bradley-Terry Ensemble).
"""

import os
import sys
import unittest
import numpy as np
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from preference.segment_sampler import (
    TrajectorySegment,
    sample_segment_pairs,
    PREF_STATE_FEATURES,
    PREF_ACTION_FEATURES,
)
from preference.labeling_rubric import evaluate_rubric
from preference.train_reward import bradley_terry_loss, TriRewardEvaluator
from models.reward_net import RewardNet, RewardEnsemble


class TestPreferenceSuite(unittest.TestCase):
    def setUp(self):
        # Create dummy trajectories
        rng = np.random.RandomState(42)
        self.dummy_trajectories = []
        for r_id in range(4):
            n_steps = 100
            states = np.zeros((n_steps, len(PREF_STATE_FEATURES)), dtype=np.float32)
            # COD decays from 3000 to 200
            states[:, 0] = np.linspace(3000.0, 200.0, n_steps)
            # BOD = 0.5 * COD
            states[:, 1] = 0.5 * states[:, 0]
            # pH around 8.5
            states[:, 4] = 8.5 + rng.normal(0.0, 0.2, n_steps)
            # DO around 6.0
            states[:, 5] = 6.0
            # Biomass increases from 0.5 to 1.8
            states[:, -1] = np.linspace(0.5, 1.8, n_steps)

            actions = rng.uniform(0.0, 2.0, (n_steps, len(PREF_ACTION_FEATURES))).astype(np.float32)
            violations = np.zeros(n_steps, dtype=np.float32)
            d_cv = np.zeros(n_steps, dtype=np.float32)
            costs = np.ones(n_steps, dtype=np.float32) * 20.0

            # Inject artificial violation in run 1
            if r_id == 1:
                violations[10:15] = 1.0
                d_cv[10:15] = 1.5
                states[10:15, 4] = 5.2  # Acidic violation

            self.dummy_trajectories.append({
                "run_id": f"dummy_{r_id}",
                "states": states,
                "actions": actions,
                "violations": violations,
                "d_cv": d_cv,
                "costs": costs,
            })

    def test_segment_sampler_lengths(self):
        # Test exact length 4
        pairs_4 = sample_segment_pairs(self.dummy_trajectories, num_pairs=15, segment_length=4)
        for t1, t2 in pairs_4:
            self.assertEqual(len(t1), 4)
            self.assertEqual(len(t2), 4)

        # Test exact length 8
        pairs_8 = sample_segment_pairs(self.dummy_trajectories, num_pairs=15, segment_length=8)
        for t1, t2 in pairs_8:
            self.assertEqual(len(t1), 8)
            self.assertEqual(len(t2), 8)

        # Test exact length 20
        pairs_20 = sample_segment_pairs(self.dummy_trajectories, num_pairs=15, segment_length=20)
        for t1, t2 in pairs_20:
            self.assertEqual(len(t1), 20)
            self.assertEqual(len(t2), 20)

        # Test combined mixture (lengths must be in {4, 8, 20})
        pairs_comb = sample_segment_pairs(self.dummy_trajectories, num_pairs=60, segment_length="comb")
        lengths = set(len(t1) for t1, _ in pairs_comb)
        self.assertTrue(lengths.issubset({4, 8, 20}))
        self.assertGreater(len(lengths), 1)

    def test_rubric_rule_safety_first(self):
        # Physiological bioreactor states: [cod, bod, toc, tss, ph, do, tds, nh3, water_temp, biomass]
        base = np.array([1000.0, 500.0, 320.0, 400.0, 8.5, 5.0, 4000.0, 60.0, 30.0, 1.0], dtype=np.float32)
        s1 = np.tile(base, (8, 1))
        s1[:, 4] = 5.2  # Tier-2 biological safety violation (pH < 5.5)
        s2 = np.tile(base, (8, 1))
        s2[:, 4] = 8.5  # Biologically safe

        tau_viol = TrajectorySegment(s1, np.zeros((8, 5)), {"violations": np.ones(8), "d_cv": np.ones(8) * 0.3, "total_cost": 50.0})
        tau_safe = TrajectorySegment(s2, np.zeros((8, 5)), {"violations": np.zeros(8), "d_cv": np.zeros(8), "total_cost": 50.0})

        # Safe segment MUST defeat violating segment
        self.assertEqual(evaluate_rubric(tau_viol, tau_safe), 1)
        self.assertEqual(evaluate_rubric(tau_safe, tau_viol), 0)

    def test_rubric_rule_compliance_milestone(self):
        # Snippet 1 finishes with COD=280 mg/L (compliant), Snippet 2 finishes with COD=600 mg/L
        s1 = np.ones((8, 10), dtype=np.float32) * 400.0
        s1[-1, 0] = 280.0  # Compliant <= 350
        s1[:, 4] = 8.2

        s2 = np.ones((8, 10), dtype=np.float32) * 800.0
        s2[-1, 0] = 600.0  # Non-compliant
        s2[:, 4] = 8.2

        tau_comp = TrajectorySegment(s1, np.zeros((8, 5)), {"violations": np.zeros(8), "total_cost": 50.0})
        tau_noncomp = TrajectorySegment(s2, np.zeros((8, 5)), {"violations": np.zeros(8), "total_cost": 50.0})

        self.assertEqual(evaluate_rubric(tau_comp, tau_noncomp), 0)
        self.assertEqual(evaluate_rubric(tau_noncomp, tau_comp), 1)

    def test_rubric_rule_pareto_dominance(self):
        # Snippet 1 has higher COD reduction AND lower cost
        s1 = np.ones((8, 10), dtype=np.float32) * 1000.0
        s1[0, 0] = 2500.0
        s1[-1, 0] = 1200.0  # Reduction = 1300
        s1[:, 4] = 8.5

        s2 = np.ones((8, 10), dtype=np.float32) * 1000.0
        s2[0, 0] = 2500.0
        s2[-1, 0] = 2100.0  # Reduction = 400
        s2[:, 4] = 8.5

        tau_pareto_win = TrajectorySegment(s1, np.zeros((8, 5)), {"violations": np.zeros(8), "total_cost": 30.0})
        tau_pareto_lose = TrajectorySegment(s2, np.zeros((8, 5)), {"violations": np.zeros(8), "total_cost": 80.0})

        self.assertEqual(evaluate_rubric(tau_pareto_win, tau_pareto_lose), 0)
        self.assertEqual(evaluate_rubric(tau_pareto_lose, tau_pareto_win), 1)

    def test_bradley_terry_loss(self):
        r1 = torch.tensor([5.0])
        r2 = torch.tensor([1.0])
        pref_0 = torch.tensor([0])  # tau_1 > tau_2 (consistent with r1 > r2)
        pref_1 = torch.tensor([1])  # tau_2 > tau_1 (inconsistent with r1 > r2)

        loss_consistent = bradley_terry_loss(r1, r2, pref_0)
        loss_inconsistent = bradley_terry_loss(r1, r2, pref_1)

        # Consistent ranking must yield substantially lower loss
        self.assertLess(loss_consistent.item(), loss_inconsistent.item())
        self.assertLess(loss_consistent.item(), 0.05)
        self.assertGreater(loss_inconsistent.item(), 3.5)

    def test_tri_reward_evaluator(self):
        evaluator = TriRewardEvaluator()
        s_dict = {"cod": 300.0, "bod": 140.0, "total_suspended_solids": 180.0, "ph": 8.0, "dissolved_oxygen": 6.0, "biomass": 1.2}
        a_vec = np.array([3.0, 0.5, 0.0, 0.0, 3000.0], dtype=np.float32)
        s_vec = np.array([300.0, 140.0, 80.0, 180.0, 8.0, 6.0, 5500.0, 45.0, 30.0, 1.2], dtype=np.float32)

        # r_rig test
        r_rig_val = evaluator.r_rig(s_dict, a_vec)
        self.assertIsInstance(r_rig_val, float)

        # r_pref test (fallback when no checkpoint)
        r_pref_val, std_val = evaluator.r_pref(s_vec, a_vec)
        self.assertIsInstance(r_pref_val, float)
        self.assertGreaterEqual(std_val, 0.0)

        # r_hybrid test
        r_hyb_val = evaluator.r_hybrid(s_dict, s_vec, a_vec, alpha=1.0)
        self.assertAlmostEqual(r_hyb_val, r_rig_val + r_pref_val, places=4)


if __name__ == "__main__":
    unittest.main()
