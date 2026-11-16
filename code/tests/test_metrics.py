"""
Unit tests for standardized LAPSE evaluation metrics, PerMenLHK compliance,
5-treatment action breakdown (T1-T5), and step-by-step reward/punishment ledger.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.metrics import (
    compute_lapse_metrics,
    compute_action_treatment_breakdown,
    build_step_reward_punishment_ledger,
    verify_permenlhk_compliance,
    compute_step_violation_magnitude,
)


class TestLAPSEMetrics(unittest.TestCase):
    def test_compute_lapse_metrics_zero_and_nonzero_violations(self):
        # Trajectory 1: Safe & compliant
        states_safe = []
        for t in range(10):
            states_safe.append({
                "cod": max(200.0, 1200.0 - 200.0 * t),
                "bod": max(80.0, 600.0 - 100.0 * t),
                "total_suspended_solids": 180.0,
                "dissolved_ammonia": 30.0,
                "ph": 7.8,
                "dissolved_oxygen": 5.5,
                "water_temperature": 30.0,
            })
        acts_safe = np.tile([3.0, 0.5, 0.0, 0.0, 3000.0], (9, 1))
        rews_safe = np.ones(9) * 2.5

        # Tier-1 vs Tier-2 separation: pH=5.5 and pH=9.6 fail PerMenLH effluent [6.0, 9.0]
        # while staying within Chlorella vulgaris survival bounds [5.5, 10.5] -> n_cv == 0.
        states_effluent_only = [dict(s) for s in states_safe]
        states_effluent_only[-2]["ph"] = 5.5
        states_effluent_only[-1]["ph"] = 9.6

        res_eff = compute_lapse_metrics([
            {"rewards": rews_safe, "states": states_effluent_only, "actions": acts_safe}
        ], step_hours=6.0)
        self.assertEqual(res_eff["n_cv"], 0)
        self.assertEqual(res_eff["compliance_rate_pct"], 0.0)

        # Tier-2 Biological Safety failures: pH=5.2 (<5.5) and pH=10.8 (>10.5) -> n_cv == 2.
        states_viol = [dict(s) for s in states_safe]
        states_viol[2]["ph"] = 5.2
        states_viol[3]["ph"] = 10.8

        res_safe = compute_lapse_metrics([
            {"rewards": rews_safe, "states": states_safe, "actions": acts_safe}
        ], step_hours=6.0)
        self.assertAlmostEqual(res_safe["G_mean"], 22.5, places=4)
        self.assertEqual(res_safe["n_cv"], 0)
        self.assertAlmostEqual(res_safe["d_cv_mean"], 0.0, places=5)
        self.assertTrue(res_safe["compliance_rate_pct"] == 100.0)

        res_mixed = compute_lapse_metrics([
            {"rewards": rews_safe, "states": states_safe, "actions": acts_safe},
            {"rewards": rews_safe - 1.0, "states": states_viol, "actions": acts_safe},
        ], step_hours=6.0)
        self.assertEqual(res_mixed["n_cv"], 2)
        self.assertGreater(res_mixed["d_cv_mean"], 0.0)
        self.assertAlmostEqual(res_mixed["G_mean"], 18.0, places=4)

    def test_action_treatment_breakdown(self):
        # 8 steps at 6h/step = 2 days
        actions = np.array([
            [4.0, 1.0, 20.0, 5.0, 3500.0],
            [4.0, 0.0, 0.0, 0.0, 3500.0],
            [4.0, 0.5, 0.0, 0.0, 3500.0],
            [4.0, 0.0, 0.0, 0.0, 3500.0],
            [3.0, 0.0, 0.0, 2.0, 2500.0],
            [3.0, 0.0, 0.0, 0.0, 2500.0],
            [3.0, 0.0, 0.0, 0.0, 2500.0],
            [3.0, 0.0, 0.0, 0.0, 2500.0],
        ])
        bd = compute_action_treatment_breakdown(actions, step_hours=6.0)
        self.assertEqual(bd["total_days"], 2)
        self.assertAlmostEqual(bd["T1_aeration_total_h"], 28.0)
        self.assertAlmostEqual(bd["T4_nahco3_total_g"], 1.5)
        self.assertAlmostEqual(bd["T5_inoculum_total_ml"], 20.0)
        self.assertAlmostEqual(bd["T3_molasses_total_ml"], 7.0)
        self.assertAlmostEqual(bd["T2_light_mean_lux"], 3000.0)
        self.assertEqual(len(bd["daily_profile"]), 2)

    def test_reward_punishment_ledger(self):
        states = [
            {"cod": 3000.0, "bod": 1500.0, "total_suspended_solids": 1200.0, "dissolved_ammonia": 80.0, "ph": 8.2, "dissolved_oxygen": 3.0, "water_temperature": 30.0, "biomass": 0.4},
            {"cod": 2200.0, "bod": 1100.0, "total_suspended_solids": 1000.0, "dissolved_ammonia": 65.0, "ph": 8.5, "dissolved_oxygen": 5.2, "water_temperature": 30.0, "biomass": 0.65},
            {"cod": 1800.0, "bod": 900.0, "total_suspended_solids": 900.0, "dissolved_ammonia": 55.0, "ph": 5.4, "dissolved_oxygen": 4.8, "water_temperature": 30.0, "biomass": 0.70},
        ]
        actions = np.array([
            [4.0, 0.5, 10.0, 0.0, 3000.0],
            [4.0, 0.0, 0.0, 0.0, 3000.0],
        ])
        ledger = build_step_reward_punishment_ledger(states, actions, step_hours=6.0, safety_penalty_weight=50.0)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger[0]["violation_flag"], 0)
        self.assertEqual(ledger[1]["violation_flag"], 1)
        self.assertAlmostEqual(ledger[1]["constraint_penalty"], 50.0)
        self.assertGreater(ledger[0]["E_effluent_idr"], 0.0)
        self.assertGreater(ledger[0]["B_biomass_idr"], 0.0)
        self.assertGreater(ledger[0]["Q_co2_idr"], 0.0)


if __name__ == "__main__":
    unittest.main()
