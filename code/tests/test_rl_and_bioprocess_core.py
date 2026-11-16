import os
import sys
import tempfile
import unittest

import numpy as np
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import (
    ACTION_BOUNDS,
    ACTION_COLUMNS,
    BIOLOGICAL_SAFETY_LIMITS,
    EFFLUENT_COMPLIANCE_LIMITS,
    REACTOR_TOTAL_CAPACITY_LITERS,
    REACTOR_VOLUME_LITERS,
)
from evaluation.metrics import compute_step_violation_magnitude, verify_permenlhk_compliance
from models.bioprocess_core import (
    beer_lambert_i_avg,
    compute_conservative_charge_z,
    ctmi_cardinal_np,
    do_saturation_mgL,
    f_light_haldane_bernard,
    free_nh3_fraction,
    lux_to_ppfd,
    solve_ph,
)
from models.critics import FailureCriticEnsemble, RewardCritic
from models.policy_sft import HurdlePolicy
from models.reward_net import RewardNet
from preference.train_reward import bradley_terry_loss
from rl.cal_ql import compute_cal_ql_loss, compute_empirical_mc_returns
from rl.dppo import DPPOAgent
from rl.sac_kl import SACKLAgent


class TestBioprocessCoreAndTwoTierLimits(unittest.TestCase):
    def test_standard_volume_and_benson_krause_do(self):
        self.assertAlmostEqual(REACTOR_VOLUME_LITERS, 2.5, places=4)
        self.assertAlmostEqual(REACTOR_TOTAL_CAPACITY_LITERS, 3.0, places=4)
        # Benson & Krause (1984) standard freshwater benchmark at 20 C, 1013.25 hPa, 0 PSU -> 9.092 mg/L
        self.assertAlmostEqual(float(do_saturation_mgL(20.0, 1013.25, 0.0)), 9.092, places=2)
        self.assertGreater(float(do_saturation_mgL(25.0, 1013.25, 0.0)), float(do_saturation_mgL(30.0, 1013.25, 0.0)))

    def test_emerson_nh3_and_carbonate_ph_inversion(self):
        # Emerson et al. (1975): pKa at 25 C ~ 9.245, so at pH 9.245 & I=0 fraction ~ 0.5
        self.assertAlmostEqual(float(free_nh3_fraction(25.0, 9.245, ionic_strength=0.0)), 0.5, places=2)
        # Charge balance inversion recovers initial pH accurately
        z_net = compute_conservative_charge_z(ph=7.4, dic_mol_l=0.015, tan_mol_l=0.004, temp_c=30.0)
        ph_rec = solve_ph(dic_mol_l=0.015, z_net_eq_l=z_net, tan_mol_l=0.004, temp_c=30.0)
        self.assertAlmostEqual(ph_rec, 7.4, places=3)

    def test_photokinetics_and_ctmi_bounds(self):
        self.assertEqual(float(beer_lambert_i_avg(0.0, 1.0, 0.1, 12.5)), 0.0)
        self.assertEqual(float(f_light_haldane_bernard(0.0, 150.0)), 0.0)
        self.assertAlmostEqual(float(lux_to_ppfd(1000.0)), 14.5, places=3)
        self.assertEqual(float(ctmi_cardinal_np(10.0, 15.0, 30.0, 42.0)), 0.0)
        self.assertAlmostEqual(float(ctmi_cardinal_np(30.0, 15.0, 30.0, 42.0)), 1.0, places=5)

    def test_two_tier_compliance_vs_biological_safety(self):
        # pH 9.8 and DO 3.0 are biologically safe for C. vulgaris ([5.5, 10.5]),
        # yet fail PerMenLH No. 5/2014 effluent discharge ([6.0, 9.0]).
        state_bio_ok_effluent_fail = {
            "cod": 280.0,
            "bod": 80.0,
            "total_suspended_solids": 180.0,
            "dissolved_ammonia": 30.0,
            "ph": 9.8,
            "dissolved_oxygen": 3.0,
            "water_temperature": 31.0,
        }
        v_flag, d_mag = compute_step_violation_magnitude(state_bio_ok_effluent_fail)
        self.assertEqual(v_flag, 0)
        self.assertEqual(d_mag, 0.0)

        comp = verify_permenlhk_compliance([state_bio_ok_effluent_fail] * 4, step_hours=6.0)
        self.assertFalse(comp["compliance_check"])

        # pH 10.8 exceeds BIOLOGICAL_SAFETY_LIMITS["ph"]["max"] (10.5) -> hard violation
        state_bio_fail = dict(state_bio_ok_effluent_fail, ph=10.8)
        v_flag_2, d_mag_2 = compute_step_violation_magnitude(state_bio_fail)
        self.assertEqual(v_flag_2, 1)
        self.assertAlmostEqual(d_mag_2, 10.8 - BIOLOGICAL_SAFETY_LIMITS["ph"]["max"], places=4)
        self.assertEqual(EFFLUENT_COMPLIANCE_LIMITS["ph_max"], 9.0)


class TestRLAlgorithmsAndHurdlePolicy(unittest.TestCase):
    def test_hurdle_policy_gradients_bounds_and_onnx_parity(self):
        import onnxruntime as ort

        torch.manual_seed(0)
        policy = HurdlePolicy(state_dim=224, action_dim=5)
        x = torch.randn(4, 224)
        actions, log_probs = policy.sample(x, deterministic=False)

        self.assertEqual(tuple(actions.shape), (4, 5))
        self.assertEqual(tuple(log_probs.shape), (4,))
        self.assertTrue(torch.all(torch.isfinite(actions)))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        highs = torch.tensor([ACTION_BOUNDS[c][1] for c in ACTION_COLUMNS], dtype=torch.float32)
        self.assertTrue(torch.all(actions >= 0.0))
        self.assertTrue(torch.all(actions <= highs + 1e-4))

        loss = -log_probs.mean() + actions.mean()
        loss.backward()
        grad_norms = [p.grad.abs().sum().item() for p in policy.parameters() if p.grad is not None]
        self.assertTrue(any(g > 0.0 for g in grad_norms))

        # ONNX export parity
        policy.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_file = os.path.join(tmpdir, "hurdle.onnx")
            dummy = torch.randn(1, 224)
            torch.onnx.export(
                policy,
                dummy,
                onnx_file,
                export_params=True,
                opset_version=14,
                input_names=["state"],
                output_names=["gate_logits", "mu_phys", "std_phys"],
                dynamo=False,
            )
            sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"state": dummy.numpy()})
            with torch.no_grad():
                pt_out = policy(dummy)
            for pt_t, o_arr in zip(pt_out, ort_out):
                self.assertLess(float(np.max(np.abs(pt_t.numpy() - o_arr))), 1e-3)

    def test_residual_bradley_terry_ordering(self):
        net = RewardNet(state_dim=10, action_dim=5)
        s1 = torch.randn(8, 10)
        a1 = torch.rand(8, 5)
        s2 = torch.randn(8, 10)
        a2 = torch.rand(8, 5)
        r1 = torch.full((8,), 3.0) + 0.5 * net(s1, a1)
        r2 = torch.full((8,), 0.5) + 0.5 * net(s2, a2)
        loss_win = bradley_terry_loss(r1.sum().unsqueeze(0), r2.sum().unsqueeze(0), torch.tensor([0]))
        loss_lose = bradley_terry_loss(r1.sum().unsqueeze(0), r2.sum().unsqueeze(0), torch.tensor([1]))
        self.assertLess(float(loss_win.item()), float(loss_lose.item()))

    def test_calql_logsumexp_and_mc_returns(self):
        mc = compute_empirical_mc_returns(
            rewards=torch.tensor([1.0, 2.0, 3.0]),
            dones=torch.tensor([0.0, 0.0, 1.0]),
            gamma=0.9,
        )
        expected_v0 = 1.0 + 0.9 * (2.0 + 0.9 * 3.0)
        self.assertAlmostEqual(float(mc[0].item()), expected_v0, places=4)

        q = RewardCritic(state_dim=16, action_dim=5)
        s = torch.randn(4, 16)
        a = torch.rand(4, 5)
        cands = torch.rand(4, 6, 5)
        target_q = torch.ones(4) * 2.0
        v_mu = torch.ones(4) * 5.0
        loss_lse, info = compute_cal_ql_loss(q, s, a, target_q, cands, v_mu, alpha_cql=1.0, use_logsumexp=True)
        self.assertGreater(float(loss_lse.item()), 0.0)
        self.assertIn("cql_loss", info)

    def test_sackl_failure_critic_undiscounted_target_and_dppo_guard(self):
        agent = SACKLAgent(state_dim=32, action_dim=5, num_q=2, ensemble_size_m=2)
        # Constant target_failure_ensemble output = 0.5 check isn't needed; verify zero-violation terminal target vs step violation
        bs = torch.randn(4, 32)
        ba = torch.rand(4, 5)
        bns = torch.randn(4, 32)
        bd = torch.tensor([0.0, 0.0, 1.0, 1.0])
        bc = torch.tensor([1.0, 0.0, 0.0, 1.0])
        bce_loss = agent.update_failure_critics(bs, ba, bns, bd, bc)
        self.assertGreater(bce_loss, 0.0)
        self.assertTrue(np.isfinite(bce_loss))

        # DPPO must reject mismatched state dimensions instead of silent zero padding
        dppo = DPPOAgent(state_dim=224, action_dim=5)
        loader = [{
            "tau_1_states": torch.randn(2, 4, 10),
            "tau_1_actions": torch.rand(2, 4, 5),
            "tau_2_states": torch.randn(2, 4, 10),
            "tau_2_actions": torch.rand(2, 4, 5),
            "preferred_idx": torch.tensor([0, 1]),
        }]
        with self.assertRaises(ValueError):
            dppo.train_epoch(loader)

    def test_scale_invariant_critic_and_analytical_hurdle_kl(self):
        torch.manual_seed(7)
        q_net = RewardCritic(state_dim=224, action_dim=5)
        f_net = FailureCriticEnsemble(state_dim=224, action_dim=5, ensemble_size=2)
        extreme_state = torch.randn(1, 224)
        extreme_state[0, 168:176] = 350000.0  # MQ O2 ppm telemetry magnitude

        a_zero = torch.zeros(1, 5)
        a_mol = torch.tensor([[0.0, 0.0, 0.0, 20.0, 0.0]])
        a_buf = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]])

        with torch.no_grad():
            q_diff_mol = abs(float(q_net(extreme_state, a_mol) - q_net(extreme_state, a_zero)))
            q_diff_buf = abs(float(q_net(extreme_state, a_buf) - q_net(extreme_state, a_zero)))
            f_diff_mol = abs(float(f_net(extreme_state, a_mol)[0] - f_net(extreme_state, a_zero)[0]))
        self.assertGreater(q_diff_mol, 1e-3)
        self.assertGreater(q_diff_buf, 1e-3)
        self.assertGreater(f_diff_mol, 1e-4)

        p1 = HurdlePolicy(state_dim=32, action_dim=5)
        p2 = HurdlePolicy(state_dim=32, action_dim=5)
        st = torch.randn(6, 32) * 100.0
        kl_self = p1.kl_divergence(p1, st)
        self.assertTrue(torch.allclose(kl_self, torch.zeros_like(kl_self), atol=1e-5))
        kl_cross = p1.kl_divergence(p2, st)
        self.assertTrue(torch.all(kl_cross >= 0.0))
        self.assertGreater(float(kl_cross.mean().item()), 0.0)
        kl_cross.mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in p1.parameters()))


if __name__ == "__main__":
    unittest.main()
