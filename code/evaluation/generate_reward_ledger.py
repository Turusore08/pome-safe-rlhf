#!/usr/bin/env python3
"""
Step-by-Step Reward and Punishment Ledger Generator (2.5 L POME Bioreactor)
===========================================================================
Generates a transparent 64-step operational ledger directly from the closed-loop
POMEReactorEnv simulation using strictly loaded trained Safe RLHF checkpoints:
  - Environmental State: COD (mg/L), BOD (mg/L), TSS (mg/L), NH3 (mg/L), pH, DO (mg/L), Biomass (g/L)
  - Canonical Actuators (config.ACTION_COLUMNS):
      [0] T1: Aeration (0-6 h)
      [1] T4: NaHCO3 (0-5 g)
      [2] T5: Microalgae Inoculum (0-100 mL)
      [3] T3: Molasses (0-20 mL)
      [4] T2: LED Light Intensity (0-5000 Lux)
  - Exact State Reward r_rig = (E + B + Q) - (CT + CO) from environment.reward_rig
  - Tier-2 Biological Safety Violation Penalty (-50 * I_viol) from evaluation.metrics
  - Residual Neural Preference Score r_pref & Hybrid Reward r_hybrid from TriRewardEvaluator
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import ACTION_BOUNDS, ACTION_COLUMNS, BIOLOGICAL_SAFETY_LIMITS, DECISION_HORIZON_HOURS
from models.policy_sft import HurdlePolicy
from models.critics import RewardCritic, FailureCriticEnsemble
from environment.reward_rig import (
    compute_state_reward,
    compute_material_cost,
    compute_operational_cost,
    compute_action_cost,
)
from evaluation.metrics import compute_step_violation_magnitude, verify_permenlhk_compliance
from preference.train_reward import TriRewardEvaluator
from experiments.evaluator import evaluate_policy_on_bioreactor, load_checkpoint_strict


def _resolve_reward_ensemble_path() -> str:
    """Resolve trained preference reward ensemble checkpoint or fail strictly."""
    for fname in ("reward_ensemble.pt", "reward_ensemble_Ls_comb.pt"):
        cand = os.path.join(CODE_DIR, "checkpoints", fname)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "[Strict Ledger Error] Reward ensemble checkpoint ('reward_ensemble.pt' or "
        "'reward_ensemble_Ls_comb.pt') tidak ditemukan. Jalankan pelatihan preferensi terlebih dahulu."
    )


def generate_ledger(output_csv: str, seed: int = 42) -> pd.DataFrame:
    device = torch.device("cpu")
    ckpt_dir = os.path.join(CODE_DIR, "checkpoints")

    pref_ckpt = _resolve_reward_ensemble_path()
    tri_reward = TriRewardEvaluator(ensemble_checkpoint_path=pref_ckpt, device=device)
    if tri_reward.ensemble is None:
        raise RuntimeError(f"[Strict Ledger Error] Gagal memuat ensemble dari '{pref_ckpt}'.")

    # Strictly load trained Safe RLHF policy & critics (refuse random weights)
    state_dim = 224
    action_dim = 5
    safe_policy = HurdlePolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    reward_critic = RewardCritic(state_dim=state_dim, action_dim=action_dim).to(device)
    failure_ensemble = FailureCriticEnsemble(state_dim=state_dim, action_dim=action_dim, ensemble_size=4).to(device)

    load_checkpoint_strict(safe_policy, os.path.join(ckpt_dir, "safe_rlhf_policy.pt"), ["model_state_dict", "policy_state_dict"])
    load_checkpoint_strict(reward_critic, os.path.join(ckpt_dir, "safe_rlhf_reward_critic.pt"), ["model_state_dict", "q_state_dict", "critic_state_dict"])
    load_checkpoint_strict(failure_ensemble, os.path.join(ckpt_dir, "safe_rlhf_failure_ensemble.pt"), ["model_state_dict", "failure_ensemble_state_dict", "ensemble_state_dict"])

    print("[Ledger] Executing Safe RLHF closed-loop episode on 2.5 L POMEReactorEnv...")
    res = evaluate_policy_on_bioreactor(
        policy=safe_policy,
        reward_critic=reward_critic,
        failure_ensemble=failure_ensemble,
        num_candidates=16,
        lambda_dual=2.0,
        safety_budget=0.10,
        gate_threshold=0.50,
        device=device,
        decision_horizon_hours=DECISION_HORIZON_HOURS,
        algorithm="safe_rlhf",
        seed=seed,
    )

    state_hist = res["state_history"]
    act_low = np.array([ACTION_BOUNDS[c][0] for c in ACTION_COLUMNS], dtype=np.float32)
    act_high = np.array([ACTION_BOUNDS[c][1] for c in ACTION_COLUMNS], dtype=np.float32)
    act_hist = np.clip(np.asarray(res["action_history"], dtype=np.float32), act_low, act_high)

    ph_bio_min = BIOLOGICAL_SAFETY_LIMITS["ph"]["min"]
    ph_bio_max = BIOLOGICAL_SAFETY_LIMITS["ph"]["max"]

    records = []
    cum_cost_action = 0.0
    cum_cost_total = 0.0

    for step in range(len(act_hist)):
        day = (step + 1) * (DECISION_HORIZON_HOURS / 24.0)
        s_curr = state_hist[step]
        s_next = state_hist[step + 1]
        act = act_hist[step]

        # Canonical physical action order: [aeration, sodium_bicarbonate, microalgae_inoculum, molasses, light_intensity]
        t1_aeration_h = float(act[0])
        t4_nahco3_g = float(act[1])
        t5_inoculum_ml = float(act[2])
        t3_molasses_ml = float(act[3])
        t2_light_lux = float(act[4])

        # Exact costs from environment.reward_rig
        ct_idr = compute_material_cost(act)
        co_idr = compute_operational_cost(act, dt_hours=DECISION_HORIZON_HOURS, include_overhead=True)
        step_action_cost = compute_action_cost(act)
        step_total_cost = ct_idr + co_idr
        cum_cost_action += step_action_cost
        cum_cost_total += step_total_cost

        # Exact state reward decomposition from environment.reward_rig
        s_curr_for_rig = dict(s_curr)
        s_curr_for_rig["light_intensity"] = t2_light_lux
        s_next_for_rig = dict(s_next)
        s_next_for_rig["light_intensity"] = t2_light_lux
        r_state_base, comp = compute_state_reward(
            s_curr_dict=s_curr_for_rig,
            a_t=act,
            s_next_dict=s_next_for_rig,
            biomass_curr=float(s_curr["biomass"]),
            biomass_next=float(s_next["biomass"]),
            dt_hours=DECISION_HORIZON_HOURS,
        )

        # Tier-2 Biological Safety Violation & Penalty
        v_flag, d_mag = compute_step_violation_magnitude(s_next)
        p_safety = 50.0 * float(v_flag)

        # Residual Bradley-Terry preference score and hybrid reward
        r_rig_eval, r_pref, r_hybrid = tri_reward.evaluate(s_curr_for_rig, act, next_state_dict=s_next_for_rig)

        # Tier-1 PerMenLH No. 5/2014 step effluent check
        step_comp = verify_permenlhk_compliance([s_next], step_hours=DECISION_HORIZON_HOURS)
        is_bio_safe = (
            v_flag == 0
            and ph_bio_min <= float(s_next["ph"]) <= ph_bio_max
        )
        if step_comp["compliance_check"]:
            status_reg = "PATUH_PERMENLH"
        elif is_bio_safe:
            status_reg = "AMAN_BIOLOGIS_PROSES"
        else:
            status_reg = "PELANGGARAN_KESELAMATAN"

        records.append({
            "Step": step + 1,
            "Hari": round(day, 2),
            "COD_mg_L": round(float(s_curr["cod"]), 1),
            "COD_Next_mg_L": round(float(s_next["cod"]), 1),
            "BOD_Next_mg_L": round(float(s_next["bod"]), 1),
            "TSS_Next_mg_L": round(float(s_next["total_suspended_solids"]), 1),
            "NH3_Next_mg_L": round(float(s_next["dissolved_ammonia"]), 1),
            "pH": round(float(s_curr["ph"]), 2),
            "pH_Next": round(float(s_next["ph"]), 2),
            "DO_Next_mg_L": round(float(s_next["dissolved_oxygen"]), 2),
            "Biomass_Next_g_L": round(float(s_next["biomass"]), 3),
            "T1_Aerasi_jam": round(t1_aeration_h, 2),
            "T2_Lampu_lux": round(t2_light_lux, 1),
            "T3_Molase_mL": round(t3_molasses_ml, 2),
            "T4_NaHCO3_g": round(t4_nahco3_g, 2),
            "T5_Inokulum_mL": round(t5_inoculum_ml, 2),
            "Biaya_Aksi_Step_IDR": round(step_action_cost, 1),
            "Biaya_Total_Step_IDR": round(step_total_cost, 1),
            "Biaya_Kumulatif_Aksi_IDR": round(cum_cost_action, 0),
            "Biaya_Kumulatif_Total_IDR": round(cum_cost_total, 0),
            "Reward_E_norm": round(comp["E_norm"], 4),
            "Reward_B_norm": round(comp["B_norm"], 4),
            "Reward_Q_norm": round(comp["Q_norm"], 4),
            "Cost_CT_norm": round(comp["CT_norm"], 4),
            "Cost_CO_norm": round(comp["CO_norm"], 4),
            "Penalty_Keselamatan": round(p_safety, 2),
            "Violation_Mag": round(float(d_mag), 4),
            "Reward_State_Base": round(float(r_state_base), 4),
            "Reward_Rigorous_r_rig": round(float(r_rig_eval), 4),
            "Reward_Preference_r_pref": round(float(r_pref), 4),
            "Reward_Hybrid_r_hybrid": round(float(r_hybrid), 4),
            "Env_Step_Reward": round(float(res["step_rig_rewards"][step]), 4),
            "Status_Regulasi": status_reg,
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Ledger] {len(df)}-step transparent ledger exported to '{output_csv}'.")
    return df


if __name__ == "__main__":
    out_csv = os.path.join(CODE_DIR, "evaluation", "reward_punishment_ledger_sample.csv")
    generate_ledger(out_csv, seed=42)
