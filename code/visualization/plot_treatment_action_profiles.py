#!/usr/bin/env python3
"""
Treatment Action Profiler & Trajectory Visualizer (2.5 L POME Bioreactor)
========================================================================
Profiles canonical actuator interventions across the 16-day cultivation horizon:
  - [0] T1: Aeration Duration (0-6 hours/interval)
  - [4] T2: LED Photoperiod / Intensity (0-5000 lux)
  - [3] T3: Molasses Organic Carbon (0-20 mL/interval)
  - [1] T4: NaHCO3 Alkaline Buffer (0-5 g/interval)
  - [2] T5: Chlorella vulgaris Inoculum (0-100 mL/interval)
And links them to COD degradation, two-tier pH stabilization, and cumulative operating costs.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import ACTION_BOUNDS, ACTION_COLUMNS
from environment.reward_rig import compute_action_cost
from models.policy_sft import HurdlePolicy
from models.critics import RewardCritic, FailureCriticEnsemble
from experiments.evaluator import evaluate_policy_on_bioreactor, load_checkpoint_strict


def extract_action_profiles(seed: int = 42):
    """
    Strictly loads trained checkpoints and runs Safe RLHF, DPPO, and HCR trajectories.
    """
    device = torch.device("cpu")
    state_dim = 224
    action_dim = 5
    ckpt_dir = os.path.join(CODE_DIR, "checkpoints")

    safe_policy = HurdlePolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    reward_critic = RewardCritic(state_dim=state_dim, action_dim=action_dim).to(device)
    failure_ensemble = FailureCriticEnsemble(state_dim=state_dim, action_dim=action_dim, ensemble_size=4).to(device)

    load_checkpoint_strict(safe_policy, os.path.join(ckpt_dir, "safe_rlhf_policy.pt"), ["model_state_dict", "policy_state_dict"])
    load_checkpoint_strict(reward_critic, os.path.join(ckpt_dir, "safe_rlhf_reward_critic.pt"), ["model_state_dict", "q_state_dict", "critic_state_dict"])
    load_checkpoint_strict(failure_ensemble, os.path.join(ckpt_dir, "safe_rlhf_failure_ensemble.pt"), ["model_state_dict", "failure_ensemble_state_dict", "ensemble_state_dict"])

    dppo_policy = HurdlePolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    load_checkpoint_strict(dppo_policy, os.path.join(ckpt_dir, "dppo_policy.pt"), ["model_state_dict", "policy_state_dict"])

    hcr_policy = HurdlePolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    load_checkpoint_strict(hcr_policy, os.path.join(ckpt_dir, "sft_policy.pt"), ["model_state_dict", "policy_state_dict"])

    print("[Profiler] Running Safe RLHF trajectory...")
    res_safe = evaluate_policy_on_bioreactor(
        policy=safe_policy,
        reward_critic=reward_critic,
        failure_ensemble=failure_ensemble,
        num_candidates=16,
        lambda_dual=2.0,
        safety_budget=0.10,
        gate_threshold=0.50,
        device=device,
        algorithm="safe_rlhf",
        seed=seed,
    )
    
    print("\n[Profiler] Running DPPO trajectory...")
    res_dppo = evaluate_policy_on_bioreactor(
        policy=dppo_policy,
        reward_critic=reward_critic,
        failure_ensemble=failure_ensemble,
        num_candidates=1,
        lambda_dual=0.0,
        safety_budget=0.10,
        gate_threshold=0.50,
        device=device,
        algorithm="dppo",
        seed=seed,
    )
    
    print("\n[Profiler] Running HCR trajectory...")
    res_hcr = evaluate_policy_on_bioreactor(
        policy=hcr_policy,
        reward_critic=reward_critic,
        failure_ensemble=failure_ensemble,
        num_candidates=1,
        lambda_dual=0.0,
        safety_budget=0.50,
        gate_threshold=0.50,
        device=device,
        algorithm="hcr",
        seed=seed,
    )

    return res_safe, res_dppo, res_hcr


def plot_treatment_profiles(res_safe, res_dppo, res_hcr, output_path: str):
    """
    Renders 5-panel actuator time series + environmental state response.
    Handles variable-length trajectories seamlessly.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dt_days = 0.25 # 6.0 hours

    def get_time_arrays(res):
        n_states = len(res["cod_trajectory"])
        t_states = np.linspace(0, (n_states - 1) * dt_days, n_states)
        n_actions = len(res["action_history"])
        t_actions = np.linspace(dt_days, n_actions * dt_days, n_actions)
        act = np.array(res["action_history"])
        return t_states, t_actions, act

    t_s_safe, t_a_safe, act_safe = get_time_arrays(res_safe)
    t_s_dppo, t_a_dppo, act_dppo = get_time_arrays(res_dppo)
    t_s_hcr,  t_a_hcr,  act_hcr  = get_time_arrays(res_hcr)

    # Canonical physical actuator ranges (ACTION_COLUMNS, ALREADY in physical units from evaluator.py):
    # [0] T1: Aeration (0-6 h), [4] T2: Light (0-5000 lux), [3] T3: Molasses (0-20 mL),
    # [1] T4: NaHCO3 (0-5 g), [2] T5: Inoculum (0-100 mL)
    t1_safe = act_safe[:, 0]
    t1_dppo = act_dppo[:, 0]
    t1_hcr  = act_hcr[:, 0]

    t2_safe = act_safe[:, 4]
    t2_dppo = act_dppo[:, 4]
    t2_hcr  = act_hcr[:, 4]

    t3_safe = act_safe[:, 3]
    t3_dppo = act_dppo[:, 3]
    t3_hcr  = act_hcr[:, 3]

    t4_safe = act_safe[:, 1]
    t4_dppo = act_dppo[:, 1]
    t4_hcr  = act_hcr[:, 1]

    t5_safe = act_safe[:, 2]
    t5_dppo = act_dppo[:, 2]
    t5_hcr  = act_hcr[:, 2]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(15, 16), dpi=300)
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0], hspace=0.35)

    c_safe = "#1b9e77"   # Teal / Green (Proposed)
    c_dppo = "#7570b3"   # Purple (LAPSE DPPO)
    c_hcr  = "#d95f02"   # Orange / Red (Baseline)

    # -------------------------------------------------------------
    # Panel 1: Environmental Response (COD & pH)
    # -------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0_ph = ax0.twinx()

    l1 = ax0.plot(t_s_safe, res_safe["cod_trajectory"], color=c_safe, linewidth=2.5, label="Safe RLHF COD (mg/L)")
    l2 = ax0.plot(t_s_dppo, res_dppo["cod_trajectory"], color=c_dppo, linewidth=2.0, linestyle="--", label="DPPO COD (mg/L)")
    l3 = ax0.plot(t_s_hcr,  res_hcr["cod_trajectory"],  color=c_hcr,  linewidth=2.0, linestyle=":", label="HCR Baseline COD (mg/L)")
    
    l_reg = ax0.axhline(350.0, color="#e41a1c", linestyle="-.", linewidth=1.5, alpha=0.9, label="PerMenLHK COD Limit (350 mg/L)")

    # Compliance Day Annotations
    comp_safe = res_safe["compliance_day"]
    comp_dppo = res_dppo["compliance_day"]
    if comp_safe < 16.0:
        ax0.axvline(comp_safe, color=c_safe, linestyle=":", alpha=0.7)
        ax0.annotate(f"Safe RLHF: {comp_safe:.1f} Hari", xy=(comp_safe, 350), xytext=(comp_safe + 0.3, 800),
                     arrowprops=dict(arrowstyle="->", color=c_safe, lw=1.2), fontweight="bold", color=c_safe)
    if comp_dppo < 16.0:
        ax0.axvline(comp_dppo, color=c_dppo, linestyle=":", alpha=0.7)
        ax0.annotate(f"DPPO: {comp_dppo:.1f} Hari", xy=(comp_dppo, 350), xytext=(comp_dppo + 0.3, 1400),
                     arrowprops=dict(arrowstyle="->", color=c_dppo, lw=1.2), fontweight="bold", color=c_dppo)

    # pH Trajectories on twin axis
    ax0_ph.plot(t_s_safe, res_safe["ph_trajectory"], color=c_safe, alpha=0.45, linewidth=1.4)
    ax0_ph.plot(t_s_dppo, res_dppo["ph_trajectory"], color=c_dppo, alpha=0.45, linewidth=1.4, linestyle="--")
    ax0_ph.plot(t_s_hcr,  res_hcr["ph_trajectory"],  color=c_hcr,  alpha=0.45, linewidth=1.4, linestyle=":")
    ax0_ph.axhspan(6.0, 9.0, color="#4daf4a", alpha=0.08, label="PerMenLH pH Discharge Band [6.0, 9.0]")
    ax0_ph.axhline(5.5, color="#ff7f00", linestyle=":", linewidth=1.2, alpha=0.85, label="Biological Survival pH [5.5, 10.5]")
    ax0_ph.axhline(10.5, color="#ff7f00", linestyle=":", linewidth=1.2, alpha=0.85)

    ax0.set_title("(A) Bioprocess State Trajectory: COD Bioremediation & Regulatory Compliance ($T_{comp}$)", fontsize=13, fontweight="bold", pad=8)
    ax0.set_ylabel("COD Concentration (mg/L)", fontsize=11, fontweight="bold")
    ax0_ph.set_ylabel("pH Level", fontsize=11, fontweight="bold", color="#555555")
    ax0.set_xlim(0, 16)
    ax0.set_ylim(0, max(res_safe["cod_trajectory"][0] * 1.1, 4500))
    ax0_ph.set_ylim(4.8, 11.2)

    lines = l1 + l2 + l3 + [l_reg]
    labels = [l.get_label() for l in lines]
    ax0.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: T1 Aeration Duration (Hours/Interval)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t_a_safe, t1_safe, color=c_safe, linewidth=2.0, label="Safe RLHF (Adaptive Aeration)")
    ax1.plot(t_a_dppo, t1_dppo, color=c_dppo, linewidth=1.8, linestyle="--", label="DPPO")
    ax1.plot(t_a_hcr,  t1_hcr,  color=c_hcr,  linewidth=1.8, linestyle=":", label="HCR Operator")
    ax1.fill_between(t_a_safe, 0, t1_safe, color=c_safe, alpha=0.15)
    ax1.set_title("(B) Actuator T1: Aeration Duration (Hours per 6h Interval, Max: 6.0h)", fontsize=12, fontweight="bold", pad=6)
    ax1.set_ylabel("Aeration (h)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 6.2)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: T2 LED Photoperiod & Intensity (Lux)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.plot(t_a_safe, t2_safe, color=c_safe, linewidth=2.0, label="Safe RLHF")
    ax2.plot(t_a_dppo, t2_dppo, color=c_dppo, linewidth=1.8, linestyle="--", label="DPPO")
    ax2.plot(t_a_hcr,  t2_hcr,  color=c_hcr,  linewidth=1.8, linestyle=":", label="HCR Operator")
    ax2.fill_between(t_a_safe, 0, t2_safe, color=c_safe, alpha=0.15)
    ax2.set_title("(C) Actuator T2: LED Lighting Intensity (Lux, Max: 5,000 Lux)", fontsize=12, fontweight="bold", pad=6)
    ax2.set_ylabel("Light (Lux)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 5500)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 4: T3 (Molasses) & T4 (NaHCO3 Buffer) Dosing
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3_b = ax3.twinx()

    p_t4_safe = ax3.plot(t_a_safe, t4_safe, color=c_safe, linewidth=2.0, label="Safe RLHF T4 NaHCO3 (g)")
    p_t4_dppo = ax3.plot(t_a_dppo, t4_dppo, color=c_dppo, linewidth=1.8, linestyle="--", label="DPPO T4 NaHCO3 (g)")
    p_t4_hcr  = ax3.plot(t_a_hcr,  t4_hcr,  color=c_hcr,  linewidth=1.8, linestyle=":", label="HCR T4 NaHCO3 (g)")

    p_t3_safe = ax3_b.plot(t_a_safe, t3_safe, color="#386cb0", linewidth=1.5, alpha=0.8, linestyle="-.", label="Safe RLHF T3 Molasses (mL)")

    ax3.set_title("(D) Actuator T4: Alkaline Buffer (NaHCO3, g) & T3: Carbon Booster (Molasses, mL)", fontsize=12, fontweight="bold", pad=6)
    ax3.set_ylabel("NaHCO3 Buffer (g)", fontsize=10, fontweight="bold")
    ax3_b.set_ylabel("Molasses (mL)", fontsize=10, fontweight="bold", color="#386cb0")
    ax3.set_ylim(0, 6.0)
    ax3_b.set_ylim(0, 23)

    lines3 = p_t4_safe + p_t4_dppo + p_t4_hcr + p_t3_safe
    labels3 = [l.get_label() for l in lines3]
    ax3.legend(lines3, labels3, loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # Panel 5: T5 Inoculum & Economic Intervention Cost
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    
    def compute_cum_cost(act):
        step_cost = np.array([compute_action_cost(step_a) for step_a in act], dtype=float)
        return np.cumsum(step_cost)

    cost_safe = compute_cum_cost(act_safe)
    cost_dppo = compute_cum_cost(act_dppo)
    cost_hcr  = compute_cum_cost(act_hcr)

    ax4.plot(t_a_safe, cost_safe, color=c_safe, linewidth=2.2, label=f"Safe RLHF Total: Rp {cost_safe[-1]:,.0f}")
    ax4.plot(t_a_dppo, cost_dppo, color=c_dppo, linewidth=2.0, linestyle="--", label=f"DPPO Total: Rp {cost_dppo[-1]:,.0f}")
    ax4.plot(t_a_hcr,  cost_hcr,  color=c_hcr,  linewidth=2.0, linestyle=":", label=f"HCR Operator Total: Rp {cost_hcr[-1]:,.0f}")

    # Microalgae Inoculation spikes (T5)
    ax4_t5 = ax4.twinx()
    ax4_t5.stem(t_a_safe, t5_safe, linefmt="C2-", markerfmt="C2o", basefmt=" ", label="Safe RLHF T5 Inoculum (mL)")
    ax4_t5.set_ylabel("Inoculum T5 (mL)", fontsize=10, fontweight="bold", color="C2")
    ax4_t5.set_ylim(0, 115)

    ax4.set_title("(E) Cumulative Operational Cost (Rp) & Bioaugmentation Interventions (T5 Inoculum)", fontsize=12, fontweight="bold", pad=6)
    ax4.set_xlabel("Waktu Kultivasi Bioreaktor (Hari)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Total Biaya (IDR)", fontsize=10, fontweight="bold")
    ax4.set_ylim(0, max(cost_safe[-1], cost_dppo[-1], cost_hcr[-1]) * 1.25)
    ax4.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Profiler] Treatment profile visualization saved to '{output_path}'.")


def generate_action_breakdown_csv(res_safe, res_dppo, res_hcr, csv_path: str):
    """
    Exports a table breaking down actuator dosages across 4 cultivation phases.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    phases = [
        ("Fase 1: Inokulasi & Aklimatisasi (Hari 0-4)", 0, 16),
        ("Fase 2: Pertumbuhan Eksponensial (Hari 4-8)", 16, 32),
        ("Fase 3: Stabilisasi & Remediasi (Hari 8-12)", 32, 48),
        ("Fase 4: Polishing & Regulasi (Hari 12-16)", 48, 64),
    ]

    act_safe = np.array(res_safe["action_history"])
    act_dppo = np.array(res_dppo["action_history"])
    act_hcr  = np.array(res_hcr["action_history"])

    records = []
    for phase_name, start_idx, end_idx in phases:
        for name, act in [("Safe RLHF (Proposed)", act_safe), ("DPPO (LAPSE)", act_dppo), ("HCR (Operator Baseline)", act_hcr)]:
            if start_idx < len(act):
                sub = act[start_idx:min(end_idx, len(act))]
                records.append({
                    "Fase": phase_name,
                    "Algoritma": name,
                    "T1 Aerasi Rata2 (jam/step)": round(float(np.mean(sub[:, 0])), 2),
                    "T2 Lampu Rata2 (Lux)": round(float(np.mean(sub[:, 4])), 0),
                    "T3 Molase Total (mL)": round(float(np.sum(sub[:, 3])), 1),
                    "T4 NaHCO3 Total (g)": round(float(np.sum(sub[:, 1])), 1),
                    "T5 Inokulum Total (mL)": round(float(np.sum(sub[:, 2])), 1),
                })
            else:
                records.append({
                    "Fase": phase_name,
                    "Algoritma": name,
                    "T1 Aerasi Rata2 (jam/step)": 0.0,
                    "T2 Lampu Rata2 (Lux)": 0.0,
                    "T3 Molase Total (mL)": 0.0,
                    "T4 NaHCO3 Total (g)": 0.0,
                    "T5 Inokulum Total (mL)": 0.0,
                })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[Profiler] Treatment breakdown CSV exported to '{csv_path}'.")
    return df


if __name__ == "__main__":
    out_fig = os.path.join(CODE_DIR, "evaluation", "figures", "treatment_action_profiles_lapse.png")
    out_csv = os.path.join(CODE_DIR, "evaluation", "treatment_action_breakdown.csv")
    
    res_safe, res_dppo, res_hcr = extract_action_profiles(seed=42)
    plot_treatment_profiles(res_safe, res_dppo, res_hcr, out_fig)
    generate_action_breakdown_csv(res_safe, res_dppo, res_hcr, out_csv)
