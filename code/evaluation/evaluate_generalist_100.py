#!/usr/bin/env python3
"""
Generalist Evaluation Across 100 Unseen Bioprocess Batches (2.5 L POME Reactor)
================================================================================
Evaluates Safe RLHF vs DPPO vs HCR Operator vs Unconstrained SAC across 100
randomly sampled POME batches spanning wide variations in initial pollutant
loads (COD 2,500-6,500 mg/L, pH 6.2-8.2, DO 0.3-2.8 mg/L, Biomass 0.08-0.45 g/L)
via genuine closed-loop rollouts in POMEReactorEnv (no post-hoc linear rescaling).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import DECISION_HORIZON_HOURS
from data.real_data_loader import OfflineDataset
from models.policy_sft import HurdlePolicy
from models.critics import RewardCritic, FailureCriticEnsemble
from experiments.evaluator import (
    evaluate_policy_on_bioreactor,
    load_checkpoint_strict,
    train_algorithm_agent,
)


def generate_100_initial_batches(seed: int = 42):
    """
    Generates 100 realistic initial conditions matching empirical K1-K16 variability.
    """
    rng = np.random.default_rng(seed)
    batches = []
    for b in range(100):
        initial_cod = float(np.clip(rng.normal(4200.0, 750.0), 2500.0, 6500.0))
        initial_ph = float(np.clip(rng.normal(7.2, 0.45), 6.2, 8.2))
        initial_do = float(np.clip(rng.normal(1.2, 0.4), 0.3, 2.8))
        initial_x = float(np.clip(rng.normal(0.25, 0.08), 0.08, 0.45))

        batches.append({
            "batch_id": b + 1,
            "initial_cod": initial_cod,
            "initial_ph": initial_ph,
            "initial_do": initial_do,
            "initial_x": initial_x,
        })
    return batches


def _load_or_train_sac(device: torch.device) -> HurdlePolicy:
    """Strictly load standalone Unconstrained SAC policy, or train it authentically if missing."""
    sac_ckpt = os.path.join(CODE_DIR, "checkpoints", "sac_policy.pt")
    sac_policy = HurdlePolicy(state_dim=224, action_dim=5).to(device)
    if os.path.exists(sac_ckpt):
        load_checkpoint_strict(sac_policy, sac_ckpt, ["model_state_dict", "policy_state_dict"])
        return sac_policy

    print("[Generalist 100] checkpoints/sac_policy.pt not found — training authentic SAC baseline...")
    d_hist = OfflineDataset(pt_path=os.path.join(CODE_DIR, "data", "d_hist.pt"))
    sac_policy, _, _, _ = train_algorithm_agent(
        algorithm="sac",
        dataset=d_hist,
        params={"use_vision": True, "cost_weight": 1.0},
        state_dim=224,
        device=device,
        quick_mode=False,
        seed=42,
    )
    os.makedirs(os.path.dirname(sac_ckpt), exist_ok=True)
    torch.save({"model_state_dict": sac_policy.state_dict()}, sac_ckpt)
    return sac_policy


def run_generalist_100(batches):
    """
    Evaluates Safe RLHF, DPPO, HCR, and Unconstrained SAC across all 100 unseen batches.
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

    sac_policy = _load_or_train_sac(device)

    algorithms = [
        ("Safe RLHF (Proposed)", safe_policy, "safe_rlhf", 16, 2.0, 0.10, 0.50),
        ("DPPO (LAPSE Reference)", dppo_policy, "dppo", 1, 0.0, 0.10, 0.50),
        ("HCR (Operator Baseline)", hcr_policy, "hcr", 1, 0.0, 0.50, 0.50),
        ("Standard SAC (Unconstrained)", sac_policy, "sac", 1, 0.0, 0.10, 0.50),
    ]

    all_results = []
    print(f"\n[Generalist 100] Starting authentic closed-loop evaluation across {len(batches)} random POME batches...")

    for alg_name, policy, alg_key, num_cands, lam_dual, s_budget, g_thresh in algorithms:
        print(f"\n  Evaluating {alg_name} across {len(batches)} batches:")

        for idx, b in enumerate(batches):
            seed = 1000 + b["batch_id"]
            init_state = {
                "cod": b["initial_cod"],
                "ph": b["initial_ph"],
                "dissolved_oxygen": b["initial_do"],
                "biomass": b["initial_x"],
            }

            res = evaluate_policy_on_bioreactor(
                policy=policy,
                reward_critic=reward_critic,
                failure_ensemble=failure_ensemble,
                num_candidates=num_cands,
                lambda_dual=lam_dual,
                safety_budget=s_budget,
                gate_threshold=g_thresh,
                device=device,
                decision_horizon_hours=DECISION_HORIZON_HOURS,
                algorithm=alg_key,
                seed=seed,
                initial_state=init_state,
            )

            success = bool(res["permenlh_compliant"]) and (res["violation_rate_pct"] == 0.0)

            all_results.append({
                "Batch_ID": b["batch_id"],
                "Algoritma": alg_name,
                "Initial_COD_mg_L": round(float(res["initial_cod_mg_l"]), 1),
                "Final_COD_mg_L": round(float(res["final_cod_mg_l"]), 1),
                "Final_BOD_mg_L": round(float(res["final_bod_mg_l"]), 1),
                "Final_TSS_mg_L": round(float(res["final_tss_mg_l"]), 1),
                "Final_NH3_mg_L": round(float(res["final_nh3_mg_l"]), 1),
                "Final_pH": round(float(res["final_ph"]), 2),
                "COD_Removal_Pct": round(float(res["cod_removal_pct"]), 2),
                "Compliance_Day": round(float(res["compliance_day"]), 2),
                "PerMenLH_Compliant": bool(res["permenlh_compliant"]),
                "Is_Compliant": success,
                "Violation_Rate_Pct": round(float(res["violation_rate_pct"]), 2),
                "Total_Cost_IDR": round(float(res["total_cost_idr"]), 1),
                "Pareto_Score": round(float(res["pareto_score"]), 2),
            })

            if (idx + 1) % 25 == 0:
                print(
                    f"    Batch {idx+1:03d}/{len(batches)} | COD: {res['final_cod_mg_l']:6.1f} mg/L | "
                    f"pH: {res['final_ph']:4.2f} | T_comp: {res['compliance_day']:4.1f}d | Success: {success}"
                )

    return pd.DataFrame(all_results)


def summarize_and_plot(df: pd.DataFrame, out_csv: str, out_summary_csv: str, out_fig: str):
    """
    Exports CSV results and generates publication-quality 4-panel robustness plot.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[Generalist 100] Full 100-batch metrics saved to '{out_csv}'.")

    summary = df.groupby("Algoritma").agg(
        Success_Rate_Pct=("Is_Compliant", lambda x: round(float(np.mean(x) * 100.0), 1)),
        PerMenLH_Compliance_Rate_Pct=("PerMenLH_Compliant", lambda x: round(float(np.mean(x) * 100.0), 1)),
        Mean_Compliance_Day=("Compliance_Day", lambda x: round(float(np.mean(x)), 2)),
        Std_Compliance_Day=("Compliance_Day", lambda x: round(float(np.std(x)), 2)),
        Mean_COD_Removal_Pct=("COD_Removal_Pct", lambda x: round(float(np.mean(x)), 2)),
        Std_COD_Removal_Pct=("COD_Removal_Pct", lambda x: round(float(np.std(x)), 2)),
        Mean_Final_COD_mg_L=("Final_COD_mg_L", lambda x: round(float(np.mean(x)), 1)),
        Mean_Final_pH=("Final_pH", lambda x: round(float(np.mean(x)), 2)),
        Mean_Violation_Rate_Pct=("Violation_Rate_Pct", lambda x: round(float(np.mean(x)), 2)),
        Mean_Total_Cost_IDR=("Total_Cost_IDR", lambda x: round(float(np.mean(x)), 0)),
        Mean_Pareto_Score=("Pareto_Score", lambda x: round(float(np.mean(x)), 2)),
        Std_Pareto_Score=("Pareto_Score", lambda x: round(float(np.std(x)), 2)),
    ).reset_index()

    summary.to_csv(out_summary_csv, index=False)
    print(f"[Generalist 100] Summary statistics saved to '{out_summary_csv}'.")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    algorithms = summary["Algoritma"].tolist()
    colors = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]
    palette = dict(zip(algorithms, colors))

    # 1. Success Rate Bar Chart
    ax1 = axes[0, 0]
    bars = ax1.bar(
        summary["Algoritma"],
        summary["Success_Rate_Pct"],
        color=[palette[a] for a in summary["Algoritma"]],
        width=0.55,
        edgecolor="black",
        alpha=0.85,
    )
    ax1.set_title("(A) Full PerMenLH No. 5/2014 + Safe Operation Across 100 Batches", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Success Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 112)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels([a.replace(" ", "\n") for a in algorithms], fontsize=9)

    # 2. Days to Compliance Distribution
    ax2 = axes[0, 1]
    box_data = [df[df["Algoritma"] == a]["Compliance_Day"].values for a in algorithms]
    bp = ax2.boxplot(box_data, patch_artist=True, tick_labels=[a.replace(" ", "\n") for a in algorithms], widths=0.5)
    for patch, alg in zip(bp["boxes"], algorithms):
        patch.set_facecolor(palette[alg])
        patch.set_alpha(0.7)
    ax2.set_title("(B) Bioremediation Duration Distribution ($T_{comp}$, Days)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Days to PerMenLH Compliance", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 17)

    # 3. Final Effluent COD Distribution
    ax3 = axes[1, 0]
    box_cod = [df[df["Algoritma"] == a]["Final_COD_mg_L"].values for a in algorithms]
    bp_cod = ax3.boxplot(box_cod, patch_artist=True, tick_labels=[a.replace(" ", "\n") for a in algorithms], widths=0.5)
    for patch, alg in zip(bp_cod["boxes"], algorithms):
        patch.set_facecolor(palette[alg])
        patch.set_alpha(0.7)
    ax3.axhline(350.0, color="#e41a1c", linestyle="--", linewidth=1.5, label="Baku Mutu PerMenLH (350 mg/L)")
    ax3.set_title("(C) Final Effluent COD Distribution vs Statutory Limit", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Final Effluent COD (mg/L)", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # 4. Pareto Score Distribution
    ax4 = axes[1, 1]
    box_pareto = [df[df["Algoritma"] == a]["Pareto_Score"].values for a in algorithms]
    bp_p = ax4.boxplot(box_pareto, patch_artist=True, tick_labels=[a.replace(" ", "\n") for a in algorithms], widths=0.5)
    for patch, alg in zip(bp_p["boxes"], algorithms):
        patch.set_facecolor(palette[alg])
        patch.set_alpha(0.7)
    ax4.set_title("(D) Overall Bioprocess Pareto Score ($J_{Pareto}$) Across 100 Batches", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Pareto Score [0 - 100]", fontsize=10, fontweight="bold")
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Generalist 100] Figure saved to '{out_fig}'.")


if __name__ == "__main__":
    batches = generate_100_initial_batches(seed=42)
    df_results = run_generalist_100(batches)

    out_csv = os.path.join(CODE_DIR, "evaluation", "generalist_100_batches_metrics.csv")
    out_sum = os.path.join(CODE_DIR, "evaluation", "generalist_summary_table.csv")
    out_fig = os.path.join(CODE_DIR, "evaluation", "figures", "generalist_100_batches_comparison.png")

    summarize_and_plot(df_results, out_csv, out_sum, out_fig)
