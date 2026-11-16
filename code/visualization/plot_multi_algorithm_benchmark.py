"""
Publication Visualization for Multi-Algorithm RL Benchmark & Reward Ablations.
Replicates Figure 4 and Table 3 of Gerold, Brandner, & Lucia (ESCAPE 36 / LAPSE 2026).

Generates a comprehensive 4-panel publication figure:
1. Panel A (Closed-Loop COD Degradation Trajectories):
   - Compares COD remediation over 16 days (64 steps) for:
     * Safe RLHF (Proposed)
     * DPPO (LAPSE Reference)
     * Lagrangian-SAC (M=1)
     * Standard SAC (Unconstrained)
     * HCR Operator Heuristic
   - Overlays PerMenLHK No. 5/2014 threshold (COD = 350 mg/L).
2. Panel B (Closed-Loop Photobioreactor pH Regulation Trajectories):
   - Compares broth pH trajectories against safety envelope [6.0, 9.2].
   - Highlights how Safe RLHF stays strictly within safe boundaries, whereas
     unconstrained SAC and DPPO suffer constraint boundary excursions.
3. Panel C (Safety vs Performance Tradeoff: G_mean vs n_cv):
   - Scatter/bar plot mapping Cumulative Return G_mean vs Total Violations n_cv
   - Displays all 9 algorithms with error bars (+/- sigma_G).
4. Panel D (Reward Design Ablation & Training Convergence):
   - Comparison of Rigorous (r_rig), Preference (r_pref), and Hybrid (r_hybrid) rewards
   - Dual-axis training convergence curve showing Critic Loss decay and Return ascent.
"""

import os
import sys
import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def generate_benchmark_figures(
    benchmark_json: Optional[str] = None,
    curves_json: Optional[str] = None,
    output_png: Optional[str] = None,
) -> str:
    """Renders the comprehensive 4-panel benchmark figure."""
    if benchmark_json is None:
        benchmark_json = os.path.join(CODE_DIR, "evaluation", "multi_algorithm_benchmark_results.json")
    if curves_json is None:
        curves_json = os.path.join(CODE_DIR, "evaluation", "training_curves.json")
    if output_png is None:
        output_png = os.path.join(CODE_DIR, "evaluation", "figures", "multi_algorithm_benchmark_lapse.png")

    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    with open(benchmark_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    with open(curves_json, "r", encoding="utf-8") as f:
        curves = json.load(f)

    df = pd.DataFrame(results)

    fig = plt.figure(figsize=(19, 13), dpi=180)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    time_days = np.linspace(0.0, 16.0, 65)  # 64 steps + initial state

    # Color palette matching LAPSE publication styling
    colors = {
        "safe_rlhf": "#00897b",      # Teal (Proposed)
        "dppo": "#7b1fa2",           # Purple (LAPSE baseline)
        "lagrangian_sac": "#1976d2", # Blue
        "sac": "#e53935",            # Red (Unconstrained)
        "ppo": "#fb8c00",            # Orange
        "ddpg": "#8d6e63",           # Brown
        "cql": "#43a047",            # Green
        "iql": "#00acc1",            # Cyan
        "hcr": "#546e7a",            # Grey/Slate (Human baseline)
    }

    # --------------------------------------------------------------------------
    # Panel A: Closed-Loop COD Removals (Top Left)
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    highlight_keys = ["safe_rlhf", "dppo", "lagrangian_sac", "sac", "hcr"]
    styles = {
        "safe_rlhf": {"lw": 2.8, "ls": "-"},
        "dppo": {"lw": 2.2, "ls": "--"},
        "lagrangian_sac": {"lw": 2.0, "ls": "-."},
        "sac": {"lw": 1.8, "ls": ":"},
        "hcr": {"lw": 2.0, "ls": "-"},
    }

    for row in results:
        k = row["algorithm_key"]
        if k in highlight_keys:
            traj = row.get("representative_cod_traj", [])
            t_axis = np.linspace(0.0, 16.0, len(traj))
            c = colors.get(k, "#333333")
            st = styles.get(k, {"lw": 1.8, "ls": "-"})
            lbl = f"{row['algorithm_name']} (Rem: {row['cod_removal_mean']}%)"
            ax_a.plot(t_axis, traj, color=c, lw=st["lw"], ls=st["ls"], label=lbl)

    ax_a.axhline(350.0, color="#d50000", lw=2.2, ls="--", label="PerMenLHK COD Limit (350 mg/L)")
    ax_a.set_title("A. Closed-Loop COD Bioremediation Trajectories over 16 Days", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Cultivation Time (Days)")
    ax_a.set_ylabel("Chemical Oxygen Demand COD (mg/L)")
    ax_a.set_xlim(0, 16)
    ax_a.set_ylim(0, 4500)
    ax_a.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_a.grid(True, alpha=0.25)

    # --------------------------------------------------------------------------
    # Panel B: Closed-Loop Broth pH Stability (Top Right)
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    for row in results:
        k = row["algorithm_key"]
        if k in highlight_keys:
            traj = row.get("representative_ph_traj", [])
            t_axis = np.linspace(0.0, 16.0, len(traj))
            c = colors.get(k, "#333333")
            st = styles.get(k, {"lw": 1.8, "ls": "-"})
            lbl = f"{row['algorithm_name']}"
            ax_b.plot(t_axis, traj, color=c, lw=st["lw"], ls=st["ls"], label=lbl)

    ax_b.axhline(6.0, color="#d50000", lw=1.8, ls="--", label="PerMenLH Lower Limit (pH 6.0)")
    ax_b.axhline(9.0, color="#d50000", lw=1.8, ls="--", label="PerMenLH Upper Limit (pH 9.0)")
    ax_b.axhline(5.5, color="#7f0000", lw=1.2, ls=":", label="C. vulgaris Survival [5.5, 10.5]")
    ax_b.axhline(10.5, color="#7f0000", lw=1.2, ls=":")
    ax_b.axhspan(6.0, 9.0, color="#4caf50", alpha=0.10, label="PerMenLH Regulatory Band [6.0, 9.0]")
    ax_b.axhspan(4.5, 5.5, color="#d50000", alpha=0.15, hatch="//")
    ax_b.axhspan(10.5, 11.0, color="#d50000", alpha=0.15, hatch="//")
    ax_b.set_title("B. Closed-Loop Broth pH Homeostasis & Two-Tier Boundaries", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Cultivation Time (Days)")
    ax_b.set_ylabel("Photobioreactor Broth pH")
    ax_b.set_xlim(0, 16)
    ax_b.set_ylim(5.0, 10.8)
    ax_b.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax_b.grid(True, alpha=0.25)

    # --------------------------------------------------------------------------
    # Panel C: Safety vs Cumulative Return Tradeoff (Bottom Left)
    # Replicating Table 3 and trade-off analysis of LAPSE
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    main_algs = [r for r in results if r["category"] != "Reward Ablation"]

    alg_names = [r["algorithm_name"].replace(" (Proposed)", "").replace(" (LAPSE Baseline)", "").replace(" (Operator Heuristic)", "") for r in main_algs]
    g_means = [r["G_mean"] for r in main_algs]
    g_stds = [r["G_std"] for r in main_algs]
    n_cvs = [r["n_cv"] for r in main_algs]
    bar_colors = [colors.get(r["algorithm_key"], "#1976d2") for r in main_algs]

    x_pos = np.arange(len(alg_names))
    width = 0.38

    ax_c_twin = ax_c.twinx()
    rects1 = ax_c.bar(x_pos - width/2, g_means, width, yerr=g_stds, capsize=3, color=bar_colors, alpha=0.85, label="Cumulative Return $\\bar{G}$")
    rects2 = ax_c_twin.bar(x_pos + width/2, n_cvs, width, color="#b71c1c", alpha=0.75, hatch="///", label="Constraint Violations $n_{\\text{CV}}$")

    ax_c.set_title("C. Multi-Algorithm Return $\\bar{G} \\pm \\sigma_G$ vs Safety Violations $n_{\\text{CV}}$", fontsize=11, fontweight="bold")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(alg_names, rotation=35, ha="right", fontsize=8.5)
    ax_c.set_ylabel("Undiscounted Cumulative Return $\\bar{G}$", color="#1a237e", fontweight="bold")
    ax_c_twin.set_ylabel("Total Constraint Violations $n_{\\text{CV}}$ (CNCV)", color="#b71c1c", fontweight="bold")
    ax_c.grid(True, alpha=0.2)

    # Combined legend
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c_twin.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel D: Reward Ablation Performance & Training Convergence (Bottom Right)
    # --------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    # Plot genuine per-algorithm training convergence curves
    for key, label, col, ls, lw in [
        ("safe_rlhf", "Safe RLHF Return", "#00897b", "-", 2.4),
        ("dppo", "DPPO Return", "#7b1fa2", "--", 2.2),
        ("sac", "Standard SAC Return", "#e53935", ":", 1.8),
    ]:
        if key in curves and curves[key].get("return_mean"):
            rets = curves[key]["return_mean"]
            it_list = curves[key].get("iterations", curves.get("iterations", list(range(1, len(rets) + 1))))
            # Normalize progress 0->100% if algorithms use different iteration units (epochs vs episodes)
            if len(it_list) > 1:
                x_prog = np.linspace(0.0, 100.0, len(rets))
            else:
                x_prog = np.array([100.0])
            ax_d.plot(x_prog, rets, color=col, lw=lw, ls=ls, marker="o", markersize=4, label=label)

    ax_d.set_title("D. Multi-Seed Authentic Training Convergence & Reward Ablation", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Training Progress (%)")
    ax_d.set_ylabel("Average Episode Return $\\bar{G}$")
    ax_d.set_xlim(0, 100)
    ax_d.legend(loc="lower right", fontsize=8.5)
    ax_d.grid(True, alpha=0.25)

    # Inset table summarizing reward ablations
    r_abls = [r for r in results if r["category"] == "Reward Ablation"]
    if r_abls:
        inset_text = "Reward Ablation (Safe RLHF):\n" + "\n".join([
            f"• {r['algorithm_name'].split('(')[-1].replace(')', '')}: G={r['G_mean']:.1f}, Viol={r['n_cv']}, Pareto={r['pareto_score_mean']:.1f}"
            for r in r_abls
        ])
        ax_d.text(
            0.04, 0.95, inset_text,
            transform=ax_d.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc", alpha=0.9),
        )

    fig.suptitle(
        "Benchmark Komparatif 9 Algoritma RL & Ablasi Reward pada Bioreaktor POME — Replikasi Metodologi Paper LAPSE (Gerold et al., 2026)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Benchmark Plot] Generated publication figure: {output_png}")
    return output_png


if __name__ == "__main__":
    generate_benchmark_figures()
