"""
2D Reward Landscape Heatmaps & Segment Length Ablation Visualization (LAPSE Adaptation).

Replicates Figures 1, 2, and 3 from Gerold, Brandner, & Lucia (ESCAPE 36 / LAPSE 2026):
1. Heatmap 1 (COD vs Biomassa):
   - Maps predicted scalar reward over COD in [0, 4000 mg/L] and Biomassa in [0.2, 2.5 g/L].
   - Overlays PerMenLHK No. 5/2014 statutory compliance threshold (COD = 350 mg/L).
2. Heatmap 2 (pH vs NaHCO3 Dosing):
   - Maps predicted scalar reward over pH in [5.0, 10.5] and NaHCO3 in [0, 5 g/step].
   - Overlays safety constraint boundaries at pH = 6.0 and pH = 9.2.
   - Demonstrates the sharp penalty cliff outside the safe operating envelope.
3. Heatmap 3 (Ablation of Segment Length L_s in {4, 8, 20, comb} - Replicating Figure 3 LAPSE):
   - 4-panel contour plot comparing reward shapes learned by:
     * L_s = 4   : sharp, localized constraint boundaries (local safety)
     * L_s = 8   : transient diurnal dynamic response
     * L_s = 20  : smooth long-term biomass growth slope
     * L_s = comb: balanced combination achieving crisp safety cliff + positive growth gradient.
"""

import os
import sys
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.reward_net import RewardEnsemble
from preference.train_reward import TriRewardEvaluator


def load_ensemble_checkpoint(ckpt_path: str, device: torch.device) -> RewardEnsemble:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[Strict Visualization Error] Ensemble checkpoint '{ckpt_path}' tidak ditemukan. "
            "Visualisasi fallback dilarang; jalankan train_preference_ablation_suite / run_pipeline.py terlebih dahulu."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ensemble = RewardEnsemble(
        state_dim=ckpt.get("state_dim", 10),
        action_dim=ckpt.get("action_dim", 5),
        ensemble_size=ckpt.get("ensemble_size", 3),
    ).to(device)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    return ensemble


def generate_reward_heatmaps(
    checkpoints_dir: Optional[str] = None,
    output_png: Optional[str] = None,
    grid_res: int = 50,
):
    """Generates the comprehensive 3-figure scientific reward heatmaps suite."""
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(CODE_DIR, "checkpoints")

    if output_png is None:
        output_png = os.path.join(CODE_DIR, "evaluation", "figures", "reward_heatmaps_lapse.png")

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the 4 ensembles
    ensembles = {}
    for name in ["4", "8", "20", "comb"]:
        ckpt_p = os.path.join(checkpoints_dir, f"reward_ensemble_Ls_{name}.pt")
        ensembles[name] = load_ensemble_checkpoint(ckpt_p, device)

    # Use L_s=comb or first available for Heatmap 1 & Heatmap 2
    primary_ens = ensembles.get("comb") or ensembles.get("20") or ensembles.get("8") or ensembles.get("4")

    # Baseline nominal state and action vectors
    # S: [cod, bod, toc, tss, ph, do, tds, nh3, water_temp, biomass_gL]
    # A: [T1_aer, T4_bic, T5_ino, T3_mol, T2_light]
    s_nom = np.array([650.0, 325.0, 205.0, 1145.0, 8.8, 6.2, 6080.0, 60.0, 30.0, 1.15], dtype=np.float32)
    a_nom = np.array([3.5, 0.5, 0.0, 0.0, 3500.0], dtype=np.float32)

    fig = plt.figure(figsize=(18, 12), dpi=180)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.28)

    # --------------------------------------------------------------------------
    # Panel 1: Heatmap 1 - COD vs Biomassa (Top Left, 2 columns wide)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])
    cod_vals = np.linspace(50.0, 4000.0, grid_res)
    bio_vals = np.linspace(0.2, 2.5, grid_res)
    COD_G, BIO_G = np.meshgrid(cod_vals, bio_vals)

    R_grid_1 = np.zeros_like(COD_G)
    for i in range(grid_res):
        for j in range(grid_res):
            s_eval = s_nom.copy()
            s_eval[0] = COD_G[i, j]
            s_eval[1] = 0.50 * COD_G[i, j]
            s_eval[-1] = BIO_G[i, j]

            if primary_ens is not None:
                st = torch.tensor(s_eval, dtype=torch.float32, device=device).unsqueeze(0)
                at = torch.tensor(a_nom, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    mu, _ = primary_ens(st, at)
                r_val = float(mu.item())
            else:
                # Rigorous prior shaping
                r_val = - (COD_G[i, j] - 350.0) / 400.0 + (BIO_G[i, j] * 2.0)
            R_grid_1[i, j] = r_val

    c1 = ax1.contourf(COD_G, BIO_G, R_grid_1, levels=25, cmap="viridis")
    fig.colorbar(c1, ax=ax1, label="Predicted Reward $\\hat{r}(s, a)$")
    ax1.axvline(350.0, color="#d50000", lw=2.2, ls="--", label="PerMenLHK COD Limit (350 mg/L)")
    ax1.scatter([350.0], [1.8], color="#ffeb3b", s=110, edgecolors="black", zorder=5, label="Optimal Discharged Effluent")
    ax1.set_title("Heatmap 1: Predicted Reward Landscape over COD vs Biomassa ($L_s = \\text{comb}$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Chemical Oxygen Demand COD (mg/L)")
    ax1.set_ylabel("Microalgae Dry Biomass $X$ (g/L)")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, alpha=0.25)

    # --------------------------------------------------------------------------
    # Panel 2: Heatmap 2 - pH vs NaHCO3 Dosing (Top Right, 2 columns wide)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2:])
    ph_vals = np.linspace(5.0, 10.5, grid_res)
    bic_vals = np.linspace(0.0, 5.0, grid_res)
    PH_G, BIC_G = np.meshgrid(ph_vals, bic_vals)

    R_grid_2 = np.zeros_like(PH_G)
    for i in range(grid_res):
        for j in range(grid_res):
            s_eval = s_nom.copy()
            s_eval[4] = PH_G[i, j]
            a_eval = a_nom.copy()
            a_eval[1] = BIC_G[i, j]

            if primary_ens is not None:
                st = torch.tensor(s_eval, dtype=torch.float32, device=device).unsqueeze(0)
                at = torch.tensor(a_eval, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    mu, _ = primary_ens(st, at)
                r_val = float(mu.item())
            else:
                pen = -35.0 if (PH_G[i, j] < 6.0 or PH_G[i, j] > 9.2) else 5.0
                r_val = pen - 0.5 * BIC_G[i, j]
            R_grid_2[i, j] = r_val

    c2 = ax2.contourf(PH_G, BIC_G, R_grid_2, levels=25, cmap="magma")
    fig.colorbar(c2, ax=ax2, label="Predicted Reward $\\hat{r}(s, a)$")
    ax2.axvline(6.0, color="#d50000", lw=2.0, ls="--", label="PerMenLH pH [6.0, 9.0]")
    ax2.axvline(9.0, color="#d50000", lw=2.0, ls="--")
    ax2.axvline(5.5, color="#ff6d00", lw=1.4, ls=":", label="C. vulgaris Survival [5.5, 10.5]")
    ax2.axvline(10.5, color="#ff6d00", lw=1.4, ls=":")
    ax2.axvspan(5.0, 5.5, color="#d50000", alpha=0.18, hatch="//", label="Acidic Collapse Hazard (<5.5)")
    ax2.axvspan(9.0, 10.5, color="#ff6d00", alpha=0.12, hatch="\\\\", label="Above Effluent Limit (>9.0)")
    ax2.set_title("Heatmap 2: Reward Landscape over Broth pH vs $\\text{NaHCO}_3$ Dosing", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Photobioreactor Broth pH")
    ax2.set_ylabel("$\\text{NaHCO}_3$ Buffer Dosing $T4$ (g/step)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.25)

    # --------------------------------------------------------------------------
    # Bottom Row: Heatmap 3 - Segment Length Ablation (4 Panels: L_s = 4, 8, 20, comb)
    # Replicating Figure 3 of LAPSE (Gerold et al., 2026)
    # --------------------------------------------------------------------------
    labels = ["$L_s = 4$ (24h Snippets)", "$L_s = 8$ (48h Snippets)", "$L_s = 20$ (120h Snippets)", "$L_s = \\text{comb}$ (Balanced 1/3)"]
    names = ["4", "8", "20", "comb"]

    # Shared COD vs pH evaluation grid for direct contour comparison across horizons
    cod_abl = np.linspace(100.0, 3500.0, grid_res)
    ph_abl = np.linspace(5.5, 10.2, grid_res)
    COD_A, PH_A = np.meshgrid(cod_abl, ph_abl)

    # Compute min/max reward across all 4 models for unified color normalization
    all_grids = []
    for n in names:
        ens = ensembles.get(n)
        grid_n = np.zeros_like(COD_A)
        for i in range(grid_res):
            for j in range(grid_res):
                s_eval = s_nom.copy()
                s_eval[0] = COD_A[i, j]
                s_eval[1] = 0.50 * COD_A[i, j]
                s_eval[4] = PH_A[i, j]
                if ens is not None:
                    st = torch.tensor(s_eval, dtype=torch.float32, device=device).unsqueeze(0)
                    at = torch.tensor(a_nom, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        mu, _ = ens(st, at)
                    grid_n[i, j] = float(mu.item())
                else:
                    pen = -30.0 if (PH_A[i, j] < 6.0 or PH_A[i, j] > 9.2) else 5.0
                    grid_n[i, j] = pen - (COD_A[i, j] / 400.0)
        all_grids.append(grid_n)

    vmin = min(np.min(g) for g in all_grids)
    vmax = max(np.max(g) for g in all_grids)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for idx, (n_lbl, g_data) in enumerate(zip(labels, all_grids)):
        ax = fig.add_subplot(gs[1, idx])
        cf = ax.contourf(COD_A, PH_A, g_data, levels=20, cmap="Spectral_r", norm=norm)
        ax.axhline(6.0, color="#d50000", ls=":", lw=1.6)
        ax.axhline(9.0, color="#d50000", ls=":", lw=1.6)
        ax.axvline(350.0, color="#2e7d32", ls="--", lw=1.8)
        ax.set_title(f"Model {n_lbl}", fontsize=10, fontweight="bold")
        ax.set_xlabel("COD (mg/L)")
        if idx == 0:
            ax.set_ylabel("Broth pH")
        ax.grid(True, alpha=0.2)

    # Colorbar for the bottom 4-panel comparison
    cbar_ax = fig.add_axes([0.915, 0.08, 0.015, 0.32])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="Spectral_r"), cax=cbar_ax, label="Reward $\\hat{r}_{\\text{pref}}(s, a)$")

    fig.suptitle(
        "Interpretabilitas Model Reward & Ablasi Panjang Segmen ($L_s \\in \\{4, 8, 20, \\text{comb}\\}$) — Replikasi Metodologi Paper LAPSE",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Reward Heatmaps] Saved 3-figure publication suite to: {output_png}")
    return output_png


if __name__ == "__main__":
    generate_reward_heatmaps()
