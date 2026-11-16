"""
Figure 4: Cultivation Performance and MCDM Analyses (Individual Single-Plot Figures).
Replicates Fig. 6 from Mirda et al. (2026), Cleaner Engineering and Technology:
Generates EACH graph as an independent, high-resolution (300 DPI) standalone figure:
- (a) Treatment Duration (Days to Comply across K1–K8)
- (b) COD Removal (%) across K1–K8
- (c) Treatment Cost (Cost IDR across K1–K8)
- (d) Biomass Productivity (Biomass g/L across K1–K8)
- (e) TOPSIS Score across 4 weighting scenarios
- (f) VIKOR Score across 4 weighting scenarios
- (g) DEA Efficiency Score across 4 weighting scenarios
- (h) Normalized Comparison: K8 (HCR) vs K5 (Best Manual) vs K7 (Direct)
- (i) Total First-Place Rankings across 39 MCDM Evaluations
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

CULTIVATIONS = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"]

BAR_COLORS = [
    "#78909c", "#78909c", "#78909c", "#78909c", "#78909c", "#78909c",
    "#e57373",  # K7 (Direct ML)
    "#2e7d32",  # K8 (Proposed HCR)
]

DAYS_TO_COMPLY = [16, 25, 9, 14, 8, 14, 16, 5]
COD_REMOVAL = [96.0, 96.8, 98.8, 93.6, 98.6, 92.2, 91.5, 95.7]
TREATMENT_COST = [12115, 15300, 9089, 11035, 9872, 11645, 19748, 6170]
BIOMASS_PROD = [1.82, 1.45, 2.15, 0.89, 2.10, 1.72, 1.68, 2.49]
SCENARIOS = ["Equal\n(33:33:33)", "Cost\nPriority", "Time\nPriority", "Biomass\nPriority"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(CODE_ROOT, "evaluation", "figures", "fig4_individual")


def plot_single_fig4_graph(key: str, output_dir: str = None, show: bool = True):
    """Generates ONE single standalone figure from Figure 4."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
    save_png = os.path.join(output_dir, f"{key}.png")
    save_pdf = os.path.join(output_dir, f"{key}.pdf")

    if key == "fig4a_treatment_duration":
        bars = ax.bar(CULTIVATIONS, DAYS_TO_COMPLY, color=BAR_COLORS, edgecolor="#374151", linewidth=0.7)
        ax.set_title("Treatment Duration (Days to Comply)", fontweight="bold", pad=12)
        ax.set_ylabel("Days to Comply", fontweight="bold")
        ax.set_xlabel("Cultivation Batch", fontweight="bold")
        ax.set_ylim(0, 28)
        for b, val in zip(bars, DAYS_TO_COMPLY):
            c_txt = "#2e7d32" if val == 5 else "black"
            ax.text(b.get_x() + b.get_width()/2, val + 0.6, f"{val}d", ha="center", va="bottom", fontweight="bold", color=c_txt)

    elif key == "fig4b_cod_removal":
        bars = ax.bar(CULTIVATIONS, COD_REMOVAL, color=BAR_COLORS, edgecolor="#374151", linewidth=0.7)
        ax.set_title("COD Pollutant Removal Efficiency (%)", fontweight="bold", pad=12)
        ax.set_ylabel("COD Removal (%)", fontweight="bold")
        ax.set_xlabel("Cultivation Batch", fontweight="bold")
        ax.set_ylim(88, 102)
        for b, val in zip(bars, COD_REMOVAL):
            ax.text(b.get_x() + b.get_width()/2, val + 0.3, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    elif key == "fig4c_treatment_cost":
        bars = ax.bar(CULTIVATIONS, TREATMENT_COST, color=BAR_COLORS, edgecolor="#374151", linewidth=0.7)
        ax.set_title("Operational Treatment Cost (Chemical Reagents)", fontweight="bold", pad=12)
        ax.set_ylabel("Total Cost (IDR)", fontweight="bold")
        ax.set_xlabel("Cultivation Batch", fontweight="bold")
        ax.set_ylim(0, 22000)
        for b, val in zip(bars, TREATMENT_COST):
            c_txt = "#2e7d32" if val == 6170 else ("#c62828" if val == 19748 else "black")
            ax.text(b.get_x() + b.get_width()/2, val + 400, f"Rp {val:,}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=c_txt)

    elif key == "fig4d_biomass_productivity":
        bars = ax.bar(CULTIVATIONS, BIOMASS_PROD, color=BAR_COLORS, edgecolor="#374151", linewidth=0.7)
        ax.set_title("Harvested Microalgal Biomass Concentration", fontweight="bold", pad=12)
        ax.set_ylabel("Biomass (g/L)", fontweight="bold")
        ax.set_xlabel("Cultivation Batch", fontweight="bold")
        ax.set_ylim(0, 2.9)
        for b, val in zip(bars, BIOMASS_PROD):
            c_txt = "#2e7d32" if val == 2.49 else "black"
            ax.text(b.get_x() + b.get_width()/2, val + 0.06, f"{val:.2f} g/L", ha="center", va="bottom", fontweight="bold", color=c_txt)

    elif key == "fig4e_topsis_score":
        topsis_data = np.array([
            [0.45, 0.42, 0.41, 0.42], [0.35, 0.32, 0.30, 0.31], [0.65, 0.61, 0.71, 0.61],
            [0.40, 0.38, 0.38, 0.35], [0.70, 0.65, 0.72, 0.58], [0.48, 0.45, 0.48, 0.43],
            [0.28, 0.22, 0.28, 0.32], [0.86, 0.88, 0.88, 0.83],
        ])
        w = 0.09
        x = np.arange(len(SCENARIOS))
        for k in range(8):
            pos = x - 0.35 + (k * w)
            ax.bar(pos, topsis_data[k], w, label=CULTIVATIONS[k], color=BAR_COLORS[k], edgecolor="#333", linewidth=0.4)
        ax.set_title("TOPSIS Multi-Criteria Score (Higher is Better)", fontweight="bold", pad=12)
        ax.set_ylabel("TOPSIS Score", fontweight="bold")
        ax.set_xlabel("Weighting Scenario", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_ylim(0, 1.15)
        ax.legend(ncol=4, loc="upper right", framealpha=0.9, fontsize=8)

    elif key == "fig4f_vikor_score":
        vikor_data = np.array([
            [0.55, 0.52, 0.51, 0.52], [0.70, 0.72, 0.71, 0.66], [0.35, 0.38, 0.32, 0.39],
            [0.68, 0.65, 0.65, 0.60], [0.38, 0.40, 0.35, 0.48], [0.61, 0.60, 0.59, 0.59],
            [0.78, 0.82, 0.80, 0.76], [0.02, 0.01, 0.01, 0.08],
        ])
        w = 0.09
        x = np.arange(len(SCENARIOS))
        for k in range(8):
            pos = x - 0.35 + (k * w)
            ax.bar(pos, vikor_data[k], w, label=CULTIVATIONS[k], color=BAR_COLORS[k], edgecolor="#333", linewidth=0.4)
        ax.set_title("VIKOR Regret Score (Lower is Better - K8 Near 0)", fontweight="bold", pad=12)
        ax.set_ylabel("VIKOR Score", fontweight="bold")
        ax.set_xlabel("Weighting Scenario", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_ylim(0, 1.05)
        ax.legend(ncol=4, loc="upper right", framealpha=0.9, fontsize=8)

    elif key == "fig4g_dea_score":
        dea_data = np.array([
            [0.65, 0.61, 0.60, 0.69], [0.45, 0.41, 0.39, 0.42], [0.85, 0.82, 0.90, 0.82],
            [0.51, 0.49, 0.47, 0.45], [0.78, 0.76, 0.80, 0.70], [0.62, 0.58, 0.58, 0.55],
            [0.50, 0.46, 0.46, 0.48], [1.00, 1.00, 1.00, 1.00],
        ])
        w = 0.09
        x = np.arange(len(SCENARIOS))
        for k in range(8):
            pos = x - 0.35 + (k * w)
            ax.bar(pos, dea_data[k], w, label=CULTIVATIONS[k], color=BAR_COLORS[k], edgecolor="#333", linewidth=0.4)
        ax.set_title("DEA Technical Efficiency Score (K8 = 1.0 Frontier)", fontweight="bold", pad=12)
        ax.set_ylabel("DEA Score", fontweight="bold")
        ax.set_xlabel("Weighting Scenario", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_ylim(0, 1.25)
        ax.legend(ncol=4, loc="upper right", framealpha=0.9, fontsize=8)

    elif key == "fig4h_k8_vs_k5_vs_k7_normalized":
        crit_labels = ["Compliance\nDuration⁻¹", "COD\nRemoval", "Biomass\nProductivity", "Cost\nEfficiency⁻¹"]
        x = np.arange(len(crit_labels))
        w = 0.25
        k5_norm = [0.62, 1.00, 0.84, 0.62]
        k8_norm = [1.00, 0.97, 1.00, 1.00]
        k7_norm = [0.31, 0.93, 0.67, 0.31]

        b1 = ax.bar(x - w, k5_norm, w, label="K5 (Best Manual)", color="#90a4ae", edgecolor="#374151")
        b2 = ax.bar(x,     k8_norm, w, label="K8 (HCR Proposed)", color="#2e7d32", edgecolor="#1b5e20")
        b3 = ax.bar(x + w, k7_norm, w, label="K7 (Direct ML)", color="#e57373", edgecolor="#c62828")

        ax.set_title("Multi-Objective Normalized Comparison (K8 vs K5 vs K7)", fontweight="bold", pad=14)
        ax.set_ylabel("Normalized Score", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(crit_labels)
        ax.set_ylim(0, 1.22)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, framealpha=0.92, fontsize=8.5)

        for bars in [b1, b2, b3]:
            for b in bars:
                v = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    elif key == "fig4i_mcdm_rankings":
        first_places = [0, 0, 0, 0, 0, 0, 0, 39]
        bars = ax.bar(CULTIVATIONS, first_places, color=BAR_COLORS, edgecolor="#374151", linewidth=0.7)
        ax.set_title("Total First-Place Rankings across 39 MCDM Evaluators", fontweight="bold", pad=12)
        ax.set_ylabel("First-Place Count", fontweight="bold")
        ax.set_xlabel("Cultivation Batch", fontweight="bold")
        ax.set_ylim(0, 46)
        for b, val in zip(bars, first_places):
            c_txt = "#2e7d32" if val == 39 else "black"
            ax.text(b.get_x() + b.get_width()/2, val + 1.0, str(val), ha="center", va="bottom", fontsize=9, fontweight="bold", color=c_txt)

    plt.tight_layout()
    plt.savefig(save_png, bbox_inches="tight", dpi=300)
    plt.savefig(save_pdf, bbox_inches="tight")

    print(f"✓ Grafik terpisah berhasil disimpan:")
    print(f"  → PNG: {save_png}")
    print(f"  → PDF: {save_pdf}")

    if show:
        print(f"[INFO] Menampilkan grafik '{key}' di layar... (Tutup jendela untuk lanjut)")
        plt.show()
    plt.close()


FIG4_KEYS = [
    "fig4a_treatment_duration",
    "fig4b_cod_removal",
    "fig4c_treatment_cost",
    "fig4d_biomass_productivity",
    "fig4e_topsis_score",
    "fig4f_vikor_score",
    "fig4g_dea_score",
    "fig4h_k8_vs_k5_vs_k7_normalized",
    "fig4i_mcdm_rankings",
]


def plot_all_individual_fig4(output_dir: str = None, show_each: bool = False):
    """Generates ALL 9 separate individual figures."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"📊 MEMBUAT 9 GAMBAR INDIVIDUAL UNTUK FIGURE 4 (K1–K8 & MCDM)")
    print(f"Direktori Output: {output_dir}")
    print("=" * 70)

    for idx, k in enumerate(FIG4_KEYS, 1):
        print(f"\n[{idx}/9] Rendering: {k}...")
        plot_single_fig4_graph(k, output_dir=output_dir, show=show_each)

    print("\n" + "=" * 70)
    print(f"🎉 SUKSES! 9 gambar individual terpisah berhasil dibuat di:\n   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Individual Figures for Figure 4")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all"] + FIG4_KEYS,
        help="Pilih grafik individual atau 'all' untuk semua"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder penyimpanan gambar individual"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Tampilkan jendela pop-up untuk setiap gambar"
    )
    args = parser.parse_args()

    if args.target == "all":
        plot_all_individual_fig4(output_dir=args.output_dir, show_each=args.show)
        if not args.show:
            print("\n[TIPS] Menampilkan pratinjau grafik pertama (Treatment Duration) di layar:")
            plot_single_fig4_graph("fig4a_treatment_duration", output_dir=args.output_dir, show=True)
    else:
        plot_single_fig4_graph(args.target, output_dir=args.output_dir, show=True)
