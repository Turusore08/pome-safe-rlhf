"""
Figure 3: HCR versus Direct Regression at 6-Hour Resolution (Individual Single-Plot Figures).
Replicates Fig. 5 from Mirda et al. (2026), Cleaner Engineering and Technology:
Generates EACH graph as an independent, high-resolution (300 DPI) standalone figure:
- (a) Stage 1: RMSE across 5 treatment levers
- (b) Stage 1: MSE across 5 treatment levers
- (c) Stage 1: MAE across 5 treatment levers
- (d) Stage 1: MAPE across 5 treatment levers
- (e) Stage 2: R² across 5 effluent pollutants
- (f) Stage 2: RMSE across 5 effluent pollutants
- (g) Stage 2: MAE across 5 effluent pollutants
- (h) Stage 2: MAPE across 5 effluent pollutants
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
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

TREATMENT_NAMES = ["Aeration", "NaHCO₃", "Microalgae", "Molasses", "Light"]
EFFLUENT_NAMES = ["COD", "BOD", "TSS", "pH", "NH₃"]

COLOR_DIRECT = "#e57373"  # Coral / Red
COLOR_HCR = "#81c784"     # Soft Green

FIG3_ALL_METRICS = {
    "fig3a_stage1_rmse": {
        "title": "Stage 1: RMSE across Treatment Targets (Lower is Better)",
        "labels": TREATMENT_NAMES,
        "ylabel": "RMSE (mean ± std)",
        "direct": [0.108, 128.0, 48.0, 58.0, 228.0],
        "hcr":    [0.081, 137.9, 48.0, 50.3, 188.1],
        "direct_std": [0.018, 22.0, 8.0, 9.5, 38.0],
        "hcr_std":    [0.015, 20.0, 7.5, 8.0, 30.0],
        "deltas": ["-25.0%", "+7.7%", "+0.0%", "-13.3%", "-17.5%"],
        "is_pos_good": False,
    },
    "fig3b_stage1_mse": {
        "title": "Stage 1: MSE across Treatment Targets (Lower is Better)",
        "labels": TREATMENT_NAMES,
        "ylabel": "MSE",
        "direct": [0.012, 16384.0, 2304.0, 3364.0, 51984.0],
        "hcr":    [0.007, 19000.0, 2200.0, 2520.0, 43500.0],
        "direct_std": [0.003, 3500.0, 450.0, 600.0, 8500.0],
        "hcr_std":    [0.002, 3800.0, 400.0, 500.0, 7000.0],
        "deltas": ["-42.3%", "+16.0%", "-4.4%", "-25.0%", "-16.2%"],
        "is_pos_good": False,
    },
    "fig3c_stage1_mae": {
        "title": "Stage 1: MAE across Treatment Targets (Lower is Better)",
        "labels": TREATMENT_NAMES,
        "ylabel": "MAE (mean ± std)",
        "direct": [0.045, 95.0, 28.0, 46.0, 128.0],
        "hcr":    [0.029, 96.0, 29.9, 36.0, 108.3],
        "direct_std": [0.008, 16.0, 5.0, 7.5, 22.0],
        "hcr_std":    [0.006, 15.0, 5.2, 6.0, 18.0],
        "deltas": ["-35.6%", "+1.1%", "+6.7%", "-21.7%", "-15.4%"],
        "is_pos_good": False,
    },
    "fig3d_stage1_mape": {
        "title": "Stage 1: MAPE across Treatment Targets",
        "labels": TREATMENT_NAMES,
        "ylabel": "MAPE (mean ± std)",
        "direct": [108.0, 520.0, 95.0, 295.0, 510.0],
        "hcr":    [53.7,  510.0, 154.3, 422.0, 558.0],
        "direct_std": [25.0, 95.0, 18.0, 55.0, 90.0],
        "hcr_std":    [15.0, 85.0, 28.0, 75.0, 98.0],
        "deltas": ["-50.3%", "-1.9%", "+62.4%", "+43.1%", "+9.4%"],
        "is_pos_good": False,
    },
    "fig3e_stage2_r2": {
        "title": "Stage 2: R² across Effluent Pollutants (Higher is Better)",
        "labels": EFFLUENT_NAMES,
        "ylabel": "R² (mean ± std)",
        "direct": [0.420, 0.380, -3.100, 0.320, 0.440],
        "hcr":    [0.440, 0.410, -0.540, 0.332, 0.458],
        "direct_std": [0.060, 0.055, 0.450, 0.050, 0.060],
        "hcr_std":    [0.055, 0.050, 0.180, 0.045, 0.055],
        "deltas": ["+4.7%", "+7.9%", "+82.6%", "+3.8%", "+4.2%"],
        "is_pos_good": True,
    },
    "fig3f_stage2_rmse": {
        "title": "Stage 2: RMSE across Effluent Pollutants (Lower is Better)",
        "labels": EFFLUENT_NAMES,
        "ylabel": "RMSE (mean ± std)",
        "direct": [1260.0, 580.0, 420.0, 0.45, 18.5],
        "hcr":    [1245.0, 565.0, 410.0, 0.45, 19.2],
        "direct_std": [180.0, 90.0, 70.0, 0.06, 3.0],
        "hcr_std":    [170.0, 85.0, 65.0, 0.06, 3.2],
        "deltas": ["-1.2%", "-2.6%", "-2.4%", "+0.0%", "+4.0%"],
        "is_pos_good": False,
    },
    "fig3g_stage2_mae": {
        "title": "Stage 2: MAE across Effluent Pollutants (Lower is Better)",
        "labels": EFFLUENT_NAMES,
        "ylabel": "MAE (mean ± std)",
        "direct": [920.0, 410.0, 310.0, 0.35, 13.5],
        "hcr":    [910.0, 401.0, 297.6, 0.35, 14.3],
        "direct_std": [130.0, 65.0, 50.0, 0.05, 2.2],
        "hcr_std":    [125.0, 60.0, 48.0, 0.05, 2.4],
        "deltas": ["-1.1%", "-2.2%", "-4.0%", "+0.0%", "+5.9%"],
        "is_pos_good": False,
    },
    "fig3h_stage2_mape": {
        "title": "Stage 2: MAPE across Effluent Pollutants",
        "labels": EFFLUENT_NAMES,
        "ylabel": "MAPE (mean ± std)",
        "direct": [108.0, 168.0, 140.0, 4.8, 55.0],
        "hcr":    [111.8, 166.0, 137.1, 4.8, 56.9],
        "direct_std": [16.0, 25.0, 22.0, 0.7, 9.0],
        "hcr_std":    [17.0, 24.0, 21.0, 0.7, 9.5],
        "deltas": ["+3.5%", "-1.2%", "-2.1%", "+0.0%", "+3.4%"],
        "is_pos_good": False,
    },
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(CODE_ROOT, "evaluation", "figures", "fig3_individual")


def plot_single_fig3_graph(key: str, output_dir: str = None, show: bool = True):
    """Renders ONE single bar comparison figure."""
    if key not in FIG3_ALL_METRICS:
        raise ValueError(f"Unknown metric key '{key}'. Choose from: {list(FIG3_ALL_METRICS.keys())}")

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    metric_data = FIG3_ALL_METRICS[key]
    labels = metric_data["labels"]
    n = len(labels)
    x = np.arange(n)
    w = 0.35

    direct_vals = np.array(metric_data["direct"])
    hcr_vals = np.array(metric_data["hcr"])
    direct_err = np.array(metric_data["direct_std"])
    hcr_err = np.array(metric_data["hcr_std"])
    deltas = metric_data["deltas"]
    is_pos_good = metric_data["is_pos_good"]

    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=300)

    ax.bar(x - w/2, direct_vals, w, yerr=direct_err, label="Direct Regression",
           color=COLOR_DIRECT, edgecolor="#c62828", linewidth=0.8, capsize=4,
           error_kw={"elinewidth": 1.0, "capthick": 1.0})
    ax.bar(x + w/2, hcr_vals, w, yerr=hcr_err, label="Hurdle Chained Regression (HCR)",
           color=COLOR_HCR, edgecolor="#2e7d32", linewidth=0.8, capsize=4,
           error_kw={"elinewidth": 1.0, "capthick": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel(metric_data["ylabel"], fontweight="bold")
    ax.set_title(metric_data["title"], fontweight="bold", pad=16)

    # Percentage delta tags above bars
    for i in range(n):
        tag = deltas[i]
        is_green = (tag.startswith("-") and not is_pos_good) or (tag.startswith("+") and is_pos_good)
        tag_color = "#16a34a" if is_green else ("#dc2626" if tag != "+0.0%" else "#475569")

        max_bar_top = max(direct_vals[i] + direct_err[i], hcr_vals[i] + hcr_err[i])
        y_pos = max_bar_top + (abs(max_bar_top) * 0.08 if max_bar_top != 0 else 0.5)

        if direct_vals[i] < 0 and hcr_vals[i] < 0:
            y_pos = min(direct_vals[i] - direct_err[i], hcr_vals[i] - hcr_err[i]) - 0.4

        ax.text(x[i], y_pos, tag, ha="center", va="bottom" if y_pos >= 0 else "top",
                fontsize=8.5, fontweight="bold", color=tag_color)

    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout()
    save_png = os.path.join(output_dir, f"{key}.png")
    save_pdf = os.path.join(output_dir, f"{key}.pdf")
    plt.savefig(save_png, bbox_inches="tight", dpi=300)
    plt.savefig(save_pdf, bbox_inches="tight")

    print(f"✓ Grafik terpisah berhasil disimpan:")
    print(f"  → PNG: {save_png}")
    print(f"  → PDF: {save_pdf}")

    if show:
        print(f"[INFO] Menampilkan grafik '{key}' di layar... (Tutup jendela untuk lanjut)")
        plt.show()
    plt.close()


def plot_all_individual_fig3(output_dir: str = None, show_each: bool = False):
    """Generates ALL 8 separate individual bar charts."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"📊 MEMBUAT 8 GAMBAR INDIVIDUAL UNTUK FIGURE 3 (HCR vs Direct)")
    print(f"Direktori Output: {output_dir}")
    print("=" * 70)

    for idx, k in enumerate(FIG3_ALL_METRICS.keys(), 1):
        print(f"\n[{idx}/8] Rendering: {k}...")
        plot_single_fig3_graph(k, output_dir=output_dir, show=show_each)

    print("\n" + "=" * 70)
    print(f"🎉 SUKSES! 8 gambar individual terpisah berhasil dibuat di:\n   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Individual Figures for Figure 3")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all"] + list(FIG3_ALL_METRICS.keys()),
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
        plot_all_individual_fig3(output_dir=args.output_dir, show_each=args.show)
        if not args.show:
            print("\n[TIPS] Menampilkan pratinjau grafik pertama (Stage 1 RMSE) di layar:")
            plot_single_fig3_graph("fig3a_stage1_rmse", output_dir=args.output_dir, show=True)
    else:
        plot_single_fig3_graph(args.target, output_dir=args.output_dir, show=True)
