"""
Figure 1: Multi-Resolution Temporal Performance Comparison (Individual Single-Plot Figures).
Replicates Fig. 3 from Mirda et al. (2026), Cleaner Engineering and Technology:
Generates EACH graph as an independent, high-resolution (300 DPI) standalone figure:
- Stage 1 RMSE: Aeration, NaHCO3, Microalgae, Molasses, Light
- Stage 2 R²: COD, BOD, TSS, pH, NH3
Across temporal resolutions: 2h, 4h, 6h, 24h for XGBoost, RF, LightGBM, GBR, and DT.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Style configuration for Elsevier / Nature publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

RESOLUTIONS = ["2h", "4h", "6h", "24h"]
MODELS = ["XGBoost", "RF", "LightGBM", "GBR", "DT"]
COLORS = {
    "XGBoost": "#1f77b4",   # Blue
    "RF": "#2ca02c",        # Green
    "LightGBM": "#ff7f0e",  # Orange
    "GBR": "#9467bd",       # Purple
    "DT": "#d62728",        # Red
}
MARKERS = {
    "XGBoost": "s",
    "RF": "o",
    "LightGBM": "D",
    "GBR": "^",
    "DT": "v",
}

# Empirical benchmark data from paper runs across temporal windows
STAGE1_DATA = {
    "aeration": {
        "id": "fig1a_aeration_rmse",
        "title": "Aeration Dosing Prediction (Stage 1 RMSE)",
        "unit": "ratio",
        "ylabel": "RMSE (mean ± std)",
        "y_lim": (0.04, 0.35),
        "mean": {
            "XGBoost":  [0.060, 0.070, 0.080, 0.150],
            "RF":       [0.080, 0.090, 0.108, 0.200],
            "LightGBM": [0.090, 0.100, 0.120, 0.230],
            "GBR":      [0.100, 0.115, 0.135, 0.250],
            "DT":       [0.120, 0.138, 0.162, 0.300],
        },
        "std": {
            "XGBoost":  [0.015, 0.018, 0.019, 0.035],
            "RF":       [0.018, 0.020, 0.022, 0.040],
            "LightGBM": [0.020, 0.022, 0.025, 0.045],
            "GBR":      [0.022, 0.025, 0.028, 0.048],
            "DT":       [0.025, 0.028, 0.030, 0.055],
        },
    },
    "nahco3": {
        "id": "fig1b_nahco3_rmse",
        "title": "NaHCO₃ Dosing Prediction (Stage 1 RMSE)",
        "unit": "mL",
        "ylabel": "RMSE (mean ± std)",
        "y_lim": (50, 440),
        "mean": {
            "XGBoost":  [72.0,  85.0,  100.0, 185.0],
            "RF":       [95.0,  108.0, 128.0, 235.0],
            "LightGBM": [108.0, 125.0, 146.0, 272.0],
            "GBR":      [125.0, 142.0, 168.0, 315.0],
            "DT":       [150.0, 172.0, 202.0, 375.0],
        },
        "std": {
            "XGBoost":  [18.0, 22.0, 25.0, 45.0],
            "RF":       [22.0, 26.0, 30.0, 52.0],
            "LightGBM": [25.0, 30.0, 35.0, 58.0],
            "GBR":      [28.0, 34.0, 40.0, 65.0],
            "DT":       [32.0, 38.0, 45.0, 72.0],
        },
    },
    "microalgae": {
        "id": "fig1c_microalgae_rmse",
        "title": "Microalgae Inoculum Prediction (Stage 1 RMSE)",
        "unit": "mL",
        "ylabel": "RMSE (mean ± std)",
        "y_lim": (20, 170),
        "mean": {
            "XGBoost":  [31.0, 37.0, 42.0, 78.0],
            "RF":       [38.0, 44.0, 52.0, 95.0],
            "LightGBM": [46.0, 53.0, 62.0, 115.0],
            "GBR":      [52.0, 60.0, 70.0, 130.0],
            "DT":       [60.0, 69.0, 81.0, 150.0],
        },
        "std": {
            "XGBoost":  [8.0,  9.5,  11.0, 18.0],
            "RF":       [9.5,  11.5, 13.0, 22.0],
            "LightGBM": [11.0, 13.0, 15.0, 25.0],
            "GBR":      [13.0, 15.0, 17.0, 28.0],
            "DT":       [15.0, 17.0, 20.0, 32.0],
        },
    },
    "molasses": {
        "id": "fig1d_molasses_rmse",
        "title": "Molasses Carbon Dosing Prediction (Stage 1 RMSE)",
        "unit": "mL",
        "ylabel": "RMSE (mean ± std)",
        "y_lim": (30, 200),
        "mean": {
            "XGBoost":  [35.0, 41.0, 48.0, 88.0],
            "RF":       [44.0, 51.0, 60.0, 110.0],
            "LightGBM": [52.0, 60.0, 71.0, 130.0],
            "GBR":      [58.0, 68.0, 79.0, 146.0],
            "DT":       [68.0, 79.0, 92.0, 170.0],
        },
        "std": {
            "XGBoost":  [9.0,  10.5, 12.0, 20.0],
            "RF":       [11.0, 13.0, 15.0, 25.0],
            "LightGBM": [13.0, 15.0, 17.5, 29.0],
            "GBR":      [15.0, 17.5, 20.0, 32.0],
            "DT":       [17.0, 20.0, 23.0, 36.0],
        },
    },
    "light": {
        "id": "fig1e_light_rmse",
        "title": "Light Intensity Prediction (Stage 1 RMSE)",
        "unit": "lux",
        "ylabel": "RMSE (mean ± std)",
        "y_lim": (80, 620),
        "mean": {
            "XGBoost":  [108.0, 124.0, 148.0, 272.0],
            "RF":       [145.0, 168.0, 198.0, 365.0],
            "LightGBM": [172.0, 198.0, 230.0, 420.0],
            "GBR":      [190.0, 220.0, 255.0, 465.0],
            "DT":       [215.0, 248.0, 288.0, 525.0],
        },
        "std": {
            "XGBoost":  [28.0, 32.0, 38.0, 68.0],
            "RF":       [35.0, 42.0, 48.0, 85.0],
            "LightGBM": [42.0, 50.0, 58.0, 95.0],
            "GBR":      [48.0, 55.0, 64.0, 105.0],
            "DT":       [55.0, 62.0, 72.0, 118.0],
        },
    },
}

STAGE2_DATA = {
    "cod": {
        "id": "fig1f_cod_r2",
        "title": "COD Degradation Effluent Prediction (Stage 2 R²)",
        "ylabel": "R² (mean ± std)",
        "y_lim": (0.20, 0.85),
        "mean": {
            "XGBoost":  [0.720, 0.665, 0.595, 0.405],
            "RF":       [0.680, 0.625, 0.555, 0.365],
            "LightGBM": [0.650, 0.595, 0.525, 0.340],
            "GBR":      [0.610, 0.560, 0.495, 0.320],
            "DT":       [0.550, 0.510, 0.450, 0.285],
        },
        "std": {
            "XGBoost":  [0.080, 0.085, 0.090, 0.110],
            "RF":       [0.085, 0.090, 0.095, 0.115],
            "LightGBM": [0.090, 0.095, 0.100, 0.120],
            "GBR":      [0.095, 0.100, 0.105, 0.125],
            "DT":       [0.105, 0.110, 0.115, 0.135],
        },
    },
    "bod": {
        "id": "fig1g_bod_r2",
        "title": "BOD Removal Effluent Prediction (Stage 2 R²)",
        "ylabel": "R² (mean ± std)",
        "y_lim": (0.22, 0.88),
        "mean": {
            "XGBoost":  [0.740, 0.680, 0.610, 0.415],
            "RF":       [0.700, 0.640, 0.575, 0.380],
            "LightGBM": [0.665, 0.610, 0.540, 0.355],
            "GBR":      [0.630, 0.580, 0.510, 0.335],
            "DT":       [0.575, 0.525, 0.465, 0.300],
        },
        "std": {
            "XGBoost":  [0.075, 0.080, 0.085, 0.105],
            "RF":       [0.080, 0.085, 0.090, 0.110],
            "LightGBM": [0.085, 0.090, 0.095, 0.115],
            "GBR":      [0.090, 0.095, 0.100, 0.120],
            "DT":       [0.100, 0.105, 0.110, 0.130],
        },
    },
    "tss": {
        "id": "fig1h_tss_r2",
        "title": "TSS Effluent Prediction (Stage 2 R²)",
        "ylabel": "R² (mean ± std)",
        "y_lim": (0.25, 0.92),
        "mean": {
            "XGBoost":  [0.775, 0.720, 0.640, 0.430],
            "RF":       [0.730, 0.675, 0.600, 0.395],
            "LightGBM": [0.690, 0.635, 0.565, 0.370],
            "GBR":      [0.640, 0.590, 0.525, 0.345],
            "DT":       [0.580, 0.535, 0.475, 0.315],
        },
        "std": {
            "XGBoost":  [0.070, 0.075, 0.080, 0.100],
            "RF":       [0.075, 0.080, 0.085, 0.105],
            "LightGBM": [0.080, 0.085, 0.090, 0.110],
            "GBR":      [0.085, 0.090, 0.095, 0.115],
            "DT":       [0.095, 0.100, 0.105, 0.125],
        },
    },
    "ph": {
        "id": "fig1i_ph_r2",
        "title": "pH Equilibrium Effluent Prediction (Stage 2 R²)",
        "ylabel": "R² (mean ± std)",
        "y_lim": (0.12, 0.78),
        "mean": {
            "XGBoost":  [0.620, 0.570, 0.510, 0.335],
            "RF":       [0.550, 0.505, 0.450, 0.290],
            "LightGBM": [0.490, 0.450, 0.395, 0.255],
            "GBR":      [0.440, 0.400, 0.355, 0.230],
            "DT":       [0.375, 0.340, 0.300, 0.195],
        },
        "std": {
            "XGBoost":  [0.085, 0.090, 0.095, 0.115],
            "RF":       [0.090, 0.095, 0.100, 0.120],
            "LightGBM": [0.095, 0.100, 0.105, 0.125],
            "GBR":      [0.100, 0.105, 0.110, 0.130],
            "DT":       [0.110, 0.115, 0.120, 0.140],
        },
    },
    "nh3": {
        "id": "fig1j_nh3_r2",
        "title": "Dissolved Ammonia NH₃ Prediction (Stage 2 R²)",
        "ylabel": "R² (mean ± std)",
        "y_lim": (0.35, 1.05),
        "mean": {
            "XGBoost":  [0.910, 0.850, 0.770, 0.540],
            "RF":       [0.880, 0.825, 0.745, 0.510],
            "LightGBM": [0.860, 0.805, 0.725, 0.490],
            "GBR":      [0.835, 0.780, 0.700, 0.470],
            "DT":       [0.795, 0.740, 0.665, 0.435],
        },
        "std": {
            "XGBoost":  [0.060, 0.065, 0.070, 0.090],
            "RF":       [0.065, 0.070, 0.075, 0.095],
            "LightGBM": [0.070, 0.075, 0.080, 0.100],
            "GBR":      [0.075, 0.080, 0.085, 0.105],
            "DT":       [0.085, 0.090, 0.095, 0.115],
        },
    },
}

ALL_DATA = {**STAGE1_DATA, **STAGE2_DATA}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(CODE_ROOT, "evaluation", "figures", "fig1_individual")


def plot_single_graph(target_key: str, output_dir: str = None, show: bool = True):
    """
    Renders ONE single graph as its own independent figure file.
    """
    if target_key not in ALL_DATA:
        raise ValueError(f"Unknown target key '{target_key}'. Choose from: {list(ALL_DATA.keys())}")

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    data = ALL_DATA[target_key]
    filename = data["id"]
    save_png = os.path.join(output_dir, f"{filename}.png")
    save_pdf = os.path.join(output_dir, f"{filename}.pdf")

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
    x = np.arange(len(RESOLUTIONS))

    for m in MODELS:
        mean = data["mean"][m]
        std = data["std"][m]
        ax.errorbar(
            x, mean, yerr=std,
            label=m, color=COLORS[m], marker=MARKERS[m],
            linewidth=1.6, markersize=6.0, capsize=4, capthick=1.2, elinewidth=1.2
        )

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTIONS, fontweight="bold")
    ax.set_xlabel("Temporal Resolution", fontweight="bold", fontsize=11, labelpad=8)
    ax.set_ylabel(data["ylabel"], fontweight="bold", fontsize=11, labelpad=8)
    ax.set_title(data["title"], fontweight="bold", fontsize=11.5, pad=12)

    if "y_lim" in data:
        ax.set_ylim(data["y_lim"])

    ax.legend(loc="upper left" if "RMSE" in data["ylabel"] else "lower left",
              frameon=True, framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(save_png, bbox_inches="tight", dpi=300)
    plt.savefig(save_pdf, bbox_inches="tight")

    print(f"✓ Tersimpan terpisah:")
    print(f"  → PNG: {save_png}")
    print(f"  → PDF: {save_pdf}")

    if show:
        print(f"[INFO] Menampilkan grafik '{data['title']}' di layar... (Tutup jendela untuk lanjut)")
        plt.show()
    plt.close()


def plot_all_individual_graphs(output_dir: str = None, show_each: bool = False):
    """
    Renders ALL 10 graphs, saving EACH graph as its own independent image file.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"📊 MEMBUAT 10 GAMBAR INDIVIDUAL UNTUK FIGURE 1 (Multi-Resolution)")
    print(f"Direktori Output: {output_dir}")
    print("=" * 70)

    for idx, key in enumerate(ALL_DATA.keys(), 1):
        print(f"\n[{idx}/10] Rendering: {ALL_DATA[key]['title']}...")
        plot_single_graph(key, output_dir=output_dir, show=show_each)

    print("\n" + "=" * 70)
    print(f"🎉 SUKSES! 10 gambar individual terpisah berhasil dibuat di:\n   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Individual Figures for Figure 1")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all"] + list(ALL_DATA.keys()),
        help="Pilih satu grafik spesifik (misal: aeration, cod, ph) atau 'all' untuk semua"
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
        help="Tampilkan jendela pop-up grafik satu per satu"
    )
    args = parser.parse_args()

    if args.target == "all":
        # Jika run default, buat semua gambar terpisah dan tampilkan grafik pertama di layar
        plot_all_individual_graphs(output_dir=args.output_dir, show_each=args.show)
        if not args.show:
            print("\n[TIPS] Menampilkan pratinjau grafik pertama (Aeration RMSE) di layar:")
            plot_single_graph("aeration", output_dir=args.output_dir, show=True)
    else:
        plot_single_graph(args.target, output_dir=args.output_dir, show=True)
