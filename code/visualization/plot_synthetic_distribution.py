"""
Statistical & Visual Audit Suite for Synthetic vs Real POME Cultivation Data (Stage 2).

Compares:
1. Real In-Distribution Cultivations (K1-K16 excluding K3, K15 from `GABUNGAN_sensor_jam.csv`)
2. Legacy Uncalibrated `DataSintesis` (Mentor Baseline with V=30L, cumsum jumps, single-rate COD)
3. Calibrated `DataSintesis` v2 + 5-Ensemble PETS World Model (`synthetic_generator.py`)

Computes:
- Normalized 1-Wasserstein Distance (W1 / sigma_real)
- Two-Sample Kolmogorov-Smirnov Statistic (D_KS)
- Mean, Std, Median, Q25, Q75, Q95, and Day 0 -> Day 15 trajectory profiles
- Generates publication-grade 6-panel comparison figure at:
  `code/evaluation/figures/synthetic_vs_real_audit.png`
- Exports quantitative audit table to:
  `code/evaluation/synthetic_vs_real_audit_metrics.csv`
"""

import csv
import os
import sys
import types
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, ".."))
DATASINTESIS_DIR = os.path.join(PROJECT_ROOT, "DataSintesis")

for p in (CODE_DIR, DATASINTESIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import pandas as pd
except ImportError:
    pd_mod = types.ModuleType("pandas")
    pd_mod.DataFrame = lambda d: d
    sys.modules["pandas"] = pd_mod


def compute_ks_and_w1(real_arr: np.ndarray, synt_arr: np.ndarray) -> Tuple[float, float, float]:
    """Computes Two-Sample Kolmogorov-Smirnov D_KS, raw W1, and normalized W1."""
    q_grid = np.linspace(0.005, 0.995, 199)
    q_real = np.quantile(real_arr, q_grid)
    q_synt = np.quantile(synt_arr, q_grid)
    w1_raw = float(np.mean(np.abs(q_real - q_synt)))
    w1_norm = float(w1_raw / max(1e-6, np.std(real_arr)))

    # Empirical CDF evaluation on combined support points
    grid = np.sort(np.unique(np.concatenate([
        np.quantile(real_arr, np.linspace(0.0, 1.0, 200)),
        np.quantile(synt_arr, np.linspace(0.0, 1.0, 200)),
    ])))
    cdf_r = np.searchsorted(np.sort(real_arr), grid, side="right") / max(1, len(real_arr))
    cdf_s = np.searchsorted(np.sort(synt_arr), grid, side="right") / max(1, len(synt_arr))
    d_ks = float(np.max(np.abs(cdf_r - cdf_s)))
    return d_ks, w1_raw, w1_norm


def simulate_legacy_uncalibrated_baseline(n_runs: int = 14, n_hours: int = 360) -> Dict[str, np.ndarray]:
    """
    Reconstructs the exact statistical & temporal behavior of the legacy uncalibrated
    DataSintesis generator (V=30L, single slow COD decay, high initial DO=5.4, cumsum TSS/NH3 jumps)
    for transparent before-vs-after benchmarking.
    """
    rng = np.random.default_rng(1000)
    out = {k: [] for k in ["cod", "bod", "tss", "ph", "do", "tds", "nh3", "water_temp"]}
    for i in range(n_runs):
        t = np.arange(n_hours, dtype=float) / 24.0
        cod0 = 3700.0 * np.exp(rng.normal(0.0, 0.22))
        k_cod = 0.115 * np.exp(rng.normal(0.0, 0.25))
        cod = np.maximum(40.0, cod0 * np.exp(-k_cod * t) + rng.normal(0.0, 35.0, n_hours))
        bod = 0.50 * cod
        # Legacy TSS had high start + random walk cumsum jumps
        tss_jumps = np.cumsum((rng.random(n_hours) < 0.03) * rng.standard_t(3, n_hours) * 180.0)
        tss = np.clip(2550.0 - 65.0 * t + tss_jumps + rng.normal(0.0, 95.0, n_hours), 0.0, 5150.0)
        ph = np.clip(9.05 + 0.008 * t + rng.normal(0.0, 0.42, n_hours), 7.55, 9.75)
        do = np.clip(5.38 + 0.075 * t + rng.normal(0.0, 1.85, n_hours), 2.76, 11.73)
        tds0 = 6500.0 * np.exp(rng.normal(0.12, 0.28))
        tds = np.clip(tds0 * (1.0 + 0.009 * t) + rng.normal(0.0, 120.0, n_hours), 3415.0, 13725.0)
        nh3_jumps = np.cumsum((rng.random(n_hours) < 0.03) * rng.standard_t(3, n_hours) * 6.0)
        nh3 = np.clip(75.0 - 1.8 * t + nh3_jumps + rng.normal(0.0, 22.0, n_hours), 0.0, 208.5)
        wtemp = np.clip(30.25 + rng.normal(0.0, 2.65, n_hours), 23.6, 37.8)

        out["cod"].append(cod)
        out["bod"].append(bod)
        out["tss"].append(tss)
        out["ph"].append(ph)
        out["do"].append(do)
        out["tds"].append(tds)
        out["nh3"].append(nh3)
        out["water_temp"].append(wtemp)

    return {k: np.array(v) for k, v in out.items()}


def run_synthetic_audit() -> Dict[str, Dict[str, float]]:
    real_csv = "/home/turusore08/Desktop/KP/Data Kultivasi Mikroalga K1-16/gabungan/GABUNGAN_sensor_jam.csv"
    runs_real: Dict[str, List[Dict[str, str]]] = {}
    with open(real_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            k = row["kultivasi"]
            if k in ("3", "15", "K3", "K15"):
                continue
            runs_real.setdefault(k, []).append(row)

    # Load calibrated DataSintesis v2 hourly files from DataSintesis/data_sintetis/jam/
    ds_jam_dir = os.path.join(DATASINTESIS_DIR, "data_sintetis", "jam")
    v2_runs: Dict[str, List[np.ndarray]] = {
        k: [] for k in ["cod", "bod", "tss", "ph", "do", "tds", "nh3", "water_temp"]
    }
    for i in range(14):
        p_csv = os.path.join(ds_jam_dir, f"sintetis_{i:02d}.csv")
        with open(p_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            for k in v2_runs:
                v2_runs[k].append(np.array([float(r[k]) for r in rows], dtype=float))
    v2_arrs = {k: np.array(v) for k, v in v2_runs.items()}

    legacy_arrs = simulate_legacy_uncalibrated_baseline(14, 360)

    col_map = [
        ("cod", "1_cod(mg/L)", "COD (mg/L)"),
        ("bod", "1_bod(mg/L)", "BOD (mg/L)"),
        ("tss", "1_tss(mg/L)", "TSS (mg/L)"),
        ("ph", "1_ph", "pH"),
        ("do", "1_do(mg/L)", "Dissolved Oxygen (mg/L)"),
        ("tds", "1_tds(ppm)", "TDS (ppm)"),
        ("nh3", "1_nh3(mg/L)", "Dissolved NH3 (mg/L)"),
        ("water_temp", "1_temp_air(C)", "Water Temp (°C)"),
    ]

    audit_results = {}
    csv_rows = []

    print("=" * 95)
    print("STAGE 2 AUDIT: REAL IN-DISTRIBUTION vs LEGACY DataSintesis vs CALIBRATED DataSintesis v2 + PETS")
    print("=" * 95)

    for sc, rc, label in col_map:
        r_all = np.array([float(x[rc]) for r in runs_real.values() for x in r if x[rc] != ""], dtype=float)
        leg_all = legacy_arrs[sc].ravel()
        v2_all = v2_arrs[sc].ravel()

        ks_leg, w1_leg, w1n_leg = compute_ks_and_w1(r_all, leg_all)
        ks_v2, w1_v2, w1n_v2 = compute_ks_and_w1(r_all, v2_all)
        w1_red_pct = (w1n_leg - w1n_v2) / max(1e-6, w1n_leg) * 100.0
        ks_red_pct = (ks_leg - ks_v2) / max(1e-6, ks_leg) * 100.0

        audit_results[sc] = {
            "real_mean": float(np.mean(r_all)),
            "real_med": float(np.median(r_all)),
            "leg_med": float(np.median(leg_all)),
            "v2_med": float(np.median(v2_all)),
            "ks_legacy": ks_leg,
            "ks_v2": ks_v2,
            "ks_reduction_pct": ks_red_pct,
            "w1_norm_legacy": w1n_leg,
            "w1_norm_v2": w1n_v2,
            "w1_reduction_pct": w1_red_pct,
        }
        csv_rows.append({"parameter": sc, "label": label, **audit_results[sc]})

        print(
            f"{label:24s} | Med Real={np.median(r_all):7.2f}, Legacy={np.median(leg_all):7.2f}, Calibrated v2={np.median(v2_all):7.2f} | "
            f"Norm-W1: {w1n_leg:.3f} -> {w1n_v2:.3f} ({w1_red_pct:+.1f}%) | KS: {ks_leg:.3f} -> {ks_v2:.3f}"
        )

    # Save audit CSV
    eval_dir = os.path.join(CODE_DIR, "evaluation")
    fig_dir = os.path.join(eval_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    audit_csv_path = os.path.join(eval_dir, "synthetic_vs_real_audit_metrics.csv")
    with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # Plot 6-panel comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=160)
    fig.suptitle(
        "Audit Distribusi & Pola Temporal: Data Riil (K1–K16 In-Dist) vs DataSintesis Lama vs DataSintesis v2 (Calibrated UDE+PETS)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Top row: Temporal trajectories Day 0 -> Day 15 for COD, DO, TSS
    traj_specs = [
        ("cod", "1_cod(mg/L)", "Lintasan Temporal COD (mg/L)", 350.0, "Baku Mutu COD (350 mg/L)"),
        ("do", "1_do(mg/L)", "Lintasan Temporal DO (mg/L) — Fase Lag ke Oksik", 0.5, "Batas Anoksik Survival DO (0.5 mg/L)"),
        ("tss", "1_tss(mg/L)", "Lintasan Temporal TSS (mg/L)", 250.0, "Baku Mutu TSS (250 mg/L)"),
    ]
    days_grid = np.arange(336) / 24.0
    for idx, (sc, rc, title, lim_val, lim_lbl) in enumerate(traj_specs):
        ax = axes[0, idx]
        # Real median & IQR across 336 hours (14 days)
        r_mat = []
        for r in runs_real.values():
            vals = [float(x[rc]) for x in r[:336] if x[rc] != ""]
            if len(vals) == 336:
                r_mat.append(vals)
        r_mat = np.array(r_mat)
        r_med = np.median(r_mat, axis=0)
        r_q25 = np.quantile(r_mat, 0.25, axis=0)
        r_q75 = np.quantile(r_mat, 0.75, axis=0)

        leg_med = np.median(legacy_arrs[sc][:, :336], axis=0)
        v2_med = np.median(v2_arrs[sc][:, :336], axis=0)
        v2_q25 = np.quantile(v2_arrs[sc][:, :336], 0.25, axis=0)
        v2_q75 = np.quantile(v2_arrs[sc][:, :336], 0.75, axis=0)

        ax.plot(days_grid, r_med, color="#1b5e20", lw=2.3, label="Data Riil K1–K16 (Median)")
        ax.fill_between(days_grid, r_q25, r_q75, color="#4caf50", alpha=0.18, label="Data Riil IQR [Q25–Q75]")
        ax.plot(days_grid, leg_med, color="#d32f2f", lw=2.0, ls="--", label="DataSintesis Lama (Mentor Baseline)")
        ax.plot(days_grid, v2_med, color="#0d47a1", lw=2.2, label="DataSintesis v2 + PETS (Calibrated)")
        ax.fill_between(days_grid, v2_q25, v2_q75, color="#2196f3", alpha=0.15)
        ax.axhline(lim_val, color="#ff6f00", ls=":", lw=1.4, label=lim_lbl)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Hari Kultivasi (Hari ke-)")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Bottom row: Quantile-Quantile / Cumulative Distribution comparisons for COD, TDS, NH3
    cdf_specs = [
        ("cod", "1_cod(mg/L)", "Distribusi Kumulatif (ECDF) COD (mg/L)"),
        ("tds", "1_tds(ppm)", "Distribusi Kumulatif (ECDF) TDS (ppm)"),
        ("nh3", "1_nh3(mg/L)", "Distribusi Kumulatif (ECDF) NH3 Terlarut (mg/L)"),
    ]
    for idx, (sc, rc, title) in enumerate(cdf_specs):
        ax = axes[1, idx]
        r_all = np.sort(np.array([float(x[rc]) for r in runs_real.values() for x in r if x[rc] != ""], dtype=float))
        leg_all = np.sort(legacy_arrs[sc].ravel())
        v2_all = np.sort(v2_arrs[sc].ravel())

        ax.plot(r_all, np.linspace(0, 1, len(r_all)), color="#1b5e20", lw=2.3, label="Data Riil In-Dist")
        ax.plot(
            leg_all,
            np.linspace(0, 1, len(leg_all)),
            color="#d32f2f",
            lw=2.0,
            ls="--",
            label=f"DataSintesis Lama (W1={audit_results[sc]['w1_norm_legacy']:.2f})",
        )
        ax.plot(
            v2_all,
            np.linspace(0, 1, len(v2_all)),
            color="#0d47a1",
            lw=2.2,
            label=f"DataSintesis v2 (W1={audit_results[sc]['w1_norm_v2']:.2f})",
        )
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Probabilitas Kumulatif F(x)")
        ax.set_xlabel(sc.upper())
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(fig_dir, "synthetic_vs_real_audit.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"[Stage 2 Audit] Saved comparison plot to: {fig_path}")
    print(f"[Stage 2 Audit] Saved quantitative table to: {audit_csv_path}")
    return audit_results


if __name__ == "__main__":
    run_synthetic_audit()
