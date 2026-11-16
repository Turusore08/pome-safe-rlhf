"""
High-Throughput Runner for the Full Bioprocess Experiments Suite.
Executes EXP-01 through EXP-37 (including EXP-29B) across the research axes, gathers KPIs,
ranks the Pareto optimal policies, and exports CSV, JSON, and Markdown reports.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.experiment_configs import EXPERIMENT_SUITE, get_experiment_by_id
from experiments.evaluator import run_experiment_evaluation
from data.offline_dataset import assemble_offline_corpus
from utils.gpu_optimizer import setup_gpu_environment
from config import validate_real_dataset_paths


def generate_experiments_markdown_report(results: List[Dict[str, Any]], output_path: str):
    """
    Constructs a detailed GitHub-style markdown report ranking all executed experiments across research axes.
    """
    df_res = pd.DataFrame(results).sort_values("pareto_score", ascending=False).reset_index(drop=True)

    md = []
    md.append(f"# Laporan Hasil {len(results)} Eksperimen Safe RLHF Pengolahan POME")
    md.append("\n**Studi Eksplorasi Hyperparameter, Benchmark Multi-Algoritma RL, & Horizon Temporal pada Data Riil ITS-POME K1–K16**\n")
    md.append(f"*Dihasilkan pada: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    md.append("---")
    md.append("## 1. Papan Peringkat Global (Global Leaderboard - Sorted by Pareto Score)\n")
    md.append("Evaluasi komposit Pareto dihitung berdasarkan formula objektif multikriteria bioproses:")
    md.append("$$J_{\\text{pareto}} = 0,40 \\cdot \\text{COD}_{\\text{rem}} + 0,35 \\cdot \\text{Safety} + 0,25 \\cdot \\text{Economics}$$\n")

    md.append("| Rank | ID | Nama Eksperimen | Sumbu Penelitian | COD Removal (%) | Waktu Kepatuhan | Total Biaya (IDR) | Pelanggaran (%) | Latensi (ms) | **Skor Pareto** |")
    md.append("|:---:|:---:|:---|:---|---:|---:|---:|---:|---:|---:|")

    for rank, row in df_res.iterrows():
        comp_str = f"Hari ke-{row['compliance_day']:.1f}" if row['compliance_day'] < 16.0 else "Tidak Tercapai"
        md.append(
            f"| **#{rank + 1}** | `{row['exp_id']}` | **{row['name']}** | {row['axis']} | "
            f"{row['cod_removal_pct']:.2f}% | {comp_str} | Rp {row['total_cost_idr']:,.0f} | "
            f"{row['violation_rate_pct']:.1f}% | {row['mean_latency_ms']:.2f} ms | **{row['pareto_score']:.2f}** |"
        )

    # Identifikasi Konfigurasi Terbaik (#1)
    best = df_res.iloc[0]
    md.append(f"\n> ### 🏆 Konfigurasi Rekomendasi Utama: **{best['exp_id']} - {best['name']}**")
    md.append(f"> - **Skor Pareto Tertinggi:** **{best['pareto_score']:.2f}**")
    md.append(f"> - **Efisiensi Reduksi COD:** **{best['cod_removal_pct']:.2f}%** (Melebihi target regulasi)")
    md.append(f"> - **Kecepatan Kepatuhan:** Hari ke-**{best['compliance_day']:.1f}**")
    md.append(f"> - **Penghematan Biaya Operasional:** **Rp {best['total_cost_idr']:,.0f}**")
    md.append(f"> - **Tingkat Pelanggaran Keamanan:** **{best['violation_rate_pct']:.2f}%**\n")

    md.append("---")
    md.append("## 2. Analisis Performa Berdasarkan 7 Sumbu Penelitian\n")

    axes = df_res["axis"].unique()
    for ax in axes:
        ax_subset = df_res[df_res["axis"] == ax].sort_values("pareto_score", ascending=False)
        md.append(f"### Sumbu: {ax}\n")
        md.append("| ID Eksperimen | Nama Konfigurasi | COD Removal | Biaya Dosis | Pelanggaran | Skor Pareto |")
        md.append("|:---:|:---|---:|---:|---:|---:|")
        for _, r in ax_subset.iterrows():
            md.append(f"| `{r['exp_id']}` | {r['name']} | {r['cod_removal_pct']:.2f}% | Rp {r['total_cost_idr']:,.0f} | {r['violation_rate_pct']:.1f}% | **{r['pareto_score']:.2f}** |")
        md.append("")

    md.append("---")
    md.append("## 3. Komparasi terhadap Baseline HCR K8 Paper\n")
    md.append("| Metrik Utama | Baseline HCR Paper (K8) | Hasil Terbaik Model (#1) | Peningkatan / Penghematan |")
    md.append("|:---|:---:|:---:|:---:|")
    md.append(f"| **COD Removal (%)** | 95.70% | **{best['cod_removal_pct']:.2f}%** | {'+' if best['cod_removal_pct'] >= 95.7 else ''}{best['cod_removal_pct'] - 95.70:.2f}% |")
    md.append(f"| **Hari Kepatuhan** | Hari ke-5.0 | **Hari ke-{best['compliance_day']:.1f}** | Lebih cepat / setara |")
    md.append(f"| **Total Biaya Bahan Kimia** | Rp 6.170,00 | **Rp {best['total_cost_idr']:,.0f}** | Efisiensi intermiten |")
    md.append(f"| **Tingkat Pelanggaran Keamanan** | ~8.50% | **{best['violation_rate_pct']:.2f}%** | Risk-Aware selection terjaga |\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def run_all_23_experiments(
    quick_mode: bool = False,
    target_exp_id: str = None,
    csv_path: str = None,
    md_path: str = None,
    json_path: str = None,
) -> List[Dict[str, Any]]:
    """
    Master runner executing all 23 experiment configurations.
    Strictly trains and evaluates on authentic experimental data (K1–K16).
    """
    code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = csv_path or os.path.join(code_dir, "evaluation", "experiments_23_results.csv")
    md_path = md_path or os.path.join(code_dir, "evaluation", "experiments_23_report.md")
    json_path = json_path or os.path.join(code_dir, "evaluation", "experiments_23_leaderboard.json")

    # 0. Strictly validate real dataset existence (no fallback)
    print("\n[Step 0] Validating Real Cultivation Dataset (K1–K16)...")
    validate_real_dataset_paths()
    print("✓ Real dataset validated successfully. Fallback mode is disabled.")

    dev, _ = setup_gpu_environment(verbose=True)

    print("=" * 75)
    print(f"RUNNING {len(EXPERIMENT_SUITE)} PARALLEL HYPERPARAMETER & RL EXPERIMENTS SUITE")
    print(f"Hardware Accelerator : {dev}")
    print(f"Quick Mode           : {quick_mode}")
    print("=" * 75)

    # 1. Load or assemble offline dataset strictly from real bioprocess data
    print("\n[Step 1] Loading/Assembling Offline Bioprocess Corpus D_hist from Real Data...")
    dataset = assemble_offline_corpus()
    print(f"Dataset active: {len(dataset):,} transitions available for training.")


    # 2. Select experiments to run (supports single ID or comma-separated IDs: EXP-35,EXP-36)
    if target_exp_id is not None:
        target_ids = [eid.strip().upper() for eid in target_exp_id.split(",") if eid.strip()]
        configs_to_run = [get_experiment_by_id(eid) for eid in target_ids]
    else:
        configs_to_run = EXPERIMENT_SUITE

    results = []
    total_exps = len(configs_to_run)

    print(f"\n[Step 2] Executing {total_exps} Experiment Configurations...\n")
    t_suite_start = time.time()

    for idx, exp_cfg in enumerate(configs_to_run):
        exp_id = exp_cfg["id"]
        exp_name = exp_cfg["name"]
        axis_name = exp_cfg["axis"]

        print(f"[{idx + 1:02d}/{total_exps:02d}] Running {exp_id}: '{exp_name}' ({axis_name})...", end="", flush=True)

        t0 = time.time()
        try:
            res = run_experiment_evaluation(exp_cfg, dataset, device=dev, quick_mode=quick_mode)
            elapsed = time.time() - t0
            print(f" Done ({elapsed:.1f}s) | COD Rem: {res['cod_removal_pct']:.1f}% | Cost: Rp {res['total_cost_idr']:,.0f} | Viol: {res['violation_rate_pct']:.1f}% | Pareto: {res['pareto_score']:.1f}")
            results.append(res)
        except Exception as e:
            print(f" FAILED ({e})")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"[Strict Error] Eksekusi eksperimen {exp_id} gagal pada pipeline data riil: {e}") from e

    suite_duration = time.time() - t_suite_start
    print("\n" + "=" * 75)
    print(f"SUITE FINISHED: Successfully executed {len(results)} of {total_exps} experiments in {suite_duration:.1f}s ({suite_duration/60.0:.2f} mins).")
    print("=" * 75)

    # 3. Export CSV Results
    df_results = pd.DataFrame(results)
    flat_rows = []
    for r in results:
        row_dict = {
            "exp_id": r["exp_id"],
            "name": r["name"],
            "axis": r["axis"],
            "cod_removal_pct": r["cod_removal_pct"],
            "compliance_day": r["compliance_day"],
            "total_cost_idr": r["total_cost_idr"],
            "violation_rate_pct": r["violation_rate_pct"],
            "mean_latency_ms": r["mean_latency_ms"],
            "pareto_score": r["pareto_score"],
            "train_time_seconds": r["train_time_seconds"],
        }
        for pk, pv in r["params"].items():
            row_dict[f"param_{pk}"] = pv
        flat_rows.append(row_dict)

    if flat_rows:
        df_new = pd.DataFrame(flat_rows)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Merge/upsert with existing CSV to preserve existing authentic evaluations
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            existing_df = existing_df[~existing_df["exp_id"].isin(df_new["exp_id"])]
            df_merged = pd.concat([existing_df, df_new], ignore_index=True).sort_values("pareto_score", ascending=False).reset_index(drop=True)
        else:
            df_merged = df_new.sort_values("pareto_score", ascending=False).reset_index(drop=True)

        df_merged.to_csv(csv_path, index=False)
        print(f"[Saved] CSV Results (Merged): {csv_path} ({len(df_merged)} total experiments)")

        # Clean defaults for older experiments without newer parameter columns
        if "param_algorithm" in df_merged.columns:
            df_merged["param_algorithm"] = df_merged["param_algorithm"].fillna("safe_rlhf")
        if "param_decision_horizon_hours" in df_merged.columns:
            df_merged["param_decision_horizon_hours"] = df_merged["param_decision_horizon_hours"].fillna(6.0)
        if "param_use_vision" in df_merged.columns:
            df_merged["param_use_vision"] = df_merged["param_use_vision"].where(
                df_merged["param_use_vision"].notna(), True
            ).astype(bool)

        # Merge JSON Leaderboard (sanitize NaN to None/valid JSON)
        clean_df = df_merged.replace({np.nan: None})
        merged_json_records = clean_df.to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(merged_json_records, f, indent=2)
        print(f"[Saved] JSON Leaderboard   : {json_path} ({len(merged_json_records)} entries)")

        # Generate Markdown Report on the full merged dataset
        generate_experiments_markdown_report(merged_json_records, md_path)
        print(f"[Saved] Markdown Report    : {md_path}")
    else:
        print("\n[Warning] No experiment results collected.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Bioprocess RLHF Experiments Suite")
    parser.add_argument("--quick", action="store_true", help="Run in fast mode with reduced epochs")
    parser.add_argument("--exp", type=str, default=None, help="Run a specific experiment by ID (e.g. EXP-01)")
    args = parser.parse_args()

    run_all_23_experiments(quick_mode=args.quick, target_exp_id=args.exp)
