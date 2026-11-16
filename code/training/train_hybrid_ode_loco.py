"""
Leave-One-Cultivation-Out (LOCO) Cross-Validation & Training Entrypoint
for Physics-Informed Hybrid ODE / Universal Differential Equation (UDE).

Implements the complete validation and training pipeline from `hybrid_ode_loco.ipynb`:
1. Evaluates 3 model variants:
   - `mech`     : Pure mechanistic ODE with bounded learnable parameters (±1 decade)
   - `ude`      : Mechanistic ODE + zero-initialized Neural Network growth correction
   - `ude_amm`  : UDE + bacterial ammonification coupling (+ k_amm * r_S)
2. 14-fold Leave-One-Cultivation-Out (LOCO) across in-distribution cultivations
   TRAIN = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16] (K3 & K15 = OOD):
   - Loss normalization scale (`scale_fold`) is computed strictly from the 13 training cultivations per fold
   - Reported NRMSE uses the 14-cultivation std (`scale_report`) for direct comparison with in-sample metrics
   - Checkpoints per (mode, fold) are cached in `evaluation/hybrid_ode_loco_parts/`
3. Produces:
   - `evaluation/hybrid_ode_metrics.csv` (in-sample & OOD metrics)
   - `evaluation/hybrid_ode_loco.csv` (14-fold LOCO metrics)
   - `checkpoints/hybrid_ode_ude_amm.pt` (trained full-data HybridODE checkpoint)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    REAL_GABUNGAN_SENSOR_JAM,
    LOCO_TRAIN_CULTIVATIONS as TRAIN,
    OOD_CULTIVATIONS as OOD,
    LOCO_WINDOW_HOURS as WINDOW,
    LOCO_MODES as MODES,
    validate_real_dataset_paths,
)
from models.hybrid_ode import (
    HybridODE,
    culture_arrays,
    train_hybrid_ode,
    selfcheck,
    evaluate_one,
)


def tag_of(mode: str, amm: bool, seed: int = 0) -> str:
    return f"{mode}{'_amm' if amm else ''}_seed{seed}"


def run_fold(
    mode: str,
    amm: bool,
    k: int,
    arrs_by_k: Dict[int, List[torch.Tensor]],
    scale_report: torch.Tensor,
    epochs: int = 200,
    window: int = WINDOW,
    lr: float = 1e-2,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Train HybridODE from scratch on 13 cultivations (excluding k) and evaluate
    full-trajectory rollout on held-out cultivation k.
    """
    arrs = [arrs_by_k[kk] for kk in TRAIN if kk != k]
    # Scale for training loss is computed strictly from the 13 training cultivations (no data leakage)
    scale_fold = torch.cat([a[0] for a in arrs]).std(0)
    torch.manual_seed(seed)
    model = train_hybrid_ode(
        HybridODE(mode, amm),
        arrs,
        scale_fold,
        epochs=epochs,
        W=window,
        lr=lr,
        log_every=max(epochs // 4, 1),
        device=device,
    )
    res = pd.DataFrame(evaluate_one(model, arrs_by_k[k], scale_report))
    res.insert(0, "kultivasi", k)
    res.insert(0, "model", tag_of(mode, amm, seed))
    return res


def run_in_sample_and_ood(
    df: pd.DataFrame,
    arrs_by_k: Dict[int, List[torch.Tensor]],
    scale_report: torch.Tensor,
    epochs: int = 200,
    window: int = WINDOW,
    lr: float = 1e-2,
    seed: int = 0,
    output_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    quick_test: bool = False,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Train each HybridODE mode on all 14 training cultivations, evaluate in-sample
    on the 14 training runs and out-of-distribution on K3 & K15, and save
    `hybrid_ode_metrics.csv` + model checkpoints.
    """
    train_arrs = [arrs_by_k[k] for k in TRAIN]
    ood_arrs_by_k = {k: culture_arrays(df, k) for k in OOD if k in df["kultivasi"].values}

    rows = []
    for mode, amm in MODES:
        tag = tag_of(mode, amm, seed)
        torch.manual_seed(seed)
        print(f"=== [In-Sample Full Train] {tag} ({len(train_arrs)} kultivasi) ===")
        model = train_hybrid_ode(
            HybridODE(mode, amm),
            train_arrs,
            scale_report,
            epochs=epochs,
            W=window,
            lr=lr,
            log_every=max(epochs // 4, 1),
            device=device,
        )

        if not quick_test and checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / f"hybrid_ode_{mode}{'_amm' if amm else ''}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": mode,
                    "amm": amm,
                    "learned_params": {k: float(v.item()) for k, v in model.params().items()},
                    "scale_report": scale_report.cpu(),
                },
                ckpt_path,
            )

        for k in TRAIN:
            df_k = pd.DataFrame(evaluate_one(model, arrs_by_k[k], scale_report))
            df_k.insert(0, "split", "train")
            df_k.insert(0, "kultivasi", k)
            df_k.insert(0, "model", tag)
            rows.append(df_k)

        for k, ood_arr in ood_arrs_by_k.items():
            df_k = pd.DataFrame(evaluate_one(model, ood_arr, scale_report))
            df_k.insert(0, "split", "ood")
            df_k.insert(0, "kultivasi", k)
            df_k.insert(0, "model", tag)
            rows.append(df_k)

    metrics_df = pd.concat(rows, ignore_index=True)
    if not quick_test and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "hybrid_ode_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"tersimpan in-sample/OOD metrics: {metrics_path}")
    return metrics_df


def run_hybrid_ode_loco(
    csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: int = 200,
    window: int = WINDOW,
    lr: float = 1e-2,
    seed: int = 0,
    quick_test: bool = False,
    force_rerun: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the complete LOCO validation pipeline from `hybrid_ode_loco.ipynb`.
    """
    validate_real_dataset_paths()
    import config

    resolved_csv = Path(csv_path or config.REAL_GABUNGAN_SENSOR_JAM)
    code_root = Path(__file__).resolve().parent.parent
    out_path = Path(output_dir) if output_dir else (code_root / "evaluation")
    ckpt_dir = code_root / "checkpoints"

    df = pd.read_csv(resolved_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["kultivasi", "timestamp"]).reset_index(drop=True)
    assert df.isna().sum().sum() == 0, "tabel jam harus tanpa nilai hilang"
    for k, d in df.groupby("kultivasi"):
        assert (d["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all(), f"celah waktu di K{k}"

    print("data  :", resolved_csv)
    print("output:", out_path)

    arrs_by_k = {k: culture_arrays(df, k) for k in TRAIN}
    scale_report = torch.cat([arrs_by_k[k][0] for k in TRAIN]).std(0)
    selfcheck([arrs_by_k[k] for k in TRAIN])

    folds = TRAIN[:2] if quick_test else TRAIN
    epochs_run = 2 if quick_test else epochs
    parts_dir = out_path / "hybrid_ode_loco_parts"
    if not quick_test:
        parts_dir.mkdir(parents=True, exist_ok=True)

    print("lipatan:", folds, "| epoch:", epochs_run)

    # 1. Run in-sample & OOD benchmark if hybrid_ode_metrics.csv does not exist or force_rerun
    ins_path = out_path / "hybrid_ode_metrics.csv"
    if force_rerun or not ins_path.exists() or quick_test:
        ins_df = run_in_sample_and_ood(
            df=df,
            arrs_by_k=arrs_by_k,
            scale_report=scale_report,
            epochs=epochs_run,
            window=window,
            lr=lr,
            seed=seed,
            output_dir=out_path,
            checkpoint_dir=ckpt_dir,
            quick_test=quick_test,
        )
    else:
        ins_df = pd.read_csv(ins_path)

    # 2. Run 14-fold LOCO validation loop
    results = []
    for mode, amm in MODES:
        tag = tag_of(mode, amm, seed)
        for k in folds:
            part = parts_dir / f"{tag}_K{k:02d}.csv"
            if not quick_test and not force_rerun and part.exists():
                results.append(pd.read_csv(part))
                print(f"[lewati] {tag} K{k}")
                continue
            t0 = time.time()
            print(f"=== {tag} | tahan K{k}")
            res = run_fold(
                mode=mode,
                amm=amm,
                k=k,
                arrs_by_k=arrs_by_k,
                scale_report=scale_report,
                epochs=epochs_run,
                window=window,
                lr=lr,
                seed=seed,
            )
            if not quick_test:
                res.to_csv(part, index=False)
            results.append(res)
            print(f"    selesai {time.time() - t0:.0f} s")

    loco = pd.concat(results, ignore_index=True)
    if not quick_test:
        loco_csv = out_path / "hybrid_ode_loco.csv"
        loco.to_csv(loco_csv, index=False)
        print("tersimpan:", loco_csv)

    # 3. Summary & Comparison: LOCO vs In-Sample
    summary = (
        loco.groupby(["model", "state"])["nrmse"]
        .agg(["mean", "std", "median"])
        .round(3)
        .unstack("state")
    )
    print("\n=== Ringkasan LOCO NRMSE (mean / std / median) ===")
    print(summary)

    ins_train = ins_df[ins_df["split"] == "train"].groupby(["model", "state"])["nrmse"].mean()
    cmp = pd.DataFrame(
        {
            "in_sample": ins_train,
            "loco": loco.groupby(["model", "state"])["nrmse"].mean(),
        }
    ).dropna()
    cmp["selisih"] = cmp["loco"] - cmp["in_sample"]
    print("\n=== Perbandingan LOCO vs In-Sample NRMSE ===")
    print(cmp.round(3).unstack("state"))

    return {"loco": loco, "summary": summary, "comparison": cmp, "in_sample": ins_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validasi LOCO — Hybrid ODE (mech / UDE / UDE + amonifikasi)")
    parser.add_argument("--quick", action="store_true", help="Uji cepat: 2 lipatan x 2 epoch tanpa menulis file")
    parser.add_argument("--epochs", type=int, default=200, help="Jumlah epoch per lipatan (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate Adam (default: 1e-2)")
    parser.add_argument("--force", action="store_true", help="Latih ulang semua lipatan meskipun file part sudah ada")
    args = parser.parse_args()

    run_hybrid_ode_loco(epochs=args.epochs, lr=args.lr, quick_test=args.quick, force_rerun=args.force)

