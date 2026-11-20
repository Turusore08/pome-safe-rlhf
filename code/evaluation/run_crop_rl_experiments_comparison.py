#!/usr/bin/env python3
"""
End-to-End Safe-RLHF & Multi-Algorithm Experiment Comparison Across Photo Dataset Variants:
  1. Baseline (foto/ Uncropped Full Frame - existing d_hist.pt & vision_encoder.pt)
  2. FOTO_crop_K1-K16 (128x128 Main Camera Liquid ROI Crop - 16/16 batches, 5,680 transitions)
  3. FOTO1_crop_16runs (168x168 Side Camera OBSBOT ROI Crop - all 16 batches, 5 missing zero-imputed)
  4. FOTO1_crop_11runs (168x168 Side Camera OBSBOT ROI Crop - pure 11 active batches, 3,922 transitions)
  5. Dual_Crop_Fusion (Unified ConvNeXt encoder fusing FOTO_crop + FOTO1_crop - 16/16 batches)

Runs:
  - Stage 1: Builds VisionEncoder checkpoints & offline datasets (d_hist_*.pt), plus held-out LOCO/OOD VisionEncoder R2
  - Stage 2: Parallel multi-seed training (seeds=[42, 101, 202]) across 6 distinct algorithms
             (safe_rlhf, lagrangian_sac, dppo, hcr, cql, sac) on each dataset variant
  - Stage 3: Evaluates both standard POMEReactorEnv closed-loop & real-photo-grounded closed-loop rollouts,
             saves CSV benchmarks, and renders 300-DPI mentor-ready figures.
"""

import os
import sys
import json
import time
import shutil
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import LOCO_TRAIN_CULTIVATIONS, OOD_CULTIVATIONS, REAL_DATA_DIR
from models.vision_encoder import VisionEncoder
from data.offline_dataset import OfflineDataset
from evaluation.metrics import compute_lapse_metrics, compute_step_violation_magnitude


DATASET_ROOT = "/home/turusore08/Desktop/KP/Data Kultivasi Mikroalga K1-16"
if not os.path.exists(DATASET_ROOT):
    DATASET_ROOT = REAL_DATA_DIR

SENSOR_JAM_CSV = os.path.join(DATASET_ROOT, "gabungan", "GABUNGAN_sensor_jam.csv")
EVAL_DIR = os.path.join(CODE_DIR, "evaluation")
FIG_DIR = os.path.join(EVAL_DIR, "figures")
DATA_DIR = os.path.join(CODE_DIR, "data")
CKPT_DIR = os.path.join(CODE_DIR, "checkpoints")
ARTIFACT_DIR = "/home/turusore08/.gemini/antigravity/brain/aa15dda2-5b39-4e2c-9f57-28bc8edf6488"
OBSIDIAN_ROOT = "/home/turusore08/Documents/Obsidian Vault/POME-Safe-RLHF"

BASELINE_DHIST_PATH = os.path.join(DATA_DIR, "d_hist.pt")
BASELINE_VENC_PATH = os.path.join(CKPT_DIR, "vision_encoder.pt")
PHOTO_BANK_PATH = os.path.join(DATA_DIR, "real_photo_eval_bank.pt")

DATASET_VARIANTS: List[Dict[str, Any]] = [
    {
        "key": "Baseline_foto",
        "label": "Baseline (foto/ Uncropped)",
        "short": "Baseline\n(foto/)",
        "photo_source": "Baseline",
        "dhist_path": BASELINE_DHIST_PATH,
        "venc_path": os.path.join(CKPT_DIR, "vision_encoder_baseline.pt"),
        "color": "#64748b",
        "batches_active": "16/16",
    },
    {
        "key": "FOTO_crop_K1-K16",
        "label": "FOTO_crop_K1-K16 (128x128)",
        "short": "FOTO_crop\n(16/16 Batches)",
        "photo_source": "FOTO_crop",
        "dhist_path": os.path.join(DATA_DIR, "d_hist_foto_crop.pt"),
        "venc_path": os.path.join(CKPT_DIR, "vision_encoder_foto_crop.pt"),
        "color": "#0284c7",
        "batches_active": "16/16",
    },
    {
        "key": "FOTO1_crop_16runs",
        "label": "FOTO1_crop (16-Batch Zero-Imputed)",
        "short": "FOTO1_crop\n(16 Batches, Zeros)",
        "photo_source": "FOTO1_crop",
        "dhist_path": os.path.join(DATA_DIR, "d_hist_foto1_crop.pt"),
        "venc_path": os.path.join(CKPT_DIR, "vision_encoder_foto1_crop.pt"),
        "color": "#f59e0b",
        "batches_active": "11/16 (5 zero-filled)",
    },
    {
        "key": "FOTO1_crop_11runs",
        "label": "FOTO1_crop (11-Batch Active Pure)",
        "short": "FOTO1_crop\n(11 Active Pure)",
        "photo_source": "FOTO1_crop",
        "dhist_path": os.path.join(DATA_DIR, "d_hist_foto1_crop_11runs.pt"),
        "venc_path": os.path.join(CKPT_DIR, "vision_encoder_foto1_crop.pt"),
        "color": "#059669",
        "batches_active": "11/11 (4,180 trans)",
    },
    {
        "key": "Dual_Crop_Fusion",
        "label": "Dual-Crop Fusion (FOTO + FOTO1)",
        "short": "Dual-Crop\n(Fusion 16/16)",
        "photo_source": "Dual_Crop",
        "dhist_path": os.path.join(DATA_DIR, "d_hist_dual_crop.pt"),
        "venc_path": os.path.join(CKPT_DIR, "vision_encoder_dual_crop.pt"),
        "color": "#7c3aed",
        "batches_active": "16/16 (Hybrid ROI)",
    },
]

ALGORITHMS_TO_RUN: List[Dict[str, Any]] = [
    {
        "key": "safe_rlhf",
        "name": "Safe RLHF (Cal-QL+SAC-KL+RAS)",
        "short": "Safe RLHF",
        "params": {
            "kl_beta": 0.04, "adaptive_kl": True, "target_kl": 1.20,
            "safety_budget": 0.10, "lambda_dual_init": 2.0, "eta_dual": 0.50,
            "ensemble_size_m": 4, "num_candidates_n": 16, "alpha_cql": 1.0,
            "steps_online": 3840,
        },
    },
    {
        "key": "lagrangian_sac",
        "name": "Lagrangian-SAC (M=1)",
        "short": "Lag-SAC",
        "params": {
            "kl_beta": 0.25, "safety_budget": 0.10, "lambda_dual_init": 2.0,
            "eta_dual": 0.05, "ensemble_size_m": 1, "num_candidates_n": 16,
            "steps_online": 3840,
        },
    },
    {
        "key": "dppo",
        "name": "DPPO (Preference Opt)",
        "short": "DPPO",
        "params": {"num_candidates_n": 1},
    },
    {
        "key": "hcr",
        "name": "HCR (SFT Cloning)",
        "short": "HCR/SFT",
        "params": {"epochs_sft": 15, "num_candidates_n": 1},
    },
    {
        "key": "cql",
        "name": "CQL (Offline Only)",
        "short": "CQL",
        "params": {"alpha_cql": 1.0, "num_candidates_n": 1},
    },
    {
        "key": "sac",
        "name": "Standard SAC (Unconstrained)",
        "short": "Std SAC",
        "params": {"kl_beta": 0.0, "adaptive_kl": False, "num_candidates_n": 1, "steps_online": 3840},
    },
]

FOTO1_ACTIVE_K: List[int] = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 14]
IMG_TRANSFORM = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_photo_index_by_source(source_key: str) -> Dict[int, pd.DataFrame]:
    """Build per-cultivation timestamp -> photo_path index for Baseline, FOTO_crop, or FOTO1_crop."""
    indices: Dict[int, pd.DataFrame] = {}
    for k in range(1, 17):
        k_str = f"K{k:02d}"
        if source_key == "Baseline":
            csv_path = os.path.join(DATASET_ROOT, "foto", k_str, f"{k_str}_daftar_foto.csv")
            if not os.path.exists(csv_path):
                continue
            df_idx = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "dalam_periode_kultivasi" in df_idx.columns:
                df_idx = df_idx[df_idx["dalam_periode_kultivasi"] == True].copy()
            df_idx["dt"] = pd.to_datetime(df_idx["waktu"])
            df_idx["photo_path"] = [
                os.path.join(DATASET_ROOT, "foto", k_str, str(cam), str(fname))
                for cam, fname in zip(df_idx["kamera"], df_idx["file"])
            ]
        elif source_key == "FOTO_crop":
            csv_path = os.path.join(DATASET_ROOT, "FOTO_crop_K1-K16", k_str, f"{k_str}_daftar_crop.csv")
            if not os.path.exists(csv_path):
                continue
            df_idx = pd.read_csv(csv_path, encoding="utf-8-sig")
            df_idx["dt"] = pd.to_datetime(df_idx["waktu"])
            df_idx["photo_path"] = [
                os.path.join(DATASET_ROOT, "FOTO_crop_K1-K16", k_str, str(fc))
                for fc in df_idx["file_crop"]
            ]
        elif source_key == "FOTO1_crop":
            csv_path = os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "gambar", k_str, f"{k_str}_daftar_crop.csv")
            if not os.path.exists(csv_path):
                continue
            df_idx = pd.read_csv(csv_path, encoding="utf-8-sig")
            df_idx["dt"] = pd.to_datetime(df_idx["waktu"])
            df_idx["photo_path"] = [
                os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "gambar", k_str, str(fc))
                for fc in df_idx["file_crop"]
            ]
        elif source_key == "Dual_Crop":
            idx_a = build_photo_index_by_source("FOTO_crop").get(k)
            idx_b = build_photo_index_by_source("FOTO1_crop").get(k)
            frames = [x for x in (idx_a, idx_b) if x is not None and len(x) > 0]
            if not frames:
                continue
            df_idx = pd.concat(frames, ignore_index=True)
        else:
            raise ValueError(f"Unknown source_key: {source_key}")

        if len(df_idx) > 0:
            indices[k] = df_idx[["dt", "photo_path"]].sort_values("dt").reset_index(drop=True)
    return indices


def _load_image_tensor(p_path: str) -> Optional[torch.Tensor]:
    """Load RGB image safely with draft resize for large raw JPEGs and zero crash on corrupt frames."""
    if not p_path or not os.path.exists(p_path):
        return None
    try:
        with Image.open(p_path) as im:
            im.draft("RGB", (128, 128))
            return IMG_TRANSFORM(im.convert("RGB"))
    except Exception:
        return None


def _sample_paired_vision_tensors(
    df: pd.DataFrame,
    photo_indices: Dict[int, pd.DataFrame],
    cultivations: List[int],
    max_samples: int = 640,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Collect evenly spaced (image, [log1p(COD), log1p(TSS), OD680]) samples across cultivations."""
    imgs: List[torch.Tensor] = []
    targets: List[List[float]] = []
    per_cult_cap = max(16, max_samples // max(1, len(cultivations)))

    for cult_id in cultivations:
        df_p = photo_indices.get(cult_id)
        cult_df = df[df["kultivasi"] == cult_id].sort_values("dt").reset_index(drop=True)
        if df_p is None or len(df_p) == 0 or len(cult_df) == 0:
            continue

        merged = pd.merge_asof(
            cult_df[["dt", "1_cod(mg/L)", "1_tss(mg/L)", "od680_estimasi(AU)"]],
            df_p,
            on="dt",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        ).dropna(subset=["photo_path", "1_cod(mg/L)", "1_tss(mg/L)", "od680_estimasi(AU)"])

        if len(merged) == 0:
            continue

        idxs = np.linspace(0, len(merged) - 1, min(per_cult_cap, len(merged)), dtype=int)
        for idx in idxs:
            row = merged.iloc[idx]
            t_img = _load_image_tensor(str(row["photo_path"]))
            if t_img is None:
                continue
            targets.append([
                float(np.log1p(max(0.0, float(row["1_cod(mg/L)"])))),
                float(np.log1p(max(0.0, float(row["1_tss(mg/L)"])))),
                float(np.clip(float(row["od680_estimasi(AU)"]), 0.0, 5.0)),
            ])
            imgs.append(t_img)

    if len(imgs) < 16:
        return None, None
    return torch.stack(imgs), torch.tensor(targets, dtype=torch.float32)


def extract_diff_optical(x: torch.Tensor) -> torch.Tensor:
    """
    Extract 16 differentiable global Beer-Lambert & algal pigment features directly from
    ImageNet-normalized RGB tensors (B, 3, 64, 64), preserving color intensity lost in GroupNorm.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    rgb = torch.clamp(x * std + mean, 0.0, 1.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rm, gm, bm = r.mean(dim=(1, 2)), g.mean(dim=(1, 2)), b.mean(dim=(1, 2))
    rs, gs, bs = r.std(dim=(1, 2)), g.std(dim=(1, 2)), b.std(dim=(1, 2))
    exg = 2.0 * gm - rm - bm
    gr = gm / (rm + 1e-4)
    ngrdi = (gm - rm) / (gm + rm + 1e-4)
    bg = bm / (gm + 1e-4)
    dark = 1.0 - (rm + gm + bm) / 3.0
    dx = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    dy = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    c_rgb = rgb[:, :, 16:48, 16:48].mean(dim=(2, 3))
    return torch.stack(
        [rm, gm, bm, rs, gs, bs, exg, gr, ngrdi, bg, dark, dx, dy, c_rgb[:, 0], c_rgb[:, 1], c_rgb[:, 2]],
        dim=1,
    )


class OpticalConvNeXtEncoder(nn.Module):
    """
    Hybrid Beer-Lambert Optical + ConvNeXt Encoder (latent_dim=32) for cropped ROI liquid frames.
    Combines 16 global optical reflectance/pigment descriptors with 16-dim spatial ConvNeXt latents.
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = VisionEncoder(latent_dim=16)
        self.register_buffer("opt_mean", torch.zeros(1, 16))
        self.register_buffer("opt_std", torch.ones(1, 16))
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        opt_f = (extract_diff_optical(x) - self.opt_mean) / torch.clamp(self.opt_std, min=1e-4)
        spatial_f = self.conv(x)
        # Scale by 0.5 so LayerNorm unit-variance outputs match ~0.50 std of baseline d_hist.pt
        return self.mlp(torch.cat([spatial_f, opt_f], dim=1)) * 0.5


def train_and_evaluate_vision_encoder(
    source_key: str,
    df: pd.DataFrame,
    ckpt_path: str,
    epochs: int = 35,
    seed: int = 42,
    reuse_baseline_ckpt: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train (or load baseline) VisionEncoder on in-distribution batches (`LOCO_TRAIN_CULTIVATIONS`)
    and evaluate both internal 20% validation R2 and unseen OOD (K3, K15) R2.
    """
    photo_indices = build_photo_index_by_source(source_key)

    if reuse_baseline_ckpt and os.path.exists(BASELINE_VENC_PATH):
        encoder: nn.Module = VisionEncoder(latent_dim=32)
        ckpt = torch.load(BASELINE_VENC_PATH, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["model_state_dict"])
        reg_head = nn.Linear(32, 3)
        reg_head.load_state_dict(ckpt["reg_head_state_dict"])
        y_mean = ckpt["target_mean"].unsqueeze(0)
        y_std = ckpt["target_std"].unsqueeze(0)
        val_r2_saved = float(ckpt.get("val_r2", 0.0))
        val_mse_saved = float(ckpt.get("val_mse", 0.0))
        n_train = int(ckpt.get("n_train", 0))
        n_val = int(ckpt.get("n_val", 0))
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        shutil.copy2(BASELINE_VENC_PATH, ckpt_path)
    else:
        encoder = OpticalConvNeXtEncoder(latent_dim=32)
        x_all, y_all = _sample_paired_vision_tensors(df, photo_indices, LOCO_TRAIN_CULTIVATIONS, max_samples=1120)
        if x_all is None or y_all is None:
            raise RuntimeError(f"Insufficient images to train VisionEncoder for {source_key}")

        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        perm = rng.permutation(len(x_all))
        split = int(0.8 * len(perm))
        tr_idx, va_idx = perm[:split], perm[split:]

        with torch.no_grad():
            opt_tr = extract_diff_optical(x_all[tr_idx])
            encoder.opt_mean.copy_(opt_tr.mean(dim=0, keepdim=True))
            encoder.opt_std.copy_(torch.clamp(opt_tr.std(dim=0, keepdim=True), min=1e-4))

        y_mean = y_all[tr_idx].mean(dim=0, keepdim=True)
        y_std = torch.clamp(y_all[tr_idx].std(dim=0, keepdim=True), min=1e-4)
        y_norm = (y_all - y_mean) / y_std

        reg_head = nn.Linear(32, 3)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(reg_head.parameters()),
            lr=2e-3,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        batch_size = 64
        for _ in range(epochs):
            encoder.train()
            reg_head.train()
            ep_perm = tr_idx[rng.permutation(len(tr_idx))]
            for start in range(0, len(ep_perm), batch_size):
                b_ids = ep_perm[start : start + batch_size]
                preds = reg_head(encoder(x_all[b_ids]))
                loss = F.mse_loss(preds, y_norm[b_ids])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        encoder.eval()
        reg_head.eval()
        with torch.no_grad():
            val_pred = reg_head(encoder(x_all[va_idx]))
            val_true = y_norm[va_idx]
            val_mse_saved = float(F.mse_loss(val_pred, val_true).item())
            ss_res = float(((val_true - val_pred) ** 2).sum().item())
            ss_tot = float(((val_true - val_true.mean(dim=0, keepdim=True)) ** 2).sum().item())
            val_r2_saved = 1.0 - ss_res / max(1e-8, ss_tot)

        n_train, n_val = len(tr_idx), len(va_idx)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(
            {
                "encoder_type": "optical_convnext",
                "model_state_dict": encoder.state_dict(),
                "reg_head_state_dict": reg_head.state_dict(),
                "target_mean": y_mean.squeeze(0),
                "target_std": y_std.squeeze(0),
                "visual_dim": 32,
                "val_mse": val_mse_saved,
                "val_r2": val_r2_saved,
                "n_train": n_train,
                "n_val": n_val,
                "source_key": source_key,
            },
            ckpt_path,
        )

    # Evaluate on unseen OOD runs (K03, K15) or available OOD subset
    encoder.eval()
    reg_head.eval()
    x_ood, y_ood = _sample_paired_vision_tensors(df, photo_indices, OOD_CULTIVATIONS, max_samples=200)
    ood_r2 = np.nan
    ood_cod_r2 = np.nan
    ood_od680_r2 = np.nan
    if x_ood is not None and len(x_ood) >= 8:
        with torch.no_grad():
            pred_norm = reg_head(encoder(x_ood))
            pred_raw = (pred_norm * y_std + y_mean).numpy()
            true_raw = y_ood.numpy()
        ood_cod_r2 = float(r2_score(true_raw[:, 0], pred_raw[:, 0]))
        ood_od680_r2 = float(r2_score(true_raw[:, 2], pred_raw[:, 2]))
        ood_r2 = float(np.mean([ood_cod_r2, r2_score(true_raw[:, 1], pred_raw[:, 1]), ood_od680_r2]))

    stats = {
        "source_key": source_key,
        "ckpt_path": ckpt_path,
        "n_train": n_train,
        "n_val": n_val,
        "val_mse": round(val_mse_saved, 4),
        "val_r2": round(val_r2_saved, 4),
        "ood_r2": round(ood_r2, 4) if not np.isnan(ood_r2) else np.nan,
        "ood_cod_r2": round(ood_cod_r2, 4) if not np.isnan(ood_cod_r2) else np.nan,
        "ood_od680_r2": round(ood_od680_r2, 4) if not np.isnan(ood_od680_r2) else np.nan,
        "indexed_cultivations": len(photo_indices),
    }
    return encoder, stats


def _extract_cultivation_hz(
    cult_df: pd.DataFrame,
    df_p: Optional[pd.DataFrame],
    encoder: VisionEncoder,
    batch_size: int = 128,
) -> Tuple[np.ndarray, int]:
    """Extract 32-dim visual latents per hour using 2h tolerance + Zero-Order Hold (ZOH)."""
    n_rows = len(cult_df)
    hz = np.zeros((n_rows, 32), dtype=np.float32)
    matched = 0

    if df_p is not None and len(df_p) > 0:
        merged = pd.merge_asof(
            cult_df[["dt"]].reset_index(drop=True),
            df_p,
            on="dt",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )
        tensors: List[torch.Tensor] = []
        idxs: List[int] = []
        for idx in range(n_rows):
            p_path = merged["photo_path"].iloc[idx]
            if isinstance(p_path, str) and os.path.exists(p_path):
                t_img = _load_image_tensor(p_path)
                if t_img is not None:
                    tensors.append(t_img)
                    idxs.append(idx)
            if len(tensors) >= batch_size or (idx == n_rows - 1 and len(tensors) > 0):
                with torch.no_grad():
                    out = encoder(torch.stack(tensors)).float().cpu().numpy()
                for b_i, orig_i in enumerate(idxs):
                    hz[orig_i] = out[b_i]
                    matched += 1
                tensors.clear()
                idxs.clear()

        # Back-fill then forward-fill via ZOH
        first_valid = None
        for idx in range(n_rows):
            if not np.all(hz[idx] == 0.0):
                first_valid = hz[idx].copy()
                hz[:idx] = first_valid
                break
        if first_valid is not None:
            curr = first_valid
            for idx in range(n_rows):
                if np.all(hz[idx] == 0.0):
                    hz[idx] = curr
                else:
                    curr = hz[idx].copy()

    return hz, matched


def assemble_variant_dhist_files(
    base_dhist: Dict[str, torch.Tensor],
    df: pd.DataFrame,
    enc_foto_crop: VisionEncoder,
    enc_foto1_crop: VisionEncoder,
    enc_dual_crop: VisionEncoder,
) -> Dict[str, Dict[str, Any]]:
    """
    Preserve exact 192-dim sensor windows, actions, rewards, dones, violations from base d_hist.pt,
    and replace the 32-dim visual slice ([:, 192:224]) per cultivation for each crop variant.
    """
    idx_foto = build_photo_index_by_source("FOTO_crop")
    idx_foto1 = build_photo_index_by_source("FOTO1_crop")

    def _build_from_hz_dict(hz_per_k: Dict[int, np.ndarray], save_path: str, subset_k: Optional[List[int]] = None):
        states = base_dhist["states"].clone()
        next_states = base_dhist["next_states"].clone()
        cids = base_dhist["cultivation_ids"].numpy()

        for k in range(1, 17):
            mask = np.where(cids == k)[0]
            hz_k = hz_per_k[k]
            assert len(mask) == len(hz_k) - 1, f"Row mismatch on K{k:02d}: {len(mask)} vs {len(hz_k)-1}"
            states[mask, 192:224] = torch.from_numpy(hz_k[:-1])
            next_states[mask, 192:224] = torch.from_numpy(hz_k[1:])

        out = {
            "states": states,
            "actions": base_dhist["actions"].clone(),
            "rewards": base_dhist["rewards"].clone(),
            "next_states": next_states,
            "dones": base_dhist["dones"].clone(),
            "violations": base_dhist["violations"].clone(),
            "cultivation_ids": base_dhist["cultivation_ids"].clone(),
            "lab_masks": base_dhist["lab_masks"].clone(),
        }
        if subset_k is not None:
            k_mask = torch.isin(out["cultivation_ids"], torch.tensor(subset_k, dtype=torch.long))
            out = {key: val[k_mask] for key, val in out.items()}

        torch.save(out, save_path)
        return len(out["states"]), int((out["states"][:, 192:224].abs().sum(dim=1) > 0).sum().item())

    hz_foto_k: Dict[int, np.ndarray] = {}
    hz_foto1_k: Dict[int, np.ndarray] = {}
    hz_dual_k: Dict[int, np.ndarray] = {}
    match_stats: Dict[str, int] = {"foto_matched": 0, "foto1_matched": 0, "dual_matched": 0}

    for k in range(1, 17):
        cult_df = df[df["kultivasi"] == k].sort_values("dt").reset_index(drop=True)
        hz_f, m_f = _extract_cultivation_hz(cult_df, idx_foto.get(k), enc_foto_crop)
        hz_f1, m_f1 = _extract_cultivation_hz(cult_df, idx_foto1.get(k), enc_foto1_crop)

        # Dual crop uses unified enc_dual_crop on both cameras and averages when both exist
        hz_df, _ = _extract_cultivation_hz(cult_df, idx_foto.get(k), enc_dual_crop)
        hz_df1, m_df1 = _extract_cultivation_hz(cult_df, idx_foto1.get(k), enc_dual_crop)
        has_f1 = np.linalg.norm(hz_df1, axis=1, keepdims=True) > 1e-6
        hz_dual = np.where(has_f1, 0.5 * (hz_df + hz_df1), hz_df).astype(np.float32)

        hz_foto_k[k] = hz_f
        hz_foto1_k[k] = hz_f1
        hz_dual_k[k] = hz_dual
        match_stats["foto_matched"] += m_f
        match_stats["foto1_matched"] += m_f1
        match_stats["dual_matched"] += int(np.sum(np.linalg.norm(hz_dual, axis=1) > 1e-6))

    summary = {}
    n_f, nz_f = _build_from_hz_dict(hz_foto_k, os.path.join(DATA_DIR, "d_hist_foto_crop.pt"))
    summary["FOTO_crop_K1-K16"] = {"transitions": n_f, "nonzero_hz_rows": nz_f, "direct_photo_hours": match_stats["foto_matched"]}

    n_f1_16, nz_f1_16 = _build_from_hz_dict(hz_foto1_k, os.path.join(DATA_DIR, "d_hist_foto1_crop.pt"))
    summary["FOTO1_crop_16runs"] = {"transitions": n_f1_16, "nonzero_hz_rows": nz_f1_16, "direct_photo_hours": match_stats["foto1_matched"]}

    n_f1_11, nz_f1_11 = _build_from_hz_dict(hz_foto1_k, os.path.join(DATA_DIR, "d_hist_foto1_crop_11runs.pt"), subset_k=FOTO1_ACTIVE_K)
    summary["FOTO1_crop_11runs"] = {"transitions": n_f1_11, "nonzero_hz_rows": nz_f1_11, "direct_photo_hours": match_stats["foto1_matched"]}

    n_d, nz_d = _build_from_hz_dict(hz_dual_k, os.path.join(DATA_DIR, "d_hist_dual_crop.pt"))
    summary["Dual_Crop_Fusion"] = {"transitions": n_d, "nonzero_hz_rows": nz_d, "direct_photo_hours": match_stats["dual_matched"]}
    return summary


def build_real_photo_eval_bank(
    df: pd.DataFrame,
    encoders: Dict[str, VisionEncoder],
    max_bank: int = 360,
) -> Dict[str, Any]:
    """
    Build a fast stratified in-memory lookup bank mapping physical bioreactor states
    (log1p(COD), log1p(TSS), pH) -> precomputed real-photo 32-dim latents hz for Track 2 evaluation.
    """
    bank: Dict[str, Any] = {}
    for src_key, enc in encoders.items():
        photo_idx = build_photo_index_by_source(src_key)
        all_rows = []
        for k in sorted(photo_idx.keys()):
            cult_df = df[df["kultivasi"] == k].sort_values("dt").reset_index(drop=True)
            merged = pd.merge_asof(
                cult_df[["dt", "1_cod(mg/L)", "1_tss(mg/L)", "1_ph", "kultivasi"]],
                photo_idx[k],
                on="dt",
                direction="nearest",
                tolerance=pd.Timedelta("2h"),
            ).dropna(subset=["photo_path", "1_cod(mg/L)", "1_tss(mg/L)", "1_ph"])
            if len(merged) > 0:
                all_rows.append(merged)
        cat_df = pd.concat(all_rows, ignore_index=True)
        # Stratify across COD quantiles so the bank covers raw (4200+) down to treated (<100)
        cat_df["cod_bin"] = pd.qcut(cat_df["1_cod(mg/L)"].rank(method="first"), q=min(20, len(cat_df)), labels=False)
        sampled_frames = []
        per_bin = max(2, max_bank // max(1, cat_df["cod_bin"].nunique()))
        for _, grp in cat_df.groupby("cod_bin"):
            idxs = np.linspace(0, len(grp) - 1, min(per_bin, len(grp)), dtype=int)
            sampled_frames.append(grp.iloc[idxs])
        samp_df = pd.concat(sampled_frames, ignore_index=True)

        states_feat = []
        tensors = []
        for _, r in samp_df.iterrows():
            t_img = _load_image_tensor(str(r["photo_path"]))
            if t_img is not None:
                tensors.append(t_img)
                states_feat.append([
                    float(np.log1p(max(0.0, float(r["1_cod(mg/L)"])))),
                    float(np.log1p(max(0.0, float(r["1_tss(mg/L)"])))),
                    float(r["1_ph"]),
                ])
        enc.eval()
        with torch.no_grad():
            hz_bank = enc(torch.stack(tensors)).float().cpu().numpy()
        bank[src_key] = {
            "state_feats": np.asarray(states_feat, dtype=np.float32),
            "hz_bank": hz_bank.astype(np.float32),
        }
    torch.save(bank, PHOTO_BANK_PATH)
    return bank


class _NormalizedVisionAdapter(nn.Module):
    """Applies ImageNet normalization to [0, 1] POMEReactorEnv synthetic swatches for OpticalConvNeXtEncoder."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner((x - self.mean) / self.std)


def _make_variant_env_class(venc_path: str, real_photo_bank: Optional[Dict[str, np.ndarray]] = None):
    """
    Create an isolated POMEReactorEnv subclass inside a worker process so each worker uses its
    exact dataset VisionEncoder (zero disk contention with checkpoints/vision_encoder.pt).
    """
    from environment.pome_reactor_env import POMEReactorEnv as BaseEnv

    ckpt = torch.load(venc_path, map_location="cpu", weights_only=False)
    vdim = int(ckpt.get("visual_dim", 32))
    if ckpt.get("encoder_type") == "optical_convnext":
        raw_enc: nn.Module = OpticalConvNeXtEncoder(latent_dim=vdim)
        raw_enc.load_state_dict(ckpt["model_state_dict"])
        raw_enc.eval()
        enc: nn.Module = _NormalizedVisionAdapter(raw_enc).eval()
    else:
        enc = VisionEncoder(latent_dim=vdim)
        enc.load_state_dict(ckpt["model_state_dict"])
        enc.eval()

    class VariantPOMEReactorEnv(BaseEnv):
        def __init__(self, *args, **kwargs):
            if kwargs.get("vision_encoder") is None:
                kwargs["vision_encoder"] = enc
            super().__init__(*args, **kwargs)
            if real_photo_bank is not None:
                feats = real_photo_bank["state_feats"]
                hz_arr = real_photo_bank["hz_bank"]
                feat_std = np.maximum(feats.std(axis=0, keepdims=True), 1e-4)

                def _real_photo_generate(cod: float, ph: float, turbidity: float, step: int, is_violated: bool = False):
                    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
                    est_tss = max(10.0, turbidity / 0.665)
                    q = np.array([[np.log1p(max(0.0, cod)), np.log1p(est_tss), float(ph)]], dtype=np.float32)
                    dists = np.sum(((feats - q) / feat_std) ** 2, axis=1)
                    best_idx = int(np.argmin(dists))
                    return dummy_img, hz_arr[best_idx].copy()

                self.vision_gen.generate_visual_state = _real_photo_generate  # type: ignore

    return VariantPOMEReactorEnv


def _worker_train_and_eval(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry point running isolated single-thread PyTorch RL training + dual evaluation."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)

    import experiments.evaluator as ev

    variant = task["variant"]
    algo_cfg = task["algo_cfg"]
    seed = int(task["seed"])
    algo = algo_cfg["key"]
    params = algo_cfg["params"]

    t0 = time.time()

    # Load offline corpus for this variant
    raw = torch.load(variant["dhist_path"], map_location="cpu", weights_only=False)
    dataset = OfflineDataset(**{k: v.numpy() for k, v in raw.items()})

    # Inject dataset-specific VisionEncoder into evaluator.POMEReactorEnv (process-local)
    VariantEnvSynthetic = _make_variant_env_class(variant["venc_path"], real_photo_bank=None)
    ev.POMEReactorEnv = VariantEnvSynthetic

    policy, r_critic, f_ens, _ = ev.train_algorithm_agent(
        algorithm=algo,
        dataset=dataset,
        params=params,
        state_dim=224,
        device=torch.device("cpu"),
        quick_mode=False,
        seed=seed,
    )
    train_sec = time.time() - t0

    # Evaluation 1: Standard POMEReactorEnv (synthetic swath + variant VisionEncoder)
    res_std = ev.evaluate_policy_on_bioreactor(
        policy=policy,
        reward_critic=r_critic,
        failure_ensemble=f_ens,
        num_candidates=int(params.get("num_candidates_n", 16)),
        lambda_dual=float(params.get("lambda_dual_init", 2.0)),
        safety_budget=float(params.get("safety_budget", 0.10)),
        gate_threshold=float(params.get("gate_threshold", 0.50)),
        device=torch.device("cpu"),
        max_steps=64,
        decision_horizon_hours=6.0,
        algorithm=algo,
        use_vision=True,
        seed=seed + 1000,
    )
    ep_g_std = float(sum(res_std["step_rig_rewards"]))
    lapse_std = compute_lapse_metrics(
        [{"states": res_std["state_history"], "actions": res_std["action_history"], "rewards": [ep_g_std]}],
        step_hours=6.0,
    )

    # Evaluation 2: Real-Photo-Grounded POMEReactorEnv (nearest real photograph + variant VisionEncoder)
    bank_all = torch.load(PHOTO_BANK_PATH, map_location="cpu", weights_only=False)
    bank_src = bank_all[variant["photo_source"]]
    VariantEnvRealPhoto = _make_variant_env_class(variant["venc_path"], real_photo_bank=bank_src)
    ev.POMEReactorEnv = VariantEnvRealPhoto

    res_real = ev.evaluate_policy_on_bioreactor(
        policy=policy,
        reward_critic=r_critic,
        failure_ensemble=f_ens,
        num_candidates=int(params.get("num_candidates_n", 16)),
        lambda_dual=float(params.get("lambda_dual_init", 2.0)),
        safety_budget=float(params.get("safety_budget", 0.10)),
        gate_threshold=float(params.get("gate_threshold", 0.50)),
        device=torch.device("cpu"),
        max_steps=64,
        decision_horizon_hours=6.0,
        algorithm=algo,
        use_vision=True,
        seed=seed + 1000,
    )
    ep_g_real = float(sum(res_real["step_rig_rewards"]))
    lapse_real = compute_lapse_metrics(
        [{"states": res_real["state_history"], "actions": res_real["action_history"], "rewards": [ep_g_real]}],
        step_hours=6.0,
    )

    # Evaluation 3: Offline Risk & Visual Sensitivity on dataset
    policy.eval()
    f_ens.eval()
    with torch.no_grad():
        s_all = dataset.states
        a_all = dataset.actions
        pred_act, _ = policy.sample(s_all, deterministic=True)
        # Visual sensitivity: perturb visual latent hz -> zeros and measure delta action norm
        s_novis = s_all.clone()
        s_novis[:, 192:224] = 0.0
        pred_novis, _ = policy.sample(s_novis, deterministic=True)
        act_scale = torch.tensor([6.0, 5.0, 100.0, 20.0, 5000.0])
        vis_sensitivity = float((((pred_act - pred_novis) / act_scale).abs().mean()).item())

        # Offline safety risk detection AUC on real dataset violations (if both classes exist)
        v_true = dataset.violations.numpy()
        fail_auc = np.nan
        if len(np.unique(v_true)) == 2:
            mean_risk, _ = f_ens(s_all, a_all)
            fail_auc = float(roc_auc_score(v_true, mean_risk.numpy()))

    row = {
        "dataset_key": variant["key"],
        "dataset_label": variant["label"],
        "algorithm_key": algo,
        "algorithm_name": algo_cfg["name"],
        "seed": seed,
        "train_time_sec": round(train_sec, 2),
        # Track 1: Standard POMEReactorEnv
        "std_return_g": round(ep_g_std, 4),
        "std_cod_removal_pct": round(float(res_std["cod_removal_pct"]), 4),
        "std_bod_removal_pct": round(float(res_std["bod_reduction_pct"]), 4),
        "std_tss_removal_pct": round(float(res_std["tss_removal_pct"]), 4),
        "std_final_cod_mgl": round(float(res_std["final_cod_mg_l"]), 2),
        "std_cost_idr": round(float(res_std["total_cost_idr"]), 2),
        "std_violation_rate_pct": round(float(res_std["violation_rate_pct"]), 2),
        "std_had_violation": int(bool(res_std["had_violation"])),
        "std_d_cv_mean": round(float(lapse_std["d_cv_mean"]), 4),
        "std_compliance_day": round(float(res_std["compliance_day"]), 2),
        "std_permenlh_compliant": int(bool(res_std["permenlh_compliant"])),
        "std_pareto_score": round(float(res_std["pareto_score"]), 4),
        # Track 2: Real-Photo-Grounded POMEReactorEnv
        "realphoto_return_g": round(ep_g_real, 4),
        "realphoto_cod_removal_pct": round(float(res_real["cod_removal_pct"]), 4),
        "realphoto_bod_removal_pct": round(float(res_real["bod_reduction_pct"]), 4),
        "realphoto_tss_removal_pct": round(float(res_real["tss_removal_pct"]), 4),
        "realphoto_cost_idr": round(float(res_real["total_cost_idr"]), 2),
        "realphoto_violation_rate_pct": round(float(res_real["violation_rate_pct"]), 2),
        "realphoto_had_violation": int(bool(res_real["had_violation"])),
        "realphoto_pareto_score": round(float(res_real["pareto_score"]), 4),
        # Diagnostics
        "visual_action_sensitivity": round(vis_sensitivity, 5),
        "offline_failure_auc": round(fail_auc, 4) if not np.isnan(fail_auc) else np.nan,
        # Trajectories for Representative Plot (only returned cleanly for JSON serialization)
        "std_cod_traj": json.dumps([round(float(x), 2) for x in res_std["cod_trajectory"]]),
        "std_ph_traj": json.dumps([round(float(x), 3) for x in res_std["ph_trajectory"]]),
        "std_do_traj": json.dumps([round(float(s["dissolved_oxygen"]), 3) for s in res_std["state_history"]]),
        "std_cum_g_traj": json.dumps([round(float(x), 3) for x in np.cumsum([0.0] + res_std["step_rig_rewards"])]),
        "realphoto_cod_traj": json.dumps([round(float(x), 2) for x in res_real["cod_trajectory"]]),
        "realphoto_ph_traj": json.dumps([round(float(x), 3) for x in res_real["ph_trajectory"]]),
        "realphoto_do_traj": json.dumps([round(float(s["dissolved_oxygen"]), 3) for s in res_real["state_history"]]),
        "realphoto_cum_g_traj": json.dumps([round(float(x), 3) for x in np.cumsum([0.0] + res_real["step_rig_rewards"])]),
    }
    return row


def render_comparison_figures(
    df_runs: pd.DataFrame,
    df_agg: pd.DataFrame,
    df_venc: pd.DataFrame,
):
    """Generate two 300-DPI publication figures comparing all 5 dataset variants."""
    os.makedirs(FIG_DIR, exist_ok=True)
    obs_assets = os.path.join(OBSIDIAN_ROOT, "06_Assets")
    os.makedirs(obs_assets, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#fcfcfd",
    })

    ds_order = [v["key"] for v in DATASET_VARIANTS]
    ds_short = {v["key"]: v["short"] for v in DATASET_VARIANTS}
    ds_tick = {
        "Baseline_foto": "Baseline\n(foto/)",
        "FOTO_crop_K1-K16": "FOTO_crop\n(16/16)",
        "FOTO1_crop_16runs": "FOTO1_crop\n(16, Zeros)",
        "FOTO1_crop_11runs": "FOTO1_crop\n(11 Pure)",
        "Dual_Crop_Fusion": "Dual-Crop\n(16/16)",
    }
    ds_colors = {v["key"]: v["color"] for v in DATASET_VARIANTS}
    algo_order = [a["key"] for a in ALGORITHMS_TO_RUN]
    algo_short = {a["key"]: a["short"] for a in ALGORITHMS_TO_RUN}

    # =========================================================================
    # Figure 5: 4-Panel RL Pipeline Benchmark Across Image Dataset Versions
    # =========================================================================
    fig = plt.figure(figsize=(16.5, 11.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.25)

    # Panel A: Primary Controller (Safe RLHF) Return G (Standard vs Real-Photo Loop) + Vision R2
    ax1 = fig.add_subplot(gs[0, 0])
    sub_rlhf = df_agg[df_agg["algorithm_key"] == "safe_rlhf"].set_index("dataset_key").loc[ds_order]
    x = np.arange(len(ds_order))
    w = 0.34
    bars1 = ax1.bar(x - w/2, sub_rlhf["std_return_g_mean"], width=w, yerr=sub_rlhf["std_return_g_std"],
                    capsize=3, color=[ds_colors[k] for k in ds_order], edgecolor="#1e293b", lw=0.8,
                    label="Standard POMEReactorEnv Loop")
    bars2 = ax1.bar(x + w/2, sub_rlhf["realphoto_return_g_mean"], width=w, yerr=sub_rlhf["realphoto_return_g_std"],
                    capsize=3, color=[ds_colors[k] for k in ds_order], edgecolor="#0f172a", lw=0.8,
                    alpha=0.52, hatch="///", label="Real-Photo-Grounded Loop")
    for i, b in enumerate(bars1):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + sub_rlhf["std_return_g_std"].iloc[i] + 1.5,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for i, b in enumerate(bars2):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + sub_rlhf["realphoto_return_g_std"].iloc[i] + 1.5,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([ds_tick[k] for k in ds_order])
    ax1.set_ylabel("Episode Return G (Mean ± SD, 3 Seeds)")
    ax1.set_title("A. Safe RLHF Episode Return G: Standard vs Real-Photo Loop")
    ax1.axhline(0, color="#94a3b8", lw=0.8, ls="--")
    ax1.grid(axis="y", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(min(ymin - 5, -5), ymax + 10)

    # Panel B: Heatmap of Return G across (6 Algorithms x 5 Datasets) - Real-Photo Loop
    ax2 = fig.add_subplot(gs[0, 1])
    piv_g = df_agg.pivot(index="algorithm_key", columns="dataset_key", values="realphoto_return_g_mean").loc[algo_order, ds_order]
    piv_vr = df_agg.pivot(index="algorithm_key", columns="dataset_key", values="realphoto_episode_viol_pct").loc[algo_order, ds_order]
    im = ax2.imshow(piv_g.values, cmap="RdYlGn", aspect="auto", vmin=-65, vmax=75)
    ax2.set_xticks(np.arange(len(ds_order)))
    ax2.set_xticklabels([ds_tick[k] for k in ds_order], fontsize=8.3)
    ax2.set_yticks(np.arange(len(algo_order)))
    ax2.set_yticklabels([algo_short[k] for k in algo_order])
    for r_i in range(len(algo_order)):
        for c_i in range(len(ds_order)):
            g_val = piv_g.values[r_i, c_i]
            vr_val = piv_vr.values[r_i, c_i]
            txt_col = "white" if (g_val < -25 or g_val > 58) else "#0f172a"
            ax2.text(c_i, r_i, f"G={g_val:.1f}\nVR={vr_val:.0f}%", ha="center", va="center",
                     fontsize=7.8, fontweight="bold", color=txt_col)
    cb = plt.colorbar(im, ax=ax2, fraction=0.045, pad=0.03)
    cb.set_label("Real-Photo Loop Return G", fontsize=8.5)
    ax2.set_title("B. Multi-Algorithm Matrix (Return G & Episode Crash Rate VR%)")

    # Panel C: Pareto Frontier across All Runs per Dataset Variant (Safe RLHF highlighted + All Algo Mean)
    ax3 = fig.add_subplot(gs[1, 0])
    other = df_agg[df_agg["algorithm_key"] != "safe_rlhf"]
    ax3.scatter(other["realphoto_cost_idr_mean"], other["realphoto_pareto_score_mean"],
                s=42, color="#94a3b8", alpha=0.5, marker="^", label="Baselines (HCR/DPPO/CQL/SAC)")
    label_positions = {
        "FOTO_crop_K1-K16": (1500, 91.5),
        "FOTO1_crop_11runs": (12200, 93.0),
        "Dual_Crop_Fusion": (22500, 89.5),
        "FOTO1_crop_16runs": (23500, 79.0),
        "Baseline_foto": (22500, 67.5),
    }
    for k in ds_order:
        s_sub = df_agg[(df_agg["dataset_key"] == k) & (df_agg["algorithm_key"] == "safe_rlhf")].iloc[0]
        cx = float(s_sub["realphoto_cost_idr_mean"])
        cy = float(s_sub["realphoto_pareto_score_mean"])
        ax3.scatter(
            cx, cy, s=190, color=ds_colors[k], edgecolor="#0f172a", linewidth=1.3, zorder=5,
            label=f"{ds_tick[k].replace(chr(10), ' ')}: Pareto={cy:.1f}, VR={s_sub['realphoto_episode_viol_pct']:.0f}%",
        )
        tx, ty = label_positions[k]
        ax3.annotate(
            f"{ds_tick[k].replace(chr(10), ' ')}\n({cy:.1f}, IDR {cx:,.0f})",
            (cx, cy),
            xytext=(tx, ty),
            fontsize=7.6, fontweight="bold", color=ds_colors[k],
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=ds_colors[k], alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=ds_colors[k], lw=1.0),
        )
    ax3.set_xlabel("Operating Cost per 16-Day Batch (IDR) — Lower is Better")
    ax3.set_ylabel("Pareto Composite Score (0–100) — Higher is Better")
    ax3.set_title("C. Cost vs Pareto Frontier (Safe RLHF Highlighted, Real-Photo Loop)")
    ax3.grid(alpha=0.35)
    ax3.legend(loc="lower right", fontsize=7.6, framealpha=0.94)
    ax3.set_xlim(-1000, 39500)
    ax3.set_ylim(-2, 102)

    # Panel D: Multi-Stage Comparison Normalized Bar Chart (Vision R2 vs Pareto vs Safety vs Visual Sensitivity)
    ax4 = fig.add_subplot(gs[1, 1])
    venc_map = df_venc.set_index("source_key")
    # Build clean 4-metric summary across variants for Safe RLHF
    metrics_rows = []
    for k in ds_order:
        src = [v["photo_source"] for v in DATASET_VARIANTS if v["key"] == k][0]
        r_agg = df_agg[(df_agg["dataset_key"] == k) & (df_agg["algorithm_key"] == "safe_rlhf")].iloc[0]
        val_r2 = float(venc_map.loc[src, "val_r2"]) * 100.0
        safety_rate = 100.0 - float(r_agg["realphoto_episode_viol_pct"])
        cod_rem = float(r_agg["realphoto_cod_removal_pct_mean"])
        pareto = float(r_agg["realphoto_pareto_score_mean"])
        metrics_rows.append((val_r2, cod_rem, safety_rate, pareto))
    mat = np.array(metrics_rows)  # (5, 4)
    metric_names = ["VisionEnc Val R² (%)", "COD Removal (%)", "Zero-Crash Safety (%)", "Pareto Score"]
    xm = np.arange(len(metric_names))
    bw = 0.15
    for idx_k, k in enumerate(ds_order):
        offset = (idx_k - 2.0) * bw
        vals = mat[idx_k]
        bars = ax4.bar(xm + offset, vals, width=bw, color=ds_colors[k], edgecolor="#1e293b", lw=0.6,
                       label=ds_short[k].replace("\n", " "))
        for b in bars:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.0f}", ha="center", va="bottom", fontsize=6.8)
    ax4.set_xticks(xm)
    ax4.set_xticklabels(metric_names)
    ax4.set_ylabel("Score / Percentage (%)")
    ax4.set_ylim(0, 115)
    ax4.set_title("D. 4-Pillar Scorecard: Perception, Efficacy, Safety & Pareto")
    ax4.grid(axis="y", alpha=0.35)
    ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7.8, framealpha=0.92)

    fig.suptitle(
        "POME Safe-RLHF End-to-End Experimental Benchmark: Baseline foto/ vs FOTO_crop vs FOTO1_crop vs Dual-Crop",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    fig5_path = os.path.join(FIG_DIR, "fig_crop_5_rl_pipeline_comparison.png")
    fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(fig5_path, os.path.join(ARTIFACT_DIR, os.path.basename(fig5_path)))
    shutil.copy2(fig5_path, os.path.join(obs_assets, os.path.basename(fig5_path)))

    # =========================================================================
    # Figure 6: Closed-Loop Bioreactor Trajectories for Safe RLHF (Seed 42)
    # =========================================================================
    fig2, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), sharex=True)
    axes = axes.flatten()
    sub_seed = df_runs[(df_runs["algorithm_key"] == "safe_rlhf") & (df_runs["seed"] == 42)].set_index("dataset_key")

    for k in ds_order:
        row = sub_seed.loc[k]
        cod_t = np.array(json.loads(row["realphoto_cod_traj"]), dtype=float)
        ph_t = np.array(json.loads(row["realphoto_ph_traj"]), dtype=float)
        do_t = np.array(json.loads(row["realphoto_do_traj"]), dtype=float)
        g_t = np.array(json.loads(row["realphoto_cum_g_traj"]), dtype=float)
        days = np.arange(len(cod_t)) * 6.0 / 24.0

        lbl = f"{ds_short[k].replace(chr(10), ' ')} (G={row['realphoto_return_g']:.1f}, COD Rem={row['realphoto_cod_removal_pct']:.1f}%)"
        col = ds_colors[k]
        lw = 2.4 if k in ("FOTO_crop_K1-K16", "Dual_Crop_Fusion") else 1.8
        ls = "--" if k == "Baseline_foto" else "-"

        axes[0].plot(days, cod_t, color=col, lw=lw, ls=ls, label=lbl)
        axes[1].plot(days, ph_t, color=col, lw=lw, ls=ls)
        axes[2].plot(days, do_t, color=col, lw=lw, ls=ls)
        axes[3].plot(days, g_t, color=col, lw=lw, ls=ls)

    axes[0].axhline(350.0, color="#dc2626", ls=":", lw=1.3, label="PerMenLH No.5/2014 COD Limit (350 mg/L)")
    axes[0].set_title("A. COD Concentration Trajectory (Real-Photo Closed-Loop)")
    axes[0].set_ylabel("COD (mg/L)")
    axes[0].grid(alpha=0.35)
    axes[0].legend(loc="upper right", fontsize=7.8, framealpha=0.93)

    axes[1].axhspan(5.5, 10.5, color="#bbf7d0", alpha=0.25, label="Tier-2 Biological pH Envelope (5.5–10.5)")
    axes[1].axhline(5.5, color="#dc2626", ls="--", lw=1.0)
    axes[1].axhline(10.5, color="#dc2626", ls="--", lw=1.0)
    axes[1].set_title("B. Reactor pH Stability vs Tier-2 Envelope")
    axes[1].set_ylabel("pH (-)")
    axes[1].grid(alpha=0.35)
    axes[1].legend(loc="lower right", fontsize=8)

    axes[2].axhline(0.5, color="#dc2626", ls="--", lw=1.1, label="Tier-2 Anoxic Floor (0.5 mg/L)")
    axes[2].set_title("C. Dissolved Oxygen (DO) Dynamics")
    axes[2].set_xlabel("Cultivation Time (Days)")
    axes[2].set_ylabel("Dissolved Oxygen (mg/L)")
    axes[2].set_ylim(-0.4, 17.8)
    axes[2].grid(alpha=0.35)
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].axhline(0.0, color="#94a3b8", ls=":", lw=1.0)
    axes[3].set_title("D. Cumulative Episode Return G Trajectory (Real-Photo Loop)")
    axes[3].set_xlabel("Cultivation Time (Days)")
    axes[3].set_ylabel("Cumulative Return G")
    axes[3].grid(alpha=0.35)

    fig2.suptitle(
        "Closed-Loop Safe-RLHF Trajectory Comparison Across Dataset Versions (Seed 42)",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig6_path = os.path.join(FIG_DIR, "fig_crop_6_closedloop_trajectories_by_dataset.png")
    fig2.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    shutil.copy2(fig6_path, os.path.join(ARTIFACT_DIR, os.path.basename(fig6_path)))
    shutil.copy2(fig6_path, os.path.join(obs_assets, os.path.basename(fig6_path)))
    print(f"[Figures] Saved {fig5_path} and {fig6_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Safe-RLHF & RL Baselines across Baseline vs Cropped photo datasets")
    parser.add_argument("--workers", type=int, default=14, help="Parallel CPU worker count")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset/VisionEncoder build if already generated")
    parser.add_argument("--plots-only", action="store_true", help="Re-render figures from existing CSV results")
    args = parser.parse_args()

    if args.plots_only:
        df_runs = pd.read_csv(os.path.join(EVAL_DIR, "crop_rl_experiments_all_runs.csv"))
        df_agg = pd.read_csv(os.path.join(EVAL_DIR, "crop_rl_experiments_comparison.csv"))
        df_venc = pd.read_csv(os.path.join(EVAL_DIR, "crop_rl_vision_encoder_stats.csv"))
        render_comparison_figures(df_runs, df_agg, df_venc)
        return

    t_global = time.time()
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    obs_csv_dir = os.path.join(OBSIDIAN_ROOT, "04_Experiments_and_Benchmarking", "csv_tables")
    os.makedirs(obs_csv_dir, exist_ok=True)

    df = pd.read_csv(SENSOR_JAM_CSV, encoding="utf-8-sig")
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["kultivasi", "dt"]).reset_index(drop=True)
    base_dhist = torch.load(BASELINE_DHIST_PATH, map_location="cpu", weights_only=False)

    # ------------------------------------------------------------------
    # STAGE 1: Train/Evaluate VisionEncoders + Assemble Variant D_hist
    # ------------------------------------------------------------------
    venc_stats_list = []
    venc_csv = os.path.join(EVAL_DIR, "crop_rl_vision_encoder_stats.csv")
    if args.skip_build and os.path.exists(venc_csv) and os.path.exists(PHOTO_BANK_PATH):
        print("[Stage 1] Reusing existing crop VisionEncoder checkpoints & datasets.")
        df_venc = pd.read_csv(venc_csv)
    else:
        print("=" * 80)
        print("[Stage 1/3] Training ConvNeXt VisionEncoders & Building Offline Corpora D_hist")
        print("=" * 80)
        enc_base, st_base = train_and_evaluate_vision_encoder(
            "Baseline", df, os.path.join(CKPT_DIR, "vision_encoder_baseline.pt"),
            epochs=12, seed=42, reuse_baseline_ckpt=True,
        )
        print("  Baseline   :", st_base)
        enc_foto, st_foto = train_and_evaluate_vision_encoder(
            "FOTO_crop", df, os.path.join(CKPT_DIR, "vision_encoder_foto_crop.pt"),
            epochs=35, seed=42, reuse_baseline_ckpt=False,
        )
        print("  FOTO_crop  :", st_foto)
        enc_foto1, st_foto1 = train_and_evaluate_vision_encoder(
            "FOTO1_crop", df, os.path.join(CKPT_DIR, "vision_encoder_foto1_crop.pt"),
            epochs=35, seed=42, reuse_baseline_ckpt=False,
        )
        print("  FOTO1_crop :", st_foto1)
        enc_dual, st_dual = train_and_evaluate_vision_encoder(
            "Dual_Crop", df, os.path.join(CKPT_DIR, "vision_encoder_dual_crop.pt"),
            epochs=35, seed=42, reuse_baseline_ckpt=False,
        )
        print("  Dual_Crop  :", st_dual)

        venc_stats_list = [st_base, st_foto, st_foto1, st_dual]
        df_venc = pd.DataFrame(venc_stats_list)
        df_venc.to_csv(venc_csv, index=False)
        shutil.copy2(venc_csv, os.path.join(obs_csv_dir, os.path.basename(venc_csv)))

        print("\nAssembling 224-dim D_hist corpora (preserving 192-dim sensors & updating 32-dim vision)...")
        dhist_summary = assemble_variant_dhist_files(base_dhist, df, enc_foto, enc_foto1, enc_dual)
        for k, v in dhist_summary.items():
            print(f"  {k:20s}: {v}")

        print("\nBuilding real-photo retrieval evaluation bank (Track 2 Real-Photo Loop)...")
        build_real_photo_eval_bank(
            df,
            {"Baseline": enc_base, "FOTO_crop": enc_foto, "FOTO1_crop": enc_foto1, "Dual_Crop": enc_dual},
            max_bank=360,
        )

    # ------------------------------------------------------------------
    # STAGE 2: Parallel Multi-Seed Multi-Algorithm RL Benchmark
    # ------------------------------------------------------------------
    seeds = [42, 101, 202]
    tasks = []
    for var in DATASET_VARIANTS:
        for algo in ALGORITHMS_TO_RUN:
            for sd in seeds:
                tasks.append({"variant": var, "algo_cfg": algo, "seed": sd})

    print("\n" + "=" * 80)
    print(f"[Stage 2/3] Launching {len(tasks)} Full RL Training + Dual Evaluation Runs across {args.workers} Workers")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        fut_map = {ex.submit(_worker_train_and_eval, t): t for t in tasks}
        done_cnt = 0
        for fut in as_completed(fut_map):
            t_info = fut_map[fut]
            res = fut.result()
            results.append(res)
            done_cnt += 1
            if done_cnt % 10 == 0 or done_cnt == len(tasks):
                print(
                    f"  Progress [{done_cnt:2d}/{len(tasks)}] | Last: {res['dataset_key']} / {res['algorithm_key']} (seed={res['seed']}) "
                    f"-> Std_G={res['std_return_g']:.2f}, Real_G={res['realphoto_return_g']:.2f}, VR={res['realphoto_violation_rate_pct']:.1f}% ({res['train_time_sec']}s)"
                )

    df_runs = pd.DataFrame(results)
    runs_csv = os.path.join(EVAL_DIR, "crop_rl_experiments_all_runs.csv")
    df_runs.to_csv(runs_csv, index=False)

    # ------------------------------------------------------------------
    # STAGE 3: Aggregate Benchmarks, Summary Ranking & Figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Stage 3/3] Aggregating Comparison CSVs & Rendering 300-DPI Figures")
    print("=" * 80)

    agg_rows = []
    for var in DATASET_VARIANTS:
        for algo in ALGORITHMS_TO_RUN:
            sub = df_runs[(df_runs["dataset_key"] == var["key"]) & (df_runs["algorithm_key"] == algo["key"])]
            agg_rows.append({
                "dataset_key": var["key"],
                "dataset_label": var["label"],
                "batches_active": var["batches_active"],
                "algorithm_key": algo["key"],
                "algorithm_name": algo["name"],
                "n_seeds": len(sub),
                "std_return_g_mean": round(float(sub["std_return_g"].mean()), 2),
                "std_return_g_std": round(float(sub["std_return_g"].std(ddof=0)), 2),
                "std_cod_removal_pct_mean": round(float(sub["std_cod_removal_pct"].mean()), 2),
                "std_bod_removal_pct_mean": round(float(sub["std_bod_removal_pct"].mean()), 2),
                "std_tss_removal_pct_mean": round(float(sub["std_tss_removal_pct"].mean()), 2),
                "std_cost_idr_mean": round(float(sub["std_cost_idr"].mean()), 1),
                "std_episode_viol_pct": round(float(sub["std_had_violation"].mean() * 100.0), 1),
                "std_pareto_score_mean": round(float(sub["std_pareto_score"].mean()), 2),
                "realphoto_return_g_mean": round(float(sub["realphoto_return_g"].mean()), 2),
                "realphoto_return_g_std": round(float(sub["realphoto_return_g"].std(ddof=0)), 2),
                "realphoto_cod_removal_pct_mean": round(float(sub["realphoto_cod_removal_pct"].mean()), 2),
                "realphoto_bod_removal_pct_mean": round(float(sub["realphoto_bod_removal_pct"].mean()), 2),
                "realphoto_tss_removal_pct_mean": round(float(sub["realphoto_tss_removal_pct"].mean()), 2),
                "realphoto_cost_idr_mean": round(float(sub["realphoto_cost_idr"].mean()), 1),
                "realphoto_episode_viol_pct": round(float(sub["realphoto_had_violation"].mean() * 100.0), 1),
                "realphoto_pareto_score_mean": round(float(sub["realphoto_pareto_score"].mean()), 2),
                "visual_action_sensitivity_mean": round(float(sub["visual_action_sensitivity"].mean()), 5),
                "offline_failure_auc_mean": round(float(sub["offline_failure_auc"].mean()), 4),
            })

    df_agg = pd.DataFrame(agg_rows)
    comp_csv = os.path.join(EVAL_DIR, "crop_rl_experiments_comparison.csv")
    df_agg.to_csv(comp_csv, index=False)
    shutil.copy2(comp_csv, os.path.join(obs_csv_dir, os.path.basename(comp_csv)))

    # Dataset-level summary comparing Safe RLHF hero + Cross-Algorithm Averages + VisionEncoder R2
    venc_map = df_venc.set_index("source_key")
    sum_rows = []
    for var in DATASET_VARIANTS:
        k = var["key"]
        src = var["photo_source"]
        sub_all = df_agg[df_agg["dataset_key"] == k]
        sub_hero = sub_all[sub_all["algorithm_key"] == "safe_rlhf"].iloc[0]
        sum_rows.append({
            "dataset_key": k,
            "dataset_label": var["label"],
            "batches_active": var["batches_active"],
            "vision_val_r2": float(venc_map.loc[src, "val_r2"]),
            "vision_ood_r2": float(venc_map.loc[src, "ood_r2"]),
            "hero_safe_rlhf_std_G": sub_hero["std_return_g_mean"],
            "hero_safe_rlhf_realphoto_G": sub_hero["realphoto_return_g_mean"],
            "hero_safe_rlhf_realphoto_COD_rem": sub_hero["realphoto_cod_removal_pct_mean"],
            "hero_safe_rlhf_realphoto_cost_idr": sub_hero["realphoto_cost_idr_mean"],
            "hero_safe_rlhf_realphoto_VR_pct": sub_hero["realphoto_episode_viol_pct"],
            "hero_safe_rlhf_realphoto_Pareto": sub_hero["realphoto_pareto_score_mean"],
            "hero_visual_sensitivity": sub_hero["visual_action_sensitivity_mean"],
            "hero_failure_critic_auc": sub_hero["offline_failure_auc_mean"],
            "all6_algos_mean_realphoto_G": round(float(sub_all["realphoto_return_g_mean"].mean()), 2),
            "all6_algos_mean_realphoto_Pareto": round(float(sub_all["realphoto_pareto_score_mean"].mean()), 2),
            "all6_algos_mean_VR_pct": round(float(sub_all["realphoto_episode_viol_pct"].mean()), 1),
        })
    df_sum = pd.DataFrame(sum_rows)
    sum_csv = os.path.join(EVAL_DIR, "crop_rl_summary_by_dataset.csv")
    df_sum.to_csv(sum_csv, index=False)
    shutil.copy2(sum_csv, os.path.join(obs_csv_dir, os.path.basename(sum_csv)))

    render_comparison_figures(df_runs, df_agg, df_venc)

    print("\n=== SUMMARY BY DATASET VARIANT ===")
    print(df_sum.to_string(index=False))
    print(f"\nCompleted full pipeline in {time.time() - t_global:.1f}s.")


if __name__ == "__main__":
    main()
