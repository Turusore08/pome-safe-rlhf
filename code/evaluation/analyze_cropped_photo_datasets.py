#!/usr/bin/env python3
"""
Feature Extraction, Importance Mapping, Feature Selection, and 3-Way Benchmark
Comparing POME Microalgae Photobioreactor Image Datasets:
  1. Baseline (foto/ Uncropped Full Frame: FOTO + FOTO1 merged, 93,134 in-period images)
  2. FOTO_crop_K1-K16 (128x128 Liquid ROI Crop from main camera FOTO, 61,879 images)
  3. FOTO1_crop_K1-K16 (168x168 Quality-Cleaned Liquid ROI Crop from OBSBOT camera FOTO1, 27,703 images)

Outputs CSV tables and 4 mentor-ready 300-DPI figures to code/evaluation/.
"""

import os
import sys
import glob
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, skew
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torchvision.transforms as T

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import LOCO_TRAIN_CULTIVATIONS, OOD_CULTIVATIONS, REAL_DATA_DIR
from models.vision_encoder import VisionEncoder

# Exact paths verified on disk
DATASET_ROOT = "/home/turusore08/Desktop/KP/Data Kultivasi Mikroalga K1-16"
if not os.path.exists(DATASET_ROOT):
    DATASET_ROOT = REAL_DATA_DIR

SENSOR_JAM_CSV = os.path.join(DATASET_ROOT, "gabungan", "GABUNGAN_sensor_jam.csv")
EVAL_DIR = os.path.join(CODE_DIR, "evaluation")
FIG_DIR = os.path.join(EVAL_DIR, "figures")
ARTIFACT_DIR = "/home/turusore08/.gemini/antigravity/brain/aa15dda2-5b39-4e2c-9f57-28bc8edf6488"
VISION_CKPT = os.path.join(CODE_DIR, "checkpoints", "vision_encoder.pt")

OPTICAL_FEATURES: List[str] = [
    # Group A: RGB Reflectance (6)
    "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
    # Group B: Algal & Chlorophyll Pigment Indices (4)
    "ExG_index", "G_R_ratio", "NGRDI", "B_G_ratio",
    # Group C: Perceptual Color Spaces HSV & CIE L*a*b* (6)
    "HSV_H_mean", "HSV_S_mean", "HSV_V_mean", "LAB_L_mean", "LAB_a_mean", "LAB_b_mean",
    # Group D: Optical Absorbance & Spatial Texture/Clutter (4)
    "Optical_Darkness", "Shannon_Entropy", "Gradient_Edge_Energy", "Center_Border_Contrast",
]

FEATURE_GROUPS: Dict[str, str] = {
    "R_mean": "RGB Reflectance", "G_mean": "RGB Reflectance", "B_mean": "RGB Reflectance",
    "R_std": "RGB Reflectance", "G_std": "RGB Reflectance", "B_std": "RGB Reflectance",
    "ExG_index": "Algal Pigment Indices", "G_R_ratio": "Algal Pigment Indices",
    "NGRDI": "Algal Pigment Indices", "B_G_ratio": "Algal Pigment Indices",
    "HSV_H_mean": "HSV & CIE L*a*b*", "HSV_S_mean": "HSV & CIE L*a*b*", "HSV_V_mean": "HSV & CIE L*a*b*",
    "LAB_L_mean": "HSV & CIE L*a*b*", "LAB_a_mean": "HSV & CIE L*a*b*", "LAB_b_mean": "HSV & CIE L*a*b*",
    "Optical_Darkness": "Optical & Spatial Texture", "Shannon_Entropy": "Optical & Spatial Texture",
    "Gradient_Edge_Energy": "Optical & Spatial Texture", "Center_Border_Contrast": "Optical & Spatial Texture",
}

TARGET_SPECS: List[Tuple[str, str]] = [
    ("target_od680", "OD680 / Biomass (AU)"),
    ("target_log_cod", "log1p(COD) (mg/L)"),
    ("target_log_tss", "log1p(TSS) (mg/L)"),
    ("target_ph", "Water pH (-)"),
]

DATASET_CONFIGS: List[Dict[str, str]] = [
    {"key": "Baseline", "label": "Baseline (foto/ Uncropped)", "short": "Baseline\n(Uncropped)", "color": "#64748b"},
    {"key": "FOTO_crop", "label": "FOTO_crop_K1-K16 (128x128)", "short": "FOTO_crop\n(128x128)", "color": "#0284c7"},
    {"key": "FOTO1_crop", "label": "FOTO1_crop_K1-K16 (168x168)", "short": "FOTO1_crop\n(168x168)", "color": "#059669"},
]

DATASET_COLORS: Dict[str, str] = {
    "Baseline": "#64748b",
    "FOTO_crop": "#0284c7",
    "FOTO1_crop": "#059669",
    "Dual_Crop": "#7c3aed",
}


def load_hourly_telemetry() -> pd.DataFrame:
    """Load and prepare 5,696 hourly telemetry rows across K01-K16."""
    df = pd.read_csv(SENSOR_JAM_CSV, encoding="utf-8-sig")
    df["dt"] = pd.to_datetime(df["timestamp"])
    df["kultivasi"] = df["kultivasi"].astype(int)
    df["target_od680"] = df["od680_estimasi(AU)"].astype(float)
    df["target_log_cod"] = np.log1p(np.clip(df["1_cod(mg/L)"].astype(float), 0.0, None))
    df["target_log_tss"] = np.log1p(np.clip(df["1_tss(mg/L)"].astype(float), 0.0, None))
    df["target_ph"] = df["1_ph"].astype(float)
    df["row_uid"] = df["kultivasi"].astype(str) + "_" + df["dt"].dt.strftime("%Y%m%d%H%M%S")
    return df.sort_values(["kultivasi", "dt"]).reset_index(drop=True)


def build_dataset_index(dataset_key: str) -> Tuple[Dict[int, pd.DataFrame], Dict[int, int]]:
    """
    Build per-cultivation timestamp -> image path index for the requested dataset.
    Handles exact disk path quirks:
      - Baseline: foto/Kxx/Kxx_daftar_foto.csv -> foto/Kxx/<kamera>/<file>
      - FOTO_crop: FOTO_crop_K1-K16/Kxx/Kxx_daftar_crop.csv -> FOTO_crop_K1-K16/Kxx/<file_crop>
      - FOTO1_crop: FOTO1_crop_K1-K16/gambar/Kxx/Kxx_daftar_crop.csv -> FOTO1_crop_K1-K16/gambar/Kxx/<file_crop>
    """
    indices: Dict[int, pd.DataFrame] = {}
    raw_counts: Dict[int, int] = {}

    for k in range(1, 17):
        k_str = f"K{k:02d}"
        if dataset_key == "Baseline":
            csv_path = os.path.join(DATASET_ROOT, "foto", k_str, f"{k_str}_daftar_foto.csv")
            if not os.path.exists(csv_path):
                raw_counts[k] = 0
                continue
            df_idx = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "dalam_periode_kultivasi" in df_idx.columns:
                df_idx = df_idx[df_idx["dalam_periode_kultivasi"] == True].copy()
            df_idx["dt"] = pd.to_datetime(df_idx["waktu"])
            df_idx["photo_path"] = [
                os.path.join(DATASET_ROOT, "foto", k_str, str(cam), str(fname))
                for cam, fname in zip(df_idx["kamera"], df_idx["file"])
            ]
            df_idx["x0"] = np.nan
            df_idx["y0"] = np.nan
            df_idx["x1"] = np.nan
            df_idx["y1"] = np.nan
            df_idx["cam_src"] = df_idx["kamera"].astype(str)
        elif dataset_key == "FOTO_crop":
            csv_path = os.path.join(DATASET_ROOT, "FOTO_crop_K1-K16", k_str, f"{k_str}_daftar_crop.csv")
            if not os.path.exists(csv_path):
                raw_counts[k] = 0
                continue
            df_idx = pd.read_csv(csv_path, encoding="utf-8-sig")
            df_idx["dt"] = pd.to_datetime(df_idx["waktu"])
            df_idx["photo_path"] = [
                os.path.join(DATASET_ROOT, "FOTO_crop_K1-K16", k_str, str(fc))
                for fc in df_idx["file_crop"]
            ]
            df_idx["cam_src"] = "FOTO_crop"
        elif dataset_key == "FOTO1_crop":
            csv_path = os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "gambar", k_str, f"{k_str}_daftar_crop.csv")
            if not os.path.exists(csv_path):
                raw_counts[k] = 0
                continue
            df_idx = pd.read_csv(csv_path, encoding="utf-8-sig")
            df_idx["dt"] = pd.to_datetime(df_idx["waktu"])
            df_idx["photo_path"] = [
                os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "gambar", k_str, str(fc))
                for fc in df_idx["file_crop"]
            ]
            df_idx["cam_src"] = "FOTO1_crop"
        else:
            raise ValueError(f"Unknown dataset_key: {dataset_key}")

        raw_counts[k] = len(df_idx)
        if len(df_idx) > 0:
            cols = ["dt", "photo_path", "x0", "y0", "x1", "y1", "cam_src"]
            indices[k] = df_idx.sort_values("dt")[cols].reset_index(drop=True)

    return indices, raw_counts


def align_hourly_images(df_jam: pd.DataFrame, indices: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Align hourly telemetry rows to nearest photograph within +-2 hours per cultivation."""
    matched_parts = []
    for k in range(1, 17):
        sub_jam = df_jam[df_jam["kultivasi"] == k].sort_values("dt").copy()
        df_p = indices.get(k)
        if df_p is None or len(df_p) == 0:
            continue
        merged = pd.merge_asof(
            sub_jam,
            df_p,
            on="dt",
            direction="nearest",
            tolerance=pd.Timedelta(hours=2),
        )
        merged = merged.dropna(subset=["photo_path"]).copy()
        if len(merged) > 0:
            matched_parts.append(merged)

    if not matched_parts:
        return pd.DataFrame()
    return pd.concat(matched_parts, ignore_index=True)


def _rgb_to_lab_numpy(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sRGB [0,1] (H, W, 3) -> CIE L*a*b*."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r_l, g_l, b_l = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (0.4124564 * r_l + 0.3575761 * g_l + 0.1804375 * b_l) / 0.95047
    y = (0.2126729 * r_l + 0.7151522 * g_l + 0.0721750 * b_l) / 1.00000
    z = (0.0193339 * r_l + 0.1191920 * g_l + 0.9503041 * b_l) / 1.08883

    eps = 0.008856
    fx = np.where(x > eps, np.cbrt(x), 7.787 * x + 16.0 / 116.0)
    fy = np.where(y > eps, np.cbrt(y), 7.787 * y + 16.0 / 116.0)
    fz = np.where(z > eps, np.cbrt(z), 7.787 * z + 16.0 / 116.0)

    lab_l = 116.0 * fy - 16.0
    lab_a = 500.0 * (fx - fy)
    lab_b = 200.0 * (fy - fz)
    return lab_l, lab_a, lab_b


def _rgb_to_hsv_numpy(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sRGB [0,1] (H, W, 3) -> HSV in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0.0)
    h = np.zeros_like(maxc)

    nz = delta > 1e-6
    mask_r = nz & (maxc == r)
    mask_g = nz & (maxc == g) & ~mask_r
    mask_b = nz & ~mask_r & ~mask_g

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    h = (h / 6.0) % 1.0
    return h, s, v


def _process_single_image(args: Tuple[int, str, bool]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load one image safely on CPU, resize to 64x64 RGB, compute 20 optical features,
    and return quantized uint8 (64, 64, 3) array for batched ConvNeXt inference.
    """
    idx, path, use_draft = args
    if not isinstance(path, str) or not os.path.exists(path):
        return idx, None, None
    try:
        with Image.open(path) as im:
            if use_draft and im.format == "JPEG":
                im.draft("RGB", (128, 128))
            im_rgb = im.convert("RGB").resize((64, 64), Image.BILINEAR)
            arr_u8 = np.asarray(im_rgb, dtype=np.uint8)
    except Exception:
        return idx, None, None

    rgb = arr_u8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Group A: RGB stats
    r_m, g_m, b_m = float(r.mean()), float(g.mean()), float(b.mean())
    r_s, g_s, b_s = float(r.std()), float(g.std()), float(b.std())

    # Group B: Algal & chlorophyll pigment indices
    exg = float((2.0 * g - r - b).mean())
    g_r = float(g_m / (r_m + 1e-4))
    ngrdi = float((g_m - r_m) / (g_m + r_m + 1e-4))
    b_g = float(b_m / (g_m + 1e-4))

    # Group C: HSV & CIE L*a*b*
    h_arr, s_arr, v_arr = _rgb_to_hsv_numpy(rgb)
    hsv_h, hsv_s, hsv_v = float(h_arr.mean()), float(s_arr.mean()), float(v_arr.mean())
    lab_l, lab_a, lab_b = _rgb_to_lab_numpy(rgb)
    l_m, a_m, b_lab_m = float(lab_l.mean()), float(lab_a.mean()), float(lab_b.mean())

    # Group D: Optical absorbance & spatial clutter
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    darkness = float(1.0 - lum.mean())

    hist, _ = np.histogram(np.clip(lum, 0.0, 1.0), bins=64, range=(0.0, 1.0))
    prob = hist.astype(np.float64)
    prob = prob[prob > 0] / max(1.0, float(prob.sum()))
    shannon_ent = float(-np.sum(prob * np.log2(prob)))

    dx = np.abs(lum[:, 1:] - lum[:, :-1])
    dy = np.abs(lum[1:, :] - lum[:-1, :])
    edge_energy = float(0.5 * (dx.mean() + dy.mean()))

    center_lum = lum[16:48, 16:48].mean()
    border_mask = np.ones((64, 64), dtype=bool)
    border_mask[16:48, 16:48] = False
    border_lum = lum[border_mask].mean()
    cb_contrast = float(abs(center_lum - border_lum))

    feats = np.array([
        r_m, g_m, b_m, r_s, g_s, b_s,
        exg, g_r, ngrdi, b_g,
        hsv_h, hsv_s, hsv_v, l_m, a_m, b_lab_m,
        darkness, shannon_ent, edge_energy, cb_contrast,
    ], dtype=np.float32)

    return idx, feats, arr_u8


def extract_dataset_features(
    dataset_key: str,
    df_aligned: pd.DataFrame,
    vision_encoder: VisionEncoder,
    device: torch.device,
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """Extract 20 optical features + 32-dim ConvNeXt latents for an aligned dataframe."""
    use_draft = (dataset_key == "Baseline")
    tasks = [(i, str(p), use_draft) for i, p in enumerate(df_aligned["photo_path"].tolist())]
    n = len(tasks)
    opt_mat = np.zeros((n, len(OPTICAL_FEATURES)), dtype=np.float32)
    img_u8 = np.zeros((n, 64, 64, 3), dtype=np.uint8)
    valid_mask = np.zeros(n, dtype=bool)

    with ThreadPoolExecutor(max_workers=8) as ex:
        for idx, feats, arr in ex.map(_process_single_image, tasks, chunksize=64):
            if feats is not None and arr is not None:
                opt_mat[idx] = feats
                img_u8[idx] = arr
                valid_mask[idx] = True

    corrupted_count = int((~valid_mask).sum())
    df_valid = df_aligned.iloc[valid_mask].copy().reset_index(drop=True)
    opt_valid = opt_mat[valid_mask]
    img_valid_u8 = img_u8[valid_mask]

    for j, col in enumerate(OPTICAL_FEATURES):
        df_valid[col] = opt_valid[:, j]

    # Batched ConvNeXt VisionEncoder inference (exact ImageNet norm matching real_data_loader.py)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    hz_list = []
    batch_sz = 256
    vision_encoder.eval()
    with torch.no_grad():
        for start in range(0, len(img_valid_u8), batch_sz):
            b_np = img_valid_u8[start : start + batch_sz]
            b_t = torch.from_numpy(b_np).permute(0, 3, 1, 2).float() / 255.0
            b_t = ((b_t - mean) / std).to(device)
            hz_b = vision_encoder(b_t).cpu().numpy()
            hz_list.append(hz_b)

    hz_mat = np.vstack(hz_list) if hz_list else np.zeros((0, 32), dtype=np.float32)
    return df_valid, img_valid_u8, hz_mat, corrupted_count


def train_dataset_specific_convnext(
    df_valid: pd.DataFrame,
    img_u8: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    device: torch.device,
    epochs: int = 15,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Fairly train a dataset-specific ConvNeXt VisionEncoder (32-dim latent) strictly on `train_mask`
    (capped evenly up to 800 samples) and evaluate on `test_mask` for [log1p(COD), log1p(TSS), OD680].
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    tr_indices = np.where(train_mask)[0]
    te_indices = np.where(test_mask)[0]
    if len(tr_indices) > 800:
        pick = rng.choice(len(tr_indices), size=800, replace=False)
        tr_indices = tr_indices[pick]

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    y_all = df_valid[["target_od680", "target_log_cod", "target_log_tss"]].values.astype(np.float32)
    y_mean = y_all[tr_indices].mean(axis=0, keepdims=True)
    y_std = np.maximum(y_all[tr_indices].std(axis=0, keepdims=True), 1e-4)
    y_norm = torch.from_numpy((y_all - y_mean) / y_std)

    def to_tensor_batch(idxs: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img_u8[idxs]).permute(0, 3, 1, 2).float() / 255.0
        return (t - mean) / std

    enc = VisionEncoder(latent_dim=32).to(device)
    head = torch.nn.Linear(32, 3).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1.5e-3, weight_decay=1e-4)

    x_tr = to_tensor_batch(tr_indices)
    y_tr = y_norm[tr_indices]
    bs = 64
    for _ in range(epochs):
        enc.train()
        head.train()
        perm = rng.permutation(len(tr_indices))
        for s in range(0, len(perm), bs):
            b_idx = perm[s : s + bs]
            bx = x_tr[b_idx].to(device)
            by = y_tr[b_idx].to(device)
            pred = head(enc(bx))
            loss = torch.nn.functional.mse_loss(pred, by)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()

    enc.eval()
    head.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(te_indices), 256):
            bx = to_tensor_batch(te_indices[s : s + 256]).to(device)
            preds.append(head(enc(bx)).cpu().numpy())
    pred_norm = np.vstack(preds)
    pred_orig = pred_norm * y_std + y_mean
    true_orig = y_all[te_indices]

    return {
        "r2_od680": float(r2_score(true_orig[:, 0], pred_orig[:, 0])),
        "r2_log_cod": float(r2_score(true_orig[:, 1], pred_orig[:, 1])),
        "r2_log_tss": float(r2_score(true_orig[:, 2], pred_orig[:, 2])),
    }


def compute_summary_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stage 1 output CSV: descriptive stats + correlations with 4 bioprocess targets."""
    rows = []
    for ds_key, df in dfs.items():
        for feat in OPTICAL_FEATURES:
            vals = df[feat].values.astype(float)
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            row = {
                "dataset": ds_key,
                "n_samples": len(df),
                "feature": feat,
                "group": FEATURE_GROUPS[feat],
                "mean": round(float(np.mean(vals)), 5),
                "std": round(float(np.std(vals)), 5),
                "median": round(float(q50), 5),
                "iqr": round(float(q75 - q25), 5),
                "skewness": round(float(skew(vals)), 4),
            }
            for t_col, _ in TARGET_SPECS:
                r_p = np.corrcoef(vals, df[t_col].values.astype(float))[0, 1]
                r_s, _ = spearmanr(vals, df[t_col].values.astype(float))
                row[f"pearson_{t_col}"] = round(float(0.0 if np.isnan(r_p) else r_p), 4)
                row[f"spearman_{t_col}"] = round(float(0.0 if np.isnan(r_s) else r_s), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_feature_importance_all(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Stage 2: Compute feature importance of 20 optical features across 3 datasets & 4 targets
    using RF impurity, Mutual Information, and standardized |Ridge| coefficients.
    """
    records = []
    for ds_key, df in dfs.items():
        X = df[OPTICAL_FEATURES].values.astype(np.float64)
        X_scaled = StandardScaler().fit_transform(X)

        for t_col, t_label in TARGET_SPECS:
            y = df[t_col].values.astype(np.float64)
            rf = RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            rf_imp = rf.feature_importances_
            rf_imp_norm = rf_imp / max(1e-12, rf_imp.max()) * 100.0

            mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
            mi_norm = mi / max(1e-12, mi.max()) * 100.0

            ridge = RidgeCV(alphas=np.logspace(-2, 3, 12))
            ridge.fit(X_scaled, (y - y.mean()) / max(1e-8, y.std()))
            rcoef = np.abs(ridge.coef_)
            rcoef_norm = rcoef / max(1e-12, rcoef.max()) * 100.0

            consensus = 0.45 * rf_imp_norm + 0.35 * mi_norm + 0.20 * rcoef_norm
            for i, feat in enumerate(OPTICAL_FEATURES):
                records.append({
                    "dataset": ds_key,
                    "target": t_col,
                    "target_label": t_label,
                    "feature": feat,
                    "group": FEATURE_GROUPS[feat],
                    "rf_importance_norm": round(float(rf_imp_norm[i]), 3),
                    "mi_score_norm": round(float(mi_norm[i]), 3),
                    "ridge_abs_norm": round(float(rcoef_norm[i]), 3),
                    "consensus_score": round(float(consensus[i]), 3),
                })

    df_imp = pd.DataFrame(records)
    return df_imp


def perform_feature_selection(
    dfs: Dict[str, pd.DataFrame],
    df_imp: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str], Dict[str, Dict[int, Dict[str, float]]]]:
    """
    Stage 3: Collinearity-aware feature selection (removing |r| > 0.92 redundancies while preserving priority)
    plus top-k subset ablation curves via 5-fold CV R2.
    """
    agg_imp = (
        df_imp.groupby(["dataset", "feature", "group"])["consensus_score"]
        .mean()
        .reset_index()
        .rename(columns={"consensus_score": "mean_consensus_score"})
    )

    global_imp = (
        df_imp.groupby(["feature", "group"])["consensus_score"]
        .mean()
        .reset_index()
        .sort_values("consensus_score", ascending=False)
    )

    selection_rows = []
    selected_per_ds: Dict[str, List[str]] = {}
    ranked_per_ds: Dict[str, List[str]] = {}

    for ds_key, df in dfs.items():
        sub = agg_imp[agg_imp["dataset"] == ds_key].sort_values("mean_consensus_score", ascending=False).copy()
        ranked_feats = sub["feature"].tolist()
        ranked_per_ds[ds_key] = ranked_feats
        corr_mat = df[OPTICAL_FEATURES].corr().abs()

        kept = []
        pruned_reason: Dict[str, str] = {}
        for feat in ranked_feats:
            redundant_with = None
            for k_f in kept:
                if corr_mat.loc[feat, k_f] > 0.92:
                    redundant_with = f"Redundant (|r|={corr_mat.loc[feat, k_f]:.2f} with {k_f})"
                    break
            if redundant_with is None and len(kept) < 10:
                kept.append(feat)
                pruned_reason[feat] = "Selected (Top-10 Non-Redundant)"
            elif redundant_with is not None:
                pruned_reason[feat] = redundant_with
            else:
                pruned_reason[feat] = "Not Selected (Rank > Top-10)"

        # If collinearity pruning left < 10 features, top up with highest remaining consensus rank
        for feat in ranked_feats:
            if len(kept) >= 10:
                break
            if feat not in kept:
                kept.append(feat)
                pruned_reason[feat] = pruned_reason.get(feat, "") + " -> Retained to fill Top-10"

        selected_per_ds[ds_key] = kept

        for rank_idx, row in enumerate(sub.itertuples(), start=1):
            max_abs_target_r = max(
                abs(float(np.corrcoef(df[row.feature].values, df[t_col].values)[0, 1]))
                for t_col, _ in TARGET_SPECS
            )
            selection_rows.append({
                "dataset": ds_key,
                "rank": rank_idx,
                "feature": row.feature,
                "group": row.group,
                "mean_consensus_score": round(float(row.mean_consensus_score), 3),
                "max_abs_target_corr": round(max_abs_target_r, 4),
                "selected_top10": row.feature in kept,
                "selection_note": pruned_reason[row.feature],
            })

    # Global consensus Top-10 (prioritizing FOTO_crop & FOTO1_crop consensus)
    df_all_crops = pd.concat([dfs["FOTO_crop"], dfs["FOTO1_crop"]], ignore_index=True)
    corr_crops = df_all_crops[OPTICAL_FEATURES].corr().abs()
    global_top10: List[str] = []
    for feat in global_imp["feature"].tolist():
        if all(corr_crops.loc[feat, kf] <= 0.92 for kf in global_top10):
            global_top10.append(feat)
        if len(global_top10) == 10:
            break
    for feat in global_imp["feature"].tolist():
        if len(global_top10) >= 10:
            break
        if feat not in global_top10:
            global_top10.append(feat)

    # Top-k ablation curves (k in [3, 5, 8, 10, 12, 15, 20]) evaluated on Common Intersection or full dataset
    k_values = [3, 5, 8, 10, 12, 15, 20]
    k_curves: Dict[str, Dict[int, Dict[str, float]]] = {}
    for ds_key, df in dfs.items():
        k_curves[ds_key] = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        ranked = ranked_per_ds[ds_key]
        # Order features so non-redundant Top-10 come first, then remainder
        ordered = selected_per_ds[ds_key] + [f for f in ranked if f not in selected_per_ds[ds_key]]
        for k_sub in k_values:
            cols = ordered[:k_sub]
            X_sub = df[cols].values.astype(np.float64)
            t_r2s = {}
            for t_col, _ in TARGET_SPECS:
                y = df[t_col].values.astype(np.float64)
                oof = np.zeros_like(y)
                for tr, te in kf.split(X_sub):
                    rf = RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
                    rf.fit(X_sub[tr], y[tr])
                    oof[te] = rf.predict(X_sub[te])
                t_r2s[t_col] = float(r2_score(y, oof))
            t_r2s["mean_r2"] = float(np.mean(list(t_r2s.values())))
            k_curves[ds_key][k_sub] = t_r2s

    return pd.DataFrame(selection_rows), selected_per_ds, global_top10, k_curves


def evaluate_cv_suite(
    X: np.ndarray,
    df: pd.DataFrame,
    model_builder,
    eval_mode: str = "5fold",
) -> Dict[str, Dict[str, float]]:
    """Run either standard 5-Fold CV or Cross-Cultivation GroupKFold CV across all 4 targets."""
    results: Dict[str, Dict[str, float]] = {}
    if eval_mode == "5fold":
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(X))
    elif eval_mode == "group_cult":
        n_groups = df["kultivasi"].nunique()
        splitter = GroupKFold(n_splits=min(5, n_groups))
        splits = list(splitter.split(X, groups=df["kultivasi"].values))
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    for t_col, _ in TARGET_SPECS:
        y = df[t_col].values.astype(np.float64)
        oof = np.zeros_like(y)
        for tr, te in splits:
            model = model_builder()
            model.fit(X[tr], y[tr])
            oof[te] = model.predict(X[te])
        r2 = float(r2_score(y, oof))
        rmse = float(np.sqrt(mean_squared_error(y, oof)))
        rho, _ = spearmanr(y, oof)
        results[t_col] = {
            "r2": r2,
            "rmse": rmse,
            "spearman": float(0.0 if np.isnan(rho) else rho),
        }
    results["mean_r2"] = float(np.mean([results[t]["r2"] for t, _ in TARGET_SPECS]))
    return results


def run_3way_benchmark(
    dfs: Dict[str, pd.DataFrame],
    hz_mats: Dict[str, np.ndarray],
    img_u8_mats: Dict[str, np.ndarray],
    selected_per_ds: Dict[str, List[str]],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Stage 4: Fair 3-Way + Dual-Crop Fusion Benchmark on:
      (A) Full Dataset Matched Hours
      (B) Strict Common Timestamp Intersection across all 3 datasets
    """
    common_uids = (
        set(dfs["Baseline"]["row_uid"])
        & set(dfs["FOTO_crop"]["row_uid"])
        & set(dfs["FOTO1_crop"]["row_uid"])
    )
    common_uid_list = sorted(common_uids)

    dfs_common: Dict[str, pd.DataFrame] = {}
    hz_common: Dict[str, np.ndarray] = {}
    img_common: Dict[str, np.ndarray] = {}
    for ds_key in ["Baseline", "FOTO_crop", "FOTO1_crop"]:
        df = dfs[ds_key]
        uid_to_pos = {u: idx for idx, u in enumerate(df["row_uid"])}
        pos = [uid_to_pos[u] for u in common_uid_list]
        dfs_common[ds_key] = df.iloc[pos].copy().reset_index(drop=True)
        hz_common[ds_key] = hz_mats[ds_key][pos]
        img_common[ds_key] = img_u8_mats[ds_key][pos]

    bench_rows = []

    def rf_all_fn():
        return RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1)

    def gbr_fn():
        return GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.06, random_state=42)

    def ridge_fn():
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 15)))

    # Evaluate on BOTH scopes
    for scope_name, ds_map, hz_map in [
        ("Full_Matched", dfs, hz_mats),
        ("Common_Intersection", dfs_common, hz_common),
    ]:
        for ds_key in ["Baseline", "FOTO_crop", "FOTO1_crop"]:
            df_cur = ds_map[ds_key]
            hz_cur = hz_map[ds_key]
            top10_cols = selected_per_ds[ds_key]

            X_opt20 = df_cur[OPTICAL_FEATURES].values.astype(np.float64)
            X_top10 = df_cur[top10_cols].values.astype(np.float64)
            X_hyb = np.hstack([X_top10, hz_cur])

            configs = [
                ("RF (20 Optical Features)", X_opt20, rf_all_fn),
                ("RF (Top-10 Selected)", X_top10, rf_all_fn),
                ("GBR (Top-10 Selected)", X_top10, gbr_fn),
                ("ConvNeXt Baseline h_z (32-Dim)", hz_cur, ridge_fn),
                ("Hybrid (Top-10 + ConvNeXt 32-Dim)", X_hyb, rf_all_fn),
            ]
            for eval_mode in ["5fold", "group_cult"]:
                for rep_name, X_in, m_builder in configs:
                    res = evaluate_cv_suite(X_in, df_cur, m_builder, eval_mode=eval_mode)
                    for t_col, t_label in TARGET_SPECS:
                        bench_rows.append({
                            "scope": scope_name,
                            "eval_mode": eval_mode,
                            "dataset": ds_key,
                            "n_samples": len(df_cur),
                            "n_cultivations": int(df_cur["kultivasi"].nunique()),
                            "representation": rep_name,
                            "target": t_col,
                            "target_label": t_label,
                            "r2": round(res[t_col]["r2"], 4),
                            "rmse": round(res[t_col]["rmse"], 4),
                            "spearman": round(res[t_col]["spearman"], 4),
                        })

        # Add Dual-Crop Fusion on Common Intersection
        if scope_name == "Common_Intersection":
            X_dual_opt = np.hstack([
                ds_map["FOTO_crop"][selected_per_ds["FOTO_crop"]].values.astype(np.float64),
                ds_map["FOTO1_crop"][selected_per_ds["FOTO1_crop"]].values.astype(np.float64),
            ])
            X_dual_hyb = np.hstack([X_dual_opt, hz_map["FOTO_crop"], hz_map["FOTO1_crop"]])
            for eval_mode in ["5fold", "group_cult"]:
                for rep_name, X_in, m_builder in [
                    ("RF (Dual-Crop Top-10x2)", X_dual_opt, rf_all_fn),
                    ("Hybrid Dual-Crop (+ConvNeXt)", X_dual_hyb, rf_all_fn),
                ]:
                    res = evaluate_cv_suite(X_in, ds_map["FOTO_crop"], m_builder, eval_mode=eval_mode)
                    for t_col, t_label in TARGET_SPECS:
                        bench_rows.append({
                            "scope": scope_name,
                            "eval_mode": eval_mode,
                            "dataset": "Dual_Crop",
                            "n_samples": len(ds_map["FOTO_crop"]),
                            "n_cultivations": int(ds_map["FOTO_crop"]["kultivasi"].nunique()),
                            "representation": rep_name,
                            "target": t_col,
                            "target_label": t_label,
                            "r2": round(res[t_col]["r2"], 4),
                            "rmse": round(res[t_col]["rmse"], 4),
                            "spearman": round(res[t_col]["spearman"], 4),
                        })

    # Fair dataset-specific ConvNeXt retraining benchmark on Common Intersection (LOCO split vs remaining)
    print("[Benchmark] Training dataset-specific ConvNeXt VisionEncoder per dataset (Common Intersection LOCO/held-out split)...")
    # Held-out cultivations that exist in Common Intersection: K03, K06, K12 (~25% of common rows)
    held_out = {3, 6, 12}
    convnext_rows = []
    for ds_key in ["Baseline", "FOTO_crop", "FOTO1_crop"]:
        df_c = dfs_common[ds_key]
        te_mask = df_c["kultivasi"].isin(held_out).values
        tr_mask = ~te_mask
        metrics = train_dataset_specific_convnext(
            df_valid=df_c,
            img_u8=img_common[ds_key],
            train_mask=tr_mask,
            test_mask=te_mask,
            device=device,
            epochs=16,
            seed=42,
        )
        for k_m, t_col, t_label in [
            ("r2_od680", "target_od680", "OD680 / Biomass (AU)"),
            ("r2_log_cod", "target_log_cod", "log1p(COD) (mg/L)"),
            ("r2_log_tss", "target_log_tss", "log1p(TSS) (mg/L)"),
        ]:
            convnext_rows.append({
                "scope": "Common_Intersection_HeldOut_K03_K06_K12",
                "eval_mode": "loco_heldout",
                "dataset": ds_key,
                "n_samples": int(te_mask.sum()),
                "n_cultivations": len(held_out),
                "representation": "ConvNeXt Fine-Tuned (Dataset-Specific)",
                "target": t_col,
                "target_label": t_label,
                "r2": round(metrics[k_m], 4),
                "rmse": np.nan,
                "spearman": np.nan,
            })

    df_bench = pd.DataFrame(bench_rows + convnext_rows)
    return df_bench, dfs_common, img_common, hz_common


def _find_sample_triplets() -> List[Dict[str, Any]]:
    """
    Find Day 0, Day 6, and Day 13/14 daytime sample triplets in K01 where Baseline (FOTO),
    FOTO_crop, and FOTO1_crop all exist at matching timestamps.
    """
    df_fc = pd.read_csv(os.path.join(DATASET_ROOT, "FOTO_crop_K1-K16", "K01", "K01_daftar_crop.csv"), encoding="utf-8-sig")
    df_f1c = pd.read_csv(os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "gambar", "K01", "K01_daftar_crop.csv"), encoding="utf-8-sig")
    df_base = pd.read_csv(os.path.join(DATASET_ROOT, "foto", "K01", "K01_daftar_foto.csv"), encoding="utf-8-sig")
    df_base = df_base[(df_base["kamera"] == "FOTO") & (df_base["dalam_periode_kultivasi"] == True)].copy()

    df_fc["dt"] = pd.to_datetime(df_fc["waktu"])
    df_f1c["dt"] = pd.to_datetime(df_f1c["waktu"])
    df_base["dt"] = pd.to_datetime(df_base["waktu"])

    triplets = []
    # Check days 1, 7, 13 around midday (12:00 - 15:00)
    for target_day in [1, 7, 13]:
        sub_fc = df_fc[df_fc["hari_ke"] == target_day]
        sub_f1c = df_f1c[df_f1c["hari_ke"] == target_day]
        if len(sub_fc) == 0 or len(sub_f1c) == 0:
            continue
        # Pick row closest to 12:00
        mid_fc = sub_fc.iloc[(sub_fc["dt"].dt.hour - 13).abs().argsort().iloc[0]]
        t_ref = mid_fc["dt"]

        # Closest base FOTO & closest FOTO1_crop
        base_row = df_base.iloc[(df_base["dt"] - t_ref).abs().argsort().iloc[0]]
        f1_row = sub_f1c.iloc[(sub_f1c["dt"] - t_ref).abs().argsort().iloc[0]]

        p_base = os.path.join(DATASET_ROOT, "foto", "K01", "FOTO", str(base_row["file"]))
        p_fc = os.path.join(DATASET_ROOT, "FOTO_crop_K1-K16", "K01", str(mid_fc["file_crop"]))
        p_f1c = os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "gambar", "K01", str(f1_row["file_crop"]))

        triplets.append({
            "day": target_day,
            "time_str": t_ref.strftime("%Y-%m-%d %H:%M"),
            "base_path": p_base,
            "bbox": (int(mid_fc["x0"]), int(mid_fc["y0"]), int(mid_fc["x1"]), int(mid_fc["y1"])),
            "fc_path": p_fc,
            "f1c_path": p_f1c,
        })
    return triplets


def plot_fig1_roi_and_mapping(
    raw_counts: Dict[str, Dict[int, int]],
    dfs: Dict[str, pd.DataFrame],
    dfs_common: Dict[str, pd.DataFrame],
) -> None:
    """Render Fig 1: ROI Visual Strip + K1-K16 Coverage + 2D PCA Feature Mapping."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(19, 12.5), dpi=300)
    fig.patch.set_facecolor("white")

    gs = GridSpec(3, 1, height_ratios=[1.15, 0.95, 1.25], hspace=0.42, left=0.05, right=0.96, top=0.90, bottom=0.06)

    fig.suptitle(
        "Stage 1 — Feature Extraction & Visual Domain Mapping: Baseline vs FOTO_crop vs FOTO1_crop",
        fontsize=16, fontweight="bold", color="#0f172a", y=0.965,
    )
    fig.text(
        0.5, 0.935,
        "Comparison of Full-Frame Camera Input (2560x1440 / 3840x2160) vs Isolated Liquid ROIs (128x128 & 168x168) across K01–K16",
        ha="center", fontsize=11, color="#475569",
    )

    # Row 1: Sample Visual Comparison (3 Days x 3 Image Views = 9 columns)
    gs_top = gs[0].subgridspec(1, 9, wspace=0.14)
    triplets = _find_sample_triplets()
    for t_i, trp in enumerate(triplets[:3]):
        c_base = t_i * 3
        # Baseline with yellow ROI bbox
        ax_b = fig.add_subplot(gs_top[0, c_base])
        with Image.open(trp["base_path"]) as im_b:
            im_draw = im_b.convert("RGB").copy()
            x0, y0, x1, y1 = trp["bbox"]
            draw = ImageDraw.Draw(im_draw)
            draw.rectangle([x0, y0, x1, y1], outline="#facc15", width=18)
            im_draw.thumbnail((480, 270))
            ax_b.imshow(im_draw)
        ax_b.set_title(f"K01 Day {trp['day']} Baseline\n(Full Frame + ROI Box)", fontsize=9, fontweight="bold", color="#334155")
        ax_b.axis("off")

        ax_fc = fig.add_subplot(gs_top[0, c_base + 1])
        with Image.open(trp["fc_path"]) as im_fc:
            ax_fc.imshow(im_fc.convert("RGB"))
        ax_fc.set_title(f"FOTO_crop\n(128x128 Main ROI)", fontsize=9, fontweight="bold", color="#0284c7")
        ax_fc.axis("off")

        ax_f1c = fig.add_subplot(gs_top[0, c_base + 2])
        with Image.open(trp["f1c_path"]) as im_f1c:
            ax_f1c.imshow(im_f1c.convert("RGB"))
        ax_f1c.set_title(f"FOTO1_crop\n(168x168 OBSBOT ROI)", fontsize=9, fontweight="bold", color="#059669")
        ax_f1c.axis("off")

    # Row 2: Image Count & Matched Hourly Telemetry Coverage per Cultivation K01-K16
    ax_cov = fig.add_subplot(gs[1])
    ks = np.arange(1, 17)
    w = 0.27
    base_cnt = [raw_counts["Baseline"].get(k, 0) for k in ks]
    fc_cnt = [raw_counts["FOTO_crop"].get(k, 0) for k in ks]
    f1c_cnt = [raw_counts["FOTO1_crop"].get(k, 0) for k in ks]

    b1 = ax_cov.bar(ks - w, base_cnt, width=w, color=DATASET_COLORS["Baseline"], alpha=0.88, label="Baseline foto/ In-Period (Total 93,134 | 5,362 jam cocok)")
    b2 = ax_cov.bar(ks, fc_cnt, width=w, color=DATASET_COLORS["FOTO_crop"], alpha=0.90, label="FOTO_crop_K1-K16 (Total 61,879 | 5,328 jam cocok)")
    b3 = ax_cov.bar(ks + w, f1c_cnt, width=w, color=DATASET_COLORS["FOTO1_crop"], alpha=0.90, label="FOTO1_crop_K1-K16 (Total 27,703 | 2,613 jam cocok, 11/16 kultivasi)")

    for rect in b2:
        h = rect.get_height()
        if h > 0:
            ax_cov.text(rect.get_x() + rect.get_width() / 2.0, h + 150, f"{h:,}", ha="center", va="bottom", fontsize=7.2, color="#0369a1", rotation=35)
    for rect in b3:
        h = rect.get_height()
        if h > 0:
            ax_cov.text(rect.get_x() + rect.get_width() / 2.0, h + 150, f"{h:,}", ha="center", va="bottom", fontsize=7.2, color="#047857", rotation=35)

    ax_cov.set_xticks(ks)
    ax_cov.set_xticklabels([f"K{k:02d}" for k in ks], fontsize=9.5, fontweight="bold")
    ax_cov.set_ylabel("Jumlah Foto Valid", fontsize=10, fontweight="bold")
    ax_cov.set_title("Cakupan Foto Valid per Kultivasi (K01–K16): FOTO_crop Menjaga 16/16 Siklus vs FOTO1_crop 11/16 Siklus", fontsize=11.5, fontweight="bold", pad=10)
    ax_cov.set_ylim(0, max(base_cnt) * 1.18)
    ax_cov.grid(axis="y", linestyle="--", alpha=0.35)
    ax_cov.legend(loc="upper right", fontsize=9.2, frameon=True, framealpha=0.95)

    # Row 3: 2D PCA Feature Space Mapping across Common Intersection colored by OD680 Biomass
    gs_bot = gs[2].subgridspec(1, 3, wspace=0.22)
    sc_last = None
    od_vals = dfs_common["FOTO_crop"]["target_od680"].values
    vmin, vmax = np.percentile(od_vals, [2, 98])

    for p_i, cfg in enumerate(DATASET_CONFIGS):
        ax_p = fig.add_subplot(gs_bot[0, p_i])
        df_c = dfs_common[cfg["key"]]
        X_s = StandardScaler().fit_transform(df_c[OPTICAL_FEATURES].values)
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(X_s)
        ev1, ev2 = pca.explained_variance_ratio_ * 100.0
        # Correlation of PC1/PC2 linear combo with OD680
        pc_r = np.corrcoef(proj[:, 0], df_c["target_od680"].values)[0, 1]

        sc_last = ax_p.scatter(
            proj[:, 0], proj[:, 1],
            c=df_c["target_od680"].values,
            cmap="viridis", vmin=vmin, vmax=vmax,
            s=11, alpha=0.68, edgecolors="none",
        )
        ax_p.set_title(
            f"PCA Ruang 20 Fitur: {cfg['key']}\nPC1={ev1:.1f}%, PC2={ev2:.1f}% | |r(PC1, OD680)|={abs(pc_r):.3f}",
            fontsize=10.5, fontweight="bold", color=cfg["color"], pad=8,
        )
        ax_p.set_xlabel(f"PC1 ({ev1:.1f}% var)", fontsize=9)
        ax_p.set_ylabel(f"PC2 ({ev2:.1f}% var)", fontsize=9)
        ax_p.grid(True, linestyle="--", alpha=0.3)

    cbar = fig.colorbar(sc_last, ax=fig.axes[-3:], fraction=0.015, pad=0.02)
    cbar.set_label("OD680 Estimasi (Biomassa AU)", fontsize=9.5, fontweight="bold")

    out_path = os.path.join(FIG_DIR, "fig_crop_1_dataset_roi_and_mapping.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, os.path.basename(out_path)))
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_fig2_feature_importance(df_imp: pd.DataFrame) -> None:
    """Render Fig 2: Feature Importance Ranking, 20x8 Target Mapping Heatmap, and Group Share."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(20, 11.5), dpi=300)
    fig.patch.set_facecolor("white")

    gs = GridSpec(1, 3, width_ratios=[1.25, 1.35, 0.95], wspace=0.45, left=0.125, right=0.97, top=0.86, bottom=0.07)
    fig.suptitle(
        "Stage 2 — Feature Importance & Bioprocess Target Mapping (RF + Mutual Info + Ridge Consensus)",
        fontsize=16, fontweight="bold", color="#0f172a", y=0.955,
    )
    fig.text(
        0.5, 0.918,
        "Konsensus Skor Kepentingan (0–100) untuk Memprediksi OD680, log1p(COD), log1p(TSS), dan pH pada Ketiga Dataset",
        ha="center", fontsize=11, color="#475569",
    )

    # Order features by average consensus across cropped datasets
    crop_order = (
        df_imp[df_imp["dataset"].isin(["FOTO_crop", "FOTO1_crop"])]
        .groupby("feature")["consensus_score"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # Panel A: Grouped Horizontal Bar Chart
    ax_a = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(crop_order))
    h = 0.27
    mean_ds = df_imp.groupby(["dataset", "feature"])["consensus_score"].mean()

    for idx, (ds_key, off) in enumerate([("Baseline", -h), ("FOTO_crop", 0.0), ("FOTO1_crop", +h)]):
        vals = [float(mean_ds.loc[(ds_key, f)]) for f in crop_order]
        ax_a.barh(y_pos + off, vals, height=h, color=DATASET_COLORS[ds_key], alpha=0.9, label=ds_key)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(crop_order, fontsize=9, fontweight="bold")
    ax_a.set_xlabel("Mean Consensus Importance Score (0–100)", fontsize=10, fontweight="bold")
    ax_a.set_title("(A) Peringkat 20 Fitur Optik per Dataset", fontsize=11.5, fontweight="bold", pad=10)
    ax_a.set_xlim(0, 102)
    ax_a.grid(axis="x", linestyle="--", alpha=0.35)
    ax_a.legend(loc="lower right", fontsize=9.2, frameon=True)

    # Panel B: 20 x 8 Feature-to-Target Importance Mapping Heatmap (FOTO_crop & FOTO1_crop)
    ax_b = fig.add_subplot(gs[0, 1])
    rev_order = list(reversed(crop_order))
    col_specs = [
        ("FOTO_crop", "target_od680", "FC\nOD680"),
        ("FOTO_crop", "target_log_cod", "FC\nCOD"),
        ("FOTO_crop", "target_log_tss", "FC\nTSS"),
        ("FOTO_crop", "target_ph", "FC\npH"),
        ("FOTO1_crop", "target_od680", "F1C\nOD680"),
        ("FOTO1_crop", "target_log_cod", "F1C\nCOD"),
        ("FOTO1_crop", "target_log_tss", "F1C\nTSS"),
        ("FOTO1_crop", "target_ph", "F1C\npH"),
    ]
    hm = np.zeros((len(rev_order), len(col_specs)), dtype=float)
    piv = df_imp.set_index(["dataset", "target", "feature"])["consensus_score"]
    for r_i, feat in enumerate(rev_order):
        for c_i, (dk, tc, _) in enumerate(col_specs):
            hm[r_i, c_i] = float(piv.loc[(dk, tc, feat)])

    im = ax_b.imshow(hm, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax_b.axvline(3.5, color="#0f172a", linewidth=2.2)
    ax_b.set_xticks(np.arange(len(col_specs)))
    ax_b.set_xticklabels([c[2] for c in col_specs], fontsize=8.8, fontweight="bold")
    ax_b.set_yticks(np.arange(len(rev_order)))
    ax_b.set_yticklabels(rev_order, fontsize=8.5)
    ax_b.set_title("(B) Peta Fitur → Target Bioproses\n(Kiri: FOTO_crop | Kanan: FOTO1_crop)", fontsize=11.5, fontweight="bold", pad=10)

    for r_i in range(hm.shape[0]):
        for c_i in range(hm.shape[1]):
            v = hm[r_i, c_i]
            txt_col = "white" if v >= 58 else "#0f172a"
            ax_b.text(c_i, r_i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=txt_col, fontweight="bold" if v >= 50 else "normal")

    cbar_b = fig.colorbar(im, ax=ax_b, fraction=0.035, pad=0.02)
    cbar_b.set_label("Consensus Importance (0–100)", fontsize=9)

    # Panel C: Category Share (%) Comparison
    ax_c = fig.add_subplot(gs[0, 2])
    grp_df = df_imp.groupby(["dataset", "group"])["consensus_score"].sum().unstack(fill_value=0.0)
    grp_pct = grp_df.div(grp_df.sum(axis=1), axis=0) * 100.0
    ds_order = ["Baseline", "FOTO_crop", "FOTO1_crop"]
    grp_pct = grp_pct.loc[ds_order]

    grp_colors = {
        "Algal Pigment Indices": "#10b981",
        "HSV & CIE L*a*b*": "#0ea5e9",
        "RGB Reflectance": "#f59e0b",
        "Optical & Spatial Texture": "#8b5cf6",
    }
    bottom = np.zeros(len(ds_order))
    x_idx = np.arange(len(ds_order))
    for g_name in ["Algal Pigment Indices", "HSV & CIE L*a*b*", "RGB Reflectance", "Optical & Spatial Texture"]:
        vals = grp_pct[g_name].values
        bars = ax_c.bar(x_idx, vals, bottom=bottom, width=0.55, color=grp_colors[g_name], edgecolor="white", linewidth=1.2, label=g_name)
        for bi, val in enumerate(vals):
            if val >= 7.0:
                ax_c.text(bi, bottom[bi] + val / 2.0, f"{val:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        bottom += vals

    ax_c.set_xticks(x_idx)
    ax_c.set_xticklabels(["Baseline\n(Uncropped)", "FOTO_crop\n(128x128)", "FOTO1_crop\n(168x168)"], fontsize=9.5, fontweight="bold")
    ax_c.set_ylabel("Kontribusi Kepentingan Total (%)", fontsize=10, fontweight="bold")
    ax_c.set_title("(C) Kontribusi per Kategori Fitur", fontsize=11.5, fontweight="bold", pad=10)
    ax_c.set_ylim(0, 122)
    ax_c.legend(loc="upper center", ncol=2, fontsize=8.4, frameon=True, framealpha=0.95)
    ax_c.grid(axis="y", linestyle="--", alpha=0.3)

    out_path = os.path.join(FIG_DIR, "fig_crop_2_feature_importance_mapping.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, os.path.basename(out_path)))
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_fig3_selection_heatmap(
    dfs: Dict[str, pd.DataFrame],
    global_top10: List[str],
    k_curves: Dict[str, Dict[int, Dict[str, float]]],
) -> None:
    """Render Fig 3: Correlation/Selection Heatmaps across 3 datasets + Top-k Elbow Curve."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(21, 15), dpi=300)
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 2, wspace=0.31, hspace=0.36, left=0.115, right=0.96, top=0.90, bottom=0.08)
    fig.suptitle(
        "Stage 3 — Feature Selection & Correlation Heatmaps: Top-10 Selected Optical Features + 4 Bioprocess Targets",
        fontsize=16, fontweight="bold", color="#0f172a", y=0.96,
    )
    fig.text(
        0.5, 0.93,
        "Perbandingan Korelasi Pearson (r) Fitur Terpilih Non-Redundan vs Target Bioproses serta Kurva Seleksi Top-k",
        ha="center", fontsize=11, color="#475569",
    )

    cols = global_top10 + [t[0] for t in TARGET_SPECS]
    short_labels = global_top10 + ["OD680", "log(COD)", "log(TSS)", "pH"]
    positions = [(0, 0, "Baseline", "(A) Correlation Heatmap — Baseline (foto/ Uncropped)"),
                 (0, 1, "FOTO_crop", "(B) Correlation Heatmap — FOTO_crop_K1-K16 (128x128)"),
                 (1, 0, "FOTO1_crop", "(C) Correlation Heatmap — FOTO1_crop_K1-K16 (168x168)")]

    for r_pos, c_pos, ds_key, title in positions:
        ax = fig.add_subplot(gs[r_pos, c_pos])
        cmat = dfs[ds_key][cols].corr().values
        im = ax.imshow(cmat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.axhline(9.5, color="#0f172a", linewidth=2.0)
        ax.axvline(9.5, color="#0f172a", linewidth=2.0)

        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=42, ha="right", fontsize=8.2, fontweight="bold")
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_yticklabels(short_labels, fontsize=8.2, fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=DATASET_COLORS[ds_key], pad=10)

        for i in range(len(short_labels)):
            for j in range(len(short_labels)):
                val = cmat[i, j]
                txt_color = "white" if abs(val) > 0.58 else "#0f172a"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.6, color=txt_color)

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Pearson r", fontsize=8.5)

    # Panel D: Top-k Selection Ablation Curve
    ax_d = fig.add_subplot(gs[1, 1])
    k_vals = sorted(list(k_curves["FOTO_crop"].keys()))
    for ds_key, label in [("Baseline", "Baseline (Uncropped)"), ("FOTO_crop", "FOTO_crop (128x128)"), ("FOTO1_crop", "FOTO1_crop (168x168)")]:
        mean_r2s = [k_curves[ds_key][k]["mean_r2"] for k in k_vals]
        od_r2s = [k_curves[ds_key][k]["target_od680"] for k in k_vals]
        ax_d.plot(k_vals, mean_r2s, marker="o", linewidth=2.5, markersize=7, color=DATASET_COLORS[ds_key], label=f"{label} — Mean 4-Target R²")
        ax_d.plot(k_vals, od_r2s, marker="s", linestyle="--", linewidth=1.6, markersize=5.5, color=DATASET_COLORS[ds_key], alpha=0.65, label=f"{label} — OD680 R²")
        # Annotate at k=10 and k=20
        ax_d.annotate(f"{mean_r2s[k_vals.index(10)]:.3f}", (10, mean_r2s[k_vals.index(10)]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color=DATASET_COLORS[ds_key])

    ax_d.axvline(10, color="#dc2626", linestyle=":", linewidth=2.0, label="Optimal Top-10 Selected Elbow (k=10)")
    ax_d.set_xticks(k_vals)
    ax_d.set_xlabel("Jumlah Fitur Terpilih (Top-k Ranking + Collinearity Pruning)", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("5-Fold Cross-Validation R²", fontsize=10, fontweight="bold")
    ax_d.set_title("(D) Kurva Seleksi Fitur Optimal (R² vs Jumlah Fitur k)", fontsize=11.5, fontweight="bold", pad=10)
    ax_d.grid(True, linestyle="--", alpha=0.35)
    ax_d.legend(loc="lower right", fontsize=8.3, frameon=True, ncol=1)

    out_path = os.path.join(FIG_DIR, "fig_crop_3_correlation_and_selection_heatmap.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, os.path.basename(out_path)))
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_fig4_3dataset_benchmark(
    dfs_common: Dict[str, pd.DataFrame],
    df_bench: pd.DataFrame,
) -> None:
    """Render Fig 4: Feature Distributions + 3-Dataset Predictive Benchmark + ConvNeXt + Cleanup Audit."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(19, 12.5), dpi=300)
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 2, wspace=0.22, hspace=0.34, left=0.06, right=0.97, top=0.89, bottom=0.07)
    fig.suptitle(
        "Stage 4 — Komparasi Komprehensif 3 Dataset (Baseline vs FOTO_crop vs FOTO1_crop & Dual-Crop)",
        fontsize=16, fontweight="bold", color="#0f172a", y=0.96,
    )
    n_common = len(dfs_common["FOTO_crop"])
    fig.text(
        0.5, 0.925,
        f"Evaluasi Fair pada Irisan Timestamp Sama (n = {n_common:,} jam sinkron) serta Audit Quality Filtering FOTO1_crop",
        ha="center", fontsize=11, color="#475569",
    )

    # Panel A: Standardized Distribution Shift (Pigment vs Background Clutter)
    ax_a = fig.add_subplot(gs[0, 0])
    dist_feats = ["ExG_index", "LAB_a_mean", "Optical_Darkness", "Gradient_Edge_Energy", "Center_Border_Contrast"]
    # Z-score across concatenated common rows so all 5 features fit cleanly on one axis
    all_cat = pd.concat([dfs_common[k][dist_feats] for k in ["Baseline", "FOTO_crop", "FOTO1_crop"]], ignore_index=True)
    means = all_cat.mean()
    stds = all_cat.std().replace(0, 1.0)

    offsets = [-0.26, 0.0, +0.26]
    ds_keys = ["Baseline", "FOTO_crop", "FOTO1_crop"]
    for d_i, dk in enumerate(ds_keys):
        z_data = ((dfs_common[dk][dist_feats] - means) / stds).values
        pos = np.arange(len(dist_feats)) + offsets[d_i]
        bp = ax_a.boxplot(
            [z_data[:, j] for j in range(len(dist_feats))],
            positions=pos, widths=0.22, patch_artist=True, showfliers=False,
            medianprops={"color": "#0f172a", "linewidth": 1.6},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(DATASET_COLORS[dk])
            patch.set_alpha(0.78)
        ax_a.plot([], [], color=DATASET_COLORS[dk], linewidth=8, label=dk)

    ax_a.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0)
    ax_a.set_xticks(np.arange(len(dist_feats)))
    ax_a.set_xticklabels(["ExG\nIndex", "CIE LAB a*\n(Green-Red)", "Optical\nDarkness", "Edge Clutter\nEnergy", "Center-Border\nContrast"], fontsize=9, fontweight="bold")
    ax_a.set_ylabel("Distribusi Fitur Terstandarisasi (Z-Score)", fontsize=9.8, fontweight="bold")
    ax_a.set_title("(A) Distribusi Sinyal Pigmen vs Noise/Clutter Latar (n sama)", fontsize=11.5, fontweight="bold", pad=10)
    ax_a.legend(loc="upper right", fontsize=9, frameon=True)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: 5-Fold CV Predictive Accuracy (R²) across 4 Targets (Common Intersection, RF Top-10)
    ax_b = fig.add_subplot(gs[0, 1])
    sub_b = df_bench[
        (df_bench["scope"] == "Common_Intersection")
        & (df_bench["eval_mode"] == "5fold")
        & (df_bench["representation"].isin(["RF (Top-10 Selected)", "RF (Dual-Crop Top-10x2)"]))
    ]
    t_cols = ["target_od680", "target_log_cod", "target_log_tss", "target_ph"]
    t_names = ["OD680\n(Biomassa)", "log1p(COD)\n(Limbah)", "log1p(TSS)\n(Padatan)", "Water\npH"]
    x_b = np.arange(len(t_cols))
    bw = 0.19
    ds_b_list = [("Baseline", -1.5 * bw, "Baseline"),
                 ("FOTO_crop", -0.5 * bw, "FOTO_crop"),
                 ("FOTO1_crop", +0.5 * bw, "FOTO1_crop"),
                 ("Dual_Crop", +1.5 * bw, "Dual-Crop Fusion")]

    for dk, off, lbl in ds_b_list:
        vals = [float(sub_b[(sub_b["dataset"] == dk) & (sub_b["target"] == tc)]["r2"].iloc[0]) for tc in t_cols]
        bars = ax_b.bar(x_b + off, vals, width=bw, color=DATASET_COLORS[dk], alpha=0.92, edgecolor="white", label=lbl)
        for rect, val in zip(bars, vals):
            y_txt = max(0.01, val) + 0.015 if val >= 0 else val - 0.025
            ax_b.text(rect.get_x() + rect.get_width() / 2.0, y_txt, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5, fontweight="bold", color=DATASET_COLORS[dk])

    ax_b.axhline(0.0, color="#0f172a", linewidth=1.0)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(t_names, fontsize=9.5, fontweight="bold")
    ax_b.set_ylabel("5-Fold Cross-Validation R²", fontsize=9.8, fontweight="bold")
    ax_b.set_title("(B) Akurasi Prediksi 4 Target Bioproses (RF Top-10, Irisan Sama)", fontsize=11.5, fontweight="bold", pad=10)
    ax_b.set_ylim(0.0, 1.06)
    ax_b.grid(axis="y", linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper left", fontsize=8.8, frameon=True, ncol=2)

    # Panel C: Model Representation Benchmark (Mean 3-Target R2 on Common Intersection: Optical vs ConvNeXt vs Hybrid)
    ax_c = fig.add_subplot(gs[1, 0])
    bio_targets = ["target_od680", "target_log_cod", "target_log_tss"]
    sub_c1 = df_bench[(df_bench["scope"] == "Common_Intersection") & (df_bench["eval_mode"] == "5fold") & (df_bench["target"].isin(bio_targets))]
    sub_c2 = df_bench[(df_bench["scope"] == "Common_Intersection_HeldOut_K03_K06_K12") & (df_bench["target"].isin(bio_targets))]

    rep_groups = [
        ("RF (20 Optical Features)", "20 Optical\n(5-Fold CV)", sub_c1),
        ("RF (Top-10 Selected)", "Top-10 Opt\n(5-Fold CV)", sub_c1),
        ("ConvNeXt Baseline h_z (32-Dim)", "ConvNeXt\nBaseline Pt", sub_c1),
        ("Hybrid (Top-10 + ConvNeXt 32-Dim)", "Hybrid Opt+\nConvNeXt", sub_c1),
        ("ConvNeXt Fine-Tuned (Dataset-Specific)", "ConvNeXt\nLOCO Val", sub_c2),
    ]
    x_c = np.arange(len(rep_groups))
    cw = 0.25
    for d_i, (dk, off) in enumerate([("Baseline", -cw), ("FOTO_crop", 0.0), ("FOTO1_crop", +cw)]):
        vals = []
        for rep_key, _, src_df in rep_groups:
            v = src_df[(src_df["dataset"] == dk) & (src_df["representation"] == rep_key)]["r2"].mean()
            vals.append(float(v))
        bars = ax_c.bar(x_c + off, vals, width=cw, color=DATASET_COLORS[dk], alpha=0.9, label=dk)
        for rect, val in zip(bars, vals):
            y_txt = max(0.01, val) + 0.02 if val >= 0 else val - 0.025
            ax_c.text(rect.get_x() + rect.get_width() / 2.0, y_txt, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.6, fontweight="bold", color=DATASET_COLORS[dk])

    ax_c.axhline(0.0, color="#0f172a", linewidth=1.0)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([r[1] for r in rep_groups], fontsize=8.8, fontweight="bold")
    ax_c.set_ylabel("Mean R² (OD680, log COD, log TSS)", fontsize=9.8, fontweight="bold")
    ax_c.set_title("(C) Komparasi Metode: Fitur Optik vs ConvNeXt vs Hybrid", fontsize=11.5, fontweight="bold", pad=10)
    ax_c.set_ylim(-0.42, 1.18)
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper center", ncol=3, fontsize=8.8, frameon=True, framealpha=0.95)

    # Panel D: FOTO1_crop Quality Cleanup Audit vs FOTO_crop Coverage
    ax_d = fig.add_subplot(gs[1, 1])
    excl_csv = os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "daftar_foto_tidak_masuk.csv")
    df_excl = pd.read_csv(excl_csv, encoding="utf-8-sig")
    out_period = int((df_excl["alasan"] == "di_luar_periode").sum())
    led_off = int(df_excl["penyebab"].fillna("").str.contains("lampu mati").sum())
    dark_occl = int((df_excl["alasan"] == "dihapus_cleanup").sum()) - led_off
    cant_crop = int((df_excl["alasan"] == "tidak_bisa_dicrop").sum())

    cats = [
        "Baseline In-Period\n(FOTO + FOTO1)",
        "FOTO_crop Valid\n(16/16 Kultivasi)",
        "FOTO1_crop Valid\n(11/16 Kultivasi)",
        "FOTO1 Dibuang:\nDi Luar Periode",
        "FOTO1 Cleanup:\nLampu Mati (Hitam)",
        "FOTO1 Cleanup:\nGelap/Tangan/Tepi",
    ]
    cnts = [93134, 61879, 27703, out_period, led_off, dark_occl + cant_crop]
    bar_cols = ["#64748b", "#0284c7", "#059669", "#94a3b8", "#f59e0b", "#ef4444"]
    y_d = np.arange(len(cats))[::-1]
    bars_d = ax_d.barh(y_d, cnts, height=0.58, color=bar_cols, alpha=0.9)
    for rect, cnt in zip(bars_d, cnts):
        ax_d.text(cnt + 1400, rect.get_y() + rect.get_height() / 2.0, f"{cnt:,}", va="center", ha="left", fontsize=9, fontweight="bold", color="#1e293b")

    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(cats, fontsize=8.8, fontweight="bold")
    ax_d.set_xlabel("Jumlah File Gambar", fontsize=9.8, fontweight="bold")
    ax_d.set_title("(D) Audit Penyaringan Kualitas vs Cakupan Data (FOTO1 vs FOTO_crop)", fontsize=11.5, fontweight="bold", pad=10)
    ax_d.set_xlim(0, max(cnts) * 1.18)
    ax_d.grid(axis="x", linestyle="--", alpha=0.35)

    out_path = os.path.join(FIG_DIR, "fig_crop_4_3dataset_distribution_and_benchmark.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, os.path.basename(out_path)))
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 72)
    print(" POME PHOTO DATASETS FEATURE ANALYSIS & 3-WAY BENCHMARK")
    print("=" * 72)

    df_jam = load_hourly_telemetry()
    print(f"[1/6] Loaded GABUNGAN_sensor_jam.csv: {len(df_jam):,} hourly rows across K01-K16.")

    # Load existing VisionEncoder checkpoint
    vision_encoder = VisionEncoder(latent_dim=32).to(device)
    if os.path.exists(VISION_CKPT):
        ckpt = torch.load(VISION_CKPT, map_location=device, weights_only=False)
        vision_encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[1/6] Loaded existing ConvNeXt checkpoint: {VISION_CKPT}")

    raw_counts: Dict[str, Dict[int, int]] = {}
    dfs: Dict[str, pd.DataFrame] = {}
    img_u8_mats: Dict[str, np.ndarray] = {}
    hz_mats: Dict[str, np.ndarray] = {}

    for cfg in DATASET_CONFIGS:
        dk = cfg["key"]
        indices, r_cnt = build_dataset_index(dk)
        raw_counts[dk] = r_cnt
        df_aligned = align_hourly_images(df_jam, indices)
        print(f"[2/6] Extracting features for {dk} ({len(df_aligned):,} matched hours)...")
        df_valid, img_u8, hz_mat, corrupted = extract_dataset_features(dk, df_aligned, vision_encoder, device)
        dfs[dk] = df_valid
        img_u8_mats[dk] = img_u8
        hz_mats[dk] = hz_mat
        print(f"      -> Valid rows: {len(df_valid):,} | Skipped corrupted: {corrupted} | ConvNeXt shape: {hz_mat.shape}")

    replot_only = ("--replot" in sys.argv)
    p_imp = os.path.join(EVAL_DIR, "photo_datasets_feature_importance.csv")
    p_bench = os.path.join(EVAL_DIR, "photo_datasets_3way_benchmark.csv")

    if "--fig4-only" in sys.argv and os.path.exists(p_bench):
        print("[Fig4-Only Mode] Extracting Common Intersection rows for quick re-render...")
        common_uids = sorted(set(dfs["Baseline"]["row_uid"]) & set(dfs["FOTO_crop"]["row_uid"]) & set(dfs["FOTO1_crop"]["row_uid"]))
        dfs_common = {
            dk: dfs[dk].set_index("row_uid").loc[common_uids].reset_index()
            for dk in ["Baseline", "FOTO_crop", "FOTO1_crop"]
        }
        df_bench = pd.read_csv(p_bench)
        plot_fig4_3dataset_benchmark(dfs_common, df_bench)
        return

    if replot_only and os.path.exists(p_imp) and os.path.exists(p_bench):
        print("[Replot Mode] Loading saved importance & benchmark CSVs and re-extracting lightweight features for heatmaps/plots...")
        df_imp = pd.read_csv(p_imp)
        df_bench = pd.read_csv(p_bench)
        df_sel, selected_per_ds, global_top10, k_curves = perform_feature_selection(dfs, df_imp)
        common_uids = sorted(set(dfs["Baseline"]["row_uid"]) & set(dfs["FOTO_crop"]["row_uid"]) & set(dfs["FOTO1_crop"]["row_uid"]))
        dfs_common = {
            dk: dfs[dk].set_index("row_uid").loc[common_uids].reset_index()
            for dk in ["Baseline", "FOTO_crop", "FOTO1_crop"]
        }
        print("[Replot Mode] Rendering 4 Mentor-Ready Figures (300 DPI)...")
        plot_fig1_roi_and_mapping(raw_counts, dfs, dfs_common)
        plot_fig2_feature_importance(df_imp)
        plot_fig3_selection_heatmap(dfs, global_top10, k_curves)
        plot_fig4_3dataset_benchmark(dfs_common, df_bench)
        return

    # Stage 1 CSV
    df_summary = compute_summary_table(dfs)
    p_sum = os.path.join(EVAL_DIR, "photo_datasets_extracted_features_summary.csv")
    df_summary.to_csv(p_sum, index=False)
    print(f"[Saved] {p_sum}")

    # Stage 2 CSV
    print("[3/6] Computing Feature Importance (RF + Mutual Info + Ridge)...")
    df_imp = compute_feature_importance_all(dfs)
    df_imp.to_csv(p_imp, index=False)
    print(f"[Saved] {p_imp}")

    # Stage 3 CSV
    print("[4/6] Performing Collinearity-Aware Feature Selection & Top-k Curves...")
    df_sel, selected_per_ds, global_top10, k_curves = perform_feature_selection(dfs, df_imp)
    p_sel = os.path.join(EVAL_DIR, "photo_datasets_feature_selection.csv")
    df_sel.to_csv(p_sel, index=False)
    print(f"[Saved] {p_sel}")
    print("      -> Global Top-10 Selected Features:", global_top10)

    # Stage 4 CSV
    print("[5/6] Running 3-Way + Dual-Crop Benchmark (Full Matched & Common Intersection)...")
    df_bench, dfs_common, img_common, hz_common = run_3way_benchmark(dfs, hz_mats, img_u8_mats, selected_per_ds, device)
    df_bench.to_csv(p_bench, index=False)
    print(f"[Saved] {p_bench}")

    # Render all 4 Mentor-Ready Figures
    print("[6/6] Rendering 4 Mentor-Ready Figures (300 DPI)...")
    plot_fig1_roi_and_mapping(raw_counts, dfs, dfs_common)
    plot_fig2_feature_importance(df_imp)
    plot_fig3_selection_heatmap(dfs, global_top10, k_curves)
    plot_fig4_3dataset_benchmark(dfs_common, df_bench)

    # Print key console highlights for verification
    print("\n--- KEY VERIFIED METRICS SUMMARY ---")
    for scope in ["Common_Intersection"]:
        for em in ["5fold", "group_cult"]:
            print(f"\n[{scope} | {em}] Mean R2 across 4 targets:")
            sub = df_bench[(df_bench["scope"] == scope) & (df_bench["eval_mode"] == em)]
            print(sub.groupby(["dataset", "representation"])["r2"].mean().unstack(level=0).round(4))


if __name__ == "__main__":
    main()
