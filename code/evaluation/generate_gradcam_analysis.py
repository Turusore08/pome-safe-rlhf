#!/usr/bin/env python3
"""
Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al. ICCV 2017)
Visual Attribution & Quantitative Attention Audit on ConvNeXt VisionEncoder (`stage2`):
  1. Baseline (foto/ Uncropped Full Frame)
  2. FOTO_crop_K1-K16 (128x128 Main Camera Liquid ROI)
  3. FOTO1_crop_K1-K16 (168x168 OBSBOT Quality-Cleaned Liquid ROI)

Outputs:
  - code/evaluation/gradcam_quantitative_metrics.csv
  - code/evaluation/figures/fig_gradcam_1_baseline_shortcut_vs_cropped_roi.png
  - code/evaluation/figures/fig_gradcam_2_multitarget_bioprocess_attribution.png
  - code/evaluation/figures/fig_gradcam_3_quantitative_attention_audit.png
"""

import os
import sys
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import REAL_DATA_DIR
from models.vision_encoder import VisionEncoder

DATASET_ROOT = "/home/turusore08/Desktop/KP/Data Kultivasi Mikroalga K1-16"
if not os.path.exists(DATASET_ROOT):
    DATASET_ROOT = REAL_DATA_DIR

SENSOR_JAM_CSV = os.path.join(DATASET_ROOT, "gabungan", "GABUNGAN_sensor_jam.csv")
EVAL_DIR = os.path.join(CODE_DIR, "evaluation")
FIG_DIR = os.path.join(EVAL_DIR, "figures")
ARTIFACT_DIR = "/home/turusore08/.gemini/antigravity/brain/aa15dda2-5b39-4e2c-9f57-28bc8edf6488"
VISION_CKPT = os.path.join(CODE_DIR, "checkpoints", "vision_encoder.pt")

TARGET_COLS = ["target_od680", "target_log_cod", "target_log_tss", "target_ph"]
TARGET_LABELS = ["OD680 / Biomassa", "log1p(COD)", "log1p(TSS)", "Water pH"]
DATASET_KEYS = ["Baseline", "FOTO_crop", "FOTO1_crop"]
DATASET_COLORS = {"Baseline": "#64748b", "FOTO_crop": "#0284c7", "FOTO1_crop": "#059669"}


class ConvNeXtGradCAM:
    """
    Exact Grad-CAM hook attached to the final spatial convolutional stage (`stage2`)
    of VisionEncoder before Global Average Pooling (`pool`).
    """

    def __init__(self, encoder: VisionEncoder, reg_head: nn.Linear):
        self.encoder = encoder
        self.reg_head = reg_head
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            self.encoder.stage2.register_forward_hook(self._fwd_hook),
            self.encoder.stage2.register_full_backward_hook(self._bwd_hook),
        ]

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        for h in self._handles:
            h.remove()

    def compute_batch_cam(
        self,
        x_batch: torch.Tensor,
        target_idx: int,
        out_size: Tuple[int, int] = (128, 128),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute normalized Grad-CAM heatmaps (B, H, W) in [0, 1], channel weights alpha (B, 64),
        and normalized predictions (B,) for target_idx.
        """
        self.encoder.eval()
        self.reg_head.eval()
        self.encoder.zero_grad(set_to_none=True)
        self.reg_head.zero_grad(set_to_none=True)

        x_in = x_batch.clone().detach().requires_grad_(True)
        latents = self.encoder(x_in)
        preds = self.reg_head(latents)
        score = preds[:, target_idx].sum()
        score.backward()

        acts = self.activations.detach()   # (B, 64, 16, 16)
        grads = self.gradients.detach()    # (B, 64, 16, 16)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, 64, 1, 1)

        # Standard Grad-CAM ReLU linear combination
        raw_cam = torch.sum(weights * acts, dim=1, keepdim=True)  # (B, 1, 16, 16)
        cam = F.relu(raw_cam)

        # Fallback if strictly zero on strong negative-regime samples: use |raw_cam| magnitude
        flat_max = cam.view(cam.size(0), -1).max(dim=1)[0]
        zero_mask = flat_max < 1e-8
        if zero_mask.any():
            cam[zero_mask] = torch.abs(raw_cam[zero_mask])

        # Bicubic upsample to (128, 128)
        cam_up = F.interpolate(cam, size=out_size, mode="bicubic", align_corners=False).squeeze(1)
        cam_np = cam_up.cpu().numpy()
        # Normalize each sample to [0, 1]
        for i in range(cam_np.shape[0]):
            c = np.maximum(cam_np[i], 0.0)
            c_min, c_max = float(c.min()), float(c.max())
            if c_max - c_min > 1e-8:
                cam_np[i] = (c - c_min) / (c_max - c_min)
            else:
                cam_np[i] = np.zeros_like(c)

        alpha_np = weights.squeeze(-1).squeeze(-1).cpu().numpy()  # (B, 64)
        pred_np = preds[:, target_idx].detach().cpu().numpy()
        return cam_np, alpha_np, pred_np


def load_synchronized_common_dataset() -> pd.DataFrame:
    """
    Build the exact hourly synchronized dataset across K01-K16 where Baseline (raw FOTO),
    FOTO_crop_K1-K16 (128x128), and FOTO1_crop_K1-K16 (168x168) ALL have valid images within +-2h,
    attaching exact liquid ROI box [u0, v0, u1, v1] for Baseline FOTO frames.
    """
    df_jam = pd.read_csv(SENSOR_JAM_CSV, encoding="utf-8-sig")
    df_jam["dt"] = pd.to_datetime(df_jam["timestamp"])
    df_jam["kultivasi"] = df_jam["kultivasi"].astype(int)
    df_jam["target_od680"] = df_jam["od680_estimasi(AU)"].astype(float)
    df_jam["target_log_cod"] = np.log1p(np.clip(df_jam["1_cod(mg/L)"].astype(float), 0.0, None))
    df_jam["target_log_tss"] = np.log1p(np.clip(df_jam["1_tss(mg/L)"].astype(float), 0.0, None))
    df_jam["target_ph"] = df_jam["1_ph"].astype(float)

    rows = []
    for k in range(1, 17):
        k_str = f"K{k:02d}"
        p_base = os.path.join(DATASET_ROOT, "foto", k_str, f"{k_str}_daftar_foto.csv")
        p_fc = os.path.join(DATASET_ROOT, "FOTO_crop_K1-K16", k_str, f"{k_str}_daftar_crop.csv")
        p_f1c = os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "gambar", k_str, f"{k_str}_daftar_crop.csv")
        if not (os.path.exists(p_base) and os.path.exists(p_fc) and os.path.exists(p_f1c)):
            continue

        df_j = df_jam[df_jam["kultivasi"] == k].sort_values("dt").copy()

        df_b = pd.read_csv(p_base, encoding="utf-8-sig")
        df_b = df_b[(df_b["dalam_periode_kultivasi"] == True) & (df_b["kamera"] == "FOTO")].copy()
        df_b["dt"] = pd.to_datetime(df_b["waktu"])
        df_b["base_path"] = [os.path.join(DATASET_ROOT, "foto", k_str, "FOTO", str(f)) for f in df_b["file"]]
        df_b = df_b.sort_values("dt")[["dt", "base_path"]]

        df_c0 = pd.read_csv(p_fc, encoding="utf-8-sig")
        df_c0["dt"] = pd.to_datetime(df_c0["waktu"])
        df_c0["fc_path"] = [os.path.join(DATASET_ROOT, "FOTO_crop_K1-K16", k_str, str(f)) for f in df_c0["file_crop"]]
        # Normalized liquid ROI fraction [u0, v0, u1, v1] inside Baseline frame
        res_wh = df_c0["resolusi_asli"].astype(str).str.split("x", expand=True)
        w_orig = res_wh[0].astype(float).replace(0, 2560.0)
        h_orig = res_wh[1].astype(float).replace(0, 1440.0)
        df_c0["roi_u0"] = np.clip(df_c0["x0"].astype(float) / w_orig, 0.0, 1.0)
        df_c0["roi_v0"] = np.clip(df_c0["y0"].astype(float) / h_orig, 0.0, 1.0)
        df_c0["roi_u1"] = np.clip(df_c0["x1"].astype(float) / w_orig, 0.0, 1.0)
        df_c0["roi_v1"] = np.clip(df_c0["y1"].astype(float) / h_orig, 0.0, 1.0)
        df_c0 = df_c0.sort_values("dt")[["dt", "fc_path", "roi_u0", "roi_v0", "roi_u1", "roi_v1"]]

        df_c1 = pd.read_csv(p_f1c, encoding="utf-8-sig")
        df_c1["dt"] = pd.to_datetime(df_c1["waktu"])
        df_c1["f1c_path"] = [os.path.join(DATASET_ROOT, "FOTO1_crop_K1-K16", "gambar", k_str, str(f)) for f in df_c1["file_crop"]]
        df_c1 = df_c1.sort_values("dt")[["dt", "f1c_path"]]

        if len(df_b) == 0 or len(df_c0) == 0 or len(df_c1) == 0:
            continue

        m = pd.merge_asof(df_j, df_b, on="dt", direction="nearest", tolerance=pd.Timedelta(hours=2))
        m = pd.merge_asof(m, df_c0, on="dt", direction="nearest", tolerance=pd.Timedelta(hours=2))
        m = pd.merge_asof(m, df_c1, on="dt", direction="nearest", tolerance=pd.Timedelta(hours=2))
        m = m.dropna(subset=["base_path", "fc_path", "f1c_path"]).copy()
        if len(m) > 0:
            rows.append(m)

    df_sync = pd.concat(rows, ignore_index=True)
    return df_sync


def _load_image_128(args: Tuple[int, str, bool]) -> Tuple[int, Optional[np.ndarray]]:
    idx, path, use_draft = args
    if not os.path.exists(path):
        return idx, None
    try:
        with Image.open(path) as im:
            if use_draft and im.format == "JPEG":
                im.draft("RGB", (256, 256))
            im_128 = im.convert("RGB").resize((128, 128), Image.BILINEAR)
            return idx, np.asarray(im_128, dtype=np.uint8)
    except Exception:
        return idx, None


def load_images_parallel(paths: List[str], use_draft: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load RGB 128x128 uint8 tensor array + boolean validity mask."""
    n = len(paths)
    arr = np.zeros((n, 128, 128, 3), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)
    tasks = [(i, p, use_draft) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for idx, img_u8 in ex.map(_load_image_128, tasks, chunksize=32):
            if img_u8 is not None:
                arr[idx] = img_u8
                valid[idx] = True
    return arr, valid


def to_normalized_tensor(img_u8: np.ndarray) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    t = torch.from_numpy(img_u8).permute(0, 3, 1, 2).float() / 255.0
    return (t - mean) / std


def train_or_finetune_cnn(
    dataset_key: str,
    x_tr_u8: np.ndarray,
    y_tr: np.ndarray,
    x_te_u8: np.ndarray,
    y_te: np.ndarray,
    device: torch.device,
    seed: int = 42,
) -> Tuple[VisionEncoder, nn.Linear, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Prepare 4-target ConvNeXt VisionEncoder + Regression Head per dataset:
      - Baseline: load pretrained vision_encoder.pt, attach 4-target head, fine-tune gently (12 ep)
      - FOTO_crop & FOTO1_crop: train fresh VisionEncoder(32) + 4-target head (20 ep)
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    y_mean = y_tr.mean(axis=0, keepdims=True)
    y_std = np.maximum(y_tr.std(axis=0, keepdims=True), 1e-4)
    y_tr_norm = torch.from_numpy(((y_tr - y_mean) / y_std).astype(np.float32))

    x_tr_t = to_normalized_tensor(x_tr_u8)
    x_te_t = to_normalized_tensor(x_te_u8)

    enc = VisionEncoder(latent_dim=32).to(device)
    head = nn.Linear(32, 4).to(device)

    if dataset_key == "Baseline" and os.path.exists(VISION_CKPT):
        ckpt = torch.load(VISION_CKPT, map_location=device, weights_only=False)
        enc.load_state_dict(ckpt["model_state_dict"])
        epochs = 12
        opt = torch.optim.AdamW([
            {"params": enc.parameters(), "lr": 3.5e-4, "weight_decay": 1e-4},
            {"params": head.parameters(), "lr": 2.5e-3, "weight_decay": 1e-4},
        ])
    else:
        epochs = 20
        opt = torch.optim.AdamW(
            list(enc.parameters()) + list(head.parameters()),
            lr=2.5e-3,
            weight_decay=1e-4,
        )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = 32
    n_tr = len(x_tr_t)
    for ep in range(epochs):
        enc.train()
        head.train()
        perm = rng.permutation(n_tr)
        for s in range(0, n_tr, bs):
            b_idx = perm[s : s + bs]
            bx = x_tr_t[b_idx].to(device)
            # Lightweight horizontal flip augmentation during training
            if rng.rand() < 0.5:
                bx = torch.flip(bx, dims=[3])
            by = y_tr_norm[b_idx].to(device)
            pred = head(enc(bx))
            loss = F.mse_loss(pred, by)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
        sched.step()

    enc.eval()
    head.eval()
    preds_te = []
    with torch.no_grad():
        for s in range(0, len(x_te_t), 128):
            bx = x_te_t[s : s + 128].to(device)
            preds_te.append(head(enc(bx)).cpu().numpy())
    pred_orig = np.vstack(preds_te) * y_std + y_mean
    r2s = {t_col: float(r2_score(y_te[:, i], pred_orig[:, i])) for i, t_col in enumerate(TARGET_COLS)}
    return enc, head, y_mean, y_std, r2s


def compute_spatial_metrics(
    cam_maps: np.ndarray,
    dataset_key: str,
    df_eval: pd.DataFrame,
) -> List[Dict[str, float]]:
    """
    Compute per-sample Grad-CAM spatial metrics:
      - roi_energy_pct: % of total Grad-CAM activation inside exact liquid crop box [u0,v0,u1,v1]
      - roi_2x_energy_pct: % inside 2x expanded reactor jar box
      - core_vs_border_ratio: mean activation inside center 60% vs border 40%
      - active_area_pct: % of pixels with normalized activation >= 0.50
      - spatial_entropy: Shannon entropy (bits) of spatial attention probability map
    """
    n, h, w = cam_maps.shape  # 128, 128
    # Center 60% box (~[14:114, 14:114] -> 100x100 / 16384 = 61.0% area)
    core_mask = np.zeros((h, w), dtype=bool)
    core_mask[14:114, 14:114] = True
    border_mask = ~core_mask

    records = []
    for i in range(n):
        c = cam_maps[i]
        tot_e = max(1e-8, float(c.sum()))

        if dataset_key == "Baseline":
            u0 = float(df_eval["roi_u0"].iloc[i])
            v0 = float(df_eval["roi_v0"].iloc[i])
            u1 = float(df_eval["roi_u1"].iloc[i])
            v1 = float(df_eval["roi_v1"].iloc[i])

            x0_p = max(0, min(w - 1, int(round(u0 * w))))
            x1_p = max(x0_p + 1, min(w, int(round(u1 * w))))
            y0_p = max(0, min(h - 1, int(round(v0 * h))))
            y1_p = max(y0_p + 1, min(h, int(round(v1 * h))))
            roi_e = float(c[y0_p:y1_p, x0_p:x1_p].sum()) / tot_e * 100.0

            # 2x expanded reactor jar box around ROI center
            cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            bw, bh = max(0.08, (u1 - u0)), max(0.08, (v1 - v0))
            ex0 = max(0, min(w - 1, int(round((cx - 1.5 * bw) * w))))
            ex1 = max(ex0 + 1, min(w, int(round((cx + 1.5 * bw) * w))))
            ey0 = max(0, min(h - 1, int(round((cy - 1.5 * bh) * h))))
            ey1 = max(ey0 + 1, min(h, int(round((cy + 1.5 * bh) * h))))
            roi_2x_e = float(c[ey0:ey1, ex0:ex1].sum()) / tot_e * 100.0
        else:
            roi_e = 100.0
            roi_2x_e = 100.0

        core_mean = float(c[core_mask].mean())
        border_mean = float(c[border_mask].mean())
        cb_ratio = core_mean / max(0.02, border_mean)
        core_energy_pct = float(c[core_mask].sum()) / tot_e * 100.0
        active_pct = float((c >= 0.50).mean() * 100.0)

        p = c.ravel() / tot_e
        p = p[p > 0]
        ent = float(-np.sum(p * np.log2(p)))

        records.append({
            "roi_energy_pct": roi_e,
            "roi_2x_energy_pct": roi_2x_e,
            "background_leak_pct": 100.0 - roi_e,
            "outer_room_leak_pct": 100.0 - roi_2x_e,
            "core_energy_pct": core_energy_pct,
            "core_vs_border_ratio": cb_ratio,
            "active_area_pct": active_pct,
            "spatial_entropy": ent,
        })
    return records


def overlay_gradcam(rgb_u8: np.ndarray, cam_2d: np.ndarray, alpha_blend: float = 0.46) -> np.ndarray:
    """Blend RGB uint8 image with turbo/jet Grad-CAM heatmap cleanly."""
    h_img, w_img = rgb_u8.shape[:2]
    if cam_2d.shape != (h_img, w_img):
        cam_pil = Image.fromarray((np.clip(cam_2d, 0, 1) * 255).astype(np.uint8)).resize((w_img, h_img), Image.BICUBIC)
        cam_resized = np.asarray(cam_pil, dtype=np.float32) / 255.0
    else:
        cam_resized = cam_2d
    cmap = matplotlib.colormaps["turbo"]
    heat_rgb = (cmap(np.clip(cam_resized, 0.0, 1.0))[..., :3] * 255.0).astype(np.float32)
    blended = (1.0 - alpha_blend) * rgb_u8.astype(np.float32) + alpha_blend * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def _get_k01_phase_indices(df_eval_full: pd.DataFrame) -> List[Tuple[int, str, int]]:
    """Find Day 1, Day 7, Day 13 daytime rows in df_eval_full (K01 prioritized)."""
    k01 = df_eval_full[df_eval_full["kultivasi"] == 1].copy()
    picked = []
    for target_day, phase_name in [(1, "Hari 1 — Awal Limbah Pekat"), (7, "Hari 7 — Pertumbuhan Eksponensial"), (13, "Hari 13 — Pematangan Biomassa")]:
        sub = k01[k01["hari_ke"] == target_day]
        if len(sub) == 0:
            sub = k01.iloc[(k01["hari_ke"] - target_day).abs().argsort().iloc[:5]]
        # Prefer midday ~13:00
        pick = sub.iloc[(sub["dt"].dt.hour - 13).abs().argsort().iloc[0]]
        picked.append((int(pick.name), phase_name, int(pick["hari_ke"])))
    return picked


def plot_gradcam_fig1(
    df_all: pd.DataFrame,
    cams_sample: Dict[str, Dict[int, np.ndarray]],
    sample_list: List[Tuple[int, str, int]],
) -> None:
    """
    Fig Grad-CAM 1: K01 Day 1, Day 7, Day 13 across Baseline, FOTO_crop, FOTO1_crop (target=OD680).
    """
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(21, 10.5), dpi=300)
    fig.patch.set_facecolor("white")

    gs = GridSpec(3, 6, wspace=0.10, hspace=0.30, left=0.04, right=0.92, top=0.84, bottom=0.05)
    fig.suptitle(
        "Grad-CAM CNN Visual Attribution: Bukti Shortcut Learning Baseline vs Fokus Cairan Terisolasi (Target: OD680)",
        fontsize=15.5, fontweight="bold", color="#0f172a", y=0.96,
    )
    fig.text(
        0.48, 0.92,
        "Lapisan Konvolusi Terakhir stage2 (16x16 upsampled) pada 3 Fase Kultivasi K01 — Membandingkan Full Frame vs ROI Crop",
        ha="center", fontsize=11, color="#475569",
    )

    col_titles = [
        ("Baseline RGB\n(+ Kotak ROI)", "#334155"),
        ("Baseline Grad-CAM\n(Full Frame)", "#334155"),
        ("FOTO_crop RGB\n(128x128)", "#0284c7"),
        ("FOTO_crop Grad-CAM\n(Main ROI)", "#0284c7"),
        ("FOTO1_crop RGB\n(168x168)", "#059669"),
        ("FOTO1_crop Grad-CAM\n(OBSBOT ROI)", "#059669"),
    ]

    for r_i, (row_idx, phase_lbl, day_num) in enumerate(sample_list):
        row = df_all.loc[row_idx]
        # Load natural aspect-ratio Baseline (320x180) and native crops
        with Image.open(row["base_path"]) as im_b:
            im_b.draft("RGB", (640, 360))
            base_320 = np.asarray(im_b.convert("RGB").resize((320, 180), Image.BILINEAR), dtype=np.uint8)
        with Image.open(row["fc_path"]) as im_fc:
            fc_128 = np.asarray(im_fc.convert("RGB").resize((160, 160), Image.BILINEAR), dtype=np.uint8)
        with Image.open(row["f1c_path"]) as im_f1c:
            f1c_168 = np.asarray(im_f1c.convert("RGB").resize((160, 160), Image.BILINEAR), dtype=np.uint8)

        u0, v0, u1, v1 = float(row["roi_u0"]), float(row["roi_v0"]), float(row["roi_u1"]), float(row["roi_v1"])
        # Compute ROI & 2x Jar Energy for this Baseline image
        cam_b = cams_sample["Baseline"][row_idx]
        x0_c, x1_c = int(round(u0 * 128)), max(int(round(u0 * 128)) + 1, int(round(u1 * 128)))
        y0_c, y1_c = int(round(v0 * 128)), max(int(round(v0 * 128)) + 1, int(round(v1 * 128)))
        tot_b = max(1e-8, float(cam_b.sum()))
        roi_hit = float(cam_b[y0_c:y1_c, x0_c:x1_c].sum()) / tot_b * 100.0
        cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        bw, bh = max(0.08, u1 - u0), max(0.08, v1 - v0)
        ex0 = max(0, int(round((cx - 1.5 * bw) * 128)))
        ex1 = min(128, max(ex0 + 1, int(round((cx + 1.5 * bw) * 128))))
        ey0 = max(0, int(round((cy - 1.5 * bh) * 128)))
        ey1 = min(128, max(ey0 + 1, int(round((cy + 1.5 * bh) * 128))))
        jar_hit = float(cam_b[ey0:ey1, ex0:ex1].sum()) / tot_b * 100.0

        base_overlay = overlay_gradcam(base_320, cam_b, alpha_blend=0.50)
        fc_overlay = overlay_gradcam(fc_128, cams_sample["FOTO_crop"][row_idx], alpha_blend=0.50)
        f1c_overlay = overlay_gradcam(f1c_168, cams_sample["FOTO1_crop"][row_idx], alpha_blend=0.50)

        # Draw ROI box cleanly via matplotlib patches on both Baseline columns
        panels = [
            (base_320, True, f"{phase_lbl}\nOD680={row['target_od680']:.2f} | COD={row['1_cod(mg/L)']:.0f}"),
            (base_overlay, True, f"ROI Hit: {roi_hit:.1f}% | Jar: {jar_hit:.1f}%\nBg Leak: {100.0 - jar_hit:.1f}%"),
            (fc_128, False, "Input Cairan 128x128"),
            (fc_overlay, False, f"100% Liquid ROI\nPeak={cams_sample['FOTO_crop'][row_idx].max():.2f}"),
            (f1c_168, False, "Input Cairan 168x168"),
            (f1c_overlay, False, f"100% Liquid ROI\nPeak={cams_sample['FOTO1_crop'][row_idx].max():.2f}"),
        ]

        for c_i, (img_arr, draw_roi, sub_cap) in enumerate(panels):
            ax = fig.add_subplot(gs[r_i, c_i])
            ax.imshow(img_arr)
            if draw_roi:
                ih, iw = img_arr.shape[:2]
                rect_roi = patches.Rectangle((u0 * iw, v0 * ih), (u1 - u0) * iw, (v1 - v0) * ih,
                                             linewidth=2.0, edgecolor="#facc15", facecolor="none")
                ax.add_patch(rect_roi)
            if r_i == 0:
                ax.set_title(col_titles[c_i][0], fontsize=10.2, fontweight="bold", color=col_titles[c_i][1], pad=10)
            ax.set_xlabel(sub_cap, fontsize=8.3, fontweight="bold" if c_i in (0, 1) else "normal", color="#1e293b", labelpad=4)
            ax.set_xticks([])
            ax.set_yticks([])

    # Shared Grad-CAM intensity colorbar on the right
    cax = fig.add_axes([0.935, 0.14, 0.015, 0.65])
    sm = cm.ScalarMappable(cmap=matplotlib.colormaps["turbo"], norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Normalized Grad-CAM Activation Intensity", fontsize=10, fontweight="bold")

    out_path = os.path.join(FIG_DIR, "fig_gradcam_1_baseline_shortcut_vs_cropped_roi.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, os.path.basename(out_path)))
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_gradcam_fig2(
    row_rep: pd.Series,
    cams_multi: Dict[str, Dict[str, np.ndarray]],
    alphas_multi: Dict[str, np.ndarray],
) -> None:
    """
    Fig Grad-CAM 2: Multi-Target Attribution across 4 Bioprocess Targets + 64-Channel Stage2 Gradient Profile.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(20, 12.5), dpi=300)
    fig.patch.set_facecolor("white")

    gs_outer = GridSpec(2, 1, height_ratios=[2.3, 1.0], hspace=0.28, left=0.06, right=0.92, top=0.88, bottom=0.06)
    fig.suptitle(
        "Grad-CAM Multi-Target Bioprocess Attribution: OD680, log(COD), log(TSS), dan pH (K01 Hari 7)",
        fontsize=15.5, fontweight="bold", color="#0f172a", y=0.96,
    )
    fig.text(
        0.49, 0.922,
        f"Sampel Sinkron {row_rep['dt'].strftime('%Y-%m-%d %H:%M')} | OD680={row_rep['target_od680']:.2f} AU, COD={row_rep['1_cod(mg/L)']:.0f} mg/L, TSS={row_rep['1_tss(mg/L)']:.0f} mg/L, pH={row_rep['target_ph']:.2f}",
        ha="center", fontsize=10.5, color="#475569",
    )

    gs_top = gs_outer[0].subgridspec(3, 5, wspace=0.10, hspace=0.18)

    with Image.open(row_rep["base_path"]) as im_b:
        im_b.draft("RGB", (640, 360))
        base_320 = np.asarray(im_b.convert("RGB").resize((320, 180), Image.BILINEAR), dtype=np.uint8)
    with Image.open(row_rep["fc_path"]) as im_fc:
        fc_160 = np.asarray(im_fc.convert("RGB").resize((160, 160), Image.BILINEAR), dtype=np.uint8)
    with Image.open(row_rep["f1c_path"]) as im_f1c:
        f1c_160 = np.asarray(im_f1c.convert("RGB").resize((160, 160), Image.BILINEAR), dtype=np.uint8)

    row_specs = [
        ("Baseline", "Baseline\n(Uncropped)", base_320, True),
        ("FOTO_crop", "FOTO_crop\n(128x128 ROI)", fc_160, False),
        ("FOTO1_crop", "FOTO1_crop\n(168x168 ROI)", f1c_160, False),
    ]
    col_headers = ["Input RGB", "Target: OD680", "Target: log1p(COD)", "Target: log1p(TSS)", "Target: Water pH"]
    u0, v0, u1, v1 = float(row_rep["roi_u0"]), float(row_rep["roi_v0"]), float(row_rep["roi_u1"]), float(row_rep["roi_v1"])

    for r_i, (dk, row_lbl, rgb_arr, is_base) in enumerate(row_specs):
        # Col 0: RGB
        ax0 = fig.add_subplot(gs_top[r_i, 0])
        ax0.imshow(rgb_arr)
        if is_base:
            ih, iw = rgb_arr.shape[:2]
            ax0.add_patch(patches.Rectangle((u0 * iw, v0 * ih), (u1 - u0) * iw, (v1 - v0) * ih, linewidth=2.0, edgecolor="#facc15", facecolor="none"))
        ax0.set_ylabel(row_lbl, fontsize=10, fontweight="bold", color=DATASET_COLORS[dk], labelpad=8)
        if r_i == 0:
            ax0.set_title(col_headers[0], fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
        ax0.set_xticks([])
        ax0.set_yticks([])

        for t_i, t_col in enumerate(TARGET_COLS):
            ax = fig.add_subplot(gs_top[r_i, t_i + 1])
            cam_map = cams_multi[dk][t_col]
            ov = overlay_gradcam(rgb_arr, cam_map, alpha_blend=0.52)
            ax.imshow(ov)
            if is_base:
                ih, iw = ov.shape[:2]
                ax.add_patch(patches.Rectangle((u0 * iw, v0 * ih), (u1 - u0) * iw, (v1 - v0) * ih, linewidth=1.8, edgecolor="#facc15", facecolor="none"))
                cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
                bw, bh = max(0.08, u1 - u0), max(0.08, v1 - v0)
                ex0 = max(0, int(round((cx - 1.5 * bw) * 128)))
                ex1 = min(128, max(ex0 + 1, int(round((cx + 1.5 * bw) * 128))))
                ey0 = max(0, int(round((cy - 1.5 * bh) * 128)))
                ey1 = min(128, max(ey0 + 1, int(round((cy + 1.5 * bh) * 128))))
                jar_hit = float(cam_map[ey0:ey1, ex0:ex1].sum()) / max(1e-8, float(cam_map.sum())) * 100.0
                ax.set_xlabel(f"Jar ROI Energy: {jar_hit:.1f}%", fontsize=8.3, fontweight="bold", color="#b91c1c")
            else:
                cb_ratio = float(cam_map[14:114, 14:114].mean()) / max(1e-5, float(cam_map.mean()))
                ax.set_xlabel(f"Mean Act={cam_map.mean():.2f} | CoreIdx={cb_ratio:.2f}", fontsize=8.2, color="#1e293b")
            if r_i == 0:
                ax.set_title(col_headers[t_i + 1], fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
            ax.set_xticks([])
            ax.set_yticks([])

    cax = fig.add_axes([0.935, 0.34, 0.015, 0.52])
    sm = cm.ScalarMappable(cmap=matplotlib.colormaps["turbo"], norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Normalized Grad-CAM Activation", fontsize=9.8, fontweight="bold")

    # Bottom row: 64-Channel Stage2 Gradient Sensitivity |alpha_k| across 3 datasets (averaged across 4 targets)
    ax_bot = fig.add_subplot(gs_outer[1])
    ch = np.arange(64)
    for dk in DATASET_KEYS:
        prof = alphas_multi[dk]
        prof_norm = prof / max(1e-8, float(prof.max())) * 100.0
        ax_bot.plot(ch, prof_norm, marker="o", markersize=3.5, linewidth=2.0, color=DATASET_COLORS[dk], alpha=0.9, label=f"{dk} (Top-3 kanal aktif: {np.argsort(prof)[-3:][::-1].tolist()})")
        ax_bot.fill_between(ch, 0, prof_norm, color=DATASET_COLORS[dk], alpha=0.10)

    ax_bot.set_xticks(np.arange(0, 64, 4))
    ax_bot.set_xlabel("Indeks Kanal Konvolusi Terakhir `stage2` (Kanal 0 – 63)", fontsize=10, fontweight="bold")
    ax_bot.set_ylabel("Sensitivitas Gradien |alpha_k| (%)", fontsize=9.8, fontweight="bold")
    ax_bot.set_title("Profil Aktivasi Gradien 64 Kanal Konvolusi (`stage2`) — Spesialisasi Filter CNN Antar Dataset", fontsize=11.5, fontweight="bold", pad=8)
    ax_bot.set_xlim(-0.5, 63.5)
    ax_bot.set_ylim(0, 114)
    ax_bot.grid(True, linestyle="--", alpha=0.35)
    ax_bot.legend(loc="upper right", fontsize=9, frameon=True, ncol=3)

    out_path = os.path.join(FIG_DIR, "fig_gradcam_2_multitarget_bioprocess_attribution.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, os.path.basename(out_path)))
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_gradcam_fig3(
    df_quant: pd.DataFrame,
    df_day_trends: pd.DataFrame,
) -> None:
    """
    Fig Grad-CAM 3: Quantitative Grad-CAM Attention Audit across n=300 samples x 4 targets.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(19, 12), dpi=300)
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 2, wspace=0.23, hspace=0.33, left=0.06, right=0.97, top=0.89, bottom=0.08)
    fig.suptitle(
        "Stage 2B — Audit Kuantitatif Grad-CAM Across 300 Sampel Terstruktur (Baseline vs FOTO_crop vs FOTO1_crop)",
        fontsize=15.5, fontweight="bold", color="#0f172a", y=0.96,
    )
    fig.text(
        0.5, 0.923,
        "Evaluasi Energi ROI Cairan, Kebocoran Latar Belakang, Konsentrasi Inti-vs-Tepi, Entropi Spasial, & Stabilitas Harian",
        ha="center", fontsize=10.8, color="#475569",
    )

    # Panel A: Liquid ROI vs Jar vs Background Leak Breakdown (%)
    ax_a = fig.add_subplot(gs[0, 0])
    agg_ds = df_quant[df_quant["target"] == "MEAN_ALL_TARGETS"].set_index("dataset")
    x_a = np.arange(len(DATASET_KEYS))
    roi_exact = [float(agg_ds.loc[dk, "roi_energy_pct"]) for dk in DATASET_KEYS]
    jar_extra = [max(0.0, float(agg_ds.loc[dk, "roi_2x_energy_pct"]) - float(agg_ds.loc[dk, "roi_energy_pct"])) for dk in DATASET_KEYS]
    bg_leak = [max(0.0, float(agg_ds.loc[dk, "outer_room_leak_pct"])) for dk in DATASET_KEYS]

    ax_a.bar(x_a, roi_exact, width=0.52, color="#10b981", edgecolor="white", label="Dalam Kotak ROI Cairan Tepat (%)")
    ax_a.bar(x_a, jar_extra, bottom=roi_exact, width=0.52, color="#f59e0b", edgecolor="white", label="Zona Wadah Kaca / Leher 2x ROI (%)")
    ax_a.bar(x_a, bg_leak, bottom=np.array(roi_exact) + np.array(jar_extra), width=0.52, color="#ef4444", edgecolor="white", label="Bocor ke Latar Belakang Ruangan (%)")

    for i, dk in enumerate(DATASET_KEYS):
        if bg_leak[i] > 5.0:
            ax_a.text(i, roi_exact[i] + jar_extra[i] + bg_leak[i] / 2.0, f"Leak Ruangan:\n{bg_leak[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.2)
            ax_a.text(i, roi_exact[i] + jar_extra[i] / 2.0, f"Jar 2x: {jar_extra[i]:.1f}% (ROI {roi_exact[i]:.1f}%)", ha="center", va="center", color="#0f172a", fontweight="bold", fontsize=7.8)
        else:
            ax_a.text(i, 50.0, "100,0%\nROI Cairan", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(["Baseline\n(Uncropped)", "FOTO_crop\n(128x128)", "FOTO1_crop\n(168x168)"], fontsize=9.8, fontweight="bold")
    ax_a.set_ylabel("Distribusi Energi Gradien Grad-CAM (%)", fontsize=9.8, fontweight="bold")
    ax_a.set_title("(A) Fokus ROI Cairan vs Kebocoran Latar (n=300 x 4 Target)", fontsize=11.5, fontweight="bold", pad=10)
    ax_a.set_ylim(0, 122)
    ax_a.legend(loc="upper center", ncol=2, fontsize=8.3, frameon=True)
    ax_a.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel B: Core Liquid Attention Share (%) across 4 targets
    ax_b = fig.add_subplot(gs[0, 1])
    x_b = np.arange(len(TARGET_COLS))
    bw = 0.25
    sub_t = df_quant[df_quant["target"].isin(TARGET_COLS)]
    for d_i, (dk, off) in enumerate([("Baseline", -bw), ("FOTO_crop", 0.0), ("FOTO1_crop", +bw)]):
        vals = [float(sub_t[(sub_t["dataset"] == dk) & (sub_t["target"] == tc)]["core_energy_pct"].iloc[0]) for tc in TARGET_COLS]
        bars = ax_b.bar(x_b + off, vals, width=bw, color=DATASET_COLORS[dk], alpha=0.92, label=dk)
        for rect, val in zip(bars, vals):
            ax_b.text(rect.get_x() + rect.get_width() / 2.0, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=DATASET_COLORS[dk])

    ax_b.axhline(61.0, color="#0f172a", linestyle="--", linewidth=1.2, label="Netral Geometris (61,0% Luas Inti)")
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(["OD680\n(Biomassa)", "log1p(COD)\n(Limbah)", "log1p(TSS)\n(Padatan)", "Water\npH"], fontsize=9.5, fontweight="bold")
    ax_b.set_ylabel("Energi Atensi di Area Inti 61% (%)", fontsize=9.8, fontweight="bold")
    ax_b.set_title("(B) Konsentrasi Energi Atensi pada Area Inti vs Tepian Gambar", fontsize=11.5, fontweight="bold", pad=10)
    ax_b.set_ylim(0, 114)
    ax_b.legend(loc="upper center", fontsize=8.4, frameon=True, ncol=2)
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel C: Active Coverage Area (>= 0.50, %) & Spatial Attention Entropy
    ax_c = fig.add_subplot(gs[1, 0])
    x_c = np.arange(len(DATASET_KEYS))
    cw = 0.34
    act_vals = [float(agg_ds.loc[dk, "active_area_pct"]) for dk in DATASET_KEYS]
    ent_vals = [float(agg_ds.loc[dk, "spatial_entropy"]) for dk in DATASET_KEYS]
    r2_vals = [float(agg_ds.loc[dk, "cnn_val_r2"]) * 100.0 for dk in DATASET_KEYS]

    bars1 = ax_c.bar(x_c - cw / 2, act_vals, width=cw, color="#0284c7", alpha=0.88, label="Area Aktif L >= 0.50 (%)")
    bars2 = ax_c.bar(x_c + cw / 2, r2_vals, width=cw, color="#059669", alpha=0.88, label="Akurasi CNN Held-Out R² (x100%)")
    for rect, v in zip(bars1, act_vals):
        ax_c.text(rect.get_x() + rect.get_width() / 2.0, max(0, v) + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0369a1")
    for rect, v in zip(bars2, r2_vals):
        ax_c.text(rect.get_x() + rect.get_width() / 2.0, max(0, v) + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#047857")

    ax_c_twin = ax_c.twinx()
    ax_c_twin.plot(x_c, ent_vals, color="#dc2626", marker="D", markersize=8, linewidth=2.3, label="Entropi Spasial (bits)")
    for i, ev in enumerate(ent_vals):
        ax_c_twin.annotate(f"{ev:.2f} b", (x_c[i], ev), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#b91c1c")

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(["Baseline\n(Uncropped)", "FOTO_crop\n(128x128)", "FOTO1_crop\n(168x168)"], fontsize=9.8, fontweight="bold")
    ax_c.set_ylabel("Persentase (%)", fontsize=9.8, fontweight="bold")
    ax_c_twin.set_ylabel("Entropi Spasial Peta Grad-CAM (bits)", fontsize=9.5, fontweight="bold", color="#b91c1c")
    ax_c.set_title("(C) Sebaran Area Aktif, Entropi Spasial, & Akurasi CNN", fontsize=11.5, fontweight="bold", pad=10)
    ax_c.set_ylim(0, max(max(act_vals), max(r2_vals)) * 1.50)
    ax_c_twin.set_ylim(min(ent_vals) - 2.2, max(ent_vals) + 1.25)
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c_twin.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, fontsize=8.0, frameon=True)
    ax_c.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel D: Daily Trend Across Cultivation Days (H0 - H14)
    ax_d = fig.add_subplot(gs[1, 1])
    for dk in DATASET_KEYS:
        sub_d = df_day_trends[df_day_trends["dataset"] == dk].sort_values("hari_ke")
        ax_d.plot(sub_d["hari_ke"], sub_d["core_energy_pct"], marker="o", linewidth=2.2, color=DATASET_COLORS[dk], label=f"{dk}: Energi Inti (%)")
    ax_d.axhline(61.0, color="#64748b", linestyle=":", linewidth=1.2, label="Batas Seragam (61%)")

    ax_d_twin = ax_d.twinx()
    sub_base = df_day_trends[df_day_trends["dataset"] == "Baseline"].sort_values("hari_ke")
    ax_d_twin.plot(sub_base["hari_ke"], sub_base["outer_room_leak_pct"], marker="^", linestyle="--", linewidth=2.0, color="#ef4444", label="Baseline: Bocor Latar Ruangan (%)")

    ax_d.set_xlabel("Hari Kultivasi (`hari_ke`)", fontsize=9.8, fontweight="bold")
    ax_d.set_ylabel("Energi Atensi pada Area Inti (%)", fontsize=9.8, fontweight="bold")
    ax_d_twin.set_ylabel("Kebocoran Latar Ruangan Baseline (%)", fontsize=9.5, fontweight="bold", color="#dc2626")
    ax_d.set_title("(D) Dinamika Fokus Spasial Sepanjang Siklus Kultivasi", fontsize=11.5, fontweight="bold", pad=10)
    ax_d.set_ylim(30, 108)
    ax_d_twin.set_ylim(0, 114)
    hd1, ld1 = ax_d.get_legend_handles_labels()
    hd2, ld2 = ax_d_twin.get_legend_handles_labels()
    ax_d.legend(hd1 + hd2, ld1 + ld2, loc="lower right", fontsize=8.0, frameon=True, ncol=2)
    ax_d.grid(True, linestyle="--", alpha=0.3)

    out_path = os.path.join(FIG_DIR, "fig_gradcam_3_quantitative_attention_audit.png")
    fig.savefig(out_path, dpi=300)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, os.path.basename(out_path)))
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 72)
    print(" POME CONVNEXT GRAD-CAM VISUAL ATTRIBUTION & ATTENTION AUDIT")
    print("=" * 72)

    df_sync = load_synchronized_common_dataset()
    print(f"[1/5] Synchronized dataset across Baseline FOTO + FOTO_crop + FOTO1_crop: {len(df_sync):,} hours.")

    # Stratified split across the synchronized hours into n_train=960, n_eval=300 (+ preserve K01 sample rows)
    rng = np.random.RandomState(42)
    all_idx = np.arange(len(df_sync))
    perm = rng.permutation(all_idx)
    eval_pool = perm[:320]
    train_pool = perm[320:]
    if len(train_pool) > 960:
        pick_tr = rng.choice(len(train_pool), size=960, replace=False)
        train_idx = train_pool[pick_tr]
    else:
        train_idx = train_pool
    eval_idx = eval_pool[:300]

    # Force include K01 Day 1, Day 7, Day 13 sample triplets in evaluation inspection set
    sample_triplets = _get_k01_phase_indices(df_sync)
    sample_indices = [s[0] for s in sample_triplets]
    # Remove any sample_indices from train_idx to guarantee strict visual evaluation purity
    train_idx = np.array([i for i in train_idx if i not in sample_indices], dtype=int)

    print(f"[1/5] Train subset: {len(train_idx)} | Eval audit subset: {len(eval_idx)} | K01 visual samples: {sample_indices}")

    y_all = df_sync[TARGET_COLS].values.astype(np.float32)
    path_cols = {"Baseline": ("base_path", True), "FOTO_crop": ("fc_path", False), "FOTO1_crop": ("f1c_path", False)}

    quant_rows = []
    day_trend_rows = []
    cams_k01_od680: Dict[str, Dict[int, np.ndarray]] = {dk: {} for dk in DATASET_KEYS}
    cams_day7_multi: Dict[str, Dict[str, np.ndarray]] = {dk: {} for dk in DATASET_KEYS}
    alphas_multi: Dict[str, np.ndarray] = {}

    day7_row_pos = sample_triplets[1][0] if len(sample_triplets) >= 2 else sample_triplets[0][0]

    for dk in DATASET_KEYS:
        p_col, use_draft = path_cols[dk]
        print(f"\n[2-4/5] Processing {dk}...")
        tr_paths = df_sync.iloc[train_idx][p_col].tolist()
        ev_paths = df_sync.iloc[eval_idx][p_col].tolist()
        k01_paths = df_sync.iloc[sample_indices][p_col].tolist()

        x_tr_u8, v_tr = load_images_parallel(tr_paths, use_draft=use_draft)
        x_ev_u8, v_ev = load_images_parallel(ev_paths, use_draft=use_draft)
        x_k01_u8, v_k01 = load_images_parallel(k01_paths, use_draft=use_draft)

        x_tr_u8 = x_tr_u8[v_tr]
        y_tr = y_all[train_idx][v_tr]
        x_ev_u8 = x_ev_u8[v_ev]
        y_ev = y_all[eval_idx][v_ev]
        df_ev_valid = df_sync.iloc[eval_idx[v_ev]].reset_index(drop=True)

        enc, head, y_mean, y_std, val_r2s = train_or_finetune_cnn(
            dataset_key=dk,
            x_tr_u8=x_tr_u8,
            y_tr=y_tr,
            x_te_u8=x_ev_u8,
            y_te=y_ev,
            device=device,
            seed=42,
        )
        print(f"    -> CNN Held-Out R2 on eval set: { {k: round(v, 3) for k, v in val_r2s.items()} }")

        gradcam = ConvNeXtGradCAM(enc, head)
        x_ev_t = to_normalized_tensor(x_ev_u8).to(device)
        x_k01_t = to_normalized_tensor(x_k01_u8).to(device)

        ch_alphas = np.zeros(64, dtype=np.float64)
        ds_target_metrics = []

        for t_i, t_col in enumerate(TARGET_COLS):
            # Batched Grad-CAM across all n_eval images
            cam_batches = []
            alpha_batches = []
            for s in range(0, len(x_ev_t), 64):
                c_b, a_b, _ = gradcam.compute_batch_cam(x_ev_t[s : s + 64], target_idx=t_i, out_size=(128, 128))
                cam_batches.append(c_b)
                alpha_batches.append(a_b)
            cam_all = np.concatenate(cam_batches, axis=0)
            alpha_all = np.vstack(alpha_batches)
            ch_alphas += np.mean(np.abs(alpha_all), axis=0)

            recs = compute_spatial_metrics(cam_all, dk, df_ev_valid)
            df_rec = pd.DataFrame(recs)
            df_rec["hari_ke"] = df_ev_valid["hari_ke"].values
            ds_target_metrics.append(df_rec)

            quant_rows.append({
                "dataset": dk,
                "n_eval": len(df_ev_valid),
                "target": t_col,
                "target_label": TARGET_LABELS[t_i],
                "cnn_val_r2": round(val_r2s[t_col], 4),
                "roi_energy_pct": round(float(df_rec["roi_energy_pct"].mean()), 2),
                "roi_2x_energy_pct": round(float(df_rec["roi_2x_energy_pct"].mean()), 2),
                "background_leak_pct": round(float(df_rec["background_leak_pct"].mean()), 2),
                "outer_room_leak_pct": round(float(df_rec["outer_room_leak_pct"].mean()), 2),
                "core_energy_pct": round(float(df_rec["core_energy_pct"].mean()), 2),
                "core_vs_border_ratio": round(float(df_rec["core_vs_border_ratio"].mean()), 3),
                "active_area_pct": round(float(df_rec["active_area_pct"].mean()), 2),
                "spatial_entropy": round(float(df_rec["spatial_entropy"].mean()), 3),
            })

            # Visual Grad-CAM for K01 samples
            k01_c_b, _, _ = gradcam.compute_batch_cam(x_k01_t, target_idx=t_i, out_size=(128, 128))
            if t_i == 0:
                for idx_k, orig_r_idx in enumerate(sample_indices):
                    cams_k01_od680[dk][orig_r_idx] = k01_c_b[idx_k]
            day7_pos_local = sample_indices.index(day7_row_pos)
            cams_day7_multi[dk][t_col] = k01_c_b[day7_pos_local]

        alphas_multi[dk] = ch_alphas / len(TARGET_COLS)
        df_all_t = pd.concat(ds_target_metrics, ignore_index=True)

        # Dataset overall mean row
        quant_rows.append({
            "dataset": dk,
            "n_eval": len(df_ev_valid),
            "target": "MEAN_ALL_TARGETS",
            "target_label": "Mean (4 Targets)",
            "cnn_val_r2": round(float(np.mean(list(val_r2s.values()))), 4),
            "roi_energy_pct": round(float(df_all_t["roi_energy_pct"].mean()), 2),
            "roi_2x_energy_pct": round(float(df_all_t["roi_2x_energy_pct"].mean()), 2),
            "background_leak_pct": round(float(df_all_t["background_leak_pct"].mean()), 2),
            "outer_room_leak_pct": round(float(df_all_t["outer_room_leak_pct"].mean()), 2),
            "core_energy_pct": round(float(df_all_t["core_energy_pct"].mean()), 2),
            "core_vs_border_ratio": round(float(df_all_t["core_vs_border_ratio"].mean()), 3),
            "active_area_pct": round(float(df_all_t["active_area_pct"].mean()), 2),
            "spatial_entropy": round(float(df_all_t["spatial_entropy"].mean()), 3),
        })

        # Daily trends (days 0..14)
        for day_val, grp_d in df_all_t.groupby("hari_ke"):
            if int(day_val) <= 14:
                day_trend_rows.append({
                    "dataset": dk,
                    "hari_ke": int(day_val),
                    "core_energy_pct": float(grp_d["core_energy_pct"].mean()),
                    "core_vs_border_ratio": float(grp_d["core_vs_border_ratio"].mean()),
                    "outer_room_leak_pct": float(grp_d["outer_room_leak_pct"].mean()),
                })

        gradcam.remove()

    df_quant = pd.DataFrame(quant_rows)
    p_csv = os.path.join(EVAL_DIR, "gradcam_quantitative_metrics.csv")
    df_quant.to_csv(p_csv, index=False)
    print(f"\n[Saved] {p_csv}")
    print(df_quant[df_quant["target"] == "MEAN_ALL_TARGETS"].to_string(index=False))

    df_day_trends = pd.DataFrame(day_trend_rows)

    print("\n[5/5] Rendering 3 Mentor-Ready Grad-CAM Figures (300 DPI)...")
    plot_gradcam_fig1(df_sync, cams_k01_od680, sample_triplets)
    plot_gradcam_fig2(df_sync.loc[day7_row_pos], cams_day7_multi, alphas_multi)
    plot_gradcam_fig3(df_quant, df_day_trends)


if __name__ == "__main__":
    main()
