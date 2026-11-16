"""
Real Cultivation Data Ingestion & Preprocessing Pipeline (K1–K16).
Processes the mentor's genuine experimental dataset from 'Data Kultivasi Mikroalga K1-16':
1. Ingests GABUNGAN_sensor_jam.csv (5,696 time steps across 16 cultivation runs)
2. Maps real telemetry headers to the 16-variable schema and 5-action treatment schema
3. Cleans data via IQR outlier rejection and linear interpolation
4. Extracts 192-dimensional multi-resolution rolling statistical features
5. Aligns real photobioreactor broth photography (72,000+ photos in foto/Kxx/FOTO)
   using ConvNeXt VisionEncoder on CUDA to construct visual latent embeddings h_z (32 dims)
6. Assembles D_hist = {(s_t, a_t, r_t, s_{t+1}, d_t, c_t)} across all 16 bioprocess runs.
"""

import os
import sys
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    SENSOR_COLUMNS,
    ACTION_COLUMNS,
    ACTION_BOUNDS,
    AERATOR_FLOW_LS,
    NAHCO3_STOCK_G_PER_ML,
    MOLASSES_RAW_TO_WINDOW_SCALE,
    BIOLOGICAL_SAFETY_LIMITS,
    LOCO_TRAIN_CULTIVATIONS,
    REAL_DATA_DIR,
    REAL_GABUNGAN_SENSOR_JAM,
    REAL_FOTO_DIR,
    REAL_SENSOR_MAP,
    REAL_ACTION_MAP,
)
from data.cleaning import clean_telemetry_pipeline
from data.windowing import build_windowed_features
from data.offline_dataset import OfflineDataset
from environment.reward_rig import compute_state_reward
from evaluation.metrics import compute_step_violation_magnitude
from models.vision_encoder import VisionEncoder
from utils.gpu_optimizer import setup_gpu_environment, get_amp_context


def build_photo_index(foto_dir: str = REAL_FOTO_DIR) -> Dict[int, pd.DataFrame]:
    """
    Builds a high-speed timestamp lookup index for photos in each cultivation K01–K16
    using official metadata CSVs (Kxx_daftar_foto.csv) covering both FOTO and FOTO1 cameras (93,000+ photos).
    """
    photo_indices = {}

    for k in range(1, 17):
        k_str = f"K{k:02d}"
        csv_file = os.path.join(foto_dir, k_str, f"{k_str}_daftar_foto.csv")
        if not os.path.exists(csv_file):
            continue

        try:
            df_k = pd.read_csv(csv_file)
            # Filter photos within cultivation period if marked, or use all
            if "dalam_periode_kultivasi" in df_k.columns and df_k["dalam_periode_kultivasi"].any():
                df_k = df_k[df_k["dalam_periode_kultivasi"] == True].copy()

            # Format datetime
            df_k["dt"] = pd.to_datetime(df_k["waktu"])
            # Construct absolute file path
            df_k["photo_path"] = df_k.apply(
                lambda r: os.path.join(foto_dir, k_str, str(r["kamera"]), str(r["file"])),
                axis=1
            )
            df_k = df_k.sort_values("dt").reset_index(drop=True)
            photo_indices[k] = df_k[["photo_path", "dt"]]
        except Exception as e:
            print(f"Warning: Failed reading photo index for {k_str}: {e}")

    return photo_indices


def extract_real_visual_features(
    cultivation_df: pd.DataFrame,
    cultivation_id: int,
    photo_indices: Dict[int, pd.DataFrame],
    vision_encoder: VisionEncoder,
    device: torch.device,
    visual_dim: int = 32,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Aligns and pre-extracts 32-dim visual latent embeddings h_z for each hourly row.
    Uses GPU acceleration and Zero-Order Hold (ZOH) fallback.
    """
    n_rows = len(cultivation_df)
    visual_features = np.zeros((n_rows, visual_dim), dtype=np.float32)

    df_p = photo_indices.get(cultivation_id, None)

    # Transform for ConvNeXt
    img_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Temporal alignment with tolerance of 2 hours
    if df_p is not None and len(df_p) > 0:
        cult_sorted = cultivation_df[["dt"]].copy().reset_index()
        merged = pd.merge_asof(
            cult_sorted,
            df_p,
            on="dt",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )
    else:
        merged = None

    last_valid_hz = np.zeros(visual_dim, dtype=np.float32)
    # Batch processing for GPU
    img_tensors = []
    tensor_indices = []

    for idx in range(n_rows):
        photo_path = merged["photo_path"].iloc[idx] if merged is not None else None

        if photo_path is not None and isinstance(photo_path, str) and os.path.exists(photo_path):
            try:
                img = Image.open(photo_path).convert("RGB")
                t_img = img_transform(img)
                img_tensors.append(t_img)
                tensor_indices.append(idx)
            except Exception:
                pass

        if len(img_tensors) >= batch_size or (idx == n_rows - 1 and len(img_tensors) > 0):
            # Run batch forward pass on GPU
            batch_tensor = torch.stack(img_tensors).to(device)
            with torch.no_grad():
                with get_amp_context(device):
                    hz_batch = vision_encoder(batch_tensor).float().cpu().numpy()

            for b_i, orig_idx in enumerate(tensor_indices):
                visual_features[orig_idx] = hz_batch[b_i]
                last_valid_hz = hz_batch[b_i]

            img_tensors = []
            tensor_indices = []

    # Apply authentic Zero-Order Hold (ZOH) temporal alignment
    # If a timestep lacks a photo within tolerance, hold the last observed real photo embedding
    current_hz = None
    # 1. Back-fill initial cold start steps if the first photo appears after t=0
    for idx in range(n_rows):
        if not np.all(visual_features[idx] == 0.0):
            current_hz = visual_features[idx].copy()
            for back_i in range(idx):
                visual_features[back_i] = current_hz.copy()
            break

    # 2. Forward-fill subsequent missing photo intervals via ZOH
    if current_hz is not None:
        for idx in range(n_rows):
            if np.all(visual_features[idx] == 0.0):
                visual_features[idx] = current_hz.copy()
            else:
                current_hz = visual_features[idx].copy()
    else:
        print(f"Warning: No valid photographs found within tolerance for cultivation K{cultivation_id:02d}.")

    return visual_features


def pretrain_or_load_vision_encoder(
    df: pd.DataFrame,
    photo_indices: Dict[int, pd.DataFrame],
    device: torch.device,
    visual_dim: int = 32,
    checkpoint_path: Optional[str] = None,
    force_retrain: bool = False,
    max_samples: int = 640,
    epochs: int = 12,
    seed: int = 42,
) -> VisionEncoder:
    """
    Pre-trains ConvNeXt VisionEncoder on paired in-distribution K1-K16 broth photography
    to predict normalized biological state targets [log1p(COD), log1p(TSS), OD680].
    Strictly excludes OOD cultivations (K3, K15) from vision pre-training.
    """
    code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_path = checkpoint_path or os.path.join(code_root, "checkpoints", "vision_encoder.pt")
    vision_encoder = VisionEncoder(latent_dim=visual_dim).to(device)

    if not force_retrain and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt.get("visual_dim", visual_dim) == visual_dim:
                vision_encoder.load_state_dict(ckpt["model_state_dict"])
                vision_encoder.eval()
                r2_info = ckpt.get("val_r2", None)
                print(
                    f"[VisionEncoder] Loaded pre-trained broth encoder from {ckpt_path}"
                    + (f" (Val R2={r2_info:.3f})" if isinstance(r2_info, (int, float)) else "")
                )
                return vision_encoder
        except Exception as exc:
            print(f"[VisionEncoder] Existing checkpoint load skipped ({exc}); retraining...")

    img_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    paired_imgs: List[torch.Tensor] = []
    paired_targets: List[List[float]] = []
    per_cult_cap = max(16, max_samples // max(1, len(LOCO_TRAIN_CULTIVATIONS)))

    for cult_id in LOCO_TRAIN_CULTIVATIONS:
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
            p_path = str(row["photo_path"])
            if not os.path.exists(p_path):
                continue
            try:
                with Image.open(p_path) as im:
                    t_img = img_transform(im.convert("RGB"))
                target = [
                    float(np.log1p(max(0.0, float(row["1_cod(mg/L)"])))),
                    float(np.log1p(max(0.0, float(row["1_tss(mg/L)"])))),
                    float(np.clip(float(row["od680_estimasi(AU)"]), 0.0, 5.0)),
                ]
                paired_imgs.append(t_img)
                paired_targets.append(target)
            except Exception:
                continue

    if len(paired_imgs) < 32:
        print("[VisionEncoder] Warning: Insufficient paired broth photos found; using fresh initialization.")
        vision_encoder.eval()
        return vision_encoder

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    perm = rng.permutation(len(paired_imgs))
    split = int(0.8 * len(perm))
    tr_idx, va_idx = perm[:split], perm[split:]

    x_all = torch.stack(paired_imgs)
    y_all = torch.tensor(paired_targets, dtype=torch.float32)
    y_mean = y_all[tr_idx].mean(dim=0, keepdim=True)
    y_std = torch.clamp(y_all[tr_idx].std(dim=0, keepdim=True), min=1e-4)
    y_norm = (y_all - y_mean) / y_std

    reg_head = torch.nn.Linear(visual_dim, 3).to(device)
    optimizer = torch.optim.AdamW(
        list(vision_encoder.parameters()) + list(reg_head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    batch_size = 32
    print(f"[VisionEncoder] Pre-training ConvNeXt on {len(tr_idx)} paired K1-K16 broth photos ({len(va_idx)} val)...")
    for epoch in range(epochs):
        vision_encoder.train()
        reg_head.train()
        ep_perm = tr_idx[rng.permutation(len(tr_idx))]
        for start in range(0, len(ep_perm), batch_size):
            b_ids = ep_perm[start : start + batch_size]
            bx = x_all[b_ids].to(device)
            by = y_norm[b_ids].to(device)
            preds = reg_head(vision_encoder(bx))
            loss = torch.nn.functional.mse_loss(preds, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vision_encoder.parameters(), 1.0)
            optimizer.step()

    vision_encoder.eval()
    reg_head.eval()
    with torch.no_grad():
        val_preds = []
        for start in range(0, len(va_idx), batch_size):
            bx = x_all[va_idx[start : start + batch_size]].to(device)
            val_preds.append(reg_head(vision_encoder(bx)).cpu())
        val_pred_t = torch.cat(val_preds, dim=0)
        val_true_t = y_norm[va_idx]
        val_mse = float(torch.nn.functional.mse_loss(val_pred_t, val_true_t).item())
        ss_res = float(((val_true_t - val_pred_t) ** 2).sum().item())
        ss_tot = float(((val_true_t - val_true_t.mean(dim=0, keepdim=True)) ** 2).sum().item())
        val_r2 = 1.0 - ss_res / max(1e-8, ss_tot)

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": vision_encoder.state_dict(),
            "reg_head_state_dict": reg_head.state_dict(),
            "target_mean": y_mean.squeeze(0).cpu(),
            "target_std": y_std.squeeze(0).cpu(),
            "visual_dim": visual_dim,
            "val_mse": val_mse,
            "val_r2": val_r2,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
        },
        ckpt_path,
    )
    print(f"[VisionEncoder] Saved pre-trained encoder to {ckpt_path} | Val MSE={val_mse:.4f} | Val R2={val_r2:.3f}")
    return vision_encoder


def load_and_preprocess_real_dataset(
    csv_path: Optional[str] = None,
    foto_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    visual_dim: int = 32,
) -> OfflineDataset:
    """
    End-to-end real dataset assembler:
    - Reads GABUNGAN_sensor_jam.csv
    - Cleans sensor telemetry with IQR & MICE
    - Generates 192-dim statistical features and 32-dim visual embeddings
    - Builds 5,680 real transitions across K1–K16
    """
    import config
    resolved_csv = csv_path or config.REAL_GABUNGAN_SENSOR_JAM
    resolved_foto = foto_dir or config.REAL_FOTO_DIR

    default_dir = os.path.abspath(os.path.dirname(__file__))
    target_save_path = save_path or os.path.join(default_dir, "d_hist.pt")

    dev = device if device is not None else setup_gpu_environment(verbose=False)[0]

    print("=" * 70)
    print("INGESTING & PREPROCESSING REAL CULTIVATION DATASET (K1–K16)")
    print(f"Source file : {resolved_csv}")
    print(f"Photos root : {resolved_foto}")
    print(f"Accelerator : {dev}")
    print("=" * 70)

    if not os.path.exists(resolved_csv):
        raise FileNotFoundError(
            f"Dataset riil sensor tidak ditemukan di: {resolved_csv}\n"
            f"Pastikan folder 'gabungan' berada di dalam lokasi dataset yang terdeteksi: {config.REAL_DATA_DIR}"
        )

    # 1. Load full raw telemetry
    raw_df = pd.read_csv(resolved_csv)
    print(f"Loaded {len(raw_df):,} hourly telemetry records across cultivations: {sorted(raw_df['kultivasi'].unique())}")

    # 2. Prepare timestamp datetime column & verify hourly continuity (hybrid_ode_loco.ipynb)
    df = raw_df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["kultivasi", "dt"]).reset_index(drop=True)
    for k_chk, d_chk in df.groupby("kultivasi"):
        assert (d_chk["dt"].diff().dropna() == pd.Timedelta(hours=1)).all(), f"celah waktu di K{k_chk}"

    # 3. Build photo index
    print("\n[1/4] Indexing real photobioreactor photographs (FOTO)...")
    photo_indices = build_photo_index(resolved_foto)

    total_indexed_photos = sum(len(p_df) for p_df in photo_indices.values())
    print(f"Indexed {total_indexed_photos:,} photographs across {len(photo_indices)} cultivations.")

    # 4. Pre-train or load supervised ConvNeXt VisionEncoder on in-distribution broth photos
    vision_encoder = pretrain_or_load_vision_encoder(
        df=df,
        photo_indices=photo_indices,
        device=dev,
        visual_dim=visual_dim,
        force_retrain=(save_path is None),
    )

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    all_violations = []
    all_cultivation_ids = []
    all_lab_masks = []

    cultivations = sorted(df["kultivasi"].unique())

    print("\n[2/4] Processing cultivation episodes, feature extraction, and visual embeddings...")
    for cult_id in cultivations:
        cult_raw = df[df["kultivasi"] == cult_id].sort_values("dt").reset_index(drop=True)
        n_steps = len(cult_raw)
        if n_steps < 5:
            continue

        # A. Separate sensor telemetry DataFrame (16 variables)
        sensor_dict = {"dt": cult_raw["dt"]}
        for r_col, s_col in REAL_SENSOR_MAP.items():
            if r_col in cult_raw.columns:
                sensor_dict[s_col] = cult_raw[r_col].values
            else:
                sensor_dict[s_col] = np.zeros(n_steps, dtype=np.float32)
        telemetry_df = pd.DataFrame(sensor_dict)

        # B. Separate action treatments DataFrame (5 variables)
        act_dict = {}
        for r_act, s_act in REAL_ACTION_MAP.items():
            if r_act in cult_raw.columns:
                act_dict[s_act] = cult_raw[r_act].values
            else:
                act_dict[s_act] = np.zeros(n_steps, dtype=np.float32)
        raw_actions_df = pd.DataFrame(act_dict)

        # C. Clean telemetry (IQR outlier filtering + linear interpolation + MICE)
        cleaned_cult_df = clean_telemetry_pipeline(telemetry_df, sensor_columns=SENSOR_COLUMNS)
        cleaned_cult_df["dt"] = cult_raw["dt"]

        # D. Multi-resolution statistical windowing (192 descriptors)
        hx_features = build_windowed_features(cleaned_cult_df, sensor_columns=SENSOR_COLUMNS)

        # E. Align real visual embeddings (32 latents)
        hz_features = extract_real_visual_features(
            cleaned_cult_df,
            cultivation_id=cult_id,
            photo_indices=photo_indices,
            vision_encoder=vision_encoder,
            device=dev,
            visual_dim=visual_dim,
        )

        # F. Fused multimodal state s_t = [h_x, h_z] (224 dims)
        fused_states = np.concatenate([hx_features, hz_features], axis=1).astype(np.float32)

        # G. Extract and normalize actions strictly using canonical config bounds
        raw_acts = raw_actions_df[ACTION_COLUMNS]
        a0 = np.clip(
            raw_acts["aeration"].to_numpy(dtype=np.float32)
            / max(1e-6, AERATOR_FLOW_LS)
            * ACTION_BOUNDS["aeration"][1],
            ACTION_BOUNDS["aeration"][0],
            ACTION_BOUNDS["aeration"][1],
        )
        a1 = np.clip(
            raw_acts["sodium_bicarbonate"].to_numpy(dtype=np.float32) * NAHCO3_STOCK_G_PER_ML,
            ACTION_BOUNDS["sodium_bicarbonate"][0],
            ACTION_BOUNDS["sodium_bicarbonate"][1],
        )
        a2 = np.clip(
            raw_acts["microalgae_inoculum"].to_numpy(dtype=np.float32),
            ACTION_BOUNDS["microalgae_inoculum"][0],
            ACTION_BOUNDS["microalgae_inoculum"][1],
        )
        a3 = np.clip(
            raw_acts["molasses"].to_numpy(dtype=np.float32) * MOLASSES_RAW_TO_WINDOW_SCALE,
            ACTION_BOUNDS["molasses"][0],
            ACTION_BOUNDS["molasses"][1],
        )
        a4 = np.clip(
            raw_acts["light_intensity"].to_numpy(dtype=np.float32),
            ACTION_BOUNDS["light_intensity"][0],
            ACTION_BOUNDS["light_intensity"][1],
        )
        actions = np.column_stack([a0, a1, a2, a3, a4]).astype(np.float32)

        if "biomassa_estimasi(g/L)" in cult_raw.columns:
            biomass_series = cult_raw["biomassa_estimasi(g/L)"].to_numpy(dtype=np.float32)
        else:
            biomass_series = np.maximum(0.05, cult_raw["od680_estimasi(AU)"].to_numpy(dtype=np.float32) * 1.266)

        # H. Build transitions: (s_t, a_t, r_t, s_{t+1}, done, violation)
        for t in range(n_steps - 1):
            s_t = fused_states[t]
            a_t = actions[t]
            s_next = fused_states[t + 1]

            row_curr = cleaned_cult_df.iloc[t]
            row_next = cleaned_cult_df.iloc[t + 1]

            # 1. Two-tier Tier 2 Biological Safety violation check (pH, DO, thermal limits)
            s_curr_dict = {col: float(row_curr[col]) for col in SENSOR_COLUMNS if col in row_curr}
            s_next_dict = {col: float(row_next[col]) for col in SENSOR_COLUMNS if col in row_next}
            viol_flag, _ = compute_step_violation_magnitude(s_next_dict)
            violation = float(viol_flag)

            # 2. State Reward: (E + B + Q) - (CT + CO) - absorbing penalty
            b_curr = float(biomass_series[t])
            b_next = float(biomass_series[t + 1])
            r_t, _ = compute_state_reward(
                s_curr_dict=s_curr_dict,
                a_t=a_t,
                s_next_dict=s_next_dict,
                biomass_curr=b_curr,
                biomass_next=b_next,
                dt_hours=1.0,
            )
            if violation == 1.0:
                r_t -= 50.0

            done = 1.0 if (t == n_steps - 2) else 0.0

            all_states.append(s_t)
            all_actions.append(a_t)
            all_rewards.append(r_t)
            all_next_states.append(s_next)
            all_dones.append(done)
            all_violations.append(violation)
            all_cultivation_ids.append(int(cult_id))
            all_lab_masks.append(bool((t % 24) == 0))

        print(f"  • K{cult_id:02d}: {n_steps} hourly points -> {n_steps - 1} transitions (Initial COD: {cleaned_cult_df.iloc[0]['cod']:.1f} mg/L, Final COD: {cleaned_cult_df.iloc[-1]['cod']:.1f} mg/L)")

    # 5. Assemble final OfflineDataset
    d_hist = OfflineDataset(
        states=np.array(all_states, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_states=np.array(all_next_states, dtype=np.float32),
        dones=np.array(all_dones, dtype=np.float32),
        violations=np.array(all_violations, dtype=np.float32),
        cultivation_ids=np.array(all_cultivation_ids, dtype=np.int64),
        lab_masks=np.array(all_lab_masks, dtype=bool),
    )

    print(f"\n[3/4] Successfully assembled real corpus D_hist:")
    print(f"  • Total transitions  : {len(d_hist):,}")
    print(f"  • State dimension    : {d_hist.states.shape[1]} (192 sensor descriptors + 32 visual latents)")
    print(f"  • Action dimension   : {d_hist.actions.shape[1]} continuous dosing levers")
    print(f"  • Violation rate     : {(d_hist.violations.sum() / len(d_hist) * 100.0):.2f}%")
    print(f"  • Average step reward: {d_hist.rewards.mean().item():.3f}")

    # 6. Save dataset
    print(f"\n[4/4] Saving real dataset artifacts...")
    os.makedirs(os.path.dirname(target_save_path), exist_ok=True)
    torch.save(
        {
            "states": d_hist.states,
            "actions": d_hist.actions,
            "rewards": d_hist.rewards,
            "next_states": d_hist.next_states,
            "dones": d_hist.dones,
            "violations": d_hist.violations,
            "cultivation_ids": d_hist.cultivation_ids,
            "lab_masks": d_hist.lab_masks,
        },
        target_save_path,
    )
    print(f"Saved primary dataset to: {target_save_path}")

    # Also save to default data/d_hist.pt so downstream scripts seamlessly pick it up
    default_dhist_path = os.path.join(os.path.dirname(target_save_path), "d_hist.pt")
    if os.path.abspath(target_save_path) != os.path.abspath(default_dhist_path):
        torch.save(
            {
                "states": d_hist.states,
                "actions": d_hist.actions,
                "rewards": d_hist.rewards,
                "next_states": d_hist.next_states,
                "dones": d_hist.dones,
                "violations": d_hist.violations,
                "cultivation_ids": d_hist.cultivation_ids,
                "lab_masks": d_hist.lab_masks,
            },
            default_dhist_path,
        )
        print(f"Updated default corpus: {default_dhist_path}")


    print("=" * 70)
    return d_hist


if __name__ == "__main__":
    load_and_preprocess_real_dataset()
