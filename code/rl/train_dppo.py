"""
Training script for Direct Preference-based Policy Optimization (DPPO).

Constructs preference pairs in the EXACT 224-dim multimodal observation space
of POMEReactorEnv (192 multi-resolution rolling sensor descriptors + 32 ConvNeXt visual
latents) paired with continuous physical actions in [0, action_high], initializes from
SFT reference pi_sft when available, and optimizes DPPOAgent (LAPSE Eqs. 5-6) without
zero-padding hacks. Saves checkpoint to checkpoints/dppo_policy.pt.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from config import SENSOR_COLUMNS
from data.windowing import compute_step_window_features
from environment.bioprocess_dynamics import POMEVisionGenerator
from models.policy_sft import HurdlePolicy
from models.vision_encoder import VisionEncoder
from preference.labeling_rubric import PreferenceDataset, build_preference_dataset
from preference.segment_sampler import load_runs_from_transitions_csv
from rl.dppo import train_dppo_model


def _load_trained_vision_encoder(device: torch.device) -> Optional[VisionEncoder]:
    ckpt_path = os.path.join(CODE_DIR, "checkpoints", "vision_encoder.pt")
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc = VisionEncoder(latent_dim=int(ckpt.get("visual_dim", 32))).to(device)
        enc.load_state_dict(ckpt["model_state_dict"])
        enc.eval()
        return enc
    except Exception:
        return None


def _attach_224_observations(
    runs: Dict[str, Dict[str, np.ndarray]],
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Reconstruct 224-dim POMEReactorEnv-aligned observations [h_x (192), h_z (32)]
    for each synthetic cultivation run from its 10-dim physical telemetry and action history.
    """
    dev = device or torch.device("cpu")
    vision_enc = _load_trained_vision_encoder(dev)
    vision_gen = POMEVisionGenerator(visual_dim=32, seed=seed, vision_encoder=vision_enc)

    for run_id, run in runs.items():
        states_10 = run["states"]      # [cod, bod, toc, tss, ph, do, tds, nh3, temp, bio]
        actions = run["actions"]       # [T1_aer, T4_bic, T5_ino, T3_mol, T2_light]
        n_steps = len(states_10)

        history: List[Dict[str, float]] = []
        obs_list: List[np.ndarray] = []

        for t in range(n_steps):
            row10 = states_10[t]
            light_val = float(actions[t, 4]) if len(actions[t]) >= 5 else 0.0
            s_dict = {
                "cod": float(row10[0]),
                "bod": float(row10[1]),
                "total_organic_carbon": float(row10[2]),
                "total_suspended_solids": float(row10[3]),
                "turbidity": float(max(0.0, row10[3] * 0.65)),
                "ph": float(row10[4]),
                "dissolved_oxygen": float(row10[5]),
                "total_dissolved_solids": float(row10[6]),
                "dissolved_ammonia": float(row10[7]),
                "water_temperature": float(row10[8]),
                "light_intensity": light_val,
                "ambient_temperature": 28.0,
                "relative_humidity": 65.0,
                "atmospheric_pressure": 1009.0,
                "air_oxygen": 20.9,
                "air_ammonia": 0.0,
            }
            history.append(s_dict)
            h_x = compute_step_window_features(history, SENSOR_COLUMNS, window_size=4)
            is_violated = bool(
                s_dict["ph"] < 5.5
                or s_dict["ph"] > 10.5
                or s_dict["dissolved_oxygen"] < 0.5
                or s_dict["water_temperature"] < 18.0
                or s_dict["water_temperature"] > 38.0
            )
            _, h_z = vision_gen.generate_visual_state(
                cod=s_dict["cod"],
                ph=s_dict["ph"],
                turbidity=s_dict["turbidity"],
                step=t,
                is_violated=is_violated,
            )
            obs_list.append(np.concatenate([h_x, h_z]).astype(np.float32))

        run["obs_224"] = np.asarray(obs_list, dtype=np.float32)

    return runs


def build_dppo_224_preference_dataset(
    transitions_csv: Optional[str] = None,
    num_pairs: int = 800,
    segment_length: int = 8,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> PreferenceDataset:
    """Build preference dataset containing exact 224-dim POMEReactorEnv observations."""
    csv_path = transitions_csv or os.path.join(CODE_DIR, "data", "synthetic_pome_transitions.csv")
    runs = load_runs_from_transitions_csv(csv_path)
    runs = _attach_224_observations(runs, seed=seed, device=device)
    return build_preference_dataset(
        trajectories=runs,
        num_pairs=num_pairs,
        segment_length=segment_length,
        seed=seed,
    )


def main(
    epochs: int = 15,
    num_pairs: int = 800,
    segment_length: int = 8,
    seed: int = 42,
):
    checkpoints_dir = os.path.join(CODE_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    dppo_ckpt = os.path.join(checkpoints_dir, "dppo_policy.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DPPO] Constructing 224-dim multimodal preference pairs (L_s={segment_length})...")
    pref_dataset = build_dppo_224_preference_dataset(
        num_pairs=num_pairs,
        segment_length=segment_length,
        seed=seed,
        device=device,
    )

    init_policy = None
    sft_ckpt = os.path.join(checkpoints_dir, "sft_policy.pt")
    if os.path.exists(sft_ckpt):
        try:
            init_policy = HurdlePolicy(state_dim=224, action_dim=5).to(device)
            ckpt = torch.load(sft_ckpt, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            init_policy.load_state_dict(state_dict)
            print(f"[DPPO] Warm-starting policy and ref_policy from {sft_ckpt}")
        except Exception as exc:
            print(f"[DPPO] SFT warm-start skipped ({exc}); using fresh HurdlePolicy")
            init_policy = None

    print(f"[DPPO] Training pure DPPO policy on device: {device} (no post-hoc BC overwrite)...")
    train_dppo_model(
        preference_dataset=pref_dataset,
        state_dim=224,
        action_dim=5,
        epochs=epochs,
        batch_size=32,
        lr=5e-4,
        lambda_reg=0.1,
        init_policy=init_policy,
        save_path=dppo_ckpt,
        device=device,
    )
    print(f"[DPPO] Training completed. Model saved to: {dppo_ckpt}")
    return dppo_ckpt


if __name__ == "__main__":
    main()
