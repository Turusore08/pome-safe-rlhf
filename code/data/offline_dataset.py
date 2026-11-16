"""
Offline Historical Corpus D_hist Assembly and Management.
Constructs D_hist = {(s_t, a_t, s_{t+1}, c_t)} from historical expert cultivations (K1–K6).
Handles multimodal state fusion (192 sensor descriptors + visual latents h_z)
and preserves the realistic zero-inflated dosing sparsity (~1:15 ratio).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    SENSOR_COLUMNS,
    ACTION_COLUMNS,
    ACTION_DIM,
    LOCO_TRAIN_CULTIVATIONS,
    OOD_CULTIVATIONS,
    LAB_MEASUREMENT_INTERVAL_HOURS,
)
from data.cleaning import clean_telemetry_pipeline
from data.windowing import build_windowed_features


def _reconstruct_cultivation_and_lab_masks(
    dones: torch.Tensor,
    interval_hours: int = LAB_MEASUREMENT_INTERVAL_HOURS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct per-transition cultivation IDs (K1..K16) and 24-hour laboratory
    measurement masks from `dones` episode boundaries when loading legacy checkpoints.
    """
    n = len(dones)
    cult_ids = torch.zeros(n, dtype=torch.long)
    lab_masks = torch.zeros(n, dtype=torch.bool)
    current_k = 1
    step_in_k = 0
    dones_np = dones.detach().cpu().numpy()
    for i in range(n):
        cult_ids[i] = current_k
        lab_masks[i] = (step_in_k % interval_hours) == 0
        step_in_k += 1
        if dones_np[i] == 1.0:
            current_k += 1
            step_in_k = 0
    return cult_ids, lab_masks


class OfflineDataset(Dataset):
    """
    PyTorch Dataset wrapper for the historical offline corpus D_hist.
    Stores transitions (state, action, reward, next_state, done, violation)
    along with cultivation_id and 24h lab_mask for LOCO cross-validation and
    sparse laboratory supervision.
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        violations: np.ndarray,
        cultivation_ids: Optional[np.ndarray] = None,
        lab_masks: Optional[np.ndarray] = None,
    ):
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.rewards = torch.as_tensor(rewards, dtype=torch.float32)
        self.next_states = torch.as_tensor(next_states, dtype=torch.float32)
        self.dones = torch.as_tensor(dones, dtype=torch.float32)
        self.violations = torch.as_tensor(violations, dtype=torch.float32)

        if cultivation_ids is None or lab_masks is None:
            rec_ids, rec_masks = _reconstruct_cultivation_and_lab_masks(self.dones)
            self.cultivation_ids = (
                torch.as_tensor(cultivation_ids, dtype=torch.long)
                if cultivation_ids is not None
                else rec_ids
            )
            self.lab_masks = (
                torch.as_tensor(lab_masks, dtype=torch.bool)
                if lab_masks is not None
                else rec_masks
            )
        else:
            self.cultivation_ids = torch.as_tensor(cultivation_ids, dtype=torch.long)
            self.lab_masks = torch.as_tensor(lab_masks, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
            "done": self.dones[idx],
            "violation": self.violations[idx],
            "cultivation_id": self.cultivation_ids[idx],
            "lab_mask": self.lab_masks[idx],
        }

    def subset_by_cultivations(self, allowed_k: List[int]) -> "OfflineDataset":
        """Filter transitions to only those belonging to `allowed_k` cultivations."""
        allowed_tensor = torch.tensor(allowed_k, dtype=torch.long)
        mask = torch.isin(self.cultivation_ids, allowed_tensor)
        return OfflineDataset(
            states=self.states[mask].numpy(),
            actions=self.actions[mask].numpy(),
            rewards=self.rewards[mask].numpy(),
            next_states=self.next_states[mask].numpy(),
            dones=self.dones[mask].numpy(),
            violations=self.violations[mask].numpy(),
            cultivation_ids=self.cultivation_ids[mask].numpy(),
            lab_masks=self.lab_masks[mask].numpy(),
        )

    def get_loco_split(
        self,
        held_out_k: int,
        train_cultivations: Optional[List[int]] = None,
    ) -> Tuple["OfflineDataset", "OfflineDataset"]:
        """
        Leave-One-Cultivation-Out (LOCO) split (hybrid_ode_loco.ipynb):
        Trains on 13 in-distribution cultivations (`train_cultivations` excluding `held_out_k`)
        and evaluates on `held_out_k`. OOD runs (K3 & K15) are excluded from LOCO training.
        """
        pool = train_cultivations if train_cultivations is not None else LOCO_TRAIN_CULTIVATIONS
        train_k = [k for k in pool if k != held_out_k]
        return self.subset_by_cultivations(train_k), self.subset_by_cultivations([held_out_k])

    def get_in_distribution_subset(self) -> "OfflineDataset":
        """Return the 14 in-distribution cultivations (excluding OOD K3 & K15)."""
        return self.subset_by_cultivations(LOCO_TRAIN_CULTIVATIONS)

    def get_ood_subset(self) -> "OfflineDataset":
        """Return the out-of-distribution (OOD) cultivations (K3 & K15)."""
        return self.subset_by_cultivations(OOD_CULTIVATIONS)



def assemble_offline_corpus(
    visual_dim: int = 32,
    save_path: Optional[str] = "data/d_hist.pt",
    force_rebuild: bool = False,
) -> OfflineDataset:
    """
    Assemble the complete historical offline corpus D_hist strictly from the real
    experimental dataset (ITS-POME K1–K16).
    All synthetic fallbacks have been removed. If the real dataset is not found,
    a FileNotFoundError is raised immediately.
    """
    # Robustly resolve target path to avoid CWD mismatch
    default_dir = os.path.abspath(os.path.dirname(__file__))
    if save_path is None or save_path in ("data/d_hist.pt", "d_hist.pt"):
        target_path = os.path.join(default_dir, "d_hist.pt")
    elif not os.path.isabs(save_path) and not os.path.exists(save_path):
        target_path = os.path.join(default_dir, os.path.basename(save_path))
    else:
        target_path = save_path

    # If already cached and not forcing rebuild, verify that cache matches real dataset scale (>=5000 transitions, 224 dims)
    if not force_rebuild and os.path.exists(target_path):

        data_dict = torch.load(target_path, map_location="cpu")
        states = data_dict.get("states", None)
        if states is not None and len(states) >= 5000 and states.shape[1] == 224:
            print(f"[OfflineDataset] Loaded verified real offline corpus D_hist from '{target_path}' ({len(states):,} transitions, 224 dims).")
            cult_ids = data_dict.get("cultivation_ids", None)
            lab_masks = data_dict.get("lab_masks", None)
            return OfflineDataset(
                states=states.numpy() if isinstance(states, torch.Tensor) else states,
                actions=data_dict["actions"].numpy() if isinstance(data_dict["actions"], torch.Tensor) else data_dict["actions"],
                rewards=data_dict["rewards"].numpy() if isinstance(data_dict["rewards"], torch.Tensor) else data_dict["rewards"],
                next_states=data_dict["next_states"].numpy() if isinstance(data_dict["next_states"], torch.Tensor) else data_dict["next_states"],
                dones=data_dict["dones"].numpy() if isinstance(data_dict["dones"], torch.Tensor) else data_dict["dones"],
                violations=data_dict["violations"].numpy() if isinstance(data_dict["violations"], torch.Tensor) else data_dict["violations"],
                cultivation_ids=(cult_ids.numpy() if isinstance(cult_ids, torch.Tensor) else cult_ids) if cult_ids is not None else None,
                lab_masks=(lab_masks.numpy() if isinstance(lab_masks, torch.Tensor) else lab_masks) if lab_masks is not None else None,
            )
        else:
            n_rows = len(states) if states is not None else 0
            print(f"[OfflineDataset] Target cache '{target_path}' contains only {n_rows} transitions (synthetic stub). Rebuilding strictly from real dataset K1–K16...")

    from config import REAL_GABUNGAN_SENSOR_JAM, validate_real_dataset_paths
    from data.real_data_loader import load_and_preprocess_real_dataset

    # Enforce real dataset existence (no fallback)
    validate_real_dataset_paths()

    return load_and_preprocess_real_dataset(save_path=target_path, visual_dim=visual_dim)


if __name__ == "__main__":
    assemble_offline_corpus(force_rebuild=True)

