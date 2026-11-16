"""
Supervised Fine-Tuning (SFT) / Behavioral Cloning Entrypoint.
Pre-trains the reference policy pi_sft on offline corpus D_hist
using negative log-likelihood (bc_loss) to clone expert dosing habits.
Saves frozen reference policy checkpoint to checkpoints/sft_policy.pt.
(penjelasan.md §3.1, Chen & Luo, 2026).
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.offline_dataset import assemble_offline_corpus, OfflineDataset
from models.policy_sft import HurdlePolicy
from utils.gpu_optimizer import setup_gpu_environment, get_amp_context, create_optimized_dataloader


def train_sft(
    dataset: Optional[OfflineDataset] = None,
    epochs: int = 35,
    batch_size: int = 256,
    lr: float = 1e-3,
    state_dim: int = 224,
    action_dim: int = 5,
    save_path: str = "checkpoints/sft_policy.pt",
    device: Optional[torch.device] = None,
) -> HurdlePolicy:
    """
    Supervised pre-training execution entrypoint for reference policy pi_sft.
    Minimizes behavioral cloning loss:
    L_BC(theta) = - E_{(s, a) ~ D_hist} [ log pi_theta(a | s) ]
    Optimized for high-throughput CUDA execution with TF32/BF16 Tensor Cores.
    """
    dev, _ = setup_gpu_environment(verbose=True) if device is None else (device, torch.float32)
    print(f"Starting Behavioral Cloning / SFT pre-training on device: {dev}")

    if dataset is None:
        # Load or assemble historical offline dataset
        dhist_path = "data/d_hist.pt"
        if os.path.exists(dhist_path):
            data_dict = torch.load(dhist_path, map_location="cpu")
            dataset = OfflineDataset(
                states=data_dict["states"].numpy() if isinstance(data_dict["states"], torch.Tensor) else data_dict["states"],
                actions=data_dict["actions"].numpy() if isinstance(data_dict["actions"], torch.Tensor) else data_dict["actions"],
                rewards=data_dict["rewards"].numpy() if isinstance(data_dict["rewards"], torch.Tensor) else data_dict["rewards"],
                next_states=data_dict["next_states"].numpy() if isinstance(data_dict["next_states"], torch.Tensor) else data_dict["next_states"],
                dones=data_dict["dones"].numpy() if isinstance(data_dict["dones"], torch.Tensor) else data_dict["dones"],
                violations=data_dict["violations"].numpy() if isinstance(data_dict["violations"], torch.Tensor) else data_dict["violations"],
            )
        else:
            dataset = assemble_offline_corpus()

    # Automatically adapt batch size for GPU vs CPU
    actual_batch_size = batch_size if dev.type == "cuda" else min(batch_size, 64)
    dataloader = create_optimized_dataloader(dataset, batch_size=actual_batch_size, shuffle=True, device=dev)

    # Initialize reference policy model
    pi_sft = HurdlePolicy(state_dim=state_dim, action_dim=action_dim).to(dev)
    optimizer = torch.optim.AdamW(pi_sft.parameters(), lr=lr, weight_decay=1e-4)

    pi_sft.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            states = batch["state"].to(dev, non_blocking=True)
            actions = batch["action"].to(dev, non_blocking=True)

            with get_amp_context(dev):
                # Compute behavioral cloning negative log-likelihood loss
                log_prob = pi_sft.log_prob(states, actions)
                bc_loss = -log_prob.mean()

            optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(pi_sft.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += bc_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch + 1}/{epochs}] - Behavioral Cloning Loss (bc_loss): {avg_loss:.4f}")

    # Save frozen reference policy checkpoint when requested
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": pi_sft.state_dict(),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "loss": avg_loss,
                "description": "Pre-trained reference policy pi_sft via behavioral cloning",
            },
            save_path,
        )
        print(f"Reference policy pi_sft saved successfully to {save_path}")

    return pi_sft


if __name__ == "__main__":
    train_sft()
