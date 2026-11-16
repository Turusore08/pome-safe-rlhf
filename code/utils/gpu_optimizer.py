"""
GPU Optimization Engine for High-Throughput NVIDIA Hardware.
Tuned specifically for NVIDIA GeForce RTX 5090 (Blackwell GB202, 32GB GDDR7, 5th-gen Tensor Cores)
with backwards compatibility for Ada Lovelace, Ampere, and CPU fallback.
"""

import os
import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader, Dataset


def setup_gpu_environment(verbose: bool = True) -> Tuple[torch.device, torch.dtype]:
    """
    Initializes optimal PyTorch CUDA execution settings:
    - Enables TensorFloat-32 (TF32) on modern Tensor Cores (Blackwell / Ada / Ampere)
    - Enables cuDNN benchmark autotuner
    - Selects optimal mixed precision dtype (bfloat16 preferred for Blackwell)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)

        # 1. TF32 Matrix Multiplication acceleration on Tensor Cores
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # 2. Select optimal mixed precision format
        # Blackwell / Ada / Ampere natively support BF16 with zero underflow risk
        if torch.cuda.is_bf16_supported():
            mixed_dtype = torch.bfloat16
            precision_name = "BFloat16 (Native Blackwell / Ada Tensor Cores)"
        else:
            mixed_dtype = torch.float16
            precision_name = "Float16 (AMP)"

        if verbose:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[GPU Optimizer] Active Device: {device_name} ({total_vram_gb:.2f} GB VRAM)")
            print(f"[GPU Optimizer] Acceleration: TF32 Enabled | cuDNN Benchmark: True | Precision: {precision_name}")
    else:
        device = torch.device("cpu")
        mixed_dtype = torch.float32
        if verbose:
            print("[GPU Optimizer] CUDA not detected. Falling back to CPU execution.")

    return device, mixed_dtype


def get_amp_context(device: torch.device, dtype: Optional[torch.dtype] = None):
    """
    Returns Automatic Mixed Precision (AMP) autocast context manager.
    """
    if device.type == "cuda":
        amp_dtype = dtype if dtype is not None else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int = 512,
    shuffle: bool = True,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Constructs high-throughput DataLoader with pinned host memory and asynchronous prefetching.
    """
    is_cuda = (device is not None and device.type == "cuda") or torch.cuda.is_available()
    pin_mem = is_cuda
    workers = 0 if not is_cuda else min(num_workers, os.cpu_count() or 1)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_mem,
        num_workers=workers,
        drop_last=drop_last,
        persistent_workers=(workers > 0),
    )


class VectorizedCandidateSampler:
    """
    Vectorized candidate action generation directly on GPU Tensor Cores.
    Replaces sequential Python loops with a single batched forward pass.
    Shape: (B, N, action_dim) where N is candidate count.
    """

    @staticmethod
    def sample_candidates_vectorized(
        policy: nn.Module = None,
        states: torch.Tensor = None,
        num_candidates: int = 16,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            policy: HurdlePolicy instance
            states: (B, state_dim) or (1, state_dim) tensor on GPU
            num_candidates: N candidates to propose per state
        Returns:
            candidates: (B, num_candidates, action_dim) tensor on GPU
        """
        if policy is None and "policy" in kwargs:
            policy = kwargs["policy"]
        if states is None:
            states = kwargs.get("states", kwargs.get("state"))

        if states.dim() == 1:
            states = states.unsqueeze(0)

        B, D = states.shape
        # Expand states to (B * num_candidates, D)
        expanded_states = states.unsqueeze(1).repeat(1, num_candidates, 1).view(B * num_candidates, D)

        gate_threshold = kwargs.get("gate_threshold", 0.50)
        with torch.no_grad():
            actions, _ = policy.sample(expanded_states, deterministic=deterministic)
            if hasattr(policy, "dose_gate") and gate_threshold != 0.50:
                gate_logits, _, _ = policy(expanded_states)
                gate_probs = torch.sigmoid(gate_logits)
                gate_mask = (gate_probs >= gate_threshold).float()
                actions = actions * gate_mask

        # Reshape to (B, num_candidates, action_dim)
        action_dim = actions.shape[-1]
        return actions.view(B, num_candidates, action_dim)

    @classmethod
    def sample_candidates(cls, *args, **kwargs) -> torch.Tensor:
        return cls.sample_candidates_vectorized(*args, **kwargs)


class VectorizedRiskSelector:
    """
    Fully GPU-vectorized Lagrangian Risk-Aware Selection across N candidates.
    Evaluates RewardCritic and FailureCriticEnsemble in a single fused kernel.
    """

    @staticmethod
    def select_best_actions(
        states: torch.Tensor,
        candidates: torch.Tensor,
        reward_critic: nn.Module,
        failure_ensemble: nn.Module,
        lambda_dual: float = 2.0,
        safety_budget: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (B, state_dim)
            candidates: (B, N, action_dim)
            reward_critic: RewardCritic
            failure_ensemble: FailureCriticEnsemble (M critics)
        Returns:
            chosen_actions: (B, action_dim)
            chosen_q_vals: (B,)
            chosen_risks: (B,)
        """
        B, N, A_dim = candidates.shape
        D = states.shape[-1]

        # Flatten batch and candidates for unified forward pass
        flat_states = states.unsqueeze(1).repeat(1, N, 1).view(B * N, D)
        flat_actions = candidates.view(B * N, A_dim)

        with torch.no_grad():
            # 1. Evaluate Q_r
            q_vals = reward_critic(flat_states, flat_actions).view(B, N)

            # 2. Evaluate Failure Critic Ensemble
            conservative_risk, _ = failure_ensemble(flat_states, flat_actions)
            conservative_risk = conservative_risk.view(B, N)

        # 3. Lagrangian utility calculation
        util = q_vals - (lambda_dual * conservative_risk)

        # 4. Safe mask filtering
        safe_mask = (conservative_risk <= safety_budget)

        # For batch rows with at least one safe action, penalize unsafe actions
        has_safe = safe_mask.any(dim=-1)
        util_penalized = util.clone()
        util_penalized[~safe_mask] = -1e9

        # Best index for rows with safe candidates
        best_safe_idx = torch.argmax(util_penalized, dim=-1)
        # Fallback index for rows where no candidate meets safety budget (minimum risk)
        best_fallback_idx = torch.argmin(conservative_risk, dim=-1)

        best_idx = torch.where(has_safe, best_safe_idx, best_fallback_idx)

        # Gather chosen actions
        batch_indices = torch.arange(B, device=states.device)
        chosen_actions = candidates[batch_indices, best_idx]
        chosen_q = q_vals[batch_indices, best_idx]
        chosen_risk = conservative_risk[batch_indices, best_idx]

        return chosen_actions, chosen_q, chosen_risk
