"""
INT8 Model Quantization for Edge Hardware Optimization.
Applies dynamic quantization (qint8 / int8) to trained neural policy and critic weights.
Reduces memory footprint and enables efficient execution on Raspberry Pi 4 edge nodes.
(penjelasan.md §6).
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.policy_sft import HurdlePolicy


def apply_int8_quantization(
    policy: HurdlePolicy,
    save_path: Optional[str] = "deployment/models/policy_quantized_int8.pt",
) -> nn.Module:
    """
    Applies affine INT8 dynamic quantization to PyTorch Linear layers.
    Maps 32-bit floating point weights to 8-bit signed integers (qint8).
    """
    policy_cpu = policy.to("cpu")
    policy_cpu.eval()

    # Apply PyTorch dynamic quantization
    quantized_policy = torch.ao.quantization.quantize_dynamic(
        policy_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(quantized_policy.state_dict(), save_path)
        print(f"Quantized INT8 policy checkpoint saved to {save_path}")

    return quantized_policy


def compare_quantization_performance(
    fp32_policy: nn.Module,
    int8_policy: nn.Module,
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    Verify numerical consistency between float32 and quantized int8 policies on random states.
    """
    dummy_states = torch.randn(num_samples, 224, dtype=torch.float32)

    with torch.no_grad():
        a_fp32, _ = fp32_policy.sample(dummy_states, deterministic=True)
        a_int8, _ = int8_policy.sample(dummy_states, deterministic=True)

    abs_diff = torch.abs(a_fp32 - a_int8).mean().item()
    max_diff = torch.abs(a_fp32 - a_int8).max().item()

    results = {
        "mean_absolute_difference": float(abs_diff),
        "max_absolute_difference": float(max_diff),
        "int8_consistency": bool(abs_diff < 0.25),
    }
    print(f"INT8 Quantization verification: Mean diff = {abs_diff:.4f}, Max diff = {max_diff:.4f}")
    return results


if __name__ == "__main__":
    pol = HurdlePolicy(state_dim=224, action_dim=5)
    q_pol = apply_int8_quantization(pol)
    compare_quantization_performance(pol, q_pol)
