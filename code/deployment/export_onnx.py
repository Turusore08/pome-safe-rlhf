"""
ONNX Model Export for Edge Bioreactor Deployment.
Exports PyTorch policy and critic computational graphs to portable ONNX format
for low-latency deterministic edge execution on Raspberry Pi 4 (penjelasan.md §6).
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.policy_sft import HurdlePolicy


class ExportableHurdlePolicy(nn.Module):
    """Wrapper that outputs deterministic actions directly for ONNX export."""

    def __init__(self, policy: HurdlePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        gate_logits, mu, _ = self.policy(state)
        gate_probs = torch.sigmoid(gate_logits)
        # Deterministic actuation: if p > 0.5 dose mu, else 0
        is_active = (gate_probs > 0.5).float()
        action = is_active * mu
        return action


def export_policy(
    policy: HurdlePolicy,
    output_path: str = "deployment/models/policy_edge.onnx",
    state_dim: int = 224,
    opset_version: int = 14,
) -> str:
    """
    Export the trained policy network to ONNX runtime format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    exportable = ExportableHurdlePolicy(policy.to("cpu"))
    exportable.eval()

    dummy_state = torch.randn(1, state_dim, dtype=torch.float32, device="cpu")

    try:
        # Standard legacy exporter (fast, reliable on Windows without emoji print errors)
        torch.onnx.export(
            exportable,
            dummy_state,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={"state": {0: "batch_size"}, "action": {0: "batch_size"}},
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            exportable,
            dummy_state,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={"state": {0: "batch_size"}, "action": {0: "batch_size"}},
        )

    print(f"Exported ONNX policy model to {output_path}")
    return output_path


if __name__ == "__main__":
    dummy_pol = HurdlePolicy(state_dim=224, action_dim=5)
    export_policy(dummy_pol)
