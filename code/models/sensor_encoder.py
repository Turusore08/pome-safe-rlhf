"""
Sensor Encoder for Telemetry Dynamics.
Compresses multi-variable temporal IoT streams into a compact latent vector h_x
using 1D Convolutional layers (Conv1d) with residual connections.
(penjelasan.md §2 · Sensor + Vision)
"""

import torch
import torch.nn as nn
from typing import Optional


class SensorEncoder(nn.Module):
    """
    1D-Convolutional sensor encoder for bioreactor probe series x_t.
    Projects high-frequency 16-variable time series into latent vector h_x.
    """

    def __init__(
        self,
        in_channels: int = 16,
        history_len: int = 12,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # 1D Temporal Convolutional feature extractor
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu2 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Latent projection head
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Linear projection for direct 192-dim tabular window descriptor input
        self.tabular_proj = nn.Sequential(
            nn.Linear(192, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode_sensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes sensor observations into latent h_x.
        Supports both raw 3D sequential shape (B, C, L) and 2D windowed features (B, 192).
        """
        if x.dim() == 2 and x.shape[-1] == 192:
            # Tabular windowed statistical descriptors
            h_x = self.tabular_proj(x)
        elif x.dim() == 2 and x.shape[-1] == self.in_channels:
            # Single-step observation (B, 16) -> expand to (B, 16, 1)
            x_seq = x.unsqueeze(-1)
            feat = self.relu1(self.conv1(x_seq))
            feat = self.relu2(self.conv2(feat))
            pooled = self.pool(feat).squeeze(-1)
            h_x = self.proj(pooled)
        else:
            # 3D Sequential stream (B, in_channels, history_len)
            feat = self.relu1(self.conv1(x))
            feat = self.relu2(self.conv2(feat))
            pooled = self.pool(feat).squeeze(-1)
            h_x = self.proj(pooled)

        return h_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_sensor(x)
