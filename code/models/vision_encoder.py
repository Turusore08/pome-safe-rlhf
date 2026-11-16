"""
Vision Encoder for Surface Photography of POME Effluent.
Extracts latent representations h_z from camera images of POME culture
using ConvNeXt architecture (penjelasan.md §2 · Sensor + Vision).
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvNeXtBlock(nn.Module):
    """Depthwise separable ConvNeXt-style building block for fast visual processing."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # back to (N, C, H, W)
        return residual + x


class VisionEncoder(nn.Module):
    """
    ConvNeXt-based vision encoder extracting latent vector h_z from POME images.
    Can also instantiate pre-trained timm backbones if available.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 32,
        use_timm: bool = False,
        timm_model_name: str = "convnext_tiny",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_timm = use_timm

        if use_timm:
            try:
                import timm
                self.backbone = timm.create_model(
                    timm_model_name,
                    pretrained=False,
                    num_classes=0,
                    in_chans=in_channels,
                )
                feat_dim = self.backbone.num_features
                self.proj = nn.Linear(feat_dim, latent_dim)
            except Exception:
                self.use_timm = False

        if not self.use_timm:
            # Custom lightweight ConvNeXt architecture for edge deployment
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=4, stride=4),
                nn.GroupNorm(1, 32),
            )
            self.stage1 = nn.Sequential(
                ConvNeXtBlock(32),
                ConvNeXtBlock(32),
            )
            self.downsample = nn.Sequential(
                nn.GroupNorm(1, 32),
                nn.Conv2d(32, 64, kernel_size=2, stride=2),
            )
            self.stage2 = nn.Sequential(
                ConvNeXtBlock(64),
                ConvNeXtBlock(64),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, latent_dim),
            )

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode an RGB image tensor (B, 3, H, W) into visual latent h_z (B, latent_dim).
        """
        if self.use_timm:
            feats = self.backbone(img)
            h_z = self.proj(feats)
        else:
            x = self.stem(img)
            x = self.stage1(x)
            x = self.downsample(x)
            x = self.stage2(x)
            pooled = self.pool(x).flatten(1)
            h_z = self.proj(pooled)

        return h_z

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.encode_image(img)
