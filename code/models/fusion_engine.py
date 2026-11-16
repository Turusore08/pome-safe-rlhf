"""
Multimodal Fusion Engine with Zero-Order Hold (ZOH) Memory.
Bridges temporal sampling rate mismatch between fast IoT sensors (5s / 6h)
and slow POME surface photography by holding the last valid visual embedding in memory.
Constructs unified state s_t = [h_x, h_z] (penjelasan.md §2 · Sensor + Vision).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from models.sensor_encoder import SensorEncoder
from models.vision_encoder import VisionEncoder


class ZeroOrderHold:
    """
    Zero-Order Hold (ZOH) synchronizer for slow visual features.
    Holds the last valid feature in memory until a fresh photograph arrives,
    preventing artificial interpolation artifacts across bioprocess stages.
    """

    def __init__(self, visual_dim: int = 32):
        self.visual_dim = visual_dim
        self.last_valid_feature: Optional[torch.Tensor] = None
        self.last_seen_step: int = -1

    def update(self, h_z: torch.Tensor, step: int):
        """Update buffer with fresh visual feature embedding."""
        self.last_valid_feature = h_z.clone().detach()
        self.last_seen_step = step

    def get_last_valid(self, default_batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Retrieve held visual feature vector using Zero-Order Hold."""
        if self.last_valid_feature is None:
            # Cold-start initialization (zeros)
            dev = device if device is not None else torch.device("cpu")
            return torch.zeros((default_batch_size, self.visual_dim), device=dev)
        return self.last_valid_feature

    def hold_last(self, fresh_hz: Optional[torch.Tensor] = None, step: int = 0) -> torch.Tensor:
        """Return fresh visual feature if available, otherwise hold last."""
        if fresh_hz is not None:
            self.update(fresh_hz, step)
            return fresh_hz
        return self.get_last_valid()

    def reset(self):
        """Reset ZOH memory buffer at episode boundary."""
        self.last_valid_feature = None
        self.last_seen_step = -1


class MultimodalFusionEngine(nn.Module):
    """
    Multimodal Fusion Engine:
    Encodes sensor dynamics into h_x, holds visual embedding h_z via ZOH,
    and fuses them via concatenation: s_t = [h_x, h_z].
    """

    def __init__(
        self,
        sensor_latent_dim: int = 64,
        visual_latent_dim: int = 32,
    ):
        super().__init__()
        self.sensor_latent_dim = sensor_latent_dim
        self.visual_latent_dim = visual_latent_dim
        self.total_state_dim = sensor_latent_dim + visual_latent_dim

        self.sensor_encoder = SensorEncoder(latent_dim=sensor_latent_dim)
        self.vision_encoder = VisionEncoder(latent_dim=visual_latent_dim)
        self.zoh = ZeroOrderHold(visual_dim=visual_latent_dim)

    def fuse_state(
        self,
        h_x: torch.Tensor,
        h_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multimodal state concatenation: s_t = [h_x, h_z].
        """
        s_t = torch.cat([h_x, h_z], dim=-1)
        return s_t

    def forward(
        self,
        sensor_obs: torch.Tensor,
        vision_img: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass:
        1. h_x = SensorEncoder(sensor_obs)
        2. h_z = VisionEncoder(vision_img) with ZOH fallback
        3. s_t = torch.cat([h_x, h_z], dim=-1)
        """
        h_x = self.sensor_encoder(sensor_obs)

        if vision_img is not None:
            fresh_hz = self.vision_encoder(vision_img)
            h_z = self.zoh.hold_last(fresh_hz, step=step)
        else:
            h_z = self.zoh.hold_last(fresh_hz=None, step=step)
            if h_z.shape[0] != h_x.shape[0]:
                h_z = h_z.expand(h_x.shape[0], -1)

        s_t = self.fuse_state(h_x, h_z)
        return s_t, h_x, h_z
