"""Shared 3D U-Net representation with segmentation and SDF output heads."""

from __future__ import annotations

import torch
from monai.networks.nets import UNet


class MultiTaskUNet(torch.nn.Module):
    """A shared MONAI U-Net followed by independent segmentation and SDF heads."""

    def __init__(
        self,
        *,
        in_channels: int = 2,
        feature_channels: int = 16,
        segmentation_channels: int = 3,
        sdf_channels: int = 2,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        strides: tuple[int, ...] = (2, 2, 2),
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        act: str = "PRELU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if segmentation_channels < 2 or sdf_channels < 1 or feature_channels < 1:
            raise ValueError("Invalid multi-task output dimensions")
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
            dropout=dropout,
        )
        self.segmentation_head = torch.nn.Conv3d(feature_channels, segmentation_channels, kernel_size=1)
        self.sdf_head = torch.nn.Sequential(
            torch.nn.Conv3d(feature_channels, sdf_channels, kernel_size=1),
            torch.nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        return self.segmentation_head(features), self.sdf_head(features)

