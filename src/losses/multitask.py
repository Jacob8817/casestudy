"""Configurable segmentation and boundary-focused SDF supervision."""

from __future__ import annotations

import torch
from monai.losses import DiceCELoss


def sdf_weight_for_epoch(
    epoch: int,
    max_epochs: int,
    *,
    early: float,
    middle: float,
    late: float,
    early_fraction: float,
    middle_fraction: float,
) -> float:
    if epoch < 1 or max_epochs < 1 or not 0 < early_fraction < middle_fraction < 1:
        raise ValueError("Invalid epoch or SDF schedule fractions")
    progress = epoch / max_epochs
    return float(early if progress <= early_fraction else middle if progress <= middle_fraction else late)


class MultiTaskSegmentationSDFLoss(torch.nn.Module):
    def __init__(self, *, boundary_band: float = 0.2, boundary_multiplier: float = 4.0) -> None:
        super().__init__()
        if boundary_band <= 0 or boundary_band > 1 or boundary_multiplier < 0:
            raise ValueError("Invalid SDF boundary weighting")
        self.segmentation = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.boundary_band = float(boundary_band)
        self.boundary_multiplier = float(boundary_multiplier)

    def forward(
        self,
        segmentation_logits: torch.Tensor,
        predicted_sdf: torch.Tensor,
        labels: torch.Tensor,
        target_sdf: torch.Tensor,
        *,
        sdf_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if predicted_sdf.shape != target_sdf.shape:
            raise ValueError("Predicted and target SDF tensors must have identical shapes")
        segmentation_loss = self.segmentation(segmentation_logits, labels)
        pointwise = torch.nn.functional.smooth_l1_loss(predicted_sdf, target_sdf, reduction="none")
        weights = 1.0 + self.boundary_multiplier * (target_sdf.abs() <= self.boundary_band).to(pointwise.dtype)
        sdf_loss = (pointwise * weights).sum() / weights.sum().clamp_min(1.0)
        total = segmentation_loss + float(sdf_weight) * sdf_loss
        return total, {"segmentation": segmentation_loss, "sdf": sdf_loss}
