"""Unit tests for Phase 2 SDF, model, and dynamic loss contracts."""

from __future__ import annotations

import numpy as np
import torch

from src.dataset.sdf import multi_class_sdf, signed_distance_field
from src.losses.multitask import MultiTaskSegmentationSDFLoss, sdf_weight_for_epoch
from src.models.multitask_unet import MultiTaskUNet


def test_signed_distance_sign_truncation_and_physical_spacing() -> None:
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[3:6, 3:6, 3:6] = True
    sdf = signed_distance_field(mask, (1.0, 1.0, 2.0), max_distance_mm=4.0)
    assert sdf.shape == mask.shape
    assert sdf.dtype == np.float32
    assert sdf[4, 4, 4] > 0
    assert sdf[0, 0, 0] < 0
    assert np.max(sdf) <= 1 and np.min(sdf) >= -1
    assert sdf[4, 4, 6] < sdf[4, 6, 4]  # Two z-index steps are farther in millimetres.


def test_multiclass_sdf_channel_order() -> None:
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[1:3, 1:3, 1:3] = 1
    labels[5:7, 5:7, 5:7] = 2
    sdf = multi_class_sdf(labels, (1, 2), (1.0, 1.0, 1.0), max_distance_mm=5.0)
    assert sdf.shape == (2, 8, 8, 8)
    assert sdf[0, 1, 1, 1] > 0 and sdf[1, 1, 1, 1] < 0
    assert sdf[1, 5, 5, 5] > 0 and sdf[0, 5, 5, 5] < 0


def test_multitask_model_and_loss_shapes() -> None:
    model = MultiTaskUNet(
        in_channels=2,
        feature_channels=4,
        segmentation_channels=3,
        sdf_channels=2,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=1,
    )
    image = torch.randn(1, 2, 32, 32, 32)
    logits, sdf = model(image)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert sdf.shape == (1, 2, 32, 32, 32)
    assert torch.all(sdf <= 1) and torch.all(sdf >= -1)
    labels = torch.zeros(1, 1, 32, 32, 32, dtype=torch.long)
    target_sdf = torch.full_like(sdf, -1.0)
    objective = MultiTaskSegmentationSDFLoss()
    total, parts = objective(logits, sdf, labels, target_sdf, sdf_weight=0.1)
    assert torch.isfinite(total)
    assert set(parts) == {"segmentation", "sdf"}


def test_sdf_schedule_is_staged() -> None:
    values = [
        sdf_weight_for_epoch(
            epoch, 30, early=0.05, middle=0.15, late=0.25,
            early_fraction=1 / 3, middle_fraction=2 / 3,
        )
        for epoch in (1, 10, 11, 20, 21, 30)
    ]
    assert values == [0.05, 0.05, 0.15, 0.15, 0.25, 0.25]
