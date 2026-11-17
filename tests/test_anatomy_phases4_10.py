from __future__ import annotations

import numpy as np

from src.geometry.adaptive_mesh import build_adaptive_mesh
from src.geometry.surface import binary_surface_metrics
from src.refinement.rescue import build_rescue_priority, extract_priority_rois


def test_rescue_priority_is_bounded_and_weighted() -> None:
    prediction = np.zeros((16, 16, 16), dtype=np.uint8)
    prediction[4:12, 4:12, 4:12] = 1
    entropy = np.zeros_like(prediction, dtype=np.float32)
    entropy[7:9, 7:9, 7:9] = 1
    violation = np.zeros_like(entropy)
    violation[1:3, 1:3, 1:3] = 1
    priority, boundary = build_rescue_priority(
        entropy,
        violation,
        prediction,
        {"uncertainty": 0.45, "anatomical_violation": 0.35, "boundary_complexity": 0.20},
    )
    assert priority.shape == prediction.shape and boundary.shape == prediction.shape
    assert 0 <= priority.min() <= priority.max() <= 1
    assert priority[7, 7, 7] >= 0.45
    assert boundary.max() > 0


def test_priority_roi_extraction_is_limited_and_in_bounds() -> None:
    mask = np.zeros((32, 32, 32), dtype=bool)
    mask[3:8, 3:8, 3:8] = True
    mask[20:28, 20:28, 20:28] = True
    priority = mask.astype(np.float32)
    rois = extract_priority_rois(
        mask,
        priority,
        (1.0, 1.0, 1.0),
        roi_size=(16, 16, 16),
        minimum_component_ml=0.01,
        maximum_rois=1,
    )
    assert len(rois) == 1
    assert all(0 <= item.start < item.stop <= 32 for item in rois[0])


def test_surface_metrics_are_exact_for_identical_masks() -> None:
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:15, 5:15, 5:15] = True
    metrics = binary_surface_metrics(mask, mask, (1.0, 1.0, 1.0))
    for name in ("dice", "iou", "precision", "recall", "surface_dice_1mm", "surface_dice_2mm"):
        assert metrics[name] == 1.0
    for name in ("assd_mm", "hd95_mm", "hausdorff_mm", "surface_rmse_mm", "volume_error_percent"):
        assert metrics[name] == 0.0


def test_adaptive_mesh_reduces_low_priority_cube() -> None:
    mask = np.zeros((24, 24, 24), dtype=np.uint8)
    mask[4:20, 4:20, 4:20] = 1
    zeros = np.zeros_like(mask, dtype=np.float32)
    config = {
        "priority_weights": {
            "uncertainty": 0.35,
            "curvature": 0.30,
            "boundary_complexity": 0.20,
            "anisotropy": 0.15,
        },
        "priority_bins": [0.33, 0.66],
        "low_priority_cluster_mm": 3.0,
        "medium_priority_cluster_mm": 1.5,
        "high_priority_cluster_mm": 0.0,
    }
    baseline, adaptive, metrics = build_adaptive_mesh(
        mask, zeros, zeros, np.eye(4), (1.0, 1.0, 1.0), config
    )
    assert len(adaptive.vertices) < len(baseline.vertices)
    assert len(adaptive.faces) > 0
    assert metrics["vertex_reduction_fraction"] > 0
