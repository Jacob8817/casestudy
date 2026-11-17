"""Physical signed-distance targets for brain/skull multi-task learning."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def signed_distance_field(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    max_distance_mm: float,
) -> np.ndarray:
    """Return a truncated SDF normalized to [-1, 1], positive inside the mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError("SDF input must be a 3D mask")
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("SDF spacing must contain three positive finite values")
    if not np.isfinite(max_distance_mm) or max_distance_mm <= 0:
        raise ValueError("SDF truncation distance must be positive")
    if not np.any(binary):
        return np.full(binary.shape, -1.0, dtype=np.float32)
    inside = distance_transform_edt(binary, sampling=spacing)
    outside = distance_transform_edt(~binary, sampling=spacing)
    signed_mm = inside - outside
    result = np.clip(signed_mm / float(max_distance_mm), -1.0, 1.0)
    return result.astype(np.float32)


def multi_class_sdf(
    labels: np.ndarray,
    class_ids: tuple[int, ...],
    spacing_mm: tuple[float, float, float],
    *,
    max_distance_mm: float,
) -> np.ndarray:
    """Return one normalized physical SDF channel per requested class."""
    source = np.asarray(labels)
    if source.ndim != 3 or not class_ids:
        raise ValueError("Labels must be 3D and class_ids must be non-empty")
    return np.stack(
        [
            signed_distance_field(source == class_id, spacing_mm, max_distance_mm=max_distance_mm)
            for class_id in class_ids
        ],
        axis=0,
    )

