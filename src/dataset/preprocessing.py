"""Deterministic CT preprocessing and memory-bounded 3D patch extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy.ndimage import affine_transform

from audit_dataset import orientation_from_affine


@dataclass(frozen=True)
class PreprocessingSpec:
    orientation: str
    spacing_mm: tuple[float, float, float]
    hu_lower: float
    hu_upper: float
    normalized_lower: float
    normalized_upper: float
    roi_size: tuple[int, int, int]
    foreground_sampling_probability: float
    body_threshold_hu: float
    foreground_crop_enabled: bool
    foreground_crop_threshold_hu: float
    foreground_crop_margin_voxels: tuple[int, int, int]
    random_seed: int

    def fingerprint(self, *, manifest_sha256: str, schema_sha256: str) -> str:
        payload = {
            "spec": asdict(self),
            "manifest_sha256": manifest_sha256,
            "schema_sha256": schema_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PatchTrace:
    case_id: str
    sample_index: int
    mode: str
    selected_class_id: int
    center: tuple[int, int, int]
    requested_start: tuple[int, int, int]
    source_start: tuple[int, int, int]
    source_end: tuple[int, int, int]
    padding_before: tuple[int, int, int]
    padding_after: tuple[int, int, int]
    derived_seed: int


def _tuple3(value: object, field: str, cast: type) -> tuple[Any, Any, Any]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must be a three-item list")
    converted = tuple(cast(item) for item in value)
    return converted  # type: ignore[return-value]


def load_preprocessing_spec(path: Path) -> PreprocessingSpec:
    try:
        root = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load preprocessing config {path}: {exc}") from exc
    if not isinstance(root, dict):
        raise ValueError("Dataset configuration must be a mapping")
    prep = root.get("preprocessing")
    reproducibility = root.get("reproducibility")
    if not isinstance(prep, dict) or not isinstance(reproducibility, dict):
        raise ValueError("Dataset configuration lacks preprocessing/reproducibility mappings")
    window = prep.get("intensity_window_hu")
    normalization = prep.get("normalization")
    sampling = prep.get("patch_sampling")
    crop = prep.get("foreground_crop")
    if not all(isinstance(item, dict) for item in (window, normalization, sampling, crop)):
        raise ValueError("Preprocessing window, normalization, sampling, and crop settings are required")
    spec = PreprocessingSpec(
        orientation=str(prep.get("orientation")),
        spacing_mm=_tuple3(prep.get("spacing_mm"), "spacing_mm", float),
        hu_lower=float(window["lower"]),
        hu_upper=float(window["upper"]),
        normalized_lower=float(normalization["output_range"][0]),
        normalized_upper=float(normalization["output_range"][1]),
        roi_size=_tuple3(prep.get("roi_size"), "roi_size", int),
        foreground_sampling_probability=float(sampling["foreground_probability"]),
        body_threshold_hu=float(sampling["body_threshold_hu"]),
        foreground_crop_enabled=bool(crop["enabled"]),
        foreground_crop_threshold_hu=float(crop["threshold_hu"]),
        foreground_crop_margin_voxels=_tuple3(crop["margin_voxels"], "margin_voxels", int),
        random_seed=int(reproducibility["random_seed"]),
    )
    if spec.orientation != "RAS":
        raise ValueError("This release requires target orientation RAS")
    if any(value <= 0 for value in spec.spacing_mm + spec.roi_size):
        raise ValueError("Spacing and ROI dimensions must be positive")
    if not spec.hu_lower < spec.hu_upper:
        raise ValueError("HU lower bound must be below upper bound")
    if not spec.normalized_lower < spec.normalized_upper:
        raise ValueError("Normalized lower bound must be below upper bound")
    if not 0 <= spec.foreground_sampling_probability <= 1:
        raise ValueError("Foreground sampling probability must lie in [0, 1]")
    return spec


def normalize_ct(ct_hu: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    """Clip calibrated HU values and linearly map them to a fixed range."""
    if ct_hu.ndim != 3 or not np.all(np.isfinite(ct_hu)):
        raise ValueError("CT must be a finite 3D array")
    clipped = np.clip(ct_hu, spec.hu_lower, spec.hu_upper).astype(np.float32, copy=False)
    unit = (clipped - np.float32(spec.hu_lower)) / np.float32(spec.hu_upper - spec.hu_lower)
    result = unit * np.float32(spec.normalized_upper - spec.normalized_lower) + np.float32(
        spec.normalized_lower
    )
    return result.astype(np.float32, copy=False)


def _orientation_from_matrix(affine: np.ndarray) -> str:
    return orientation_from_affine(tuple(float(value) for value in affine[:3, :].reshape(-1)))


def canonicalize_to_ras(
    image: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Permute/flip voxel axes to closest-canonical RAS without interpolation."""
    if image.shape != label.shape or image.ndim != 3 or affine.shape != (4, 4):
        raise ValueError("Image, label, and affine geometry are inconsistent")
    orientation = _orientation_from_matrix(affine)
    if orientation == "UNKNOWN" or len(orientation) != 3:
        raise ValueError("Cannot canonicalize an unknown orientation")
    world_group = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
    positive = {"R", "A", "S"}
    try:
        permutation = tuple(next(i for i, code in enumerate(orientation) if world_group[code] == world) for world in range(3))
    except (KeyError, StopIteration) as exc:
        raise ValueError(f"Unsupported orientation {orientation}") from exc
    output_image = np.transpose(image, axes=permutation)
    output_label = np.transpose(label, axes=permutation)
    transform = np.zeros((4, 4), dtype=np.float64)
    transform[3, 3] = 1.0
    flips: list[bool] = []
    for output_axis, input_axis in enumerate(permutation):
        code = orientation[input_axis]
        flip = code not in positive
        flips.append(flip)
        transform[input_axis, output_axis] = -1.0 if flip else 1.0
        if flip:
            transform[input_axis, 3] = image.shape[input_axis] - 1
            output_image = np.flip(output_image, axis=output_axis)
            output_label = np.flip(output_label, axis=output_axis)
    output_affine = affine @ transform
    result_orientation = _orientation_from_matrix(output_affine)
    if result_orientation != "RAS":
        raise ValueError(f"Canonicalization failed: {orientation} -> {result_orientation}")
    return output_image, output_label, output_affine, {
        "source_orientation": orientation,
        "target_orientation": result_orientation,
        "axis_permutation": list(permutation),
        "axis_flips": flips,
        "was_noop": permutation == (0, 1, 2) and not any(flips),
    }


def resample_pair(
    image_hu: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
    target_spacing_mm: Sequence[float],
    *,
    image_fill_hu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Affine-aware spacing resampling with linear CT and nearest labels."""
    if image_hu.shape != label.shape or image_hu.ndim != 3 or affine.shape != (4, 4):
        raise ValueError("Image, label, and affine geometry are inconsistent")
    target = np.asarray(tuple(target_spacing_mm), dtype=np.float64)
    if target.shape != (3,) or np.any(~np.isfinite(target)) or np.any(target <= 0):
        raise ValueError("Target spacing must contain three positive finite values")
    linear = affine[:3, :3]
    source = np.linalg.norm(linear, axis=0)
    if np.allclose(source, target, rtol=1e-5, atol=1e-5):
        return image_hu, label, affine.copy(), {
            "source_spacing_mm": source.tolist(),
            "target_spacing_mm": target.tolist(),
            "source_shape": list(image_hu.shape),
            "target_shape": list(image_hu.shape),
            "was_noop": True,
            "image_interpolation": "none",
            "label_interpolation": "none",
        }
    directions = linear / source
    output_affine = affine.copy()
    output_affine[:3, :3] = directions * target
    output_shape = tuple(
        max(1, int(np.floor((size - 1) * spacing / requested + 1.0 + 1e-6)))
        for size, spacing, requested in zip(image_hu.shape, source, target)
    )
    mapping = np.linalg.inv(affine) @ output_affine
    output_image = affine_transform(
        image_hu,
        matrix=mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=float(image_fill_hu),
        prefilter=False,
    ).astype(np.float32, copy=False)
    output_label = affine_transform(
        label,
        matrix=mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=output_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.uint8, copy=False)
    if not set(int(value) for value in np.unique(output_label)) <= set(int(value) for value in np.unique(label)):
        raise ValueError("Nearest-neighbour label resampling introduced a new class id")
    return output_image, output_label, output_affine, {
        "source_spacing_mm": source.tolist(),
        "target_spacing_mm": target.tolist(),
        "source_shape": list(image_hu.shape),
        "target_shape": list(output_shape),
        "was_noop": False,
        "image_interpolation": "linear",
        "label_interpolation": "nearest",
    }


def ct_foreground_bbox(
    image_hu: np.ndarray,
    *,
    threshold_hu: float,
    margin_voxels: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return a conservative CT-derived crop box; never uses the target label."""
    if image_hu.ndim != 3:
        raise ValueError("Foreground crop expects a 3D CT")
    margin = tuple(int(value) for value in margin_voxels)
    if len(margin) != 3 or any(value < 0 for value in margin):
        raise ValueError("Crop margin must contain three non-negative integers")
    foreground = image_hu > threshold_hu
    if not bool(np.any(foreground)):
        return tuple((0, size) for size in image_hu.shape)  # type: ignore[return-value]
    bounds: list[tuple[int, int]] = []
    for axis, size in enumerate(image_hu.shape):
        reduced_axes = tuple(index for index in range(3) if index != axis)
        indices = np.flatnonzero(np.any(foreground, axis=reduced_axes))
        bounds.append((max(0, int(indices[0]) - margin[axis]), min(size, int(indices[-1]) + 1 + margin[axis])))
    return tuple(bounds)  # type: ignore[return-value]


def crop_pair(
    image: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
    bbox: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.shape != label.shape or len(bbox) != 3:
        raise ValueError("Crop geometry is inconsistent")
    slices = tuple(slice(int(start), int(end)) for start, end in bbox)
    starts = np.asarray([int(item[0]) for item in bbox], dtype=np.float64)
    output_affine = affine.copy()
    output_affine[:3, 3] = (affine @ np.asarray([*starts, 1.0]))[:3]
    return image[slices], label[slices], output_affine


def _derived_seed(base_seed: int, case_id: str, sample_index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{case_id}:{sample_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _random_coordinate(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
    flat = np.flatnonzero(mask)
    if not flat.size:
        raise ValueError("Cannot sample from an empty candidate mask")
    selected = int(flat[int(rng.integers(flat.size))])
    return tuple(int(value) for value in np.unravel_index(selected, mask.shape))


def extract_centered_patch(
    image: np.ndarray,
    label: np.ndarray,
    center: Sequence[int],
    roi_size: Sequence[int],
    *,
    image_pad_value: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int, int]]]:
    if image.shape != label.shape or image.ndim != 3:
        raise ValueError("Patch extraction requires aligned 3D image and label arrays")
    roi = tuple(int(value) for value in roi_size)
    center3 = tuple(int(value) for value in center)
    if len(roi) != 3 or len(center3) != 3 or any(value <= 0 for value in roi):
        raise ValueError("ROI and center must contain three valid coordinates")
    requested_start = tuple(center3[axis] - roi[axis] // 2 for axis in range(3))
    source_start = tuple(max(0, requested_start[axis]) for axis in range(3))
    source_end = tuple(min(image.shape[axis], requested_start[axis] + roi[axis]) for axis in range(3))
    before = tuple(source_start[axis] - requested_start[axis] for axis in range(3))
    copied = tuple(source_end[axis] - source_start[axis] for axis in range(3))
    after = tuple(roi[axis] - before[axis] - copied[axis] for axis in range(3))
    output_image = np.full(roi, image_pad_value, dtype=np.float32)
    output_label = np.zeros(roi, dtype=np.uint8)
    source_slices = tuple(slice(source_start[axis], source_end[axis]) for axis in range(3))
    target_slices = tuple(slice(before[axis], before[axis] + copied[axis]) for axis in range(3))
    output_image[target_slices] = image[source_slices]
    output_label[target_slices] = label[source_slices]
    return output_image[np.newaxis, ...], output_label, {
        "requested_start": requested_start,
        "source_start": source_start,
        "source_end": source_end,
        "padding_before": before,
        "padding_after": after,
    }


def sample_class_balanced_patch(
    image_normalized: np.ndarray,
    label: np.ndarray,
    image_hu: np.ndarray,
    spec: PreprocessingSpec,
    *,
    case_id: str,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, PatchTrace]:
    """Sample uniformly by present class, with a deterministic body fallback."""
    if image_normalized.shape != label.shape or label.shape != image_hu.shape:
        raise ValueError("Sampler inputs must share one 3D geometry")
    seed = _derived_seed(spec.random_seed, case_id, sample_index)
    rng = np.random.default_rng(seed)
    present = np.unique(label)
    present = present[present > 0]
    choose_foreground = bool(present.size) and rng.random() < spec.foreground_sampling_probability
    if choose_foreground:
        selected_class = int(present[int(rng.integers(present.size))])
        center = _random_coordinate(label == selected_class, rng)
        mode = "foreground_class_balanced"
    else:
        selected_class = 0
        candidates = (label == 0) & (image_hu > spec.body_threshold_hu)
        if not bool(np.any(candidates)):
            candidates = label == 0
        if not bool(np.any(candidates)):
            candidates = np.ones(label.shape, dtype=bool)
        center = _random_coordinate(candidates, rng)
        mode = "body_background" if present.size else "background_only_fallback"
    image_patch, label_patch, details = extract_centered_patch(
        image_normalized,
        label,
        center,
        spec.roi_size,
        image_pad_value=spec.normalized_lower,
    )
    trace = PatchTrace(
        case_id=case_id,
        sample_index=sample_index,
        mode=mode,
        selected_class_id=selected_class,
        center=center,
        requested_start=details["requested_start"],
        source_start=details["source_start"],
        source_end=details["source_end"],
        padding_before=details["padding_before"],
        padding_after=details["padding_after"],
        derived_seed=seed,
    )
    return image_patch, label_patch, trace
