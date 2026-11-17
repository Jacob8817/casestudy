"""Common development-set scoring for the head segmentation comparisons.

All methods are scored on the same label-derived head crop. This is an oracle-ROI
development benchmark, not a CT-only clinical evaluation or an independent test.
"""

from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

from .brain_skull import dice_per_class, hd95_mm


def reference_geometry(record: dict) -> tuple[tuple[int, int, int], np.ndarray]:
    shape = tuple(int(value) for value in record["shape"])
    affine = np.asarray(record["ct_affine"], dtype=np.float64)
    translation = np.eye(4)
    translation[:3, 3] = record["crop_start"]
    return shape, affine @ translation


def cropped_ct_nifti(dataset_root: Path, record: dict, output_path: Path) -> Path:
    """Materialize the same oracle crop as a HU NIfTI for external methods."""
    if record["split"] not in {"train", "val"}:
        raise ValueError("Head CT export refuses the official test split")
    shape, affine = reference_geometry(record)
    if output_path.is_file():
        existing = nib.load(str(output_path))
        if existing.shape == shape and np.allclose(existing.affine, affine, atol=1e-3, rtol=0):
            return output_path
        raise ValueError(f"Existing cropped CT has unexpected geometry: {output_path}")
    original = nib.load(str(dataset_root / record["case_id"] / "ct.nii.gz"))
    start = record["crop_start"]
    crop = tuple(slice(start[axis], start[axis] + shape[axis]) for axis in range(3))
    image = np.asarray(original.dataobj[crop], dtype=np.float32)
    if image.shape != shape or not np.isfinite(image).all():
        raise ValueError(f"Invalid cropped CT: {record['case_id']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image, affine), str(output_path))
    return output_path


def skull_hu_morphology(record: dict) -> np.ndarray:
    """Fixed >300 HU baseline with one-voxel closing and small-component removal."""
    broad_window = np.load(record["original_image_path"], mmap_mode="r")[0]
    bone = np.asarray(broad_window > 0.65, dtype=bool)  # >300 HU after [-1000,1000] scaling.
    bone = ndi.binary_closing(bone, structure=np.ones((3, 3, 3), dtype=bool))
    components, count = ndi.label(bone)
    if count:
        sizes = np.bincount(components.ravel())
        keep = sizes >= 100
        keep[0] = False
        bone = keep[components]
    return np.asarray(bone, dtype=np.uint8) * 2


def external_masks_to_labels(record: dict, brain_path: Path, skull_path: Path) -> np.ndarray:
    """Resample two external binary masks onto the benchmark crop; skull wins overlaps."""
    shape, affine = reference_geometry(record)
    result = np.zeros(shape, dtype=np.uint8)
    for class_id, path in ((1, brain_path), (2, skull_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
        image = nib.load(str(path))
        if image.shape != shape or not np.allclose(image.affine, affine, atol=1e-3, rtol=0):
            image = resample_from_to(image, (shape, affine), order=0)
        mask = np.asarray(image.dataobj) > 0.5
        result[mask] = class_id
    return result


def external_multilabel_to_labels(record: dict, prediction_path: Path) -> np.ndarray:
    shape, affine = reference_geometry(record)
    image = nib.load(str(prediction_path))
    if image.shape != shape or not np.allclose(image.affine, affine, atol=1e-3, rtol=0):
        image = resample_from_to(image, (shape, affine), order=0)
    result = np.asarray(image.dataobj, dtype=np.uint8)
    if result.shape != shape or not np.isin(np.unique(result), [0, 1, 2]).all():
        raise ValueError(f"Unexpected external labels: {prediction_path}")
    return result


def score_prediction(record: dict, prediction: np.ndarray, method: str, runtime_seconds: float | None) -> list[dict]:
    truth = np.load(record["label_path"], mmap_mode="r")
    if prediction.shape != truth.shape:
        raise ValueError(f"Prediction shape mismatch: {method}, {record['case_id']}")
    affine = np.asarray(record["ct_affine"], dtype=np.float64)
    spacing = tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))
    dice = dice_per_class(prediction, truth)
    rows = []
    for class_id, organ in ((1, "brain"), (2, "skull")):
        predicted = prediction == class_id
        target = truth == class_id
        rows.append({
            "method": method,
            "case_id": record["case_id"],
            "organ": organ,
            "dice": dice[organ],
            "hd95_mm": hd95_mm(predicted, target, spacing),
            "volume_error_pct": 100.0 * (int(predicted.sum()) - int(target.sum())) / int(target.sum()),
            "inference_seconds": runtime_seconds,
        })
    return rows


def timed_prediction(function, *args, **kwargs) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    prediction = function(*args, **kwargs)
    return prediction, time.perf_counter() - started
