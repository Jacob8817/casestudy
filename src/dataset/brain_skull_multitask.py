"""CT-only brain/skull caches and deterministic multi-task patch sampling."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from .preprocessing import ct_foreground_bbox
from .sdf import multi_class_sdf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def prepare_ct_only_multitask_case(
    dataset_root: Path,
    record: dict,
    cache_root: Path,
    *,
    threshold_hu: float,
    margin_voxels: tuple[int, int, int],
    max_distance_mm: float,
) -> dict:
    """Cache CT windows, exclusive labels, and two physical SDFs without label-based cropping."""
    if record["split"] not in {"train", "val"}:
        raise ValueError("Multi-task cache refuses official test cases")
    case_id = str(record["case_id"])
    case_root = dataset_root / case_id
    ct_path = case_root / "ct.nii.gz"
    brain_path = case_root / "segmentations" / "brain.nii.gz"
    skull_path = case_root / "segmentations" / "skull.nii.gz"
    contract = {
        "case_id": case_id,
        "split": record["split"],
        "threshold_hu": float(threshold_hu),
        "margin_voxels": list(margin_voxels),
        "max_distance_mm": float(max_distance_mm),
        "source_ct_sha256": _sha256(ct_path),
        "source_brain_sha256": _sha256(brain_path),
        "source_skull_sha256": _sha256(skull_path),
        "overlap_policy": "skull_wins",
    }
    contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    case_cache = cache_root / case_id
    image_path = case_cache / "image.npy"
    label_path = case_cache / "label.npy"
    sdf_path = case_cache / "sdf.npy"
    metadata_path = case_cache / "metadata.json"
    if all(path.is_file() for path in (image_path, label_path, sdf_path, metadata_path)):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract_sha256") == contract_sha256:
            return metadata
        raise ValueError(f"Existing Phase 2 cache has a different contract: {case_id}")

    ct_image = nib.load(str(ct_path))
    hu = np.asarray(ct_image.dataobj, dtype=np.float32)
    if hu.ndim != 3 or not np.isfinite(hu).all():
        raise ValueError(f"Invalid CT: {case_id}")
    bbox = ct_foreground_bbox(hu, threshold_hu=threshold_hu, margin_voxels=margin_voxels)
    slices = tuple(slice(start, end) for start, end in bbox)
    starts = np.asarray([start for start, _ in bbox], dtype=np.float64)
    crop_affine = np.asarray(ct_image.affine, dtype=np.float64).copy()
    crop_affine[:3, 3] = (ct_image.affine @ np.asarray([*starts, 1.0]))[:3]
    spacing = tuple(float(np.linalg.norm(crop_affine[:3, axis])) for axis in range(3))

    brain_image, skull_image = nib.load(str(brain_path)), nib.load(str(skull_path))
    if (
        brain_image.shape != ct_image.shape
        or skull_image.shape != ct_image.shape
        or not np.allclose(brain_image.affine, ct_image.affine, atol=1e-3, rtol=0)
        or not np.allclose(skull_image.affine, ct_image.affine, atol=1e-3, rtol=0)
    ):
        raise ValueError(f"CT/mask geometry mismatch: {case_id}")
    brain = np.asarray(brain_image.dataobj) > 0
    skull = np.asarray(skull_image.dataobj) > 0
    full_label = np.zeros(ct_image.shape, dtype=np.uint8)
    full_label[brain] = 1
    full_label[skull] = 2
    label = full_label[slices]
    if any(not np.any(label == class_id) for class_id in (1, 2)):
        raise ValueError(f"CT-only crop lost a target class: {case_id}")
    crop_hu = hu[slices]
    image = np.stack(
        [
            np.clip((crop_hu + 100.0) / 300.0, 0.0, 1.0),
            np.clip((crop_hu + 300.0) / 1800.0, 0.0, 1.0),
        ],
        axis=0,
    ).astype(np.float16)
    sdf = multi_class_sdf(label, (1, 2), spacing, max_distance_mm=max_distance_mm).astype(np.float16)
    case_cache.mkdir(parents=True, exist_ok=True)
    _atomic_array(image_path, image)
    _atomic_array(label_path, label)
    _atomic_array(sdf_path, sdf)
    metadata = {
        **contract,
        "contract_sha256": contract_sha256,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "sdf_path": str(sdf_path),
        "shape": list(label.shape),
        "crop_start": starts.astype(int).tolist(),
        "crop_bbox": [list(item) for item in bbox],
        "ct_affine": crop_affine.tolist(),
        "spacing_mm": list(spacing),
        "class_bounds": {
            str(class_id): [
                [int(np.where(label == class_id)[axis].min()), int(np.where(label == class_id)[axis].max()) + 1]
                for axis in range(3)
            ]
            for class_id in (1, 2)
        },
        "label_voxels": {str(class_id): int(np.count_nonzero(label == class_id)) for class_id in (1, 2)},
        "crop_selected_without_labels": True,
        "reference_retention": {
            "brain": float(np.count_nonzero(label == 1) / np.count_nonzero(full_label == 1)),
            "skull": float(np.count_nonzero(label == 2) / np.count_nonzero(full_label == 2)),
        },
    }
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata


def prepare_ct_only_multitask_inventory(
    dataset_root: Path,
    records: list[dict],
    cache_root: Path,
    *,
    threshold_hu: float,
    margin_voxels: tuple[int, int, int],
    max_distance_mm: float,
    workers: int = 1,
    progress=None,
) -> list[dict]:
    if workers < 1 or workers > 4:
        raise ValueError("SDF cache workers must lie in [1, 4]")

    def prepare(record: dict) -> dict:
        return prepare_ct_only_multitask_case(
            dataset_root,
            record,
            cache_root,
            threshold_hu=threshold_hu,
            margin_voxels=margin_voxels,
            max_distance_mm=max_distance_mm,
        )

    if workers == 1:
        result = []
        for index, record in enumerate(records, 1):
            result.append(prepare(record))
            if progress is not None:
                progress(index, len(records), record["case_id"])
        return result

    ordered: list[dict | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(prepare, record): index for index, record in enumerate(records)}
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            ordered[index] = future.result()
            completed += 1
            if progress is not None:
                progress(completed, len(records), records[index]["case_id"])
    return [item for item in ordered if item is not None]


def sample_multitask_patch(
    record: dict,
    rng: np.random.Generator,
    *,
    roi: int,
    foreground_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    image = np.load(record["image_path"], mmap_mode="r")
    label = np.load(record["label_path"], mmap_mode="r")
    sdf = np.load(record["sdf_path"], mmap_mode="r")
    if not 0 <= foreground_probability <= 1:
        raise ValueError("Foreground probability must lie in [0, 1]")
    if rng.random() < foreground_probability:
        target = int(rng.choice((1, 2)))
        coordinates = np.argwhere(label == target)
        center = coordinates[int(rng.integers(len(coordinates)))].tolist()
        mode = "foreground_brain" if target == 1 else "foreground_skull"
    else:
        candidates = (label == 0) & (image[1] > 0)
        if not np.any(candidates):
            candidates = label == 0
        coordinates = np.argwhere(candidates)
        center = coordinates[int(rng.integers(len(coordinates)))].tolist()
        mode = "ct_context_background"
    slices, padding = [], []
    for coordinate, dimension in zip(center, label.shape):
        desired_start = int(coordinate) - roi // 2
        desired_end = desired_start + roi
        start, end = max(0, desired_start), min(dimension, desired_end)
        slices.append(slice(start, end))
        padding.append((max(0, -desired_start), max(0, desired_end - dimension)))
    image_patch = np.pad(np.asarray(image[(slice(None), *slices)], dtype=np.float32), [(0, 0), *padding])
    label_patch = np.pad(np.asarray(label[tuple(slices)], dtype=np.int64), padding)
    sdf_patch = np.pad(
        np.asarray(sdf[(slice(None), *slices)], dtype=np.float32),
        [(0, 0), *padding],
        constant_values=-1.0,
    )
    for axis in range(3):
        if rng.random() < 0.5:
            image_patch = np.flip(image_patch, axis=axis + 1)
            label_patch = np.flip(label_patch, axis=axis)
            sdf_patch = np.flip(sdf_patch, axis=axis + 1)
    return (
        np.ascontiguousarray(image_patch),
        np.ascontiguousarray(label_patch),
        np.ascontiguousarray(sdf_patch),
        mode,
    )


def sample_anatomy_patch(
    record: dict,
    rng: np.random.Generator,
    *,
    roi: int,
    foreground_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Sample one spatially matched image/label/SDF/enclosure/core patch."""
    image = np.load(record["image_path"], mmap_mode="r")
    label = np.load(record["label_path"], mmap_mode="r")
    sdf = np.load(record["sdf_path"], mmap_mode="r")
    enclosure = np.load(record["enclosure_path"], mmap_mode="r")
    brain_core = np.load(record["brain_core_path"], mmap_mode="r")
    if rng.random() < foreground_probability:
        target = int(rng.choice((1, 2)))
        coordinates = np.argwhere(label == target)
        center = coordinates[int(rng.integers(len(coordinates)))].tolist()
        mode = "foreground_brain" if target == 1 else "foreground_skull"
    else:
        candidates = (label == 0) & (image[1] > 0)
        if not np.any(candidates):
            candidates = label == 0
        coordinates = np.argwhere(candidates)
        center = coordinates[int(rng.integers(len(coordinates)))].tolist()
        mode = "ct_context_background"
    slices, padding = [], []
    for coordinate, dimension in zip(center, label.shape):
        desired_start = int(coordinate) - roi // 2
        desired_end = desired_start + roi
        start, end = max(0, desired_start), min(dimension, desired_end)
        slices.append(slice(start, end))
        padding.append((max(0, -desired_start), max(0, desired_end - dimension)))
    image_patch = np.pad(np.asarray(image[(slice(None), *slices)], dtype=np.float32), [(0, 0), *padding])
    label_patch = np.pad(np.asarray(label[tuple(slices)], dtype=np.int64), padding)
    sdf_patch = np.pad(
        np.asarray(sdf[(slice(None), *slices)], dtype=np.float32),
        [(0, 0), *padding],
        constant_values=-1.0,
    )
    enclosure_patch = np.pad(np.asarray(enclosure[tuple(slices)], dtype=np.float32), padding)
    core_patch = np.pad(np.asarray(brain_core[tuple(slices)], dtype=np.float32), padding)
    for axis in range(3):
        if rng.random() < 0.5:
            image_patch = np.flip(image_patch, axis=axis + 1)
            label_patch = np.flip(label_patch, axis=axis)
            sdf_patch = np.flip(sdf_patch, axis=axis + 1)
            enclosure_patch = np.flip(enclosure_patch, axis=axis)
            core_patch = np.flip(core_patch, axis=axis)
    return tuple(
        np.ascontiguousarray(value)
        for value in (image_patch, label_patch, sdf_patch, enclosure_patch, core_patch)
    ) + (mode,)
