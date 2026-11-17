"""Phase 5 selective local flip-TTA rescue using the frozen Phase 3 model."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, label as connected_components, find_objects
from scipy.spatial import cKDTree

from src.benchmark.brain_skull import dice_per_class
from src.evaluation.metrics import select_error_threshold


def _boundary_complexity(prediction: np.ndarray) -> np.ndarray:
    foreground = prediction > 0
    boundary = foreground ^ binary_erosion(foreground)
    for class_id in (1, 2):
        mask = prediction == class_id
        boundary |= mask ^ binary_erosion(mask)
    expanded = binary_dilation(boundary, iterations=2).astype(np.float32)
    return np.clip(gaussian_filter(expanded, sigma=1.0), 0.0, 1.0)


def build_rescue_priority(
    entropy: np.ndarray,
    violation: np.ndarray,
    prediction: np.ndarray,
    weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    required = {"uncertainty", "anatomical_violation", "boundary_complexity"}
    if set(weights) != required or any(not np.isfinite(v) or v < 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Rescue priority weights must cover U, V, B and sum to one")
    if entropy.shape != violation.shape or entropy.shape != prediction.shape:
        raise ValueError("Rescue priority inputs must share geometry")
    if not np.isfinite(entropy).all() or not np.isfinite(violation).all():
        raise ValueError("Rescue priority inputs must be finite")
    boundary = _boundary_complexity(prediction)
    priority = (
        float(weights["uncertainty"]) * np.clip(entropy, 0, 1)
        + float(weights["anatomical_violation"]) * np.clip(violation, 0, 1)
        + float(weights["boundary_complexity"]) * boundary
    )
    return np.clip(priority, 0, 1).astype(np.float32), boundary


def extract_priority_rois(
    mask: np.ndarray,
    priority: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    roi_size: tuple[int, int, int],
    minimum_component_ml: float,
    maximum_rois: int,
) -> list[tuple[slice, slice, slice]]:
    components, count = connected_components(mask)
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    minimum_voxels = max(1, int(np.ceil(minimum_component_ml / voxel_ml)))
    counts = np.bincount(components.ravel())
    boxes = find_objects(components)
    candidates = []
    for component in range(1, count + 1):
        if counts[component] < minimum_voxels:
            continue
        box = boxes[component - 1]
        local = components[box] == component
        coordinates = np.argwhere(local) + np.array([s.start for s in box])
        scores = priority[box][local]
        center = np.rint(np.average(coordinates, axis=0, weights=np.maximum(scores, 1e-6))).astype(int)
        candidates.append((float(np.sum(scores)), center))
    candidates.sort(key=lambda item: item[0], reverse=True)
    rois = []
    for _, center in candidates[:maximum_rois]:
        slices = []
        for axis, size in enumerate(roi_size):
            start = int(center[axis]) - size // 2
            start = min(max(0, start), max(0, mask.shape[axis] - size))
            stop = min(mask.shape[axis], start + size)
            slices.append(slice(start, stop))
        rois.append(tuple(slices))
    return rois


def _pad_patch(image: np.ndarray, slices: tuple[slice, slice, slice], roi_size: tuple[int, int, int]):
    patch = np.asarray(image[(slice(None), *slices)], dtype=np.float32)
    padding = [(0, 0)] + [(0, roi_size[i] - patch.shape[i + 1]) for i in range(3)]
    return np.pad(patch, padding), tuple(slice(0, patch.shape[i + 1]) for i in range(3))


def _local_tta_prediction(
    model: torch.nn.Module,
    image_patch: np.ndarray,
    *,
    device: torch.device,
    axes: tuple[int, ...],
) -> np.ndarray:
    if len(set(axes)) != len(axes) or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("TTA axes must be unique spatial axes 0, 1, 2")
    tensor = torch.from_numpy(image_patch[None]).to(device)
    logits = []
    variants: list[int | None] = [None, *axes]
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for axis in variants:
            spatial_dim = None if axis is None else axis + 2
            values = tensor if spatial_dim is None else torch.flip(tensor, dims=(spatial_dim,))
            output, _ = model(values)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite local TTA logits")
            if spatial_dim is not None:
                output = torch.flip(output, dims=(spatial_dim,))
            logits.append(output.float())
    mean_logits = torch.mean(torch.stack(logits), dim=0)
    prediction = mean_logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
    del tensor, logits, mean_logits
    return prediction


def _sample_indices(size: int, count: int, seed: int, case_id: str) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{case_id}:rescue".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.choice(size, min(size, count), replace=False))


def _hd95_points(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Surface-point HD95 with deterministic caps for very large false-positive volumes."""
    p, t = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if not np.any(p) or not np.any(t):
        return float("nan")
    p_surface = p ^ binary_erosion(p, border_value=0)
    t_surface = t ^ binary_erosion(t, border_value=0)
    scale = np.asarray(spacing, dtype=np.float32)
    p_points = np.argwhere(p_surface).astype(np.float32) * scale
    t_points = np.argwhere(t_surface).astype(np.float32) * scale
    maximum = 200_000
    if len(p_points) > maximum:
        p_points = p_points[np.linspace(0, len(p_points) - 1, maximum, dtype=np.int64)]
    if len(t_points) > maximum:
        t_points = t_points[np.linspace(0, len(t_points) - 1, maximum, dtype=np.int64)]
    forward = cKDTree(t_points).query(p_points, workers=-1)[0]
    backward = cKDTree(p_points).query(t_points, workers=-1)[0]
    return float(np.percentile(np.concatenate((forward, backward)), 95))


def run_validation_rescue(
    model: torch.nn.Module,
    records: list[dict],
    phase4_report: dict,
    config: dict,
    phase3_qc_root: Path,
    phase4_root: Path,
    output_root: Path,
    *,
    device: torch.device,
    progress=print,
    frozen_priority_threshold: dict | None = None,
) -> dict:
    """Select Q threshold on validation, refine only selected ROIs, and report actual change."""
    if not records or any(item["split"] != "val" for item in records):
        raise ValueError("Phase 5 accepts validation records only")
    output_root.mkdir(parents=True, exist_ok=True)
    priorities, samples, errors = {}, [], []
    seed = int(config["random_seed"])
    sample_count = int(config["threshold_samples_per_case"])
    weights = {key: float(value) for key, value in config["priority_weights"].items()}
    for record in records:
        case_id = record["case_id"]
        images = [nib.load(str(path)) for path in (
            phase4_root / "cases" / case_id / "initial_prediction.nii.gz",
            phase4_root / "cases" / case_id / "predictive_entropy.nii.gz",
            phase3_qc_root / case_id / "anatomical_violation.nii.gz")]
        if any(tuple(record["shape"]) != im.shape or not np.allclose(im.affine, record["ct_affine"], atol=1e-4, rtol=0) for im in images):
            raise ValueError(f"Priority grid mismatch for {case_id}")
        prediction = np.asarray(images[0].dataobj, dtype=np.uint8)
        entropy = np.asarray(images[1].dataobj, dtype=np.float32)
        violation = np.asarray(images[2].dataobj, dtype=np.float32)
        target = np.load(record["label_path"], mmap_mode="r")
        priority, boundary = build_rescue_priority(entropy, violation, prediction, weights)
        priorities[case_id] = (priority, boundary, prediction)
        index = _sample_indices(target.size, sample_count, seed, case_id)
        samples.append(priority.reshape(-1)[index])
        errors.append((prediction != target).reshape(-1)[index].astype(np.uint8))
    threshold = (
        select_error_threshold(np.concatenate(samples), np.concatenate(errors))
        if frozen_priority_threshold is None else dict(frozen_priority_threshold)
    )
    selected_threshold = float(threshold["threshold"])
    if not np.isfinite(selected_threshold) or not 0 <= selected_threshold <= 1:
        raise ValueError("Invalid frozen priority threshold")

    case_reports = []
    for record in records:
        started = time.perf_counter()
        case_id = record["case_id"]
        priority, boundary, prediction = priorities[case_id]
        target = np.load(record["label_path"], mmap_mode="r")
        image = np.load(record["image_path"], mmap_mode="r")
        spacing = tuple(float(value) for value in record["spacing_mm"])
        rescue_mask = priority >= selected_threshold
        roi_size = tuple(int(value) for value in config["roi_size"])
        rois = extract_priority_rois(
            rescue_mask,
            priority,
            spacing,
            roi_size=roi_size,
            minimum_component_ml=float(config["minimum_component_ml"]),
            maximum_rois=int(config["maximum_rois_per_case"]),
        )
        refined = prediction.copy()
        selected_union = np.zeros(prediction.shape, dtype=bool)
        roi_details = []
        for slices in rois:
            patch, valid = _pad_patch(image, slices, roi_size)
            local = _local_tta_prediction(
                model,
                patch,
                device=device,
                axes=tuple(int(axis) for axis in config["tta_spatial_axes"]),
            )[valid]
            selected = rescue_mask[slices]
            selected_union[slices] |= selected
            roi_details.append({
                "slices_start_stop": [[s.start, s.stop] for s in slices],
                "padded_inference_shape": list(patch.shape[1:]),
                "selected_voxels": int(np.count_nonzero(selected)),
                "origin_ras_mm": nib.affines.apply_affine(np.asarray(record["ct_affine"]), [s.start for s in slices]).tolist(),
            })
            region = refined[slices]
            region[selected] = local[selected]
            refined[slices] = region
        changed = refined != prediction
        assert not np.any(changed & ~selected_union), "Refinement changed unselected voxels"
        assert len(rois) <= int(config["maximum_rois_per_case"])
        before_dice, after_dice = dice_per_class(prediction, target), dice_per_class(refined, target)
        before_hd95 = {organ: _hd95_points(prediction == cid, target == cid, spacing) for cid, organ in ((1, "brain"), (2, "skull"))}
        after_hd95 = {organ: _hd95_points(refined == cid, target == cid, spacing) for cid, organ in ((1, "brain"), (2, "skull"))}
        case_root = output_root / "cases" / case_id
        affine = np.asarray(record["ct_affine"], dtype=np.float64)
        case_root.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(priority, affine), str(case_root / "rescue_priority.nii.gz"))
        nib.save(nib.Nifti1Image(rescue_mask.astype(np.uint8), affine), str(case_root / "rescue_mask.nii.gz"))
        nib.save(nib.Nifti1Image(refined, affine), str(case_root / "refined_prediction.nii.gz"))
        nib.save(nib.Nifti1Image(boundary, affine), str(case_root / "boundary_signal.nii.gz"))
        nib.save(nib.Nifti1Image(selected_union.astype(np.uint8), affine), str(case_root / "selected_replacement_mask.nii.gz"))
        nib.save(nib.Nifti1Image(changed.astype(np.uint8), affine), str(case_root / "changed_voxels.nii.gz"))
        case_reports.append(
            {
                "case_id": case_id,
                "rescue_voxel_fraction": float(np.mean(rescue_mask)),
                "rois": len(rois),
                "roi_locations": roi_details,
                "priority_candidate_voxels": int(np.count_nonzero(rescue_mask)),
                "selected_replacement_voxels": int(np.count_nonzero(selected_union)),
                "changed_voxels": int(np.count_nonzero(changed)),
                "changed_outside_selection": int(np.count_nonzero(changed & ~selected_union)),
                "before_dice": before_dice,
                "after_dice": after_dice,
                "before_hd95_mm": before_hd95,
                "after_hd95_mm": after_hd95,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        progress(f"Phase 5 rescue: {case_id}, {len(rois)} ROI(s)")
    means = {}
    for organ in ("brain", "skull"):
        means[organ] = {
            "before_dice": float(np.mean([item["before_dice"][organ] for item in case_reports])),
            "after_dice": float(np.mean([item["after_dice"][organ] for item in case_reports])),
            "before_hd95_mm": float(np.mean([item["before_hd95_mm"][organ] for item in case_reports])),
            "after_hd95_mm": float(np.mean([item["after_hd95_mm"][organ] for item in case_reports])),
        }
        means[organ]["dice_delta"] = means[organ]["after_dice"] - means[organ]["before_dice"]
        means[organ]["hd95_delta_mm"] = means[organ]["after_hd95_mm"] - means[organ]["before_hd95_mm"]
    report = {
        "schema_version": 1,
        "phase": 5,
        "status": "passed",
        "priority_formula": weights,
        "selected_priority_threshold": threshold,
        "priority_threshold_reselected": frozen_priority_threshold is None,
        "tta_aggregation": "mean inverse-transformed logits; identity plus each configured single-axis flip",
        "overlapping_roi_policy": "deterministic ROI order; later ROI wins at selected overlapping voxels",
        "refinement": config["refinement"],
        "prediction_was_modified_only_inside_selected_rois": True,
        "mean_rescue_voxel_fraction": float(np.mean([item["rescue_voxel_fraction"] for item in case_reports])),
        "total_rois": int(sum(item["rois"] for item in case_reports)),
        "total_rescue_seconds": float(sum(item["elapsed_seconds"] for item in case_reports)),
        "means": means,
        "cases": case_reports,
        "official_test_split_accessed": False,
    }
    (output_root / "phase5_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
