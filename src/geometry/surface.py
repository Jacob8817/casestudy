"""Phase 6 voxel-boundary and reconstructed-surface evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def _surface(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    return source ^ binary_erosion(source, border_value=0)


def binary_surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    tolerances_mm: tuple[float, ...] = (1.0, 2.0),
) -> dict:
    p, t = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if p.shape != t.shape or p.ndim != 3 or not np.any(p) or not np.any(t):
        raise ValueError("Surface metrics require aligned non-empty 3D masks")
    intersection = int(np.count_nonzero(p & t))
    p_count, t_count = int(np.count_nonzero(p)), int(np.count_nonzero(t))
    p_surface, t_surface = _surface(p), _surface(t)
    scale = np.asarray(spacing_mm, dtype=np.float32)
    predicted_points = np.argwhere(p_surface).astype(np.float32) * scale
    target_points = np.argwhere(t_surface).astype(np.float32) * scale
    maximum = 200_000
    if len(predicted_points) > maximum:
        predicted_points = predicted_points[np.linspace(0, len(predicted_points) - 1, maximum, dtype=np.int64)]
    if len(target_points) > maximum:
        target_points = target_points[np.linspace(0, len(target_points) - 1, maximum, dtype=np.int64)]
    to_target = cKDTree(target_points).query(predicted_points, workers=-1)[0]
    to_prediction = cKDTree(predicted_points).query(target_points, workers=-1)[0]
    distances = np.concatenate((to_target, to_prediction))
    result = {
        "dice": float(2 * intersection / (p_count + t_count)),
        "iou": float(intersection / np.count_nonzero(p | t)),
        "precision": float(intersection / p_count),
        "recall": float(intersection / t_count),
        "assd_mm": float((np.mean(to_target) + np.mean(to_prediction)) / 2.0),
        "hd95_mm": float(np.percentile(distances, 95)),
        "hausdorff_mm": float(np.max(distances)),
        "surface_rmse_mm": float(np.sqrt(np.mean(np.square(distances)))),
        "volume_error_percent": float(100.0 * (p_count - t_count) / t_count),
    }
    for tolerance in tolerances_mm:
        result[f"surface_dice_{tolerance:g}mm"] = float(
            (np.count_nonzero(to_target <= tolerance) + np.count_nonzero(to_prediction <= tolerance))
            / (to_target.size + to_prediction.size)
        )
    return result


def _mesh(mask: np.ndarray, affine: np.ndarray) -> trimesh.Trimesh:
    vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), 0.5, allow_degenerate=False)
    world = nib.affines.apply_affine(affine, vertices)
    return trimesh.Trimesh(vertices=world, faces=faces, process=True, validate=True)


def _deterministic_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    if len(mesh.vertices) <= count:
        return np.asarray(mesh.vertices)
    rng = np.random.default_rng(seed)
    return np.asarray(mesh.vertices)[np.sort(rng.choice(len(mesh.vertices), count, replace=False))]


def mesh_reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    affine: np.ndarray,
    *,
    sample_points: int,
    seed: int,
) -> dict:
    predicted, reference = _mesh(prediction, affine), _mesh(target, affine)
    p = _deterministic_points(predicted, sample_points, seed)
    t = _deterministic_points(reference, sample_points, seed + 1)
    p_to_t = cKDTree(t).query(p, workers=-1)[0]
    t_to_p = cKDTree(p).query(t, workers=-1)[0]
    return {
        "symmetric_chamfer_mm": float((np.mean(p_to_t) + np.mean(t_to_p)) / 2.0),
        "mesh_hd95_mm": float(np.percentile(np.concatenate((p_to_t, t_to_p)), 95)),
        "mesh_surface_rmse_mm": float(np.sqrt(np.mean(np.square(np.concatenate((p_to_t, t_to_p)))))),
        "predicted_surface_area_mm2": float(predicted.area),
        "reference_surface_area_mm2": float(reference.area),
        "surface_area_error_percent": float(100.0 * (predicted.area - reference.area) / reference.area),
        "predicted_vertices": int(len(predicted.vertices)),
        "predicted_faces": int(len(predicted.faces)),
        "reference_vertices": int(len(reference.vertices)),
        "reference_faces": int(len(reference.faces)),
    }


def run_surface_evaluation(
    records: list[dict],
    phase4_root: Path,
    phase5_root: Path,
    config: dict,
    output_root: Path,
    *,
    progress=print,
) -> dict:
    if any(item["split"] != "val" for item in records):
        raise ValueError("Phase 6 surface evaluation accepts validation records only")
    output_root.mkdir(parents=True, exist_ok=True)
    tolerances = tuple(float(value) for value in config["surface_dice_tolerances_mm"])
    cases = []
    for case_index, record in enumerate(records):
        case_id = record["case_id"]
        target = np.load(record["label_path"], mmap_mode="r")
        initial = np.asarray(nib.load(str(phase4_root / "cases" / case_id / "initial_prediction.nii.gz")).dataobj, dtype=np.uint8)
        refined = np.asarray(nib.load(str(phase5_root / "cases" / case_id / "refined_prediction.nii.gz")).dataobj, dtype=np.uint8)
        spacing = tuple(float(value) for value in record["spacing_mm"])
        affine = np.asarray(record["ct_affine"], dtype=np.float64)
        organs = {}
        for class_id, organ in ((1, "brain"), (2, "skull")):
            organ_result = {}
            for name, prediction in (("initial", initial), ("refined", refined)):
                if not np.any(prediction == class_id) or not np.any(target == class_id):
                    organ_result[name] = None
                    continue
                voxel_metrics = binary_surface_metrics(
                    prediction == class_id, target == class_id, spacing, tolerances_mm=tolerances
                )
                mesh_metrics = mesh_reconstruction_metrics(
                    prediction == class_id,
                    target == class_id,
                    affine,
                    sample_points=int(config["mesh_sample_points"]),
                    seed=int(config["random_seed"]) + case_index * 10 + class_id,
                )
                organ_result[name] = {**voxel_metrics, **mesh_metrics}
            organs[organ] = organ_result
        cases.append({"case_id": case_id, "organs": organs})
        progress(f"Phase 6 surfaces: {case_id}")
    aggregate = {}
    metric_names = sorted(
        next(item for item in cases[0]["organs"].values())["initial"].keys()
    )
    for organ in ("brain", "skull"):
        aggregate[organ] = {}
        for stage in ("initial", "refined"):
            aggregate[organ][stage] = {
                metric: float(np.mean([case["organs"][organ][stage][metric] for case in cases]))
                for metric in metric_names
            }
    report = {
        "schema_version": 1,
        "phase": 6,
        "status": "passed",
        "surface_dice_tolerances_mm": list(tolerances),
        "reconstruction_optimization": config["reconstruction_optimization"],
        "aggregate": aggregate,
        "cases": cases,
        "official_test_split_accessed": False,
    }
    (output_root / "phase6_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
