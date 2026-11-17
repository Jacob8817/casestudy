"""Phase 7 curvature-, uncertainty-, and anisotropy-aware mesh allocation."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def _normalize(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(source, (1, 99))
    return np.clip((source - low) / max(float(high - low), 1e-6), 0, 1)


def _normal_variation(mesh: trimesh.Trimesh) -> np.ndarray:
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    total = np.zeros(len(normals), dtype=np.float64)
    count = np.zeros(len(normals), dtype=np.int64)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        left, right = faces[:, a], faces[:, b]
        value = 1.0 - np.sum(normals[left] * normals[right], axis=1)
        np.add.at(total, left, value)
        np.add.at(total, right, value)
        np.add.at(count, left, 1)
        np.add.at(count, right, 1)
    return _normalize(total / np.maximum(count, 1))


def _sample_volume(values: np.ndarray, voxel_vertices: np.ndarray) -> np.ndarray:
    return map_coordinates(values, voxel_vertices.T, order=1, mode="nearest").astype(np.float32)


def _adaptive_cluster(
    vertices: np.ndarray,
    faces: np.ndarray,
    priority: np.ndarray,
    bins: tuple[float, float],
    cluster_mm: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categories = np.digitize(priority, bins).astype(np.uint8)
    keys = []
    for index, (vertex, category) in enumerate(zip(vertices, categories)):
        size = cluster_mm[int(category)]
        key = (int(category), index) if size <= 0 else (int(category), *np.rint(vertex / size).astype(np.int64))
        keys.append(key)
    groups: dict[tuple, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(tuple(key), []).append(index)
    new_vertices, new_priority = [], []
    mapping = np.empty(len(vertices), dtype=np.int64)
    for new_index, indices in enumerate(groups.values()):
        new_vertices.append(np.mean(vertices[indices], axis=0))
        new_priority.append(np.max(priority[indices]))
        mapping[indices] = new_index
    new_faces = mapping[faces]
    keep = (new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2]) & (new_faces[:, 0] != new_faces[:, 2])
    new_faces = new_faces[keep]
    _, unique_indices = np.unique(np.sort(new_faces, axis=1), axis=0, return_index=True)
    new_faces = new_faces[np.sort(unique_indices)]
    return np.asarray(new_vertices, dtype=np.float32), new_faces.astype(np.int64), np.asarray(new_priority, dtype=np.float32)


def _priority_colors(priority: np.ndarray) -> np.ndarray:
    values = np.clip(priority, 0, 1)
    return np.column_stack(
        (
            np.rint(255 * values),
            np.rint(180 * (1 - np.abs(values - 0.5) * 2)),
            np.rint(255 * (1 - values)),
            np.full(len(values), 255),
        )
    ).astype(np.uint8)


def build_adaptive_mesh(
    mask: np.ndarray,
    entropy: np.ndarray,
    priority_volume: np.ndarray,
    affine: np.ndarray,
    spacing_mm: tuple[float, float, float],
    config: dict,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    voxel_vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), 0.5, allow_degenerate=False)
    world_vertices = nib.affines.apply_affine(affine, voxel_vertices).astype(np.float32)
    # Preserve the marching-cubes vertex order so voxel-space samples remain aligned.
    baseline = trimesh.Trimesh(vertices=world_vertices, faces=faces, process=False, validate=False)
    curvature = _normal_variation(baseline)
    sampled_uncertainty = _sample_volume(entropy, voxel_vertices)
    boundary_signal = _sample_volume(priority_volume, voxel_vertices)
    anisotropy = float(np.clip((max(spacing_mm) / min(spacing_mm) - 1.0) / 2.0, 0, 1))
    weights = config["priority_weights"]
    vertex_priority = (
        float(weights["uncertainty"]) * sampled_uncertainty
        + float(weights["curvature"]) * curvature
        + float(weights["boundary_complexity"]) * boundary_signal
        + float(weights["anisotropy"]) * anisotropy
    ).astype(np.float32)
    bins = tuple(float(value) for value in config["priority_bins"])
    clustered_vertices, clustered_faces, clustered_priority = _adaptive_cluster(
        world_vertices,
        faces,
        vertex_priority,
        bins,
        (
            float(config["low_priority_cluster_mm"]),
            float(config["medium_priority_cluster_mm"]),
            float(config["high_priority_cluster_mm"]),
        ),
    )
    adaptive = trimesh.Trimesh(
        vertices=clustered_vertices,
        faces=clustered_faces,
        vertex_colors=_priority_colors(clustered_priority),
        process=True,
        validate=True,
    )
    forward = cKDTree(baseline.vertices).query(adaptive.vertices, workers=-1)[0]
    backward = cKDTree(adaptive.vertices).query(baseline.vertices, workers=-1)[0]
    metrics = {
        "baseline_vertices": int(len(baseline.vertices)),
        "baseline_faces": int(len(baseline.faces)),
        "adaptive_vertices": int(len(adaptive.vertices)),
        "adaptive_faces": int(len(adaptive.faces)),
        "vertex_reduction_fraction": float(1 - len(adaptive.vertices) / len(baseline.vertices)),
        "face_reduction_fraction": float(1 - len(adaptive.faces) / len(baseline.faces)),
        "adaptive_to_full_assd_mm": float((np.mean(forward) + np.mean(backward)) / 2),
        "adaptive_to_full_hd95_mm": float(np.percentile(np.concatenate((forward, backward)), 95)),
        "anisotropy_ratio": float(max(spacing_mm) / min(spacing_mm)),
        "anisotropy_score": anisotropy,
        "mean_vertex_priority": float(np.mean(vertex_priority)),
        "high_priority_vertex_fraction": float(np.mean(vertex_priority >= bins[1])),
    }
    return baseline, adaptive, metrics


def run_adaptive_meshes(
    records: list[dict],
    phase4_root: Path,
    phase5_root: Path,
    config: dict,
    output_root: Path,
    *,
    progress=print,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    cases = []
    for record in records:
        case_id = record["case_id"]
        refined = np.asarray(nib.load(str(phase5_root / "cases" / case_id / "refined_prediction.nii.gz")).dataobj, dtype=np.uint8)
        entropy = np.asarray(nib.load(str(phase4_root / "cases" / case_id / "predictive_entropy.nii.gz")).dataobj, dtype=np.float32)
        priority_volume = np.asarray(nib.load(str(phase5_root / "cases" / case_id / "rescue_priority.nii.gz")).dataobj, dtype=np.float32)
        affine = np.asarray(record["ct_affine"], dtype=np.float64)
        spacing = tuple(float(value) for value in record["spacing_mm"])
        organs = {}
        for class_id, organ in ((1, "brain"), (2, "skull")):
            baseline, adaptive, metrics = build_adaptive_mesh(
                refined == class_id, entropy, priority_volume, affine, spacing, config
            )
            organ_root = output_root / "cases" / case_id / organ
            organ_root.mkdir(parents=True, exist_ok=True)
            baseline.export(organ_root / "baseline_full_resolution.ply")
            adaptive.export(organ_root / "adaptive_priority.ply")
            adaptive.export(organ_root / "adaptive_priority.stl")
            organs[organ] = metrics
        cases.append({"case_id": case_id, "organs": organs})
        progress(f"Phase 7 adaptive meshes: {case_id}")
    report = {
        "schema_version": 1,
        "phase": 7,
        "status": "passed",
        "method": "priority-binned spatial vertex clustering",
        "priority_formula": config["priority_weights"],
        "cases": cases,
        "official_test_split_accessed": False,
    }
    (output_root / "phase7_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
