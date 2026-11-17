"""Patient-space surfaces: preserve labels and topology; attach, never threshold, confidence."""
from __future__ import annotations

import hashlib
import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes


def require_same_patient(reference: dict, confidence: dict) -> None:
    for key in ("case_id", "source_ct_sha256"):
        if not reference.get(key) or reference[key] != confidence.get(key):
            raise ValueError(f"Patient provenance mismatch: {key}")


def require_same_grid(reference, other) -> None:
    if reference.shape != other.shape or not np.allclose(reference.affine, other.affine, atol=1e-4, rtol=0):
        raise ValueError("Image grids differ (shape/affine); explicit physical resampling required")


def anatomical_surface(mask: np.ndarray, affine: np.ndarray) -> trimesh.Trimesh:
    """Native-resolution 0.5 surface. No smoothing, filtering, filling or FOV padding.

    A surface intersecting the image boundary remains open, rather than capping
    missing anatomy outside the acquired field of view.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any() or mask.all():
        raise ValueError("Expected nonempty, nonconstant 3D mask")
    bounds = [np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    lo = np.maximum([b[0] - 1 for b in bounds], 0)
    hi = np.minimum([b[-1] + 2 for b in bounds], mask.shape)
    crop = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
    vertices, faces, _, _ = marching_cubes(crop.astype(np.uint8), .5, allow_degenerate=False)
    vertices += lo
    world = nib.affines.apply_affine(affine, vertices).astype(np.float32)
    # Correct reflection-induced winding; no vertex reordering or repair.
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    mesh.fix_normals(multibody=True)
    return mesh


def sample_world(volume: np.ndarray, vertices: np.ndarray, affine: np.ndarray, order: int = 1) -> np.ndarray:
    if order not in (0, 1):
        raise ValueError("Only nearest/linear interpolation preserves bounded confidence")
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(affine).all():
        raise ValueError("Invalid confidence grid")
    voxel = nib.affines.apply_affine(np.linalg.inv(affine), vertices)
    valid = np.isfinite(voxel).all(axis=1) & (voxel >= -1e-5).all(axis=1) & (voxel <= np.array(values.shape)-1+1e-5).all(axis=1)
    result = np.full(len(vertices), np.nan, dtype=np.float32)
    # Clamp only floating-point roundoff at voxel centers, never extend the FOV.
    coords = np.clip(voxel[valid], 0, np.array(values.shape)-1)
    result[valid] = map_coordinates(values, coords.T, order=order, mode="constant", cval=np.nan, prefilter=False)
    return result


def confidence_colors(values: np.ndarray) -> np.ndarray:
    """Fixed red (0), yellow (.5), green (1), opaque grey (invalid)."""
    values = np.asarray(values)
    valid = np.isfinite(values) & (values >= 0) & (values <= 1)
    rgba = np.tile(np.array([150, 150, 150, 255], dtype=np.uint8), (len(values), 1))
    v = values[valid]
    rgba[valid, :3] = np.rint(255 * np.column_stack([np.minimum(1, 2*(1-v)), np.minimum(1, 2*v), np.zeros_like(v)])).astype(np.uint8)
    return rgba


def color_surface(mesh, volume, affine, *, reference_provenance, confidence_provenance):
    require_same_patient(reference_provenance, confidence_provenance)
    sampled = sample_world(volume, mesh.vertices, affine)
    sampled[(sampled < 0) | (sampled > 1)] = np.nan
    colored = mesh.copy()
    colored.visual.vertex_colors = confidence_colors(sampled)
    assert np.array_equal(colored.vertices, mesh.vertices)
    assert np.array_equal(colored.faces, mesh.faces)
    return colored, sampled


def geometry_digest(mesh) -> str:
    digest = hashlib.sha256()
    for array, dtype in ((mesh.vertices, '<f4'), (mesh.faces, '<i8')):
        digest.update(np.asarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()
