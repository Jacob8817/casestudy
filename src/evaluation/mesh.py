"""Patient-specific surface reconstruction and portable 3D exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


ORGAN_COLORS: dict[int, tuple[int, int, int, int]] = {
    1: (194, 66, 66, 230),
    2: (135, 75, 170, 230),
    3: (56, 126, 184, 230),
    4: (39, 159, 140, 230),
    5: (240, 145, 63, 230),
    6: (231, 196, 78, 230),
    7: (191, 40, 60, 230),
    8: (75, 105, 190, 230),
    9: (112, 188, 210, 180),
    10: (88, 166, 200, 180),
}


def _safe_name(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "_-" else "_" for character in value)
    if not result:
        raise ValueError("Mesh name is empty after sanitization")
    return result


def _world_vertices(voxel_vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((voxel_vertices, np.ones(voxel_vertices.shape[0], dtype=np.float64)))
    return (homogeneous @ affine.T)[:, :3].astype(np.float32)


def _uncertainty_vertex_colors(
    voxel_vertices: np.ndarray,
    uncertainty: np.ndarray,
    base: tuple[int, int, int, int],
) -> np.ndarray:
    indices = np.rint(voxel_vertices).astype(np.int64)
    for axis in range(3):
        indices[:, axis] = np.clip(indices[:, axis], 0, uncertainty.shape[axis] - 1)
    values = np.clip(uncertainty[tuple(indices.T)], 0.0, 1.0)[:, None]
    base_rgb = np.asarray(base[:3], dtype=np.float32)[None, :]
    warning_rgb = np.asarray((255, 245, 50), dtype=np.float32)[None, :]
    rgb = np.rint(base_rgb * (1.0 - values) + warning_rgb * values).astype(np.uint8)
    alpha = np.full((rgb.shape[0], 1), base[3], dtype=np.uint8)
    return np.concatenate((rgb, alpha), axis=1)


def build_patient_meshes(
    *,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    affine: np.ndarray,
    class_names: dict[int, str],
    output_root: Path,
    case_id: str,
    max_html_faces_per_organ: int = 25_000,
) -> dict[str, Any]:
    """Create full meshes (PLY/STL/GLB), an interactive HTML viewer, and PNG preview."""
    labels = np.asarray(prediction, dtype=np.uint8)
    entropy = np.asarray(uncertainty, dtype=np.float32)
    transform = np.asarray(affine, dtype=np.float64)
    if labels.shape != entropy.shape or labels.ndim != 3 or transform.shape != (4, 4):
        raise ValueError("Mesh inputs must share one valid 3D geometry")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    individual_root = output_root / "individual_organs"
    individual_root.mkdir(exist_ok=True)
    scene = trimesh.Scene()
    plotly_traces: list[go.Mesh3d] = []
    mesh_records: list[dict[str, Any]] = []
    preview_geometry: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] = []

    for class_id, organ in sorted(class_names.items()):
        mask = labels == class_id
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            mesh_records.append({"class_id": class_id, "organ": organ, "status": "absent"})
            continue
        voxel_vertices, faces, _normals, _values = marching_cubes(
            mask.astype(np.uint8), level=0.5, step_size=1, allow_degenerate=False
        )
        vertices = _world_vertices(voxel_vertices, transform)
        color = ORGAN_COLORS.get(class_id, (180, 180, 180, 220))
        vertex_colors = _uncertainty_vertex_colors(voxel_vertices, entropy, color)
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=True,
            validate=True,
        )
        mesh.remove_unreferenced_vertices()
        safe = _safe_name(organ)
        ply_path = individual_root / f"{safe}.ply"
        stl_path = individual_root / f"{safe}.stl"
        mesh.export(ply_path)
        mesh.export(stl_path)
        scene.add_geometry(mesh.copy(), node_name=safe, geom_name=safe)

        face_indices = np.arange(len(mesh.faces))
        if face_indices.size > max_html_faces_per_organ:
            face_indices = np.linspace(
                0, face_indices.size - 1, max_html_faces_per_organ, dtype=np.int64
            )
        shown_faces = mesh.faces[face_indices]
        hex_color = "#%02x%02x%02x" % color[:3]
        plotly_traces.append(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=shown_faces[:, 0],
                j=shown_faces[:, 1],
                k=shown_faces[:, 2],
                name=organ,
                color=hex_color,
                opacity=color[3] / 255.0,
                flatshading=False,
                hovertemplate=f"{organ}<extra></extra>",
            )
        )
        preview_geometry.append((mesh.vertices, shown_faces[:5000], color))
        mesh_records.append(
            {
                "class_id": class_id,
                "organ": organ,
                "status": "exported",
                "input_voxels": voxel_count,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "surface_area_mm2": float(mesh.area),
                "volume_ml_if_watertight": float(abs(mesh.volume) / 1000.0) if mesh.is_watertight else None,
                "ply": ply_path.relative_to(output_root).as_posix(),
                "stl": stl_path.relative_to(output_root).as_posix(),
            }
        )

    if not scene.geometry:
        raise ValueError("No foreground prediction exists; no mesh can be generated")
    glb_path = output_root / f"{_safe_name(case_id)}_digital_twin.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))

    figure = go.Figure(data=plotly_traces)
    figure.update_layout(
        title=f"Patient-specific multi-organ digital twin — {case_id}",
        template="plotly_dark",
        scene={
            "aspectmode": "data",
            "xaxis_title": "R–L (mm)",
            "yaxis_title": "A–P (mm)",
            "zaxis_title": "I–S (mm)",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    html_path = output_root / f"{_safe_name(case_id)}_digital_twin.html"
    figure.write_html(html_path, include_plotlyjs=True, full_html=True)

    preview_path = output_root / f"{_safe_name(case_id)}_digital_twin_preview.png"
    fig = plt.figure(figsize=(10, 10), dpi=160)
    axis = fig.add_subplot(111, projection="3d")
    for vertices, faces, color in preview_geometry:
        collection = Poly3DCollection(vertices[faces], linewidths=0, alpha=color[3] / 255.0)
        collection.set_facecolor(np.asarray(color[:3], dtype=float) / 255.0)
        axis.add_collection3d(collection)
    all_vertices = np.concatenate([item[0] for item in preview_geometry], axis=0)
    mins, maxs = all_vertices.min(axis=0), all_vertices.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)
    axis.set_xlabel("R–L (mm)")
    axis.set_ylabel("A–P (mm)")
    axis.set_zlabel("I–S (mm)")
    axis.set_title(f"Multi-organ digital twin — {case_id}")
    axis.view_init(elev=18, azim=-58)
    fig.tight_layout()
    fig.savefig(preview_path, bbox_inches="tight")
    plt.close(fig)

    report = {
        "schema_version": 1,
        "case_id": case_id,
        "coordinate_system": "RAS patient-space millimetres",
        "uncertainty_vertex_coloring": "base organ color blended toward yellow by calibrated normalized entropy",
        "full_resolution_exports": True,
        "interactive_html_face_cap_per_organ": max_html_faces_per_organ,
        "combined_glb": glb_path.name,
        "interactive_html": html_path.name,
        "preview_png": preview_path.name,
        "organs": mesh_records,
    }
    report_path = output_root / "mesh_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
