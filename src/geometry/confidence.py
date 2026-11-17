"""Phases 8 and 9 confidence-colored meshes and model-derived envelopes."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import trimesh
from .patient_surface import sample_world, confidence_colors, geometry_digest


def _sample_world(volume: np.ndarray, vertices: np.ndarray, affine: np.ndarray, order: int) -> np.ndarray:
    return sample_world(volume, vertices, affine, order)


def _colors(confidence: np.ndarray, colormap: str) -> np.ndarray:
    # Keep the legacy argument for notebook compatibility; the legend is fixed.
    return confidence_colors(confidence)


def run_surface_confidence(
    records: list[dict],
    phase4_root: Path,
    phase7_root: Path,
    config: dict,
    output_root: Path,
    *,
    representative_case: str,
    progress=print,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    cases, representative_meshes = [], []
    for record in records:
        case_id = record["case_id"]
        entropy_image = nib.load(str(phase4_root / "cases" / case_id / "predictive_entropy.nii.gz"))
        entropy = np.asarray(entropy_image.dataobj, dtype=np.float32)
        affine = np.asarray(record["ct_affine"], dtype=np.float64)
        if not np.allclose(entropy_image.affine, affine, atol=1e-4, rtol=0) or tuple(record["shape"]) != entropy.shape:
            raise ValueError(f"Entropy grid disagrees with patient record: {case_id}")
        organs = {}
        for organ in ("brain", "skull"):
            source = phase7_root / "cases" / case_id / organ / "baseline_full_resolution.ply"
            mesh = trimesh.load_mesh(source, process=False)
            geometry_before = geometry_digest(mesh)
            uncertainty = _sample_world(entropy, mesh.vertices, affine, int(config["interpolation_order"]))
            uncertainty[(uncertainty < 0) | (uncertainty > 1)] = np.nan
            confidence = 1.0 - uncertainty
            mesh.visual.vertex_colors = _colors(confidence, str(config["colormap"]))
            assert geometry_before == geometry_digest(mesh)
            organ_root = output_root / "cases" / case_id / organ
            organ_root.mkdir(parents=True, exist_ok=True)
            mesh.export(organ_root / "confidence_colored.ply")
            np.savez_compressed(
                organ_root / "vertex_confidence.npz",
                uncertainty=uncertainty.astype(np.float32),
                confidence=confidence.astype(np.float32),
            )
            organs[organ] = {
                "vertices": int(len(mesh.vertices)),
                "geometry_source": str(source),
                "geometry_sha256": geometry_before,
                "invalid_confidence_fraction": float(np.mean(~np.isfinite(confidence))),
                "mean_uncertainty": float(np.nanmean(uncertainty)) if np.isfinite(uncertainty).any() else None,
                "mean_confidence": float(np.nanmean(confidence)) if np.isfinite(confidence).any() else None,
                "low_confidence_vertex_fraction": float(np.mean(confidence < 0.5)),
            }
            if case_id == representative_case:
                representative_meshes.append((organ, mesh.copy(), confidence))
        cases.append({"case_id": case_id, "organs": organs})
        progress(f"Phase 8 confidence: {case_id}")
    scene = trimesh.Scene({organ: mesh for organ, mesh, _ in representative_meshes})
    glb_path = output_root / f"{representative_case}_confidence.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))
    traces = []
    for organ, mesh, confidence in representative_meshes:
        traces.append(
            go.Mesh3d(
                x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                vertexcolor=mesh.visual.vertex_colors, colorscale=[[0,"red"],[.5,"yellow"],[1,"green"]], cmin=0, cmax=1,
                name=organ, opacity=1, colorbar={"title": "1 - H/log(3)"},
            )
        )
    traces.append(go.Scatter3d(x=[None, None], y=[None, None], z=[None, None], mode="markers", marker=dict(color=[0,1], cmin=0, cmax=1, colorscale=[[0,"red"],[.5,"yellow"],[1,"green"]], showscale=True, colorbar=dict(title="1 - H/log(3)")), showlegend=False))
    figure = go.Figure(traces)
    figure.update_layout(title=f"Predictive certainty (not skull probability) — {representative_case}; grey = unavailable", scene_aspectmode="data")
    html_path = output_root / f"{representative_case}_confidence.html"
    figure.write_html(html_path, include_plotlyjs=True)
    report = {
        "schema_version": 1,
        "phase": 8,
        "status": "passed",
        "confidence_definition": config["confidence_definition"],
        "representative_case": representative_case,
        "interactive_html": str(html_path),
        "combined_glb": str(glb_path),
        "cases": cases,
        "official_test_split_accessed": False,
    }
    (output_root / "phase8_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_confidence_envelopes(
    records: list[dict],
    phase8_root: Path,
    config: dict,
    output_root: Path,
    *,
    representative_case: str,
    progress=print,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    cases, scene_geometry = [], {}
    maximum = float(config["maximum_offset_mm"])
    for record in records:
        case_id = record["case_id"]
        organs = {}
        for organ in ("brain", "skull"):
            organ_source = phase8_root / "cases" / case_id / organ
            mesh = trimesh.load_mesh(organ_source / "confidence_colored.ply", process=False)
            with np.load(organ_source / "vertex_confidence.npz") as values:
                uncertainty = values["uncertainty"].astype(np.float32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            # Unavailable samples cannot define an envelope displacement.
            valid = np.isfinite(uncertainty) & (uncertainty >= 0) & (uncertainty <= 1)
            offset = maximum * np.where(valid, uncertainty, 0)
            inner = trimesh.Trimesh(vertices=mesh.vertices - normals * offset[:, None], faces=mesh.faces, process=False)
            outer = trimesh.Trimesh(vertices=mesh.vertices + normals * offset[:, None], faces=mesh.faces, process=False)
            organ_root = output_root / "cases" / case_id / organ
            organ_root.mkdir(parents=True, exist_ok=True)
            inner.export(organ_root / "inner_envelope.ply")
            outer.export(organ_root / "outer_envelope.ply")
            organs[organ] = {
                "mean_offset_mm": float(np.mean(offset)),
                "maximum_offset_mm": float(np.max(offset)),
                "vertices": int(len(mesh.vertices)),
                "unavailable_offset_vertex_fraction": float(np.mean(~valid)),
            }
            if case_id == representative_case:
                scene_geometry[f"{organ}_prediction"] = mesh
                scene_geometry[f"{organ}_inner"] = inner
                scene_geometry[f"{organ}_outer"] = outer
        cases.append({"case_id": case_id, "organs": organs})
        progress(f"Phase 9 envelopes: {case_id}")
    scene = trimesh.Scene(scene_geometry)
    glb_path = output_root / f"{representative_case}_confidence_envelope.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))
    report = {
        "schema_version": 1,
        "phase": 9,
        "status": "passed",
        "interpretation": config["interpretation"],
        "maximum_offset_mm": maximum,
        "representative_case": representative_case,
        "combined_glb": str(glb_path),
        "cases": cases,
        "official_test_split_accessed": False,
    }
    (output_root / "phase9_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
