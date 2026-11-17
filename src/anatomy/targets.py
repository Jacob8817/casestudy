"""Reference-derived training targets for non-template brain/skull relationships."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import convex_hull_image


def skull_enclosure(skull: np.ndarray) -> np.ndarray:
    """Estimate the intracranial enclosure from each axial skull cross-section."""
    source = np.asarray(skull, dtype=bool)
    if source.ndim != 3:
        raise ValueError("Skull enclosure requires a 3D mask")
    enclosure = np.zeros_like(source)
    for axial in range(source.shape[2]):
        section = source[:, :, axial]
        if np.count_nonzero(section) >= 3:
            enclosure[:, :, axial] = convex_hull_image(section)
    return enclosure


def build_anatomy_targets(
    labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    brain_core_distance_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    source = np.asarray(labels)
    if source.ndim != 3 or brain_core_distance_mm <= 0:
        raise ValueError("Invalid anatomy target inputs")
    brain, skull = source == 1, source == 2
    if not np.any(brain) or not np.any(skull):
        raise ValueError("Brain and skull must both be present")
    enclosure = skull_enclosure(skull)
    brain_core = distance_transform_edt(brain, sampling=spacing_mm) >= float(brain_core_distance_mm)
    metrics = {
        "brain_enclosure_retention": float(np.count_nonzero(brain & enclosure) / np.count_nonzero(brain)),
        "skull_inside_brain_core_fraction": float(
            np.count_nonzero(skull & brain_core) / max(1, np.count_nonzero(skull))
        ),
    }
    return enclosure.astype(np.uint8), brain_core.astype(np.uint8), metrics


def prepare_anatomy_target_cache(
    records: list[dict],
    *,
    brain_core_distance_mm: float,
    progress=None,
) -> list[dict]:
    """Attach skull-derived enclosure and deep-brain targets to cached development records."""
    result = []
    for index, record in enumerate(records, 1):
        if record["split"] not in {"train", "val"}:
            raise ValueError("Anatomy target cache refuses official test records")
        case_root = Path(record["label_path"]).parent
        enclosure_path = case_root / "skull_enclosure.npy"
        brain_core_path = case_root / "brain_core.npy"
        metrics_path = case_root / "anatomy_target_metrics.json"
        contract = {
            "label_contract_sha256": record["contract_sha256"],
            "method": "axial_skull_convex_hull",
            "brain_core_distance_mm": float(brain_core_distance_mm),
        }
        if all(path.is_file() for path in (enclosure_path, brain_core_path, metrics_path)):
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            if saved.get("contract") != contract:
                raise ValueError(f"Existing anatomy target cache contract differs: {record['case_id']}")
            metrics = saved["metrics"]
        else:
            labels = np.load(record["label_path"], mmap_mode="r")
            enclosure, brain_core, metrics = build_anatomy_targets(
                labels,
                tuple(float(value) for value in record["spacing_mm"]),
                brain_core_distance_mm=brain_core_distance_mm,
            )
            for path, array in ((enclosure_path, enclosure), (brain_core_path, brain_core)):
                temporary = path.with_suffix(".tmp.npy")
                np.save(temporary, array)
                os.replace(temporary, path)
            temporary = metrics_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"contract": contract, "metrics": metrics}, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(metrics_path)
        updated = dict(record)
        updated["enclosure_path"] = str(enclosure_path)
        updated["brain_core_path"] = str(brain_core_path)
        updated["anatomy_target_metrics"] = metrics
        result.append(updated)
        if progress is not None:
            progress(index, len(records), record["case_id"])
    return result
