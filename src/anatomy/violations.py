"""Non-differentiable anatomical quality-control maps for predicted brain/skull masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label as connected_components, maximum_filter

from .targets import skull_enclosure


def fit_surface_distance_reference(
    records: list[dict],
    *,
    percentiles: tuple[float, float],
    samples_per_case: int,
    seed: int,
) -> dict[str, float | int | list[str]]:
    """Fit permissive brain-surface-to-skull distance bounds using training labels only."""
    lower_percentile, upper_percentile = (float(value) for value in percentiles)
    if not 0 <= lower_percentile < upper_percentile <= 100 or samples_per_case < 1:
        raise ValueError("Invalid anatomy reference settings")
    values, case_ids = [], []
    for record in records:
        if record["split"] != "train":
            raise ValueError("Anatomy distance reference may use training records only")
        labels = np.load(record["label_path"], mmap_mode="r")
        brain, skull = labels == 1, labels == 2
        surface = brain & ~binary_erosion(brain)
        distances = distance_transform_edt(~skull, sampling=tuple(record["spacing_mm"]))[surface]
        if distances.size:
            rng = np.random.default_rng(seed + int.from_bytes(record["case_id"].encode()[:4], "little"))
            if distances.size > samples_per_case:
                distances = distances[rng.choice(distances.size, samples_per_case, replace=False)]
            values.append(distances.astype(np.float32))
            case_ids.append(record["case_id"])
    if not values:
        raise ValueError("No training surface distances were available")
    combined = np.concatenate(values)
    return {
        "lower_mm": float(np.percentile(combined, lower_percentile)),
        "upper_mm": float(np.percentile(combined, upper_percentile)),
        "median_mm": float(np.median(combined)),
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "samples": int(combined.size),
        "training_cases": len(case_ids),
        "case_ids": case_ids,
    }


def _disconnected_map(
    prediction: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    minimum_ml: float,
    minimum_largest_fraction: float,
) -> tuple[np.ndarray, dict[str, int]]:
    result = np.zeros(prediction.shape, dtype=np.float32)
    counts = {}
    minimum_voxels = max(1, int(np.ceil(minimum_ml * 1000.0 / np.prod(spacing_mm))))
    for class_id, organ in ((1, "brain"), (2, "skull")):
        components, count = connected_components(prediction == class_id)
        sizes = np.bincount(components.ravel())
        if sizes.size <= 1:
            counts[organ] = 0
            continue
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        threshold = max(minimum_voxels, int(np.ceil(sizes[largest] * minimum_largest_fraction)))
        flagged = [index for index in range(1, count + 1) if index != largest and sizes[index] >= threshold]
        if flagged:
            result[np.isin(components, flagged)] = 1.0
        counts[organ] = len(flagged)
    return result, counts


def anatomy_violation_map(
    prediction: np.ndarray,
    probabilities: np.ndarray,
    predicted_sdf: np.ndarray,
    spacing_mm: tuple[float, float, float],
    distance_reference: dict,
    *,
    minimum_component_ml: float,
    minimum_largest_fraction: float,
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Create V(x) and auditable components without changing the prediction."""
    labels = np.asarray(prediction, dtype=np.uint8)
    probs = np.asarray(probabilities, dtype=np.float32)
    sdf = np.asarray(predicted_sdf, dtype=np.float32)
    if labels.ndim != 3 or probs.shape != (3, *labels.shape) or sdf.shape != (2, *labels.shape):
        raise ValueError("Anatomy QC inputs do not share the expected geometry")
    required = {
        "brain_outside_enclosure",
        "probability_ambiguity",
        "disconnected_components",
        "surface_distance",
        "sdf_sign_disagreement",
    }
    if set(weights) != required or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Violation weights must exactly cover all components")

    brain, skull = labels == 1, labels == 2
    enclosure = skull_enclosure(skull)
    outside = (brain & ~enclosure).astype(np.float32)
    ambiguity = np.clip(4.0 * probs[1] * probs[2], 0.0, 1.0)
    disconnected, disconnected_counts = _disconnected_map(
        labels,
        spacing_mm,
        minimum_ml=minimum_component_ml,
        minimum_largest_fraction=minimum_largest_fraction,
    )
    brain_surface = brain & ~binary_erosion(brain)
    distance_map = np.zeros(labels.shape, dtype=np.float32)
    surface_distances = np.asarray([], dtype=np.float32)
    if np.any(skull) and np.any(brain_surface):
        distances = distance_transform_edt(~skull, sampling=spacing_mm)
        surface_distances = distances[brain_surface].astype(np.float32)
        lower, upper = float(distance_reference["lower_mm"]), float(distance_reference["upper_mm"])
        local = np.zeros(labels.shape, dtype=np.float32)
        if lower > 0:
            local[brain_surface] = np.maximum(local[brain_surface], (lower - distances[brain_surface]) / lower)
        local[brain_surface] = np.maximum(
            local[brain_surface], (distances[brain_surface] - upper) / max(upper, 1e-6)
        )
        distance_map = maximum_filter(np.clip(local, 0.0, 1.0), size=3)
    sdf_disagreement = np.maximum(
        ((labels == 1) != (sdf[0] > 0)).astype(np.float32),
        ((labels == 2) != (sdf[1] > 0)).astype(np.float32),
    )
    components = {
        "brain_outside_enclosure": outside,
        "probability_ambiguity": ambiguity,
        "disconnected_components": disconnected,
        "surface_distance": np.clip(distance_map, 0.0, 1.0),
        "sdf_sign_disagreement": sdf_disagreement,
    }
    total_weight = float(sum(weights.values()))
    violation = sum(float(weights[name]) * array for name, array in components.items()) / total_weight
    violation = np.clip(violation, 0.0, 1.0).astype(np.float32)
    summary = {
        "brain_outside_enclosure_voxels": int(np.count_nonzero(outside)),
        "brain_outside_enclosure_fraction": float(np.count_nonzero(outside) / max(1, np.count_nonzero(brain))),
        "disconnected_major_components": disconnected_counts,
        "surface_distance_mean_mm": float(np.mean(surface_distances)) if surface_distances.size else None,
        "surface_distance_max_mm": float(np.max(surface_distances)) if surface_distances.size else None,
        "probability_ambiguity_mean": float(np.mean(ambiguity)),
        "sdf_sign_disagreement_fraction": float(np.mean(sdf_disagreement > 0)),
        "violation_mean": float(np.mean(violation)),
        "violation_nonzero_fraction": float(np.mean(violation > 0)),
        "violation_above_0_05_fraction": float(np.mean(violation >= 0.05)),
    }
    return violation, components, summary
