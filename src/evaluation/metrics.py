"""Numerically stable metrics for Phase 6 model evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label as connected_components
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _validate_logits_labels(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != targets.size or values.shape[1] < 2:
        raise ValueError("Logits must be [samples, classes] and align with labels")
    if not np.all(np.isfinite(values)) or np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("Calibration arrays contain non-finite or invalid values")
    return values, targets


def _nll(log_temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    temperature = math.exp(float(log_temperature))
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(labels.size), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Fit one positive scalar temperature on validation voxels only."""
    values, targets = _validate_logits_labels(logits, labels)
    result = minimize_scalar(
        _nll,
        args=(values, targets),
        method="bounded",
        bounds=(math.log(0.05), math.log(10.0)),
        options={"xatol": 1e-5, "maxiter": 200},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = math.exp(float(result.x))
    return {
        "temperature": temperature,
        "nll_before": _nll(0.0, values, targets),
        "nll_after": _nll(float(result.x), values, targets),
        "optimizer_iterations": int(result.nfev),
    }


def calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    temperature: float,
    bins: int = 15,
) -> dict[str, float | int]:
    values, targets = _validate_logits_labels(logits, labels)
    if not math.isfinite(temperature) or temperature <= 0 or bins < 2:
        raise ValueError("Temperature must be positive and bins >= 2")
    probabilities = softmax(values / temperature, axis=1)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    maximum_gap = 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if np.any(selected):
            gap = abs(float(np.mean(correct[selected])) - float(np.mean(confidence[selected])))
            ece += float(np.mean(selected)) * gap
            maximum_gap = max(maximum_gap, gap)
    one_hot = np.eye(values.shape[1], dtype=np.float64)[targets]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    nll = float(np.mean(-np.log(np.clip(probabilities[np.arange(targets.size), targets], 1e-12, 1.0))))
    return {
        "samples": int(targets.size),
        "nll": nll,
        "ece_15_bin": float(ece),
        "maximum_calibration_gap": float(maximum_gap),
        "multiclass_brier": brier,
        "voxel_accuracy": float(np.mean(correct)),
        "mean_confidence": float(np.mean(confidence)),
    }


def normalized_entropy(probabilities: np.ndarray, *, class_axis: int = 0) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    classes = values.shape[class_axis]
    if classes < 2:
        raise ValueError("Entropy requires at least two classes")
    clipped = np.clip(values, np.float32(1e-7), np.float32(1.0))
    return (-np.sum(clipped * np.log(clipped), axis=class_axis) / np.float32(math.log(classes))).astype(
        np.float32
    )


def select_error_threshold(uncertainty: np.ndarray, errors: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    targets = np.asarray(errors, dtype=np.uint8).reshape(-1)
    if scores.size != targets.size or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Uncertainty and error samples must be aligned and finite")
    if np.unique(targets).size < 2:
        raise ValueError("Threshold selection requires both correct and incorrect voxels")
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.nanargmax(f1))
    threshold = float(thresholds[best])
    flagged = scores >= threshold
    return {
        "criterion": "maximum_validation_voxel_error_f1",
        "threshold": threshold,
        "f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "flagged_fraction": float(np.mean(flagged)),
        "samples": int(scores.size),
    }


def uncertainty_error_metrics(uncertainty: np.ndarray, errors: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    targets = np.asarray(errors, dtype=np.uint8).reshape(-1)
    if scores.size != targets.size or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Uncertainty and error arrays must be aligned and finite")
    unique = np.unique(targets)
    auroc = float(roc_auc_score(targets, scores)) if unique.size == 2 else math.nan
    auprc = float(average_precision_score(targets, scores)) if unique.size == 2 else math.nan
    order = np.argsort(scores, kind="mergesort")
    ordered_error = targets[order].astype(np.float64)
    cumulative_error = np.cumsum(ordered_error)
    coverage = np.arange(1, targets.size + 1, dtype=np.float64) / targets.size
    risk = cumulative_error / np.arange(1, targets.size + 1, dtype=np.float64)
    aurc = float(np.trapezoid(risk, coverage))
    return {
        "samples": int(scores.size),
        "error_prevalence": float(np.mean(targets)),
        "error_detection_auroc": auroc,
        "error_detection_auprc": auprc,
        "area_under_risk_coverage_curve": aurc,
        "mean_uncertainty_correct": float(np.mean(scores[targets == 0])) if np.any(targets == 0) else math.nan,
        "mean_uncertainty_error": float(np.mean(scores[targets == 1])) if np.any(targets == 1) else math.nan,
    }


def apply_postprocessing(
    prediction: np.ndarray,
    *,
    policy: str,
    spacing_mm: tuple[float, float, float],
    min_component_ml: float = 1.0,
) -> np.ndarray:
    """Apply a class-wise, validation-selectable connected-component policy."""
    source = np.asarray(prediction, dtype=np.uint8)
    if source.ndim != 3:
        raise ValueError("Prediction must be a 3D label map")
    if policy == "none":
        return source.copy()
    if policy not in {"largest_component", "remove_below_1ml"}:
        raise ValueError(f"Unknown postprocessing policy: {policy}")
    output = np.zeros_like(source)
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    minimum_voxels = max(1, int(math.ceil(min_component_ml / voxel_ml)))
    for class_id in sorted(int(value) for value in np.unique(source) if value > 0):
        components, count = connected_components(source == class_id)
        if count == 0:
            continue
        sizes = np.bincount(components.reshape(-1))
        sizes[0] = 0
        if policy == "largest_component":
            keep = np.asarray([int(np.argmax(sizes))], dtype=np.int64)
        else:
            keep = np.flatnonzero(sizes >= minimum_voxels)
        output[np.isin(components, keep)] = np.uint8(class_id)
    return output


def _binary_hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> float:
    if not np.any(target) or not np.any(prediction):
        return math.nan
    union = prediction | target
    coordinates = np.where(union)
    bounds = []
    for axis, size in enumerate(union.shape):
        start = max(0, int(coordinates[axis].min()) - 2)
        stop = min(size, int(coordinates[axis].max()) + 3)
        bounds.append(slice(start, stop))
    predicted = prediction[tuple(bounds)]
    truth = target[tuple(bounds)]
    predicted_surface = predicted ^ binary_erosion(predicted, border_value=0)
    truth_surface = truth ^ binary_erosion(truth, border_value=0)
    if not np.any(predicted_surface) or not np.any(truth_surface):
        return math.nan
    distance_to_truth = distance_transform_edt(~truth_surface, sampling=spacing_mm)
    forward = distance_to_truth[predicted_surface]
    del distance_to_truth
    distance_to_prediction = distance_transform_edt(~predicted_surface, sampling=spacing_mm)
    backward = distance_to_prediction[truth_surface]
    return float(np.percentile(np.concatenate((forward, backward)), 95))


def case_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    class_names: dict[int, str],
    spacing_mm: tuple[float, float, float],
    compute_hd95: bool = True,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.uint8)
    truth = np.asarray(target, dtype=np.uint8)
    if predicted.shape != truth.shape or predicted.ndim != 3:
        raise ValueError("Prediction and target must be aligned 3D arrays")
    rows: list[dict[str, Any]] = []
    for class_id, organ in sorted(class_names.items()):
        p = predicted == class_id
        t = truth == class_id
        p_count, t_count = int(np.count_nonzero(p)), int(np.count_nonzero(t))
        if t_count == 0:
            dice, hd95 = math.nan, math.nan
        else:
            denominator = p_count + t_count
            dice = float(2 * np.count_nonzero(p & t) / denominator) if denominator else math.nan
            hd95 = _binary_hd95(p, t, spacing_mm) if compute_hd95 and p_count else math.nan
        rows.append(
            {
                "class_id": class_id,
                "organ": organ,
                "dice": dice,
                "hd95_mm": hd95,
                "ground_truth_present": bool(t_count),
                "prediction_present": bool(p_count),
                "false_positive_on_empty": not bool(t_count) and bool(p_count),
                "missing_prediction": bool(t_count) and not bool(p_count),
            }
        )
    dice_values = [row["dice"] for row in rows if math.isfinite(row["dice"])]
    return {
        "mean_foreground_dice": float(np.mean(dice_values)) if dice_values else math.nan,
        "per_class": rows,
    }


def aggregate_case_metrics(cases: list[dict[str, Any]], class_names: dict[int, str]) -> dict[str, Any]:
    if not cases:
        raise ValueError("At least one case metric record is required")
    per_class: list[dict[str, Any]] = []
    for class_id, organ in sorted(class_names.items()):
        records = [next(row for row in case["segmentation"]["per_class"] if row["class_id"] == class_id) for case in cases]
        dice = [
            float(row["dice"])
            for row in records
            if row.get("dice") is not None and math.isfinite(float(row["dice"]))
        ]
        hd95 = [
            float(row["hd95_mm"])
            for row in records
            if row.get("hd95_mm") is not None and math.isfinite(float(row["hd95_mm"]))
        ]
        per_class.append(
            {
                "class_id": class_id,
                "organ": organ,
                "dice_mean": float(np.mean(dice)) if dice else math.nan,
                "dice_std": float(np.std(dice, ddof=1)) if len(dice) > 1 else 0.0,
                "hd95_mm_mean": float(np.mean(hd95)) if hd95 else math.nan,
                "valid_dice_cases": len(dice),
                "valid_hd95_cases": len(hd95),
                "false_positive_empty_cases": sum(bool(row["false_positive_on_empty"]) for row in records),
                "missing_prediction_cases": sum(bool(row["missing_prediction"]) for row in records),
            }
        )
    macro = [row["dice_mean"] for row in per_class if math.isfinite(row["dice_mean"])]
    return {
        "cases": len(cases),
        "mean_foreground_dice": float(np.mean(macro)) if macro else math.nan,
        "per_class": per_class,
    }
