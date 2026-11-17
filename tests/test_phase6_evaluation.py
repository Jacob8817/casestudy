import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset.label_mapping import load_organ_schema
from src.dataset.loader import load_exclusive_case
from src.evaluation.metrics import (
    apply_postprocessing,
    calibration_metrics,
    case_segmentation_metrics,
    fit_temperature,
    normalized_entropy,
    select_error_threshold,
    uncertainty_error_metrics,
)
from src.evaluation.pipeline import _bootstrap_macro_dice, select_predeclared_mesh_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_temperature_scaling_and_calibration_are_finite() -> None:
    logits = np.asarray([[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [2.0, 0.0]], dtype=np.float32)
    labels = np.asarray([0, 1, 1, 0], dtype=np.uint8)
    fitted = fit_temperature(logits, labels)
    assert 0.05 <= fitted["temperature"] <= 10.0
    assert fitted["nll_after"] <= fitted["nll_before"] + 1e-8
    report = calibration_metrics(logits, labels, temperature=fitted["temperature"])
    assert report["samples"] == 4
    assert all(np.isfinite(value) for value in report.values())


def test_entropy_and_error_detection_contract() -> None:
    probabilities = np.asarray([[0.99, 0.5, 0.6], [0.01, 0.5, 0.4]], dtype=np.float32)
    entropy = normalized_entropy(probabilities)
    assert entropy[0] < entropy[2] < entropy[1]
    errors = np.asarray([0, 1, 0], dtype=np.uint8)
    threshold = select_error_threshold(entropy, errors)
    metrics = uncertainty_error_metrics(entropy, errors)
    assert 0 <= threshold["threshold"] <= 1
    assert metrics["error_detection_auroc"] > 0.5


def test_postprocessing_is_classwise_and_deterministic() -> None:
    prediction = np.zeros((8, 8, 8), dtype=np.uint8)
    prediction[1:4, 1:4, 1:4] = 1
    prediction[7, 7, 7] = 1
    prediction[4:7, 4:7, 4:7] = 2
    processed = apply_postprocessing(
        prediction, policy="largest_component", spacing_mm=(1.5, 1.5, 1.5)
    )
    assert processed[7, 7, 7] == 0
    assert np.count_nonzero(processed == 1) == 27
    assert np.count_nonzero(processed == 2) == 27


def test_segmentation_metric_identity() -> None:
    target = np.zeros((7, 7, 7), dtype=np.uint8)
    target[2:5, 2:5, 2:5] = 1
    report = case_segmentation_metrics(
        target,
        target,
        class_names={1: "organ"},
        spacing_mm=(1.5, 1.5, 1.5),
    )
    assert report["mean_foreground_dice"] == 1.0
    assert report["per_class"][0]["hd95_mm"] == 0.0


def test_test_loader_stays_fail_closed_without_receipt() -> None:
    schema = load_organ_schema(PROJECT_ROOT / "configs" / "organs_v1.yaml")
    manifest = json.loads((PROJECT_ROOT / "data" / "splits" / "dataset_manifest.json").read_text())
    test_case = next(case for case in manifest["cases"] if case["split"] == "test")
    with pytest.raises(PermissionError, match="one-shot authorization"):
        load_exclusive_case(PROJECT_ROOT / "path_must_not_be_resolved", test_case, schema)


def test_mesh_case_selection_uses_manifest_geometry_only() -> None:
    manifest = json.loads((PROJECT_ROOT / "data" / "splits" / "dataset_manifest.json").read_text())
    selected = select_predeclared_mesh_case(manifest)
    assert selected["case_id"] == "s0244"
    assert selected["selection_used_labels"] is False


def test_patient_bootstrap_excludes_undefined_empty_target_cases() -> None:
    cases = [
        {"segmentation": {"mean_foreground_dice": 0.8}},
        {"segmentation": {"mean_foreground_dice": None}},
        {"segmentation": {"mean_foreground_dice": 1.0}},
    ]
    report = _bootstrap_macro_dice(cases, seed=42, replicates=100)
    assert report["valid_patients"] == 2
    assert report["excluded_patients"] == 1
    assert report["point_estimate"] == pytest.approx(0.9)
    assert np.isfinite(report["ci95_lower"])
    assert np.isfinite(report["ci95_upper"])
