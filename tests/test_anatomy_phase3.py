from __future__ import annotations

import numpy as np
import pytest
import torch

from src.anatomy.constraints import AnatomyRelationalLoss
from src.anatomy.targets import build_anatomy_targets, skull_enclosure
from src.anatomy.violations import anatomy_violation_map, fit_surface_distance_reference


def test_skull_enclosure_fills_axial_ring() -> None:
    skull = np.zeros((24, 24, 3), dtype=bool)
    skull[4:20, 4, 1] = True
    skull[4:20, 19, 1] = True
    skull[4, 4:20, 1] = True
    skull[19, 4:20, 1] = True
    enclosure = skull_enclosure(skull)
    assert enclosure[12, 12, 1]
    assert not enclosure[1, 1, 1]
    assert not enclosure[:, :, 0].any()


def test_anatomy_targets_use_physical_brain_core() -> None:
    labels = np.zeros((24, 24, 5), dtype=np.uint8)
    labels[5:19, 5:19, 1:4] = 1
    labels[3:21, 3, 1:4] = 2
    labels[3:21, 20, 1:4] = 2
    labels[3, 3:21, 1:4] = 2
    labels[20, 3:21, 1:4] = 2
    enclosure, core, metrics = build_anatomy_targets(
        labels, (1.0, 1.0, 2.0), brain_core_distance_mm=2.0
    )
    assert enclosure.shape == labels.shape and core.shape == labels.shape
    assert enclosure.dtype == np.uint8 and core.dtype == np.uint8
    assert metrics["brain_enclosure_retention"] == pytest.approx(1.0)
    assert core[12, 12, 2] == 1
    assert core[5, 5, 2] == 0


def test_relational_loss_penalizes_anatomically_invalid_probabilities() -> None:
    objective = AnatomyRelationalLoss(
        brain_outside_enclosure_weight=1.0,
        skull_in_brain_core_weight=1.0,
        probability_ambiguity_weight=0.25,
        sdf_sign_consistency_weight=0.25,
    )
    enclosure = torch.zeros((1, 1, 8, 8, 8))
    enclosure[:, :, 2:6, 2:6, 2:6] = 1
    core = torch.zeros_like(enclosure)
    core[:, :, 3:5, 3:5, 3:5] = 1
    valid_logits = torch.full((1, 3, 8, 8, 8), -6.0)
    valid_logits[:, 0] = 3.0
    valid_logits[:, 1, 2:6, 2:6, 2:6] = 8.0
    valid_logits[:, 2, 1:7, 1:7, 1:7] = 6.0
    valid_logits[:, 1, 3:5, 3:5, 3:5] = 10.0
    valid_sdf = torch.where(valid_logits[:, 1:3] > 0, 1.0, -1.0)
    invalid_logits = valid_logits.clone()
    invalid_logits[:, 1, :2, :2, :2] = 12.0
    invalid_logits[:, 2, 3:5, 3:5, 3:5] = 12.0
    invalid_sdf = -torch.where(invalid_logits[:, 1:3] > 0, 1.0, -1.0)
    valid, _ = objective(valid_logits, valid_sdf, enclosure, core)
    invalid, parts = objective(invalid_logits, invalid_sdf, enclosure, core)
    assert torch.isfinite(valid) and torch.isfinite(invalid)
    assert invalid > valid
    assert all(torch.isfinite(value) for value in parts.values())


def test_violation_map_is_bounded_and_does_not_modify_prediction() -> None:
    prediction = np.zeros((24, 24, 3), dtype=np.uint8)
    prediction[4:20, 4, 1] = 2
    prediction[4:20, 19, 1] = 2
    prediction[4, 4:20, 1] = 2
    prediction[19, 4:20, 1] = 2
    prediction[8:16, 8:16, 1] = 1
    prediction[1:3, 1:3, 1] = 1
    original = prediction.copy()
    probabilities = np.eye(3, dtype=np.float32)[prediction].transpose(3, 0, 1, 2)
    sdf = np.stack(
        [np.where(prediction == class_id, 1.0, -1.0) for class_id in (1, 2)], axis=0
    ).astype(np.float32)
    violation, components, summary = anatomy_violation_map(
        prediction,
        probabilities,
        sdf,
        (1.0, 1.0, 1.0),
        {"lower_mm": 1.0, "upper_mm": 8.0},
        minimum_component_ml=0.001,
        minimum_largest_fraction=0.01,
        weights={
            "brain_outside_enclosure": 0.30,
            "probability_ambiguity": 0.15,
            "disconnected_components": 0.20,
            "surface_distance": 0.20,
            "sdf_sign_disagreement": 0.15,
        },
    )
    assert np.array_equal(prediction, original)
    assert violation.shape == prediction.shape
    assert 0.0 <= float(violation.min()) <= float(violation.max()) <= 1.0
    assert set(components) == {
        "brain_outside_enclosure",
        "probability_ambiguity",
        "disconnected_components",
        "surface_distance",
        "sdf_sign_disagreement",
    }
    assert summary["brain_outside_enclosure_voxels"] > 0
    assert summary["disconnected_major_components"]["brain"] == 1
    assert 0.0 <= summary["violation_above_0_05_fraction"] <= 1.0


def test_distance_reference_rejects_nontraining_records() -> None:
    with pytest.raises(ValueError, match="training records only"):
        fit_surface_distance_reference(
            [{"split": "val", "label_path": "unused.npy"}],
            percentiles=(1.0, 99.0),
            samples_per_case=10,
            seed=1,
        )
