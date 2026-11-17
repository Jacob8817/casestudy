"""Phase 4 temperature calibration and brain/skull predictive entropy."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy.special import softmax

from src.evaluation.metrics import (
    calibration_metrics,
    fit_temperature,
    normalized_entropy,
    select_error_threshold,
    uncertainty_error_metrics,
)


def infer_multitask_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    device: torch.device,
    roi: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-volume segmentation logits and SDFs from the multi-head model."""
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 4 or source.shape[0] != 2:
        raise ValueError("Expected two-channel [C,H,W,D] CT input")
    tensor = torch.from_numpy(source[None]).to(device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, sdf = sliding_window_inference(
            inputs=tensor,
            roi_size=(roi, roi, roi),
            sw_batch_size=1,
            predictor=model,
            overlap=float(overlap),
            mode="gaussian",
        )
    if not torch.isfinite(logits).all() or not torch.isfinite(sdf).all():
        raise FloatingPointError("Non-finite Phase 4 inference output")
    output_logits = logits[0].float().cpu().numpy()
    output_sdf = sdf[0].float().cpu().numpy()
    del tensor, logits, sdf
    return output_logits, output_sdf


def _indices(size: int, count: int, seed: int, case_id: str, purpose: str) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{case_id}:{purpose}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.choice(size, min(size, count), replace=False))


def _sample(logits: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return logits.reshape(logits.shape[0], -1)[:, indices].T, labels.reshape(-1)[indices]


def _save_nifti(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), str(path))


def run_validation_uncertainty(
    model: torch.nn.Module,
    records: list[dict],
    config: dict,
    output_root: Path,
    *,
    device: torch.device,
    roi: int,
    overlap: float,
    progress=print,
    frozen_calibration_report: dict | None = None,
    artifact_observer=None,
) -> dict:
    """Fit scalar temperature and save entropy maps on validation only."""
    if not records or any(item["split"] != "val" for item in records):
        raise ValueError("Phase 4 accepts validation records only")
    output_root.mkdir(parents=True, exist_ok=True)
    sample_count = int(config["calibration_samples_per_case"])
    seed = int(config["random_seed"])
    sampled_logits, sampled_labels = [], []
    for record in records:
        image = np.load(record["image_path"], mmap_mode="r")
        labels = np.load(record["label_path"], mmap_mode="r")
        logits, _ = infer_multitask_logits(model, image, device=device, roi=roi, overlap=overlap)
        index = _indices(labels.size, sample_count, seed, record["case_id"], "calibration")
        values, targets = _sample(logits, labels, index)
        sampled_logits.append(values.astype(np.float32))
        sampled_labels.append(targets.astype(np.uint8))
        progress(f"Phase 4 calibration inference: {record['case_id']}")
    calibration_logits = np.concatenate(sampled_logits)
    calibration_labels = np.concatenate(sampled_labels)
    fitted = (
        fit_temperature(calibration_logits, calibration_labels)
        if frozen_calibration_report is None
        else dict(frozen_calibration_report["temperature_scaling"])
    )
    temperature = float(fitted["temperature"])
    before = calibration_metrics(
        calibration_logits, calibration_labels, temperature=1.0, bins=int(config["ece_bins"])
    )
    after = calibration_metrics(
        calibration_logits, calibration_labels, temperature=temperature, bins=int(config["ece_bins"])
    )
    np.savez_compressed(output_root / "calibration_samples.npz", logits=calibration_logits, labels=calibration_labels)

    uncertainty_samples, error_samples, cases = [], [], []
    error_count = int(config["error_analysis_samples_per_case"])
    for record in records:
        started = time.perf_counter()
        image = np.load(record["image_path"], mmap_mode="r")
        labels = np.load(record["label_path"], mmap_mode="r")
        logits, sdf = infer_multitask_logits(model, image, device=device, roi=roi, overlap=overlap)
        probabilities = softmax(logits / temperature, axis=0).astype(np.float32)
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        entropy = normalized_entropy(probabilities, class_axis=0)
        if artifact_observer is not None:
            artifact_observer(record, logits, sdf, probabilities, entropy)
        error = prediction != labels
        index = _indices(labels.size, error_count, seed, record["case_id"], "error-analysis")
        uncertainty_samples.append(entropy.reshape(-1)[index])
        error_samples.append(error.reshape(-1)[index].astype(np.uint8))
        case_root = output_root / "cases" / record["case_id"]
        affine = np.asarray(record["ct_affine"], dtype=np.float64)
        _save_nifti(case_root / "initial_prediction.nii.gz", prediction, affine)
        _save_nifti(case_root / "predictive_entropy.nii.gz", entropy, affine)
        _save_nifti(case_root / "segmentation_error.nii.gz", error.astype(np.uint8), affine)
        np.savez_compressed(case_root / "geometry_outputs.npz", sdf=sdf.astype(np.float16))
        cases.append(
            {
                "case_id": record["case_id"],
                "mean_entropy": float(np.mean(entropy)),
                "mean_entropy_correct": float(np.mean(entropy[~error])) if np.any(~error) else None,
                "mean_entropy_error": float(np.mean(entropy[error])) if np.any(error) else None,
                "error_fraction": float(np.mean(error)),
                "inference_seconds": float(time.perf_counter() - started),
            }
        )
        progress(f"Phase 4 entropy saved: {record['case_id']}")
    uncertainty = np.concatenate(uncertainty_samples)
    errors = np.concatenate(error_samples)
    threshold = (
        select_error_threshold(uncertainty, errors)
        if frozen_calibration_report is None
        else dict(frozen_calibration_report["selected_error_threshold"])
    )
    error_metrics = uncertainty_error_metrics(uncertainty, errors)
    report = {
        "schema_version": 1,
        "phase": 4,
        "status": "passed",
        "method": config["method"],
        "interpretation": config["interpretation"],
        "temperature_scaling": fitted,
        "temperature_refitted": frozen_calibration_report is None,
        "error_threshold_reselected": frozen_calibration_report is None,
        "calibration_before": before,
        "calibration_after": after,
        "selected_error_threshold": threshold,
        "uncertainty_error_analysis": error_metrics,
        "cases": cases,
        "validation_cases": len(records),
        "official_test_split_accessed": False,
    }
    (output_root / "phase4_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
