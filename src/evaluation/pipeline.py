"""Ordered Phase 6 workflow: validation calibration, frozen protocol, test, mesh."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from monai.inferers import sliding_window_inference

from src.dataset import (
    canonicalize_to_ras,
    load_exclusive_case,
    normalize_ct,
    resample_pair,
)
from src.training.config import BaselineConfig
from src.training.data import prepare_case
from src.training.model import build_model

from .metrics import (
    aggregate_case_metrics,
    apply_postprocessing,
    calibration_metrics,
    case_segmentation_metrics,
    fit_temperature,
    normalized_entropy,
    select_error_threshold,
    uncertainty_error_metrics,
)


Progress = Callable[[str], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_phase6_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Phase 6 config must be a schema-version 1 mapping")
    expected = {"experiment", "locks", "checkpoint", "uncertainty", "postprocessing", "test", "mesh", "outputs", "reproducibility"}
    if set(raw) != expected | {"schema_version"}:
        raise ValueError(f"Unexpected Phase 6 config keys: {sorted(set(raw) - expected - {'schema_version'})}")
    if raw["test"].get("authorization_token") != "FINAL_ONE_SHOT_TEST_EVALUATION":
        raise ValueError("The one-shot test authorization token changed")
    if raw["uncertainty"].get("method") != "temperature_scaled_softmax_predictive_entropy":
        raise ValueError("Unsupported uncertainty method")
    return raw


def validate_phase6_context(
    *,
    project_root: Path,
    phase6_config: dict[str, Any],
    baseline_config: BaselineConfig,
    manifest: dict[str, Any],
    schema: Any,
    preprocessing: Any,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    locks = phase6_config["locks"]
    preprocessing_sha = preprocessing.fingerprint(
        manifest_sha256=manifest["manifest_sha256"], schema_sha256=schema.fingerprint()
    )
    checks = {
        "manifest": locks["manifest_sha256"] == manifest["manifest_sha256"] == baseline_config.manifest_sha256,
        "organ_schema": locks["organ_schema_sha256"] == schema.fingerprint() == baseline_config.organ_schema_sha256,
        "preprocessing": locks["preprocessing_sha256"] == preprocessing_sha == baseline_config.preprocessing_sha256,
        "training_config": locks["training_config_sha256"] == baseline_config.config_sha256,
        "checkpoint_epoch": int(phase6_config["checkpoint"]["expected_epoch"]) == 50,
        "test_lock": manifest["split_policy"]["test_lock"]["locked"] is True,
        "cuda": torch.cuda.is_available(),
        "dropout_zero": baseline_config.dropout == 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Phase 6 preflight failed: {failed}")
    checkpoint_path = (project_root / phase6_config["checkpoint"]["path"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_checks = {
        "kind_best": checkpoint.get("kind") == "best",
        "epoch": int(checkpoint.get("epoch", -1)) == int(phase6_config["checkpoint"]["expected_epoch"]),
        "config_sha256": checkpoint.get("config_sha256") == baseline_config.config_sha256,
        "validation_score": math.isclose(
            float(checkpoint.get("best_validation_mean_foreground_dice")),
            float(phase6_config["checkpoint"]["expected_validation_mean_foreground_dice"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    if not all(checkpoint_checks.values()):
        raise RuntimeError(f"Checkpoint contract failed: {checkpoint_checks}")
    model = build_model(baseline_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(torch.device("cuda:0"))
    del checkpoint
    return {
        "checks": checks | checkpoint_checks,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model": model,
        "device": torch.device("cuda:0"),
        "preprocessing_sha256": preprocessing_sha,
    }


def _infer_logits(
    image: np.ndarray,
    *,
    model: torch.nn.Module,
    device: torch.device,
    config: BaselineConfig,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image[np.newaxis, np.newaxis], dtype=np.float32))
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda", dtype=torch.float16, enabled=config.automatic_mixed_precision
    ):
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=config.sliding_window_roi_size,
            sw_batch_size=config.sliding_window_batch_size,
            predictor=model,
            overlap=config.sliding_window_overlap,
            mode=config.sliding_window_blend_mode,
            sw_device=device,
            device="cpu",
        )
    if logits.shape[0] != 1 or logits.shape[1] != config.out_channels or not torch.isfinite(logits).all():
        raise FloatingPointError(f"Invalid full-volume logits: {tuple(logits.shape)}")
    return logits.float().cpu()


def _entropy_from_logits(logits: torch.Tensor, temperature: float, chunk_voxels: int = 1_000_000) -> np.ndarray:
    flattened = logits[0].reshape(logits.shape[1], -1)
    output = np.empty(flattened.shape[1], dtype=np.float32)
    for start in range(0, flattened.shape[1], chunk_voxels):
        stop = min(flattened.shape[1], start + chunk_voxels)
        probabilities = torch.softmax(flattened[:, start:stop] / float(temperature), dim=0).numpy()
        output[start:stop] = normalized_entropy(probabilities, class_axis=0)
    return output.reshape(tuple(logits.shape[2:]))


def _sample_indices(size: int, count: int, *, seed: int, case_id: str, purpose: str) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{case_id}:{purpose}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.choice(size, size=min(size, count), replace=False))


def _sample_logits(logits: torch.Tensor, labels: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = logits[0].reshape(logits.shape[1], -1)
    sampled = flat[:, torch.from_numpy(indices)].T.numpy().astype(np.float32, copy=False)
    return sampled, labels.reshape(-1)[indices].astype(np.uint8, copy=False)


def _validation_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = [case for case in manifest["cases"] if case["split"] == "val"]
    if len(records) != 57:
        raise ValueError(f"Expected 57 validation cases, found {len(records)}")
    return sorted(records, key=lambda item: item["case_id"])


def _prepare_validation(
    case: dict[str, Any],
    *,
    project_root: Path,
    dataset_root: Path,
    schema: Any,
    preprocessing: Any,
    preprocessing_sha256: str,
    config: BaselineConfig,
) -> Any:
    return prepare_case(
        project_root=project_root,
        dataset_root=dataset_root,
        case=case,
        schema=schema,
        preprocessing=preprocessing,
        preprocessing_sha256=preprocessing_sha256,
        config=config,
        allow_cache_write=True,
        verify_source_hashes=False,
        verify_cache_hashes=False,
    )


def _overlay_figure(
    *,
    ct_hu: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    class_names: dict[int, str],
    case_id: str,
    output_path: Path,
) -> None:
    foreground_by_slice = np.count_nonzero(target > 0, axis=(0, 1))
    slice_index = int(np.argmax(foreground_by_slice))
    ct = np.clip(ct_hu[:, :, slice_index], -200, 350).T
    truth = target[:, :, slice_index].T
    predicted = prediction[:, :, slice_index].T
    entropy = uncertainty[:, :, slice_index].T
    error = predicted != truth
    colors = ["#00000000", "#c24242", "#874baa", "#387eb8", "#279f8c", "#f0913f", "#e7c44e", "#bf283c", "#4b69be", "#70bcd2", "#58a6c8"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=150)
    axes[0].imshow(ct, cmap="gray", origin="lower")
    axes[0].set_title("CT (HU -200 to 350)")
    axes[1].imshow(ct, cmap="gray", origin="lower")
    axes[1].imshow(np.ma.masked_where(truth == 0, truth), cmap=cmap, norm=norm, alpha=0.65, origin="lower")
    axes[1].set_title("Ground truth")
    axes[2].imshow(ct, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(predicted == 0, predicted), cmap=cmap, norm=norm, alpha=0.65, origin="lower")
    axes[2].set_title("Prediction")
    axes[3].imshow(ct, cmap="gray", origin="lower")
    axes[3].imshow(np.ma.masked_where(~error, error), cmap="Reds", alpha=0.72, origin="lower")
    axes[3].set_title("Segmentation error")
    shown = axes[4].imshow(entropy, cmap="magma", vmin=0, vmax=1, origin="lower")
    axes[4].set_title("Calibrated normalized entropy")
    fig.colorbar(shown, ax=axes[4], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{case_id} — axial slice {slice_index} | {len(class_names)} foreground organs")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_validation_analysis(
    *,
    project_root: Path,
    dataset_root: Path,
    phase6_config: dict[str, Any],
    baseline_config: BaselineConfig,
    manifest: dict[str, Any],
    schema: Any,
    preprocessing: Any,
    context: dict[str, Any],
    progress: Progress = print,
) -> dict[str, Any]:
    root = (project_root / phase6_config["outputs"]["validation"]).resolve()
    report_path = root / "validation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("checkpoint_sha256") != context["checkpoint_sha256"]:
            raise RuntimeError("Existing validation report belongs to a different checkpoint")
        progress("Validation report already complete; verified and reused.")
        return report
    root.mkdir(parents=True, exist_ok=True)
    calibration_root = root / "calibration_samples"
    analysis_root = root / "case_analysis"
    sample_root = root / "error_samples"
    overlay_root = root / "qualitative_overlays"
    for directory in (calibration_root, analysis_root, sample_root, overlay_root):
        directory.mkdir(exist_ok=True)
    records = _validation_records(manifest)
    class_names = {label.id: label.name for label in schema.labels}
    seed = int(phase6_config["reproducibility"]["random_seed"])
    calibration_count = int(phase6_config["uncertainty"]["calibration_sample_voxels_per_case"])
    error_count = int(phase6_config["uncertainty"]["error_analysis_sample_voxels_per_case"])
    qualitative_ids = ("s0249", "s1017", "s1161")

    progress("Validation pass 1/2: collecting deterministic voxel logits for temperature scaling.")
    for number, case in enumerate(records, start=1):
        sample_path = calibration_root / f"{case['case_id']}.npz"
        if sample_path.is_file():
            continue
        prepared = _prepare_validation(
            case,
            project_root=project_root,
            dataset_root=dataset_root,
            schema=schema,
            preprocessing=preprocessing,
            preprocessing_sha256=context["preprocessing_sha256"],
            config=baseline_config,
        )
        logits = _infer_logits(
            normalize_ct(prepared.ct_hu, preprocessing),
            model=context["model"],
            device=context["device"],
            config=baseline_config,
        )
        indices = _sample_indices(prepared.label.size, calibration_count, seed=seed, case_id=prepared.case_id, purpose="calibration")
        sampled_logits, sampled_labels = _sample_logits(logits, prepared.label, indices)
        np.savez_compressed(sample_path, logits=sampled_logits.astype(np.float16), labels=sampled_labels)
        progress(f"  calibration {number:02d}/{len(records)}: {prepared.case_id}")
        del logits, prepared

    calibration_files = [calibration_root / f"{case['case_id']}.npz" for case in records]
    calibration_payloads = [np.load(path) for path in calibration_files]
    calibration_logits = np.concatenate([item["logits"].astype(np.float32) for item in calibration_payloads], axis=0)
    calibration_labels = np.concatenate([item["labels"] for item in calibration_payloads], axis=0)
    for item in calibration_payloads:
        item.close()
    fitted = fit_temperature(calibration_logits, calibration_labels)
    before = calibration_metrics(calibration_logits, calibration_labels, temperature=1.0)
    after = calibration_metrics(calibration_logits, calibration_labels, temperature=float(fitted["temperature"]))
    temperature = float(fitted["temperature"])

    progress(f"Fitted validation temperature: T={temperature:.5f}. Validation pass 2/2: error analysis and postprocessing.")
    for number, case in enumerate(records, start=1):
        case_path = analysis_root / f"{case['case_id']}.json"
        sample_path = sample_root / f"{case['case_id']}.npz"
        if case_path.is_file() and sample_path.is_file():
            continue
        prepared = _prepare_validation(
            case,
            project_root=project_root,
            dataset_root=dataset_root,
            schema=schema,
            preprocessing=preprocessing,
            preprocessing_sha256=context["preprocessing_sha256"],
            config=baseline_config,
        )
        logits = _infer_logits(
            normalize_ct(prepared.ct_hu, preprocessing),
            model=context["model"],
            device=context["device"],
            config=baseline_config,
        )
        prediction = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        entropy = _entropy_from_logits(logits, temperature)
        policies: dict[str, Any] = {}
        for policy in phase6_config["postprocessing"]["candidates"]:
            processed = apply_postprocessing(prediction, policy=policy, spacing_mm=preprocessing.spacing_mm)
            policies[policy] = case_segmentation_metrics(
                processed,
                prepared.label,
                class_names=class_names,
                spacing_mm=preprocessing.spacing_mm,
                compute_hd95=False,
            )
        indices = _sample_indices(prepared.label.size, error_count, seed=seed, case_id=prepared.case_id, purpose="uncertainty_error")
        sampled_uncertainty = entropy.reshape(-1)[indices].astype(np.float32)
        sampled_error = (prediction.reshape(-1)[indices] != prepared.label.reshape(-1)[indices]).astype(np.uint8)
        np.savez_compressed(sample_path, uncertainty=sampled_uncertainty.astype(np.float16), error=sampled_error)
        atomic_json(case_path, {"case_id": prepared.case_id, "split": "val", "policies": policies})
        if prepared.case_id in qualitative_ids:
            _overlay_figure(
                ct_hu=prepared.ct_hu,
                target=prepared.label,
                prediction=prediction,
                uncertainty=entropy,
                class_names=class_names,
                case_id=prepared.case_id,
                output_path=overlay_root / f"{prepared.case_id}_overlay.png",
            )
        progress(f"  analysis    {number:02d}/{len(records)}: {prepared.case_id}")
        del logits, entropy, prediction, prepared

    case_payloads = [json.loads((analysis_root / f"{case['case_id']}.json").read_text(encoding="utf-8")) for case in records]
    policy_summaries: dict[str, Any] = {}
    for policy in phase6_config["postprocessing"]["candidates"]:
        wrapped = [{"segmentation": item["policies"][policy]} for item in case_payloads]
        policy_summaries[policy] = aggregate_case_metrics(wrapped, class_names)
    tie_order = list(phase6_config["postprocessing"]["tie_break_order"])
    selected_policy = max(
        phase6_config["postprocessing"]["candidates"],
        key=lambda name: (float(policy_summaries[name]["mean_foreground_dice"]), -tie_order.index(name)),
    )
    error_payloads = [np.load(sample_root / f"{case['case_id']}.npz") for case in records]
    uncertainty = np.concatenate([item["uncertainty"].astype(np.float32) for item in error_payloads])
    errors = np.concatenate([item["error"] for item in error_payloads])
    for item in error_payloads:
        item.close()
    threshold = select_error_threshold(uncertainty, errors)
    uncertainty_metrics = uncertainty_error_metrics(uncertainty, errors)
    report = {
        "schema_version": 1,
        "status": "passed",
        "generated_at_utc": utc_now(),
        "split": "val",
        "cases": len(records),
        "checkpoint_sha256": context["checkpoint_sha256"],
        "uncertainty_method": phase6_config["uncertainty"]["method"],
        "interpretation": phase6_config["uncertainty"]["interpretation"],
        "temperature_scaling": fitted,
        "calibration_before": before,
        "calibration_after": after,
        "uncertainty_error_analysis": uncertainty_metrics,
        "selected_error_threshold": threshold,
        "postprocessing_candidates": policy_summaries,
        "selected_postprocessing": selected_policy,
        "selection_metric": phase6_config["postprocessing"]["selection_metric"],
        "qualitative_case_ids": list(qualitative_ids),
        "qualitative_overlays": [f"qualitative_overlays/{case_id}_overlay.png" for case_id in qualitative_ids],
        "test_split_accessed": False,
    }
    atomic_json(report_path, report)
    return json.loads(report_path.read_text(encoding="utf-8"))


def select_predeclared_mesh_case(manifest: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in manifest["cases"] if case["split"] == "test"]
    volumes = np.asarray([np.prod(case["ct"]["shape"]) * np.prod(case["ct"]["spacing_mm"]) for case in cases])
    target = float(np.quantile(volumes, 0.75))
    eligible = [case for case in cases if min(case["ct"]["shape"]) >= 200]
    selected = min(
        eligible,
        key=lambda case: (
            abs(float(np.prod(case["ct"]["shape"]) * np.prod(case["ct"]["spacing_mm"])) - target),
            case["case_id"],
        ),
    )
    return {
        "case_id": selected["case_id"],
        "selection_used_labels": False,
        "target_75th_percentile_physical_volume_mm3": target,
        "native_shape": selected["ct"]["shape"],
        "native_spacing_mm": selected["ct"]["spacing_mm"],
    }


def freeze_protocol(
    *,
    project_root: Path,
    phase6_config_path: Path,
    phase6_config: dict[str, Any],
    baseline_config_path: Path,
    manifest: dict[str, Any],
    schema: Any,
    context: dict[str, Any],
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    if validation_report.get("test_split_accessed") is not False:
        raise RuntimeError("Validation report does not prove the test remained locked")
    mesh_selection = select_predeclared_mesh_case(manifest)
    if mesh_selection["case_id"] != phase6_config["mesh"]["expected_case_id"]:
        raise RuntimeError("Predeclared mesh selection rule no longer resolves to the expected case")
    protocol_path = (project_root / phase6_config["outputs"]["protocol"]).resolve()
    test_ids = sorted(case["case_id"] for case in manifest["cases"] if case["split"] == "test")
    payload = {
        "schema_version": 1,
        "status": "frozen_before_test_access",
        "frozen_at_utc": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "organ_schema_sha256": schema.fingerprint(),
        "preprocessing_sha256": context["preprocessing_sha256"],
        "baseline_config_sha256": baseline_config_path and sha256_file(baseline_config_path),
        "phase6_config_sha256": sha256_file(phase6_config_path),
        "checkpoint_sha256": context["checkpoint_sha256"],
        "uncertainty_method": validation_report["uncertainty_method"],
        "temperature": float(validation_report["temperature_scaling"]["temperature"]),
        "error_flag_threshold": float(validation_report["selected_error_threshold"]["threshold"]),
        "postprocessing_policy": validation_report["selected_postprocessing"],
        "postprocessing_selection_metric": validation_report["selection_metric"],
        "validation_report_sha256": canonical_sha256(validation_report),
        "test_case_count": len(test_ids),
        "test_case_ids_sha256": hashlib.sha256(("\n".join(test_ids) + "\n").encode("utf-8")).hexdigest(),
        "test_rerun_policy": phase6_config["test"]["rerun_policy"],
        "mesh_selection_rule": phase6_config["mesh"]["selection_rule"],
        "mesh_selection": mesh_selection,
        "test_labels_used_for_any_selection": False,
    }
    comparable = dict(payload)
    comparable.pop("frozen_at_utc")
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        existing_comparable = dict(existing)
        expected_hash = existing_comparable.pop("protocol_sha256", None)
        existing_comparable.pop("frozen_at_utc", None)
        if existing_comparable != comparable or canonical_sha256({key: value for key, value in existing.items() if key != "protocol_sha256"}) != expected_hash:
            raise RuntimeError("Frozen protocol exists and differs; refusing to overwrite it")
        return existing
    payload["protocol_sha256"] = canonical_sha256(payload)
    atomic_json(protocol_path, payload)
    protocol_path.with_suffix(".sha256").write_text(payload["protocol_sha256"] + "\n", encoding="ascii")
    return payload


def authorize_one_shot_test(
    *, project_root: Path, phase6_config: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    if protocol.get("status") != "frozen_before_test_access":
        raise PermissionError("The final protocol is not frozen")
    test_root = (project_root / phase6_config["outputs"]["test"]).resolve()
    test_root.mkdir(parents=True, exist_ok=True)
    receipt_path = test_root / "test_access_receipt.json"
    receipt = {
        "schema_version": 1,
        "status": "authorized",
        "authorization_token": phase6_config["test"]["authorization_token"],
        "frozen_protocol_sha256": protocol["protocol_sha256"],
        "authorized_at_utc": utc_now(),
        "scope": "exact official test split; one final evaluation; exact case-level resume allowed",
    }
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("frozen_protocol_sha256") != protocol["protocol_sha256"]:
            raise PermissionError("Existing test receipt belongs to a different frozen protocol")
        return existing
    atomic_json(receipt_path, receipt)
    return receipt


def _prepare_test_case(
    case: dict[str, Any],
    *,
    dataset_root: Path,
    schema: Any,
    preprocessing: Any,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    loaded = load_exclusive_case(dataset_root, case, schema, test_authorization=receipt)
    ras_hu, ras_label, ras_affine, _ = canonicalize_to_ras(loaded.ct_hu, loaded.label, loaded.affine)
    ct_hu, label, affine, _ = resample_pair(
        ras_hu,
        ras_label,
        ras_affine,
        preprocessing.spacing_mm,
        image_fill_hu=preprocessing.hu_lower,
    )
    return {
        "case_id": loaded.case_id,
        "ct_hu": np.ascontiguousarray(ct_hu, dtype=np.float32),
        "label": np.ascontiguousarray(label, dtype=np.uint8),
        "affine": np.ascontiguousarray(affine, dtype=np.float64),
    }


def _bootstrap_macro_dice(cases: list[dict[str, Any]], seed: int, replicates: int = 2000) -> dict[str, float | int]:
    raw_values = [case["segmentation"].get("mean_foreground_dice") for case in cases]
    values = np.asarray(
        [float(value) for value in raw_values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("Patient-level bootstrap requires at least one defined macro Dice")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        estimates[index] = np.mean(values[rng.integers(0, len(values), size=len(values))])
    return {
        "method": "patient_level_nonparametric_bootstrap",
        "empty_target_policy": "patients_with_no_selected_ground_truth_organs_excluded",
        "valid_patients": int(values.size),
        "excluded_patients": int(len(raw_values) - values.size),
        "replicates": replicates,
        "point_estimate": float(np.mean(values)),
        "ci95_lower": float(np.percentile(estimates, 2.5)),
        "ci95_upper": float(np.percentile(estimates, 97.5)),
    }


def run_locked_test_evaluation(
    *,
    project_root: Path,
    dataset_root: Path,
    phase6_config: dict[str, Any],
    baseline_config: BaselineConfig,
    manifest: dict[str, Any],
    schema: Any,
    preprocessing: Any,
    context: dict[str, Any],
    protocol: dict[str, Any],
    receipt: dict[str, Any],
    progress: Progress = print,
) -> dict[str, Any]:
    test_root = (project_root / phase6_config["outputs"]["test"]).resolve()
    report_path = test_root / "test_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("frozen_protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError("Completed test report belongs to another protocol")
        progress("Locked test evaluation already complete; verified and reused.")
        return report
    case_root, sample_root = test_root / "cases", test_root / "uncertainty_samples"
    case_root.mkdir(parents=True, exist_ok=True)
    sample_root.mkdir(exist_ok=True)
    records = sorted([case for case in manifest["cases"] if case["split"] == "test"], key=lambda item: item["case_id"])
    if len(records) != int(phase6_config["test"]["expected_cases"]):
        raise RuntimeError("Official test split count changed")
    class_names = {label.id: label.name for label in schema.labels}
    temperature = float(protocol["temperature"])
    threshold = float(protocol["error_flag_threshold"])
    policy = str(protocol["postprocessing_policy"])
    seed = int(phase6_config["reproducibility"]["random_seed"])
    sample_count = int(phase6_config["uncertainty"]["error_analysis_sample_voxels_per_case"])
    access_log_path = test_root / "test_access_log.json"
    access_log = {
        "schema_version": 1,
        "status": "running_or_resumable",
        "started_at_utc": utc_now(),
        "frozen_protocol_sha256": protocol["protocol_sha256"],
        "accessed_case_ids": [],
        "completed_case_ids": [],
    }
    if access_log_path.is_file():
        access_log = json.loads(access_log_path.read_text(encoding="utf-8"))
        if access_log.get("frozen_protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError("Test access log belongs to another protocol")
    atomic_json(access_log_path, access_log)

    mesh_case_id = protocol["mesh_selection"]["case_id"]
    for number, case in enumerate(records, start=1):
        case_id = case["case_id"]
        case_path, sample_path = case_root / f"{case_id}.json", sample_root / f"{case_id}.npz"
        if case_path.is_file() and sample_path.is_file():
            continue
        if case_id not in access_log["accessed_case_ids"]:
            access_log["accessed_case_ids"].append(case_id)
            atomic_json(access_log_path, access_log)
        prepared = _prepare_test_case(
            case,
            dataset_root=dataset_root,
            schema=schema,
            preprocessing=preprocessing,
            receipt=receipt,
        )
        logits = _infer_logits(
            normalize_ct(prepared["ct_hu"], preprocessing),
            model=context["model"],
            device=context["device"],
            config=baseline_config,
        )
        raw_prediction = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        prediction = apply_postprocessing(raw_prediction, policy=policy, spacing_mm=preprocessing.spacing_mm)
        entropy = _entropy_from_logits(logits, temperature)
        segmentation = case_segmentation_metrics(
            prediction,
            prepared["label"],
            class_names=class_names,
            spacing_mm=preprocessing.spacing_mm,
            compute_hd95=True,
        )
        indices = _sample_indices(prepared["label"].size, sample_count, seed=seed, case_id=case_id, purpose="locked_test")
        sampled_logits, sampled_labels = _sample_logits(logits, prepared["label"], indices)
        sampled_uncertainty = entropy.reshape(-1)[indices]
        sampled_error = (prediction.reshape(-1)[indices] != prepared["label"].reshape(-1)[indices]).astype(np.uint8)
        np.savez_compressed(
            sample_path,
            logits=sampled_logits.astype(np.float16),
            labels=sampled_labels,
            uncertainty=sampled_uncertainty.astype(np.float16),
            error=sampled_error,
        )
        atomic_json(
            case_path,
            {
                "case_id": case_id,
                "split": "test",
                "frozen_protocol_sha256": protocol["protocol_sha256"],
                "segmentation": segmentation,
            },
        )
        if case_id == mesh_case_id:
            np.savez_compressed(
                test_root / "predeclared_mesh_case_volume.npz",
                case_id=np.asarray(case_id),
                ct_hu=prepared["ct_hu"].astype(np.float16),
                prediction=prediction,
                uncertainty=entropy.astype(np.float16),
                affine=prepared["affine"],
            )
        if case_id not in access_log["completed_case_ids"]:
            access_log["completed_case_ids"].append(case_id)
        atomic_json(access_log_path, access_log)
        progress(f"  locked test {number:02d}/{len(records)}: {case_id}")
        del logits, entropy, prediction, raw_prediction, prepared

    case_payloads = [json.loads((case_root / f"{case['case_id']}.json").read_text(encoding="utf-8")) for case in records]
    sample_payloads = [np.load(sample_root / f"{case['case_id']}.npz") for case in records]
    logits = np.concatenate([item["logits"].astype(np.float32) for item in sample_payloads])
    labels = np.concatenate([item["labels"] for item in sample_payloads])
    uncertainty = np.concatenate([item["uncertainty"].astype(np.float32) for item in sample_payloads])
    errors = np.concatenate([item["error"] for item in sample_payloads])
    for item in sample_payloads:
        item.close()
    aggregate = aggregate_case_metrics(case_payloads, class_names)
    flags = uncertainty >= threshold
    true_error = errors.astype(bool)
    threshold_test = {
        "frozen_threshold": threshold,
        "precision": float(np.count_nonzero(flags & true_error) / max(1, np.count_nonzero(flags))),
        "recall": float(np.count_nonzero(flags & true_error) / max(1, np.count_nonzero(true_error))),
        "flagged_fraction": float(np.mean(flags)),
    }
    report = {
        "schema_version": 1,
        "status": "complete_one_shot",
        "completed_at_utc": utc_now(),
        "split": "test",
        "test_split_accessed": True,
        "cases": len(records),
        "frozen_protocol_sha256": protocol["protocol_sha256"],
        "checkpoint_sha256": context["checkpoint_sha256"],
        "temperature": temperature,
        "postprocessing_policy": policy,
        "segmentation": aggregate,
        "patient_level_macro_dice_ci95": _bootstrap_macro_dice(case_payloads, seed),
        "calibration_at_frozen_temperature": calibration_metrics(logits, labels, temperature=temperature),
        "uncertainty_error_analysis": uncertainty_error_metrics(uncertainty, errors),
        "frozen_error_threshold_performance": threshold_test,
        "mesh_case_id_predeclared_before_test": mesh_case_id,
        "no_test_time_tuning": True,
    }
    atomic_json(report_path, report)
    access_log["status"] = "complete"
    access_log["completed_at_utc"] = utc_now()
    atomic_json(access_log_path, access_log)
    return json.loads(report_path.read_text(encoding="utf-8"))
