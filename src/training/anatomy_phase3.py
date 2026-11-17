"""Phase 3 anatomy-aware fine-tuning and independent validation QC."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.anatomy.constraints import AnatomyRelationalLoss
from src.anatomy.violations import anatomy_violation_map
from src.dataset.brain_skull_multitask import sample_anatomy_patch
from src.losses.multitask import MultiTaskSegmentationSDFLoss, sdf_weight_for_epoch
from src.training.anatomy_phase2 import (
    build_phase2_model,
    canonical_sha256,
    evaluate_multitask,
    predict_multitask_outputs,
    sha256_file,
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_phase3(
    records: list[dict],
    phase2_config: dict,
    phase3_config: dict,
    output_root: Path,
    *,
    phase2_checkpoint: Path,
    phase3_config_sha256: str,
    progress=print,
) -> dict:
    """Fine-tune the Phase 2 model with a scheduled relational anatomy loss."""
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 training requires CUDA")
    train_records = [item for item in records if item["split"] == "train"]
    val_records = [item for item in records if item["split"] == "val"]
    if len(train_records) != 98 or len(val_records) != 5:
        raise ValueError("Phase 3 requires the frozen 98/5 development cohort")
    if any("enclosure_path" not in item or "brain_core_path" not in item for item in records):
        raise ValueError("Phase 3 anatomy targets have not been prepared")

    training = phase3_config["training"]
    anatomy_config = phase3_config["loss"]
    base_loss_config = phase2_config["loss"]["sdf"]
    data_config = phase2_config["data"]
    inference = phase2_config["inference"]
    epochs = int(training["epochs"])
    validation_interval = int(training["validation_interval_epochs"])
    roi = int(data_config["roi_size"][0])
    seed = int(training["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")

    phase2_saved = torch.load(phase2_checkpoint, map_location="cpu", weights_only=True)
    model = build_phase2_model(phase2_config).to(device)
    model.load_state_dict(phase2_saved["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    base_objective = MultiTaskSegmentationSDFLoss(
        boundary_band=float(base_loss_config["boundary_band_normalized"]),
        boundary_multiplier=float(base_loss_config["boundary_multiplier"]),
    )
    anatomy_objective = AnatomyRelationalLoss(
        brain_outside_enclosure_weight=float(anatomy_config["brain_outside_enclosure_weight"]),
        skull_in_brain_core_weight=float(anatomy_config["skull_in_brain_core_weight"]),
        probability_ambiguity_weight=float(anatomy_config["probability_ambiguity_weight"]),
        sdf_sign_consistency_weight=float(anatomy_config["sdf_sign_consistency_weight"]),
    )
    contract = {
        "experiment_id": phase3_config["experiment"]["id"],
        "phase2_checkpoint_sha256": sha256_file(phase2_checkpoint),
        "phase2_contract_sha256": phase2_saved["contract_sha256"],
        "phase3_config_sha256": phase3_config_sha256,
        "train_case_ids": [item["case_id"] for item in train_records],
        "validation_case_ids": [item["case_id"] for item in val_records],
        "model": phase2_config["model"],
        "data": phase2_config["data"],
        "base_loss": phase2_config["loss"],
        "phase3": phase3_config,
    }
    contract_sha256 = canonical_sha256(contract)
    best_path, last_path = output_root / "best.pt", output_root / "last.pt"
    history_path = output_root / "history.json"
    output_root.mkdir(parents=True, exist_ok=True)
    history, start_epoch, best_score = [], 1, -1.0
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["contract_sha256"] != contract_sha256:
            raise ValueError("Existing Phase 3 checkpoint has a different frozen contract")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved["best_score"])
        history = json.loads(history_path.read_text(encoding="utf-8"))

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        rng = np.random.default_rng(seed + epoch)
        losses, base_losses, anatomy_losses = [], [], []
        anatomy_parts: dict[str, list[float]] = {
            "brain_outside_enclosure": [],
            "skull_in_brain_core": [],
            "probability_ambiguity": [],
            "sdf_sign_consistency": [],
        }
        modes = {"foreground_brain": 0, "foreground_skull": 0, "ct_context_background": 0}
        sdf_weight = float(base_loss_config["late_weight"])
        anatomy_weight = sdf_weight_for_epoch(
            epoch,
            epochs,
            early=float(anatomy_config["early_weight"]),
            middle=float(anatomy_config["middle_weight"]),
            late=float(anatomy_config["late_weight"]),
            early_fraction=float(anatomy_config["early_fraction"]),
            middle_fraction=float(anatomy_config["middle_fraction"]),
        )
        for index in rng.permutation(len(train_records)):
            image, label, target_sdf, enclosure, brain_core, mode = sample_anatomy_patch(
                train_records[int(index)],
                rng,
                roi=roi,
                foreground_probability=float(data_config["foreground_patch_probability"]),
            )
            modes[mode] += 1
            inputs = torch.from_numpy(image[None]).to(device, non_blocking=True)
            labels = torch.from_numpy(label[None, None]).to(device, non_blocking=True)
            sdf_target = torch.from_numpy(target_sdf[None]).to(device, non_blocking=True)
            enclosure_target = torch.from_numpy(enclosure[None, None]).to(device, non_blocking=True)
            core_target = torch.from_numpy(brain_core[None, None]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                segmentation_logits, predicted_sdf = model(inputs)
                base_loss, base_parts = base_objective(
                    segmentation_logits,
                    predicted_sdf,
                    labels,
                    sdf_target,
                    sdf_weight=sdf_weight,
                )
                anatomy_loss, relation_parts = anatomy_objective(
                    segmentation_logits, predicted_sdf, enclosure_target, core_target
                )
                loss = base_loss + anatomy_weight * anatomy_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Phase 3 loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            base_losses.append(float(base_loss.detach().cpu()))
            anatomy_losses.append(float(anatomy_loss.detach().cpu()))
            for name, value in relation_parts.items():
                anatomy_parts[name].append(float(value.detach().cpu()))
            del inputs, labels, sdf_target, enclosure_target, core_target
            del segmentation_logits, predicted_sdf, base_loss, anatomy_loss, loss, base_parts, relation_parts

        validation = None
        improved = False
        if epoch % validation_interval == 0 or epoch == epochs:
            validation = evaluate_multitask(
                model,
                val_records,
                device=device,
                roi=roi,
                overlap=float(inference["overlap"]),
                include_hd95=False,
            )
            score = float(validation["macro_mean_dice"])
            improved = score > best_score
            if improved:
                best_score = score
        checkpoint = {
            "format_version": 1,
            "contract": contract,
            "contract_sha256": contract_sha256,
            "epoch": epoch,
            "best_score": best_score,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "validation": validation,
        }
        _atomic_torch_save(checkpoint, last_path)
        if improved:
            _atomic_torch_save(checkpoint, best_path)
        record = {
            "epoch": epoch,
            "total_loss": float(np.mean(losses)),
            "base_multitask_loss": float(np.mean(base_losses)),
            "anatomy_loss": float(np.mean(anatomy_losses)),
            "anatomy_terms": {name: float(np.mean(values)) for name, values in anatomy_parts.items()},
            "sdf_weight": sdf_weight,
            "anatomy_weight": anatomy_weight,
            "sampling_modes": modes,
            "validation_macro_dice": None if validation is None else validation["macro_mean_dice"],
            "validation_brain_dice": None if validation is None else validation["mean_dice"]["brain"],
            "validation_skull_dice": None if validation is None else validation["mean_dice"]["skull"],
            "best": improved,
            "elapsed_seconds": float(time.perf_counter() - epoch_started),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        validation_text = "" if validation is None else f", val macro Dice={validation['macro_mean_dice']:.4f}"
        progress(
            f"Phase 3 epoch {epoch}/{epochs}: loss={record['total_loss']:.4f}, "
            f"anatomy w={anatomy_weight:.2f}{validation_text}, {record['elapsed_seconds']:.1f}s"
        )

    if not best_path.is_file():
        raise RuntimeError("No Phase 3 best checkpoint was selected")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    detailed = evaluate_multitask(
        model,
        val_records,
        device=device,
        roi=roi,
        overlap=float(inference["overlap"]),
        include_hd95=True,
    )
    report = {
        "schema_version": 1,
        "status": "trained",
        "phase": 3,
        "selected_epoch": int(best["epoch"]),
        "completed_epochs": len(history),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_cases": len(train_records),
        "validation_cases": len(val_records),
        "validation": detailed,
        "total_training_seconds": float(sum(item["elapsed_seconds"] for item in history)),
        "peak_gpu_memory_mb": float(max(item["peak_gpu_memory_mb"] for item in history)),
        "initial_checkpoint": str(phase2_checkpoint),
        "initial_checkpoint_sha256": sha256_file(phase2_checkpoint),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "contract_sha256": contract_sha256,
        "official_test_split_accessed": False,
    }
    (output_root / "phase3_model_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_anatomy_qc(
    model: torch.nn.Module,
    records: list[dict],
    distance_reference: dict,
    qc_config: dict,
    *,
    device: torch.device,
    roi: int,
    overlap: float,
    output_root: Path,
) -> dict:
    """Measure and save V(x) without changing any model prediction."""
    if any(item["split"] != "val" for item in records):
        raise ValueError("Phase 3 QC comparison accepts validation records only")
    output_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    cases = []
    for record in records:
        prediction, predicted_sdf, probabilities = predict_multitask_outputs(
            model, record, device=device, roi=roi, overlap=overlap
        )
        violation, components, summary = anatomy_violation_map(
            prediction,
            probabilities,
            predicted_sdf,
            tuple(float(value) for value in record["spacing_mm"]),
            distance_reference,
            minimum_component_ml=float(qc_config["disconnected_minimum_ml"]),
            minimum_largest_fraction=float(qc_config["disconnected_minimum_largest_fraction"]),
            weights={name: float(value) for name, value in qc_config["violation_weights"].items()},
        )
        case_root = output_root / record["case_id"]
        case_root.mkdir(parents=True, exist_ok=True)
        affine = np.asarray(record["ct_affine"], dtype=np.float64)
        nib.save(nib.Nifti1Image(violation, affine), str(case_root / "anatomical_violation.nii.gz"))
        np.savez_compressed(case_root / "violation_components.npz", **components)
        cases.append({"case_id": record["case_id"], **summary})
    scalar_keys = (
        "brain_outside_enclosure_fraction",
        "surface_distance_mean_mm",
        "surface_distance_max_mm",
        "probability_ambiguity_mean",
        "sdf_sign_disagreement_fraction",
        "violation_mean",
        "violation_nonzero_fraction",
        "violation_above_0_05_fraction",
    )
    means = {}
    for key in scalar_keys:
        values = [item[key] for item in cases if item[key] is not None]
        means[key] = float(np.mean(values)) if values else None
    means["disconnected_major_components_brain"] = float(
        np.mean([item["disconnected_major_components"]["brain"] for item in cases])
    )
    means["disconnected_major_components_skull"] = float(
        np.mean([item["disconnected_major_components"]["skull"] for item in cases])
    )
    result = {
        "cases": cases,
        "means": means,
        "prediction_was_modified": False,
        "official_test_split_accessed": False,
    }
    (output_root / "qc_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
