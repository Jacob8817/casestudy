"""Resumable Phase 2 training and full-volume validation for segmentation + SDF."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from src.benchmark.brain_skull import dice_per_class, hd95_mm
from src.dataset.brain_skull_multitask import sample_multitask_patch
from src.losses.multitask import MultiTaskSegmentationSDFLoss, sdf_weight_for_epoch
from src.models.multitask_unet import MultiTaskUNet


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_phase2_model(config: dict) -> MultiTaskUNet:
    model = config["model"]
    return MultiTaskUNet(
        in_channels=int(model["in_channels"]),
        feature_channels=int(model["feature_channels"]),
        segmentation_channels=int(model["segmentation_channels"]),
        sdf_channels=int(model["sdf_channels"]),
        channels=tuple(int(value) for value in model["channels"]),
        strides=tuple(int(value) for value in model["strides"]),
        num_res_units=int(model["num_res_units"]),
        norm=str(model["normalization"]),
        act=str(model["activation"]),
        dropout=float(model["dropout"]),
    )


def predict_multitask(
    model: MultiTaskUNet,
    record: dict,
    *,
    device: torch.device,
    roi: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    prediction, sdf_array, _ = predict_multitask_outputs(
        model, record, device=device, roi=roi, overlap=overlap
    )
    return prediction, sdf_array


def predict_multitask_outputs(
    model: MultiTaskUNet,
    record: dict,
    *,
    device: torch.device,
    roi: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hard labels, SDF regression, and softmax probabilities."""
    image = np.asarray(np.load(record["image_path"], mmap_mode="r"), dtype=np.float32)
    tensor = torch.from_numpy(image[None]).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        segmentation, sdf = sliding_window_inference(
            inputs=tensor,
            roi_size=(roi, roi, roi),
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )
    probabilities = torch.softmax(segmentation.float(), dim=1)[0].cpu().numpy()
    prediction = probabilities.argmax(axis=0).astype(np.uint8, copy=False)
    sdf_array = sdf[0].float().cpu().numpy()
    del tensor, segmentation, sdf
    return prediction, sdf_array, probabilities


def evaluate_multitask(
    model: MultiTaskUNet,
    records: list[dict],
    *,
    device: torch.device,
    roi: int,
    overlap: float,
    include_hd95: bool,
) -> dict:
    model.eval()
    cases = []
    for record in records:
        started = time.perf_counter()
        prediction, predicted_sdf = predict_multitask(model, record, device=device, roi=roi, overlap=overlap)
        target = np.load(record["label_path"], mmap_mode="r")
        target_sdf = np.load(record["sdf_path"], mmap_mode="r")
        dice = dice_per_class(prediction, target)
        item = {
            "case_id": record["case_id"],
            "dice": dice,
            "sdf_mae_normalized": {
                "brain": float(np.mean(np.abs(predicted_sdf[0] - target_sdf[0]))),
                "skull": float(np.mean(np.abs(predicted_sdf[1] - target_sdf[1]))),
            },
            "inference_seconds": float(time.perf_counter() - started),
        }
        for class_index, organ in enumerate(("brain", "skull")):
            boundary = np.abs(target_sdf[class_index]) <= 0.2
            item.setdefault("boundary_sdf_mae_normalized", {})[organ] = float(
                np.mean(np.abs(predicted_sdf[class_index][boundary] - target_sdf[class_index][boundary]))
            )
        if include_hd95:
            spacing = tuple(float(value) for value in record["spacing_mm"])
            item["hd95_mm"] = {
                "brain": hd95_mm(prediction == 1, target == 1, spacing),
                "skull": hd95_mm(prediction == 2, target == 2, spacing),
            }
        cases.append(item)
    mean_dice = {
        organ: float(np.mean([item["dice"][organ] for item in cases])) for organ in ("brain", "skull")
    }
    result = {
        "cases": cases,
        "mean_dice": mean_dice,
        "macro_mean_dice": float(np.mean(list(mean_dice.values()))),
        "mean_sdf_mae_normalized": {
            organ: float(np.mean([item["sdf_mae_normalized"][organ] for item in cases]))
            for organ in ("brain", "skull")
        },
        "mean_boundary_sdf_mae_normalized": {
            organ: float(np.mean([item["boundary_sdf_mae_normalized"][organ] for item in cases]))
            for organ in ("brain", "skull")
        },
        "mean_inference_seconds": float(np.mean([item["inference_seconds"] for item in cases])),
    }
    if include_hd95:
        result["mean_hd95_mm"] = {
            organ: float(np.mean([item["hd95_mm"][organ] for item in cases]))
            for organ in ("brain", "skull")
        }
    return result


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_phase2(
    records: list[dict],
    config: dict,
    output_root: Path,
    *,
    phase1_contract_sha256: str,
    config_sha256: str,
    progress=print,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2 training requires CUDA")
    train_records = [item for item in records if item["split"] == "train"]
    val_records = [item for item in records if item["split"] == "val"]
    if len(train_records) != 98 or len(val_records) != 5:
        raise ValueError("Phase 2 requires the frozen 98/5 cohort")
    training = config["training"]
    loss_config = config["loss"]
    data_config = config["data"]
    inference = config["inference"]
    epochs = int(training["epochs"])
    validation_interval = int(training["validation_interval_epochs"])
    roi = int(data_config["roi_size"][0])
    if tuple(data_config["roi_size"]) != (roi, roi, roi):
        raise ValueError("Current Phase 2 implementation requires a cubic ROI")
    seed = int(training["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = build_phase2_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    sdf_loss_config = loss_config["sdf"]
    objective = MultiTaskSegmentationSDFLoss(
        boundary_band=float(sdf_loss_config["boundary_band_normalized"]),
        boundary_multiplier=float(sdf_loss_config["boundary_multiplier"]),
    )
    contract = {
        "experiment_id": config["experiment"]["id"],
        "phase1_contract_sha256": phase1_contract_sha256,
        "config_sha256": config_sha256,
        "train_case_ids": [item["case_id"] for item in train_records],
        "validation_case_ids": [item["case_id"] for item in val_records],
        "model": config["model"],
        "data": config["data"],
        "sdf": config["sdf"],
        "loss": config["loss"],
        "training": config["training"],
    }
    contract_sha256 = canonical_sha256(contract)
    best_path, last_path = output_root / "best.pt", output_root / "last.pt"
    history_path = output_root / "history.json"
    output_root.mkdir(parents=True, exist_ok=True)
    history, start_epoch, best_score = [], 1, -1.0
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["contract_sha256"] != contract_sha256:
            raise ValueError("Existing Phase 2 checkpoint has a different frozen contract")
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
        order = rng.permutation(len(train_records))
        losses, segmentation_losses, sdf_losses = [], [], []
        modes = {"foreground_brain": 0, "foreground_skull": 0, "ct_context_background": 0}
        weight = sdf_weight_for_epoch(
            epoch,
            epochs,
            early=float(sdf_loss_config["early_weight"]),
            middle=float(sdf_loss_config["middle_weight"]),
            late=float(sdf_loss_config["late_weight"]),
            early_fraction=float(sdf_loss_config["early_fraction"]),
            middle_fraction=float(sdf_loss_config["middle_fraction"]),
        )
        for index in order:
            image, label, target_sdf, mode = sample_multitask_patch(
                train_records[int(index)],
                rng,
                roi=roi,
                foreground_probability=float(data_config["foreground_patch_probability"]),
            )
            modes[mode] += 1
            inputs = torch.from_numpy(image[None]).to(device, non_blocking=True)
            labels = torch.from_numpy(label[None, None]).to(device, non_blocking=True)
            sdf_target = torch.from_numpy(target_sdf[None]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                segmentation_logits, predicted_sdf = model(inputs)
                loss, parts = objective(
                    segmentation_logits,
                    predicted_sdf,
                    labels,
                    sdf_target,
                    sdf_weight=weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Phase 2 loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            segmentation_losses.append(float(parts["segmentation"].detach().cpu()))
            sdf_losses.append(float(parts["sdf"].detach().cpu()))
            del inputs, labels, sdf_target, segmentation_logits, predicted_sdf, loss

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
            "segmentation_loss": float(np.mean(segmentation_losses)),
            "sdf_loss": float(np.mean(sdf_losses)),
            "sdf_weight": weight,
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
            f"Phase 2 epoch {epoch}/{epochs}: loss={record['total_loss']:.4f}, "
            f"SDF w={weight:.2f}{validation_text}, {record['elapsed_seconds']:.1f}s"
        )

    if not best_path.is_file():
        raise RuntimeError("No Phase 2 best checkpoint was selected")
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
        "phase": 2,
        "selected_epoch": int(best["epoch"]),
        "completed_epochs": len(history),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_cases": len(train_records),
        "validation_cases": len(val_records),
        "validation": detailed,
        "total_training_seconds": float(sum(item["elapsed_seconds"] for item in history)),
        "peak_gpu_memory_mb": float(max(item["peak_gpu_memory_mb"] for item in history)),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "contract_sha256": contract_sha256,
        "official_test_split_accessed": False,
    }
    (output_root / "phase2_model_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
