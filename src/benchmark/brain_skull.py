"""Brain/skull CT benchmark helpers. The official test split is never opened."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


LABELS = {0: "background", 1: "brain", 2: "skull"}
INPUT_CHANNELS = ("brain_window_-100_200_HU", "bone_window_-300_1500_HU")
ORIGINAL_INPUT_CHANNEL = "original_window_-1000_1000_HU"
ORIGINAL_TASKS = {
    "original_monai_brain_only": "brain_only",
    "original_monai_skull_only": "skull_only",
    "original_monai_brain_skull": "joint",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_cases(manifest_path: Path) -> list[dict]:
    """Return only the official train/val cases, rejecting unexpected split values."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    allowed = {"train", "val", "test"}
    if any(case["split"] not in allowed for case in cases):
        raise ValueError("Manifest contains an unexpected split")
    return [case for case in cases if case["split"] in {"train", "val"}]


def _binary_mask(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    if values.ndim != 3 or not np.all(np.isin(np.unique(values), (0, 1))):
        raise ValueError(f"Expected a binary 3D mask: {path}")
    affine = np.asarray(image.affine, dtype=np.float64)
    voxel_ml = float(abs(np.linalg.det(affine[:3, :3])) / 1000.0)
    return values.astype(bool), affine, voxel_ml


def scan_development_inventory(
    dataset_root: Path,
    manifest_path: Path,
    *,
    min_brain_ml: float = 800.0,
    min_skull_ml: float = 300.0,
) -> dict:
    """Find complete-head train/val cases without resolving a test-case path."""
    selected: list[dict] = []
    scanned = {"train": 0, "val": 0}
    for case in development_cases(manifest_path):
        case_id = case["case_id"]
        split = case["split"]
        scanned[split] += 1
        case_root = dataset_root / case_id
        brain, brain_affine, voxel_ml = _binary_mask(case_root / "segmentations" / "brain.nii.gz")
        brain_ml = float(brain.sum() * voxel_ml)
        if brain_ml < min_brain_ml:
            continue
        skull, skull_affine, skull_voxel_ml = _binary_mask(case_root / "segmentations" / "skull.nii.gz")
        skull_ml = float(skull.sum() * skull_voxel_ml)
        if skull_ml < min_skull_ml:
            continue
        if brain.shape != skull.shape or not np.allclose(brain_affine, skull_affine, atol=1e-3, rtol=0):
            raise ValueError(f"Brain/skull geometry differs for {case_id}")
        overlap = int(np.count_nonzero(brain & skull))
        selected.append(
            {
                "case_id": case_id,
                "split": split,
                "brain_ml": brain_ml,
                "skull_ml": skull_ml,
                "overlap_voxels_skull_wins": overlap,
                "shape": list(brain.shape),
                "ct_path": case["ct"]["path"],
            }
        )
    selected.sort(key=lambda item: (item["split"], item["case_id"]))
    if not any(item["split"] == "train" for item in selected):
        raise ValueError("No complete-head training cases found")
    if not any(item["split"] == "val" for item in selected):
        raise ValueError("No complete-head validation cases found")
    return {
        "schema_version": 1,
        "selection": "official train/val only; brain >= 800 mL and skull >= 300 mL",
        "overlap_policy": "skull wins; brain is removed where the two source masks overlap",
        "min_brain_ml": min_brain_ml,
        "min_skull_ml": min_skull_ml,
        "scanned": scanned,
        "selected_counts": {
            split: sum(item["split"] == split for item in selected) for split in ("train", "val")
        },
        "cases": selected,
        "test_split_accessed": False,
    }


def load_or_create_inventory(dataset_root: Path, manifest_path: Path, output_path: Path) -> dict:
    manifest_hash = sha256_file(manifest_path)
    if output_path.is_file():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if saved.get("manifest_file_sha256") == manifest_hash:
            return saved
        raise ValueError("Existing benchmark inventory belongs to a different manifest")
    inventory = scan_development_inventory(dataset_root, manifest_path)
    inventory["manifest_file_sha256"] = manifest_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return inventory


def _bounding_box(mask: np.ndarray, margin: int = 16) -> tuple[slice, slice, slice]:
    locations = np.where(mask)
    return tuple(
        slice(max(0, int(axis.min()) - margin), min(mask.shape[index], int(axis.max()) + margin + 1))
        for index, axis in enumerate(locations)
    )


def prepare_case(dataset_root: Path, record: dict, cache_root: Path) -> dict:
    """Cache two CT windows and exclusive labels on a head crop; never alter source files."""
    if record["split"] not in {"train", "val"}:
        raise ValueError("Benchmark cache refuses non-development cases")
    case_id = record["case_id"]
    case_root = dataset_root / case_id
    image_path = cache_root / f"{case_id}_image.npy"
    label_path = cache_root / f"{case_id}_label.npy"
    meta_path = cache_root / f"{case_id}_meta.json"
    if image_path.is_file() and label_path.is_file() and meta_path.is_file():
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if "class_bounds" in cached and "ct_affine" in cached:
            return cached

    brain, brain_affine, _ = _binary_mask(case_root / "segmentations" / "brain.nii.gz")
    skull, skull_affine, _ = _binary_mask(case_root / "segmentations" / "skull.nii.gz")
    ct_image = nib.load(str(dataset_root / record["ct_path"]))
    if (
        brain.shape != skull.shape
        or brain.shape != ct_image.shape
        or not np.allclose(brain_affine, skull_affine, atol=1e-3, rtol=0)
        or not np.allclose(brain_affine, ct_image.affine, atol=1e-3, rtol=0)
    ):
        raise ValueError(f"CT and masks do not share a grid: {case_id}")
    crop = _bounding_box(brain | skull)
    label = np.zeros(tuple(item.stop - item.start for item in crop), dtype=np.uint8)
    label[brain[crop]] = 1
    label[skull[crop]] = 2  # Explicit skull-wins overlap policy.
    hu = np.asarray(ct_image.dataobj[crop], dtype=np.float32)
    if not np.isfinite(hu).all():
        raise ValueError(f"Non-finite CT values: {case_id}")
    image = np.stack(
        [
            np.clip((hu + 100.0) / 300.0, 0.0, 1.0),
            np.clip((hu + 300.0) / 1800.0, 0.0, 1.0),
        ],
        axis=0,
    ).astype(np.float16)
    cache_root.mkdir(parents=True, exist_ok=True)
    # A partial cache is never considered valid unless its metadata exists.
    image_temp = image_path.with_suffix(".tmp.npy")
    label_temp = label_path.with_suffix(".tmp.npy")
    np.save(image_temp, image)
    np.save(label_temp, label)
    os.replace(image_temp, image_path)
    os.replace(label_temp, label_path)
    metadata = {
        "case_id": case_id,
        "split": record["split"],
        "image_path": str(image_path),
        "label_path": str(label_path),
        "shape": list(label.shape),
        "label_voxels": {str(class_id): int((label == class_id).sum()) for class_id in (1, 2)},
        "class_bounds": {
            str(class_id): [
                [int(np.where(label == class_id)[axis].min()), int(np.where(label == class_id)[axis].max()) + 1]
                for axis in range(3)
            ]
            for class_id in (1, 2)
        },
        "crop_start": [item.start for item in crop],
        "ct_affine": np.asarray(ct_image.affine, dtype=np.float64).tolist(),
        "input_channels": list(INPUT_CHANNELS),
        "source_ct_sha256": sha256_file(dataset_root / record["ct_path"]),
    }
    meta_temp = meta_path.with_suffix(".tmp")
    meta_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    meta_temp.replace(meta_path)
    return metadata


def prepare_inventory(dataset_root: Path, inventory: dict, cache_root: Path) -> list[dict]:
    return [prepare_case(dataset_root, record, cache_root) for record in inventory["cases"]]


def prepare_original_inputs(dataset_root: Path, prepared_records: list[dict], cache_root: Path) -> list[dict]:
    """Add the prior 10-organ U-Net's single fixed-HU CT channel on the same frozen crops."""
    result = []
    for record in prepared_records:
        if record["split"] not in {"train", "val"}:
            raise ValueError("Original MONAI benchmark refuses non-development cases")
        case_id = record["case_id"]
        output_path = cache_root / f"{case_id}_original_image.npy"
        if not output_path.is_file():
            ct_path = dataset_root / case_id / "ct.nii.gz"
            ct_image = nib.load(str(ct_path))
            start = record["crop_start"]
            shape = record["shape"]
            crop = tuple(slice(start[axis], start[axis] + shape[axis]) for axis in range(3))
            hu = np.asarray(ct_image.dataobj[crop], dtype=np.float32)
            if not np.isfinite(hu).all():
                raise ValueError(f"Non-finite CT values: {case_id}")
            image = np.clip((hu + 1000.0) / 2000.0, 0.0, 1.0)[None].astype(np.float16)
            temporary = output_path.with_suffix(".tmp.npy")
            np.save(temporary, image)
            os.replace(temporary, output_path)
        updated = dict(record)
        updated["original_image_path"] = str(output_path)
        result.append(updated)
    return result


def sample_patch(
    record: dict,
    rng: np.random.Generator,
    roi: int = 96,
    *,
    image_key: str = "image_path",
    task: str = "joint",
    sampling_targets: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.load(record[image_key], mmap_mode="r")
    label = np.load(record["label_path"], mmap_mode="r")
    if task not in {"brain_only", "skull_only", "joint"}:
        raise ValueError(f"Unknown task: {task}")
    candidate_targets = (
        list(sampling_targets)
        if sampling_targets is not None
        else {"brain_only": [1], "skull_only": [2], "joint": [1, 2]}[task]
    )
    if not candidate_targets or any(value not in (1, 2) for value in candidate_targets):
        raise ValueError("Sampling targets must be a non-empty subset of brain/skull")
    target = int(rng.choice(candidate_targets))
    bounds = record["class_bounds"][str(target)]
    for _ in range(100):
        center = [int(rng.integers(lower, upper)) for lower, upper in bounds]
        if label[tuple(center)] == target:
            break
    else:
        coords = np.argwhere(label == target)
        center = coords[int(rng.integers(0, len(coords)))].tolist()
    slices = []
    padding = []
    for coordinate, dimension in zip(center, label.shape):
        desired_start = coordinate - roi // 2
        desired_end = desired_start + roi
        start = max(0, desired_start)
        end = min(dimension, desired_end)
        slices.append(slice(start, end))
        padding.append((max(0, -desired_start), max(0, desired_end - dimension)))
    image_patch = np.pad(np.asarray(image[(slice(None), *slices)], dtype=np.float32), [(0, 0), *padding])
    label_patch = np.pad(np.asarray(label[tuple(slices)], dtype=np.int64), padding)
    for axis in range(3):
        if rng.random() < 0.5:
            image_patch = np.flip(image_patch, axis=axis + 1)
            label_patch = np.flip(label_patch, axis=axis)
    if task == "brain_only":
        label_patch = (label_patch == 1).astype(np.int64)
    elif task == "skull_only":
        label_patch = (label_patch == 2).astype(np.int64)
    return np.ascontiguousarray(image_patch), np.ascontiguousarray(label_patch)


def dice_per_class(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scores = {}
    for class_id in (1, 2):
        predicted = prediction == class_id
        truth = target == class_id
        denominator = int(predicted.sum() + truth.sum())
        scores[LABELS[class_id]] = float(2 * np.count_nonzero(predicted & truth) / denominator) if denominator else float("nan")
    return scores


def hd95_mm(prediction: np.ndarray, target: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    if not prediction.any() or not target.any():
        return float("nan")
    predicted_surface = prediction & ~binary_erosion(prediction)
    target_surface = target & ~binary_erosion(target)
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing_mm)
    distance_to_predicted = distance_transform_edt(~predicted_surface, sampling=spacing_mm)
    distances = np.concatenate((distance_to_target[predicted_surface], distance_to_predicted[target_surface]))
    return float(np.percentile(distances, 95))


def make_model(name: str):
    from monai.networks.nets import SegResNet, SwinUNETR, UNet

    if name == "unet3d":
        return UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
        )
    if name == "segresnet3d":
        return SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.0,
        )
    if name == "swinunetr3d":
        return SwinUNETR(
            in_channels=2,
            out_channels=3,
            feature_size=12,
            use_checkpoint=True,
            spatial_dims=3,
        )
    if name in ORIGINAL_TASKS:
        output_channels = 3 if ORIGINAL_TASKS[name] == "joint" else 2
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=output_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
        )
    raise ValueError(f"Unknown benchmark model: {name}")


def _atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def predict_case(
    model,
    record: dict,
    *,
    device,
    roi: int = 96,
    image_key: str = "image_path",
) -> np.ndarray:
    import torch
    from monai.inferers import sliding_window_inference

    image = np.asarray(np.load(record[image_key], mmap_mode="r"), dtype=np.float32)
    tensor = torch.from_numpy(image[None]).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=(roi, roi, roi),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            mode="gaussian",
        )
    prediction = logits.argmax(dim=1)[0].to(dtype=torch.uint8).cpu().numpy()
    del tensor, logits
    return prediction


def _target_for_task(label: np.ndarray, task: str) -> np.ndarray:
    if task == "brain_only":
        return np.asarray(label == 1, dtype=np.uint8)
    if task == "skull_only":
        return np.asarray(label == 2, dtype=np.uint8)
    if task == "joint":
        return np.asarray(label, dtype=np.uint8)
    raise ValueError(f"Unknown task: {task}")


def evaluate_model(
    model,
    val_records: list[dict],
    *,
    device,
    roi: int = 96,
    include_hd95: bool = False,
    task: str = "joint",
    image_key: str = "image_path",
) -> dict:
    model.eval()
    cases = []
    for record in val_records:
        prediction = predict_case(model, record, device=device, roi=roi, image_key=image_key)
        target = _target_for_task(np.load(record["label_path"], mmap_mode="r"), task)
        if task == "joint":
            dice = dice_per_class(prediction, target)
            classes = {1: "brain", 2: "skull"}
        else:
            organ = "brain" if task == "brain_only" else "skull"
            predicted = prediction == 1
            truth = target == 1
            denominator = int(predicted.sum() + truth.sum())
            dice = {organ: float(2 * np.count_nonzero(predicted & truth) / denominator) if denominator else float("nan")}
            classes = {1: organ}
        item = {"case_id": record["case_id"], "dice": dice}
        if include_hd95:
            affine = np.asarray(record["ct_affine"], dtype=np.float64)
            spacing = tuple(float(np.linalg.norm(affine[:3, index])) for index in range(3))
            item["hd95_mm"] = {
                organ: hd95_mm(prediction == class_id, target == class_id, spacing)
                for class_id, organ in classes.items()
            }
        cases.append(item)
    means = {
        organ: float(np.nanmean([item["dice"][organ] for item in cases]))
        for organ in classes.values()
    }
    report = {
        "cases": cases,
        "mean_dice": means,
        "macro_mean_dice": float(np.mean(list(means.values()))),
    }
    if include_hd95:
        report["mean_hd95_mm"] = {
            organ: float(np.nanmean([item["hd95_mm"][organ] for item in cases]))
            for organ in classes.values()
        }
    return report


def threshold_skull_baseline(val_records: list[dict]) -> dict:
    """A simple >300 HU threshold; intentionally has no brain output."""
    cases = []
    for record in val_records:
        bone_window = np.load(record["image_path"], mmap_mode="r")[1]
        truth = np.load(record["label_path"], mmap_mode="r") == 2
        prediction = bone_window > (600.0 / 1800.0)
        denominator = int(prediction.sum() + truth.sum())
        dice = float(2 * np.count_nonzero(prediction & truth) / denominator) if denominator else float("nan")
        cases.append({"case_id": record["case_id"], "skull_dice": dice})
    return {
        "method": "threshold_skull_gt_300_HU",
        "brain_dice": None,
        "skull_dice": float(np.nanmean([item["skull_dice"] for item in cases])),
        "cases": cases,
    }


def train_model(
    name: str,
    prepared_records: list[dict],
    output_root: Path,
    *,
    inventory_sha256: str,
    max_epochs: int = 8,
    roi: int = 96,
    learning_rate: float = 2e-4,
    seed: int = 42,
) -> dict:
    """Train/resume one model, checkpoint each epoch, and select by validation macro Dice."""
    import time

    import torch
    from monai.losses import DiceCELoss

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark training requires CUDA; refusing silent CPU fallback")
    train_records = [item for item in prepared_records if item["split"] == "train"]
    val_records = [item for item in prepared_records if item["split"] == "val"]
    if not train_records or not val_records:
        raise ValueError("Both train and validation cases are required")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = make_model(name).cuda()
    task = ORIGINAL_TASKS.get(name, "joint")
    image_key = "original_image_path" if name in ORIGINAL_TASKS else "image_path"
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    run_root = output_root / name
    last_path = run_root / "last.pt"
    best_path = run_root / "best.pt"
    history_path = run_root / "history.json"
    run_root.mkdir(parents=True, exist_ok=True)
    output_classes = (
        {0: "background", 1: "brain"}
        if task == "brain_only"
        else {0: "background", 1: "skull"}
        if task == "skull_only"
        else LABELS
    )
    contract = {
        "model": name,
        "input_channels": [ORIGINAL_INPUT_CHANNEL] if name in ORIGINAL_TASKS else list(INPUT_CHANNELS),
        "output_classes": output_classes,
        "inventory_sha256": inventory_sha256,
        "roi": roi,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    if name in ORIGINAL_TASKS:
        contract["sampling_policy"] = "matched_brain_or_skull_centers_for_all_three_tasks"
    contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    start_epoch = 1
    best_score = -1.0
    history = []
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["contract_sha256"] != contract_sha256:
            raise ValueError(f"Existing {name} checkpoint has a different data/model contract")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved["best_score"])
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []

    for epoch in range(start_epoch, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(train_records))
        losses = []
        for index in order:
            image, label = sample_patch(
                train_records[int(index)], rng, roi=roi, image_key=image_key, task=task,
                sampling_targets=(1, 2) if name in ORIGINAL_TASKS else None,
            )
            inputs = torch.from_numpy(image[None]).cuda(non_blocking=True)
            target = torch.from_numpy(label[None, None]).cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = loss_function(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss for {name}, epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            del inputs, target, logits, loss
        validation = evaluate_model(
            model, val_records, device=torch.device("cuda"), roi=roi, task=task, image_key=image_key
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
            "validation": validation,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        }
        _atomic_torch_save(checkpoint, last_path)
        if improved:
            _atomic_torch_save(checkpoint, best_path)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_brain_dice": validation["mean_dice"].get("brain"),
            "validation_skull_dice": validation["mean_dice"].get("skull"),
            "validation_macro_dice": score,
            "best": improved,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"{name} epoch {epoch}/{max_epochs}: loss={record['train_loss']:.4f}, val macro Dice={score:.4f}, {record['elapsed_seconds']:.1f}s", flush=True)

    if not best_path.is_file():
        raise RuntimeError(f"Missing best checkpoint for {name}")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    detailed = evaluate_model(
        model,
        val_records,
        device=torch.device("cuda"),
        roi=roi,
        include_hd95=True,
        task=task,
        image_key=image_key,
    )
    report = {
        "model": name,
        "task": task,
        "status": "trained",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "selected_epoch": int(best["epoch"]),
        "completed_epochs": len(history),
        "total_epoch_seconds": float(sum(item["elapsed_seconds"] for item in history)),
        "train_cases": len(train_records),
        "validation_cases": len(val_records),
        "validation": detailed,
        "best_checkpoint": str(best_path),
        "best_checkpoint_bytes": best_path.stat().st_size,
        "best_checkpoint_sha256": sha256_file(best_path),
        "contract_sha256": contract_sha256,
        "test_split_accessed": False,
    }
    (run_root / "benchmark_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
