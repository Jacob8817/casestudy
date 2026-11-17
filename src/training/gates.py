"""Train-only diagnostic gates used before baseline model selection.

The overfit gate deliberately has no validation-loader argument.  It memorizes
a fixed, audited patch bank from one official training case and therefore
cannot be mistaken for validation evidence or touch the locked test split.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from monai.utils import set_determinism
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.dataset import OrganSchema, PreprocessingSpec

from .config import BaselineConfig, validate_baseline_contract
from .data import ExactSplitPatchDataset
from .engine import TrainingCounters, assert_training_authorized
from .model import build_loss, build_model, build_optimizer, build_scheduler, model_state_sha256
from .provenance import atomic_write_json


_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class OverfitGateFailed(RuntimeError):
    """Raised after a completed diagnostic fails its frozen acceptance rules."""


@dataclass(frozen=True)
class OverfitGateCriteria:
    minimum_loss_reduction_fraction: float
    minimum_mean_represented_class_dice: float
    minimum_optimizer_steps: int
    require_checkpoint_reload_parity: bool

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "OverfitGateCriteria":
        expected = {
            "minimum_loss_reduction_fraction",
            "minimum_mean_represented_class_dice",
            "minimum_optimizer_steps",
            "require_checkpoint_reload_parity",
        }
        if set(value) != expected:
            raise ValueError(
                "Overfit acceptance keys differ: "
                f"missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
            )
        result = cls(
            minimum_loss_reduction_fraction=float(value["minimum_loss_reduction_fraction"]),
            minimum_mean_represented_class_dice=float(value["minimum_mean_represented_class_dice"]),
            minimum_optimizer_steps=int(value["minimum_optimizer_steps"]),
            require_checkpoint_reload_parity=bool(value["require_checkpoint_reload_parity"]),
        )
        if not 0 < result.minimum_loss_reduction_fraction < 1:
            raise ValueError("minimum_loss_reduction_fraction must lie in (0,1)")
        if not 0 < result.minimum_mean_represented_class_dice <= 1:
            raise ValueError("minimum_mean_represented_class_dice must lie in (0,1]")
        if result.minimum_optimizer_steps <= 0:
            raise ValueError("minimum_optimizer_steps must be positive")
        return result


def overfit_scope_sha256(
    *,
    case_id: str,
    fixed_sample_indices: tuple[int, ...],
    expected_selected_class_ids: tuple[int, ...],
) -> str:
    """Fingerprint the train-only patch bank without any absolute path."""
    if not fixed_sample_indices or len(fixed_sample_indices) != len(expected_selected_class_ids):
        raise ValueError("Fixed indices and expected classes must be non-empty and aligned")
    payload = {
        "gate": "one_case_overfit",
        "train_case_ids": [case_id],
        "validation_case_ids": [],
        "test_case_ids": [],
        "fixed_sample_indices": list(fixed_sample_indices),
        "expected_selected_class_ids": list(expected_selected_class_ids),
        "augmentation_enabled": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_run_dir(project_root: Path, config: BaselineConfig, run_id: str) -> Path:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be 1-64 safe filename characters")
    root = project_root.resolve()
    run_dir = (root / config.run_root / run_id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("Overfit output directory escapes the project root") from exc
    return run_dir


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _capture_rng(loader: DataLoader) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1],
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "loader_generator": loader.generator.get_state() if loader.generator is not None else None,
    }


def _restore_rng(state: dict[str, Any], loader: DataLoader) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state[0]),
            np.asarray(numpy_state[1], dtype=np.uint32),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("loader_generator") is not None:
        if loader.generator is None:
            raise ValueError("Resume checkpoint expects a DataLoader generator")
        loader.generator.set_state(state["loader_generator"])


def _sequential_bank_loader(loader: DataLoader) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=SequentialSampler(loader.dataset),
        num_workers=0,
        collate_fn=loader.collate_fn,
        pin_memory=bool(loader.pin_memory),
        drop_last=False,
    )


def _scalar(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected a scalar tensor in fixed-patch metadata")
        return int(value.item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected one fixed-patch metadata value")
        return _scalar(value[0])
    return int(value)


def _materialize_fixed_bank(loader: DataLoader) -> tuple[DataLoader, DataLoader]:
    """Load the fixed patches once, then train from a bounded in-memory bank.

    Recomputing every item would rescan the complete source label to select one
    voxel on every epoch. The authorized bank is only 32 x 96^3 image/label
    patches, so materializing it is both deterministic and substantially faster.
    """
    source = _sequential_bank_loader(loader)
    items: list[dict[str, Any]] = []
    for batch in source:
        if batch["image"].shape[0] != 1 or batch["label"].shape[0] != 1:
            raise ValueError("Fixed-bank materialization requires batch size one")
        trace = batch["patch_trace"]
        items.append(
            {
                "image": batch["image"][0].clone().contiguous(),
                "label": batch["label"][0].clone().contiguous(),
                "sample_index": _scalar(batch["sample_index"]),
                "patch_trace": {
                    "selected_class_id": _scalar(trace["selected_class_id"]),
                },
            }
        )
    if not items:
        raise RuntimeError("Fixed patch bank is empty")
    shuffled = isinstance(loader.sampler, RandomSampler)
    training = DataLoader(
        items,
        batch_size=loader.batch_size,
        shuffle=shuffled,
        num_workers=0,
        pin_memory=bool(loader.pin_memory),
        generator=loader.generator,
        drop_last=False,
    )
    evaluation = DataLoader(
        items,
        batch_size=loader.batch_size,
        sampler=SequentialSampler(items),
        num_workers=0,
        pin_memory=bool(loader.pin_memory),
        drop_last=False,
    )
    return training, evaluation


def _evaluate_bank(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: BaselineConfig,
    collect_trace: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batches = 0
    intersections = torch.zeros(config.out_channels, dtype=torch.int64)
    prediction_counts = torch.zeros(config.out_channels, dtype=torch.int64)
    truth_counts = torch.zeros(config.out_channels, dtype=torch.int64)
    traces: list[dict[str, int]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=config.automatic_mixed_precision
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss during fixed-bank evaluation")
            predictions = logits.argmax(dim=1, keepdim=True)
            total_loss += float(loss.item())
            batches += 1
            for class_id in range(1, config.out_channels):
                predicted = predictions == class_id
                truth = labels == class_id
                intersections[class_id] += torch.logical_and(predicted, truth).sum().cpu()
                prediction_counts[class_id] += predicted.sum().cpu()
                truth_counts[class_id] += truth.sum().cpu()
            if collect_trace:
                trace = batch["patch_trace"]
                traces.append(
                    {
                        "sample_index": _scalar(batch["sample_index"]),
                        "selected_class_id": _scalar(trace["selected_class_id"]),
                    }
                )
    if batches == 0:
        raise RuntimeError("Fixed patch bank is empty")
    per_class: list[dict[str, Any]] = []
    represented_dice: list[float] = []
    for class_id in range(1, config.out_channels):
        truth_count = int(truth_counts[class_id])
        predicted_count = int(prediction_counts[class_id])
        if truth_count:
            dice = 2.0 * int(intersections[class_id]) / (truth_count + predicted_count)
            represented_dice.append(dice)
            per_class.append(
                {
                    "class_id": class_id,
                    "dice": dice,
                    "truth_voxels": truth_count,
                    "prediction_voxels": predicted_count,
                }
            )
    return {
        "loss": total_loss / batches,
        "mean_represented_class_dice": float(np.mean(represented_dice)),
        "represented_classes": per_class,
        "patches": batches,
        "traces": traces,
    }


def _train_bank_epoch(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader: DataLoader,
    device: torch.device,
    config: BaselineConfig,
    counters: TrainingCounters,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batches = 0
    loader_length = len(loader)
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=config.automatic_mixed_precision
        ):
            logits = model(images)
            raw_loss = criterion(logits, labels)
            group_start = (batch_index // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
            group_size = min(config.gradient_accumulation_steps, loader_length - group_start)
            loss = raw_loss / group_size
        if not torch.isfinite(raw_loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}")
        counters.model_forward_calls += 1
        counters.loss_evaluations += 1
        scaler.scale(loss).backward()
        counters.backward_calls += 1
        counters.training_batches_consumed += 1
        total_loss += float(raw_loss.detach().item())
        batches += 1
        boundary = (batch_index + 1) % config.gradient_accumulation_steps == 0
        final_batch = batch_index + 1 == loader_length
        if boundary or final_batch:
            scaler.unscale_(optimizer)
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"Missing or non-finite gradient at batch {batch_index}")
            counters.gradient_checks += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient norm at batch {batch_index}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            counters.optimizer_steps += 1
    if not batches:
        raise RuntimeError("Fixed patch loader produced no batches")
    return total_loss / batches


def run_overfit_gate(
    *,
    project_root: Path,
    config: BaselineConfig,
    train_loader: DataLoader,
    case_id: str,
    fixed_sample_indices: tuple[int, ...],
    expected_selected_class_ids: tuple[int, ...],
    criteria: OverfitGateCriteria,
    provenance: dict[str, Any],
    workflow_config_sha256: str,
    manifest: dict[str, Any],
    schema: OrganSchema,
    preprocessing: PreprocessingSpec,
    gpu_report: dict[str, Any],
    run_id: str,
    resume_from: Path | None = None,
    allow_training: bool = False,
) -> dict[str, Any]:
    """Run or resume the fixed train-only diagnostic; never accepts val/test input."""
    started_monotonic = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    assert_training_authorized(config, allow_training=allow_training)
    if config.stage != "one_case_overfit":
        raise ValueError("run_overfit_gate requires the one_case_overfit config")
    if any((config.affine_probability, config.noise_probability, config.contrast_probability)):
        raise ValueError("The overfit gate forbids augmentation")
    contract = validate_baseline_contract(
        config, manifest=manifest, schema=schema, preprocessing=preprocessing, gpu_report=gpu_report
    )
    dataset = getattr(train_loader, "dataset", None)
    if not isinstance(dataset, ExactSplitPatchDataset):
        raise TypeError("Overfit loader must wrap ExactSplitPatchDataset")
    record_ids = tuple(str(record["case_id"]) for record in dataset.records)
    if record_ids != (case_id,) or any(record.get("split") != "train" for record in dataset.records):
        raise ValueError("Overfit loader must contain exactly its declared official training case")
    if dataset.config.config_sha256 != config.config_sha256 or dataset.augment:
        raise ValueError("Overfit dataset must use the gate config with augmentation disabled")
    if tuple(getattr(dataset, "fixed_sample_indices", ())) != fixed_sample_indices:
        raise ValueError("Overfit dataset does not expose the authorized fixed patch bank")
    if train_loader.batch_size != config.batch_size or train_loader.num_workers != 0:
        raise ValueError("Notebook overfit loader requires the configured batch size and zero workers")
    if train_loader.generator is None:
        raise ValueError("Overfit loader requires an explicit seeded generator for resumable shuffle order")
    scope_sha256 = overfit_scope_sha256(
        case_id=case_id,
        fixed_sample_indices=fixed_sample_indices,
        expected_selected_class_ids=expected_selected_class_ids,
    )
    expected_provenance = {
        "manifest_sha256": config.manifest_sha256,
        "organ_schema_sha256": config.organ_schema_sha256,
        "preprocessing_sha256": config.preprocessing_sha256,
        "baseline_config_sha256": config.config_sha256,
        "experiment_sha256": config.experiment_fingerprint(),
        "workflow_config_sha256": workflow_config_sha256,
        "scope_sha256": scope_sha256,
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("Overfit provenance does not match the authorized contracts")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; the overfit gate has no CPU fallback")
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    set_determinism(seed=config.random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_loader.generator.manual_seed(config.random_seed)
    train_loader, bank_loader = _materialize_fixed_bank(train_loader)
    run_dir = _safe_run_dir(project_root, config, run_id)
    checkpoint_dir = run_dir / "checkpoints"
    report_path = run_dir / "gate_report.json"
    last_path = checkpoint_dir / "last.pt"
    if resume_from is None:
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing overfit run: {run_dir}")
    else:
        resume_path = resume_from.resolve()
        if resume_path != last_path.resolve() or not resume_path.is_file():
            raise ValueError("resume_from must be this run's existing checkpoints/last.pt")
    device = torch.device("cuda:0")
    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    # The PyTorch default (65536) overflowed the audited 3D Dice+CE baseline
    # during the first fixed-patch epoch.  Keep the lower scale explicit in the
    # fingerprinted config; genuine non-finite gradients still fail closed.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.automatic_mixed_precision,
        init_scale=config.amp_initial_scale,
    )
    counters = TrainingCounters()
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume_from is None:
        initial = _evaluate_bank(
            model=model,
            criterion=criterion,
            loader=bank_loader,
            device=device,
            config=config,
            collect_trace=True,
        )
        actual_pairs = tuple((item["sample_index"], item["selected_class_id"]) for item in initial["traces"])
        expected_pairs = tuple(zip(fixed_sample_indices, expected_selected_class_ids, strict=True))
        if actual_pairs != expected_pairs:
            raise ValueError(f"Fixed patch trace differs from the authorized class-centered bank: {actual_pairs}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        counters.checkpoint_write_attempts += 1
        initial_checkpoint_counters = asdict(counters)
        initial_checkpoint_counters["checkpoints_written"] += 1
        _atomic_checkpoint(
            last_path,
            {
                "kind": "one_case_overfit_last",
                "schema_version": 1,
                "config_sha256": config.config_sha256,
                "workflow_config_sha256": workflow_config_sha256,
                "scope_sha256": scope_sha256,
                "provenance": provenance,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": 0,
                "initial_bank_metrics": initial,
                "history": history,
                "counters": initial_checkpoint_counters,
                "rng_state": _capture_rng(train_loader),
            },
        )
        counters.checkpoints_written += 1
    else:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        checkpoint_locks = {
            "config_sha256": config.config_sha256,
            "workflow_config_sha256": workflow_config_sha256,
            "scope_sha256": scope_sha256,
            "provenance": provenance,
        }
        if any(checkpoint.get(key) != value for key, value in checkpoint_locks.items()):
            raise ValueError("Overfit resume checkpoint contract mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        counters = TrainingCounters(**checkpoint["counters"])
        counters.resume_events += 1
        history = list(checkpoint["history"])
        initial = dict(checkpoint["initial_bank_metrics"])
        start_epoch = int(checkpoint["epoch"])
        if start_epoch < 0 or start_epoch > config.max_epochs:
            raise ValueError("Overfit resume checkpoint epoch is outside the configured run")
        _restore_rng(checkpoint["rng_state"], train_loader)
    for epoch in range(start_epoch, config.max_epochs):
        training_loss = _train_bank_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            device=device,
            config=config,
            counters=counters,
        )
        scheduler.step()
        counters.scheduler_steps += 1
        counters.epochs_completed += 1
        bank_metrics = _evaluate_bank(
            model=model,
            criterion=criterion,
            loader=bank_loader,
            device=device,
            config=config,
            collect_trace=False,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": training_loss,
                "bank_loss": bank_metrics["loss"],
                "mean_represented_class_dice": bank_metrics["mean_represented_class_dice"],
                "represented_classes": bank_metrics["represented_classes"],
            }
        )
        counters.checkpoint_write_attempts += 1
        checkpoint_counters = asdict(counters)
        checkpoint_counters["checkpoints_written"] += 1
        _atomic_checkpoint(
            last_path,
            {
                "kind": "one_case_overfit_last",
                "schema_version": 1,
                "config_sha256": config.config_sha256,
                "workflow_config_sha256": workflow_config_sha256,
                "scope_sha256": scope_sha256,
                "provenance": provenance,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "initial_bank_metrics": initial,
                "history": history,
                "counters": checkpoint_counters,
                "rng_state": _capture_rng(train_loader),
            },
        )
        counters.checkpoints_written += 1
    final = _evaluate_bank(
        model=model,
        criterion=criterion,
        loader=bank_loader,
        device=device,
        config=config,
        collect_trace=False,
    )
    loss_reduction = 1.0 - final["loss"] / initial["loss"]
    saved_hash = model_state_sha256(model)
    reload_model = build_model(config).to(device)
    reloaded = torch.load(last_path, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(reloaded["model"])
    reloaded_hash = model_state_sha256(reload_model)
    reload_metrics = _evaluate_bank(
        model=reload_model,
        criterion=criterion,
        loader=bank_loader,
        device=device,
        config=config,
        collect_trace=False,
    )
    checkpoint_reload_parity = (
        saved_hash == reloaded_hash
        and math.isclose(final["loss"], reload_metrics["loss"], rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(
            final["mean_represented_class_dice"],
            reload_metrics["mean_represented_class_dice"],
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )
    acceptance = {
        "completed_configured_epochs": counters.epochs_completed == config.max_epochs,
        "finite_initial_and_final_loss": math.isfinite(initial["loss"]) and math.isfinite(final["loss"]),
        "minimum_loss_reduction": loss_reduction >= criteria.minimum_loss_reduction_fraction,
        "minimum_mean_represented_class_dice": (
            final["mean_represented_class_dice"] >= criteria.minimum_mean_represented_class_dice
        ),
        "minimum_optimizer_steps": counters.optimizer_steps >= criteria.minimum_optimizer_steps,
        "checkpoint_reload_parity": checkpoint_reload_parity,
        "train_only_scope": record_ids == (case_id,),
        "validation_split_not_accessed": True,
        "test_split_not_accessed": True,
    }
    if not criteria.require_checkpoint_reload_parity:
        acceptance["checkpoint_reload_parity"] = True
    passed = all(acceptance.values())
    report = {
        "schema_version": 1,
        "started_at_utc": started_utc,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration_seconds": time.perf_counter() - started_monotonic,
        "status": "passed" if passed else "failed",
        "gate": "one_case_overfit",
        "claim": "train-only fixed-patch memorization diagnostic; not validation evidence",
        "run_dir": run_dir.relative_to(project_root.resolve()).as_posix(),
        "last_checkpoint": last_path.relative_to(project_root.resolve()).as_posix(),
        "config_sha256": config.config_sha256,
        "workflow_config_sha256": workflow_config_sha256,
        "scope_sha256": scope_sha256,
        "provenance_sha256": hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "case_id": case_id,
        "fixed_sample_indices": list(fixed_sample_indices),
        "expected_selected_class_ids": list(expected_selected_class_ids),
        "initial_bank_metrics": initial,
        "final_bank_metrics": final,
        "loss_reduction_fraction": loss_reduction,
        "checkpoint_reload_parity": checkpoint_reload_parity,
        "counters": asdict(counters),
        "history": history,
        "acceptance": acceptance,
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "contract_checks": contract["checks"],
    }
    atomic_write_json(report_path, report)
    model.cpu()
    reload_model.cpu()
    torch.cuda.empty_cache()
    if not passed:
        failed = [name for name, value in acceptance.items() if not value]
        raise OverfitGateFailed(f"One-case overfit gate failed: {failed}. See {report_path}")
    return report
