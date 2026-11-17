"""Fail-closed, restartable baseline training engine.

The engine owns model construction so the experiment seed is applied before any
parameter is initialized.  Merely importing this module remains side-effect
free; a run directory is created only after authorization, data, provenance,
CUDA, and loader reproducibility contracts have all passed.

``fit`` is intentionally for the deterministic pilot and full baseline.  The
one-case overfit gate is a train-only fixed-patch diagnostic with different
metrics and must use its dedicated runner rather than this full-volume
validation loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance
from monai.utils import set_determinism
from torch.utils.data import RandomSampler, SequentialSampler

from src.dataset import OrganSchema, PreprocessingSpec

from .config import BaselineConfig, validate_baseline_contract
from .model import build_loss, build_model, build_optimizer, build_scheduler


class TrainingDisabledError(RuntimeError):
    """Raised before any training side effect when authorization is absent."""


class NonFiniteTrainingError(FloatingPointError):
    """Raised before an optimizer step when a loss or gradient is non-finite."""


CHECKPOINT_SCHEMA_VERSION = 2
RUN_METADATA_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_FIT_GATES = frozenset({"deterministic_pilot", "full_baseline"})


@dataclass(frozen=True)
class TrainingScope:
    """Exact official-split case selection authorized for one ``fit`` run.

    Case IDs are treated as sets for the scope fingerprint. Dataset order is
    recorded separately in the loader contract because it also affects exact
    reproducibility.
    """

    gate: str
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        train_ids = tuple(str(value) for value in self.train_case_ids)
        validation_ids = tuple(str(value) for value in self.validation_case_ids)
        object.__setattr__(self, "train_case_ids", train_ids)
        object.__setattr__(self, "validation_case_ids", validation_ids)
        if self.gate not in _FIT_GATES:
            raise ValueError(
                "fit scope must be deterministic_pilot or full_baseline; "
                "one_case_overfit uses the dedicated train-only diagnostic"
            )
        for role, case_ids in (("train", train_ids), ("validation", validation_ids)):
            if not case_ids:
                raise ValueError(f"{role} scope must contain at least one case")
            if len(case_ids) != len(set(case_ids)):
                raise ValueError(f"{role} scope contains duplicate case IDs")
            if any(not case_id or case_id.strip() != case_id for case_id in case_ids):
                raise ValueError(f"{role} scope contains an invalid case ID")

    @property
    def canonical(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "gate": self.gate,
            "train_case_ids": sorted(self.train_case_ids),
            "validation_case_ids": sorted(self.validation_case_ids),
        }

    def fingerprint(self) -> str:
        return _canonical_sha256(self.canonical)


@dataclass
class TrainingCounters:
    epochs_completed: int = 0
    training_batches_consumed: int = 0
    model_forward_calls: int = 0
    loss_evaluations: int = 0
    backward_calls: int = 0
    gradient_checks: int = 0
    amp_overflow_skips: int = 0
    optimizer_steps: int = 0
    scheduler_steps: int = 0
    validation_runs_completed: int = 0
    checkpoint_write_attempts: int = 0
    checkpoints_written: int = 0
    resume_events: int = 0


def assert_training_authorized(config: BaselineConfig, *, allow_training: bool) -> None:
    if config.execution_mode != "training" or not config.training_enabled or not allow_training:
        raise TrainingDisabledError(
            "Training is disabled. It requires a versioned config with execution_mode=training, "
            "training_enabled=true, and an explicit allow_training=True call."
        )


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Value is not finite canonical JSON") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_run_root(project_root: Path, config: BaselineConfig) -> Path:
    project_root = project_root.resolve()
    root = (project_root / Path(config.run_root)).resolve()
    try:
        relative = root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Training output root escapes the project directory") from exc
    if not relative.parts:
        raise ValueError("Training output root cannot be the project root")
    return root


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-96 characters containing only letters, digits, '.', '_', or '-' "
            "and must start with a letter or digit"
        )
    if run_id in {".", ".."}:
        raise ValueError("run_id cannot be '.' or '..'")
    return run_id


def _automatic_run_id(config: BaselineConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{config.config_sha256[:8]}-{uuid4().hex[:8]}"


def _resolve_resume_checkpoint(
    *,
    project_root: Path,
    base_root: Path,
    resume_from: Path,
    run_id: str | None,
) -> tuple[Path, Path, str]:
    candidate = resume_from if resume_from.is_absolute() else project_root.resolve() / resume_from
    checkpoint = candidate.resolve()
    try:
        checkpoint.relative_to(base_root)
    except ValueError as exc:
        raise ValueError("Resume checkpoint escapes the configured training output root") from exc
    if checkpoint.name not in {"last.pt", "best.pt"} or checkpoint.parent.name != "checkpoints":
        raise ValueError("Resume checkpoint must be a run's checkpoints/last.pt or checkpoints/best.pt")
    run_directory = checkpoint.parent.parent
    if run_directory.parent != base_root:
        raise ValueError("Resume checkpoint must belong to a direct child run directory")
    resolved_run_id = _validate_run_id(run_directory.name)
    if run_id is not None and _validate_run_id(run_id) != resolved_run_id:
        raise ValueError("run_id does not match the resume checkpoint run directory")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    return checkpoint, run_directory, resolved_run_id


def _create_unique_run_directory(
    base_root: Path,
    config: BaselineConfig,
    run_id: str | None,
) -> tuple[Path, str]:
    base_root.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        resolved_run_id = _validate_run_id(run_id)
        run_directory = base_root / resolved_run_id
        run_directory.mkdir(exist_ok=False)
        return run_directory, resolved_run_id
    for _ in range(10):
        # Keep generation inside the loop so an exceptionally unlikely collision
        # creates a fresh UUID rather than silently reusing an existing directory.
        resolved_run_id = _automatic_run_id(config)
        run_directory = base_root / resolved_run_id
        try:
            run_directory.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_directory, resolved_run_id
    raise FileExistsError("Could not allocate a unique training run directory")


def _seed_everything(config: BaselineConfig) -> None:
    """Apply all experiment seeds before the engine constructs its model."""

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    set_determinism(seed=config.random_seed)
    torch.use_deterministic_algorithms(True, warn_only=not config.strict_cuda_determinism)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_and_build_model(
    config: BaselineConfig,
    supplied_model: torch.nn.Module | None,
) -> torch.nn.Module:
    if supplied_model is not None:
        raise ValueError(
            "fit constructs the model internally so seeding precedes parameter initialization; "
            "pass model=None"
        )
    _seed_everything(config)
    return build_model(config)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("Checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise ValueError("Checkpoint NumPy RNG state is invalid")
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available():
        if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG state does not match available devices")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise ValueError("Checkpoint contains CUDA RNG state but CUDA is unavailable")


def _loader_rng_state(train_loader: Any, validation_loader: Any) -> dict[str, torch.Tensor]:
    return {
        "train": train_loader.generator.get_state(),
        "validation": validation_loader.generator.get_state(),
    }


def _restore_loader_rng_state(
    state: dict[str, torch.Tensor],
    train_loader: Any,
    validation_loader: Any,
) -> None:
    if not isinstance(state, dict) or set(state) != {"train", "validation"}:
        raise ValueError("Checkpoint loader RNG state is incomplete")
    train_loader.generator.set_state(state["train"].cpu())
    validation_loader.generator.set_state(state["validation"].cpu())


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _model_signature(model: torch.nn.Module) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        name: (tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in model.state_dict().items()
    }


def _json_model_signature(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in model.state_dict().items()
    }


def _official_case_ids(manifest: dict[str, Any], split: str) -> tuple[str, ...]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Manifest cases are missing")
    case_ids = tuple(str(case["case_id"]) for case in cases if case.get("split") == split)
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Manifest {split} case IDs are empty or duplicated")
    return case_ids


def _resolve_training_scope(
    config: BaselineConfig,
    manifest: dict[str, Any],
    scope: TrainingScope | None,
) -> TrainingScope:
    official_train = _official_case_ids(manifest, config.train_split)
    official_validation = _official_case_ids(manifest, config.validation_split)
    resolved = scope or TrainingScope(
        gate="full_baseline",
        train_case_ids=official_train,
        validation_case_ids=official_validation,
    )
    train_set = set(resolved.train_case_ids)
    validation_set = set(resolved.validation_case_ids)
    if not train_set <= set(official_train):
        raise ValueError("Training scope includes a case outside the official train split")
    if not validation_set <= set(official_validation):
        raise ValueError("Training scope includes a case outside the official validation split")
    if resolved.gate == "full_baseline":
        if train_set != set(official_train) or validation_set != set(official_validation):
            raise ValueError("full_baseline scope must contain every official train and validation case")
    elif len(train_set) == len(official_train) and len(validation_set) == len(official_validation):
        raise ValueError("deterministic_pilot must be a proper subset of the full baseline")
    return resolved


def _dataset_case_ids(dataset: Any) -> tuple[str, ...]:
    if hasattr(dataset, "selected_case_ids"):
        case_ids = tuple(str(value) for value in dataset.selected_case_ids)
    else:
        records = getattr(dataset, "records", None)
        if not isinstance(records, (list, tuple)):
            raise ValueError("Dataset does not expose auditable case records")
        case_ids = tuple(str(case["case_id"]) for case in records)
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("Dataset case IDs are empty or duplicated")
    return case_ids


def _callable_name(value: Any) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


def _loader_contract(
    loader: Any,
    *,
    role: str,
    expected_seed: int,
    config: BaselineConfig,
) -> dict[str, Any]:
    generator = getattr(loader, "generator", None)
    if not isinstance(generator, torch.Generator):
        raise ValueError(f"{role} loader must expose a dedicated torch.Generator")
    if int(generator.initial_seed()) != expected_seed:
        raise ValueError(f"{role} loader generator seed does not match the reproducibility contract")
    sampler = getattr(loader, "sampler", None)
    if role == "train":
        if not isinstance(sampler, RandomSampler) or sampler.replacement:
            raise ValueError("Training loader must use non-replacement RandomSampler")
        if getattr(sampler, "generator", None) is not generator:
            raise ValueError("Training sampler and loader must share the dedicated generator")
        expected_batch_size = config.batch_size
    elif role == "validation":
        if not isinstance(sampler, SequentialSampler):
            raise ValueError("Validation loader must use SequentialSampler")
        expected_batch_size = 1
    else:
        raise ValueError(f"Unknown loader role: {role}")
    if getattr(loader, "batch_size", None) != expected_batch_size:
        raise ValueError(f"{role} loader batch size violates the baseline contract")
    if bool(getattr(loader, "drop_last", True)):
        raise ValueError(f"{role} loader must not drop cases or patches")
    workers = int(getattr(loader, "num_workers", -1))
    allowed_workers = {config.notebook_workers, config.production_workers}
    if workers not in allowed_workers:
        raise ValueError(f"{role} loader workers must match a configured notebook/production mode")
    if bool(getattr(loader, "pin_memory", False)) != config.pin_memory:
        raise ValueError(f"{role} loader pin_memory differs from the config")
    expected_persistent = config.persistent_workers if workers > 0 else False
    if bool(getattr(loader, "persistent_workers", False)) != expected_persistent:
        raise ValueError(f"{role} loader persistent_workers differs from the config")
    prefetch_factor = getattr(loader, "prefetch_factor", None)
    if workers > 0 and prefetch_factor != config.prefetch_factor:
        raise ValueError(f"{role} loader prefetch_factor differs from the config")
    worker_init = getattr(loader, "worker_init_fn", None)
    if workers > 0 and not callable(worker_init):
        raise ValueError(f"{role} loader requires an explicit deterministic worker initializer")
    declared = getattr(loader, "reproducibility_contract", None)
    if declared is not None:
        expected_declared = {
            "schema_version": 1,
            "role": role,
            "seed": expected_seed,
            "shuffle": role == "train",
            "worker_seeded": workers == 0 or callable(worker_init),
        }
        if declared != expected_declared:
            raise ValueError(f"{role} loader declared a mismatched reproducibility contract")
    return {
        "schema_version": 1,
        "role": role,
        "generator_initial_seed": expected_seed,
        "sampler": type(sampler).__name__,
        "sampler_replacement": bool(getattr(sampler, "replacement", False)),
        "batch_size": expected_batch_size,
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": expected_persistent,
        "prefetch_factor": prefetch_factor,
        "worker_init_fn": _callable_name(worker_init),
        "collate_fn": _callable_name(getattr(loader, "collate_fn", None)),
        "case_ids": list(_dataset_case_ids(loader.dataset)),
    }


def _validate_loaders(
    *,
    train_loader: Any,
    validation_loader: Any,
    config: BaselineConfig,
    scope: TrainingScope,
) -> dict[str, Any]:
    from .data import ExactSplitPatchDataset, ExactSplitVolumeDataset

    if not isinstance(getattr(train_loader, "dataset", None), ExactSplitPatchDataset):
        raise TypeError("Training loader must wrap ExactSplitPatchDataset")
    if not isinstance(getattr(validation_loader, "dataset", None), ExactSplitVolumeDataset):
        raise TypeError("Validation loader must wrap ExactSplitVolumeDataset")
    train_dataset = train_loader.dataset
    validation_dataset = validation_loader.dataset
    if (
        train_dataset.config.config_sha256 != config.config_sha256
        or validation_dataset.config.config_sha256 != config.config_sha256
    ):
        raise ValueError("Data loaders were not built from the authorized training config")
    train_ids = _dataset_case_ids(train_dataset)
    validation_ids = _dataset_case_ids(validation_dataset)
    if set(train_ids) != set(scope.train_case_ids) or len(train_ids) != len(scope.train_case_ids):
        raise ValueError("Training loader does not exactly match the authorized scope")
    if set(validation_ids) != set(scope.validation_case_ids) or len(validation_ids) != len(
        scope.validation_case_ids
    ):
        raise ValueError("Validation loader does not exactly match the authorized scope")
    if any(case.get("split") != config.train_split for case in train_dataset.records):
        raise ValueError("Training loader violates the exact train split")
    if any(case.get("split") != config.validation_split for case in validation_dataset.records):
        raise ValueError("Validation loader violates the exact validation split")
    train_contract = _loader_contract(
        train_loader,
        role="train",
        expected_seed=config.random_seed,
        config=config,
    )
    validation_contract = _loader_contract(
        validation_loader,
        role="validation",
        expected_seed=config.random_seed + 1,
        config=config,
    )
    if train_loader.generator is validation_loader.generator:
        raise ValueError("Training and validation loaders require independent generators")
    return {"train": train_contract, "validation": validation_contract}


def _expected_provenance(config: BaselineConfig) -> dict[str, str]:
    return {
        "manifest_sha256": config.manifest_sha256,
        "organ_schema_sha256": config.organ_schema_sha256,
        "preprocessing_sha256": config.preprocessing_sha256,
        "baseline_config_sha256": config.config_sha256,
        "experiment_sha256": config.experiment_fingerprint(),
    }


def _validate_provenance(config: BaselineConfig, provenance: dict[str, Any]) -> str:
    if not isinstance(provenance, dict):
        raise TypeError("Training provenance must be a mapping")
    expected = _expected_provenance(config)
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Training provenance does not match the authorized contracts")
    return _canonical_sha256(provenance)


def _train_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: BaselineConfig,
    counters: TrainingCounters,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batch_count = 0
    loader_length = len(loader)
    consecutive_amp_overflows = 0
    if loader_length <= 0:
        raise RuntimeError("Training loader produced no batches")
    for batch_index, batch in enumerate(loader):
        counters.training_batches_consumed += 1
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=config.automatic_mixed_precision,
        ):
            counters.model_forward_calls += 1
            logits = model(images)
            counters.loss_evaluations += 1
            raw_loss = criterion(logits, labels)
            if raw_loss.numel() != 1:
                raise ValueError("Training criterion must return one scalar loss")
            if not bool(torch.isfinite(raw_loss.detach()).all().item()):
                optimizer.zero_grad(set_to_none=True)
                raise NonFiniteTrainingError(
                    f"Non-finite training loss at batch {batch_index}; optimizer step was blocked"
                )
            group_start = (batch_index // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
            group_size = min(config.gradient_accumulation_steps, loader_length - group_start)
            loss = raw_loss / group_size
        scaler.scale(loss).backward()
        counters.backward_calls += 1
        total_loss += float(raw_loss.detach().item())
        batch_count += 1
        boundary = (batch_index + 1) % config.gradient_accumulation_steps == 0
        final_batch = batch_index + 1 == len(loader)
        if boundary or final_batch:
            scaler.unscale_(optimizer)
            counters.gradient_checks += 1
            gradients = [
                (name, parameter.grad)
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
            if not gradients:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("Backward produced no parameter gradients")
            nan_gradients = [
                name
                for name, gradient in gradients
                if bool(torch.isnan(gradient).any().item())
            ]
            if nan_gradients:
                optimizer.zero_grad(set_to_none=True)
                preview = ", ".join(nan_gradients[:5])
                raise NonFiniteTrainingError(
                    f"NaN gradients before optimizer step at batch {batch_index}: {preview}"
                )
            infinite_gradients = [
                name
                for name, gradient in gradients
                if bool(torch.isinf(gradient).any().item())
            ]
            if infinite_gradients:
                if not config.automatic_mixed_precision or not hasattr(scaler, "get_scale"):
                    optimizer.zero_grad(set_to_none=True)
                    preview = ", ".join(infinite_gradients[:5])
                    raise NonFiniteTrainingError(
                        f"Infinite gradients without AMP backoff at batch {batch_index}: {preview}"
                    )
                previous_scale = float(scaler.get_scale())
                # unscale_ has already recorded the overflow. GradScaler.step
                # therefore skips the optimizer update, and update halves the
                # scale. This is the standard recoverable AMP-overflow path.
                scaler.step(optimizer)
                scaler.update()
                reduced_scale = float(scaler.get_scale())
                optimizer.zero_grad(set_to_none=True)
                if not math.isfinite(reduced_scale) or reduced_scale >= previous_scale:
                    preview = ", ".join(infinite_gradients[:5])
                    raise NonFiniteTrainingError(
                        f"AMP failed to back off an infinite gradient at batch {batch_index}: {preview}"
                    )
                counters.amp_overflow_skips += 1
                consecutive_amp_overflows += 1
                if consecutive_amp_overflows > 8:
                    raise NonFiniteTrainingError(
                        "More than eight consecutive AMP gradient overflows; training is unstable"
                    )
                continue
            consecutive_amp_overflows = 0
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm).all().item()):
                optimizer.zero_grad(set_to_none=True)
                raise NonFiniteTrainingError(
                    f"Non-finite clipped gradient norm at batch {batch_index}; optimizer step was blocked"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            counters.optimizer_steps += 1
    return total_loss / batch_count


def _case_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    spacing_mm: tuple[float, float, float],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for class_id in range(1, num_classes):
        predicted = prediction == class_id
        truth = target == class_id
        predicted_count = int(predicted.sum().item())
        truth_count = int(truth.sum().item())
        if truth_count == 0:
            dice = math.nan
            hd95 = math.nan
        else:
            intersection = int(torch.logical_and(predicted, truth).sum().item())
            dice = 2.0 * intersection / (predicted_count + truth_count)
            if predicted_count == 0:
                hd95 = math.nan
            else:
                hd = compute_hausdorff_distance(
                    predicted.float(),
                    truth.float(),
                    include_background=True,
                    distance_metric="euclidean",
                    percentile=95,
                    spacing=[spacing_mm],
                )
                hd95 = float(hd.item())
        records.append(
            {
                "class_id": class_id,
                "organ": class_names[class_id],
                "dice": dice,
                "hd95_mm": hd95,
                "ground_truth_present": truth_count > 0,
                "prediction_present": predicted_count > 0,
                "false_positive_on_empty": truth_count == 0 and predicted_count > 0,
                "missing_prediction": truth_count > 0 and predicted_count == 0,
            }
        )
    return records


def _validate(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    config: BaselineConfig,
    spacing_mm: tuple[float, float, float],
    class_names: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    class_records: list[list[dict[str, Any]]] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"]
            label = batch["label"].cpu()
            if image.shape[0] != 1 or label.shape[0] != 1:
                raise ValueError("Full-volume validation requires batch size one")
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=config.automatic_mixed_precision,
            ):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=config.sliding_window_roi_size,
                    sw_batch_size=config.sliding_window_batch_size,
                    predictor=model,
                    overlap=config.sliding_window_overlap,
                    mode=config.sliding_window_blend_mode,
                    sw_device=device,
                    device="cpu",
                )
            prediction = logits.argmax(dim=1, keepdim=True)
            class_records.append(
                _case_metrics(
                    prediction,
                    label,
                    num_classes=config.out_channels,
                    spacing_mm=spacing_mm,
                    class_names=class_names,
                )
            )
    per_class: list[dict[str, Any]] = []
    for class_index in range(config.out_channels - 1):
        records = [case[class_index] for case in class_records]
        valid_dice = [record["dice"] for record in records if math.isfinite(record["dice"])]
        valid_hd = [record["hd95_mm"] for record in records if math.isfinite(record["hd95_mm"])]
        per_class.append(
            {
                "class_id": class_index + 1,
                "organ": class_names[class_index + 1],
                "dice": float(np.mean(valid_dice)) if valid_dice else math.nan,
                "hd95_mm": float(np.mean(valid_hd)) if valid_hd else math.nan,
                "valid_dice_cases": len(valid_dice),
                "valid_hd95_cases": len(valid_hd),
                "false_positive_empty_cases": sum(record["false_positive_on_empty"] for record in records),
                "ground_truth_empty_cases": sum(not record["ground_truth_present"] for record in records),
                "missing_prediction_cases": sum(record["missing_prediction"] for record in records),
            }
        )
    valid_means = [record["dice"] for record in per_class if math.isfinite(record["dice"])]
    return {
        "mean_foreground_dice": float(np.mean(valid_means)) if valid_means else math.nan,
        "per_class": per_class,
        "validation_cases": len(class_records),
    }


def _run_metadata(
    *,
    config: BaselineConfig,
    provenance: dict[str, Any],
    provenance_sha256: str,
    scope: TrainingScope,
    loader_contracts: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "config_sha256": config.config_sha256,
        "experiment_sha256": config.experiment_fingerprint(),
        "config": config.canonical_config,
        "provenance_sha256": provenance_sha256,
        "provenance": provenance,
        "scope_sha256": scope.fingerprint(),
        "scope": scope.canonical,
        "loader_contracts_sha256": _canonical_sha256(loader_contracts),
        "loader_contracts": loader_contracts,
        "random_seed": config.random_seed,
        "created_utc": _utc_now(),
    }


def _validate_run_metadata(
    metadata: dict[str, Any],
    *,
    config: BaselineConfig,
    provenance: dict[str, Any],
    provenance_sha256: str,
    scope: TrainingScope,
    loader_contracts: dict[str, Any],
    run_id: str,
) -> None:
    checks = {
        "schema_version": metadata.get("schema_version") == RUN_METADATA_SCHEMA_VERSION,
        "run_id": metadata.get("run_id") == run_id,
        "experiment_id": metadata.get("experiment_id") == config.experiment_id,
        "config_sha256": metadata.get("config_sha256") == config.config_sha256,
        "experiment_sha256": metadata.get("experiment_sha256") == config.experiment_fingerprint(),
        "config": metadata.get("config") == config.canonical_config,
        "provenance_sha256": metadata.get("provenance_sha256") == provenance_sha256,
        "provenance": metadata.get("provenance") == provenance,
        "scope_sha256": metadata.get("scope_sha256") == scope.fingerprint(),
        "scope": metadata.get("scope") == scope.canonical,
        "loader_contracts_sha256": metadata.get("loader_contracts_sha256")
        == _canonical_sha256(loader_contracts),
        "loader_contracts": metadata.get("loader_contracts") == loader_contracts,
        "random_seed": metadata.get("random_seed") == config.random_seed,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Run metadata does not match this resume request: {failed}")


def _checkpoint_payload(
    *,
    kind: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best: float,
    validations_without_improvement: int,
    config: BaselineConfig,
    provenance: dict[str, Any],
    provenance_sha256: str,
    scope: TrainingScope,
    loader_contracts: dict[str, Any],
    run_id: str,
    counters: TrainingCounters,
    projected_checkpoints_written: int,
    history: list[dict[str, Any]],
    train_loader: Any,
    validation_loader: Any,
) -> dict[str, Any]:
    checkpoint_counters = asdict(counters)
    checkpoint_counters["checkpoints_written"] = projected_checkpoints_written
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "run_id": run_id,
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "best_validation_mean_foreground_dice": best,
        "validations_without_improvement": validations_without_improvement,
        "model": model.state_dict(),
        "model_signature": _json_model_signature(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config_sha256": config.config_sha256,
        "experiment_sha256": config.experiment_fingerprint(),
        "config": config.canonical_config,
        "provenance_sha256": provenance_sha256,
        "provenance": provenance,
        "scope_sha256": scope.fingerprint(),
        "scope": scope.canonical,
        "loader_contracts_sha256": _canonical_sha256(loader_contracts),
        "loader_contracts": loader_contracts,
        "counters": checkpoint_counters,
        "history": history,
        "rng_state": _rng_state(),
        "loader_rng_state": _loader_rng_state(train_loader, validation_loader),
        "created_utc": _utc_now(),
    }


def _write_checkpoint(
    path: Path,
    *,
    kind: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best: float,
    validations_without_improvement: int,
    config: BaselineConfig,
    provenance: dict[str, Any],
    provenance_sha256: str,
    scope: TrainingScope,
    loader_contracts: dict[str, Any],
    run_id: str,
    counters: TrainingCounters,
    history: list[dict[str, Any]],
    train_loader: Any,
    validation_loader: Any,
) -> None:
    counters.checkpoint_write_attempts += 1
    projected_written = counters.checkpoints_written + 1
    payload = _checkpoint_payload(
        kind=kind,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        best=best,
        validations_without_improvement=validations_without_improvement,
        config=config,
        provenance=provenance,
        provenance_sha256=provenance_sha256,
        scope=scope,
        loader_contracts=loader_contracts,
        run_id=run_id,
        counters=counters,
        projected_checkpoints_written=projected_written,
        history=history,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )
    _atomic_checkpoint(path, payload)
    counters.checkpoints_written = projected_written


def _load_resume_state(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: BaselineConfig,
    provenance: dict[str, Any],
    provenance_sha256: str,
    scope: TrainingScope,
    loader_contracts: dict[str, Any],
    run_id: str,
    train_loader: Any,
    validation_loader: Any,
) -> tuple[int, float, int, TrainingCounters, list[dict[str, Any]]]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot load resume checkpoint: {checkpoint_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Resume checkpoint payload is not a mapping")
    checks = {
        "schema_version": payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "kind": payload.get("kind") in {"last", "best"},
        "run_id": payload.get("run_id") == run_id,
        "config_sha256": payload.get("config_sha256") == config.config_sha256,
        "experiment_sha256": payload.get("experiment_sha256") == config.experiment_fingerprint(),
        "config": payload.get("config") == config.canonical_config,
        "provenance_sha256": payload.get("provenance_sha256") == provenance_sha256,
        "provenance": payload.get("provenance") == provenance,
        "scope_sha256": payload.get("scope_sha256") == scope.fingerprint(),
        "scope": payload.get("scope") == scope.canonical,
        "loader_contracts_sha256": payload.get("loader_contracts_sha256")
        == _canonical_sha256(loader_contracts),
        "loader_contracts": payload.get("loader_contracts") == loader_contracts,
        "model_signature": payload.get("model_signature") == _json_model_signature(model),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Resume checkpoint contract mismatch: {failed}")
    epoch = payload.get("epoch")
    history = payload.get("history")
    if not isinstance(epoch, int) or epoch < 0 or epoch > config.max_epochs:
        raise ValueError("Resume checkpoint epoch is outside the authorized range")
    if payload.get("next_epoch") != epoch + 1:
        raise ValueError("Resume checkpoint next_epoch is inconsistent")
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Resume checkpoint history is inconsistent with its epoch")
    try:
        counters = TrainingCounters(**payload["counters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resume checkpoint counters are invalid") from exc
    if counters.epochs_completed != epoch or counters.scheduler_steps != epoch:
        raise ValueError("Resume checkpoint counters are inconsistent with its epoch")
    best = payload.get("best_validation_mean_foreground_dice")
    if not isinstance(best, (float, int)) or math.isnan(float(best)):
        raise ValueError("Resume checkpoint best validation metric is invalid")
    validations_without_improvement = payload.get("validations_without_improvement")
    if not isinstance(validations_without_improvement, int) or validations_without_improvement < 0:
        raise ValueError("Resume checkpoint early-stopping state is invalid")
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        _restore_rng_state(payload["rng_state"])
        _restore_loader_rng_state(payload["loader_rng_state"], train_loader, validation_loader)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Resume checkpoint state restoration failed") from exc
    counters.resume_events += 1
    return epoch, float(best), validations_without_improvement, counters, history


def _status_record(
    *,
    status: str,
    run_id: str,
    config: BaselineConfig,
    scope: TrainingScope,
    counters: TrainingCounters,
    started_utc: str,
    resumed_from: str | None,
    message: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "config_sha256": config.config_sha256,
        "scope_sha256": scope.fingerprint(),
        "gate": scope.gate,
        "started_utc": started_utc,
        "updated_utc": _utc_now(),
        "resumed_from": resumed_from,
        "counters": asdict(counters),
    }
    if message is not None:
        record["message"] = message
    return record


def fit(
    *,
    project_root: Path,
    config: BaselineConfig,
    train_loader: Any,
    validation_loader: Any,
    provenance: dict[str, Any],
    spacing_mm: tuple[float, float, float],
    manifest: dict[str, Any],
    schema: OrganSchema,
    preprocessing: PreprocessingSpec,
    gpu_report: dict[str, Any],
    model: torch.nn.Module | None = None,
    allow_training: bool = False,
    scope: TrainingScope | None = None,
    run_id: str | None = None,
    resume_from: Path | None = None,
    pause_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Train an internally constructed model after all fail-closed checks pass.

    Parameters added for the executable training notebook:

    - ``scope`` authorizes an exact pilot subset or the complete official split.
    - ``run_id`` selects a safe, unique child of ``config.run_root``. When it is
      omitted a timestamp/config/UUID identifier is generated.
    - ``resume_from`` must name ``checkpoints/last.pt`` or ``best.pt`` inside an
      existing run. Config, provenance, scope, loader construction, model
      signature, optimizer state, counters, and RNG streams are validated before
      the next epoch.
    - ``pause_after_epoch`` is restricted to the deterministic pilot. It creates
      a deliberate complete-epoch interruption so the notebook can prove that a
      fresh process would resume from ``last.pt`` before the full run is allowed.

    ``model`` remains in the signature to turn older callers into a clear error;
    it must be ``None`` because engine-owned construction is what guarantees that
    seeding happens before parameter initialization.
    """
    project_root = project_root.resolve()
    assert_training_authorized(config, allow_training=allow_training)
    contract = validate_baseline_contract(
        config,
        manifest=manifest,
        schema=schema,
        preprocessing=preprocessing,
        gpu_report=gpu_report,
    )
    resolved_scope = _resolve_training_scope(config, manifest, scope)
    if pause_after_epoch is not None:
        if isinstance(pause_after_epoch, bool) or not isinstance(pause_after_epoch, int):
            raise TypeError("pause_after_epoch must be an integer or None")
        if resolved_scope.gate != "deterministic_pilot":
            raise ValueError("Only the deterministic pilot may request a deliberate pause")
        if pause_after_epoch < 1 or pause_after_epoch >= config.max_epochs:
            raise ValueError("Pilot pause_after_epoch must be within [1, max_epochs-1]")
    loader_contracts = _validate_loaders(
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        scope=resolved_scope,
    )
    provenance_sha256 = _validate_provenance(config, provenance)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    base_root = _safe_run_root(project_root, config)
    resume_checkpoint: Path | None = None
    if resume_from is not None:
        resume_checkpoint, run_directory, resolved_run_id = _resolve_resume_checkpoint(
            project_root=project_root,
            base_root=base_root,
            resume_from=Path(resume_from),
            run_id=run_id,
        )
    else:
        if run_id is not None:
            _validate_run_id(run_id)
        run_directory = Path()
        resolved_run_id = run_id or ""

    # This is deliberately the first model-factory call in fit.
    trained_model = _seed_and_build_model(config, model)
    if not all(parameter.requires_grad for parameter in trained_model.parameters()):
        raise ValueError("Training requires a fresh model with trainable parameters")

    if resume_checkpoint is None:
        run_directory, resolved_run_id = _create_unique_run_directory(
            base_root, config, resolved_run_id or None
        )
        checkpoints = run_directory / "checkpoints"
        logs = run_directory / "logs"
        checkpoints.mkdir(exist_ok=False)
        logs.mkdir(exist_ok=False)
        metadata = _run_metadata(
            config=config,
            provenance=provenance,
            provenance_sha256=provenance_sha256,
            scope=resolved_scope,
            loader_contracts=loader_contracts,
            run_id=resolved_run_id,
        )
        _atomic_json(run_directory / "run_metadata.json", metadata)
        started_utc = metadata["created_utc"]
        resumed_from_text = None
    else:
        checkpoints = run_directory / "checkpoints"
        logs = run_directory / "logs"
        if not checkpoints.is_dir() or not logs.is_dir():
            raise ValueError("Resume run is missing checkpoints/logs directories")
        try:
            metadata = json.loads((run_directory / "run_metadata.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Resume run metadata is missing or corrupt") from exc
        _validate_run_metadata(
            metadata,
            config=config,
            provenance=provenance,
            provenance_sha256=provenance_sha256,
            scope=resolved_scope,
            loader_contracts=loader_contracts,
            run_id=resolved_run_id,
        )
        started_utc = str(metadata["created_utc"])
        resumed_from_text = str(resume_checkpoint.relative_to(project_root)).replace("\\", "/")

    counters = TrainingCounters()
    best = -math.inf
    history: list[dict[str, Any]] = []
    validations_without_improvement = 0
    start_epoch = 0
    status_path = run_directory / "run_status.json"
    _atomic_json(
        status_path,
        _status_record(
            status="initializing",
            run_id=resolved_run_id,
            config=config,
            scope=resolved_scope,
            counters=counters,
            started_utc=started_utc,
            resumed_from=resumed_from_text,
        ),
    )

    try:
        device = torch.device("cuda:0")
        trained_model.to(device)
        criterion = build_loss(config)
        optimizer = build_optimizer(trained_model, config)
        scheduler = build_scheduler(optimizer, config)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.automatic_mixed_precision,
            init_scale=config.amp_initial_scale,
        )
        if resume_checkpoint is not None:
            (
                start_epoch,
                best,
                validations_without_improvement,
                counters,
                history,
            ) = _load_resume_state(
                resume_checkpoint,
                model=trained_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                provenance=provenance,
                provenance_sha256=provenance_sha256,
                scope=resolved_scope,
                loader_contracts=loader_contracts,
                run_id=resolved_run_id,
                train_loader=train_loader,
                validation_loader=validation_loader,
            )
        else:
            # Publish an epoch-zero recovery point before consuming the first
            # batch. If the first epoch OOMs or the kernel is interrupted, Run
            # All can safely restart from initialized model/optimizer/RNG state.
            _write_checkpoint(
                checkpoints / "last.pt",
                kind="last",
                model=trained_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=0,
                best=best,
                validations_without_improvement=validations_without_improvement,
                config=config,
                provenance=provenance,
                provenance_sha256=provenance_sha256,
                scope=resolved_scope,
                loader_contracts=loader_contracts,
                run_id=resolved_run_id,
                counters=counters,
                history=history,
                train_loader=train_loader,
                validation_loader=validation_loader,
            )
        _atomic_json(
            status_path,
            _status_record(
                status="running",
                run_id=resolved_run_id,
                config=config,
                scope=resolved_scope,
                counters=counters,
                started_utc=started_utc,
                resumed_from=resumed_from_text,
            ),
        )
        class_names = {label.id: label.name for label in schema.labels}
        for epoch in range(start_epoch, config.max_epochs):
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            training_loss = _train_epoch(
                model=trained_model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                config=config,
                counters=counters,
            )
            scheduler.step()
            counters.scheduler_steps += 1
            record: dict[str, Any] = {"epoch": epoch + 1, "training_loss": training_loss}
            improved = False
            if (epoch + 1) % config.validation_interval_epochs == 0:
                validation = _validate(
                    model=trained_model,
                    loader=validation_loader,
                    device=device,
                    config=config,
                    spacing_mm=spacing_mm,
                    class_names=class_names,
                )
                counters.validation_runs_completed += 1
                record["validation"] = validation
                score = float(validation["mean_foreground_dice"])
                if math.isfinite(score) and score > best:
                    best = score
                    validations_without_improvement = 0
                    improved = True
                else:
                    validations_without_improvement += 1
            counters.epochs_completed += 1
            history.append(record)
            _atomic_json(logs / "history.json", history)
            if improved:
                _write_checkpoint(
                    checkpoints / "best.pt",
                    kind="best",
                    model=trained_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    best=best,
                    validations_without_improvement=validations_without_improvement,
                    config=config,
                    provenance=provenance,
                    provenance_sha256=provenance_sha256,
                    scope=resolved_scope,
                    loader_contracts=loader_contracts,
                    run_id=resolved_run_id,
                    counters=counters,
                    history=history,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                )
            _write_checkpoint(
                checkpoints / "last.pt",
                kind="last",
                model=trained_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                best=best,
                validations_without_improvement=validations_without_improvement,
                config=config,
                provenance=provenance,
                provenance_sha256=provenance_sha256,
                scope=resolved_scope,
                loader_contracts=loader_contracts,
                run_id=resolved_run_id,
                counters=counters,
                history=history,
                train_loader=train_loader,
                validation_loader=validation_loader,
            )
            _atomic_json(
                status_path,
                _status_record(
                    status="running",
                    run_id=resolved_run_id,
                    config=config,
                    scope=resolved_scope,
                    counters=counters,
                    started_utc=started_utc,
                    resumed_from=resumed_from_text,
                ),
            )
            if (
                validations_without_improvement
                >= config.early_stopping_patience_validations
            ):
                break
            if pause_after_epoch is not None and counters.epochs_completed >= pause_after_epoch:
                break

        best_value = best if math.isfinite(best) else None
        best_path = checkpoints / "best.pt"
        last_path = checkpoints / "last.pt"
        paused = (
            pause_after_epoch is not None
            and counters.epochs_completed >= pause_after_epoch
            and counters.epochs_completed < config.max_epochs
        )
        early_stopped = (
            not paused
            and counters.epochs_completed < config.max_epochs
            and validations_without_improvement >= config.early_stopping_patience_validations
        )
        terminal_status = "paused" if paused else "early_stopped" if early_stopped else "completed"
        result = {
            "schema_version": 1,
            "status": terminal_status,
            "run_id": resolved_run_id,
            "run_dir": str(run_directory),
            "resumed_from": resumed_from_text,
            "last_checkpoint": str(last_path) if last_path.is_file() else None,
            "best_checkpoint": str(best_path) if best_path.is_file() else None,
            "best_validation_mean_foreground_dice": best_value,
            "stopped_early": early_stopped,
            "paused_after_epoch": counters.epochs_completed if paused else None,
            "counters": asdict(counters),
            "history": _json_safe(history),
            "contract_checks": contract["checks"],
            "scope": resolved_scope.canonical,
            "scope_sha256": resolved_scope.fingerprint(),
            "completed_utc": _utc_now(),
        }
        _atomic_json(run_directory / ("paused.json" if paused else "completed.json"), result)
        _atomic_json(
            status_path,
            _status_record(
                status=terminal_status,
                run_id=resolved_run_id,
                config=config,
                scope=resolved_scope,
                counters=counters,
                started_utc=started_utc,
                resumed_from=resumed_from_text,
            ),
        )
        return result
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "run_id": resolved_run_id,
            "run_dir": str(run_directory),
            "resumed_from": resumed_from_text,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "counters": asdict(counters),
            "last_checkpoint": str(checkpoints / "last.pt")
            if (checkpoints / "last.pt").is_file()
            else None,
            "failed_utc": _utc_now(),
        }
        try:
            _atomic_json(run_directory / "failure.json", failure)
            _atomic_json(
                status_path,
                _status_record(
                    status="failed",
                    run_id=resolved_run_id,
                    config=config,
                    scope=resolved_scope,
                    counters=counters,
                    started_utc=started_utc,
                    resumed_from=resumed_from_text,
                    message=f"{type(exc).__name__}: {exc}",
                ),
            )
        except Exception as reporting_error:
            if hasattr(exc, "add_note"):
                exc.add_note(f"Could not write atomic failure report: {reporting_error}")
        raise
