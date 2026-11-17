"""Strict, fingerprinted configuration for the Phase 4 3D U-Net baseline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from src.dataset import OrganSchema, PreprocessingSpec


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    found = set(mapping)
    if found != expected:
        raise ValueError(f"{field} keys differ: missing={sorted(expected - found)}, extra={sorted(found - expected)}")


def _tuple(value: object, length: int, cast: type, field: str) -> tuple[Any, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-item list")
    return tuple(cast(item) for item in value)


def _relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must not be absolute or contain '..'")
    return path.as_posix()


@dataclass(frozen=True)
class BaselineConfig:
    schema_version: int
    experiment_id: str
    stage: str
    status: str
    execution_mode: str
    training_enabled: bool
    description: str
    manifest_sha256: str
    organ_schema_sha256: str
    preprocessing_sha256: str
    gpu_environment_id: str
    train_split: str
    validation_split: str
    test_split_locked: bool
    expected_split_counts: dict[str, int]
    roi_size: tuple[int, int, int]
    train_patches_per_case: int
    validation_mode: str
    cache_mode: str
    cache_root: str
    notebook_cache_write_enabled: bool
    batch_size: int
    production_workers: int
    notebook_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    laterality_unsafe_flips_enabled: bool
    affine_probability: float
    affine_rotate_range: tuple[float, float, float]
    affine_translate_range: tuple[float, float, float]
    affine_scale_range: tuple[float, float, float]
    affine_image_mode: str
    affine_label_mode: str
    affine_padding_mode: str
    noise_probability: float
    noise_mean: float
    noise_std: float
    contrast_probability: float
    contrast_gamma: tuple[float, float]
    output_clamp: tuple[float, float]
    model_implementation: str
    spatial_dims: int
    in_channels: int
    out_channels: int
    channels: tuple[int, ...]
    strides: tuple[int, ...]
    num_res_units: int
    normalization: str
    activation: str
    dropout: float
    loss_implementation: str
    loss_include_background: bool
    loss_to_onehot_y: bool
    loss_softmax: bool
    lambda_dice: float
    lambda_ce: float
    label_smoothing: float
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    scheduler_t_max_epochs: int
    max_epochs: int
    gradient_accumulation_steps: int
    automatic_mixed_precision: bool
    amp_initial_scale: float
    gradient_clip_norm: float
    validation_interval_epochs: int
    early_stopping_patience_validations: int
    validation_full_volume: bool
    validation_shuffle: bool
    sliding_window_roi_size: tuple[int, int, int]
    sliding_window_overlap: float
    sliding_window_batch_size: int
    sliding_window_blend_mode: str
    metric_include_background: bool
    metric_dice: bool
    metric_hd95: bool
    empty_target_policy: str
    random_seed: int
    deterministic_controls_enabled: bool
    strict_cuda_determinism: bool
    strict_cuda_determinism_limitation: str
    run_root: str
    setup_report: str
    augmentation_figure: str
    checkpoint_metric: str
    checkpoint_mode: str
    canonical_config: dict[str, Any]
    config_sha256: str

    def experiment_fingerprint(self) -> str:
        payload = {
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "organ_schema_sha256": self.organ_schema_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "gpu_environment_id": self.gpu_environment_id,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_baseline_config(path: Path) -> BaselineConfig:
    """Load a fail-closed baseline YAML and compute its canonical fingerprint."""
    try:
        raw = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load baseline config {path}: {exc}") from exc
    root = _mapping(raw, "baseline config")
    _exact_keys(
        root,
        {"schema_version", "experiment", "locks", "data", "augmentations", "model", "loss", "optimization", "training", "validation", "reproducibility", "outputs"},
        "baseline config",
    )
    experiment = _mapping(root["experiment"], "experiment")
    locks = _mapping(root["locks"], "locks")
    data = _mapping(root["data"], "data")
    cache = _mapping(data["cache"], "data.cache")
    loader = _mapping(data["loader"], "data.loader")
    augmentations = _mapping(root["augmentations"], "augmentations")
    affine = _mapping(augmentations["spatial_affine"], "augmentations.spatial_affine")
    noise = _mapping(augmentations["gaussian_noise"], "augmentations.gaussian_noise")
    contrast = _mapping(augmentations["contrast"], "augmentations.contrast")
    model = _mapping(root["model"], "model")
    loss = _mapping(root["loss"], "loss")
    optimization = _mapping(root["optimization"], "optimization")
    training = _mapping(root["training"], "training")
    validation = _mapping(root["validation"], "validation")
    sliding = _mapping(validation["sliding_window"], "validation.sliding_window")
    metrics = _mapping(validation["metrics"], "validation.metrics")
    reproducibility = _mapping(root["reproducibility"], "reproducibility")
    outputs = _mapping(root["outputs"], "outputs")
    expected_counts = _mapping(data["expected_split_counts"], "data.expected_split_counts")
    _exact_keys(experiment, {"id", "stage", "status", "execution_mode", "training_enabled", "description"}, "experiment")
    _exact_keys(locks, {"manifest_sha256", "organ_schema_sha256", "preprocessing_sha256", "gpu_environment_id"}, "locks")
    _exact_keys(data, {"train_split", "validation_split", "test_split_locked", "expected_split_counts", "roi_size", "train_patches_per_case", "validation_mode", "cache", "loader"}, "data")
    _exact_keys(cache, {"mode", "root", "notebook_write_enabled"}, "data.cache")
    _exact_keys(loader, {"batch_size", "production_workers", "notebook_workers", "pin_memory", "persistent_workers", "prefetch_factor"}, "data.loader")
    _exact_keys(augmentations, {"laterality_unsafe_flips_enabled", "spatial_affine", "gaussian_noise", "contrast", "output_clamp"}, "augmentations")
    _exact_keys(affine, {"enabled", "probability", "rotate_range_radians", "translate_range_voxels", "scale_range", "image_mode", "label_mode", "padding_mode"}, "augmentations.spatial_affine")
    _exact_keys(noise, {"enabled", "probability", "mean", "std"}, "augmentations.gaussian_noise")
    _exact_keys(contrast, {"enabled", "probability", "gamma_range"}, "augmentations.contrast")
    _exact_keys(model, {"implementation", "spatial_dims", "in_channels", "out_channels", "channels", "strides", "num_res_units", "normalization", "activation", "dropout"}, "model")
    _exact_keys(loss, {"implementation", "include_background", "to_onehot_y", "softmax", "lambda_dice", "lambda_ce", "label_smoothing"}, "loss")
    _exact_keys(optimization, {"optimizer", "learning_rate", "weight_decay", "scheduler", "scheduler_t_max_epochs"}, "optimization")
    training_keys = {
        "max_epochs",
        "gradient_accumulation_steps",
        "automatic_mixed_precision",
        "gradient_clip_norm",
        "validation_interval_epochs",
        "early_stopping_patience_validations",
    }
    if int(root["schema_version"]) == 2:
        training_keys.add("amp_initial_scale")
    _exact_keys(training, training_keys, "training")
    _exact_keys(validation, {"full_volume", "shuffle", "sliding_window", "metrics"}, "validation")
    _exact_keys(sliding, {"roi_size", "overlap", "sw_batch_size", "blend_mode"}, "validation.sliding_window")
    _exact_keys(metrics, {"include_background", "dice", "hd95", "empty_target_policy"}, "validation.metrics")
    _exact_keys(reproducibility, {"random_seed", "deterministic_controls_enabled", "strict_cuda_determinism", "strict_cuda_determinism_limitation"}, "reproducibility")
    _exact_keys(outputs, {"run_root", "setup_report", "augmentation_figure", "checkpoint_metric", "checkpoint_mode"}, "outputs")
    _exact_keys(expected_counts, {"train", "val", "test"}, "data.expected_split_counts")
    canonical = json.loads(json.dumps(root, sort_keys=True, allow_nan=False))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    config = BaselineConfig(
        schema_version=int(root["schema_version"]),
        experiment_id=str(experiment["id"]),
        stage=str(experiment["stage"]),
        status=str(experiment["status"]),
        execution_mode=str(experiment["execution_mode"]),
        training_enabled=bool(experiment["training_enabled"]),
        description=str(experiment["description"]),
        manifest_sha256=str(locks["manifest_sha256"]),
        organ_schema_sha256=str(locks["organ_schema_sha256"]),
        preprocessing_sha256=str(locks["preprocessing_sha256"]),
        gpu_environment_id=str(locks["gpu_environment_id"]),
        train_split=str(data["train_split"]),
        validation_split=str(data["validation_split"]),
        test_split_locked=bool(data["test_split_locked"]),
        expected_split_counts={str(key): int(value) for key, value in expected_counts.items()},
        roi_size=_tuple(data["roi_size"], 3, int, "data.roi_size"),
        train_patches_per_case=int(data["train_patches_per_case"]),
        validation_mode=str(data["validation_mode"]),
        cache_mode=str(cache["mode"]),
        cache_root=_relative_path(cache["root"], "data.cache.root"),
        notebook_cache_write_enabled=bool(cache["notebook_write_enabled"]),
        batch_size=int(loader["batch_size"]),
        production_workers=int(loader["production_workers"]),
        notebook_workers=int(loader["notebook_workers"]),
        pin_memory=bool(loader["pin_memory"]),
        persistent_workers=bool(loader["persistent_workers"]),
        prefetch_factor=int(loader["prefetch_factor"]),
        laterality_unsafe_flips_enabled=bool(augmentations["laterality_unsafe_flips_enabled"]),
        affine_probability=float(affine["probability"]) if affine["enabled"] else 0.0,
        affine_rotate_range=_tuple(affine["rotate_range_radians"], 3, float, "rotate_range_radians"),
        affine_translate_range=_tuple(affine["translate_range_voxels"], 3, float, "translate_range_voxels"),
        affine_scale_range=_tuple(affine["scale_range"], 3, float, "scale_range"),
        affine_image_mode=str(affine["image_mode"]),
        affine_label_mode=str(affine["label_mode"]),
        affine_padding_mode=str(affine["padding_mode"]),
        noise_probability=float(noise["probability"]) if noise["enabled"] else 0.0,
        noise_mean=float(noise["mean"]),
        noise_std=float(noise["std"]),
        contrast_probability=float(contrast["probability"]) if contrast["enabled"] else 0.0,
        contrast_gamma=_tuple(contrast["gamma_range"], 2, float, "gamma_range"),
        output_clamp=_tuple(augmentations["output_clamp"], 2, float, "output_clamp"),
        model_implementation=str(model["implementation"]),
        spatial_dims=int(model["spatial_dims"]),
        in_channels=int(model["in_channels"]),
        out_channels=int(model["out_channels"]),
        channels=tuple(int(value) for value in model["channels"]),
        strides=tuple(int(value) for value in model["strides"]),
        num_res_units=int(model["num_res_units"]),
        normalization=str(model["normalization"]),
        activation=str(model["activation"]),
        dropout=float(model["dropout"]),
        loss_implementation=str(loss["implementation"]),
        loss_include_background=bool(loss["include_background"]),
        loss_to_onehot_y=bool(loss["to_onehot_y"]),
        loss_softmax=bool(loss["softmax"]),
        lambda_dice=float(loss["lambda_dice"]),
        lambda_ce=float(loss["lambda_ce"]),
        label_smoothing=float(loss["label_smoothing"]),
        optimizer=str(optimization["optimizer"]),
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        scheduler=str(optimization["scheduler"]),
        scheduler_t_max_epochs=int(optimization["scheduler_t_max_epochs"]),
        max_epochs=int(training["max_epochs"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        automatic_mixed_precision=bool(training["automatic_mixed_precision"]),
        amp_initial_scale=float(training.get("amp_initial_scale", 1024.0)),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        validation_interval_epochs=int(training["validation_interval_epochs"]),
        early_stopping_patience_validations=int(training["early_stopping_patience_validations"]),
        validation_full_volume=bool(validation["full_volume"]),
        validation_shuffle=bool(validation["shuffle"]),
        sliding_window_roi_size=_tuple(sliding["roi_size"], 3, int, "sliding_window.roi_size"),
        sliding_window_overlap=float(sliding["overlap"]),
        sliding_window_batch_size=int(sliding["sw_batch_size"]),
        sliding_window_blend_mode=str(sliding["blend_mode"]),
        metric_include_background=bool(metrics["include_background"]),
        metric_dice=bool(metrics["dice"]),
        metric_hd95=bool(metrics["hd95"]),
        empty_target_policy=str(metrics["empty_target_policy"]),
        random_seed=int(reproducibility["random_seed"]),
        deterministic_controls_enabled=bool(reproducibility["deterministic_controls_enabled"]),
        strict_cuda_determinism=bool(reproducibility["strict_cuda_determinism"]),
        strict_cuda_determinism_limitation=str(reproducibility["strict_cuda_determinism_limitation"]),
        run_root=_relative_path(outputs["run_root"], "outputs.run_root"),
        setup_report=_relative_path(outputs["setup_report"], "outputs.setup_report"),
        augmentation_figure=_relative_path(outputs["augmentation_figure"], "outputs.augmentation_figure"),
        checkpoint_metric=str(outputs["checkpoint_metric"]),
        checkpoint_mode=str(outputs["checkpoint_mode"]),
        canonical_config=canonical,
        config_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )
    _validate_internal(config)
    return config


def _validate_internal(config: BaselineConfig) -> None:
    setup_contract = (
        config.schema_version == 1
        and config.execution_mode == "setup_only"
        and not config.training_enabled
        and config.status == "configured_not_trained"
    )
    training_contract = (
        config.schema_version == 2
        and config.execution_mode == "training"
        and config.training_enabled
        and config.status == "authorized_not_started"
    )
    if not (setup_contract or training_contract):
        raise ValueError(
            "Use schema v1/configured_not_trained for setup-only mode or a separately versioned "
            "schema v2/authorized_not_started config for training mode"
        )
    if (config.train_split, config.validation_split) != ("train", "val") or not config.test_split_locked:
        raise ValueError("Baseline split contract must be train/val with a locked test split")
    if config.expected_split_counts != {"train": 1082, "val": 57, "test": 89}:
        raise ValueError("Expected split counts must lock the audited official split")
    if config.roi_size != config.sliding_window_roi_size or any(value <= 0 for value in config.roi_size):
        raise ValueError("Training and sliding-window ROI sizes must match and be positive")
    if config.validation_mode != "deterministic_full_volume" or not config.validation_full_volume or config.validation_shuffle:
        raise ValueError("Validation must be deterministic, full-volume, and unshuffled")
    if config.laterality_unsafe_flips_enabled:
        raise ValueError("Naive flips are forbidden because the schema contains left/right classes")
    if config.affine_label_mode != "nearest" or config.affine_image_mode != "bilinear":
        raise ValueError("Paired affine interpolation must be bilinear CT and nearest label")
    if config.affine_padding_mode != "zeros" or config.output_clamp != (0.0, 1.0):
        raise ValueError("Normalized CT augmentation must use zero padding and [0,1] clamping")
    probabilities = (config.affine_probability, config.noise_probability, config.contrast_probability)
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("Augmentation probabilities must lie in [0,1]")
    if len(config.channels) != len(config.strides) + 1 or any(value <= 0 for value in config.channels + config.strides):
        raise ValueError("UNet channels/strides are inconsistent")
    divisor = 1
    for stride in config.strides:
        divisor *= stride
    if any(size % divisor for size in config.roi_size):
        raise ValueError(f"ROI must be divisible by cumulative stride {divisor}")
    if config.spatial_dims != 3 or config.in_channels != 1 or config.out_channels != 11:
        raise ValueError("Baseline model must be 3D, single-channel, and 11-class")
    if config.model_implementation != "monai.networks.nets.UNet" or config.loss_implementation != "monai.losses.DiceCELoss":
        raise ValueError("Unexpected baseline model/loss implementation")
    if config.loss_include_background or not config.loss_to_onehot_y or not config.loss_softmax:
        raise ValueError("DiceCE contract must exclude background Dice and use one-hot softmax")
    if config.optimizer != "AdamW" or config.scheduler != "CosineAnnealingLR":
        raise ValueError("Baseline optimizer/scheduler contract changed")
    if config.checkpoint_metric != "validation_mean_foreground_dice" or config.checkpoint_mode != "max":
        raise ValueError("Baseline checkpoint selection must maximize validation mean foreground Dice")
    if not config.metric_dice or not config.metric_hd95 or config.metric_include_background:
        raise ValueError("Baseline validation must report foreground Dice and HD95")
    if min(config.batch_size, config.train_patches_per_case, config.gradient_accumulation_steps, config.max_epochs) <= 0:
        raise ValueError("Training counts must be positive")
    if not math.isfinite(config.amp_initial_scale) or config.amp_initial_scale <= 0:
        raise ValueError("AMP initial scale must be finite and positive")
    if config.cache_mode != "lazy_fingerprinted_disk":
        raise ValueError("The baseline requires the lazy fingerprinted disk cache")
    if config.execution_mode == "setup_only" and config.notebook_cache_write_enabled:
        raise ValueError("Setup-only notebook cache writes must remain disabled")
    if config.empty_target_policy != "nan_excluded_from_class_mean":
        raise ValueError("Partial-field validation requires NaN exclusion for absent targets")
    if config.strict_cuda_determinism:
        raise ValueError("Strict CUDA determinism is not supported by the validated 5D CrossEntropy kernel")


def validate_baseline_contract(
    config: BaselineConfig,
    *,
    manifest: dict[str, Any],
    schema: OrganSchema,
    preprocessing: PreprocessingSpec,
    gpu_report: dict[str, Any],
) -> dict[str, Any]:
    """Cross-check the baseline locks against already-validated project artifacts."""
    split_counts = {split: 0 for split in ("train", "val", "test")}
    for case in manifest["cases"]:
        split_counts[str(case["split"])] += 1
    preprocessing_sha256 = preprocessing.fingerprint(
        manifest_sha256=manifest["manifest_sha256"], schema_sha256=schema.fingerprint()
    )
    checks = {
        "manifest_sha256": manifest["manifest_sha256"] == config.manifest_sha256,
        "organ_schema_sha256": schema.fingerprint() == config.organ_schema_sha256,
        "preprocessing_sha256": preprocessing_sha256 == config.preprocessing_sha256,
        "split_counts": split_counts == config.expected_split_counts,
        "output_channels": schema.num_output_channels == config.out_channels,
        "roi_size": preprocessing.roi_size == config.roi_size,
        "gpu_report_passed": gpu_report.get("status") == "passed" and gpu_report.get("all_checks_passed") is True,
        "gpu_environment_id": gpu_report.get("provenance", {}).get("environment_id") == config.gpu_environment_id,
        "test_locked": manifest.get("split_policy", {}).get("test_lock", {}).get("locked") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Baseline contract checks failed: {failed}")
    return {"checks": checks, "split_counts": split_counts, "preprocessing_sha256": preprocessing_sha256}
