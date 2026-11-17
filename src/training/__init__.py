"""Reproducible Phase 4 baseline-training contracts."""

from .config import BaselineConfig, load_baseline_config, validate_baseline_contract
from .data import (
    ExactSplitPatchDataset,
    ExactSplitVolumeDataset,
    build_train_transform,
    build_validation_transform,
    prepare_case,
)
from .engine import (
    NonFiniteTrainingError,
    TrainingCounters,
    TrainingDisabledError,
    TrainingScope,
    assert_training_authorized,
    fit,
)
from .model import build_loss, build_model, build_optimizer, build_scheduler, model_state_sha256
from .provenance import atomic_write_json, hash_artifacts, sha256_file, training_artifact_inventory

__all__ = [
    "BaselineConfig",
    "load_baseline_config",
    "validate_baseline_contract",
    "ExactSplitPatchDataset",
    "ExactSplitVolumeDataset",
    "build_train_transform",
    "build_validation_transform",
    "prepare_case",
    "TrainingCounters",
    "TrainingDisabledError",
    "NonFiniteTrainingError",
    "TrainingScope",
    "assert_training_authorized",
    "fit",
    "build_loss",
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "model_state_sha256",
    "atomic_write_json",
    "hash_artifacts",
    "sha256_file",
    "training_artifact_inventory",
]
