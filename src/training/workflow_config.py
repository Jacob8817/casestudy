"""Strict configuration for the sequential training notebook.

The workflow file controls development-split scopes and gate thresholds.  It is
separate from the model YAMLs so the generic baseline configuration remains
usable by the training engine and every scope receives its own fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


GATE_SEQUENCE = ("one_case_overfit", "deterministic_pilot", "full_baseline")
AUTHORIZATION_TOKEN = "RUN_STAGED_BASELINE_TRAINING"
_CASE_ID = re.compile(r"s[0-9]{4}")


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    found = set(value)
    if found != expected:
        raise ValueError(
            f"{field} keys differ: missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )


def _relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must not be absolute or contain '..'")
    return path.as_posix()


def _case_ids(value: object, field: str, *, allow_full: bool) -> tuple[str, ...] | None:
    if allow_full and value == "official_full_split":
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a case-ID list")
    result = tuple(str(item) for item in value)
    if len(result) != len(set(result)) or any(_CASE_ID.fullmatch(item) is None for item in result):
        raise ValueError(f"{field} contains an invalid or duplicate case ID")
    return result


@dataclass(frozen=True)
class WorkflowStage:
    name: str
    baseline_config: str
    train_case_ids: tuple[str, ...] | None
    validation_case_ids: tuple[str, ...] | None
    fixed_sample_indices: tuple[int, ...]
    expected_selected_class_ids: tuple[int, ...]
    shuffle: bool
    acceptance: dict[str, Any]


@dataclass(frozen=True)
class TrainingWorkflowConfig:
    schema_version: int
    workflow_id: str
    status: str
    run_id: str
    authorization_token: str
    sequence: tuple[str, ...]
    interruption_policy: str
    passed_stage_policy: str
    test_split_locked: bool
    require_cuda: bool
    minimum_free_disk_gib: float
    estimated_complete_cache_gib: float
    runtime_note: str
    stages: dict[str, WorkflowStage]
    canonical_config: dict[str, Any]
    config_sha256: str


def load_training_workflow(path: Path) -> TrainingWorkflowConfig:
    """Load and fail-closed validate the staged notebook workflow."""
    try:
        raw = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load training workflow {path}: {exc}") from exc
    root = _mapping(raw, "training workflow")
    _exact_keys(root, {"schema_version", "workflow", "runtime", "stages"}, "training workflow")
    workflow = _mapping(root["workflow"], "workflow")
    runtime = _mapping(root["runtime"], "runtime")
    stages_raw = _mapping(root["stages"], "stages")
    _exact_keys(
        workflow,
        {
            "id", "status", "run_id", "authorization_token", "sequence",
            "interruption_policy", "passed_stage_policy", "test_split_locked",
        },
        "workflow",
    )
    _exact_keys(
        runtime,
        {"require_cuda", "minimum_free_disk_gib", "estimated_complete_cache_gib", "note"},
        "runtime",
    )
    _exact_keys(stages_raw, set(GATE_SEQUENCE), "stages")
    sequence = tuple(str(item) for item in workflow["sequence"])
    if sequence != GATE_SEQUENCE:
        raise ValueError(f"workflow.sequence must be {list(GATE_SEQUENCE)}")
    stages: dict[str, WorkflowStage] = {}
    stage_keys = {
        "baseline_config", "train_case_ids", "validation_case_ids", "fixed_sample_indices",
        "expected_selected_class_ids", "shuffle", "acceptance",
    }
    for name in GATE_SEQUENCE:
        stage = _mapping(stages_raw[name], f"stages.{name}")
        _exact_keys(stage, stage_keys, f"stages.{name}")
        fixed = tuple(int(item) for item in stage["fixed_sample_indices"])
        expected_classes = tuple(int(item) for item in stage["expected_selected_class_ids"])
        if any(item < 0 for item in fixed) or len(fixed) != len(set(fixed)):
            raise ValueError(f"stages.{name}.fixed_sample_indices must be unique and non-negative")
        if len(expected_classes) != len(fixed) or any(item not in range(1, 11) for item in expected_classes):
            raise ValueError(f"stages.{name} expected class IDs must align with its fixed foreground bank")
        stages[name] = WorkflowStage(
            name=name,
            baseline_config=_relative_path(stage["baseline_config"], f"stages.{name}.baseline_config"),
            train_case_ids=_case_ids(stage["train_case_ids"], f"stages.{name}.train_case_ids", allow_full=True),
            validation_case_ids=_case_ids(
                stage["validation_case_ids"], f"stages.{name}.validation_case_ids", allow_full=True
            ),
            fixed_sample_indices=fixed,
            expected_selected_class_ids=expected_classes,
            shuffle=bool(stage["shuffle"]),
            acceptance=dict(_mapping(stage["acceptance"], f"stages.{name}.acceptance")),
        )
    overfit, pilot, full = (stages[name] for name in GATE_SEQUENCE)
    if len(overfit.train_case_ids or ()) != 1 or overfit.validation_case_ids != () or not overfit.fixed_sample_indices:
        raise ValueError("The overfit gate must be one train case, a fixed patch bank, and no validation cases")
    if pilot.train_case_ids is None or len(pilot.train_case_ids) != 16:
        raise ValueError("The pilot must lock exactly 16 training cases")
    if pilot.validation_case_ids is None or len(pilot.validation_case_ids) != 4:
        raise ValueError("The pilot must lock exactly four validation cases")
    if full.train_case_ids is not None or full.validation_case_ids is not None:
        raise ValueError("The full baseline must use both complete official development splits")
    canonical = json.loads(json.dumps(root, sort_keys=True, allow_nan=False))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result = TrainingWorkflowConfig(
        schema_version=int(root["schema_version"]),
        workflow_id=str(workflow["id"]),
        status=str(workflow["status"]),
        run_id=str(workflow["run_id"]),
        authorization_token=str(workflow["authorization_token"]),
        sequence=sequence,
        interruption_policy=str(workflow["interruption_policy"]),
        passed_stage_policy=str(workflow["passed_stage_policy"]),
        test_split_locked=bool(workflow["test_split_locked"]),
        require_cuda=bool(runtime["require_cuda"]),
        minimum_free_disk_gib=float(runtime["minimum_free_disk_gib"]),
        estimated_complete_cache_gib=float(runtime["estimated_complete_cache_gib"]),
        runtime_note=str(runtime["note"]),
        stages=stages,
        canonical_config=canonical,
        config_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )
    if result.schema_version != 1 or result.status != "authorized_ready_to_run":
        raise ValueError("Only the authorized schema-v1 staged workflow is supported")
    if result.authorization_token != AUTHORIZATION_TOKEN:
        raise ValueError("Training workflow authorization token is invalid")
    if not result.test_split_locked or not result.require_cuda:
        raise ValueError("The staged workflow requires CUDA and a locked test split")
    if result.interruption_policy != "resume_last_completed_epoch":
        raise ValueError("The workflow must resume only from complete epoch checkpoints")
    if result.passed_stage_policy != "verify_report_then_skip":
        raise ValueError("Passed stages may be skipped only after report verification")
    if result.minimum_free_disk_gib < result.estimated_complete_cache_gib:
        raise ValueError("Free-space guard must exceed the estimated complete cache size")
    return result
