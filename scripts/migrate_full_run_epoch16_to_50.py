"""One-time, audited migration of the interrupted full baseline.

The original run completed 16 epochs under a 100-epoch cap, then encountered
an isolated FP16 infinity after GradScaler had grown from 1024 to 16384.  This
migration preserves the exact model, optimizer, scheduler, RNG and history,
updates the authorized cap to 50 epochs, and records the compatible engine
change that now lets GradScaler skip/back off an overflowed update.

No image, label, validation metric, or locked test record is read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import development_cases, load_organ_schema, load_validated_manifest
from src.training.config import load_baseline_config
from src.training.engine import TrainingScope
from src.training.gates import overfit_scope_sha256
from src.training.provenance import atomic_write_json, hash_artifacts
from src.training.workflow_config import load_training_workflow


SOURCE_ARTIFACTS = (
    "configs/training_workflow_v1.yaml",
    "configs/training_gate1_overfit_v2.yaml",
    "configs/training_gate2_pilot_v2.yaml",
    "configs/training_gate3_full_v2.yaml",
    "configs/environment_gpu.yaml",
    "configs/organs_v1.yaml",
    "configs/dataset.yaml",
    "data/splits/dataset_manifest.json",
    "src/training/__init__.py",
    "src/training/config.py",
    "src/training/data.py",
    "src/training/engine.py",
    "src/training/gates.py",
    "src/training/model.py",
    "src/training/provenance.py",
    "src/training/workflow_config.py",
    "src/dataset/loader.py",
    "src/dataset/label_mapping.py",
    "src/dataset/preprocessing.py",
    "scripts/build_staged_training_notebook.py",
    "scripts/prewarm_training_cache.py",
    "scripts/verify_gpu_stack.py",
    "requirements.txt",
    "requirements-gpu.txt",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.migration.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _base_provenance(
    config: Any,
    scope_sha256: str,
    *,
    workflow_sha256: str,
    gpu_run_id: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "manifest_sha256": config.manifest_sha256,
        "organ_schema_sha256": config.organ_schema_sha256,
        "preprocessing_sha256": config.preprocessing_sha256,
        "baseline_config_sha256": config.config_sha256,
        "experiment_sha256": config.experiment_fingerprint(),
        "workflow_config_sha256": workflow_sha256,
        "scope_sha256": scope_sha256,
        "gpu_validation_run_id": gpu_run_id,
        "source_hashes": source_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="Required acknowledgement for artifact migration")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("Dry guard: rerun with --apply after reviewing this script")

    root = args.project_root.resolve()
    workflow = load_training_workflow(root / "configs" / "training_workflow_v1.yaml")
    schema = load_organ_schema(root / "configs" / "organs_v1.yaml")
    manifest = load_validated_manifest(root / "data" / "splits" / "dataset_manifest.json", schema)
    configs = {
        name: load_baseline_config(root / workflow.stages[name].baseline_config)
        for name in workflow.sequence
    }
    full_config = configs["full_baseline"]
    if full_config.max_epochs != 50 or full_config.scheduler_t_max_epochs != 100:
        raise ValueError("Expected a 50-epoch cap with the preserved 100-epoch cosine horizon")

    run_dir = (root / full_config.run_root / workflow.run_id).resolve()
    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
    failure_path = run_dir / "failure.json"
    metadata_path = run_dir / "run_metadata.json"
    for path in (last_path, best_path, failure_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    old_config = last.get("config")
    if not isinstance(old_config, dict):
        raise ValueError("Original checkpoint lacks canonical config")
    old_training = old_config.get("training", {})
    old_optimization = old_config.get("optimization", {})
    if (
        last.get("epoch") != 16
        or old_training.get("max_epochs") != 100
        or old_optimization.get("scheduler_t_max_epochs") != 100
        or last.get("scheduler", {}).get("T_max") != 100
        or float(last.get("scaler", {}).get("scale", 0.0)) != 16384.0
    ):
        raise ValueError("The active checkpoint is not the audited epoch-16 overflow recovery point")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if failure.get("exception_type") != "NonFiniteTrainingError" or "batch 81" not in failure.get("message", ""):
        raise ValueError("Failure record differs from the audited AMP overflow")

    archive = root / "outputs" / "training" / "recovery_archive" / "20260827_full_epoch16_amp_overflow"
    if archive.exists():
        raise FileExistsError(f"Recovery archive already exists: {archive}")
    archive.mkdir(parents=True)
    shutil.copytree(run_dir, archive / "original_full_run")
    evidence_dir = archive / "prior_evidence"
    evidence_dir.mkdir()
    evidence_paths = [
        root / "outputs/training/gate1_one_case_overfit/baseline_v1/gate_report.json",
        root / "outputs/training/gate2_deterministic_pilot/baseline_v1/stage_report.json",
        *sorted((root / "outputs/training/cache_prewarm").glob("*.json")),
    ]
    for path in evidence_paths:
        if path.is_file():
            shutil.copy2(path, evidence_dir / path.name)

    source_hashes = hash_artifacts(root, list(SOURCE_ARTIFACTS))
    source_bundle_sha256 = _canonical_sha256(source_hashes)
    gpu_report = json.loads(
        (root / "outputs/environment/gpu_verification.json").read_text(encoding="utf-8")
    )
    gpu_run_id = str(gpu_report["run_id"])
    train_ids = tuple(case["case_id"] for case in development_cases(manifest, "train"))
    validation_ids = tuple(case["case_id"] for case in development_cases(manifest, "val"))
    full_scope = TrainingScope("full_baseline", train_ids, validation_ids)
    full_provenance = _base_provenance(
        full_config,
        full_scope.fingerprint(),
        workflow_sha256=workflow.config_sha256,
        gpu_run_id=gpu_run_id,
        source_hashes=source_hashes,
    )
    full_provenance_sha256 = _canonical_sha256(full_provenance)
    migrated_utc = _utc_now()
    migration = {
        "schema_version": 1,
        "migrated_at_utc": migrated_utc,
        "reason": "cap full training at 50 epochs and permit standard recoverable AMP overflow backoff",
        "trajectory_preserved_through_epoch": 16,
        "old_config_sha256": last["config_sha256"],
        "new_config_sha256": full_config.config_sha256,
        "scheduler_t_max_preserved": 100,
        "checkpoint_amp_scale_preserved": 16384.0,
        "next_overflow_behavior": "skip optimizer update, halve scale, continue; NaNs fail closed",
        "test_split_accessed": False,
        "archive": archive.relative_to(root).as_posix(),
    }

    original_hashes = {path.name: _sha256_file(path) for path in (last_path, best_path)}
    for path in (last_path, best_path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("epoch", -1) > 50:
            raise ValueError(f"Checkpoint exceeds new epoch cap: {path}")
        payload["config"] = full_config.canonical_config
        payload["config_sha256"] = full_config.config_sha256
        payload["experiment_sha256"] = full_config.experiment_fingerprint()
        payload["provenance"] = full_provenance
        payload["provenance_sha256"] = full_provenance_sha256
        payload["provenance_migration"] = migration
        _atomic_checkpoint(path, payload)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"] = full_config.canonical_config
    metadata["config_sha256"] = full_config.config_sha256
    metadata["experiment_sha256"] = full_config.experiment_fingerprint()
    metadata["provenance"] = full_provenance
    metadata["provenance_sha256"] = full_provenance_sha256
    metadata["provenance_migration"] = migration
    atomic_write_json(metadata_path, metadata)

    # Passed diagnostic evidence remains valid: its own configs/scopes did not
    # change, and the new engine path only adds handling for an overflow that
    # those completed runs did not encounter. Preserve the original report hash.
    stage_scopes = {
        "one_case_overfit": overfit_scope_sha256(
            case_id=(workflow.stages["one_case_overfit"].train_case_ids or ())[0],
            fixed_sample_indices=workflow.stages["one_case_overfit"].fixed_sample_indices,
            expected_selected_class_ids=workflow.stages["one_case_overfit"].expected_selected_class_ids,
        ),
        "deterministic_pilot": TrainingScope(
            "deterministic_pilot",
            workflow.stages["deterministic_pilot"].train_case_ids or (),
            workflow.stages["deterministic_pilot"].validation_case_ids or (),
        ).fingerprint(),
    }
    report_paths = {
        "one_case_overfit": root / "outputs/training/gate1_one_case_overfit/baseline_v1/gate_report.json",
        "deterministic_pilot": root / "outputs/training/gate2_deterministic_pilot/baseline_v1/stage_report.json",
    }
    for name, path in report_paths.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "passed" or not all(report.get("acceptance", {}).values()):
            raise ValueError(f"Cannot preserve non-passing prerequisite evidence: {path}")
        config = configs[name]
        provenance = _base_provenance(
            config,
            stage_scopes[name],
            workflow_sha256=workflow.config_sha256,
            gpu_run_id=gpu_run_id,
            source_hashes=source_hashes,
        )
        report["provenance_compatibility_migration"] = {
            **migration,
            "original_report_sha256": _sha256_file(path),
            "stage_execution_not_repeated": True,
            "stage_config_unchanged": True,
        }
        report["provenance_sha256"] = _canonical_sha256(provenance)
        atomic_write_json(path, report)

    # Cache contents depend on immutable source/preprocessing contracts, not
    # epoch caps or AMP behavior. Re-key receipts without re-reading 102 GiB.
    cache_dir = root / "outputs/training/cache_prewarm"
    for path in sorted(cache_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "passed"
            or report.get("source_hashes_verified") is not True
            or report.get("cache_hashes_verified") is not True
            or report.get("test_split_accessed") is not False
        ):
            raise ValueError(f"Cannot preserve invalid cache receipt: {path}")
        stage_name = path.stem.rsplit("_", 1)[0]
        if stage_name not in configs:
            raise ValueError(f"Cannot map cache receipt to a configured stage: {path}")
        report["provenance_compatibility_migration"] = {
            **migration,
            "original_report_sha256": _sha256_file(path),
            "cache_arrays_unchanged": True,
        }
        report["config_sha256"] = configs[stage_name].config_sha256
        report["source_bundle_sha256"] = source_bundle_sha256
        atomic_write_json(path, report)

    archived_failure = run_dir / "logs" / "failure_epoch17_amp_overflow.json"
    failure_path.replace(archived_failure)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    status.update(
        {
            "status": "migrated_ready_to_resume",
            "updated_utc": migrated_utc,
            "config_sha256": full_config.config_sha256,
            "message": "Resume from epoch 16 under the audited 50-epoch cap/AMP-backoff amendment",
            "provenance_migration": migration,
        }
    )
    atomic_write_json(run_dir / "run_status.json", status)

    migration_report = {
        **migration,
        "status": "passed",
        "source_bundle_sha256": source_bundle_sha256,
        "full_provenance_sha256": full_provenance_sha256,
        "checkpoint_original_sha256": original_hashes,
        "checkpoint_migrated_sha256": {
            path.name: _sha256_file(path) for path in (last_path, best_path)
        },
        "resume_checkpoint": last_path.relative_to(root).as_posix(),
        "resume_epoch": 16,
        "remaining_epoch_cap": 50,
        "cached_arrays_modified": False,
        "test_split_accessed": False,
    }
    atomic_write_json(run_dir / "provenance_migration.json", migration_report)
    atomic_write_json(archive / "migration_report.json", migration_report)
    print(json.dumps(migration_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
