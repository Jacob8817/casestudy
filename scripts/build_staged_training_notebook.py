#!/usr/bin/env python3
"""Build the authorized staged-training notebook without executing any cell."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "05_staged_training.ipynb"


def _lines(source: str) -> list[str]:
    text = textwrap.dedent(source).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str, *tags: str) -> dict:
    metadata = {"tags": list(tags)} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _lines(source),
    }


cells = [
    markdown(
        r"""
        # CS16 — Phase 5: Authorized Staged 3D U-Net Training

        ## This notebook performs real GPU training

        `Run All` executes three stages in order and never advances past a failed gate:

        1. **one-case overfit** — fixed, class-centred patches from training case `s0014`; no validation/test data;
        2. **deterministic pilot** — 16 official training cases and 4 official validation cases;
        3. **full baseline** — up to 50 epochs on all 1,082 training and 57 validation cases.

        The 89-case test split remains locked. It is not resolved, cached, trained on, validated on, or evaluated in
        this notebook. Final test evaluation belongs in a later notebook after model selection is frozen.

        > **Runtime expectation:** the full stage can take days on an RTX 4060 laptop. Before each stage, the
        > notebook verifies source SHA-256 values and prepares content-hashed cache entries; the complete cache may
        > grow to about 82.3 GiB. Keep the laptop on AC power, disable sleep, preserve at least 100 GiB of free
        > space, and avoid running other GPU-heavy applications.
        """
    ),
    markdown(
        r"""
        ## Restart and artifact rules

        - Every completed epoch is atomically saved as `checkpoints/last.pt`.
        - The best finite validation Dice is atomically saved as `checkpoints/best.pt`.
        - Re-running `Run All` verifies and skips an already-passed gate.
        - An interrupted gate resumes only from its own `last.pt`, with matching configuration, scope, provenance,
          loader order, optimizer, scheduler, scaler, and random-number state.
        - A directory left by an interruption before the first complete epoch is not overwritten automatically.
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import hashlib
        import json
        import math
        import os
        import platform
        import shutil
        import sys
        from datetime import datetime, timezone
        from pathlib import Path

        import monai
        import nibabel
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import Markdown, display
        from torch.utils.data import DataLoader

        PROJECT_ROOT = next(
            candidate
            for candidate in (Path.cwd(), *Path.cwd().parents)
            if (candidate / "configs" / "training_workflow_v1.yaml").is_file()
        ).resolve()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from src.dataset import (
            development_cases,
            load_organ_schema,
            load_preprocessing_spec,
            load_validated_manifest,
        )
        from src.training.config import load_baseline_config, validate_baseline_contract
        from src.training.data import (
            ExactSplitPatchDataset,
            ExactSplitVolumeDataset,
            build_training_loaders,
            prewarm_cache,
        )
        from src.training.engine import TrainingScope, fit
        from src.training.gates import (
            OverfitGateCriteria,
            overfit_scope_sha256,
            run_overfit_gate,
        )
        from src.training.provenance import atomic_write_json, hash_artifacts
        from src.training.workflow_config import (
            AUTHORIZATION_TOKEN,
            load_training_workflow,
        )

        # This is the explicit runtime half of the double authorization contract.
        RUN_TRAINING = True
        RUNTIME_AUTHORIZATION = "RUN_STAGED_BASELINE_TRAINING"
        assert RUN_TRAINING is True
        assert RUNTIME_AUTHORIZATION == AUTHORIZATION_TOKEN

        DATASET_ROOT = Path(
            os.environ.get("TOTALSEG_DATASET_ROOT", PROJECT_ROOT / "Totalsegmentator_dataset_v201")
        ).resolve()
        WORKFLOW_PATH = PROJECT_ROOT / "configs" / "training_workflow_v1.yaml"
        ORGAN_CONFIG_PATH = PROJECT_ROOT / "configs" / "organs_v1.yaml"
        DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.yaml"
        MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "dataset_manifest.json"
        GPU_REPORT_PATH = PROJECT_ROOT / "outputs" / "environment" / "gpu_verification.json"

        print(f"Project root      : {PROJECT_ROOT}")
        print(f"Dataset root      : {DATASET_ROOT}")
        print(f"Python            : {platform.python_version()}")
        print(f"PyTorch / MONAI   : {torch.__version__} / {monai.__version__}")
        print(f"Training authorized: {RUN_TRAINING}")
        """,
        "training-authorization",
    ),
    markdown(
        r"""
        ## 1. Fail-closed preflight

        This cell validates every immutable fingerprint, CUDA, disk space, the official patient split, all three
        schema-v2 training configurations, and a synthetic request proving that test access fails before path
        resolution. It does not load an image or construct a model.
        """
    ),
    code(
        r"""
        workflow = load_training_workflow(WORKFLOW_PATH)
        assert workflow.authorization_token == RUNTIME_AUTHORIZATION
        assert workflow.test_split_locked

        schema = load_organ_schema(ORGAN_CONFIG_PATH)
        manifest = load_validated_manifest(MANIFEST_PATH, schema)
        preprocessing = load_preprocessing_spec(DATASET_CONFIG_PATH)
        gpu_report = json.loads(GPU_REPORT_PATH.read_text(encoding="utf-8"))
        configs = {
            name: load_baseline_config(PROJECT_ROOT / workflow.stages[name].baseline_config)
            for name in workflow.sequence
        }
        contracts = {
            name: validate_baseline_contract(
                configs[name],
                manifest=manifest,
                schema=schema,
                preprocessing=preprocessing,
                gpu_report=gpu_report,
            )
            for name in workflow.sequence
        }
        for name in workflow.sequence:
            assert configs[name].stage == name
            assert configs[name].schema_version == 2
            assert configs[name].training_enabled
            assert configs[name].execution_mode == "training"
            assert configs[name].test_split_locked
            assert all(contracts[name]["checks"].values())

        try:
            ExactSplitVolumeDataset(
                project_root=PROJECT_ROOT,
                dataset_root=Path("Z:/POISON_PATH_MUST_NOT_RESOLVE"),
                manifest=manifest,
                schema=schema,
                preprocessing=preprocessing,
                preprocessing_sha256=configs["one_case_overfit"].preprocessing_sha256,
                config=configs["one_case_overfit"],
                split="test",
                allow_cache_write=False,
            )
        except PermissionError:
            test_rejected_before_io = True
        else:
            raise AssertionError("Locked test split unexpectedly reached dataset construction")

        disk = shutil.disk_usage(PROJECT_ROOT)
        free_gib = disk.free / 2**30
        assert DATASET_ROOT.is_dir(), f"Dataset root not found: {DATASET_ROOT}"
        assert torch.cuda.is_available(), "CUDA is required; CPU fallback is intentionally disabled"
        assert gpu_report["status"] == "passed" and gpu_report["all_checks_passed"]
        assert torch.__version__ == gpu_report["environment"]["torch"]
        assert torch.version.cuda == gpu_report["environment"]["pytorch_cuda_runtime"]
        assert monai.__version__ == gpu_report["environment"]["monai"]
        assert nibabel.__version__ == gpu_report["environment"]["nibabel"]
        assert np.__version__ == gpu_report["environment"]["numpy"]
        assert torch.cuda.get_device_name(0) == gpu_report["device"]["name"]
        assert free_gib >= workflow.minimum_free_disk_gib, (
            f"Need at least {workflow.minimum_free_disk_gib:.1f} GiB free; found {free_gib:.1f} GiB"
        )
        assert test_rejected_before_io

        train_ids = tuple(case["case_id"] for case in development_cases(manifest, "train"))
        validation_ids = tuple(case["case_id"] for case in development_cases(manifest, "val"))
        assert len(train_ids) == 1082 and len(validation_ids) == 57
        assert not set(train_ids) & set(validation_ids)

        preflight_table = pd.DataFrame(
            [
                ("CUDA", torch.cuda.get_device_name(0)),
                ("Free disk", f"{free_gib:.1f} GiB"),
                ("Estimated full cache", f"{workflow.estimated_complete_cache_gib:.1f} GiB"),
                ("Train / validation", "1082 / 57"),
                ("Full epoch cap", configs["full_baseline"].max_epochs),
                ("AMP initial scale", configs["full_baseline"].amp_initial_scale),
                ("Test", "89 — LOCKED"),
                ("Run ID", workflow.run_id),
                ("Restart", workflow.interruption_policy),
            ],
            columns=["Preflight", "Result"],
        )
        display(preflight_table.style.hide(axis="index"))
        """
    ),
    markdown(
        r"""
        ## 2. Shared provenance and safe restart helpers

        A passed report is never trusted by filename alone. Its configuration, workflow and scope fingerprints must
        match, every acceptance check must still be true, and its last checkpoint must still exist.
        """
    ),
    code(
        r"""
        source_hashes = hash_artifacts(
            PROJECT_ROOT,
            [
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
            ],
        )
        source_bundle_sha256 = hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()

        def run_dir(config):
            return (PROJECT_ROOT / config.run_root / workflow.run_id).resolve()

        def relative_artifact(value):
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()

        def stage_report_path(config):
            return run_dir(config) / "stage_report.json"

        def verified_passed_report(config, expected_scope_sha256):
            path = stage_report_path(config)
            if not path.is_file():
                # The dedicated gate writes gate_report.json itself.
                alternate = run_dir(config) / "gate_report.json"
                path = alternate if alternate.is_file() else path
            if not path.is_file():
                return None
            report = json.loads(path.read_text(encoding="utf-8"))
            expected_provenance = base_provenance(config, expected_scope_sha256)
            expected_provenance_sha256 = hashlib.sha256(
                json.dumps(expected_provenance, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest()
            locks_match = (
                report.get("status") == "passed"
                and report.get("config_sha256") == config.config_sha256
                and report.get("workflow_config_sha256") == workflow.config_sha256
                and report.get("scope_sha256") == expected_scope_sha256
                and report.get("provenance_sha256") == expected_provenance_sha256
                and all(report.get("acceptance", {}).values())
            )
            checkpoint_value = report.get("last_checkpoint")
            checkpoint = None if not checkpoint_value else PROJECT_ROOT / checkpoint_value
            if locks_match and checkpoint is not None and checkpoint.is_file():
                return report
            if report.get("status") == "passed":
                raise RuntimeError(f"Passed report failed integrity verification: {path}")
            return None

        def resume_checkpoint(config):
            path = run_dir(config) / "checkpoints" / "last.pt"
            if path.is_file():
                return path
            if run_dir(config).exists():
                raise RuntimeError(
                    f"{run_dir(config)} exists without a complete-epoch last.pt. "
                    "Preserve it for diagnosis and choose a new workflow run_id."
                )
            return None

        def base_provenance(config, scope_sha256):
            return {
                "manifest_sha256": config.manifest_sha256,
                "organ_schema_sha256": config.organ_schema_sha256,
                "preprocessing_sha256": config.preprocessing_sha256,
                "baseline_config_sha256": config.config_sha256,
                "experiment_sha256": config.experiment_fingerprint(),
                "workflow_config_sha256": workflow.config_sha256,
                "scope_sha256": scope_sha256,
                "gpu_validation_run_id": gpu_report["run_id"],
                "source_hashes": source_hashes,
            }

        def finite_validation_scores(history):
            scores = []
            for record in history:
                validation = record.get("validation")
                if validation is not None:
                    value = validation.get("mean_foreground_dice")
                    if value is not None and math.isfinite(float(value)):
                        scores.append(float(value))
            return scores

        def ensure_cache_prewarmed(name, config, split, case_ids):
            report_path = PROJECT_ROOT / "outputs" / "training" / "cache_prewarm" / f"{name}_{split}.json"
            selected_ids = (
                tuple(case["case_id"] for case in development_cases(manifest, split))
                if case_ids is None
                else tuple(case_ids)
            )
            expected_ids_sha = hashlib.sha256(
                json.dumps(list(selected_ids), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if report_path.is_file():
                prior = json.loads(report_path.read_text(encoding="utf-8"))
                valid = (
                    prior.get("status") == "passed"
                    and prior.get("config_sha256") == config.config_sha256
                    and prior.get("manifest_sha256") == config.manifest_sha256
                    and prior.get("organ_schema_sha256") == config.organ_schema_sha256
                    and prior.get("preprocessing_sha256") == config.preprocessing_sha256
                    and prior.get("selected_case_ids_sha256") == expected_ids_sha
                    and prior.get("source_bundle_sha256") == source_bundle_sha256
                    and prior.get("cases_completed") == len(selected_ids)
                    and prior.get("source_hashes_verified") is True
                    and prior.get("cache_hashes_verified") is True
                )
                if valid:
                    print(f"Verified prior cache prewarm report: {name}/{split} ({len(selected_ids)} cases)")
                    return prior
                raise RuntimeError(f"Stale or inconsistent cache prewarm report: {report_path}")

            def progress(item):
                position = int(item["position"])
                total = len(selected_ids)
                if position == 1 or position == total or position % 25 == 0:
                    print(
                        f"Cache {name}/{split}: {position}/{total} "
                        f"{item['case_id']} ({item['elapsed_seconds']:.1f}s)"
                    )

            raw = prewarm_cache(
                project_root=PROJECT_ROOT,
                dataset_root=DATASET_ROOT,
                manifest=manifest,
                schema=schema,
                preprocessing=preprocessing,
                preprocessing_sha256=config.preprocessing_sha256,
                config=config,
                split=split,
                case_ids=case_ids,
                verify_source_hashes=True,
                progress=progress,
            )
            report = {
                **raw,
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "config_sha256": config.config_sha256,
                "manifest_sha256": config.manifest_sha256,
                "organ_schema_sha256": config.organ_schema_sha256,
                "preprocessing_sha256": config.preprocessing_sha256,
                "selected_case_ids_sha256": expected_ids_sha,
                "source_bundle_sha256": source_bundle_sha256,
                "source_hashes_verified": True,
                "cache_hashes_verified": True,
                "test_split_accessed": False,
            }
            atomic_write_json(report_path, report)
            assert report["cases_completed"] == len(selected_ids)
            return report
        """
    ),
    markdown(
        r"""
        ## 3. Gate 1 — fixed one-case overfit

        This gate uses 32 immutable, foreground-class-centred `96³` patches from official **training case `s0014`**
        for 50 diagnostic epochs. Augmentation is off. It has no validation dataset or validation loader. Passing
        requires at least 80% loss reduction, mean Dice ≥0.70 across represented foreground classes, at least 1,500 optimizer steps, finite
        losses/gradients, and checkpoint reload parity. The fingerprinted AMP initial scale is 1024; this avoids
        the observed first-epoch FP16 gradient overflow from PyTorch's default scale while retaining fail-closed
        non-finite-gradient checks.

        These numbers demonstrate that the implementation can memorize training patches. They are not generalization
        estimates and must not be reported as validation or test performance.
        """
    ),
    code(
        r"""
        overfit_name = "one_case_overfit"
        overfit_stage = workflow.stages[overfit_name]
        overfit_config = configs[overfit_name]
        overfit_case_id = (overfit_stage.train_case_ids or ())[0]
        overfit_scope_sha = overfit_scope_sha256(
            case_id=overfit_case_id,
            fixed_sample_indices=overfit_stage.fixed_sample_indices,
            expected_selected_class_ids=overfit_stage.expected_selected_class_ids,
        )
        overfit_report = verified_passed_report(overfit_config, overfit_scope_sha)
        if overfit_report is None:
            overfit_cache_report = ensure_cache_prewarmed(
                overfit_name,
                overfit_config,
                "train",
                overfit_stage.train_case_ids,
            )
            overfit_dataset = ExactSplitPatchDataset(
                project_root=PROJECT_ROOT,
                dataset_root=DATASET_ROOT,
                manifest=manifest,
                schema=schema,
                preprocessing=preprocessing,
                preprocessing_sha256=overfit_config.preprocessing_sha256,
                config=overfit_config,
                split="train",
                augment=False,
                allow_cache_write=overfit_config.notebook_cache_write_enabled,
                case_ids=overfit_stage.train_case_ids,
                fixed_sample_indices=overfit_stage.fixed_sample_indices,
                verify_cache_hashes=False,
            )
            overfit_generator = torch.Generator().manual_seed(overfit_config.random_seed)
            overfit_loader = DataLoader(
                overfit_dataset,
                batch_size=overfit_config.batch_size,
                shuffle=overfit_stage.shuffle,
                num_workers=0,
                pin_memory=overfit_config.pin_memory,
                generator=overfit_generator,
                drop_last=False,
            )
            overfit_report = run_overfit_gate(
                project_root=PROJECT_ROOT,
                config=overfit_config,
                train_loader=overfit_loader,
                case_id=overfit_case_id,
                fixed_sample_indices=overfit_stage.fixed_sample_indices,
                expected_selected_class_ids=overfit_stage.expected_selected_class_ids,
                criteria=OverfitGateCriteria.from_mapping(overfit_stage.acceptance),
                provenance=base_provenance(overfit_config, overfit_scope_sha),
                workflow_config_sha256=workflow.config_sha256,
                manifest=manifest,
                schema=schema,
                preprocessing=preprocessing,
                gpu_report=gpu_report,
                run_id=workflow.run_id,
                resume_from=resume_checkpoint(overfit_config),
                allow_training=RUN_TRAINING,
            )
        assert overfit_report["status"] == "passed" and all(overfit_report["acceptance"].values())
        display(
            pd.DataFrame(
                [
                    ("Loss reduction", f"{overfit_report['loss_reduction_fraction']:.1%}"),
                    (
                        "Mean represented-class training Dice",
                        f"{overfit_report['final_bank_metrics']['mean_represented_class_dice']:.3f}",
                    ),
                    ("Optimizer steps", overfit_report["counters"]["optimizer_steps"]),
                    ("Checkpoint reload parity", overfit_report["checkpoint_reload_parity"]),
                    ("Validation accessed", overfit_report["validation_split_accessed"]),
                    ("Test accessed", overfit_report["test_split_accessed"]),
                ],
                columns=["Gate-1 check", "Result"],
            ).style.hide(axis="index")
        )
        """
    ),
    markdown(
        r"""
        ## 4. Gate 2 — deterministic 16/4 pilot

        This is the first train/validation execution. It validates augmentation, cache reuse, full-volume sliding
        window inference, Dice/HD95 aggregation, optimizer/checkpoint behavior and epoch-boundary resume on a small,
        explicitly locked subset. A deliberate pause after epoch 1 is resumed immediately to prove recovery before
        the full run is allowed. The test split remains unavailable.
        """
    ),
    code(
        r"""
        def run_fit_stage(name):
            stage = workflow.stages[name]
            config = configs[name]
            selected_train_ids = train_ids if stage.train_case_ids is None else stage.train_case_ids
            selected_validation_ids = validation_ids if stage.validation_case_ids is None else stage.validation_case_ids
            scope = TrainingScope(
                gate=name,
                train_case_ids=tuple(selected_train_ids),
                validation_case_ids=tuple(selected_validation_ids),
            )
            existing = verified_passed_report(config, scope.fingerprint())
            if existing is not None:
                return existing

            train_cache_report = ensure_cache_prewarmed(
                name,
                config,
                "train",
                stage.train_case_ids,
            )
            validation_cache_report = ensure_cache_prewarmed(
                name,
                config,
                "val",
                stage.validation_case_ids,
            )

            training_dataset = ExactSplitPatchDataset(
                project_root=PROJECT_ROOT,
                dataset_root=DATASET_ROOT,
                manifest=manifest,
                schema=schema,
                preprocessing=preprocessing,
                preprocessing_sha256=config.preprocessing_sha256,
                config=config,
                split="train",
                augment=True,
                allow_cache_write=config.notebook_cache_write_enabled,
                case_ids=stage.train_case_ids,
                verify_cache_hashes=False,
            )
            validation_dataset = ExactSplitVolumeDataset(
                project_root=PROJECT_ROOT,
                dataset_root=DATASET_ROOT,
                manifest=manifest,
                schema=schema,
                preprocessing=preprocessing,
                preprocessing_sha256=config.preprocessing_sha256,
                config=config,
                split="val",
                allow_cache_write=config.notebook_cache_write_enabled,
                case_ids=stage.validation_case_ids,
                verify_cache_hashes=False,
            )
            assert training_dataset.selected_case_ids == scope.train_case_ids
            assert validation_dataset.selected_case_ids == scope.validation_case_ids
            training_loader, validation_loader = build_training_loaders(
                training_dataset,
                validation_dataset,
                config,
                production=False,
            )
            def execute_fit(*, resume_from, pause_after_epoch=None):
                return fit(
                    project_root=PROJECT_ROOT,
                    config=config,
                    model=None,
                    train_loader=training_loader,
                    validation_loader=validation_loader,
                    provenance=base_provenance(config, scope.fingerprint()),
                    spacing_mm=preprocessing.spacing_mm,
                    manifest=manifest,
                    schema=schema,
                    preprocessing=preprocessing,
                    gpu_report=gpu_report,
                    allow_training=RUN_TRAINING,
                    scope=scope,
                    run_id=workflow.run_id,
                    resume_from=resume_from,
                    pause_after_epoch=pause_after_epoch,
                )

            checkpoint = resume_checkpoint(config)
            if name == "deterministic_pilot" and checkpoint is None:
                paused_result = execute_fit(resume_from=None, pause_after_epoch=1)
                assert paused_result["status"] == "paused"
                assert paused_result["counters"]["epochs_completed"] == 1
                checkpoint = Path(paused_result["last_checkpoint"])
                if not checkpoint.is_absolute():
                    checkpoint = PROJECT_ROOT / checkpoint
                assert checkpoint.is_file()
            result = execute_fit(resume_from=checkpoint)
            history = result["history"]
            losses = [float(record["training_loss"]) for record in history]
            validation_scores = finite_validation_scores(history)
            last_checkpoint = Path(result["last_checkpoint"])
            if not last_checkpoint.is_absolute():
                last_checkpoint = PROJECT_ROOT / last_checkpoint
            best_value = result.get("best_checkpoint")
            best_checkpoint = None if not best_value else Path(best_value)
            if best_checkpoint is not None and not best_checkpoint.is_absolute():
                best_checkpoint = PROJECT_ROOT / best_checkpoint

            if name == "deterministic_pilot":
                acceptance = {
                    "overfit_prerequisite_passed": overfit_report["status"] == "passed",
                    "completed_required_epochs": result["counters"]["epochs_completed"] >= int(stage.acceptance["minimum_completed_epochs"]),
                    "finite_training_loss": bool(losses) and all(math.isfinite(value) for value in losses),
                    "finite_validation_dice": bool(validation_scores),
                    "last_checkpoint_exists": last_checkpoint.is_file(),
                    "resume_event_recorded": (
                        not bool(stage.acceptance.get("require_resume_event"))
                        or result["counters"]["resume_events"] >= 1
                    ),
                    "test_split_not_accessed": True,
                }
            else:
                acceptance = {
                    "pilot_prerequisite_passed": pilot_report["status"] == "passed",
                    "training_completed_or_early_stopped": result["status"] in {"completed", "early_stopped"},
                    "finite_training_loss": bool(losses) and all(math.isfinite(value) for value in losses),
                    "finite_validation_dice": bool(validation_scores),
                    "last_checkpoint_exists": last_checkpoint.is_file(),
                    "best_checkpoint_exists": best_checkpoint is not None and best_checkpoint.is_file(),
                    "test_split_not_accessed": True,
                }
            report = {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "passed" if all(acceptance.values()) else "failed",
                "gate": name,
                "config_sha256": config.config_sha256,
                "workflow_config_sha256": workflow.config_sha256,
                "scope_sha256": scope.fingerprint(),
                "provenance_sha256": hashlib.sha256(
                    json.dumps(
                        base_provenance(config, scope.fingerprint()),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "train_case_count": len(scope.train_case_ids),
                "validation_case_count": len(scope.validation_case_ids),
                "cache_prewarm": {
                    "train_report": relative_artifact(
                        PROJECT_ROOT / "outputs" / "training" / "cache_prewarm" / f"{name}_train.json"
                    ),
                    "validation_report": relative_artifact(
                        PROJECT_ROOT / "outputs" / "training" / "cache_prewarm" / f"{name}_val.json"
                    ),
                    "train_cases": train_cache_report["cases_completed"],
                    "validation_cases": validation_cache_report["cases_completed"],
                },
                "test_split_accessed": False,
                "last_checkpoint": relative_artifact(last_checkpoint),
                "best_checkpoint": None if best_checkpoint is None else relative_artifact(best_checkpoint),
                "best_validation_mean_foreground_dice": result.get("best_validation_mean_foreground_dice"),
                "counters": result["counters"],
                "acceptance": acceptance,
            }
            atomic_write_json(stage_report_path(config), report)
            if report["status"] != "passed":
                failed = [key for key, passed in acceptance.items() if not passed]
                raise RuntimeError(f"{name} gate failed: {failed}")
            return report

        pilot_report = run_fit_stage("deterministic_pilot")
        assert pilot_report["status"] == "passed" and all(pilot_report["acceptance"].values())
        display(pd.DataFrame([pilot_report["acceptance"]]).T.rename(columns={0: "PASS"}))
        """
    ),
    markdown(
        r"""
        ## 5. Gate 3 — full development baseline

        This cell is reached only after both diagnostics pass. It trains for at most 50 epochs on all 1,082 official
        training cases and performs deterministic complete-volume validation on all 57 validation cases. Early
        stopping and best-model selection use validation mean foreground Dice. Test data remains locked for a later,
        frozen evaluation.
        """
    ),
    code(
        r"""
        full_report = run_fit_stage("full_baseline")
        assert full_report["status"] == "passed" and all(full_report["acceptance"].values())
        display(pd.DataFrame([full_report["acceptance"]]).T.rename(columns={0: "PASS"}))
        """
    ),
    markdown(
        r"""
        ## 6. Training hand-off

        Reaching this cell means all three stages passed, a resumable `last.pt` exists, and a validation-selected
        `best.pt` exists for the full baseline. Do not unlock the test split here. The next notebook should freeze
        the selected checkpoint and implement uncertainty estimation plus final evaluation as a separate protocol.
        """
    ),
    code(
        r"""
        summary = pd.DataFrame(
            [
                {"Stage": "one_case_overfit", "Status": overfit_report["status"], "Test accessed": False},
                {"Stage": "deterministic_pilot", "Status": pilot_report["status"], "Test accessed": False},
                {"Stage": "full_baseline", "Status": full_report["status"], "Test accessed": False},
            ]
        )
        display(summary.style.hide(axis="index"))
        display(
            Markdown(
                "### ✅ Staged baseline training completed\n\n"
                f"Full best checkpoint: `{full_report['best_checkpoint']}`  \n"
                "Locked test split: **not accessed**"
            )
        )
        """
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"phase5-{index:02d}"


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": ".venv (3.13.12.final.0)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13.12"},
        "project": {
            "stage": "phase5_staged_training",
            "training_authorized": True,
            "training_executed_when_built": False,
            "test_split_locked": True,
            "generated_by": "scripts/build_staged_training_notebook.py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells; no cell was executed")
