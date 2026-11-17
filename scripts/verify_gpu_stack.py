#!/usr/bin/env python3
"""Fail-closed CUDA/PyTorch/MONAI verification using a real Phase 3 patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai import __version__ as monai_version
from monai.networks.nets import UNet
from monai.transforms import Flipd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    canonicalize_to_ras,
    development_cases,
    get_development_case,
    load_exclusive_case,
    load_organ_schema,
    load_preprocessing_spec,
    load_validated_manifest,
    normalize_ct,
    resample_pair,
    sample_class_balanced_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "Totalsegmentator_dataset_v201",
        help="Extracted TotalSegmentator dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "environment" / "gpu_verification.json",
        help="Passed-verification JSON output",
    )
    return parser.parse_args()


def _command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "returncode": None, "output": str(exc)}
    output = (completed.stdout or completed.stderr).strip()
    return {"available": True, "returncode": completed.returncode, "output": output}


def _timed(checks: dict[str, dict[str, Any]], name: str, operation: Callable[[], dict[str, Any]]) -> None:
    started = time.perf_counter()
    details = operation()
    torch.cuda.synchronize()
    checks[name] = {"passed": True, "elapsed_seconds": time.perf_counter() - started, **details}


def _json_normalize(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _archive_existing_report(output_path: Path, started_at: datetime) -> str | None:
    if not output_path.exists():
        return None
    history_dir = output_path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    candidate = history_dir / f"{output_path.stem}_{timestamp}_previous{output_path.suffix}"
    if candidate.exists():
        candidate = history_dir / (
            f"{output_path.stem}_{timestamp}_{uuid4().hex[:8]}_previous{output_path.suffix}"
        )
    output_path.replace(candidate)
    return _safe_project_path(candidate)


def _cudnn_version_string(version: int) -> str:
    return f"{version // 10000}.{(version % 10000) // 100}.{version % 100}"


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeError(f"GPU environment contract mismatch for {name}: {actual!r} != {expected!r}")


def _pinned_version(requirements_text: str, package: str) -> str:
    match = re.search(rf"(?mi)^\s*{re.escape(package)}==([^\s;]+)\s*$", requirements_text)
    if match is None:
        raise RuntimeError(f"requirements-gpu.txt must exactly pin {package}")
    return match.group(1)


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    run_id = f"gpu-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_report_archived = _archive_existing_report(output_path, started_at)
    dataset_root = args.dataset_root.resolve()
    environment_config_path = PROJECT_ROOT / "configs" / "environment_gpu.yaml"
    gpu_requirements_path = PROJECT_ROOT / "requirements-gpu.txt"
    base_requirements_path = PROJECT_ROOT / "requirements.txt"
    environment_contract = yaml.safe_load(environment_config_path.read_text(encoding="utf-8"))
    gpu_requirements_text = gpu_requirements_path.read_text(encoding="utf-8")
    expected_framework = environment_contract["framework"]
    _require_equal("Python major.minor", ".".join(platform.python_version_tuple()[:2]), str(environment_contract["platform"]["python"]))
    _require_equal("torch YAML pin", torch.__version__, str(expected_framework["torch"]))
    _require_equal("torch requirements pin", torch.__version__, _pinned_version(gpu_requirements_text, "torch"))
    _require_equal("PyTorch CUDA runtime", torch.version.cuda, str(expected_framework["pytorch_cuda_runtime"]))
    _require_equal("MONAI YAML pin", monai_version, str(expected_framework["monai"]))
    _require_equal("MONAI requirements pin", monai_version, _pinned_version(gpu_requirements_text, "monai"))
    _require_equal("NiBabel YAML pin", nib.__version__, str(expected_framework["nibabel"]))
    _require_equal("NiBabel requirements pin", nib.__version__, _pinned_version(gpu_requirements_text, "nibabel"))
    if torch.version.cuda is None:
        raise RuntimeError("Installed PyTorch is CPU-only; CUDA build is required")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; refusing CPU fallback")
    if torch.cuda.device_count() <= args.device:
        raise RuntimeError(f"CUDA device {args.device} is unavailable")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # CUDA CrossEntropy currently has no kernel marked strictly deterministic
    # for this 5D segmentation shape. Keep deterministic selection enabled but
    # warn instead of blocking the GPU smoke step; record this in the report.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.cuda.init()
    torch.cuda.synchronize()

    if not torch.backends.cudnn.enabled or not torch.backends.cudnn.is_available():
        raise RuntimeError("cuDNN is not enabled and available")
    cudnn_version = torch.backends.cudnn.version()
    if cudnn_version is None:
        raise RuntimeError("cuDNN version is unavailable")
    cudnn_version_text = _cudnn_version_string(cudnn_version)
    _require_equal("cuDNN", cudnn_version_text, str(expected_framework["cudnn"]))

    properties = torch.cuda.get_device_properties(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
    arch_list = torch.cuda.get_arch_list()
    expected_arch = f"sm_{capability[0]}{capability[1]}"
    _require_equal("GPU name", properties.name, str(environment_contract["hardware_target"]["gpu"]))
    _require_equal(
        "GPU compute capability",
        list(capability),
        [int(value) for value in environment_contract["hardware_target"]["compute_capability"]],
    )
    checks: dict[str, dict[str, Any]] = {}

    def allocation_check() -> dict[str, Any]:
        tensor = torch.full((16, 1024, 1024), 0.25, dtype=torch.float32, device=device)
        checksum = float(tensor.sum().item())
        expected = 16 * 1024 * 1024 * 0.25
        if not math.isclose(checksum, expected, rel_tol=0, abs_tol=0):
            raise AssertionError(f"CUDA allocation/reduction checksum mismatch: {checksum} != {expected}")
        allocated_mib = tensor.numel() * tensor.element_size() / 2**20
        del tensor
        return {"allocated_tensor_mib": allocated_mib, "checksum": checksum}

    _timed(checks, "cuda_allocation_and_reduction", allocation_check)

    def matmul_check() -> dict[str, Any]:
        generator = torch.Generator().manual_seed(4201)
        left_cpu = torch.randn((1024, 1024), generator=generator, dtype=torch.float32)
        right_cpu = torch.randn((1024, 1024), generator=generator, dtype=torch.float32)
        expected = left_cpu @ right_cpu
        actual = left_cpu.to(device) @ right_cpu.to(device)
        if not torch.isfinite(actual).all():
            raise AssertionError("CUDA matrix multiplication produced non-finite values")
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)
        checksum = float(actual.sum().item())
        return {"shape": list(actual.shape), "dtype": str(actual.dtype), "checksum": checksum}

    _timed(checks, "cuda_matmul_cpu_parity", matmul_check)

    def conv3d_check() -> dict[str, Any]:
        torch.manual_seed(4202)
        cpu_model = torch.nn.Conv3d(2, 4, kernel_size=3, padding=1, bias=True)
        cpu_input = torch.randn((1, 2, 32, 32, 32), dtype=torch.float32, requires_grad=True)
        cpu_output = cpu_model(cpu_input)
        gpu_model = torch.nn.Conv3d(2, 4, kernel_size=3, padding=1, bias=True).to(device)
        gpu_model.load_state_dict(cpu_model.state_dict())
        gpu_input = cpu_input.detach().to(device).requires_grad_(True)
        gpu_output = gpu_model(gpu_input)
        torch.testing.assert_close(gpu_output.cpu(), cpu_output.detach(), rtol=1e-4, atol=1e-4)
        loss = gpu_output.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in gpu_model.parameters()]
        if not gradients or any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError("CUDA Conv3d backward produced missing/non-finite gradients")
        return {"output_shape": list(gpu_output.shape), "loss": float(loss.item())}

    _timed(checks, "cuda_conv3d_with_cudnn_enabled", conv3d_check)

    def amp_check() -> dict[str, Any]:
        torch.manual_seed(4203)
        model = torch.nn.Conv3d(2, 4, kernel_size=3, padding=1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        inputs = torch.randn((1, 2, 32, 32, 32), device=device)
        before = model.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)
            loss = output.square().mean()
        if output.dtype != torch.float16 or not torch.isfinite(loss):
            raise AssertionError("CUDA autocast did not produce finite float16 Conv3d output")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if model.weight.grad is None or not torch.isfinite(model.weight.grad).all():
            raise AssertionError("AMP backward produced a missing/non-finite gradient")
        scaler.step(optimizer)
        scaler.update()
        if torch.equal(before, model.weight.detach()):
            raise AssertionError("AMP optimizer step did not update model weights")
        return {
            "output_dtype": str(output.dtype),
            "loss_dtype": str(loss.dtype),
            "loss": float(loss.item()),
            "grad_scaler_enabled": scaler.is_enabled(),
        }

    _timed(checks, "cuda_amp_grad_scaler_optimizer", amp_check)

    schema = load_organ_schema(PROJECT_ROOT / "configs" / "organs_v1.yaml")
    spec = load_preprocessing_spec(PROJECT_ROOT / "configs" / "dataset.yaml")
    manifest = load_validated_manifest(PROJECT_ROOT / "data" / "splits" / "dataset_manifest.json", schema)
    phase3_summary = json.loads(
        (PROJECT_ROOT / "outputs" / "phase3" / "qa_summary.json").read_text(encoding="utf-8")
    )
    try:
        development_cases(manifest, "test")
    except PermissionError:
        test_lock_enforced = True
    else:
        raise AssertionError("Locked test split was not rejected before path resolution")

    case_id = str(phase3_summary["primary_case"])
    case_record = get_development_case(manifest, case_id)
    if case_record["split"] != "train":
        raise AssertionError(f"Real GPU smoke case must be train, found {case_record['split']}")
    real_case = load_exclusive_case(dataset_root, case_record, schema)
    ras_hu, ras_label, ras_affine, orientation_trace = canonicalize_to_ras(
        real_case.ct_hu, real_case.label, real_case.affine
    )
    spatial_hu, spatial_label, _spatial_affine, resample_trace = resample_pair(
        ras_hu, ras_label, ras_affine, spec.spacing_mm, image_fill_hu=spec.hu_lower
    )
    normalized = normalize_ct(spatial_hu, spec)
    image_patch, label_patch, patch_trace = sample_class_balanced_patch(
        normalized,
        spatial_label,
        spatial_hu,
        spec,
        case_id=case_id,
        sample_index=int(phase3_summary["foreground_patch_trace"]["sample_index"]),
    )
    preprocessing_sha256 = spec.fingerprint(
        manifest_sha256=manifest["manifest_sha256"], schema_sha256=schema.fingerprint()
    )
    if manifest["manifest_sha256"] != phase3_summary["manifest_sha256"]:
        raise AssertionError("Manifest fingerprint changed since Phase 3")
    if schema.fingerprint() != phase3_summary["organ_schema_sha256"]:
        raise AssertionError("Organ schema fingerprint changed since Phase 3")
    if preprocessing_sha256 != phase3_summary["preprocessing_sha256"]:
        raise AssertionError("Preprocessing fingerprint changed since Phase 3")
    if real_case.exclusive_foreground_voxels != phase3_summary["primary_exclusive_counts"]:
        raise AssertionError("Real-case exclusive counts changed since Phase 3")
    if _json_normalize(asdict(patch_trace)) != phase3_summary["foreground_patch_trace"]:
        raise AssertionError("Real patch trace changed since Phase 3")
    if not orientation_trace["was_noop"] or not resample_trace["was_noop"]:
        raise AssertionError("Audited Phase 3 spatial preprocessing should remain an exact no-op")
    if image_patch.shape != (1, 96, 96, 96) or label_patch.shape != (96, 96, 96):
        raise AssertionError("Real Phase 3 patch shape is invalid")
    if image_patch.dtype != np.float32 or label_patch.dtype != np.uint8:
        raise AssertionError("Real Phase 3 patch dtypes are invalid")
    if real_case.source_masks_checked != 13:
        raise AssertionError(f"Expected 13 selected source masks, found {real_case.source_masks_checked}")
    if not np.isfinite(image_patch).all() or float(image_patch.min()) < 0.0 or float(image_patch.max()) > 1.0:
        raise AssertionError("Real image patch contains non-finite or out-of-range normalized values")
    label_ids = {int(value) for value in np.unique(label_patch)}
    if not label_ids.issubset(set(range(11))):
        raise AssertionError(f"Real label patch contains invalid class IDs: {sorted(label_ids)}")

    image_tensor = torch.from_numpy(np.ascontiguousarray(image_patch))[None].to(device)
    label_tensor = torch.from_numpy(np.ascontiguousarray(label_patch.astype(np.int64)))[None].to(device)
    torch.cuda.synchronize()
    if image_tensor.shape != (1, 1, 96, 96, 96) or label_tensor.shape != (1, 96, 96, 96):
        raise AssertionError("GPU patch tensor shapes are invalid")
    if not torch.equal(image_tensor.cpu(), torch.from_numpy(image_patch)[None]):
        raise AssertionError("Real image patch changed during GPU round trip")
    if not torch.equal(label_tensor.cpu(), torch.from_numpy(label_patch.astype(np.int64))[None]):
        raise AssertionError("Real label patch changed during GPU round trip")
    checks["real_phase3_patch_to_cuda"] = {
        "passed": True,
        "case_id": case_id,
        "split": case_record["split"],
        "image_shape": list(image_tensor.shape),
        "label_shape": list(label_tensor.shape),
        "image_dtype": str(image_tensor.dtype),
        "label_dtype": str(label_tensor.dtype),
        "label_ids": [int(value) for value in torch.unique(label_tensor).cpu().tolist()],
        "source_masks_checked": real_case.source_masks_checked,
    }

    def monai_transform_check() -> dict[str, Any]:
        paired = {"image": image_tensor[0], "label": label_tensor[0].unsqueeze(0)}
        transformed = Flipd(keys=("image", "label"), spatial_axis=0)(paired)
        expected_image = torch.flip(paired["image"], dims=(1,))
        expected_label = torch.flip(paired["label"], dims=(1,))
        if transformed["image"].device.type != "cuda" or transformed["label"].device.type != "cuda":
            raise AssertionError("MONAI paired transform moved data off CUDA")
        if not torch.equal(transformed["image"], expected_image):
            raise AssertionError("MONAI image flip differs from expected paired transform")
        if not torch.equal(transformed["label"], expected_label):
            raise AssertionError("MONAI label flip differs from expected paired transform")
        return {
            "transform": "Flipd",
            "image_shape": list(transformed["image"].shape),
            "label_shape": list(transformed["label"].shape),
            "device": str(transformed["image"].device),
        }

    _timed(checks, "monai_paired_cuda_transform", monai_transform_check)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    def monai_model_check() -> dict[str, Any]:
        torch.manual_seed(4204)
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=11,
            channels=(8, 16, 32),
            strides=(2, 2),
            num_res_units=1,
            norm="INSTANCE",
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        first_parameter = next(model.parameters())
        before = first_parameter.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(image_tensor)
            loss = F.cross_entropy(logits, label_tensor)
        if logits.shape != (1, 11, 96, 96, 96):
            raise AssertionError(f"Unexpected MONAI logits shape {tuple(logits.shape)}")
        if logits.dtype != torch.float16 or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise AssertionError("MONAI AMP forward produced invalid logits/loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError("MONAI AMP backward produced missing/non-finite gradients")
        scaler.step(optimizer)
        scaler.update()
        if torch.equal(before, first_parameter.detach()):
            raise AssertionError("MONAI optimizer step did not update parameters")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        return {
            "network": "MONAI UNet smoke model (not research baseline)",
            "parameter_count": parameter_count,
            "input_shape": list(image_tensor.shape),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "loss": float(loss.item()),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }

    _timed(checks, "monai_real_patch_amp_training_step", monai_model_check)

    nvidia_smi = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if nvidia_smi["returncode"] != 0:
        raise RuntimeError(f"nvidia-smi query failed: {nvidia_smi}")
    nvidia_smi_banner = _command_output(["nvidia-smi"])
    maximum_cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_banner["output"])
    driver_reported_max_cuda = maximum_cuda_match.group(1) if maximum_cuda_match else None
    _require_equal(
        "driver-reported maximum CUDA version",
        driver_reported_max_cuda,
        str(environment_contract["hardware_target"]["driver_reported_max_cuda_version"]),
    )
    nvcc_path = shutil.which("nvcc")
    nvcc = _command_output([nvcc_path, "--version"]) if nvcc_path else {
        "available": False,
        "returncode": None,
        "output": "not installed (not required for prebuilt PyTorch wheels)",
    }
    gencode_getter = getattr(torch.cuda, "get_gencode_flags", None)
    artifact_hashes = {
        "verification_script": {
            "path": _safe_project_path(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
        },
        "gpu_requirements": {
            "path": _safe_project_path(gpu_requirements_path),
            "sha256": _sha256(gpu_requirements_path),
        },
        "base_requirements": {
            "path": _safe_project_path(base_requirements_path),
            "sha256": _sha256(base_requirements_path),
        },
        "environment_config": {
            "path": _safe_project_path(environment_config_path),
            "sha256": _sha256(environment_config_path),
        },
    }
    executable_path = Path(sys.executable).resolve()
    report: dict[str, Any] = {
        "status": "passed",
        "run_id": run_id,
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration_seconds": time.perf_counter() - started_perf,
        "previous_report_archived": previous_report_archived,
        "invocation": {
            "python_executable": _safe_project_path(executable_path),
            "script": _safe_project_path(Path(__file__)),
            "device": args.device,
            "dataset_root": _safe_project_path(dataset_root),
            "output": _safe_project_path(output_path),
        },
        "test_split_accessed": False,
        "test_lock_enforced": test_lock_enforced,
        "environment": {
            "python": platform.python_version(),
            "python_executable": _safe_project_path(executable_path),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "pytorch_cuda_runtime": torch.version.cuda,
            "cudnn_version": cudnn_version,
            "cudnn_version_string": cudnn_version_text,
            "monai": monai_version,
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "torch_config": torch.__config__.show(),
            "deterministic_algorithms": "enabled_warn_only",
            "determinism_note": "CUDA CrossEntropy for 5D segmentation is not marked strictly deterministic in this build.",
        },
        "driver": {
            "nvidia_smi": nvidia_smi,
            "driver_reported_max_cuda_version": driver_reported_max_cuda,
            "nvcc": nvcc,
        },
        "device": {
            "index": args.device,
            "name": properties.name,
            "compute_capability": list(capability),
            "expected_compiled_arch": expected_arch,
            "exact_arch_in_build": expected_arch in arch_list,
            "compiled_arch_list": arch_list,
            "gencode_flags": gencode_getter() if gencode_getter else None,
            "total_memory_mib": properties.total_memory / 2**20,
            "free_memory_at_start_mib": free_bytes / 2**20,
            "runtime_total_memory_mib": total_bytes / 2**20,
            "multiprocessor_count": properties.multi_processor_count,
        },
        "provenance": {
            "environment_id": environment_contract["environment_id"],
            "artifact_hashes": artifact_hashes,
            "manifest_sha256": manifest["manifest_sha256"],
            "organ_schema_sha256": schema.fingerprint(),
            "preprocessing_sha256": preprocessing_sha256,
            "real_case_id": case_id,
            "real_case_split": case_record["split"],
            "real_patch_trace": _json_normalize(asdict(patch_trace)),
        },
        "checks": checks,
        "all_checks_passed": all(item.get("passed") is True for item in checks.values()),
    }
    _require_equal(
        "GPU verification check count",
        len(checks),
        int(environment_contract["verification"]["expected_check_count"]),
    )
    if not report["all_checks_passed"]:
        raise AssertionError("At least one GPU verification check did not pass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print("GPU STACK VERIFIED")
    print(f"Torch: {torch.__version__} | runtime CUDA: {torch.version.cuda} | cuDNN: {cudnn_version_text}")
    print(f"MONAI: {monai_version}")
    print(f"Device: {properties.name} | capability: {capability} | VRAM: {properties.total_memory / 2**20:.0f} MiB")
    print(f"Checks passed: {len(checks)}/{len(checks)}")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
