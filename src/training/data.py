"""Exact-split, manifest-driven datasets for Phase 4 training and validation."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    ScaleIntensityRanged,
)
from torch.utils.data import DataLoader, Dataset

from src.dataset import (
    OrganSchema,
    PatchTrace,
    PreprocessingSpec,
    canonicalize_to_ras,
    development_cases,
    extract_centered_patch,
    load_exclusive_case,
    normalize_ct,
    resample_pair,
    sample_class_balanced_patch,
)
from src.dataset.loader import resolve_dataset_path

from .config import BaselineConfig


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    split: str
    ct_hu: np.ndarray
    label: np.ndarray
    affine: np.ndarray
    cache_hit: bool
    source_fingerprint: str


CACHE_SCHEMA_VERSION = 2
_CACHE_ARRAY_NAMES = ("ct_hu", "label", "affine")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def case_source_fingerprint(case: dict[str, Any]) -> str:
    """Fingerprint only immutable source hashes needed to prepare one case."""
    sources: list[dict[str, str]] = []
    for target_name, target in sorted(case["targets"].items()):
        for source in sorted(target["sources"], key=lambda item: item["name"]):
            sources.append(
                {
                    "target": target_name,
                    "name": str(source["name"]),
                    "path": str(source["path"]),
                    "file_sha256": str(source["file_sha256"]),
                }
            )
    payload = {
        "case_id": case["case_id"],
        "split": case["split"],
        "ct": {
            "path": case["ct"]["path"],
            "file_sha256": case["ct"]["file_sha256"],
        },
        "sources": sources,
    }
    return _canonical_sha256(payload)


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _source_records(case: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return every immutable source record in stable order."""
    ct = case.get("ct")
    if not isinstance(ct, dict):
        raise ValueError(f"{case.get('case_id', '<unknown>')}: missing CT source record")
    records: list[tuple[str, dict[str, Any]]] = [("ct", ct)]
    targets = case.get("targets")
    if not isinstance(targets, dict):
        raise ValueError(f"{case.get('case_id', '<unknown>')}: targets must be a mapping")
    for target_name, target in sorted(targets.items()):
        if not isinstance(target, dict) or not isinstance(target.get("sources"), list):
            raise ValueError(f"{case.get('case_id', '<unknown>')}/{target_name}: invalid source records")
        for source in sorted(
            target["sources"],
            key=lambda item: str(item.get("name", "")) if isinstance(item, dict) else "",
        ):
            if not isinstance(source, dict):
                raise ValueError(f"{case.get('case_id', '<unknown>')}/{target_name}: invalid source record")
            records.append((f"{target_name}/{source.get('name', '<unnamed>')}", source))
    return records


def verify_case_source_hashes(dataset_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Hash the actual compressed source files and compare them with the manifest.

    This deliberately expensive check is optional during lazy loading and intended
    for a cache-prewarm pass.  The split is rejected before resolving any path.
    """
    if case.get("split") not in {"train", "val"}:
        raise PermissionError("Source verification accepts only train/validation records")
    seen_paths: dict[Path, str] = {}
    checked: list[dict[str, Any]] = []
    for logical_name, record in _source_records(case):
        expected = record.get("file_sha256")
        if not isinstance(expected, str) or _SHA256_PATTERN.fullmatch(expected) is None:
            raise ValueError(f"{case.get('case_id')}/{logical_name}: invalid recorded SHA-256")
        path = resolve_dataset_path(dataset_root, record.get("path", ""))
        actual = seen_paths.get(path)
        if actual is None:
            actual = _sha256_file(path)
            seen_paths[path] = actual
        if actual != expected:
            raise ValueError(
                f"{case.get('case_id')}/{logical_name}: source SHA-256 mismatch "
                f"(expected {expected}, actual {actual})"
            )
        checked.append(
            {
                "logical_name": logical_name,
                "relative_path": str(record["path"]),
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
        )
    return {
        "case_id": str(case.get("case_id")),
        "split": str(case.get("split")),
        "files_checked": len(checked),
        "unique_files_checked": len(seen_paths),
        "bytes_hashed": sum(path.stat().st_size for path in seen_paths),
        "records": checked,
    }


def _cache_case_dir(project_root: Path, config: BaselineConfig, preprocessing_sha256: str, case_id: str) -> Path:
    if _SHA256_PATTERN.fullmatch(preprocessing_sha256) is None:
        raise ValueError("Preprocessing fingerprint must be a lowercase SHA-256")
    if _CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise ValueError(f"Unsafe cache case ID: {case_id!r}")
    root = (project_root / Path(config.cache_root)).resolve()
    try:
        root.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("Configured cache root escapes the project directory") from exc
    namespace = (root / preprocessing_sha256[:16]).resolve()
    case_dir = (namespace / case_id).resolve()
    try:
        case_dir.relative_to(namespace)
    except ValueError as exc:
        raise ValueError("Cache case path escapes its fingerprint namespace") from exc
    return case_dir


def _validate_prepared_arrays(
    ct_hu: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
    *,
    out_channels: int,
) -> None:
    if ct_hu.dtype != np.float32 or label.dtype != np.uint8 or affine.dtype != np.float64:
        raise ValueError("Prepared cache arrays have unexpected dtypes")
    if ct_hu.ndim != 3 or ct_hu.shape != label.shape or any(size <= 0 for size in ct_hu.shape):
        raise ValueError("Prepared CT and label geometry is inconsistent")
    if affine.shape != (4, 4) or not bool(np.isfinite(affine).all()):
        raise ValueError("Prepared affine must be a finite 4x4 matrix")
    if not bool(np.isfinite(ct_hu).all()):
        raise ValueError("Prepared CT contains non-finite values")
    label_ids = set(int(value) for value in np.unique(label))
    if not label_ids <= set(range(out_channels)):
        raise ValueError(f"Prepared label contains invalid class IDs: {sorted(label_ids)}")


def _load_cache(
    case_dir: Path,
    expected: dict[str, Any],
    *,
    verify_array_hashes: bool,
) -> PreparedCase | None:
    meta_path = case_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    if set(metadata) != set(expected) | {"generation", "arrays"}:
        return None
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    generation = metadata.get("generation")
    arrays = metadata.get("arrays")
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        return None
    if not isinstance(arrays, dict) or set(arrays) != set(_CACHE_ARRAY_NAMES):
        return None
    paths: dict[str, Path] = {}
    expected_dtypes = {"ct_hu": "float32", "label": "uint8", "affine": "float64"}
    for name in _CACHE_ARRAY_NAMES:
        descriptor = arrays.get(name)
        expected_relative = f"generations/{generation}/{name}.npy"
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "sha256",
            "bytes",
            "dtype",
            "shape",
        }:
            return None
        if descriptor.get("path") != expected_relative or descriptor.get("dtype") != expected_dtypes[name]:
            return None
        sha256 = descriptor.get("sha256")
        byte_count = descriptor.get("bytes")
        shape = descriptor.get("shape")
        if (
            not isinstance(sha256, str)
            or _SHA256_PATTERN.fullmatch(sha256) is None
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not isinstance(shape, list)
            or not shape
            or any(not isinstance(value, int) or value <= 0 for value in shape)
        ):
            return None
        path = case_dir / Path(expected_relative)
        if not path.is_file() or path.stat().st_size != byte_count:
            return None
        if verify_array_hashes and _sha256_file(path) != sha256:
            return None
        paths[name] = path
    try:
        ct_hu = np.load(paths["ct_hu"], mmap_mode="r", allow_pickle=False)
        label = np.load(paths["label"], mmap_mode="r", allow_pickle=False)
        affine = np.load(paths["affine"], allow_pickle=False)
    except (OSError, ValueError):
        return None
    for name, array in (("ct_hu", ct_hu), ("label", label), ("affine", affine)):
        if list(array.shape) != arrays[name]["shape"] or str(array.dtype) != arrays[name]["dtype"]:
            return None
    try:
        _validate_prepared_arrays(
            ct_hu,
            label,
            affine,
            out_channels=int(expected["num_output_channels"]),
        )
    except ValueError:
        return None
    return PreparedCase(
        case_id=expected["case_id"],
        split=expected["split"],
        ct_hu=ct_hu,
        label=label,
        affine=affine,
        cache_hit=True,
        source_fingerprint=expected["source_fingerprint"],
    )


def _save_array(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)


def _write_cache(
    case_dir: Path,
    metadata: dict[str, Any],
    ct_hu: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
) -> None:
    """Publish a complete immutable generation, then atomically commit metadata."""
    _validate_prepared_arrays(
        ct_hu,
        label,
        affine,
        out_channels=int(metadata["num_output_channels"]),
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex
    generation_root = case_dir / "generations"
    generation_root.mkdir(parents=True, exist_ok=True)
    staging = case_dir / f".generation-{generation}.tmp"
    staging.mkdir(parents=False, exist_ok=False)
    values = {
        "ct_hu": ct_hu.astype(np.float32, copy=False),
        "label": label.astype(np.uint8, copy=False),
        "affine": affine.astype(np.float64, copy=False),
    }
    descriptors: dict[str, dict[str, Any]] = {}
    for name, array in values.items():
        path = staging / f"{name}.npy"
        _save_array(path, array)
        descriptors[name] = {
            "path": f"generations/{generation}/{name}.npy",
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    staging.replace(generation_root / generation)
    committed_metadata = dict(metadata)
    committed_metadata["generation"] = generation
    committed_metadata["arrays"] = descriptors
    metadata_path = case_dir / "metadata.json"
    temporary = metadata_path.with_name(f".{metadata_path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(committed_metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)


def prepare_case(
    *,
    project_root: Path,
    dataset_root: Path,
    case: dict[str, Any],
    schema: OrganSchema,
    preprocessing: PreprocessingSpec,
    preprocessing_sha256: str,
    config: BaselineConfig,
    allow_cache_write: bool,
    verify_source_hashes: bool = False,
    verify_cache_hashes: bool = False,
) -> PreparedCase:
    """Load/canonicalize/resample one dev case with an optional deterministic disk cache."""
    if case.get("split") not in {config.train_split, config.validation_split}:
        raise PermissionError("Phase 4 preparation accepts only the declared train/validation splits")
    if verify_source_hashes:
        verify_case_source_hashes(dataset_root, case)
    source_fingerprint = case_source_fingerprint(case)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "case_id": str(case["case_id"]),
        "split": str(case["split"]),
        "source_fingerprint": source_fingerprint,
        "manifest_sha256": config.manifest_sha256,
        "organ_schema_sha256": config.organ_schema_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "orientation": preprocessing.orientation,
        "spacing_mm": list(preprocessing.spacing_mm),
        "num_output_channels": config.out_channels,
        "contents": ["ct_hu_float32", "exclusive_label_uint8", "affine_float64"],
    }
    case_dir = _cache_case_dir(project_root, config, preprocessing_sha256, str(case["case_id"]))
    cached = _load_cache(case_dir, metadata, verify_array_hashes=verify_cache_hashes)
    if cached is not None:
        return cached
    loaded = load_exclusive_case(dataset_root, case, schema)
    ras_hu, ras_label, ras_affine, _ = canonicalize_to_ras(loaded.ct_hu, loaded.label, loaded.affine)
    spatial_hu, spatial_label, spatial_affine, _ = resample_pair(
        ras_hu,
        ras_label,
        ras_affine,
        preprocessing.spacing_mm,
        image_fill_hu=preprocessing.hu_lower,
    )
    spatial_hu = np.ascontiguousarray(spatial_hu, dtype=np.float32)
    spatial_label = np.ascontiguousarray(spatial_label, dtype=np.uint8)
    spatial_affine = np.ascontiguousarray(spatial_affine, dtype=np.float64)
    _validate_prepared_arrays(
        spatial_hu,
        spatial_label,
        spatial_affine,
        out_channels=config.out_channels,
    )
    if allow_cache_write:
        _write_cache(case_dir, metadata, spatial_hu, spatial_label, spatial_affine)
        if verify_cache_hashes and _load_cache(case_dir, metadata, verify_array_hashes=True) is None:
            raise OSError(f"Cache commit verification failed for {case['case_id']}")
    return PreparedCase(
        case_id=str(case["case_id"]),
        split=str(case["split"]),
        ct_hu=spatial_hu,
        label=spatial_label,
        affine=spatial_affine,
        cache_hit=False,
        source_fingerprint=source_fingerprint,
    )


def build_train_transform(config: BaselineConfig) -> Compose:
    """Build paired geometry + image-only intensity transforms; no unsafe flips."""
    transforms: list[Any] = [
        EnsureTyped(
            keys=("image", "label"),
            dtype=(torch.float32, torch.int64),
            track_meta=False,
        ),
        RandAffined(
            keys=("image", "label"),
            spatial_size=config.roi_size,
            prob=config.affine_probability,
            rotate_range=config.affine_rotate_range,
            translate_range=config.affine_translate_range,
            scale_range=config.affine_scale_range,
            mode=(config.affine_image_mode, config.affine_label_mode),
            padding_mode=config.affine_padding_mode,
        ),
        RandGaussianNoised(
            keys="image",
            prob=config.noise_probability,
            mean=config.noise_mean,
            std=config.noise_std,
            sample_std=False,
        ),
        RandAdjustContrastd(
            keys="image",
            prob=config.contrast_probability,
            gamma=config.contrast_gamma,
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=config.output_clamp[0],
            a_max=config.output_clamp[1],
            b_min=config.output_clamp[0],
            b_max=config.output_clamp[1],
            clip=True,
        ),
        EnsureTyped(
            keys=("image", "label"),
            dtype=(torch.float32, torch.int64),
            track_meta=False,
        ),
    ]
    return Compose(transforms)


def build_validation_transform(_config: BaselineConfig) -> Compose:
    """Validation is type conversion only: deterministic and full-volume."""
    return Compose(
        [
            EnsureTyped(
                keys=("image", "label"),
                dtype=(torch.float32, torch.int64),
                track_meta=False,
            )
        ]
    )


def _augmentation_seed(base_seed: int, case_id: str, sample_index: int) -> int:
    digest = hashlib.sha256(f"augmentation:{base_seed}:{case_id}:{sample_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def patch_trace_record(trace: PatchTrace) -> dict[str, Any]:
    """Convert PatchTrace to a JSON-compatible, default-collatable record."""
    record = asdict(trace)
    for field in (
        "center",
        "requested_start",
        "source_start",
        "source_end",
        "padding_before",
        "padding_after",
    ):
        record[field] = list(record[field])
    # Phase-3 derives an unsigned 64-bit seed.  Decimal Python ints above
    # int64 overflow PyTorch's default collator, so preserve all bits as hex.
    record["derived_seed"] = f"{trace.derived_seed:016x}"
    return record


def sample_class_balanced_hu_patch(
    ct_hu: np.ndarray,
    label: np.ndarray,
    preprocessing: PreprocessingSpec,
    *,
    case_id: str,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, PatchTrace]:
    """Sample with the Phase-3 RNG contract, normalizing only the selected patch.

    Center selection still delegates to the frozen Phase-3 sampler.  A broadcast
    placeholder avoids allocating a normalized full volume; the extracted HU patch
    is then normalized elementwise, which is exactly equivalent to normalize-then-cut.
    """
    placeholder = np.broadcast_to(np.float32(preprocessing.normalized_lower), ct_hu.shape)
    _, reference_label, trace = sample_class_balanced_patch(
        placeholder,
        label,
        ct_hu,
        preprocessing,
        case_id=case_id,
        sample_index=sample_index,
    )
    hu_patch, label_patch, details = extract_centered_patch(
        ct_hu,
        label,
        trace.center,
        preprocessing.roi_size,
        image_pad_value=preprocessing.hu_lower,
    )
    traced_details = {
        "requested_start": trace.requested_start,
        "source_start": trace.source_start,
        "source_end": trace.source_end,
        "padding_before": trace.padding_before,
        "padding_after": trace.padding_after,
    }
    if details != traced_details or not np.array_equal(label_patch, reference_label):
        raise RuntimeError("Patch-only normalization diverged from the frozen Phase-3 sampler")
    normalized = normalize_ct(hu_patch[0], preprocessing)[np.newaxis, ...]
    return np.ascontiguousarray(normalized), label_patch, trace


def _select_split_records(
    records: Sequence[dict[str, Any]],
    case_ids: Sequence[str] | None,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], bool]:
    full_ids = tuple(str(record["case_id"]) for record in records)
    if case_ids is None:
        return tuple(records), full_ids, True
    if isinstance(case_ids, (str, bytes)):
        raise TypeError("case_ids must be a sequence of case-ID strings, not one string")
    requested = tuple(case_ids)
    if not requested or any(not isinstance(case_id, str) or not case_id for case_id in requested):
        raise ValueError("case_ids must contain one or more non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("case_ids contains duplicates")
    by_id = {str(record["case_id"]): record for record in records}
    unknown = [case_id for case_id in requested if case_id not in by_id]
    if unknown:
        raise ValueError(f"case_ids are not members of the declared split: {unknown}")
    selected = tuple(by_id[case_id] for case_id in requested)
    return selected, requested, requested == full_ids


def seed_data_loader_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic per-worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_training_loaders(
    train_dataset: Dataset[dict[str, Any]],
    validation_dataset: Dataset[dict[str, Any]],
    config: BaselineConfig,
    *,
    production: bool,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Build deterministic train/random and validation/sequential loaders."""
    workers = config.production_workers if production else config.notebook_workers
    if workers < 0:
        raise ValueError("DataLoader worker count cannot be negative")
    common: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": config.pin_memory,
        "worker_init_fn": seed_data_loader_worker,
        "persistent_workers": bool(workers and config.persistent_workers),
    }
    if workers:
        common["prefetch_factor"] = config.prefetch_factor
    train_generator = torch.Generator().manual_seed(config.random_seed)
    validation_generator = torch.Generator().manual_seed(config.random_seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        generator=validation_generator,
        **common,
    )
    train_loader.reproducibility_contract = {  # type: ignore[attr-defined]
        "schema_version": 1,
        "seed": config.random_seed,
        "role": "train",
        "shuffle": True,
        "worker_seeded": True,
    }
    validation_loader.reproducibility_contract = {  # type: ignore[attr-defined]
        "schema_version": 1,
        "seed": config.random_seed + 1,
        "role": "validation",
        "shuffle": False,
        "worker_seeded": True,
    }
    return train_loader, validation_loader


def _committed_cache_bytes(case_dir: Path) -> int:
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    return sum(int(metadata["arrays"][name]["bytes"]) for name in _CACHE_ARRAY_NAMES)


def prewarm_cache(
    *,
    project_root: Path,
    dataset_root: Path,
    manifest: dict[str, Any],
    schema: OrganSchema,
    preprocessing: PreprocessingSpec,
    preprocessing_sha256: str,
    config: BaselineConfig,
    split: str,
    case_ids: Sequence[str] | None = None,
    max_cases: int | None = None,
    verify_source_hashes: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build and strongly verify a development cache without model execution."""
    all_records = development_cases(manifest, split)  # rejects test before any path resolution
    records, selected_ids, is_full_split = _select_split_records(all_records, case_ids)
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max_cases must be positive")
        records = records[:max_cases]
        selected_ids = selected_ids[:max_cases]
        is_full_split = is_full_split and len(records) == len(all_records)
    started = time.perf_counter()
    case_reports: list[dict[str, Any]] = []
    for position, case in enumerate(records, start=1):
        case_started = time.perf_counter()
        source_report: dict[str, Any] | None = None
        if verify_source_hashes:
            source_report = verify_case_source_hashes(dataset_root, case)
        prepared = prepare_case(
            project_root=project_root,
            dataset_root=dataset_root,
            case=case,
            schema=schema,
            preprocessing=preprocessing,
            preprocessing_sha256=preprocessing_sha256,
            config=config,
            allow_cache_write=True,
            verify_source_hashes=False,
            verify_cache_hashes=True,
        )
        case_dir = _cache_case_dir(
            project_root.resolve(), config, preprocessing_sha256, prepared.case_id
        )
        report = {
            "position": position,
            "case_id": prepared.case_id,
            "split": prepared.split,
            "cache_hit_before_prewarm": prepared.cache_hit,
            "shape": list(prepared.ct_hu.shape),
            "voxels": int(np.prod(prepared.ct_hu.shape)),
            "cache_bytes": _committed_cache_bytes(case_dir),
            "source_files_hashed": 0 if source_report is None else source_report["unique_files_checked"],
            "source_bytes_hashed": 0 if source_report is None else source_report["bytes_hashed"],
            "elapsed_seconds": time.perf_counter() - case_started,
        }
        case_reports.append(report)
        if progress is not None:
            progress(dict(report))
    elapsed = time.perf_counter() - started
    total_voxels = sum(record["voxels"] for record in case_reports)
    total_cache_bytes = sum(record["cache_bytes"] for record in case_reports)
    return {
        "schema_version": 1,
        "operation": "cache_prewarm_only_no_model_execution",
        "status": "passed",
        "split": split,
        "selected_case_ids": list(selected_ids),
        "is_full_split": is_full_split,
        "cases_requested": len(records),
        "cases_completed": len(case_reports),
        "cache_hits": sum(record["cache_hit_before_prewarm"] for record in case_reports),
        "cache_misses_written": sum(not record["cache_hit_before_prewarm"] for record in case_reports),
        "verify_source_hashes": verify_source_hashes,
        "source_files_hashed": sum(record["source_files_hashed"] for record in case_reports),
        "source_bytes_hashed": sum(record["source_bytes_hashed"] for record in case_reports),
        "cache_bytes": total_cache_bytes,
        "voxels": total_voxels,
        "elapsed_seconds": elapsed,
        "voxels_per_second": total_voxels / elapsed if elapsed else None,
        "cache_mib_per_second": total_cache_bytes / (1024**2) / elapsed if elapsed else None,
        "cases": case_reports,
    }


class ExactSplitPatchDataset(Dataset[dict[str, Any]]):
    """One deterministic class-balanced patch per declared train case/repeat."""

    def __init__(
        self,
        *,
        project_root: Path,
        dataset_root: Path,
        manifest: dict[str, Any],
        schema: OrganSchema,
        preprocessing: PreprocessingSpec,
        preprocessing_sha256: str,
        config: BaselineConfig,
        split: str,
        augment: bool,
        allow_cache_write: bool,
        case_ids: Sequence[str] | None = None,
        fixed_sample_indices: Sequence[int] | None = None,
        verify_cache_hashes: bool = False,
    ) -> None:
        all_records = development_cases(manifest, split)  # rejects test before root/path resolution
        if split != config.train_split:
            raise ValueError("Patch sampling is restricted to the exact configured training split")
        records, selected_case_ids, is_full_split = _select_split_records(all_records, case_ids)
        if any(record.get("split") != split for record in records):
            raise ValueError("Patch dataset contains a cross-split record")
        if fixed_sample_indices is None:
            fixed_indices: tuple[int, ...] | None = None
        else:
            if isinstance(fixed_sample_indices, (str, bytes)):
                raise TypeError("fixed_sample_indices must be an integer sequence")
            fixed_indices = tuple(fixed_sample_indices)
            if (
                not fixed_indices
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in fixed_indices)
                or len(set(fixed_indices)) != len(fixed_indices)
            ):
                raise ValueError("fixed_sample_indices must be non-empty, unique, non-negative integers")
        if not augment:
            self.transform = build_validation_transform(config)
        else:
            self.transform = build_train_transform(config)
        self.project_root = project_root.resolve()
        self.dataset_root = dataset_root.resolve()
        self.records = records
        self.full_split_case_ids = tuple(str(record["case_id"]) for record in all_records)
        self.selected_case_ids = selected_case_ids
        self.is_full_split = is_full_split
        self.schema = schema
        self.preprocessing = preprocessing
        self.preprocessing_sha256 = preprocessing_sha256
        self.config = config
        self.augment = augment
        self.allow_cache_write = allow_cache_write
        self.verify_cache_hashes = verify_cache_hashes
        self.fixed_sample_indices = fixed_indices
        self.epoch = 0
        self.accessed_case_ids: list[str] = []

    def __len__(self) -> int:
        samples_per_case = (
            len(self.fixed_sample_indices)
            if self.fixed_sample_indices is not None
            else self.config.train_patches_per_case
        )
        return len(self.records) * samples_per_case

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        samples_per_case = (
            len(self.fixed_sample_indices)
            if self.fixed_sample_indices is not None
            else self.config.train_patches_per_case
        )
        case_index = index // samples_per_case
        repeat_index = index % samples_per_case
        case = self.records[case_index]
        if self.fixed_sample_indices is None:
            sample_index = self.epoch * len(self) + case_index * samples_per_case + repeat_index
        else:
            sample_index = self.fixed_sample_indices[repeat_index]
        prepared = prepare_case(
            project_root=self.project_root,
            dataset_root=self.dataset_root,
            case=case,
            schema=self.schema,
            preprocessing=self.preprocessing,
            preprocessing_sha256=self.preprocessing_sha256,
            config=self.config,
            allow_cache_write=self.allow_cache_write,
            verify_cache_hashes=self.verify_cache_hashes,
        )
        image, label, trace = sample_class_balanced_hu_patch(
            prepared.ct_hu,
            prepared.label,
            self.preprocessing,
            case_id=prepared.case_id,
            sample_index=sample_index,
        )
        item: dict[str, Any] = {
            "image": image,
            "label": label[np.newaxis, ...],
            "case_id": prepared.case_id,
            "split": prepared.split,
            "sample_index": sample_index,
            "patch_trace": patch_trace_record(trace),
            "cache_hit": prepared.cache_hit,
        }
        if self.augment:
            self.transform.set_random_state(
                seed=_augmentation_seed(self.config.random_seed, prepared.case_id, sample_index)
            )
        transformed = self.transform(item)
        image_tensor = transformed["image"]
        label_tensor = transformed["label"]
        if image_tensor.shape != (1, *self.config.roi_size) or label_tensor.shape != (1, *self.config.roi_size):
            raise ValueError("Transformed patch shapes violate the model contract")
        if image_tensor.dtype != torch.float32 or label_tensor.dtype != torch.int64:
            raise ValueError("Transformed patch dtypes violate the model contract")
        if not torch.isfinite(image_tensor).all() or image_tensor.min() < 0 or image_tensor.max() > 1:
            raise ValueError("Transformed image is non-finite or outside [0,1]")
        if not set(int(value) for value in torch.unique(label_tensor).tolist()) <= set(range(self.config.out_channels)):
            raise ValueError("Transformed label contains an invalid class ID")
        self.accessed_case_ids.append(prepared.case_id)
        return transformed


class ExactSplitVolumeDataset(Dataset[dict[str, Any]]):
    """Deterministic full-volume validation dataset for future sliding-window inference."""

    def __init__(
        self,
        *,
        project_root: Path,
        dataset_root: Path,
        manifest: dict[str, Any],
        schema: OrganSchema,
        preprocessing: PreprocessingSpec,
        preprocessing_sha256: str,
        config: BaselineConfig,
        split: str,
        allow_cache_write: bool,
        case_ids: Sequence[str] | None = None,
        verify_cache_hashes: bool = False,
    ) -> None:
        all_records = development_cases(manifest, split)  # rejects test before root/path resolution
        if split != config.validation_split:
            raise ValueError("Full-volume dataset is restricted to the exact validation split")
        records, selected_case_ids, is_full_split = _select_split_records(all_records, case_ids)
        if any(record.get("split") != split for record in records):
            raise ValueError("Validation dataset contains a cross-split record")
        self.project_root = project_root.resolve()
        self.dataset_root = dataset_root.resolve()
        self.records = records
        self.full_split_case_ids = tuple(str(record["case_id"]) for record in all_records)
        self.selected_case_ids = selected_case_ids
        self.is_full_split = is_full_split
        self.schema = schema
        self.preprocessing = preprocessing
        self.preprocessing_sha256 = preprocessing_sha256
        self.config = config
        self.allow_cache_write = allow_cache_write
        self.verify_cache_hashes = verify_cache_hashes
        self.transform = build_validation_transform(config)
        self.accessed_case_ids: list[str] = []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.records[index]
        prepared = prepare_case(
            project_root=self.project_root,
            dataset_root=self.dataset_root,
            case=case,
            schema=self.schema,
            preprocessing=self.preprocessing,
            preprocessing_sha256=self.preprocessing_sha256,
            config=self.config,
            allow_cache_write=self.allow_cache_write,
            verify_cache_hashes=self.verify_cache_hashes,
        )
        item = self.transform(
            {
                "image": normalize_ct(prepared.ct_hu, self.preprocessing)[np.newaxis, ...],
                "label": prepared.label[np.newaxis, ...],
                "case_id": prepared.case_id,
                "split": prepared.split,
                "affine": prepared.affine,
                "cache_hit": prepared.cache_hit,
            }
        )
        self.accessed_case_ids.append(prepared.case_id)
        return item
