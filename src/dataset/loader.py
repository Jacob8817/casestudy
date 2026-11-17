"""Leakage-safe, manifest-driven loading for selected-organ CT volumes.

The original TotalSegmentator files are opened read-only.  Source masks are
loaded one at a time and merged on the native CT grid so peak memory does not
scale with the 13 selected binary masks.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

from audit_dataset import INTEGER_DATATYPES, NUMPY_DTYPES, NiftiInfo, close_enough, inspect_nifti

from .label_mapping import OrganSchema
from .manifest import ALLOWED_SPLITS, canonical_sha256


DEVELOPMENT_SPLITS = frozenset({"train", "val"})


@dataclass(frozen=True)
class LoadedCase:
    """A CT and its exclusive 11-class label map on the audited native grid."""

    case_id: str
    split: str
    ct_hu: np.ndarray
    label: np.ndarray
    info: NiftiInfo
    affine: np.ndarray
    raw_foreground_voxels: dict[str, int]
    exclusive_foreground_voxels: dict[str, int]
    overwritten_voxels: dict[str, int]
    cross_target_conflict_voxels: int
    source_masks_checked: int


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    payload.pop("generated_at_utc", None)
    return canonical_sha256(payload)


def load_validated_manifest(path: Path, schema: OrganSchema) -> dict[str, Any]:
    """Load the immutable dataset manifest and validate its critical locks."""
    path = path.resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")
    expected = manifest.get("manifest_sha256")
    actual = _manifest_fingerprint(manifest)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"Manifest fingerprint mismatch: expected={expected!r}, actual={actual}")
    recorded_schema = manifest.get("organ_schema", {}).get("canonical_sha256")
    if recorded_schema != schema.fingerprint():
        raise ValueError("Manifest organ schema does not match the configured organ schema")
    test_lock = manifest.get("split_policy", {}).get("test_lock", {})
    if test_lock.get("locked") is not True:
        raise ValueError("Manifest test split is not locked")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain a non-empty cases list")
    split_counts = {split: 0 for split in ALLOWED_SPLITS}
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every manifest case must be an object")
        case_id = case.get("case_id")
        split = case.get("split")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"Invalid or duplicate case_id: {case_id!r}")
        if split not in split_counts:
            raise ValueError(f"Unsupported split {split!r} for {case_id}")
        seen.add(case_id)
        split_counts[split] += 1
    recorded_counts = manifest.get("split_policy", {}).get("counts")
    if recorded_counts != split_counts:
        raise ValueError(f"Manifest split counts disagree with cases: {recorded_counts} != {split_counts}")
    return manifest


def development_cases(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Return train or validation cases; reject test before resolving any path."""
    if split not in DEVELOPMENT_SPLITS:
        raise PermissionError("The test split is locked and unavailable to preprocessing QA")
    return [case for case in manifest["cases"] if case["split"] == split]


def get_development_case(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Find one predeclared train/validation case without exposing test data."""
    matches = [case for case in manifest["cases"] if case.get("case_id") == case_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one manifest entry for {case_id!r}, found {len(matches)}")
    case = matches[0]
    if case.get("split") not in DEVELOPMENT_SPLITS:
        raise PermissionError("The test split is locked and unavailable to preprocessing QA")
    return case


def resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while preventing absolute/path-traversal access."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Manifest path must be a non-empty string")
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe manifest path: {relative_path!r}")
    root = dataset_root.resolve()
    candidate = root.joinpath(*posix_path.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes dataset root: {relative_path!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def affine_from_info(info: NiftiInfo) -> np.ndarray:
    if len(info.transform) != 12:
        raise ValueError(f"NIfTI has no usable affine: {info.path}")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(info.transform, dtype=np.float64).reshape(3, 4)
    return affine


def _open_payload(path: Path, info: NiftiInfo) -> BinaryIO:
    stream: BinaryIO = gzip.open(path, "rb") if info.compressed else path.open("rb")
    stream.seek(int(round(info.vox_offset)))
    return stream


def read_nifti_volume(
    path: Path,
    info: NiftiInfo | None = None,
    *,
    apply_scaling: bool,
) -> np.ndarray:
    """Read one audited NIfTI-1 volume in native (Fortran) voxel order.

    Compressed streams are drained through EOF, which rechecks their gzip CRC.
    """
    path = path.resolve()
    info = info or inspect_nifti(path, "header")
    dtype_code = NUMPY_DTYPES.get(info.datatype)
    if dtype_code is None:
        raise ValueError(f"Unsupported NIfTI datatype {info.datatype} in {path}")
    dtype = np.dtype(info.endian + dtype_code)
    expected = math.prod(info.shape) * dtype.itemsize
    stream: BinaryIO | None = None
    try:
        stream = _open_payload(path, info)
        payload = stream.read(expected)
        if len(payload) != expected:
            raise ValueError(f"Truncated NIfTI payload in {path}: {len(payload)} != {expected} bytes")
        trailing = 0
        while chunk := stream.read(8 * 1024 * 1024):
            trailing += len(chunk)
        if trailing:
            raise ValueError(f"Unexpected {trailing} decompressed bytes after voxel payload in {path}")
    except (OSError, EOFError) as exc:
        raise ValueError(f"NIfTI decompression/CRC failure in {path}: {exc}") from exc
    finally:
        if stream is not None:
            stream.close()
    raw = np.frombuffer(payload, dtype=dtype).reshape(info.shape, order="F")
    if not apply_scaling:
        return raw.copy()
    values = raw.astype(np.float32)
    slope = info.scl_slope if math.isfinite(info.scl_slope) and info.scl_slope != 0 else 1.0
    intercept = info.scl_inter if math.isfinite(info.scl_inter) and info.scl_slope != 0 else 0.0
    if slope != 1.0:
        values *= np.float32(slope)
    if intercept != 0.0:
        values += np.float32(intercept)
    return values


def _geometry_matches(mask: NiftiInfo, ct: NiftiInfo) -> bool:
    return (
        mask.shape == ct.shape
        and close_enough(mask.voxel_spacing[:3], ct.voxel_spacing[:3], 1e-4)
        and bool(mask.transform)
        and bool(ct.transform)
        and close_enough(mask.transform, ct.transform, 1e-3)
        and mask.spatial_units == ct.spatial_units
    )


def _effective_scaling(info: NiftiInfo) -> tuple[float, float]:
    if not math.isfinite(info.scl_slope) or info.scl_slope == 0:
        return 1.0, 0.0
    return info.scl_slope, info.scl_inter if math.isfinite(info.scl_inter) else 0.0


def load_exclusive_case(
    dataset_root: Path,
    case: dict[str, Any],
    schema: OrganSchema,
    *,
    test_authorization: dict[str, Any] | None = None,
) -> LoadedCase:
    """Load CT plus the configured exclusive label map.

    Train/validation access remains the default.  A test record is accepted only
    with the narrow, explicit receipt created after the Phase 6 validation
    protocol has been frozen.  This keeps every existing preprocessing/training
    caller fail-closed while permitting the single final evaluation workflow.
    """
    split = str(case.get("split"))
    if split == "test":
        authorization = test_authorization or {}
        if (
            authorization.get("authorization_token") != "FINAL_ONE_SHOT_TEST_EVALUATION"
            or authorization.get("status") != "authorized"
            or not isinstance(authorization.get("frozen_protocol_sha256"), str)
            or len(authorization["frozen_protocol_sha256"]) != 64
        ):
            raise PermissionError(
                "The test split requires a valid post-validation one-shot authorization receipt"
            )
    elif split not in DEVELOPMENT_SPLITS:
        raise PermissionError("The test split is locked and unavailable to preprocessing QA")
    case_id = str(case.get("case_id"))
    ct_record = case.get("ct")
    if not isinstance(ct_record, dict):
        raise ValueError(f"{case_id}: missing CT manifest record")
    ct_path = resolve_dataset_path(dataset_root, ct_record.get("path", ""))
    ct_info = inspect_nifti(ct_path, "header")
    ct_hu = read_nifti_volume(ct_path, ct_info, apply_scaling=True)
    if ct_hu.dtype != np.float32 or not np.all(np.isfinite(ct_hu)):
        raise ValueError(f"{case_id}: CT must decode to finite float32 HU values")

    targets = case.get("targets")
    labels_by_name = {label.name: label for label in schema.labels}
    if not isinstance(targets, dict) or set(targets) != set(labels_by_name):
        raise ValueError(f"{case_id}: manifest targets do not match organ schema")

    label_map = np.zeros(ct_info.shape, dtype=np.uint8)
    raw_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    source_masks_checked = 0
    for target_name in schema.overlap_priority:
        definition = labels_by_name[target_name]
        target_record = targets[target_name]
        source_records = {
            source.get("name"): source
            for source in target_record.get("sources", [])
            if isinstance(source, dict)
        }
        if set(source_records) != set(definition.source_masks):
            raise ValueError(f"{case_id}/{target_name}: source-mask mapping mismatch")
        target_union = np.zeros(ct_info.shape, dtype=bool)
        for source_name in definition.source_masks:
            source_path = resolve_dataset_path(dataset_root, source_records[source_name].get("path", ""))
            source_info = inspect_nifti(source_path, "header")
            if not _geometry_matches(source_info, ct_info):
                raise ValueError(f"{case_id}/{source_name}: source mask geometry differs from CT")
            if source_info.datatype not in INTEGER_DATATYPES:
                raise ValueError(f"{case_id}/{source_name}: mask datatype must be integer")
            if not close_enough(_effective_scaling(source_info), (1.0, 0.0), 1e-6):
                raise ValueError(f"{case_id}/{source_name}: mask scaling is not identity")
            source = read_nifti_volume(source_path, source_info, apply_scaling=False)
            valid = np.logical_or(source == 0, source == 1)
            if not bool(np.all(valid)):
                raise ValueError(f"{case_id}/{source_name}: source mask is not binary")
            target_union |= source == 1
            source_masks_checked += 1
            del source, valid
        raw_counts[target_name] = int(np.count_nonzero(target_union))
        conflict_counts[target_name] = int(np.count_nonzero(target_union & (label_map != 0)))
        label_map[target_union] = np.uint8(definition.id)
        del target_union

    allowed_ids = set(range(schema.num_output_channels))
    found_ids = set(int(value) for value in np.unique(label_map))
    if not found_ids <= allowed_ids:
        raise ValueError(f"{case_id}: exclusive label map has unexpected ids {sorted(found_ids)}")
    exclusive_counts = {
        label.name: int(np.count_nonzero(label_map == label.id)) for label in schema.labels
    }
    overwritten = {name: raw_counts[name] - exclusive_counts[name] for name in raw_counts}
    return LoadedCase(
        case_id=case_id,
        split=split,
        ct_hu=ct_hu,
        label=label_map,
        info=ct_info,
        affine=affine_from_info(ct_info),
        raw_foreground_voxels=raw_counts,
        exclusive_foreground_voxels=exclusive_counts,
        overwritten_voxels=overwritten,
        cross_target_conflict_voxels=sum(conflict_counts.values()),
        source_masks_checked=source_masks_checked,
    )
