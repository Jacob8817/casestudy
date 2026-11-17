"""Full payload audit for the selected multi-organ mask subset."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import numpy as np

from audit_dataset import INTEGER_DATATYPES, NUMPY_DTYPES, NiftiInfo, close_enough, inspect_nifti

from .label_mapping import OrganDefinition, OrganSchema


SELECTED_AUDIT_VERSION = "1.0.0"


@dataclass
class SourceResult:
    name: str
    target_name: str
    path: str
    header_valid: bool = False
    geometry_valid: bool = False
    scaling_valid: bool = False
    binary_valid: bool = False
    payload_valid: bool = False
    crc_checked: bool = False
    present: bool = False
    foreground_voxels: int = 0
    nonbinary_voxels: int = 0
    raw_min: int | float | None = None
    raw_max: int | float | None = None
    trailing_decompressed_bytes: int = 0
    info: NiftiInfo | None = None
    error: str = ""


@dataclass
class TargetResult:
    id: int
    name: str
    source_masks: tuple[str, ...]
    integrity_valid: bool = False
    present: bool = False
    exclusive_present: bool | None = None
    foreground_voxels: int | None = None
    volume_ml: float | None = None
    exclusive_foreground_voxels: int | None = None
    exclusive_volume_ml: float | None = None
    voxels_removed_by_overlap_resolution: int | None = None
    merged_source_overlap_voxels: int | None = None
    cross_target_overlap_voxels: int | None = None


@dataclass
class CaseResult:
    case_id: str
    split: str
    ct_path: str
    voxel_volume_mm3: float | None = None
    spatial_units_assumed_mm: bool = False
    sources: dict[str, SourceResult] = field(default_factory=dict)
    targets: dict[str, TargetResult] = field(default_factory=dict)
    pair_overlaps: dict[str, int] = field(default_factory=dict)
    conflict_voxels: int | None = None
    metrics_valid: bool = False
    issues: list[dict[str, str]] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue["severity"] == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue["severity"] == "warning" for issue in self.issues)


def _issue(
    result: CaseResult,
    severity: str,
    category: str,
    detail: str,
    *,
    source: str = "",
    target: str = "",
) -> None:
    result.issues.append(
        {
            "severity": severity,
            "category": category,
            "source": source,
            "target": target,
            "detail": detail,
        }
    )


def _effective_scaling(info: NiftiInfo) -> tuple[float, float]:
    if not math.isfinite(info.scl_slope) or info.scl_slope == 0:
        return 1.0, 0.0
    intercept = info.scl_inter if math.isfinite(info.scl_inter) else 0.0
    return info.scl_slope, intercept


def _voxel_volume_mm3(info: NiftiInfo) -> tuple[float, bool]:
    if not info.transform:
        raise ValueError("CT has no usable spatial affine")
    a, b, c, _x, d, e, f, _y, g, h, i, _z = info.transform
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    unit_scale = {"meter": 1000.0, "mm": 1.0, "micron": 0.001}.get(info.spatial_units)
    assumed_mm = unit_scale is None
    if assumed_mm:
        unit_scale = 1.0
    return abs(determinant) * unit_scale**3, assumed_mm


def _geometry_matches(mask: NiftiInfo, ct: NiftiInfo) -> bool:
    return (
        mask.shape == ct.shape
        and close_enough(mask.voxel_spacing[:3], ct.voxel_spacing[:3], 1e-4)
        and bool(mask.transform)
        and bool(ct.transform)
        and close_enough(mask.transform, ct.transform, 1e-3)
        and mask.spatial_units == ct.spatial_units
    )


def _open_payload(path: Path, info: NiftiInfo) -> BinaryIO:
    stream: BinaryIO
    if info.compressed:
        stream = gzip.open(path, "rb")
    else:
        stream = path.open("rb")
    stream.seek(int(round(info.vox_offset)))
    return stream


def _drain_to_eof(stream: BinaryIO, chunk_bytes: int = 8 * 1024 * 1024) -> int:
    trailing = 0
    while chunk := stream.read(chunk_bytes):
        trailing += len(chunk)
    return trailing


def _scan_source_individually(path: Path, source: SourceResult, chunk_voxels: int) -> None:
    info = source.info
    if info is None or info.datatype not in NUMPY_DTYPES:
        return
    dtype = np.dtype(info.endian + NUMPY_DTYPES[info.datatype])
    remaining = math.prod(info.shape)
    stream: BinaryIO | None = None
    try:
        stream = _open_payload(path, info)
        while remaining:
            count = min(chunk_voxels, remaining)
            expected = count * dtype.itemsize
            payload = stream.read(expected)
            if len(payload) != expected:
                raise ValueError(f"truncated payload: expected {expected} bytes, received {len(payload)}")
            remaining -= count
            values = np.frombuffer(payload, dtype=dtype, count=count)
            current_min = values.min().item()
            current_max = values.max().item()
            source.raw_min = current_min if source.raw_min is None else min(source.raw_min, current_min)
            source.raw_max = current_max if source.raw_max is None else max(source.raw_max, current_max)
            valid = np.logical_or(values == 0, values == 1)
            source.nonbinary_voxels += int(values.size - np.count_nonzero(valid))
            source.foreground_voxels += int(np.count_nonzero(values == 1))
        source.trailing_decompressed_bytes = _drain_to_eof(stream)
        source.payload_valid = True
        source.crc_checked = info.compressed
        source.binary_valid = source.nonbinary_voxels == 0
        source.present = source.foreground_voxels > 0
    except (OSError, EOFError, ValueError) as exc:
        source.error = str(exc)
    finally:
        if stream is not None:
            stream.close()


def _stream_compatible_sources(
    result: CaseResult,
    schema: OrganSchema,
    source_paths: dict[str, Path],
    chunk_voxels: int,
) -> tuple[np.ndarray, dict[str, int]]:
    patterns = 1 << len(schema.labels)
    pattern_histogram = np.zeros(patterns, dtype=np.uint64)
    merged_source_overlaps = {label.name: 0 for label in schema.labels}
    readers: dict[str, BinaryIO] = {}
    dtypes: dict[str, np.dtype] = {}
    source_to_target = {
        source_name: label
        for label in schema.labels
        for source_name in label.source_masks
    }
    try:
        for source_name, source_result in result.sources.items():
            info = source_result.info
            if info is None:
                raise ValueError(f"{source_name}: missing NIfTI information")
            readers[source_name] = _open_payload(source_paths[source_name], info)
            dtypes[source_name] = np.dtype(info.endian + NUMPY_DTYPES[info.datatype])

        remaining = math.prod(next(iter(result.sources.values())).info.shape)  # type: ignore[union-attr]
        while remaining:
            count = min(chunk_voxels, remaining)
            occupancy = np.zeros(count, dtype=np.uint16)
            merged_hits = {
                label.name: np.zeros(count, dtype=np.uint8)
                for label in schema.labels
                if len(label.source_masks) > 1
            }
            remaining -= count
            for source_name in schema.source_masks:
                source = result.sources[source_name]
                dtype = dtypes[source_name]
                expected = count * dtype.itemsize
                try:
                    payload = readers[source_name].read(expected)
                    if len(payload) != expected:
                        raise ValueError(f"truncated payload: expected {expected} bytes, received {len(payload)}")
                    values = np.frombuffer(payload, dtype=dtype, count=count)
                    current_min = values.min().item()
                    current_max = values.max().item()
                    source.raw_min = current_min if source.raw_min is None else min(source.raw_min, current_min)
                    source.raw_max = current_max if source.raw_max is None else max(source.raw_max, current_max)
                    valid = np.logical_or(values == 0, values == 1)
                    source.nonbinary_voxels += int(values.size - np.count_nonzero(valid))
                    foreground = values == 1
                    source.foreground_voxels += int(np.count_nonzero(foreground))
                    label = source_to_target[source_name]
                    occupancy[foreground] |= np.uint16(1 << (label.id - 1))
                    if label.name in merged_hits:
                        merged_hits[label.name][foreground] += 1
                except (OSError, EOFError, ValueError) as exc:
                    source.error = str(exc)
                    raise

            for target_name, hits in merged_hits.items():
                merged_source_overlaps[target_name] += int(np.count_nonzero(hits > 1))
            pattern_histogram += np.bincount(occupancy, minlength=patterns).astype(np.uint64)

        for source_name, stream in readers.items():
            source = result.sources[source_name]
            try:
                source.trailing_decompressed_bytes = _drain_to_eof(stream)
                source.payload_valid = True
                source.crc_checked = source.info.compressed if source.info else False
                source.binary_valid = source.nonbinary_voxels == 0
                source.present = source.foreground_voxels > 0
            except (OSError, EOFError) as exc:
                source.error = f"payload decompression/CRC failure: {exc}"
    finally:
        for stream in readers.values():
            stream.close()
    return pattern_histogram, merged_source_overlaps


def audit_selected_case(
    case: dict[str, Any],
    dataset_root: Path,
    schema: OrganSchema,
    chunk_voxels: int,
) -> CaseResult:
    case_id = str(case["case_id"])
    split = str(case["split"])
    ct_path = dataset_root / case["ct"]["path"]
    result = CaseResult(case_id=case_id, split=split, ct_path=str(ct_path.resolve()))
    try:
        ct_info = inspect_nifti(ct_path, "header")
        result.voxel_volume_mm3, result.spatial_units_assumed_mm = _voxel_volume_mm3(ct_info)
    except (OSError, ValueError) as exc:
        _issue(result, "error", "invalid_ct_geometry", str(exc))
        return result

    source_paths: dict[str, Path] = {}
    target_by_name = {label.name: label for label in schema.labels}
    manifest_targets = case.get("targets")
    if not isinstance(manifest_targets, dict) or set(manifest_targets) != set(target_by_name):
        _issue(result, "error", "manifest_target_mismatch", "Case targets do not match organ schema")
        return result

    for label in schema.labels:
        manifest_sources = manifest_targets[label.name].get("sources", [])
        source_records = {item.get("name"): item for item in manifest_sources if isinstance(item, dict)}
        if set(source_records) != set(label.source_masks):
            _issue(
                result,
                "error",
                "manifest_source_mismatch",
                f"Expected {label.source_masks}, found {tuple(sorted(source_records))}",
                target=label.name,
            )
        for source_name in label.source_masks:
            record = source_records.get(source_name, {})
            relative_path = record.get("path", "")
            path = dataset_root / relative_path if relative_path else dataset_root / "__missing__"
            source_paths[source_name] = path
            source = SourceResult(source_name, label.name, str(path.resolve()))
            result.sources[source_name] = source
            if not path.is_file():
                source.error = "source mask is missing"
                _issue(result, "error", "missing_source_mask", source.error, source=source_name, target=label.name)
                continue
            try:
                info = inspect_nifti(path, "header")
                source.info = info
                source.header_valid = True
                source.geometry_valid = _geometry_matches(info, ct_info)
                source.scaling_valid = close_enough(_effective_scaling(info), (1.0, 0.0), 1e-6)
                if info.datatype not in INTEGER_DATATYPES or info.datatype not in NUMPY_DTYPES:
                    source.error = f"unsupported/non-integer mask datatype {info.datatype}"
                elif not source.geometry_valid:
                    source.error = "mask geometry differs from CT"
                elif not source.scaling_valid:
                    source.error = f"mask scaling must be identity, found {_effective_scaling(info)}"
            except (OSError, ValueError) as exc:
                source.error = str(exc)
            if source.error:
                category = (
                    "mask_geometry_mismatch"
                    if not source.geometry_valid and source.header_valid
                    else "invalid_mask_header_or_scaling"
                )
                _issue(result, "error", category, source.error, source=source_name, target=label.name)

    all_compatible = len(result.sources) == len(schema.source_masks) and all(
        source.header_valid
        and source.geometry_valid
        and source.scaling_valid
        and source.info is not None
        and source.info.datatype in NUMPY_DTYPES
        for source in result.sources.values()
    )

    histogram: np.ndarray | None = None
    merged_overlaps: dict[str, int] = {}
    if all_compatible:
        try:
            histogram, merged_overlaps = _stream_compatible_sources(
                result, schema, source_paths, chunk_voxels
            )
        except (OSError, EOFError, ValueError) as exc:
            _issue(result, "error", "selected_payload_stream_failure", str(exc))
    else:
        for source_name, source in result.sources.items():
            if source.header_valid and source.info and source.info.datatype in NUMPY_DTYPES:
                _scan_source_individually(source_paths[source_name], source, chunk_voxels)

    for source in result.sources.values():
        if source.error and not any(issue["source"] == source.name for issue in result.issues):
            _issue(result, "error", "mask_payload_failure", source.error, source=source.name, target=source.target_name)
        if source.nonbinary_voxels:
            detail = (
                "Mask contains values outside {0,1}; counts and range are redacted for locked test data"
                if result.split == "test"
                else f"Found {source.nonbinary_voxels} voxels outside {{0,1}}; range={source.raw_min}..{source.raw_max}"
            )
            _issue(
                result,
                "error",
                "nonbinary_mask",
                detail,
                source=source.name,
                target=source.target_name,
            )
        if source.trailing_decompressed_bytes:
            detail = (
                "Unexpected decompressed bytes occur after the expected payload; count redacted for locked test data"
                if result.split == "test"
                else f"Found {source.trailing_decompressed_bytes} decompressed bytes after the expected payload"
            )
            _issue(
                result,
                "warning",
                "unexpected_trailing_mask_bytes",
                detail,
                source=source.name,
                target=source.target_name,
            )

    sources_valid = all(
        source.header_valid
        and source.geometry_valid
        and source.scaling_valid
        and source.binary_valid
        and source.payload_valid
        for source in result.sources.values()
    ) and len(result.sources) == len(schema.source_masks)

    pattern_values = np.arange(1 << len(schema.labels), dtype=np.uint16)
    if histogram is not None and sources_valid:
        popcounts = np.array([int(value).bit_count() for value in pattern_values], dtype=np.uint8)
        result.conflict_voxels = int(histogram[popcounts > 1].sum())
        winning_label = np.zeros(pattern_values.size, dtype=np.uint8)
        labels_by_name = {label.name: label for label in schema.labels}
        for target_name in schema.overlap_priority:
            priority_label = labels_by_name[target_name]
            priority_bit = np.uint16(1 << (priority_label.id - 1))
            winning_label[(pattern_values & priority_bit) != 0] = priority_label.id
        for label in schema.labels:
            bit = np.uint16(1 << (label.id - 1))
            foreground_voxels = int(histogram[(pattern_values & bit) != 0].sum())
            exclusive_voxels = int(histogram[winning_label == label.id].sum())
            target = TargetResult(
                id=label.id,
                name=label.name,
                source_masks=label.source_masks,
                integrity_valid=all(result.sources[source].payload_valid for source in label.source_masks),
                present=foreground_voxels > 0,
                exclusive_present=exclusive_voxels > 0,
                foreground_voxels=foreground_voxels,
                volume_ml=foreground_voxels * result.voxel_volume_mm3 / 1000.0 if result.voxel_volume_mm3 else None,
                exclusive_foreground_voxels=exclusive_voxels,
                exclusive_volume_ml=exclusive_voxels * result.voxel_volume_mm3 / 1000.0 if result.voxel_volume_mm3 else None,
                voxels_removed_by_overlap_resolution=foreground_voxels - exclusive_voxels,
                merged_source_overlap_voxels=merged_overlaps.get(label.name, 0),
                cross_target_overlap_voxels=0,
            )
            result.targets[label.name] = target
        for left_index, left in enumerate(schema.labels):
            left_bit = np.uint16(1 << (left.id - 1))
            for right in schema.labels[left_index + 1 :]:
                right_bit = np.uint16(1 << (right.id - 1))
                overlap = int(histogram[((pattern_values & left_bit) != 0) & ((pattern_values & right_bit) != 0)].sum())
                result.pair_overlaps[f"{left.name}|{right.name}"] = overlap
                result.targets[left.name].cross_target_overlap_voxels = (
                    result.targets[left.name].cross_target_overlap_voxels or 0
                ) + overlap
                result.targets[right.name].cross_target_overlap_voxels = (
                    result.targets[right.name].cross_target_overlap_voxels or 0
                ) + overlap
                if overlap:
                    detail = (
                        f"{left.name} and {right.name} overlap; voxel count redacted for locked test data"
                        if result.split == "test"
                        else f"{left.name} and {right.name} overlap in {overlap} voxels"
                    )
                    _issue(
                        result,
                        "error" if schema.overlap_policy == "error" else "warning",
                        (
                            "cross_target_overlap_resolved"
                            if schema.overlap_policy == "priority"
                            else "cross_target_overlap"
                        ),
                        detail,
                        target=f"{left.name}|{right.name}",
                    )
        for label in schema.labels:
            within = result.targets[label.name].merged_source_overlap_voxels or 0
            if within:
                detail = (
                    "Merged source masks overlap; count redacted for locked test data and union remains deterministic"
                    if result.split == "test"
                    else f"Merged source masks overlap in {within} voxels; union is deterministic"
                )
                _issue(
                    result,
                    "warning",
                    "merged_source_overlap",
                    detail,
                    target=label.name,
                )
        result.metrics_valid = True
    else:
        for label in schema.labels:
            source_results = [result.sources.get(source) for source in label.source_masks]
            result.targets[label.name] = TargetResult(
                id=label.id,
                name=label.name,
                source_masks=label.source_masks,
                integrity_valid=all(source and source.payload_valid for source in source_results),
                present=any(source and source.present for source in source_results),
            )
    return result


def _descriptive_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def write_selected_audit_outputs(
    results: Sequence[CaseResult],
    schema: OrganSchema,
    manifest: dict[str, Any],
    output_csv: Path,
    output_json: Path,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item.case_id):
        case_any_present = any(target.present for target in result.targets.values())
        for label in schema.labels:
            target = result.targets[label.name]
            sources = [result.sources[source] for source in label.source_masks]
            redact = result.split == "test"
            target_issues = [
                issue
                for issue in result.issues
                if issue["target"] in {"", label.name}
                or label.name in issue["target"].split("|")
                or issue["source"] in label.source_masks
            ]
            rows.append(
                {
                    "case_id": result.case_id,
                    "split": result.split,
                    "class_id": label.id,
                    "organ": label.name,
                    "source_masks": ";".join(label.source_masks),
                    "source_files_expected": len(label.source_masks),
                    "source_files_audited": len(sources),
                    "header_valid": all(source.header_valid for source in sources),
                    "geometry_valid": all(source.geometry_valid for source in sources),
                    "identity_scaling": all(source.scaling_valid for source in sources),
                    "binary_valid": all(source.binary_valid for source in sources),
                    "payload_valid": all(source.payload_valid for source in sources),
                    "gzip_crc_checked": all(source.crc_checked for source in sources),
                    "present": target.present,
                    "exclusive_present": "" if redact else target.exclusive_present,
                    "case_any_selected_organ_present": case_any_present,
                    "source_presence": json.dumps({source.name: source.present for source in sources}, sort_keys=True),
                    "foreground_voxels": "" if redact else target.foreground_voxels,
                    "volume_ml": "" if redact else target.volume_ml,
                    "exclusive_foreground_voxels": "" if redact else target.exclusive_foreground_voxels,
                    "exclusive_volume_ml": "" if redact else target.exclusive_volume_ml,
                    "voxels_removed_by_overlap_resolution": "" if redact else target.voxels_removed_by_overlap_resolution,
                    "merged_source_overlap_voxels": "" if redact else target.merged_source_overlap_voxels,
                    "cross_target_overlap_voxels": "" if redact else target.cross_target_overlap_voxels,
                    "test_metrics_redacted": redact,
                    "issue_count": len(target_issues),
                    "issues": json.dumps(target_issues, sort_keys=True),
                }
            )

    fieldnames = list(rows[0]) if rows else []
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(output_csv)

    split_counts = Counter(result.split for result in results)
    source_results = [source for result in results for source in result.sources.values()]
    issue_counts = Counter(issue["category"] for result in results for issue in result.issues)
    severity_counts = Counter(issue["severity"] for result in results for issue in result.issues)
    organ_statistics: dict[str, Any] = {}
    development_results = [result for result in results if result.split in {"train", "val"}]
    for label in schema.labels:
        by_split: dict[str, Any] = {}
        for split in ("train", "val", "development"):
            selected = (
                [result for result in development_results if result.split == split]
                if split != "development"
                else development_results
            )
            targets = [result.targets[label.name] for result in selected if result.metrics_valid]
            present_raw_volumes = [
                target.volume_ml for target in targets if target.present and target.volume_ml is not None
            ]
            present_exclusive_volumes = [
                target.exclusive_volume_ml
                for target in targets
                if target.exclusive_present and target.exclusive_volume_ml is not None
            ]
            removed_voxels = [
                target.voxels_removed_by_overlap_resolution or 0 for target in targets
            ]
            by_split[split] = {
                "case_count": len(selected),
                "cases_with_valid_metrics": len(targets),
                "present_cases": sum(target.present for target in targets),
                "empty_cases": sum(not target.present for target in targets),
                "presence_rate": (sum(target.present for target in targets) / len(targets)) if targets else None,
                "exclusive_present_cases": sum(bool(target.exclusive_present) for target in targets),
                "present_only_raw_mask_volume_ml": _descriptive_statistics(present_raw_volumes),
                "present_only_exclusive_training_volume_ml": _descriptive_statistics(present_exclusive_volumes),
                "overlap_resolution_removed_voxels": _descriptive_statistics(removed_voxels),
            }
        organ_statistics[label.name] = {
            "class_id": label.id,
            "source_masks": list(label.source_masks),
            "train_validation": by_split,
            "test": {
                "redacted": True,
                "case_count": split_counts["test"],
                "integrity_only": True,
            },
        }

    development_pair_overlaps: Counter[str] = Counter()
    test_any_overlap = False
    for result in results:
        if result.split == "test":
            test_any_overlap = test_any_overlap or any(result.pair_overlaps.values())
        else:
            development_pair_overlaps.update(result.pair_overlaps)

    summary: dict[str, Any] = {
        "audit_version": SELECTED_AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "organ_schema_name": schema.name,
        "organ_schema_sha256": schema.fingerprint(),
        "integrity_scope": "All selected source-mask payloads streamed to EOF; gzip CRC, raw binary values, identity scaling, CT geometry, and nonempty status checked.",
        "test_policy": {
            "locked": True,
            "integrity_checked": True,
            "per_organ_voxel_counts_and_volumes_redacted": True,
            "excluded_from_volume_statistics": True,
        },
        "overlap_resolution": {
            "policy": schema.overlap_policy,
            "method": schema.overlap_method,
            "priority_low_to_high": list(schema.overlap_priority),
            "reference": schema.overlap_reference,
            "raw_and_exclusive_development_metrics_reported": True,
        },
        "case_count": len(results),
        "split_counts": {split: split_counts[split] for split in ("train", "val", "test")},
        "target_organ_count": len(schema.labels),
        "source_masks_per_case": len(schema.source_masks),
        "source_file_count": len(source_results),
        "source_header_valid_count": sum(source.header_valid for source in source_results),
        "source_geometry_valid_count": sum(source.geometry_valid for source in source_results),
        "source_binary_valid_count": sum(source.binary_valid for source in source_results),
        "source_payload_valid_count": sum(source.payload_valid for source in source_results),
        "source_crc_checked_count": sum(source.crc_checked for source in source_results),
        "source_empty_count": sum(not source.present for source in source_results if source.payload_valid),
        "cases_with_valid_merged_metrics": sum(result.metrics_valid for result in results),
        "volume_units": {
            "reported_unit": "mL",
            "unknown_nifti_units_assumed_mm_cases": sum(result.spatial_units_assumed_mm for result in results),
            "assumption": "When the NIfTI spatial-units flag is unset, affine/spacing units are treated as millimetres; the dataset-wide audit found approximately 1.5 spacing on every axis.",
        },
        "development_cases_with_no_selected_foreground": sum(
            result.metrics_valid and not any(target.present for target in result.targets.values())
            for result in development_results
        ),
        "severity_counts": dict(sorted(severity_counts.items())),
        "issue_category_counts": dict(sorted(issue_counts.items())),
        "all_integrity_checks_passed": severity_counts.get("error", 0) == 0,
        "overlaps": {
            "development_pair_voxels": {
                pair: count for pair, count in sorted(development_pair_overlaps.items()) if count
            },
            "test_any_cross_target_overlap": test_any_overlap,
            "test_overlap_counts_redacted": True,
        },
        "organs": organ_statistics,
    }
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_json.replace(output_json)
    return summary


def run_selected_audit(
    dataset_root: Path,
    manifest: dict[str, Any],
    schema: OrganSchema,
    *,
    workers: int,
    chunk_voxels: int,
) -> list[CaseResult]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Manifest cases must be a list")
    results: list[CaseResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(
            lambda case: audit_selected_case(case, dataset_root.resolve(), schema, chunk_voxels),
            cases,
        )
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index % 50 == 0 or index == len(cases):
                print(f"Audited {index}/{len(cases)} cases", flush=True)
    return results


def write_selected_audit_report(summary: dict[str, Any], path: Path) -> None:
    """Write a concise human-readable selected-organ audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    severity = summary.get("severity_counts", {})
    overlap = summary["overlap_resolution"]
    lines = [
        "# Selected-Organ Dataset Audit",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Integrity",
        "",
        f"- Cases: **{summary['case_count']:,}**",
        f"- Selected source-mask files: **{summary['source_file_count']:,}**",
        f"- Header-valid: **{summary['source_header_valid_count']:,}**",
        f"- Geometry-valid: **{summary['source_geometry_valid_count']:,}**",
        f"- Binary-valid: **{summary['source_binary_valid_count']:,}**",
        f"- Payload/CRC-valid: **{summary['source_crc_checked_count']:,}**",
        f"- Errors: **{severity.get('error', 0):,}**",
        f"- Warnings: **{severity.get('warning', 0):,}**",
        "",
        "## Coverage",
        "",
        f"- Empty selected source masks: **{summary['source_empty_count']:,}**",
        f"- Train/validation cases with no selected foreground: **{summary['development_cases_with_no_selected_foreground']:,}**",
        "- Empty masks are retained as legitimate partial-field coverage, not classified as corruption.",
        "",
        "## Overlap policy",
        "",
        f"- Method: `{overlap['method']}`",
        f"- Priority, low to high: `{', '.join(overlap['priority_low_to_high'])}`",
        f"- Reference: {overlap['reference']}",
        "- Raw mask volumes and exclusive post-resolution training volumes are both reported.",
        "",
        "## Development-set organ statistics",
        "",
        "Test-set voxel counts and volumes are deliberately excluded.",
        "",
        "| Organ | Present / 1,139 | Exclusive median mL | P05 mL | P95 mL |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, organ in summary["organs"].items():
        development = organ["train_validation"]["development"]
        volume = development["present_only_exclusive_training_volume_ml"]
        lines.append(
            f"| `{name}` | {development['exclusive_present_cases']:,} | "
            f"{volume['median']:.3f} | {volume['p05']:.3f} | {volume['p95']:.3f} |"
        )
    units = summary["volume_units"]
    lines.extend(
        [
            "",
            "## Test lock",
            "",
            "All 89 test cases received integrity checks, but per-organ voxel counts, overlap counts, and volumes are redacted from the CSV and excluded from statistics.",
            "",
            "## Spatial-unit note",
            "",
            f"The NIfTI spatial-units flag is unset in {units['unknown_nifti_units_assumed_mm_cases']:,} cases. "
            "For volume calculations, affine units are explicitly assumed to be millimetres, consistent with the dataset-wide 1.5 mm spacing audit.",
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
