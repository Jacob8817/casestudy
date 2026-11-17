"""Reproducible manifest construction for the selected-organ dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .audit import inspect_nifti
from .label_mapping import OrganSchema


MANIFEST_VERSION = "1.0.0"
ALLOWED_SPLITS = ("train", "val", "test")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_official_splits(metadata_path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames or "image_id" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError("meta.csv must contain image_id and split columns")
        for row_number, row in enumerate(reader, start=2):
            case_id = (row.get("image_id") or "").strip()
            split = (row.get("split") or "").strip()
            if not case_id:
                raise ValueError(f"meta.csv row {row_number} has an empty image_id")
            if case_id in rows:
                raise ValueError(f"meta.csv contains duplicate image_id {case_id}")
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"meta.csv case {case_id} has unsupported split {split!r}")
            rows[case_id] = split
    return rows


def _relative_file(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes dataset root: {resolved}") from exc
    if ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    return relative.as_posix()


def build_manifest(
    dataset_root: Path,
    organ_schema: OrganSchema,
    organ_config_path: Path,
    *,
    workers: int = 8,
    hash_files: bool = True,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    metadata_path = dataset_root / "meta.csv"
    if not metadata_path.is_file():
        raise ValueError(f"Metadata file not found: {metadata_path}")
    official_splits = read_official_splits(metadata_path)

    disk_case_dirs = {
        item.name: item
        for item in dataset_root.iterdir()
        if item.is_dir() and ((item / "ct.nii.gz").is_file() or (item / "ct.nii").is_file())
    }
    metadata_ids = set(official_splits)
    disk_ids = set(disk_case_dirs)
    if metadata_ids != disk_ids:
        missing_on_disk = sorted(metadata_ids - disk_ids)
        missing_in_metadata = sorted(disk_ids - metadata_ids)
        raise ValueError(
            f"Metadata/disk case mismatch: missing_on_disk={missing_on_disk[:10]}, "
            f"missing_in_metadata={missing_in_metadata[:10]}"
        )

    file_paths: list[Path] = [metadata_path]
    structural_cases: list[dict[str, Any]] = []
    for case_id in sorted(metadata_ids):
        case_dir = disk_case_dirs[case_id]
        ct_candidates = [path for path in (case_dir / "ct.nii.gz", case_dir / "ct.nii") if path.is_file()]
        if len(ct_candidates) != 1:
            raise ValueError(f"{case_id}: expected exactly one CT, found {len(ct_candidates)}")
        ct_path = ct_candidates[0]
        file_paths.append(ct_path)
        targets: dict[str, Any] = {}
        for label in organ_schema.labels:
            sources = []
            for source_name in label.source_masks:
                candidates = [
                    path
                    for path in (
                        case_dir / "segmentations" / f"{source_name}.nii.gz",
                        case_dir / "segmentations" / f"{source_name}.nii",
                    )
                    if path.is_file()
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"{case_id}/{source_name}: expected exactly one source mask, found {len(candidates)}"
                    )
                source_path = candidates[0]
                file_paths.append(source_path)
                sources.append({"name": source_name, "path": _relative_file(source_path, dataset_root)})
            targets[label.name] = {"class_id": label.id, "sources": sources}
        structural_cases.append(
            {
                "case_id": case_id,
                "split": official_splits[case_id],
                "ct_path": ct_path,
                "targets": targets,
            }
        )

    unique_paths = sorted(set(file_paths), key=lambda path: str(path).lower())
    if hash_files:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            hashes = dict(zip(unique_paths, executor.map(sha256_file, unique_paths)))
    else:
        hashes = {}

    cases: list[dict[str, Any]] = []
    content_lines: list[str] = []
    for structural in structural_cases:
        ct_path = structural.pop("ct_path")
        ct_info = inspect_nifti(ct_path, "header")
        ct_hash = hashes.get(ct_path, "not_computed")
        targets = structural["targets"]
        for target in targets.values():
            for source in target["sources"]:
                source_path = dataset_root / source["path"]
                source["size_bytes"] = source_path.stat().st_size
                source["file_sha256"] = hashes.get(source_path, "not_computed")
        case = {
            "case_id": structural["case_id"],
            "split": structural["split"],
            "ct": {
                "path": _relative_file(ct_path, dataset_root),
                "size_bytes": ct_path.stat().st_size,
                "file_sha256": ct_hash,
                "shape": list(ct_info.shape),
                "spacing_mm": list(ct_info.voxel_spacing[:3]),
                "orientation": ct_info.orientation,
                "datatype": ct_info.datatype,
            },
            "targets": targets,
        }
        cases.append(case)
        content_lines.append(f"{case['case_id']}\t{case['split']}\t{case['ct']['path']}\t{ct_hash}")
        for target_name, target in sorted(targets.items()):
            for source in target["sources"]:
                content_lines.append(
                    f"{case['case_id']}\t{target_name}\t{source['path']}\t{source['file_sha256']}"
                )

    split_counts = Counter(case["split"] for case in cases)
    split_sets = {split: {case["case_id"] for case in cases if case["split"] == split} for split in ALLOWED_SPLITS}
    if any(split_sets[left] & split_sets[right] for left in ALLOWED_SPLITS for right in ALLOWED_SPLITS if left < right):
        raise ValueError("Official split sets are not disjoint")
    if set().union(*split_sets.values()) != metadata_ids:
        raise ValueError("Official split union does not cover all cases")

    if hash_files:
        ct_hash_splits: dict[str, set[str]] = {}
        for case in cases:
            ct_hash_splits.setdefault(case["ct"]["file_sha256"], set()).add(case["split"])
        cross_split_duplicates = [digest for digest, splits in ct_hash_splits.items() if len(splits) > 1]
        if cross_split_duplicates:
            raise ValueError(f"Found {len(cross_split_duplicates)} exact compressed CT duplicates across splits")

    labels = [{"id": 0, "name": "background", "source_masks": []}] + [
        {"id": label.id, "name": label.name, "source_masks": list(label.source_masks)}
        for label in organ_schema.labels
    ]
    assignment_lines = [f"{case['case_id']}\t{case['split']}" for case in cases]
    test_cases = [case for case in cases if case["split"] == "test"]
    test_content_lines = [line for line in content_lines if line.split("\t", 1)[0] in split_sets["test"]]
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "manifest_kind": "selected_organ_dataset",
        "manifest_sha256": "",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"name": "scripts/create_manifest.py", "version": MANIFEST_VERSION},
        "dataset": {
            "id": "totalsegmentator_ct_v2.0.1",
            "release": "2.0.1",
            "modality": "CT",
            "path_anchor": "dataset_root",
            "case_count": len(cases),
            "patient_identity_basis": "meta.csv image_id",
            "patient_linkage_verified": False,
            "patient_linkage_limitation": "No independent stable patient UID is available in meta.csv.",
            "content_sha256": sha256_lines(sorted(content_lines)) if hash_files else "not_computed",
        },
        "source_metadata": {
            "path": "meta.csv",
            "delimiter": ";",
            "row_count": len(official_splits),
            "file_sha256": hashes.get(metadata_path, sha256_file(metadata_path)),
        },
        "organ_schema": {
            "name": organ_schema.name,
            "project_path": organ_config_path.as_posix(),
            "canonical_sha256": organ_schema.fingerprint(),
            "foreground_classes": len(organ_schema.labels),
            "source_mask_count_per_case": len(organ_schema.source_masks),
            "overlap_policy": organ_schema.overlap_policy,
            "overlap_method": organ_schema.overlap_method,
            "overlap_priority_low_to_high": list(organ_schema.overlap_priority),
            "overlap_reference": organ_schema.overlap_reference,
        },
        "split_policy": {
            "strategy": "official_meta_csv",
            "source_column": "split",
            "seed": None,
            "counts": {split: split_counts[split] for split in ALLOWED_SPLITS},
            "assignment_sha256": sha256_lines(assignment_lines),
            "test_lock": {
                "locked": True,
                "case_count": len(test_cases),
                "case_ids_sha256": sha256_lines(sorted(split_sets["test"])),
                "selected_content_sha256": sha256_lines(sorted(test_content_lines)) if hash_files else "not_computed",
                "per_case_target_metrics_redacted": True,
            },
        },
        "leakage_checks": {
            "split_sets_disjoint": True,
            "split_union_complete": True,
            "exact_compressed_ct_duplicates_across_splits": 0 if hash_files else "not_checked",
            "near_duplicate_ct_screening": "not_performed",
        },
        "labels": labels,
        "cases": cases,
    }
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("manifest_sha256", None)
    fingerprint_payload.pop("generated_at_utc", None)
    manifest["manifest_sha256"] = canonical_sha256(fingerprint_payload)
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any], *, overwrite: bool = False) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {path}; pass --overwrite to replace it deliberately")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
