#!/usr/bin/env python3
"""Audit an extracted TotalSegmentator-style NIfTI dataset.

The audit validates NIfTI headers, geometry, mask completeness, and metadata.
By default it also streams every CT payload to calculate scaled Hounsfield-unit
statistics without loading a whole volume into memory.  Streaming to EOF gives
every gzip-compressed CT a complete decompression/CRC check.  Use
``--integrity-mode full`` only when complete mask decompression/CRC validation
is required as well; the 117 masks per case expand to terabytes in aggregate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import itertools
import math
import os
import struct
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


NIFTI_SUFFIXES = (".nii", ".nii.gz")
AUDIT_VERSION = "1.0.0"
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "audit_outputs",
    "dataset_audit",
    "outputs",
}
DATATYPE_BITPIX = {
    1: 1,       # binary
    2: 8,       # uint8
    4: 16,      # int16
    8: 32,      # int32
    16: 32,     # float32
    32: 64,     # complex64
    64: 64,     # float64
    128: 24,    # RGB24
    256: 8,     # int8
    512: 16,    # uint16
    768: 32,    # uint32
    1024: 64,   # int64
    1280: 64,   # uint64
    1536: 128,  # float128
    1792: 128,  # complex128
    2048: 256,  # complex256
    2304: 32,   # RGBA32
}
INTEGER_DATATYPES = {1, 2, 4, 8, 256, 512, 768, 1024, 1280}
NUMPY_DTYPES = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
    1024: "i8",
    1280: "u8",
}


@dataclass
class Issue:
    severity: str
    category: str
    subject_id: str = ""
    label: str = ""
    path: str = ""
    detail: str = ""


@dataclass
class NiftiInfo:
    path: str
    file_size_bytes: int
    compressed: bool
    shape: tuple[int, ...]
    voxel_spacing: tuple[float, ...]
    datatype: int
    bitpix: int
    endian: str
    vox_offset: float
    scl_slope: float
    scl_inter: float
    qform_code: int
    sform_code: int
    transform: tuple[float, ...]
    orientation: str
    spatial_units: str
    gzip_isize: int | None = None
    expected_uncompressed_bytes: int | None = None
    full_integrity_checked: bool = False


@dataclass
class IntensityStats:
    voxel_count: int
    finite_voxel_count: int
    nonfinite_voxel_count: int
    min_hu: float
    max_hu: float
    mean_hu: float
    std_hu: float
    payload_crc_checked: bool


@dataclass
class SubjectResult:
    subject_id: str
    subject_path: str
    ct_path: str = ""
    ct_info: NiftiInfo | None = None
    ct_intensity: IntensityStats | None = None
    mask_count: int = 0
    mask_labels: list[str] = field(default_factory=list)
    mask_bytes: int = 0
    mask_uncompressed_bytes: int = 0
    issues: list[Issue] = field(default_factory=list)


def is_nifti_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(NIFTI_SUFFIXES)


def label_from_path(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".nii.gz"):
        return path.name[:-7]
    if lower.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def discover_subject_dirs(search_root: Path) -> tuple[Path, list[Path]]:
    """Find case directories without assuming the extraction directory name."""
    candidates: set[Path] = set()
    for current, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in IGNORED_DIRECTORY_NAMES and not name.startswith(".")
        ]
        current_path = Path(current)
        lower_files = {name.lower() for name in filenames}
        has_ct = "ct.nii.gz" in lower_files or "ct.nii" in lower_files
        has_seg_dir = any(name.lower() in {"segmentations", "segmentation", "masks", "labels"} for name in dirnames)
        if has_ct or has_seg_dir:
            candidates.add(current_path.resolve())
            # A case may have thousands of files below it; no nested case is
            # expected inside a subject directory.
            dirnames[:] = []

    if not candidates:
        raise RuntimeError(f"No subject directories containing CT files or segmentation folders found under {search_root}")

    ordered = sorted(candidates, key=lambda item: str(item).lower())
    common = Path(os.path.commonpath([str(item) for item in ordered]))
    if common in candidates and len(candidates) == 1:
        common = common.parent
    return common, ordered


def find_ct_files(subject_dir: Path) -> list[Path]:
    return sorted(
        [
            item
            for item in subject_dir.iterdir()
            if item.is_file() and item.name.lower() in {"ct.nii", "ct.nii.gz"}
        ],
        key=lambda item: item.name.lower(),
    )


def find_mask_files(subject_dir: Path) -> tuple[list[Path], bool]:
    conventional = [
        item
        for item in subject_dir.iterdir()
        if item.is_dir() and item.name.lower() in {"segmentations", "segmentation", "masks", "labels"}
    ]
    if conventional:
        masks: list[Path] = []
        for directory in conventional:
            masks.extend(item for item in directory.iterdir() if item.is_file() and is_nifti_name(item.name))
        return sorted(masks, key=lambda item: str(item).lower()), True

    # Fallback for a nonstandard extraction hierarchy.
    masks = []
    for current, dirnames, filenames in os.walk(subject_dir):
        if Path(current) == subject_dir:
            continue
        for filename in filenames:
            if is_nifti_name(filename):
                masks.append(Path(current) / filename)
    return sorted(masks, key=lambda item: str(item).lower()), False


def _unpack(fmt: str, data: bytes, offset: int):
    return struct.unpack_from(fmt, data, offset)


def _safe_product(values: Sequence[int]) -> int:
    product = 1
    for value in values:
        product *= value
        if product > 2**63:
            raise ValueError("voxel count is implausibly large")
    return product


def _qform_affine(
    header: bytes,
    endian: str,
    pixdim: Sequence[float],
) -> tuple[float, ...]:
    """Convert a NIfTI-1 quaternion transform into three affine rows."""
    b, c, d = (float(value) for value in _unpack(endian + "3f", header, 256))
    qx, qy, qz = (float(value) for value in _unpack(endian + "3f", header, 268))
    squared = b * b + c * c + d * d
    if squared > 1.0 + 1e-5:
        scale = 1.0 / math.sqrt(squared)
        b, c, d = b * scale, c * scale, d * scale
        a = 0.0
    else:
        a = math.sqrt(max(0.0, 1.0 - squared))

    rotation = (
        (a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c),
        (2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b),
        (2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b),
    )
    dx, dy, dz = (abs(float(value)) for value in pixdim[1:4])
    qfac = -1.0 if float(pixdim[0]) < 0 else 1.0
    scales = (dx, dy, dz * qfac)
    offsets = (qx, qy, qz)
    rows: list[float] = []
    for row_index in range(3):
        rows.extend(rotation[row_index][column] * scales[column] for column in range(3))
        rows.append(offsets[row_index])
    return tuple(rows)


def orientation_from_affine(transform: Sequence[float]) -> str:
    """Return three closest NIfTI/RAS axis codes for a 3x4 affine."""
    if len(transform) != 12 or any(not math.isfinite(value) for value in transform):
        return "UNKNOWN"
    matrix = [[float(transform[row * 4 + column]) for column in range(3)] for row in range(3)]
    for column in range(3):
        norm = math.sqrt(sum(matrix[row][column] ** 2 for row in range(3)))
        if norm <= 0 or not math.isfinite(norm):
            return "UNKNOWN"
        for row in range(3):
            matrix[row][column] /= norm

    # Choose the one-to-one voxel-to-world axis assignment with the largest
    # absolute alignment.  This is stable for the mildly oblique CT affines in
    # TotalSegmentator and avoids requiring nibabel just for axis labels.
    assignment = max(
        itertools.permutations(range(3)),
        key=lambda axes: sum(abs(matrix[axes[column]][column]) for column in range(3)),
    )
    positive = ("R", "A", "S")
    negative = ("L", "P", "I")
    return "".join(
        positive[world_axis] if matrix[world_axis][column] >= 0 else negative[world_axis]
        for column, world_axis in enumerate(assignment)
    )


def compute_ct_intensity_stats(path: Path, info: NiftiInfo, chunk_bytes: int = 16 * 1024 * 1024) -> IntensityStats:
    """Stream a 3D CT payload and calculate scaled HU statistics.

    The function reads through gzip EOF, which also verifies the stored CRC.
    It never materializes the full CT volume in RAM.
    """
    if len(info.shape) != 3:
        raise ValueError(f"CT intensity statistics require a 3D image, found shape={info.shape}")
    dtype_code = NUMPY_DTYPES.get(info.datatype)
    if dtype_code is None:
        raise ValueError(f"CT datatype {info.datatype} is not supported for streaming intensity statistics")
    if not math.isfinite(info.vox_offset) or not math.isclose(info.vox_offset, round(info.vox_offset), abs_tol=1e-4):
        raise ValueError(f"vox_offset must be finite and integer-valued, found {info.vox_offset}")

    dtype = np.dtype(info.endian + dtype_code)
    voxel_count = _safe_product(info.shape)
    itemsize = dtype.itemsize
    aligned_chunk_bytes = max(itemsize, (chunk_bytes // itemsize) * itemsize)
    remaining_bytes = voxel_count * itemsize
    slope = info.scl_slope if math.isfinite(info.scl_slope) and info.scl_slope != 0 else 1.0
    intercept = info.scl_inter if math.isfinite(info.scl_inter) and info.scl_slope != 0 else 0.0

    finite_count = 0
    nonfinite_count = 0
    running_mean = 0.0
    running_m2 = 0.0
    minimum = math.inf
    maximum = -math.inf
    opener = gzip.open if info.compressed else open
    try:
        with opener(path, "rb") as stream:
            stream.seek(int(round(info.vox_offset)))
            while remaining_bytes:
                requested = min(aligned_chunk_bytes, remaining_bytes)
                payload = stream.read(requested)
                if len(payload) != requested:
                    raise ValueError(
                        f"CT payload is truncated: requested {requested} bytes with {remaining_bytes} remaining, received {len(payload)}"
                    )
                remaining_bytes -= len(payload)
                raw = np.frombuffer(payload, dtype=dtype)
                values = raw.astype(np.float64)
                if slope != 1.0:
                    values *= slope
                if intercept != 0.0:
                    values += intercept
                finite = np.isfinite(values)
                chunk_nonfinite = int(values.size - np.count_nonzero(finite))
                nonfinite_count += chunk_nonfinite
                if chunk_nonfinite:
                    values = values[finite]
                if not values.size:
                    continue

                chunk_count = int(values.size)
                chunk_mean = float(np.mean(values, dtype=np.float64))
                chunk_m2 = float(np.sum((values - chunk_mean) ** 2, dtype=np.float64))
                delta = chunk_mean - running_mean
                combined = finite_count + chunk_count
                running_mean += delta * chunk_count / combined
                running_m2 += chunk_m2 + delta * delta * finite_count * chunk_count / combined
                finite_count = combined
                minimum = min(minimum, float(np.min(values)))
                maximum = max(maximum, float(np.max(values)))

            # Read through EOF so gzip verifies CRC.  Normally there are no
            # trailing bytes after the declared NIfTI voxel payload.
            while stream.read(aligned_chunk_bytes):
                pass
    except (OSError, EOFError) as exc:
        raise ValueError(f"CT payload decompression/CRC check failed: {exc}") from exc

    if finite_count == 0:
        raise ValueError("CT contains no finite intensity voxels")
    variance = running_m2 / finite_count
    return IntensityStats(
        voxel_count=voxel_count,
        finite_voxel_count=finite_count,
        nonfinite_voxel_count=nonfinite_count,
        min_hu=minimum,
        max_hu=maximum,
        mean_hu=running_mean,
        std_hu=math.sqrt(max(0.0, variance)),
        payload_crc_checked=info.compressed,
    )


def inspect_nifti(path: Path, integrity_mode: str) -> NiftiInfo:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("file is empty")

    compressed = path.name.lower().endswith(".gz")
    gzip_isize: int | None = None
    if compressed:
        if size < 18:
            raise ValueError("gzip file is too small to contain a valid NIfTI")
        with path.open("rb") as raw:
            if raw.read(2) != b"\x1f\x8b":
                raise ValueError(".gz file does not have gzip magic bytes")
            raw.seek(-8, os.SEEK_END)
            _crc32, gzip_isize = struct.unpack("<II", raw.read(8))
        try:
            with gzip.open(path, "rb") as stream:
                header = stream.read(352)
        except (OSError, EOFError) as exc:
            raise ValueError(f"cannot read gzip/NIfTI header: {exc}") from exc
    else:
        with path.open("rb") as stream:
            header = stream.read(352)

    if len(header) < 348:
        raise ValueError(f"NIfTI header is truncated ({len(header)} bytes available)")

    little = struct.unpack_from("<i", header, 0)[0]
    big = struct.unpack_from(">i", header, 0)[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise ValueError(f"invalid sizeof_hdr (little={little}, big={big}; expected 348)")

    magic = header[344:348]
    if magic not in {b"n+1\x00", b"ni1\x00"}:
        raise ValueError(f"invalid NIfTI-1 magic {magic!r}")
    if magic == b"ni1\x00":
        raise ValueError("two-file NIfTI (.hdr/.img) is not supported by this dataset audit")

    dimensions = _unpack(endian + "8h", header, 40)
    ndim = dimensions[0]
    if ndim != 3:
        raise ValueError(f"expected a 3D CT/mask NIfTI, found ndim={ndim}")
    shape = tuple(int(value) for value in dimensions[1 : ndim + 1])
    if any(value <= 0 for value in shape):
        raise ValueError(f"invalid shape: {shape}")

    datatype = _unpack(endian + "h", header, 70)[0]
    bitpix = _unpack(endian + "h", header, 72)[0]
    expected_bitpix = DATATYPE_BITPIX.get(datatype)
    if expected_bitpix is None:
        raise ValueError(f"unsupported/invalid datatype code: {datatype}")
    if bitpix != expected_bitpix:
        raise ValueError(f"datatype {datatype} requires bitpix={expected_bitpix}, found {bitpix}")

    pixdim = _unpack(endian + "8f", header, 76)
    spacing = tuple(abs(float(value)) for value in pixdim[1 : ndim + 1])
    if any(not math.isfinite(value) or value <= 0 for value in spacing[: min(3, len(spacing))]):
        raise ValueError(f"invalid voxel spacing: {spacing}")

    vox_offset = float(_unpack(endian + "f", header, 108)[0])
    scl_slope = float(_unpack(endian + "f", header, 112)[0])
    scl_inter = float(_unpack(endian + "f", header, 116)[0])
    spatial_units_code = header[123] & 0x07
    spatial_units = {1: "meter", 2: "mm", 3: "micron"}.get(spatial_units_code, "unknown")
    qform_code = int(_unpack(endian + "h", header, 252)[0])
    sform_code = int(_unpack(endian + "h", header, 254)[0])
    if sform_code > 0:
        transform = tuple(float(value) for value in _unpack(endian + "12f", header, 280))
    elif qform_code > 0:
        transform = _qform_affine(header, endian, pixdim)
    else:
        transform = ()
    if transform:
        if any(not math.isfinite(value) for value in transform):
            raise ValueError("spatial affine contains non-finite values")
        a, b, c, _x, d, e, f, _y, g, h, i, _z = transform
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if abs(determinant) < 1e-8:
            raise ValueError("spatial affine is singular")
    orientation = orientation_from_affine(transform)

    voxel_count = _safe_product(shape)
    expected_bytes = int(round(vox_offset)) + math.ceil(voxel_count * bitpix / 8)
    if not math.isfinite(vox_offset) or not math.isclose(vox_offset, round(vox_offset), abs_tol=1e-4):
        raise ValueError(f"invalid non-integer vox_offset: {vox_offset}")
    if magic == b"n+1\x00" and vox_offset < 352:
        raise ValueError(f"invalid single-file vox_offset: {vox_offset}")
    if not compressed and magic == b"n+1\x00" and size < expected_bytes:
        raise ValueError(f"file is truncated: {size} bytes, expected at least {expected_bytes}")
    if compressed and gzip_isize is not None and expected_bytes < 2**32:
        if gzip_isize < expected_bytes:
            raise ValueError(
                f"gzip uncompressed-size trailer indicates truncation: {gzip_isize} bytes, expected at least {expected_bytes}"
            )

    if integrity_mode == "full" and compressed:
        total = 0
        try:
            with gzip.open(path, "rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    total += len(chunk)
        except (OSError, EOFError) as exc:
            raise ValueError(f"full gzip CRC/decompression check failed: {exc}") from exc
        if total < expected_bytes:
            raise ValueError(f"decompressed stream is truncated: {total} bytes, expected at least {expected_bytes}")

    return NiftiInfo(
        path=str(path.resolve()),
        file_size_bytes=size,
        compressed=compressed,
        shape=shape,
        voxel_spacing=spacing,
        datatype=datatype,
        bitpix=bitpix,
        endian=endian,
        vox_offset=vox_offset,
        scl_slope=scl_slope,
        scl_inter=scl_inter,
        qform_code=qform_code,
        sform_code=sform_code,
        transform=transform,
        orientation=orientation,
        spatial_units=spatial_units,
        gzip_isize=gzip_isize,
        expected_uncompressed_bytes=expected_bytes,
        full_integrity_checked=integrity_mode == "full",
    )


def close_enough(left: Sequence[float], right: Sequence[float], tolerance: float) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance) for a, b in zip(left, right)
    )


def audit_subject(
    subject_dir: Path,
    dataset_root: Path,
    integrity_mode: str,
    calculate_intensity_stats: bool,
) -> SubjectResult:
    try:
        subject_id = str(subject_dir.relative_to(dataset_root)).replace("\\", "/")
    except ValueError:
        subject_id = subject_dir.name
    result = SubjectResult(subject_id=subject_id, subject_path=str(subject_dir.resolve()))

    ct_files = find_ct_files(subject_dir)
    if not ct_files:
        result.issues.append(Issue("error", "missing_ct", subject_id, path=str(subject_dir), detail="No ct.nii or ct.nii.gz found"))
    elif len(ct_files) > 1:
        result.issues.append(Issue("error", "duplicate_ct", subject_id, path=str(subject_dir), detail=f"Found {len(ct_files)} CT files"))

    if ct_files:
        result.ct_path = str(ct_files[0].resolve())
        try:
            # Reading intensities below already performs full CT payload/CRC
            # validation, so avoid decompressing each CT twice in full mode.
            ct_integrity_mode = "header" if calculate_intensity_stats else integrity_mode
            result.ct_info = inspect_nifti(ct_files[0], ct_integrity_mode)
        except (OSError, ValueError, struct.error) as exc:
            result.issues.append(Issue("error", "invalid_ct", subject_id, path=str(ct_files[0]), detail=str(exc)))
        if result.ct_info is not None:
            if result.ct_info.orientation == "UNKNOWN":
                result.issues.append(Issue("warning", "ct_orientation_unknown", subject_id, path=str(ct_files[0]), detail="No usable qform/sform affine"))
            if calculate_intensity_stats:
                try:
                    result.ct_intensity = compute_ct_intensity_stats(ct_files[0], result.ct_info)
                except (OSError, ValueError, struct.error) as exc:
                    result.issues.append(Issue("error", "invalid_ct_payload", subject_id, path=str(ct_files[0]), detail=str(exc)))
                if result.ct_intensity and result.ct_intensity.nonfinite_voxel_count:
                    result.issues.append(
                        Issue(
                            "warning",
                            "nonfinite_ct_intensity",
                            subject_id,
                            path=str(ct_files[0]),
                            detail=f"Found {result.ct_intensity.nonfinite_voxel_count} non-finite voxels",
                        )
                    )

    try:
        masks, conventional_dir_found = find_mask_files(subject_dir)
    except OSError as exc:
        masks, conventional_dir_found = [], False
        result.issues.append(Issue("error", "unreadable_segmentation_directory", subject_id, path=str(subject_dir), detail=str(exc)))

    if not conventional_dir_found:
        result.issues.append(Issue("error", "missing_segmentation_directory", subject_id, path=str(subject_dir), detail="No conventional segmentation directory found"))
    if not masks:
        result.issues.append(Issue("error", "missing_masks", subject_id, path=str(subject_dir), detail="No NIfTI masks found"))

    label_paths: dict[str, list[Path]] = defaultdict(list)
    for mask_path in masks:
        label_paths[label_from_path(mask_path)].append(mask_path)
    for label, paths in label_paths.items():
        if len(paths) > 1:
            result.issues.append(Issue("error", "duplicate_mask_label", subject_id, label, str(subject_dir), f"Found {len(paths)} files for label"))

    result.mask_count = len(masks)
    result.mask_labels = sorted(label_paths)
    for label, paths in sorted(label_paths.items()):
        mask_path = paths[0]
        try:
            result.mask_bytes += mask_path.stat().st_size
            info = inspect_nifti(mask_path, integrity_mode)
            result.mask_uncompressed_bytes += info.expected_uncompressed_bytes or 0
        except (OSError, ValueError, struct.error) as exc:
            result.issues.append(Issue("error", "invalid_mask", subject_id, label, str(mask_path), str(exc)))
            continue

        if info.datatype not in INTEGER_DATATYPES:
            result.issues.append(Issue("warning", "non_integer_mask_datatype", subject_id, label, str(mask_path), f"datatype={info.datatype}"))
        if result.ct_info is None:
            continue
        ct_info = result.ct_info
        if info.shape != ct_info.shape:
            result.issues.append(Issue("error", "mask_shape_mismatch", subject_id, label, str(mask_path), f"mask={info.shape}; ct={ct_info.shape}"))
        if not close_enough(info.voxel_spacing[:3], ct_info.voxel_spacing[:3], 1e-4):
            result.issues.append(Issue("error", "mask_spacing_mismatch", subject_id, label, str(mask_path), f"mask={info.voxel_spacing[:3]}; ct={ct_info.voxel_spacing[:3]}"))
        # TotalSegmentator masks can contain tiny affine rounding differences
        # (roughly 1e-4 in direction-cosine elements).  A 1e-3 tolerance keeps
        # these harmless serialization differences out of the error report.
        if info.transform and ct_info.transform and not close_enough(info.transform, ct_info.transform, 1e-3):
            result.issues.append(Issue("error", "mask_transform_mismatch", subject_id, label, str(mask_path), "Mask and CT spatial transforms differ"))

    return result


def read_metadata(dataset_root: Path) -> tuple[Path | None, dict[str, dict[str, str]], list[Issue]]:
    candidates = sorted(dataset_root.glob("*.csv"))
    meta_path = next((item for item in candidates if item.name.lower() == "meta.csv"), None)
    if meta_path is None:
        return None, {}, [Issue("warning", "missing_metadata", path=str(dataset_root), detail="meta.csv not found")]
    rows: dict[str, dict[str, str]] = {}
    issues: list[Issue] = []
    try:
        with meta_path.open("r", encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            delimiter = ";" if sample.count(";") > sample.count(",") else ","
            reader = csv.DictReader(stream, delimiter=delimiter)
            if not reader.fieldnames or "image_id" not in reader.fieldnames:
                issues.append(Issue("error", "invalid_metadata", path=str(meta_path), detail="Missing image_id column"))
                return meta_path, rows, issues
            for row in reader:
                image_id = (row.get("image_id") or "").strip()
                if not image_id:
                    issues.append(Issue("warning", "metadata_missing_id", path=str(meta_path), detail="Row has empty image_id"))
                elif image_id in rows:
                    issues.append(Issue("error", "duplicate_metadata_id", image_id, path=str(meta_path), detail="Duplicate image_id"))
                else:
                    rows[image_id] = dict(row)
    except (OSError, csv.Error, UnicodeError) as exc:
        issues.append(Issue("error", "invalid_metadata", path=str(meta_path), detail=str(exc)))
    return meta_path, rows, issues


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        "--search-root",
        dest="search_root",
        type=Path,
        default=Path.cwd(),
        help="Extracted dataset directory, or a parent below which it will be discovered",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated reports")
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)), help="Number of subjects audited concurrently")
    parser.add_argument(
        "--integrity-mode",
        choices=("header", "full"),
        default="header",
        help="Mask header/trailer checks, or full mask decompression and CRC validation",
    )
    parser.add_argument(
        "--skip-intensity-stats",
        action="store_true",
        help="Skip CT payload reading/HU statistics (CT payload CRC will then only be checked in full mode)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    started = time.monotonic()
    generated_at = datetime.now(timezone.utc).isoformat()
    search_root = args.search_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering dataset below: {search_root}", flush=True)
    try:
        dataset_root, subject_dirs = discover_subject_dirs(search_root)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Dataset root: {dataset_root}", flush=True)
    print(f"Candidate subjects: {len(subject_dirs)}", flush=True)

    metadata_path, metadata, metadata_issues = read_metadata(dataset_root)
    calculate_intensity_stats = not args.skip_intensity_stats
    if calculate_intensity_stats:
        print("CT intensity statistics: enabled (streaming payload + CRC validation)", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda item: audit_subject(item, dataset_root, args.integrity_mode, calculate_intensity_stats),
                subject_dirs,
            )
        )

    all_issues = list(metadata_issues)
    for result in results:
        all_issues.extend(result.issues)

    subject_ids_by_basename: dict[str, str] = {}
    for result in results:
        basename = Path(result.subject_path).name
        subject_ids_by_basename[basename] = result.subject_id
        if metadata and basename not in metadata:
            all_issues.append(Issue("warning", "subject_missing_from_metadata", result.subject_id, path=result.subject_path, detail="No matching meta.csv row"))
    for image_id in sorted(set(metadata) - set(subject_ids_by_basename)):
        all_issues.append(Issue("warning", "metadata_subject_missing_on_disk", image_id, path=str(metadata_path or ""), detail="Metadata row has no subject directory"))

    label_frequency: Counter[str] = Counter()
    for result in results:
        label_frequency.update(result.mask_labels)
    expected_labels = set(label_frequency)
    for result in results:
        missing = sorted(expected_labels - set(result.mask_labels))
        if missing:
            all_issues.append(Issue("error", "missing_expected_masks", result.subject_id, path=result.subject_path, detail=";".join(missing)))

    issue_counts = Counter(issue.category for issue in all_issues)
    severity_counts = Counter(issue.severity for issue in all_issues)
    valid_ct_count = sum(result.ct_info is not None for result in results)
    ct_count = sum(bool(result.ct_path) for result in results)
    mask_count = sum(result.mask_count for result in results)
    total_ct_bytes = sum(result.ct_info.file_size_bytes for result in results if result.ct_info)
    total_mask_bytes = sum(result.mask_bytes for result in results)
    total_mask_uncompressed_bytes = sum(result.mask_uncompressed_bytes for result in results)
    elapsed = time.monotonic() - started

    ct_shapes = Counter("x".join(map(str, result.ct_info.shape)) for result in results if result.ct_info)
    ct_datatypes = Counter(str(result.ct_info.datatype) for result in results if result.ct_info)
    ct_orientations = Counter(result.ct_info.orientation for result in results if result.ct_info)
    mask_count_distribution = Counter(str(result.mask_count) for result in results)
    spacings = [result.ct_info.voxel_spacing[:3] for result in results if result.ct_info and len(result.ct_info.voxel_spacing) >= 3]
    spacing_ranges = {
        axis: {
            "min_mm": min(values),
            "max_mm": max(values),
        }
        for axis, values in zip(
            ("x", "y", "z"),
            ([spacing[index] for spacing in spacings] for index in range(3)),
        )
    } if spacings else {}
    metadata_columns = list(next(iter(metadata.values())).keys()) if metadata else []
    metadata_missing_values = {
        column: sum(not (row.get(column) or "").strip() for row in metadata.values())
        for column in metadata_columns
    }
    metadata_distributions = {
        column: dict(sorted(Counter((row.get(column) or "<missing>").strip() or "<missing>" for row in metadata.values()).items()))
        for column in ("split", "gender", "institute", "study_type")
        if column in metadata_columns
    }
    intensity_results = [result.ct_intensity for result in results if result.ct_intensity is not None]
    total_finite_voxels = sum(item.finite_voxel_count for item in intensity_results)
    if intensity_results and total_finite_voxels:
        voxel_weighted_mean_hu = math.fsum(
            item.mean_hu * item.finite_voxel_count for item in intensity_results
        ) / total_finite_voxels
        voxel_weighted_variance_hu = math.fsum(
            item.finite_voxel_count
            * (item.std_hu * item.std_hu + (item.mean_hu - voxel_weighted_mean_hu) ** 2)
            for item in intensity_results
        ) / total_finite_voxels
        intensity_summary = {
            "cases_with_statistics": len(intensity_results),
            "payload_crc_checked_cases": sum(item.payload_crc_checked for item in intensity_results),
            "total_finite_voxels": total_finite_voxels,
            "total_nonfinite_voxels": sum(item.nonfinite_voxel_count for item in intensity_results),
            "global_min_hu": min(item.min_hu for item in intensity_results),
            "global_max_hu": max(item.max_hu for item in intensity_results),
            "voxel_weighted_mean_hu": voxel_weighted_mean_hu,
            "voxel_weighted_std_hu": math.sqrt(max(0.0, voxel_weighted_variance_hu)),
            "case_mean_hu_min": min(item.mean_hu for item in intensity_results),
            "case_mean_hu_max": max(item.mean_hu for item in intensity_results),
        }
    else:
        intensity_summary = {
            "cases_with_statistics": 0,
            "payload_crc_checked_cases": 0,
            "total_finite_voxels": 0,
            "total_nonfinite_voxels": 0,
        }

    subject_rows = []
    issues_by_subject = Counter(issue.subject_id for issue in all_issues if issue.subject_id)
    for result in results:
        info = result.ct_info
        intensity = result.ct_intensity
        metadata_row = metadata.get(Path(result.subject_path).name, {})
        shape = list(info.shape[:3]) if info else []
        spacing = list(info.voxel_spacing[:3]) if info else []
        missing_masks = sorted(expected_labels - set(result.mask_labels))
        subject_rows.append(
            {
                "case_id": result.subject_id,
                "subject_id": result.subject_id,
                "subject_path": result.subject_path,
                "ct_path": result.ct_path,
                "ct_header_valid": info is not None,
                "ct_payload_valid": intensity is not None if calculate_intensity_stats else "not_checked",
                "ct_shape": "x".join(map(str, info.shape)) if info else "",
                "ct_spacing_mm": "x".join(f"{value:.6g}" for value in info.voxel_spacing[:3]) if info else "",
                "shape_x": shape[0] if len(shape) > 0 else "",
                "shape_y": shape[1] if len(shape) > 1 else "",
                "shape_z": shape[2] if len(shape) > 2 else "",
                "spacing_x": spacing[0] if len(spacing) > 0 else "",
                "spacing_y": spacing[1] if len(spacing) > 1 else "",
                "spacing_z": spacing[2] if len(spacing) > 2 else "",
                "orientation": info.orientation if info else "",
                "ct_datatype": info.datatype if info else "",
                "ct_size_bytes": info.file_size_bytes if info else "",
                "scl_slope": info.scl_slope if info else "",
                "scl_inter": info.scl_inter if info else "",
                "voxel_count": intensity.voxel_count if intensity else (_safe_product(info.shape) if info else ""),
                "min_hu": intensity.min_hu if intensity else "",
                "max_hu": intensity.max_hu if intensity else "",
                "mean_hu": intensity.mean_hu if intensity else "",
                "std_hu": intensity.std_hu if intensity else "",
                "nonfinite_voxels": intensity.nonfinite_voxel_count if intensity else "",
                "ct_payload_crc_checked": intensity.payload_crc_checked if intensity else False,
                "num_masks": result.mask_count,
                "mask_count": result.mask_count,
                "mask_labels": ";".join(result.mask_labels),
                "missing_masks": ";".join(missing_masks),
                "mask_size_bytes": result.mask_bytes,
                "mask_uncompressed_bytes": result.mask_uncompressed_bytes,
                "issue_count": issues_by_subject[result.subject_id],
                "metadata_present": bool(metadata_row),
                "split": metadata_row.get("split", ""),
                "study_type": metadata_row.get("study_type", ""),
            }
        )

    label_rows = [
        {
            "label": label,
            "subjects_present": count,
            "subjects_missing": len(results) - count,
            "coverage_percent": round(100 * count / len(results), 4) if results else 0,
        }
        for label, count in sorted(label_frequency.items())
    ]
    issue_rows = [asdict(issue) for issue in sorted(all_issues, key=lambda item: (item.severity, item.category, item.subject_id, item.label, item.path))]

    write_csv(
        output_dir / "dataset_audit.csv",
        subject_rows,
        [
            "case_id", "subject_id", "subject_path", "ct_path", "ct_header_valid", "ct_payload_valid",
            "shape_x", "shape_y", "shape_z", "ct_shape", "spacing_x", "spacing_y", "spacing_z",
            "ct_spacing_mm", "orientation", "ct_datatype", "scl_slope", "scl_inter", "voxel_count",
            "min_hu", "max_hu", "mean_hu", "std_hu", "nonfinite_voxels", "ct_payload_crc_checked",
            "num_masks", "mask_count", "mask_labels", "missing_masks", "ct_size_bytes", "mask_size_bytes",
            "mask_uncompressed_bytes",
            "issue_count", "metadata_present", "split", "study_type",
        ],
    )
    write_csv(output_dir / "mask_inventory.csv", label_rows, ["label", "subjects_present", "subjects_missing", "coverage_percent"])
    write_csv(output_dir / "issues.csv", issue_rows, ["severity", "category", "subject_id", "label", "path", "detail"])

    summary = {
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": generated_at,
        "search_root": str(search_root),
        "dataset_root": str(dataset_root),
        "metadata_file": str(metadata_path.resolve()) if metadata_path else None,
        "integrity_mode": args.integrity_mode,
        "intensity_statistics_enabled": calculate_intensity_stats,
        "integrity_scope": {
            "ct": (
                "Every CT payload streamed to EOF with gzip CRC validation while calculating scaled intensity statistics"
                if calculate_intensity_stats
                else (
                    "Every CT payload fully decompressed with gzip CRC validation"
                    if args.integrity_mode == "full"
                    else "NIfTI header, gzip signature/trailer, and advertised uncompressed size only"
                )
            ),
            "masks": (
                "Every mask payload fully decompressed with gzip CRC validation"
                if args.integrity_mode == "full"
                else "Every mask NIfTI header, gzip signature/trailer, advertised uncompressed size, datatype, and CT-grid geometry; payload CRC not checked"
            ),
        },
        "subject_count": len(results),
        "ct_file_count": ct_count,
        "valid_ct_header_count": valid_ct_count,
        "valid_ct_payload_count": len(intensity_results) if calculate_intensity_stats else None,
        "mask_file_count": mask_count,
        "unique_mask_labels": len(label_frequency),
        "metadata_row_count": len(metadata),
        "metadata_columns": metadata_columns,
        "metadata_missing_values": metadata_missing_values,
        "metadata_distributions": metadata_distributions,
        "ct_compressed_bytes": total_ct_bytes,
        "mask_compressed_bytes": total_mask_bytes,
        "mask_advertised_uncompressed_bytes": total_mask_uncompressed_bytes,
        "total_compressed_bytes_audited": total_ct_bytes + total_mask_bytes,
        "ct_shape_counts": dict(sorted(ct_shapes.items())),
        "ct_datatype_counts": dict(sorted(ct_datatypes.items())),
        "ct_orientation_counts": dict(sorted(ct_orientations.items())),
        "ct_spacing_ranges": spacing_ranges,
        "ct_intensity_statistics": intensity_summary,
        "mask_count_distribution": dict(sorted(mask_count_distribution.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "issue_category_counts": dict(sorted(issue_counts.items())),
        "elapsed_seconds": round(elapsed, 3),
        "workers": args.workers,
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    report_lines = [
        "# TotalSegmentator CT Dataset Audit",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Audit scope",
        "",
        f"- Search root: `{search_root}`",
        f"- Detected dataset root: `{dataset_root}`",
        f"- Integrity mode: `{args.integrity_mode}`",
        f"- CT integrity coverage: {summary['integrity_scope']['ct']}",
        f"- Mask integrity coverage: {summary['integrity_scope']['masks']}",
        "",
        "## Inventory",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Subjects | {len(results):,} |",
        f"| CT files | {ct_count:,} |",
        f"| Valid CT headers | {valid_ct_count:,} |",
        f"| CT payloads with HU statistics | {len(intensity_results):,} |",
        f"| Mask files | {mask_count:,} |",
        f"| Unique mask labels | {len(label_frequency):,} |",
        f"| Metadata rows | {len(metadata):,} |",
        f"| Compressed NIfTI size | {format_bytes(total_ct_bytes + total_mask_bytes)} |",
        f"| Distinct CT matrix shapes | {len(ct_shapes):,} |",
        f"| CT datatype codes | {', '.join(f'{key}: {value:,}' for key, value in sorted(ct_datatypes.items()))} |",
        f"| CT orientations | {', '.join(f'{key}: {value:,}' for key, value in sorted(ct_orientations.items()))} |",
        "",
        "## Findings",
        "",
        f"- Errors: **{severity_counts.get('error', 0):,}**",
        f"- Warnings: **{severity_counts.get('warning', 0):,}**",
    ]
    if issue_counts:
        report_lines.extend(["", "| Category | Count |", "|---|---:|"])
        report_lines.extend(f"| `{category}` | {count:,} |" for category, count in sorted(issue_counts.items()))
    else:
        report_lines.append("- No file, header, grid-geometry, or metadata-ID inconsistencies were detected.")
    split_distribution = metadata_distributions.get("split", {})
    if split_distribution:
        report_lines.extend(
            [
                "",
                "## Source metadata splits",
                "",
                "These are split labels supplied by `meta.csv`; the audit did not create or modify a research split.",
                "",
                "| Split | Subjects |",
                "|---|---:|",
            ]
        )
        report_lines.extend(f"| `{name}` | {count:,} |" for name, count in split_distribution.items())
    if spacing_ranges:
        report_lines.extend(
            [
                "",
                "## Native CT spacing ranges",
                "",
                "| Axis | Minimum (mm) | Maximum (mm) |",
                "|---|---:|---:|",
            ]
        )
        report_lines.extend(
            f"| {axis.upper()} | {values['min_mm']:.6g} | {values['max_mm']:.6g} |"
            for axis, values in spacing_ranges.items()
        )
    if intensity_results:
        report_lines.extend(
            [
                "",
                "## CT intensity statistics",
                "",
                "Values include NIfTI `scl_slope` / `scl_inter` scaling and are aggregated over finite voxels.",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Global minimum HU | {intensity_summary['global_min_hu']:.6g} |",
                f"| Global maximum HU | {intensity_summary['global_max_hu']:.6g} |",
                f"| Voxel-weighted mean HU | {intensity_summary['voxel_weighted_mean_hu']:.6g} |",
                f"| Voxel-weighted standard deviation HU | {intensity_summary['voxel_weighted_std_hu']:.6g} |",
                f"| Non-finite voxels | {intensity_summary['total_nonfinite_voxels']:,} |",
            ]
        )
    nonzero_metadata_missing = {name: count for name, count in metadata_missing_values.items() if count}
    if nonzero_metadata_missing:
        report_lines.extend(
            [
                "",
                "## Metadata missingness",
                "",
                "These are missing descriptive values in `meta.csv`, not missing CT or mask files.",
                "",
                "| Column | Missing values |",
                "|---|---:|",
            ]
        )
        report_lines.extend(f"| `{name}` | {count:,} |" for name, count in nonzero_metadata_missing.items())
    report_lines.extend(
        [
            "",
            "## Mask coverage",
            "",
            f"All {len(label_frequency):,} discovered labels are listed in `mask_inventory.csv`.",
            "",
            "## Generated files",
            "",
            "- `dataset_summary.json`: machine-readable dataset-level summary",
            "- `dataset_audit.csv`: one row per case with geometry, HU statistics, masks, metadata, and issue count",
            "- `mask_inventory.csv`: per-label subject coverage",
            "- `issues.csv`: actionable error/warning list with exact paths",
            "",
            "## Integrity note",
            "",
            f"The default run reads every CT payload through gzip EOF, verifies its CRC, and computes HU statistics. Mask files receive header/trailer and CT-grid checks by default. Use `--integrity-mode full` for exhaustive mask decompression/CRC checking; these masks advertise {format_bytes(total_mask_uncompressed_bytes)} uncompressed and this operation can take many hours.",
            "",
            f"Audit runtime: {elapsed:.2f} seconds using {args.workers} workers.",
            "",
        ]
    )
    (output_dir / "dataset_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    console_summary = {
        "subject_count": summary["subject_count"],
        "ct_file_count": summary["ct_file_count"],
        "valid_ct_header_count": summary["valid_ct_header_count"],
        "valid_ct_payload_count": summary["valid_ct_payload_count"],
        "mask_file_count": summary["mask_file_count"],
        "unique_mask_labels": summary["unique_mask_labels"],
        "metadata_row_count": summary["metadata_row_count"],
        "severity_counts": summary["severity_counts"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    print(json.dumps(console_summary, indent=2), flush=True)
    print(f"Reports written to: {output_dir}", flush=True)
    return 1 if severity_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
