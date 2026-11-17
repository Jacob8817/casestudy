"""Dataset discovery, audit, loading, and preprocessing utilities."""

from .audit import audit_subject, compute_ct_intensity_stats, discover_subject_dirs, inspect_nifti
from .label_mapping import OrganDefinition, OrganSchema, load_organ_schema
from .loader import (
    DEVELOPMENT_SPLITS,
    LoadedCase,
    development_cases,
    get_development_case,
    load_exclusive_case,
    load_validated_manifest,
    read_nifti_volume,
)
from .preprocessing import (
    PatchTrace,
    PreprocessingSpec,
    canonicalize_to_ras,
    crop_pair,
    ct_foreground_bbox,
    extract_centered_patch,
    load_preprocessing_spec,
    normalize_ct,
    resample_pair,
    sample_class_balanced_patch,
)

__all__ = [
    "audit_subject",
    "compute_ct_intensity_stats",
    "discover_subject_dirs",
    "inspect_nifti",
    "OrganDefinition",
    "OrganSchema",
    "load_organ_schema",
    "DEVELOPMENT_SPLITS",
    "LoadedCase",
    "development_cases",
    "get_development_case",
    "load_exclusive_case",
    "load_validated_manifest",
    "read_nifti_volume",
    "PatchTrace",
    "PreprocessingSpec",
    "canonicalize_to_ras",
    "crop_pair",
    "ct_foreground_bbox",
    "extract_centered_patch",
    "load_preprocessing_spec",
    "normalize_ct",
    "resample_pair",
    "sample_class_balanced_patch",
]
