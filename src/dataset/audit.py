"""Public dataset-audit API.

The implementation remains importable from the original top-level module for
backward compatibility, while project scripts use this package path.
"""

from audit_dataset import (
    AUDIT_VERSION,
    IntensityStats,
    Issue,
    NiftiInfo,
    SubjectResult,
    audit_subject,
    compute_ct_intensity_stats,
    discover_subject_dirs,
    inspect_nifti,
    main,
    orientation_from_affine,
)

__all__ = [
    "AUDIT_VERSION",
    "IntensityStats",
    "Issue",
    "NiftiInfo",
    "SubjectResult",
    "audit_subject",
    "compute_ct_intensity_stats",
    "discover_subject_dirs",
    "inspect_nifti",
    "main",
    "orientation_from_affine",
]
