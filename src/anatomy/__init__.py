"""Brain/skull relational constraints and post-hoc anatomical quality control."""

from .constraints import AnatomyRelationalLoss
from .targets import build_anatomy_targets
from .violations import anatomy_violation_map, fit_surface_distance_reference

__all__ = [
    "AnatomyRelationalLoss",
    "anatomy_violation_map",
    "build_anatomy_targets",
    "fit_surface_distance_reference",
]

