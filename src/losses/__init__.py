"""Losses for anatomy-aware brain/skull learning."""

from .multitask import MultiTaskSegmentationSDFLoss, sdf_weight_for_epoch

__all__ = ["MultiTaskSegmentationSDFLoss", "sdf_weight_for_epoch"]

