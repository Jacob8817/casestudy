"""Differentiable, non-template brain/skull relational loss."""

from __future__ import annotations

import torch


class AnatomyRelationalLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        brain_outside_enclosure_weight: float,
        skull_in_brain_core_weight: float,
        probability_ambiguity_weight: float,
        sdf_sign_consistency_weight: float,
    ) -> None:
        super().__init__()
        weights = (
            brain_outside_enclosure_weight,
            skull_in_brain_core_weight,
            probability_ambiguity_weight,
            sdf_sign_consistency_weight,
        )
        if any(value < 0 for value in weights) or not any(value > 0 for value in weights):
            raise ValueError("Anatomy weights must be non-negative with at least one positive term")
        self.weights = weights

    def forward(
        self,
        segmentation_logits: torch.Tensor,
        predicted_sdf: torch.Tensor,
        enclosure: torch.Tensor,
        brain_core: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probabilities = torch.softmax(segmentation_logits, dim=1)
        brain, skull = probabilities[:, 1], probabilities[:, 2]
        allowed = enclosure[:, 0].to(brain.dtype)
        core = brain_core[:, 0].to(brain.dtype)
        brain_outside = (brain * (1.0 - allowed)).sum() / brain.sum().clamp_min(1.0)
        skull_in_core = (skull * core).sum() / core.sum().clamp_min(1.0)
        ambiguity = 4.0 * torch.mean(brain * skull)
        class_probabilities = probabilities[:, 1:3]
        sign_disagreement = torch.mean(
            class_probabilities * torch.relu(-predicted_sdf)
            + (1.0 - class_probabilities) * torch.relu(predicted_sdf)
        )
        terms = {
            "brain_outside_enclosure": brain_outside,
            "skull_in_brain_core": skull_in_core,
            "probability_ambiguity": ambiguity,
            "sdf_sign_consistency": sign_disagreement,
        }
        total = sum(weight * term for weight, term in zip(self.weights, terms.values()))
        return total, terms
