"""Validated organ-schema loading for multi-organ segmentation targets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OrganDefinition:
    id: int
    name: str
    source_masks: tuple[str, ...]


@dataclass(frozen=True)
class OrganSchema:
    schema_version: int
    name: str
    description: str
    background_id: int
    background_name: str
    merge_mode: str
    overlap_policy: str
    overlap_method: str
    overlap_priority: tuple[str, ...]
    overlap_reference: str
    labels: tuple[OrganDefinition, ...]

    @property
    def source_masks(self) -> tuple[str, ...]:
        return tuple(source for label in self.labels for source in label.source_masks)

    @property
    def num_output_channels(self) -> int:
        return len(self.labels) + 1

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "background": {"id": self.background_id, "name": self.background_name},
            "merge_mode": self.merge_mode,
            "overlap_policy": self.overlap_policy,
            "overlap_resolution": {
                "method": self.overlap_method,
                "priority_low_to_high": list(self.overlap_priority),
                "reference": self.overlap_reference,
            },
            "labels": [
                {"id": label.id, "name": label.name, "source_masks": list(label.source_masks)}
                for label in self.labels
            ],
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_organ_schema(path: Path) -> OrganSchema:
    """Load and strictly validate an organ mapping YAML file."""
    path = path.resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load organ schema {path}: {exc}") from exc
    root = _require_mapping(raw, "organ schema")
    background = _require_mapping(root.get("background"), "background")
    raw_labels = root.get("labels")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("labels must be a non-empty list")

    labels: list[OrganDefinition] = []
    for index, raw_label in enumerate(raw_labels):
        item = _require_mapping(raw_label, f"labels[{index}]")
        label_id = item.get("id")
        name = item.get("name")
        sources = item.get("source_masks")
        if not isinstance(label_id, int) or label_id <= 0:
            raise ValueError(f"labels[{index}].id must be a positive integer")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"labels[{index}].name must be a non-empty string")
        if not isinstance(sources, list) or not sources or any(not isinstance(source, str) or not source.strip() for source in sources):
            raise ValueError(f"labels[{index}].source_masks must be a non-empty string list")
        labels.append(OrganDefinition(label_id, name.strip(), tuple(source.strip() for source in sources)))

    ids = [label.id for label in labels]
    names = [label.name for label in labels]
    sources = [source for label in labels for source in label.source_masks]
    expected_ids = list(range(1, len(labels) + 1))
    if ids != expected_ids:
        raise ValueError(f"label ids must be ordered and contiguous: expected {expected_ids}, found {ids}")
    if len(names) != len(set(names)):
        raise ValueError("label names must be unique")
    if len(sources) != len(set(sources)):
        raise ValueError("a source mask cannot be assigned to more than one target label")
    if background.get("id") != 0 or background.get("name") != "background":
        raise ValueError("background must be exactly {id: 0, name: background}")
    if root.get("merge_mode") != "union":
        raise ValueError("only merge_mode=union is supported")
    overlap_policy = root.get("overlap_policy")
    if overlap_policy not in {"error", "warn", "priority"}:
        raise ValueError("overlap_policy must be 'error', 'warn', or 'priority'")
    overlap_resolution = root.get("overlap_resolution") or {}
    if overlap_policy == "priority":
        overlap_resolution = _require_mapping(overlap_resolution, "overlap_resolution")
        overlap_method = overlap_resolution.get("method")
        overlap_priority = overlap_resolution.get("priority_low_to_high")
        overlap_reference = overlap_resolution.get("reference")
        if overlap_method != "official_totalsegmentator_v2_last_label_wins":
            raise ValueError("priority overlap policy requires the supported official v2 last-label-wins method")
        if not isinstance(overlap_priority, list) or overlap_priority != list(dict.fromkeys(overlap_priority)):
            raise ValueError("overlap priority must be an ordered list without duplicates")
        if set(overlap_priority) != set(names):
            raise ValueError("overlap priority must contain every target label exactly once")
        if not isinstance(overlap_reference, str) or not overlap_reference.startswith("https://"):
            raise ValueError("overlap resolution requires an HTTPS reference")
    else:
        overlap_method = "none"
        overlap_priority = names
        overlap_reference = ""

    return OrganSchema(
        schema_version=int(root.get("schema_version", 0)),
        name=str(root.get("name", "")).strip(),
        description=str(root.get("description", "")).strip(),
        background_id=0,
        background_name="background",
        merge_mode="union",
        overlap_policy=str(overlap_policy),
        overlap_method=str(overlap_method),
        overlap_priority=tuple(overlap_priority),
        overlap_reference=str(overlap_reference),
        labels=tuple(labels),
    )
