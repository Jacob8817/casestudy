"""Small provenance helpers shared by Phase 4 setup and future runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


PROHIBITED_MODEL_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".safetensors", ".onnx"})
PROHIBITED_RUN_DIRECTORIES = frozenset({"checkpoints", "runs", "wandb", "mlruns"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_artifact_inventory(project_root: Path) -> list[dict[str, Any]]:
    """Inventory checkpoint/run artifacts only within the project Phase 4 output tree."""
    project_root = project_root.resolve()
    phase4_root = project_root / "outputs" / "phase4"
    if not phase4_root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(phase4_root.rglob("*")):
        relative = path.relative_to(project_root).as_posix()
        if path.is_dir() and path.name.lower() in PROHIBITED_RUN_DIRECTORIES:
            records.append({"path": relative, "kind": "prohibited_run_directory"})
        elif path.is_file() and (
            path.suffix.lower() in PROHIBITED_MODEL_SUFFIXES
            or path.name.startswith("events.out.tfevents")
        ):
            records.append(
                {
                    "path": relative,
                    "kind": "prohibited_training_file",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def hash_artifacts(project_root: Path, relative_paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    root = project_root.resolve()
    result: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Provenance path escapes project root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        result[Path(relative).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
