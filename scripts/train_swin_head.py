"""Train/resume a tiny MONAI SwinUNETR on Notebook 15's frozen head cohort."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.brain_skull import (  # noqa: E402
    load_or_create_inventory,
    prepare_inventory,
    sha256_file,
    train_model,
)


def main() -> None:
    dataset_root = Path(os.environ.get("TOTALSEG_DATASET_ROOT", ROOT / "Totalsegmentator_dataset_v201"))
    inventory_path = ROOT / "outputs" / "brain_skull_benchmark" / "development_inventory.json"
    inventory = load_or_create_inventory(
        dataset_root, ROOT / "data" / "splits" / "dataset_manifest.json", inventory_path
    )
    if inventory["test_split_accessed"] or inventory["selected_counts"] != {"train": 98, "val": 5}:
        raise ValueError("Unexpected development cohort")
    records = prepare_inventory(dataset_root, inventory, ROOT / "data" / "cache" / "brain_skull_benchmark")
    report = train_model(
        "swinunetr3d",
        records,
        ROOT / "outputs" / "brain_skull_benchmark",
        inventory_sha256=sha256_file(inventory_path),
        max_epochs=30,
        roi=96,
        learning_rate=2e-4,
        seed=42,
    )
    print(json.dumps({
        "best_checkpoint": report["best_checkpoint"],
        "selected_epoch": report["selected_epoch"],
        "validation": report["validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
