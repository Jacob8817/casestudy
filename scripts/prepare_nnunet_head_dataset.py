"""Export the frozen head-CT cohort for a fresh nnU-Net v2 comparison.

The five validation labels are available to nnU-Net only as its fixed fold-0
validation split. No official test case is exported. All scans use the same
reference-defined head crop as the MONAI development benchmark.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.brain_skull import load_or_create_inventory, prepare_inventory, sha256_file  # noqa: E402
from src.benchmark.unified_head import cropped_ct_nifti, reference_geometry  # noqa: E402


DATASET_NAME = "Dataset901_BrainSkullOracleROI"
OUTPUT_ROOT = ROOT / "outputs" / "unified_head_benchmark" / "nnunet_v2_fresh"


def main() -> None:
    dataset_root = Path(os.environ.get("TOTALSEG_DATASET_ROOT", ROOT / "Totalsegmentator_dataset_v201"))
    inventory_path = ROOT / "outputs" / "brain_skull_benchmark" / "development_inventory.json"
    inventory = load_or_create_inventory(
        dataset_root, ROOT / "data" / "splits" / "dataset_manifest.json", inventory_path
    )
    if inventory["test_split_accessed"] or inventory["selected_counts"] != {"train": 98, "val": 5}:
        raise ValueError("Unexpected development cohort")
    records = prepare_inventory(dataset_root, inventory, ROOT / "data" / "cache" / "brain_skull_benchmark")
    raw = OUTPUT_ROOT / "nnUNet_raw" / DATASET_NAME
    images_tr, labels_tr, images_ts = (raw / folder for folder in ("imagesTr", "labelsTr", "imagesTs"))
    for directory in (images_tr, labels_tr, images_ts):
        directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, 1):
        case_id = record["case_id"]
        image_path = cropped_ct_nifti(dataset_root, record, images_tr / f"{case_id}_0000.nii.gz")
        label_path = labels_tr / f"{case_id}.nii.gz"
        if not label_path.is_file():
            shape, affine = reference_geometry(record)
            label = np.asarray(np.load(record["label_path"], mmap_mode="r"), dtype=np.uint8)
            if label.shape != shape or not np.isin(np.unique(label), (0, 1, 2)).all():
                raise ValueError(f"Unexpected reference label: {case_id}")
            nib.save(nib.Nifti1Image(label, affine), str(label_path))
        if record["split"] == "val":
            validation_image = images_ts / f"{case_id}_0000.nii.gz"
            if not validation_image.is_file():
                os.link(image_path, validation_image)
        print(f"{index}/{len(records)} {case_id}", flush=True)
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "brain": 1, "skull": 2},
        "numTraining": len(records),
        "file_ending": ".nii.gz",
        "description": "Oracle-ROI brain/skull development cohort; 98 train, 5 fixed validation",
    }
    (raw / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
    split = [{
        "train": sorted(record["case_id"] for record in records if record["split"] == "train"),
        "val": sorted(record["case_id"] for record in records if record["split"] == "val"),
    }]
    for split_path in (
        raw / "splits_final.json",
        OUTPUT_ROOT / "nnUNet_preprocessed" / DATASET_NAME / "splits_final.json",
    ):
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    contract = {
        "dataset_name": DATASET_NAME,
        "inventory_sha256": sha256_file(inventory_path),
        "train_cases": len(split[0]["train"]),
        "validation_cases": len(split[0]["val"]),
        "class_priority": "skull wins overlap",
        "head_crop": "reference brain-or-skull bounding box + 16 voxels",
        "official_test_split_accessed": False,
        "raw_folder": str(raw),
    }
    (OUTPUT_ROOT / "export_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print("Exported", DATASET_NAME, "with", len(split[0]["train"]), "train and", len(split[0]["val"]), "validation scans")


if __name__ == "__main__":
    main()
