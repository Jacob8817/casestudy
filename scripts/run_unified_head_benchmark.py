"""Score available brain/skull methods on the same five oracle-ROI validation CTs.

Run with --run-totalseg to generate the released TotalSegmentator predictions.
Missing fresh nnU-Net v2 or SwinUNETR checkpoints are reported as pending, never
assigned scores. The official test split is neither read nor used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.brain_skull import (  # noqa: E402
    load_or_create_inventory,
    make_model,
    predict_case,
    prepare_inventory,
    prepare_original_inputs,
    sha256_file,
)
from src.benchmark.unified_head import (  # noqa: E402
    cropped_ct_nifti,
    external_masks_to_labels,
    external_multilabel_to_labels,
    score_prediction,
    skull_hu_morphology,
    timed_prediction,
)


METHODS = (
    "hu_threshold_morphology",
    "monai_unet_3d",
    "monai_segresnet",
    "monai_swinunetr",
    "nnunet_v2_fresh",
    "totalsegmentator_released_nnunet",
    "original_monai_unet_joint",
)
SAVED_MODELS = {
    "monai_unet_3d": ("brain_skull_benchmark", "unet3d", "image_path"),
    "monai_segresnet": ("brain_skull_benchmark", "segresnet3d", "image_path"),
    "monai_swinunetr": ("brain_skull_benchmark", "swinunetr3d", "image_path"),
    "original_monai_unet_joint": ("original_monai_head_ablation_matched", "original_monai_brain_skull", "original_image_path"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-totalseg", action="store_true", help="Infer released TotalSegmentator on each validation crop")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "Totalsegmentator_dataset_v201")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "unified_head_benchmark")
    return parser.parse_args()


def model_predictions(method: str, records: list[dict], output_root: Path) -> tuple[list[dict], dict]:
    parent, model_name, image_key = SAVED_MODELS[method]
    model_root = ROOT / "outputs" / parent / model_name
    checkpoint_path = model_root / "best.pt"
    report_path = model_root / "benchmark_report.json"
    if not checkpoint_path.is_file() or not report_path.is_file():
        return [], {"status": "pending_training", "checkpoint": None}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["best_checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["contract_sha256"] != report["contract_sha256"]:
        raise ValueError(f"Checkpoint contract mismatch: {checkpoint_path}")
    model = make_model(model_name).cuda()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = []
    prediction_dir = output_root / "predictions" / method
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        prediction, elapsed = timed_prediction(
            predict_case, model, record, device=torch.device("cuda"), roi=96, image_key=image_key
        )
        np.save(prediction_dir / f"{record['case_id']}.npy", prediction)
        rows.extend(score_prediction(record, prediction, method, elapsed))
    del model
    torch.cuda.empty_cache()
    return rows, {
        "status": "scored",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": report["best_checkpoint_sha256"],
        "selected_epoch": report["selected_epoch"],
        "parameter_count": parameter_count,
        "training_cases": report["train_cases"],
        "validation_cases": report["validation_cases"],
    }


def totalseg_predictions(records: list[dict], dataset_root: Path, output_root: Path, run: bool) -> tuple[list[dict], dict]:
    rows = []
    prediction_dir = output_root / "predictions" / "totalsegmentator_released_nnunet"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version

    for record in records:
        case_dir = prediction_dir / record["case_id"]
        brain_path, skull_path = case_dir / "brain.nii.gz", case_dir / "skull.nii.gz"
        runtime_path = case_dir / "runtime.json"
        if not (brain_path.is_file() and skull_path.is_file()):
            if not run:
                return [], {"status": "pending_inference", "available_cases": len(rows) // 2}
            from totalsegmentator.python_api import totalsegmentator

            ct_path = cropped_ct_nifti(dataset_root, record, case_dir / "oracle_head_ct.nii.gz")
            started = time.perf_counter()
            totalsegmentator(
                input=ct_path,
                output=case_dir,
                task="total",
                roi_subset=["brain", "skull"],
                device="gpu",
                ml=False,
                quiet=True,
                verbose=False,
                fast=False,
                nr_thr_resamp=1,
                nr_thr_saving=1,
            )
            runtime_path.write_text(json.dumps({"seconds": time.perf_counter() - started}, indent=2) + "\n", encoding="utf-8")
        prediction = external_masks_to_labels(record, brain_path, skull_path)
        np.save(case_dir / "labels_on_benchmark_grid.npy", prediction)
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))["seconds"] if runtime_path.is_file() else None
        rows.extend(score_prediction(record, prediction, "totalsegmentator_released_nnunet", runtime))
        print(f"TotalSegmentator: {record['case_id']} scored", flush=True)
    return rows, {"status": "scored_reference_only", "version": version("TotalSegmentator"), "input": "oracle-head-crop HU NIfTI"}


def fresh_nnunet_predictions(records: list[dict], output_root: Path) -> tuple[list[dict], dict]:
    directory = output_root / "predictions" / "nnunet_v2_fresh"
    if not all((directory / f"{item['case_id']}.nii.gz").is_file() for item in records):
        return [], {"status": "pending_training_and_inference", "prediction_dir": str(directory)}
    rows = []
    for record in records:
        prediction = external_multilabel_to_labels(record, directory / f"{record['case_id']}.nii.gz")
        rows.extend(score_prediction(record, prediction, "nnunet_v2_fresh", None))
    return rows, {"status": "scored", "prediction_dir": str(directory)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to score the trained deep models")
    inventory = load_or_create_inventory(
        args.dataset_root, ROOT / "data" / "splits" / "dataset_manifest.json",
        ROOT / "outputs" / "brain_skull_benchmark" / "development_inventory.json",
    )
    if inventory["test_split_accessed"] or inventory["selected_counts"] != {"train": 98, "val": 5}:
        raise ValueError("Unexpected development cohort")
    records = prepare_inventory(args.dataset_root, inventory, ROOT / "data" / "cache" / "brain_skull_benchmark")
    records = prepare_original_inputs(args.dataset_root, records, ROOT / "data" / "cache" / "brain_skull_benchmark")
    val_records = [item for item in records if item["split"] == "val"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    status = {}
    for method in METHODS:
        print(f"Scoring {method}", flush=True)
        if method == "hu_threshold_morphology":
            rows = []
            for record in val_records:
                prediction, elapsed = timed_prediction(skull_hu_morphology, record)
                # A skull-only baseline has no brain output, not a zero brain score.
                rows.extend(item for item in score_prediction(record, prediction, method, elapsed) if item["organ"] == "skull")
            detail = {"status": "scored_skull_only", "recipe": ">300 HU, 3x3x3 closing, remove components <100 voxels"}
        elif method in SAVED_MODELS:
            rows, detail = model_predictions(method, val_records, args.output_root)
        elif method == "totalsegmentator_released_nnunet":
            rows, detail = totalseg_predictions(val_records, args.dataset_root, args.output_root, args.run_totalseg)
        else:
            rows, detail = fresh_nnunet_predictions(val_records, args.output_root)
        all_rows.extend(rows)
        status[method] = detail
        print(f"  {detail['status']}", flush=True)
    per_case = pd.DataFrame(all_rows)
    per_case.to_csv(args.output_root / "per_case_metrics.csv", index=False)
    summary = (
        per_case.groupby(["method", "organ"], as_index=False)
        .agg(
            mean_dice=("dice", "mean"),
            mean_hd95_mm=("hd95_mm", "mean"),
            mean_volume_error_pct=("volume_error_pct", "mean"),
            mean_inference_seconds=("inference_seconds", "mean"),
            cases=("case_id", "nunique"),
        )
    )
    summary.to_csv(args.output_root / "comparison.csv", index=False)
    manifest = {
        "cohort": "98 train / 5 validation complete-head CTs from official TotalSegmentator public dataset",
        "evaluation": "same five reference-mask-defined head crops, exclusive brain=1/skull=2; skull wins overlaps",
        "test_split_accessed": False,
        "clinical_ct_only_validity": False,
        "released_totalsegmentator_independent_external_validation": False,
        "input_preprocessing_not_identical_between_models": True,
        "validation_used_for_checkpoint_selection": True,
        "method_status": status,
        "sources": {
            "nnunet": "https://github.com/MIC-DKFZ/nnUNet",
            "totalsegmentator_training": "https://github.com/wasserth/TotalSegmentator/blob/master/resources/train_nnunet.md",
            "swinunetr": "https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV",
        },
    }
    (args.output_root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
