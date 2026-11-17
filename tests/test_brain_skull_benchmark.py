"""Safety and metric tests for the independent brain/skull benchmark."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.benchmark.brain_skull import (
    development_cases,
    dice_per_class,
    prepare_case,
    prepare_original_inputs,
    sample_patch,
    scan_development_inventory,
)


class BrainSkullBenchmarkTests(unittest.TestCase):
    def test_test_split_is_not_resolved_or_opened(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            cases = [
                {"case_id": "train", "split": "train", "ct": {"path": "train/ct.nii.gz"}},
                {"case_id": "val", "split": "val", "ct": {"path": "val/ct.nii.gz"}},
                {"case_id": "missing_test", "split": "test", "ct": {"path": "missing_test/ct.nii.gz"}},
            ]
            manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            affine = np.diag([10.0, 10.0, 10.0, 1.0])
            for case_id in ("train", "val"):
                case_root = root / case_id
                masks = case_root / "segmentations"
                masks.mkdir(parents=True)
                ct = np.zeros((16, 16, 16), dtype=np.float32)
                brain = np.zeros_like(ct, dtype=np.uint8)
                skull = np.zeros_like(ct, dtype=np.uint8)
                brain[3:10, 3:10, 3:10] = 1
                skull[9:14, 9:14, 9:14] = 1
                nib.save(nib.Nifti1Image(ct, affine), case_root / "ct.nii.gz")
                nib.save(nib.Nifti1Image(brain, affine), masks / "brain.nii.gz")
                nib.save(nib.Nifti1Image(skull, affine), masks / "skull.nii.gz")
            self.assertEqual(len(development_cases(manifest)), 2)
            inventory = scan_development_inventory(root, manifest, min_brain_ml=0.01, min_skull_ml=0.01)
            self.assertEqual(inventory["selected_counts"], {"train": 1, "val": 1})
            self.assertFalse(inventory["test_split_accessed"])
            prepared = prepare_case(root, inventory["cases"][0], root / "cache")
            image, label = sample_patch(prepared, np.random.default_rng(1), roi=16)
            self.assertEqual(image.shape, (2, 16, 16, 16))
            self.assertEqual(label.shape, (16, 16, 16))
            self.assertEqual(set(np.unique(label)), {0, 1, 2})
            self.assertEqual(int(np.load(prepared["label_path"])[9, 9, 9]), 2)  # Skull wins overlap.
            original = prepare_original_inputs(root, [prepared], root / "cache")[0]
            self.assertEqual(np.load(original["original_image_path"]).shape, (1, 16, 16, 16))
            brain_image, brain_label = sample_patch(
                original, np.random.default_rng(2), roi=16,
                image_key="original_image_path", task="brain_only",
            )
            self.assertEqual(brain_image.shape, (1, 16, 16, 16))
            self.assertEqual(set(np.unique(brain_label)), {0, 1})
            skull_image, skull_label = sample_patch(
                original, np.random.default_rng(2), roi=16,
                image_key="original_image_path", task="skull_only",
            )
            self.assertEqual(skull_image.shape, (1, 16, 16, 16))
            self.assertEqual(set(np.unique(skull_label)), {0, 1})
            matched_brain, matched_brain_label = sample_patch(
                original, np.random.default_rng(3), roi=16,
                image_key="original_image_path", task="brain_only", sampling_targets=(1, 2),
            )
            matched_joint, matched_joint_label = sample_patch(
                original, np.random.default_rng(3), roi=16,
                image_key="original_image_path", task="joint", sampling_targets=(1, 2),
            )
            np.testing.assert_array_equal(matched_brain, matched_joint)
            np.testing.assert_array_equal(matched_brain_label, matched_joint_label == 1)

    def test_per_class_dice(self) -> None:
        truth = np.array([0, 1, 1, 2, 2], dtype=np.uint8)
        prediction = np.array([0, 1, 2, 2, 0], dtype=np.uint8)
        scores = dice_per_class(prediction, truth)
        self.assertAlmostEqual(scores["brain"], 2 / 3)
        self.assertAlmostEqual(scores["skull"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
