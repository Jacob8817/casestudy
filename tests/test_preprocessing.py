from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.dataset.loader import get_development_case
from src.dataset.preprocessing import (
    canonicalize_to_ras,
    crop_pair,
    ct_foreground_bbox,
    load_preprocessing_spec,
    normalize_ct,
    resample_pair,
    sample_class_balanced_patch,
)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_preprocessing_spec(Path("configs/dataset.yaml"))

    def test_locked_test_case_is_rejected(self) -> None:
        manifest = {"cases": [{"case_id": "held_out", "split": "test", "ct": {"path": "missing"}}]}
        with self.assertRaisesRegex(PermissionError, "test split is locked"):
            get_development_case(manifest, "held_out")

    def test_fixed_hu_normalization_is_finite_and_bounded(self) -> None:
        image = np.asarray([[[-2000.0, -1000.0, 0.0, 1000.0, 3000.0]]], dtype=np.float32)
        result = normalize_ct(image, self.spec)
        np.testing.assert_allclose(result, np.asarray([[[0.0, 0.0, 0.5, 1.0, 1.0]]], dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)

    def test_lps_is_reoriented_to_ras_without_interpolation(self) -> None:
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        label = (image % 3).astype(np.uint8)
        affine = np.diag([-1.5, -2.0, 3.0, 1.0])
        output_image, output_label, output_affine, trace = canonicalize_to_ras(image, label, affine)
        np.testing.assert_array_equal(output_image, np.flip(np.flip(image, axis=0), axis=1))
        np.testing.assert_array_equal(output_label, np.flip(np.flip(label, axis=0), axis=1))
        self.assertEqual(trace["source_orientation"], "LPS")
        self.assertEqual(trace["target_orientation"], "RAS")
        self.assertFalse(trace["was_noop"])
        self.assertGreater(output_affine[0, 0], 0)
        self.assertGreater(output_affine[1, 1], 0)

    def test_spacing_noop_and_nearest_label_resampling(self) -> None:
        image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        label = np.zeros((3, 3, 3), dtype=np.uint8)
        label[1:, 1:, 1:] = 7
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        same_image, same_label, _, trace = resample_pair(
            image, label, affine, (1.5, 1.5, 1.5), image_fill_hu=-1000
        )
        self.assertIs(same_image, image)
        self.assertIs(same_label, label)
        self.assertTrue(trace["was_noop"])

        resized_image, resized_label, _, resized_trace = resample_pair(
            image, label, affine, (0.75, 0.75, 0.75), image_fill_hu=-1000
        )
        self.assertFalse(resized_trace["was_noop"])
        self.assertEqual(resized_image.dtype, np.float32)
        self.assertEqual(resized_label.dtype, np.uint8)
        self.assertLessEqual(set(np.unique(resized_label)), {0, 7})

    def test_ct_crop_does_not_consult_label_and_updates_origin(self) -> None:
        image = np.full((10, 12, 14), -1000.0, dtype=np.float32)
        image[2:8, 3:9, 4:10] = 0.0
        label = np.zeros_like(image, dtype=np.uint8)
        label[4:6, 5:7, 6:8] = 1
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        bbox = ct_foreground_bbox(image, threshold_hu=-900, margin_voxels=(1, 1, 1))
        self.assertEqual(bbox, ((1, 9), (2, 10), (3, 11)))
        cropped_image, cropped_label, cropped_affine = crop_pair(image, label, affine, bbox)
        self.assertEqual(cropped_image.shape, (8, 8, 8))
        self.assertEqual(int(cropped_label.sum()), int(label.sum()))
        np.testing.assert_allclose(cropped_affine[:3, 3], (1.5, 3.0, 4.5))

    def test_class_balanced_sampler_is_deterministic_and_pads_background_only(self) -> None:
        image_hu = np.zeros((40, 42, 20), dtype=np.float32)
        image = normalize_ct(image_hu, self.spec)
        foreground = np.zeros(image.shape, dtype=np.uint8)
        foreground[20, 20, 10] = 4
        first = sample_class_balanced_patch(
            image, foreground, image_hu, self.spec, case_id="synthetic", sample_index=3
        )
        second = sample_class_balanced_patch(
            image, foreground, image_hu, self.spec, case_id="synthetic", sample_index=3
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[0].shape, (1, 96, 96, 96))
        self.assertEqual(first[1].shape, (96, 96, 96))

        background = np.zeros(image.shape, dtype=np.uint8)
        bg_image, bg_label, trace = sample_class_balanced_patch(
            image, background, image_hu, self.spec, case_id="background", sample_index=0
        )
        self.assertEqual(trace.mode, "background_only_fallback")
        self.assertEqual(bg_image.shape, (1, 96, 96, 96))
        self.assertEqual(bg_label.shape, (96, 96, 96))
        self.assertEqual(int(bg_label.sum()), 0)
        self.assertTrue(any(trace.padding_before) or any(trace.padding_after))


if __name__ == "__main__":
    unittest.main()
