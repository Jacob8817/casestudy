from __future__ import annotations

import csv
import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset.label_mapping import load_organ_schema
from src.dataset.selected_mask_audit import audit_selected_case, write_selected_audit_outputs


def write_nifti(path: Path, values: np.ndarray, *, datatype: int) -> None:
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    dimensions = [3, *values.shape, 1, 1, 1, 1]
    struct.pack_into("<8h", header, 40, *dimensions)
    if datatype == 2:
        dtype, bitpix = np.dtype("u1"), 8
    elif datatype == 4:
        dtype, bitpix = np.dtype("<i2"), 16
    else:
        raise ValueError(datatype)
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    struct.pack_into("<8f", header, 76, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    header[123] = 2  # millimetres
    struct.pack_into("<h", header, 254, 2)
    struct.pack_into("<4f", header, 280, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, 1.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, 1.0, 0.0)
    header[344:348] = b"n+1\x00"
    with gzip.open(path, "wb") as stream:
        stream.write(header)
        stream.write(values.astype(dtype).tobytes(order="F"))


class SelectedMaskAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temp_directory.name)
        self.schema = load_organ_schema(Path("configs/organs_v1.yaml"))

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def make_case(self, split: str = "train", overrides: dict[str, np.ndarray] | None = None) -> dict:
        overrides = overrides or {}
        case_id = "s0000"
        case_dir = self.dataset_root / case_id
        segmentation_dir = case_dir / "segmentations"
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        write_nifti(case_dir / "ct.nii.gz", np.zeros((2, 2, 2), dtype=np.int16), datatype=4)
        targets = {}
        for label in self.schema.labels:
            sources = []
            for source in label.source_masks:
                values = overrides.get(source, np.zeros((2, 2, 2), dtype=np.uint8))
                path = segmentation_dir / f"{source}.nii.gz"
                write_nifti(path, values, datatype=2)
                sources.append({"name": source, "path": f"{case_id}/segmentations/{source}.nii.gz"})
            targets[label.name] = {"class_id": label.id, "sources": sources}
        return {
            "case_id": case_id,
            "split": split,
            "ct": {"path": f"{case_id}/ct.nii.gz"},
            "targets": targets,
        }

    def test_composite_lung_union_and_full_integrity(self) -> None:
        upper = np.zeros((2, 2, 2), dtype=np.uint8)
        lower = np.zeros_like(upper)
        upper.flat[0] = 1
        lower.flat[1] = 1
        case = self.make_case(
            overrides={"lung_upper_lobe_left": upper, "lung_lower_lobe_left": lower}
        )
        result = audit_selected_case(case, self.dataset_root, self.schema, chunk_voxels=3)
        self.assertTrue(result.metrics_valid)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(result.targets["lung_left"].foreground_voxels, 2)
        self.assertAlmostEqual(result.targets["lung_left"].volume_ml or 0, 0.002)
        self.assertEqual(result.targets["lung_left"].merged_source_overlap_voxels, 0)
        self.assertTrue(all(source.binary_valid and source.payload_valid for source in result.sources.values()))

    def test_overlap_and_nonbinary_values_are_errors(self) -> None:
        liver = np.zeros((2, 2, 2), dtype=np.uint8)
        spleen = np.zeros_like(liver)
        liver.flat[0] = 1
        spleen.flat[0] = 1
        overlap_case = self.make_case(overrides={"liver": liver, "spleen": spleen})
        overlap_result = audit_selected_case(
            overlap_case, self.dataset_root, self.schema, chunk_voxels=4
        )
        self.assertIn(
            "cross_target_overlap_resolved",
            {issue["category"] for issue in overlap_result.issues},
        )
        self.assertEqual(overlap_result.targets["spleen"].exclusive_foreground_voxels, 0)
        self.assertEqual(overlap_result.targets["liver"].exclusive_foreground_voxels, 1)

        pancreas = np.zeros_like(liver)
        pancreas.flat[2] = 2
        nonbinary_case = self.make_case(overrides={"pancreas": pancreas})
        result = audit_selected_case(
            nonbinary_case, self.dataset_root, self.schema, chunk_voxels=4
        )
        categories = {issue["category"] for issue in result.issues}
        self.assertIn("nonbinary_mask", categories)
        self.assertGreater(result.error_count, 0)

    def test_test_case_volume_columns_and_statistics_are_redacted(self) -> None:
        liver = np.ones((2, 2, 2), dtype=np.uint8)
        case = self.make_case(split="test", overrides={"liver": liver})
        result = audit_selected_case(case, self.dataset_root, self.schema, chunk_voxels=8)
        csv_path = self.dataset_root / "audit.csv"
        json_path = self.dataset_root / "stats.json"
        summary = write_selected_audit_outputs(
            [result], self.schema, {"manifest_sha256": "test"}, csv_path, json_path
        )
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        liver_row = next(row for row in rows if row["organ"] == "liver")
        self.assertEqual(liver_row["present"], "True")
        self.assertEqual(liver_row["foreground_voxels"], "")
        self.assertEqual(liver_row["volume_ml"], "")
        self.assertTrue(summary["organs"]["liver"]["test"]["redacted"])
        self.assertEqual(summary["organs"]["liver"]["train_validation"]["development"]["case_count"], 0)
        self.assertEqual(json.loads(json_path.read_text())["test_policy"]["locked"], True)


if __name__ == "__main__":
    unittest.main()
