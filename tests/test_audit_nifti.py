from __future__ import annotations

import gzip
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_dataset import compute_ct_intensity_stats, inspect_nifti


def write_test_nifti(
    path: Path,
    values: np.ndarray,
    *,
    endian: str = "<",
    slope: float = 1.0,
    intercept: float = 0.0,
    affine_diagonal: tuple[float, float, float] = (1.5, 1.5, 1.5),
    ndim: int = 3,
    vox_offset: float = 352.0,
) -> None:
    header = bytearray(352)
    struct.pack_into(endian + "i", header, 0, 348)
    dimensions = [1] * 8
    dimensions[0] = ndim
    for index, size in enumerate(values.shape, start=1):
        dimensions[index] = size
    struct.pack_into(endian + "8h", header, 40, *dimensions)
    struct.pack_into(endian + "h", header, 70, 4)   # int16
    struct.pack_into(endian + "h", header, 72, 16)
    pixdim = [1.0, abs(affine_diagonal[0]), abs(affine_diagonal[1]), abs(affine_diagonal[2]), 1.0, 1.0, 1.0, 1.0]
    struct.pack_into(endian + "8f", header, 76, *pixdim)
    struct.pack_into(endian + "f", header, 108, vox_offset)
    struct.pack_into(endian + "f", header, 112, slope)
    struct.pack_into(endian + "f", header, 116, intercept)
    struct.pack_into(endian + "h", header, 252, 0)
    struct.pack_into(endian + "h", header, 254, 2)
    struct.pack_into(endian + "4f", header, 280, affine_diagonal[0], 0.0, 0.0, 0.0)
    struct.pack_into(endian + "4f", header, 296, 0.0, affine_diagonal[1], 0.0, 0.0)
    struct.pack_into(endian + "4f", header, 312, 0.0, 0.0, affine_diagonal[2], 0.0)
    header[344:348] = b"n+1\x00"
    dtype = np.dtype(endian + "i2")
    payload = values.astype(dtype, copy=False).tobytes(order="F")
    with gzip.open(path, "wb") as stream:
        stream.write(header)
        stream.write(payload)


class NiftiAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_scaled_intensity_statistics_and_ras_orientation(self) -> None:
        path = self.root / "scaled.nii.gz"
        raw = np.array([[[-1000, 0], [1000, 2000]]], dtype=np.int16)
        write_test_nifti(path, raw, slope=2.0, intercept=-100.0)
        info = inspect_nifti(path, "header")
        stats = compute_ct_intensity_stats(path, info, chunk_bytes=4)
        self.assertEqual(info.orientation, "RAS")
        self.assertEqual(stats.voxel_count, 4)
        self.assertEqual(stats.min_hu, -2100.0)
        self.assertEqual(stats.max_hu, 3900.0)
        self.assertAlmostEqual(stats.mean_hu, 900.0)
        self.assertAlmostEqual(stats.std_hu, math.sqrt(5_000_000.0))
        self.assertTrue(stats.payload_crc_checked)

    def test_big_endian_and_zero_slope_mean_identity_scaling(self) -> None:
        path = self.root / "big_endian.nii.gz"
        raw = np.array([[[1, 2], [3, 4]]], dtype=np.int16)
        write_test_nifti(path, raw, endian=">", slope=0.0, intercept=999.0)
        info = inspect_nifti(path, "header")
        stats = compute_ct_intensity_stats(path, info, chunk_bytes=2)
        self.assertEqual(info.endian, ">")
        self.assertEqual(stats.min_hu, 1.0)
        self.assertEqual(stats.max_hu, 4.0)
        self.assertAlmostEqual(stats.mean_hu, 2.5)

    def test_orientation_codes_follow_affine_sign(self) -> None:
        path = self.root / "lps.nii.gz"
        raw = np.zeros((2, 2, 2), dtype=np.int16)
        write_test_nifti(path, raw, affine_diagonal=(-1.5, -1.5, 1.5))
        self.assertEqual(inspect_nifti(path, "header").orientation, "LPS")

    def test_crc_corruption_is_detected_when_payload_reaches_eof(self) -> None:
        path = self.root / "bad_crc.nii.gz"
        raw = np.arange(64, dtype=np.int16).reshape((4, 4, 4))
        write_test_nifti(path, raw)
        content = bytearray(path.read_bytes())
        content[-8] ^= 0xFF  # corrupt CRC but preserve the ISIZE trailer
        path.write_bytes(content)
        info = inspect_nifti(path, "header")
        with self.assertRaisesRegex(ValueError, "CRC"):
            compute_ct_intensity_stats(path, info)

    def test_non_3d_nifti_is_rejected(self) -> None:
        path = self.root / "four_dimensional.nii.gz"
        raw = np.zeros((2, 2, 2, 2), dtype=np.int16)
        write_test_nifti(path, raw, ndim=4)
        with self.assertRaisesRegex(ValueError, "expected a 3D"):
            inspect_nifti(path, "header")

    def test_invalid_voxel_offset_is_rejected(self) -> None:
        path = self.root / "bad_offset.nii.gz"
        raw = np.zeros((2, 2, 2), dtype=np.int16)
        write_test_nifti(path, raw, vox_offset=348.0)
        with self.assertRaisesRegex(ValueError, "vox_offset"):
            inspect_nifti(path, "header")


if __name__ == "__main__":
    unittest.main()
