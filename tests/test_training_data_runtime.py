from __future__ import annotations

import hashlib
import gc
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from src.dataset import (
    load_organ_schema,
    load_preprocessing_spec,
    normalize_ct,
    sample_class_balanced_patch,
)
from src.training import load_baseline_config
from src.training.data import (
    ExactSplitPatchDataset,
    ExactSplitVolumeDataset,
    PreparedCase,
    build_training_loaders,
    prepare_case,
    prewarm_cache,
    sample_class_balanced_hu_patch,
    verify_case_source_hashes,
)
from src.training.gates import _materialize_fixed_bank


class _TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"value": torch.tensor(index)}


class _CountingFixedBank(Dataset[dict[str, object]]):
    def __init__(self) -> None:
        self.calls = 0

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        self.calls += 1
        return {
            "image": torch.full((1, 2, 2, 2), float(index), dtype=torch.float32),
            "label": torch.full((1, 2, 2, 2), index, dtype=torch.int64),
            "sample_index": index,
            "patch_trace": {"selected_class_id": index + 1},
        }


class TrainingDataRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config = load_baseline_config(cls.project_root / "configs" / "baseline_unet_v1.yaml")
        cls.schema = load_organ_schema(cls.project_root / "configs" / "organs_v1.yaml")
        cls.preprocessing = load_preprocessing_spec(cls.project_root / "configs" / "dataset.yaml")

    @staticmethod
    def _case(case_id: str = "s_train", split: str = "train") -> dict[str, object]:
        return {
            "case_id": case_id,
            "split": split,
            "ct": {"path": f"{case_id}/ct.nii.gz", "file_sha256": "0" * 64},
            "targets": {},
        }

    def test_patch_only_normalization_is_bitwise_phase3_equivalent(self) -> None:
        preprocessing = replace(self.preprocessing, roi_size=(5, 7, 9))
        rng = np.random.default_rng(7351)
        ct_hu = rng.normal(40.0, 650.0, size=(11, 13, 15)).astype(np.float32)
        label = np.zeros(ct_hu.shape, dtype=np.uint8)
        label[1:5, 2:8, 3:10] = 3
        label[7:10, 8:12, 9:14] = 8
        normalized = normalize_ct(ct_hu, preprocessing)
        for sample_index in range(24):
            expected_image, expected_label, expected_trace = sample_class_balanced_patch(
                normalized,
                label,
                ct_hu,
                preprocessing,
                case_id="parity",
                sample_index=sample_index,
            )
            actual_image, actual_label, actual_trace = sample_class_balanced_hu_patch(
                ct_hu,
                label,
                preprocessing,
                case_id="parity",
                sample_index=sample_index,
            )
            self.assertTrue(np.array_equal(actual_image, expected_image))
            self.assertTrue(np.array_equal(actual_label, expected_label))
            self.assertEqual(actual_trace, expected_trace)

    def test_overfit_fixed_bank_is_materialized_once(self) -> None:
        dataset = _CountingFixedBank()
        generator = torch.Generator().manual_seed(42)
        source = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator, num_workers=0)
        training, evaluation = _materialize_fixed_bank(source)
        self.assertEqual(dataset.calls, len(dataset))
        first_eval = [
            (int(batch["sample_index"]), int(batch["patch_trace"]["selected_class_id"]))
            for batch in evaluation
        ]
        list(training)
        list(training)
        self.assertEqual(dataset.calls, len(dataset))
        self.assertEqual(first_eval, [(0, 1), (1, 2), (2, 3)])

    def test_patch_trace_is_default_collatable_and_fixed_bank_ignores_epoch(self) -> None:
        manifest = {"cases": [self._case()]}
        prepared = PreparedCase(
            case_id="s_train",
            split="train",
            ct_hu=np.linspace(-1000, 1000, 16**3, dtype=np.float32).reshape(16, 16, 16),
            label=np.pad(np.full((4, 4, 4), 6, dtype=np.uint8), 6),
            affine=np.eye(4, dtype=np.float64),
            cache_hit=True,
            source_fingerprint="1" * 64,
        )
        dataset = ExactSplitPatchDataset(
            project_root=self.project_root,
            dataset_root=self.project_root,
            manifest=manifest,
            schema=self.schema,
            preprocessing=self.preprocessing,
            preprocessing_sha256=self.config.preprocessing_sha256,
            config=self.config,
            split="train",
            augment=False,
            allow_cache_write=False,
            case_ids=("s_train",),
            fixed_sample_indices=(14, 29),
        )
        with patch("src.training.data.prepare_case", return_value=prepared):
            first = dataset[0]
            dataset.set_epoch(99)
            repeated = dataset[0]
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
        self.assertEqual(len(dataset), 2)
        self.assertEqual(first["sample_index"], 14)
        self.assertEqual(repeated["sample_index"], 14)
        self.assertTrue(torch.equal(first["image"], repeated["image"]))
        self.assertIsInstance(first["patch_trace"], dict)
        self.assertIsInstance(first["patch_trace"]["center"], list)
        self.assertEqual(tuple(batch["image"].shape), (2, 1, 96, 96, 96))
        self.assertEqual(batch["patch_trace"]["case_id"], ["s_train", "s_train"])

    def test_case_subset_is_ordered_strict_and_test_lock_stays_pre_io(self) -> None:
        train_manifest = {"cases": [self._case("a"), self._case("b"), self._case("c")]}
        dataset = ExactSplitPatchDataset(
            project_root=self.project_root,
            dataset_root=Path("Z:/not-resolved"),
            manifest=train_manifest,
            schema=self.schema,
            preprocessing=self.preprocessing,
            preprocessing_sha256=self.config.preprocessing_sha256,
            config=self.config,
            split="train",
            augment=False,
            allow_cache_write=False,
            case_ids=("c", "a"),
        )
        self.assertEqual(dataset.selected_case_ids, ("c", "a"))
        self.assertEqual(tuple(record["case_id"] for record in dataset.records), ("c", "a"))
        self.assertFalse(dataset.is_full_split)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            ExactSplitPatchDataset(
                project_root=self.project_root,
                dataset_root=self.project_root,
                manifest=train_manifest,
                schema=self.schema,
                preprocessing=self.preprocessing,
                preprocessing_sha256=self.config.preprocessing_sha256,
                config=self.config,
                split="train",
                augment=False,
                allow_cache_write=False,
                case_ids=("a", "a"),
            )
        poison = {"cases": [self._case("poison", "test")]}
        with patch("src.training.data.resolve_dataset_path") as resolver:
            with self.assertRaises(PermissionError):
                ExactSplitPatchDataset(
                    project_root=self.project_root,
                    dataset_root=Path("Z:/not-resolved"),
                    manifest=poison,
                    schema=self.schema,
                    preprocessing=self.preprocessing,
                    preprocessing_sha256=self.config.preprocessing_sha256,
                    config=self.config,
                    split="test",
                    augment=False,
                    allow_cache_write=False,
                    case_ids=("poison",),
                )
            resolver.assert_not_called()

    def test_validation_subset_is_strict_and_ordered(self) -> None:
        manifest = {"cases": [self._case("v1", "val"), self._case("v2", "val")]}
        dataset = ExactSplitVolumeDataset(
            project_root=self.project_root,
            dataset_root=self.project_root,
            manifest=manifest,
            schema=self.schema,
            preprocessing=self.preprocessing,
            preprocessing_sha256=self.config.preprocessing_sha256,
            config=self.config,
            split="val",
            allow_cache_write=False,
            case_ids=("v2",),
        )
        self.assertEqual(dataset.selected_case_ids, ("v2",))
        self.assertEqual(len(dataset), 1)
        with self.assertRaisesRegex(ValueError, "not members"):
            ExactSplitVolumeDataset(
                project_root=self.project_root,
                dataset_root=self.project_root,
                manifest=manifest,
                schema=self.schema,
                preprocessing=self.preprocessing,
                preprocessing_sha256=self.config.preprocessing_sha256,
                config=self.config,
                split="val",
                allow_cache_write=False,
                case_ids=("train_case",),
            )

    def test_deterministic_loader_builder_contract(self) -> None:
        train_loader, validation_loader = build_training_loaders(
            _TinyDataset(5), _TinyDataset(3), self.config, production=False
        )
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(validation_loader.sampler, SequentialSampler)
        self.assertEqual(train_loader.generator.initial_seed(), self.config.random_seed)
        self.assertEqual(validation_loader.generator.initial_seed(), self.config.random_seed + 1)
        self.assertEqual(train_loader.reproducibility_contract["role"], "train")
        self.assertTrue(train_loader.reproducibility_contract["shuffle"])
        self.assertEqual(validation_loader.reproducibility_contract["role"], "validation")
        self.assertFalse(validation_loader.reproducibility_contract["shuffle"])
        self.assertEqual(train_loader.num_workers, 0)
        self.assertFalse(train_loader.drop_last)
        self.assertEqual(len(next(iter(train_loader))["value"]), 1)

    def test_cache_generation_hashes_detect_corruption_and_rebuild(self) -> None:
        preprocessing = replace(self.preprocessing, spacing_mm=(1.0, 1.0, 1.0))
        case = self._case("cache_case")
        label = np.zeros((5, 6, 7), dtype=np.uint8)
        label[1:3, 2:4, 3:5] = 10
        loaded = SimpleNamespace(
            ct_hu=np.arange(label.size, dtype=np.float32).reshape(label.shape),
            label=label,
            affine=np.eye(4, dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.training.data.load_exclusive_case", return_value=loaded) as loader:
                first = prepare_case(
                    project_root=root,
                    dataset_root=root,
                    case=case,
                    schema=self.schema,
                    preprocessing=preprocessing,
                    preprocessing_sha256=self.config.preprocessing_sha256,
                    config=self.config,
                    allow_cache_write=True,
                    verify_cache_hashes=True,
                )
                second = prepare_case(
                    project_root=root,
                    dataset_root=root,
                    case=case,
                    schema=self.schema,
                    preprocessing=preprocessing,
                    preprocessing_sha256=self.config.preprocessing_sha256,
                    config=self.config,
                    allow_cache_write=True,
                    verify_cache_hashes=True,
                )
                self.assertEqual(loader.call_count, 1)
                self.assertFalse(first.cache_hit)
                self.assertTrue(second.cache_hit)
                del second
                gc.collect()
                case_dir = (
                    root
                    / self.config.cache_root
                    / self.config.preprocessing_sha256[:16]
                    / "cache_case"
                )
                metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["schema_version"], 2)
                self.assertEqual(set(metadata["arrays"]), {"ct_hu", "label", "affine"})
                generation = metadata["generation"]
                self.assertTrue(
                    all(record["path"].startswith(f"generations/{generation}/") for record in metadata["arrays"].values())
                )
                self.assertTrue(all(len(record["sha256"]) == 64 for record in metadata["arrays"].values()))
                label_path = case_dir / metadata["arrays"]["label"]["path"]
                with label_path.open("r+b") as handle:
                    handle.seek(-1, 2)
                    original = handle.read(1)
                    handle.seek(-1, 2)
                    handle.write(bytes([original[0] ^ 0xFF]))
                rebuilt = prepare_case(
                    project_root=root,
                    dataset_root=root,
                    case=case,
                    schema=self.schema,
                    preprocessing=preprocessing,
                    preprocessing_sha256=self.config.preprocessing_sha256,
                    config=self.config,
                    allow_cache_write=True,
                    verify_cache_hashes=True,
                )
                self.assertEqual(loader.call_count, 2)
                self.assertFalse(rebuilt.cache_hit)
                new_metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
                self.assertNotEqual(new_metadata["generation"], generation)

    def test_invalid_prepared_label_id_is_rejected(self) -> None:
        preprocessing = replace(self.preprocessing, spacing_mm=(1.0, 1.0, 1.0))
        invalid = SimpleNamespace(
            ct_hu=np.zeros((3, 3, 3), dtype=np.float32),
            label=np.full((3, 3, 3), self.config.out_channels, dtype=np.uint8),
            affine=np.eye(4, dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.training.data.load_exclusive_case", return_value=invalid
        ):
            with self.assertRaisesRegex(ValueError, "invalid class IDs"):
                prepare_case(
                    project_root=Path(directory),
                    dataset_root=Path(directory),
                    case=self._case("invalid"),
                    schema=self.schema,
                    preprocessing=preprocessing,
                    preprocessing_sha256=self.config.preprocessing_sha256,
                    config=self.config,
                    allow_cache_write=False,
                )

    def test_actual_source_hash_verification_and_prewarm_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "case").mkdir()
            ct_path = root / "case" / "ct.bin"
            mask_path = root / "case" / "mask.bin"
            ct_path.write_bytes(b"ct-source")
            mask_path.write_bytes(b"mask-source")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            case = {
                "case_id": "source_case",
                "split": "train",
                "ct": {"path": "case/ct.bin", "file_sha256": digest(ct_path)},
                "targets": {
                    "organ": {
                        "sources": [
                            {"name": "organ", "path": "case/mask.bin", "file_sha256": digest(mask_path)}
                        ]
                    }
                },
            }
            report = verify_case_source_hashes(root, case)
            self.assertEqual(report["files_checked"], 2)
            self.assertEqual(report["unique_files_checked"], 2)
            mask_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
                verify_case_source_hashes(root, case)
            poison = dict(case, split="test")
            with patch("src.training.data.resolve_dataset_path") as resolver:
                with self.assertRaises(PermissionError):
                    verify_case_source_hashes(Path("Z:/poison"), poison)
                resolver.assert_not_called()

        preprocessing = replace(self.preprocessing, spacing_mm=(1.0, 1.0, 1.0))
        prewarm_case = self._case("prewarm")
        loaded = SimpleNamespace(
            ct_hu=np.zeros((3, 4, 5), dtype=np.float32),
            label=np.zeros((3, 4, 5), dtype=np.uint8),
            affine=np.eye(4, dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.training.data.load_exclusive_case", return_value=loaded
        ):
            progress: list[dict[str, object]] = []
            result = prewarm_cache(
                project_root=Path(directory),
                dataset_root=Path(directory),
                manifest={"cases": [prewarm_case]},
                schema=self.schema,
                preprocessing=preprocessing,
                preprocessing_sha256=self.config.preprocessing_sha256,
                config=self.config,
                split="train",
                verify_source_hashes=False,
                progress=progress.append,
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["operation"], "cache_prewarm_only_no_model_execution")
            self.assertEqual(result["cases_completed"], 1)
            self.assertEqual(result["cache_misses_written"], 1)
            self.assertEqual(len(progress), 1)


if __name__ == "__main__":
    unittest.main()
