from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from src.dataset import load_organ_schema, load_preprocessing_spec
    from src.training import (
        ExactSplitPatchDataset,
        TrainingDisabledError,
        assert_training_authorized,
        build_loss,
        build_model,
        build_train_transform,
        load_baseline_config,
    )

    TRAINING_STACK_AVAILABLE = True
except ImportError:
    TRAINING_STACK_AVAILABLE = False


@unittest.skipUnless(TRAINING_STACK_AVAILABLE, "Phase 4 GPU dependencies are not installed")
class TrainingSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config_path = cls.project_root / "configs" / "baseline_unet_v1.yaml"
        cls.config = load_baseline_config(cls.config_path)
        cls.schema = load_organ_schema(cls.project_root / "configs" / "organs_v1.yaml")
        cls.preprocessing = load_preprocessing_spec(cls.project_root / "configs" / "dataset.yaml")

    def test_strict_config_and_model_loss_contract(self) -> None:
        self.assertFalse(self.config.training_enabled)
        self.assertEqual(self.config.execution_mode, "setup_only")
        self.assertEqual(self.config.roi_size, (96, 96, 96))
        self.assertEqual(self.config.out_channels, self.schema.num_output_channels)
        model = build_model(self.config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 4_817_434)
        criterion = build_loss(self.config)
        self.assertTrue(criterion.dice.to_onehot_y)
        self.assertTrue(criterion.dice.softmax)

    def test_unknown_config_key_is_rejected(self) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(text + "unexpected_key: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keys differ"):
                load_baseline_config(path)

    def test_test_split_is_rejected_before_any_path_resolution(self) -> None:
        poison_manifest = {
            "cases": [
                {
                    "case_id": "poison",
                    "split": "test",
                    "ct": {"path": "Z:/must-not-resolve/ct.nii.gz"},
                    "targets": {},
                }
            ]
        }
        with patch("src.dataset.loader.resolve_dataset_path") as resolver:
            with self.assertRaises(PermissionError):
                ExactSplitPatchDataset(
                    project_root=self.project_root,
                    dataset_root=Path("Z:/must-not-resolve"),
                    manifest=poison_manifest,
                    schema=self.schema,
                    preprocessing=self.preprocessing,
                    preprocessing_sha256=self.config.preprocessing_sha256,
                    config=self.config,
                    split="test",
                    augment=True,
                    allow_cache_write=False,
                )
            resolver.assert_not_called()

    def test_train_transform_is_seed_reproducible_and_preserves_label_contract(self) -> None:
        image = torch.linspace(0, 1, 96**3, dtype=torch.float32).reshape(1, 96, 96, 96)
        label = torch.zeros((1, 96, 96, 96), dtype=torch.int64)
        label[:, 30:60, 35:65, 40:70] = 6
        transform = build_train_transform(self.config)
        self.assertFalse(any("Flip" in type(item).__name__ for item in transform.transforms))
        transform.set_random_state(seed=4242)
        first = transform({"image": image.clone(), "label": label.clone()})
        transform.set_random_state(seed=4242)
        second = transform({"image": image.clone(), "label": label.clone()})
        self.assertTrue(torch.equal(first["image"], second["image"]))
        self.assertTrue(torch.equal(first["label"], second["label"]))
        self.assertEqual(first["image"].dtype, torch.float32)
        self.assertEqual(first["label"].dtype, torch.int64)
        self.assertTrue(set(torch.unique(first["label"]).tolist()).issubset({0, 6}))
        self.assertGreaterEqual(float(first["image"].min()), 0.0)
        self.assertLessEqual(float(first["image"].max()), 1.0)

    def test_training_requires_both_config_and_runtime_authorization(self) -> None:
        with self.assertRaises(TrainingDisabledError):
            assert_training_authorized(self.config, allow_training=False)
        with self.assertRaises(TrainingDisabledError):
            assert_training_authorized(self.config, allow_training=True)
        self.assertEqual(self.config.status, "configured_not_trained")

    def test_separately_versioned_training_config_can_be_authorized(self) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        text = text.replace("schema_version: 1", "schema_version: 2", 1)
        text = text.replace("status: configured_not_trained", "status: authorized_not_started", 1)
        text = text.replace("execution_mode: setup_only", "execution_mode: training", 1)
        text = text.replace("training_enabled: false", "training_enabled: true", 1)
        text = text.replace(
            "  automatic_mixed_precision: true\n",
            "  automatic_mixed_precision: true\n  amp_initial_scale: 1024.0\n",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_v2.yaml"
            path.write_text(text, encoding="utf-8")
            training_config = load_baseline_config(path)
        with self.assertRaises(TrainingDisabledError):
            assert_training_authorized(training_config, allow_training=False)
        assert_training_authorized(training_config, allow_training=True)


if __name__ == "__main__":
    unittest.main()
