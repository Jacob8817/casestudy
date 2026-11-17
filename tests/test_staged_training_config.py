from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.training.config import load_baseline_config
from src.training.gates import OverfitGateCriteria, overfit_scope_sha256
from src.training.workflow_config import AUTHORIZATION_TOKEN, GATE_SEQUENCE, load_training_workflow


class StagedTrainingConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.workflow_path = cls.project_root / "configs" / "training_workflow_v1.yaml"
        cls.workflow = load_training_workflow(cls.workflow_path)

    def test_workflow_is_authorized_sequential_and_test_locked(self) -> None:
        self.assertEqual(self.workflow.sequence, GATE_SEQUENCE)
        self.assertEqual(self.workflow.authorization_token, AUTHORIZATION_TOKEN)
        self.assertTrue(self.workflow.test_split_locked)
        self.assertEqual(self.workflow.stages["one_case_overfit"].train_case_ids, ("s0014",))
        self.assertEqual(self.workflow.stages["one_case_overfit"].validation_case_ids, ())
        self.assertEqual(len(self.workflow.stages["deterministic_pilot"].train_case_ids or ()), 16)
        self.assertEqual(len(self.workflow.stages["deterministic_pilot"].validation_case_ids or ()), 4)
        self.assertIn("s0099", self.workflow.stages["deterministic_pilot"].train_case_ids or ())
        self.assertIn("s0287", self.workflow.stages["deterministic_pilot"].train_case_ids or ())
        self.assertIn("s1161", self.workflow.stages["deterministic_pilot"].validation_case_ids or ())
        self.assertTrue(self.workflow.stages["deterministic_pilot"].acceptance["require_resume_event"])
        self.assertIsNone(self.workflow.stages["full_baseline"].train_case_ids)
        self.assertIsNone(self.workflow.stages["full_baseline"].validation_case_ids)

    def test_every_stage_baseline_is_separately_training_authorized(self) -> None:
        for name in GATE_SEQUENCE:
            stage = self.workflow.stages[name]
            config = load_baseline_config(self.project_root / stage.baseline_config)
            self.assertEqual(config.schema_version, 2)
            self.assertEqual(config.stage, name)
            self.assertEqual(config.execution_mode, "training")
            self.assertTrue(config.training_enabled)
            self.assertTrue(config.test_split_locked)
            self.assertTrue(config.notebook_cache_write_enabled)
            self.assertEqual(config.amp_initial_scale, 1024.0)
        full = load_baseline_config(
            self.project_root / self.workflow.stages["full_baseline"].baseline_config
        )
        self.assertEqual(full.max_epochs, 50)
        self.assertEqual(full.scheduler_t_max_epochs, 100)

    def test_overfit_bank_and_acceptance_are_fingerprinted(self) -> None:
        stage = self.workflow.stages["one_case_overfit"]
        criteria = OverfitGateCriteria.from_mapping(stage.acceptance)
        self.assertEqual(criteria.minimum_loss_reduction_fraction, 0.80)
        self.assertEqual(criteria.minimum_optimizer_steps, 1500)
        config = load_baseline_config(self.project_root / stage.baseline_config)
        self.assertEqual(config.max_epochs, 50)
        self.assertEqual(config.scheduler_t_max_epochs, 50)
        first = overfit_scope_sha256(
            case_id="s0014",
            fixed_sample_indices=stage.fixed_sample_indices,
            expected_selected_class_ids=stage.expected_selected_class_ids,
        )
        second = overfit_scope_sha256(
            case_id="s0014",
            fixed_sample_indices=stage.fixed_sample_indices,
            expected_selected_class_ids=stage.expected_selected_class_ids,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_unknown_workflow_key_is_rejected(self) -> None:
        text = self.workflow_path.read_text(encoding="utf-8") + "unexpected: true\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keys differ"):
                load_training_workflow(path)

    def test_wrong_authorization_token_is_rejected(self) -> None:
        text = self.workflow_path.read_text(encoding="utf-8").replace(
            AUTHORIZATION_TOKEN, "WRONG_TOKEN", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authorization token"):
                load_training_workflow(path)


if __name__ == "__main__":
    unittest.main()
