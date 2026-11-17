from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.engine import (
    NonFiniteTrainingError,
    TrainingCounters,
    TrainingScope,
    _atomic_json,
    _create_unique_run_directory,
    _load_resume_state,
    _loader_contract,
    _resolve_training_scope,
    _restore_rng_state,
    _rng_state,
    _seed_and_build_model,
    _train_epoch,
    _validate_run_id,
    _write_checkpoint,
)


class _IdentityScaler:
    def scale(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state != {}:
            raise ValueError("unexpected scaler state")


class _OverflowBackoffScaler(_IdentityScaler):
    def __init__(self) -> None:
        self.current_scale = 1024.0

    def step(self, _optimizer: torch.optim.Optimizer) -> None:
        # Emulate GradScaler's found-inf path: optimizer.step is skipped.
        return None

    def update(self) -> None:
        self.current_scale /= 2.0

    def get_scale(self) -> float:
        return self.current_scale


class _OneBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.records = ({"case_id": "s1"},)
        self.selected_case_ids = ("s1",)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.tensor([[1.0, -1.0]], dtype=torch.float32),
            "label": torch.zeros((1, 1), dtype=torch.int64),
        }


class _FiniteForwardNanBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = value.shape  # type: ignore[attr-defined]
        return value.sum()

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (
            torch.full(
                ctx.input_shape,  # type: ignore[attr-defined]
                math.nan,
                dtype=gradient.dtype,
                device=gradient.device,
            ),
        )


class _NanGradientLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        return _FiniteForwardNanBackward.apply(logits)


class _FiniteForwardInfBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = value.shape  # type: ignore[attr-defined]
        return value.sum()

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (
            torch.full(
                ctx.input_shape,  # type: ignore[attr-defined]
                math.inf,
                dtype=gradient.dtype,
                device=gradient.device,
            ),
        )


class _InfGradientLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        return _FiniteForwardInfBackward.apply(logits)


class _NaNLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        return logits.sum() * torch.tensor(math.nan, dtype=logits.dtype)


def _epoch_config() -> SimpleNamespace:
    return SimpleNamespace(
        automatic_mixed_precision=False,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
    )


def _checkpoint_config() -> SimpleNamespace:
    return SimpleNamespace(
        config_sha256="c" * 64,
        canonical_config={"schema_version": 2, "name": "synthetic"},
        max_epochs=2,
        experiment_fingerprint=lambda: "e" * 64,
    )


class TrainingEngineHardeningTests(unittest.TestCase):
    def test_scope_is_canonical_and_never_accepts_overfit(self) -> None:
        first = TrainingScope("deterministic_pilot", ("s2", "s1"), ("v1",))
        second = TrainingScope("deterministic_pilot", ("s1", "s2"), ("v1",))
        self.assertEqual(first.fingerprint(), second.fingerprint())
        with self.assertRaisesRegex(ValueError, "dedicated train-only"):
            TrainingScope("one_case_overfit", ("s1",), ("v1",))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            TrainingScope("deterministic_pilot", ("s1", "s1"), ("v1",))

    def test_scope_must_be_an_exact_official_split_subset(self) -> None:
        config = SimpleNamespace(train_split="train", validation_split="val")
        manifest = {
            "cases": [
                {"case_id": "s1", "split": "train"},
                {"case_id": "s2", "split": "train"},
                {"case_id": "v1", "split": "val"},
            ]
        }
        pilot = TrainingScope("deterministic_pilot", ("s1",), ("v1",))
        self.assertIs(_resolve_training_scope(config, manifest, pilot), pilot)
        with self.assertRaisesRegex(ValueError, "outside the official train"):
            _resolve_training_scope(
                config,
                manifest,
                TrainingScope("deterministic_pilot", ("unknown",), ("v1",)),
            )
        full = _resolve_training_scope(config, manifest, None)
        self.assertEqual(full.gate, "full_baseline")
        self.assertEqual(set(full.train_case_ids), {"s1", "s2"})

    def test_engine_seeds_before_internal_model_construction(self) -> None:
        config = SimpleNamespace(random_seed=73, strict_cuda_determinism=False)

        def factory(_config: object) -> torch.nn.Module:
            return torch.nn.Linear(3, 2)

        with patch("src.training.engine.torch.cuda.is_available", return_value=False):
            with patch("src.training.engine.build_model", side_effect=factory):
                first = _seed_and_build_model(config, None)
                first_state = {name: value.clone() for name, value in first.state_dict().items()}
                random.random()
                np.random.random()
                torch.rand(5)
                second = _seed_and_build_model(config, None)
        for name, value in second.state_dict().items():
            self.assertTrue(torch.equal(first_state[name], value))
        with self.assertRaisesRegex(ValueError, "pass model=None"):
            _seed_and_build_model(config, torch.nn.Linear(3, 2))

    def test_nonfinite_loss_blocks_backward_and_optimizer_step(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step = Mock(wraps=optimizer.step)  # type: ignore[method-assign]
        counters = TrainingCounters()
        with self.assertRaisesRegex(NonFiniteTrainingError, "Non-finite training loss"):
            _train_epoch(
                model=model,
                loader=DataLoader(_OneBatchDataset(), batch_size=1),
                criterion=_NaNLoss(),
                optimizer=optimizer,
                scaler=_IdentityScaler(),
                device=torch.device("cpu"),
                config=_epoch_config(),
                counters=counters,
            )
        self.assertEqual(counters.training_batches_consumed, 1)
        self.assertEqual(counters.loss_evaluations, 1)
        self.assertEqual(counters.backward_calls, 0)
        self.assertEqual(counters.optimizer_steps, 0)
        optimizer.step.assert_not_called()

    def test_nonfinite_gradient_blocks_optimizer_step(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step = Mock(wraps=optimizer.step)  # type: ignore[method-assign]
        counters = TrainingCounters()
        with self.assertRaisesRegex(NonFiniteTrainingError, "NaN gradients"):
            _train_epoch(
                model=model,
                loader=DataLoader(_OneBatchDataset(), batch_size=1),
                criterion=_NanGradientLoss(),
                optimizer=optimizer,
                scaler=_IdentityScaler(),
                device=torch.device("cpu"),
                config=_epoch_config(),
                counters=counters,
            )
        self.assertEqual(counters.backward_calls, 1)
        self.assertEqual(counters.gradient_checks, 1)
        self.assertEqual(counters.optimizer_steps, 0)
        optimizer.step.assert_not_called()

    def test_amp_infinite_gradient_backs_off_and_skips_optimizer_step(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step = Mock(wraps=optimizer.step)  # type: ignore[method-assign]
        scaler = _OverflowBackoffScaler()
        counters = TrainingCounters()
        config = _epoch_config()
        config.automatic_mixed_precision = True
        loss = _train_epoch(
            model=model,
            loader=DataLoader(_OneBatchDataset(), batch_size=1),
            criterion=_InfGradientLoss(),
            optimizer=optimizer,
            scaler=scaler,
            device=torch.device("cpu"),
            config=config,
            counters=counters,
        )
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(counters.amp_overflow_skips, 1)
        self.assertEqual(counters.optimizer_steps, 0)
        self.assertEqual(scaler.get_scale(), 512.0)
        optimizer.step.assert_not_called()

    def test_rng_capture_and_restore_is_exact_on_cpu(self) -> None:
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        with patch("src.training.engine.torch.cuda.is_available", return_value=False):
            state = _rng_state()
            expected = (random.random(), float(np.random.random()), torch.rand(3))
            random.seed(99)
            np.random.seed(99)
            torch.manual_seed(99)
            _restore_rng_state(state)
            actual = (random.random(), float(np.random.random()), torch.rand(3))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_loader_contract_rejects_implicit_or_wrong_rng(self) -> None:
        config = SimpleNamespace(
            batch_size=1,
            notebook_workers=0,
            production_workers=2,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
        )
        generator = torch.Generator().manual_seed(42)
        loader = DataLoader(
            _OneBatchDataset(),
            batch_size=1,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        loader.reproducibility_contract = {  # type: ignore[attr-defined]
            "schema_version": 1,
            "seed": 42,
            "role": "train",
            "shuffle": True,
            "worker_seeded": True,
        }
        contract = _loader_contract(loader, role="train", expected_seed=42, config=config)
        self.assertEqual(contract["sampler"], "RandomSampler")
        with self.assertRaisesRegex(ValueError, "generator seed"):
            _loader_contract(loader, role="train", expected_seed=43, config=config)

    def test_unique_run_directories_never_overwrite(self) -> None:
        config = SimpleNamespace(config_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "runs"
            first, first_id = _create_unique_run_directory(base, config, "pilot-001")
            self.assertEqual(first_id, "pilot-001")
            self.assertTrue(first.is_dir())
            with self.assertRaises(FileExistsError):
                _create_unique_run_directory(base, config, "pilot-001")
            automatic, automatic_id = _create_unique_run_directory(base, config, None)
            self.assertTrue(automatic.is_dir())
            self.assertRegex(automatic_id, r"^\d{8}T\d{6}Z-[a-f0-9]{8}-[a-f0-9]{8}$")
        for unsafe in ("../escape", "has/slash", "", ".hidden"):
            with self.assertRaises(ValueError):
                _validate_run_id(unsafe)

    def test_atomic_json_replaces_complete_document_and_sanitizes_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            _atomic_json(path, {"status": "running"})
            _atomic_json(path, {"status": "completed", "metric": math.nan})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"metric": None, "status": "completed"},
            )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_checkpoint_roundtrip_validates_contract_and_restores_rng(self) -> None:
        config = _checkpoint_config()
        scope = TrainingScope("deterministic_pilot", ("s1",), ("v1",))
        provenance = {"synthetic": True}
        provenance_sha256 = "p" * 64
        loader_contracts = {"train": {"seed": 42}, "validation": {"seed": 43}}
        train_loader = SimpleNamespace(generator=torch.Generator().manual_seed(42))
        validation_loader = SimpleNamespace(generator=torch.Generator().manual_seed(43))
        model = torch.nn.Linear(2, 1)
        original_model = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = _IdentityScaler()
        counters = TrainingCounters(epochs_completed=1, scheduler_steps=1)
        history = [{"epoch": 1, "training_loss": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            with patch("src.training.engine.torch.cuda.is_available", return_value=False):
                _write_checkpoint(
                    checkpoint,
                    kind="last",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=1,
                    best=0.25,
                    validations_without_improvement=0,
                    config=config,
                    provenance=provenance,
                    provenance_sha256=provenance_sha256,
                    scope=scope,
                    loader_contracts=loader_contracts,
                    run_id="pilot-001",
                    counters=counters,
                    history=history,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                expected_torch_state = payload["rng_state"]["torch_cpu"].clone()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(10)
                torch.manual_seed(999)
                start_epoch, best, patience, restored_counters, restored_history = _load_resume_state(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    provenance=provenance,
                    provenance_sha256=provenance_sha256,
                    scope=scope,
                    loader_contracts=loader_contracts,
                    run_id="pilot-001",
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                )
            self.assertEqual(start_epoch, 1)
            self.assertEqual(best, 0.25)
            self.assertEqual(patience, 0)
            self.assertEqual(restored_history, history)
            self.assertEqual(restored_counters.resume_events, 1)
            self.assertEqual(restored_counters.checkpoint_write_attempts, 1)
            self.assertEqual(restored_counters.checkpoints_written, 1)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_state))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original_model[name]))
            with patch("src.training.engine.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    _load_resume_state(
                        checkpoint,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        provenance={"synthetic": False},
                        provenance_sha256="wrong",
                        scope=scope,
                        loader_contracts=loader_contracts,
                        run_id="pilot-001",
                        train_loader=train_loader,
                        validation_loader=validation_loader,
                    )

    def test_epoch_zero_checkpoint_is_a_valid_first_epoch_recovery_point(self) -> None:
        config = _checkpoint_config()
        scope = TrainingScope("deterministic_pilot", ("s1",), ("v1",))
        provenance = {"synthetic": True}
        loader_contracts = {"train": {"seed": 42}, "validation": {"seed": 43}}
        train_loader = SimpleNamespace(generator=torch.Generator().manual_seed(42))
        validation_loader = SimpleNamespace(generator=torch.Generator().manual_seed(43))
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = _IdentityScaler()
        counters = TrainingCounters()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            with patch("src.training.engine.torch.cuda.is_available", return_value=False):
                _write_checkpoint(
                    checkpoint,
                    kind="last",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=0,
                    best=-math.inf,
                    validations_without_improvement=0,
                    config=config,
                    provenance=provenance,
                    provenance_sha256="p" * 64,
                    scope=scope,
                    loader_contracts=loader_contracts,
                    run_id="pilot-epoch-zero",
                    counters=counters,
                    history=[],
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                )
                start_epoch, best, patience, restored_counters, restored_history = _load_resume_state(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    provenance=provenance,
                    provenance_sha256="p" * 64,
                    scope=scope,
                    loader_contracts=loader_contracts,
                    run_id="pilot-epoch-zero",
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                )
        self.assertEqual(start_epoch, 0)
        self.assertEqual(best, -math.inf)
        self.assertEqual(patience, 0)
        self.assertEqual(restored_history, [])
        self.assertEqual(restored_counters.epochs_completed, 0)
        self.assertEqual(restored_counters.resume_events, 1)


if __name__ == "__main__":
    unittest.main()
