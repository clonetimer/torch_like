"""Public training-runtime regression tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import torch_like as tl
from torch_like.training import (
    DistributedContext,
    DistributedSamplerConfig,
    Trainer,
    TrainerConfig,
    load_training_checkpoint,
    create_distributed_sampler,
    training_events,
)


def _make_loader(x: torch.Tensor, y: torch.Tensor, *, seed: int, shuffle: bool = True) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=shuffle, generator=generator)


def _tabular_adapter(model):
    return lambda raw: model.make_tabular_batch(raw[0], y=raw[1])


class TrainingRuntimeTests(unittest.TestCase):
    def test_public_default_loss_training_and_events(self):
        torch.manual_seed(1)
        model = tl.FTTransformer(
            input_dim=3,
            output_dim=2,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            dropout=0.0,
        )
        x = torch.randn(16, 3)
        y = (x[:, 0] > 0).long()
        seen = []
        dispose = training_events.on(lambda name, **payload: seen.append((name, payload)))
        try:
            trainer = Trainer(
                model,
                TrainerConfig(epochs=2, device="cpu", learning_rate=1e-2, grad_clip_norm=1.0),
                batch_adapter=_tabular_adapter(model),
            )
            result = trainer.fit(_make_loader(x, y, seed=7), _make_loader(x, y, seed=8, shuffle=False))
        finally:
            dispose()

        self.assertEqual(result.epochs_ran, 2)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.monitor, "validation_loss")
        self.assertTrue(torch.isfinite(torch.tensor(result.best_monitored_loss)))
        names = [name for name, _ in seen]
        self.assertEqual(names[0], "training_start")
        self.assertEqual(names.count("epoch_end"), 2)
        self.assertEqual(names[-1], "training_end")

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        x = torch.tensor(
            [
                [-2.0, 0.0],
                [-1.5, 0.2],
                [-1.0, -0.1],
                [-0.5, 0.3],
                [0.5, -0.3],
                [1.0, 0.1],
                [1.5, -0.2],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
        )
        y = (x[:, 0] > 0).long()

        torch.manual_seed(123)
        continuous = tl.FTTransformer(
            input_dim=2,
            output_dim=2,
            d_model=4,
            n_heads=1,
            n_layers=1,
            d_ff=8,
            dropout=0.0,
        )
        Trainer(
            continuous,
            TrainerConfig(epochs=4, device="cpu", learning_rate=5e-3),
            batch_adapter=_tabular_adapter(continuous),
        ).fit(_make_loader(x, y, seed=99))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            torch.manual_seed(123)
            resumed = tl.FTTransformer(
                input_dim=2,
                output_dim=2,
                d_model=4,
                n_heads=1,
                n_layers=1,
                d_ff=8,
                dropout=0.0,
            )
            first_loader = _make_loader(x, y, seed=99)
            first = Trainer(
                resumed,
                TrainerConfig(
                    epochs=2,
                    device="cpu",
                    learning_rate=5e-3,
                    checkpoint_dir=checkpoint_dir,
                ),
                batch_adapter=_tabular_adapter(resumed),
            ).fit(first_loader)
            self.assertIsNotNone(first.last_checkpoint)

            second_loader = _make_loader(x, y, seed=99)
            second = Trainer(
                resumed,
                TrainerConfig(
                    epochs=4,
                    device="cpu",
                    learning_rate=5e-3,
                    checkpoint_dir=checkpoint_dir,
                ),
                batch_adapter=_tabular_adapter(resumed),
            ).fit(second_loader, resume_from=first.last_checkpoint)

            self.assertEqual(second.start_epoch, 2)
            self.assertEqual(second.epochs_ran, 4)
            for expected, actual in zip(continuous.parameters(), resumed.parameters(), strict=True):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_checkpoint_loader_can_restore_model_only(self):
        torch.manual_seed(2)
        model = tl.FTTransformer(
            input_dim=2,
            output_dim=1,
            d_model=4,
            n_heads=1,
            n_layers=1,
            d_ff=8,
            dropout=0.0,
        )
        x = torch.randn(8, 2)
        y = torch.randn(8, 1)
        with tempfile.TemporaryDirectory() as directory:
            result = Trainer(
                model,
                TrainerConfig(epochs=1, device="cpu", checkpoint_dir=directory),
                batch_adapter=_tabular_adapter(model),
            ).fit(_make_loader(x, y, seed=1))
            clone = tl.FTTransformer(
                input_dim=2,
                output_dim=1,
                d_model=4,
                n_heads=1,
                n_layers=1,
                d_ff=8,
                dropout=0.0,
            )
            payload = load_training_checkpoint(result.last_checkpoint, model=clone, restore_rng=False)
            self.assertEqual(payload["epoch"], 1)
            for expected, actual in zip(model.parameters(), clone.parameters(), strict=True):
                torch.testing.assert_close(expected, actual)

    def test_checkpoint_loader_accepts_legacy_schema_v1_as_single_process(self):
        model = tl.FTTransformer(
            input_dim=2, output_dim=1, d_model=4, n_heads=1, n_layers=1, d_ff=8, dropout=0.0
        )
        x = torch.randn(4, 2)
        y = torch.randn(4, 1)
        with tempfile.TemporaryDirectory() as directory:
            result = Trainer(
                model,
                TrainerConfig(epochs=1, device="cpu", checkpoint_dir=directory),
                batch_adapter=_tabular_adapter(model),
            ).fit(_make_loader(x, y, seed=2, shuffle=False))
            try:
                payload = torch.load(result.last_checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(result.last_checkpoint, map_location="cpu")
            payload["schema_version"] = 1
            payload.pop("distributed_context", None)
            payload.pop("rng_state_scope", None)
            legacy = Path(directory) / "legacy_v1.pt"
            torch.save(payload, legacy)
            clone = tl.FTTransformer(
                input_dim=2, output_dim=1, d_model=4, n_heads=1, n_layers=1, d_ff=8, dropout=0.0
            )
            loaded = load_training_checkpoint(legacy, model=clone, restore_rng=False)
            self.assertEqual(loaded["schema_version"], 1)

    def test_forecasting_requires_explicit_task_loss_adapter(self):
        model = tl.DLinear(enc_in=1, dec_in=1, c_out=1, seq_len=8, label_len=0, pred_len=2, moving_avg=3)
        with self.assertRaisesRegex(ValueError, "provide loss_fn"):
            Trainer(model, TrainerConfig(epochs=1, device="cpu"))

        x = torch.randn(8, 8, 1)
        y = torch.randn(8, 2, 1)

        def loss_fn(current_model, raw):
            inputs, targets = raw
            prediction = current_model.predict(current_model.make_forecast_batch(inputs))
            return torch.nn.functional.mse_loss(prediction, targets)

        result = Trainer(
            model,
            TrainerConfig(epochs=1, device="cpu"),
            loss_fn=loss_fn,
        ).fit(_make_loader(x, y, seed=4))
        self.assertEqual(result.epochs_ran, 1)

    def test_distributed_sampler_uses_pytorch_rank_partitioning(self):
        dataset = TensorDataset(torch.arange(8))
        config = DistributedSamplerConfig(shuffle=False)
        rank0 = create_distributed_sampler(dataset, DistributedContext(rank=0, world_size=2), config)
        rank1 = create_distributed_sampler(dataset, DistributedContext(rank=1, world_size=2), config)
        indices0 = list(iter(rank0))
        indices1 = list(iter(rank1))
        self.assertEqual(indices0, [0, 2, 4, 6])
        self.assertEqual(indices1, [1, 3, 5, 7])
        self.assertEqual(sorted(indices0 + indices1), list(range(8)))

    def test_distributed_context_requires_an_initialized_process_group(self):
        context = DistributedContext(rank=0, world_size=2, local_rank=0)
        self.assertTrue(context.is_distributed)
        self.assertTrue(context.is_primary)
        model = tl.FTTransformer(
            input_dim=2, output_dim=1, d_model=4, n_heads=1, n_layers=1, d_ff=8, dropout=0.0
        )
        with self.assertRaisesRegex(RuntimeError, "initialized process group"):
            Trainer(model, TrainerConfig(epochs=1, device="cpu"), distributed_context=context)

    def test_checkpoint_records_process_context_and_rejects_world_size_mismatch(self):
        torch.manual_seed(4)
        model = tl.FTTransformer(
            input_dim=2, output_dim=1, d_model=4, n_heads=1, n_layers=1, d_ff=8, dropout=0.0
        )
        x = torch.randn(8, 2)
        y = torch.randn(8, 1)
        with tempfile.TemporaryDirectory() as directory:
            result = Trainer(
                model,
                TrainerConfig(epochs=1, device="cpu", checkpoint_dir=directory),
                batch_adapter=_tabular_adapter(model),
            ).fit(_make_loader(x, y, seed=3))
            clone = tl.FTTransformer(
                input_dim=2, output_dim=1, d_model=4, n_heads=1, n_layers=1, d_ff=8, dropout=0.0
            )
            payload = load_training_checkpoint(result.last_checkpoint, model=clone, restore_rng=False)
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["distributed_context"], {"rank": 0, "world_size": 1, "local_rank": 0})
            self.assertEqual(payload["rng_state_scope"], "rank_local")
            with self.assertRaisesRegex(ValueError, "world_size"):
                load_training_checkpoint(
                    result.last_checkpoint,
                    model=clone,
                    restore_rng=False,
                    distributed_context=DistributedContext(rank=0, world_size=2, local_rank=0),
                )

    def test_sklearn_is_rejected_by_torch_trainer(self):
        estimator = tl.create("RandomForestClassifier", backend="sklearn", n_estimators=2, random_state=0)
        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            Trainer(estimator)


if __name__ == "__main__":
    unittest.main()
