"""Old checkpoint schemas and additive RNG/best-state correctness regressions."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_like.training import Trainer, TrainerConfig, load_training_checkpoint


def loss(model, batch):
    x, y = batch
    return torch.nn.functional.mse_loss(model(x), y)


def model_loader():
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    dataset = TensorDataset(torch.randn(6, 2), torch.randn(6, 1))
    loader = DataLoader(dataset, batch_size=3, generator=torch.Generator().manual_seed(65))
    return model, loader


@pytest.mark.parametrize("version", [1, 2])
def test_original_v1_v2_checkpoint_still_loads(version, tmp_path):
    model, loader = model_loader()
    result = Trainer(model, TrainerConfig(epochs=1, device="cpu", checkpoint_dir=tmp_path), loss_fn=loss).fit(loader)
    payload = torch.load(result.last_checkpoint, weights_only=True)
    payload["schema_version"] = version
    for key in ("python_rng_state", "numpy_rng_state", "cuda_rng_state", "val_loader_generator_state"):
        payload.pop(key, None)
    if version == 1:
        payload.pop("distributed_context")
        payload.pop("rng_state_scope")
    legacy = tmp_path / f"schema{version}.pt"
    torch.save(payload, legacy)
    clone, _ = model_loader()
    load_training_checkpoint(legacy, model=clone, train_loader=loader)
    for a, b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rng_restores_python_numpy_torch_and_validation_generator(tmp_path):
    random.seed(73)
    np.random.seed(73)
    model, loader = model_loader()
    val = DataLoader(loader.dataset, batch_size=2, generator=torch.Generator().manual_seed(76))
    current = Trainer(model, TrainerConfig(epochs=1, device="cpu", checkpoint_dir=tmp_path), loss_fn=loss)
    result = current.fit(loader, val)
    expected = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=val.generator))
    clone, _ = model_loader()
    random.seed(987)
    np.random.seed(543)
    load_training_checkpoint(result.last_checkpoint, model=clone, train_loader=loader, val_loader=val)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3, generator=val.generator), expected[3], rtol=0, atol=0)


def test_restore_best_without_persisting_best_uses_memory_not_last(tmp_path):
    expected, loader = model_loader()
    Trainer(expected, TrainerConfig(epochs=1, device="cpu"), loss_fn=loss).fit(loader)
    model, loader = model_loader()
    result = Trainer(
        model, TrainerConfig(epochs=3, device="cpu", checkpoint_dir=tmp_path, save_best=False,
                             restore_best=True, min_delta=1e6), loss_fn=loss,
    ).fit(loader)
    assert result.best_epoch == 1
    assert result.best_checkpoint is None
    for a, b in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
