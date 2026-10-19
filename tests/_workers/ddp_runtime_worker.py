"""Real torchrun worker used by test_ddp_runtime.py (never collected directly)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

# Executable from a source checkout; installed-wheel verification uses its own PYTHONPATH.
if not os.environ.get("TORCH_LIKE_INSTALLED_TEST"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch_like as tl
from torch_like.evaluation import ClassificationMetric, RegressionMetric, classification_metric_set
from torch_like.training import (
    DDPConfig,
    DistributedContext,
    DistributedEvalSampler,
    DistributedSamplerConfig,
    DistributedSession,
    IdentityBatchAdapter,
    Trainer,
    TrainerConfig,
    create_distributed_sampler,
    load_training_checkpoint,
    training_events,
)


class Tiny(torch.nn.Module):
    def __init__(self, dropout=0.0, buffers=False):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("seen_sum", torch.zeros(1))
        self.track_buffers = buffers

    def forward(self, x):
        if self.training and self.track_buffers:
            self.seen_sum.add_(x.detach().sum())
        return self.linear(self.dropout(x))

    def loss_step(self, batch):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def loaders(ctx, *, validation_size=7, shuffle=True, dataset_size=12):
    x = torch.arange(dataset_size * 2, dtype=torch.float32).reshape(-1, 2) / 20 - 0.5
    y = x[:, :1] * 0.7 - x[:, 1:] * 0.2
    train = TensorDataset(x, y)
    sampler = create_distributed_sampler(train, ctx, DistributedSamplerConfig(seed=41, shuffle=shuffle))
    train_loader = DataLoader(train, batch_size=3, sampler=sampler, generator=torch.Generator().manual_seed(14))
    vx = torch.arange(validation_size * 2, dtype=torch.float32).reshape(-1, 2) / 10
    vy = vx[:, :1] * 0.7 - vx[:, 1:] * 0.2
    val = TensorDataset(vx, vy)
    val_loader = DataLoader(
        val, batch_size=2, sampler=DistributedEvalSampler(val, ctx), generator=torch.Generator().manual_seed(17)
    )
    return train_loader, val_loader


def equal_parameters(model):
    vector = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
    values = [torch.empty_like(vector) for _ in range(dist.get_world_size())]
    dist.all_gather(values, vector)
    for value in values[1:]:
        torch.testing.assert_close(value, values[0], rtol=0, atol=0)


def metrics():
    return {key: RegressionMetric(key) for key in ("mae", "rmse", "r2")}


def trainer(model, ctx, *, epochs=2, directory=None, metric=True, **kwargs):
    return Trainer(
        model,
        TrainerConfig(epochs=epochs, device="cpu", checkpoint_dir=directory, learning_rate=0.02, **kwargs),
        distributed_context=ctx,
        batch_adapter=IdentityBatchAdapter(),
        metrics=metrics() if metric else None,
        prediction_fn=lambda m, b: m(b[0]),
    )


def basic(ctx, root):
    seed(30 + ctx.rank)
    model = Tiny()
    train, val = loaders(ctx)
    observed = []
    dispose = training_events.on(lambda name, **data: observed.append(name))
    try:
        current = trainer(model, ctx, directory=root / "checkpoint")
        result = current.fit(train, val)
    finally:
        dispose()
    equal_parameters(model)
    assert train.sampler.epoch == 1
    assert len(observed) == (4 if ctx.is_primary else 0)
    assert current.model is model
    assert current._execution.ddp is not None
    with torch.no_grad():
        x, y = val.dataset.tensors
        predicted = model(x)
        expected = ((predicted - y) ** 2).mean().item()
    assert abs(result.history[-1].validation_loss - expected) < 1e-7
    for name in ("mae", "rmse", "r2"):
        metric = RegressionMetric(name)
        metric.update(predicted, y)
        assert abs(result.history[-1].validation_metrics[name] - metric.compute()) < 1e-10
    payload = torch.load(result.last_checkpoint, weights_only=True)
    assert payload["schema_version"] == 3
    assert len(payload["rank_states"]) == 2
    assert payload["rng_state_scope"] == "per_rank"
    assert not any(key.startswith("module.") for key in payload["model_state_dict"])
    assert not list((root / "checkpoint").glob("*.tmp"))
    return {
        "parameters_synchronized": True,
        "global_metrics_exact": True,
        "val_samples": len(val.dataset),
        "schema": 3,
        "events": len(observed),
    }


def single_equivalence(ctx, root):
    seed(45)
    model = Tiny()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    train, _ = loaders(ctx, shuffle=False)
    current = Trainer(
        model,
        TrainerConfig(epochs=3, device="cpu", optimizer="sgd", learning_rate=0.01),
        distributed_context=ctx,
        batch_adapter=IdentityBatchAdapter(),
    )
    result = current.fit(train)
    # Global batch size is per-rank batch size * world_size, using the same sample union.
    expected = Tiny()
    expected.load_state_dict(initial)
    reference = Trainer(
        expected,
        TrainerConfig(epochs=3, device="cpu", optimizer="sgd", learning_rate=0.01),
        distributed_context=DistributedContext(),
        batch_adapter=IdentityBatchAdapter(),
    )
    loader = DataLoader(train.dataset, batch_size=6)
    reference.fit(loader)
    for a, b in zip(model.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    equal_parameters(model)
    return {"matches_single_global_batch": True, "epochs": result.epochs_ran}


def resume(ctx, root):
    def make(epochs, directory, offset):
        seed(offset + ctx.rank)
        model = Tiny(dropout=0.25, buffers=True)
        train, val = loaders(ctx)

        # Test Python/NumPy process RNG as well as Torch dropout and loader RNG.
        def loss(m, batch):
            x, y = batch
            noise = (random.random() + float(np.random.random())) * 0.02 if m.training else 0.0
            return torch.nn.functional.mse_loss(m(x) + noise, y)

        current = Trainer(
            model,
            TrainerConfig(epochs=epochs, device="cpu", learning_rate=0.01, checkpoint_dir=directory),
            distributed_context=ctx,
            batch_adapter=IdentityBatchAdapter(),
            loss_fn=loss,
            metrics=metrics(),
            prediction_fn=lambda m, b: m(b[0]),
        )
        return model, current, train, val

    full, full_trainer, full_train, full_val = make(4, None, 70)
    full_result = full_trainer.fit(full_train, full_val)
    part, part_trainer, part_train, part_val = make(2, root / "resume", 70)
    part_result = part_trainer.fit(part_train, part_val)
    resumed, resumed_trainer, resumed_train, resumed_val = make(4, root / "resume", 999)
    resumed_result = resumed_trainer.fit(resumed_train, resumed_val, resume_from=part_result.last_checkpoint)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    assert full_result.history == resumed_result.history
    assert resumed_result.start_epoch == 2
    equal_parameters(resumed)
    return {
        "bitwise_parameters": True,
        "history_identical": True,
        "resume_start_epoch": 2,
        "rng": ["python", "numpy", "torch", "train_loader", "val_loader"],
    }


def empty_validation(ctx, root):
    seed(27)
    model = Tiny()
    train, val = loaders(ctx, validation_size=1)
    result = trainer(model, ctx, epochs=1).fit(train, val)
    assert len(val) == (1 if ctx.rank == 0 else 0)
    assert np.isfinite(result.history[0].validation_metrics["rmse"])
    equal_parameters(model)
    return {"empty_rank_supported": True, "val_samples": 1}


def domain_losses(ctx, root):
    seed(58)
    model = tl.FTTransformer(input_dim=2, output_dim=2, d_model=4, n_heads=1, n_layers=1, d_ff=8, dropout=0.0)
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 4 - 1
    y = (x[:, 0] > 0).long()
    data = TensorDataset(x, y)
    train = DataLoader(data, batch_size=2, sampler=create_distributed_sampler(data, ctx))
    val = DataLoader(data, batch_size=3, sampler=DistributedEvalSampler(data, ctx))
    ft = Trainer(
        model,
        TrainerConfig(epochs=2, device="cpu"),
        distributed_context=ctx,
        metrics=classification_metric_set(),
        prediction_fn=lambda m, b: m.predict(b).argmax(-1),
    )
    result = ft.fit(train, val)
    equal_parameters(model)
    predictions = model.predict(model.make_tabular_batch(x)).argmax(-1)
    expected = ClassificationMetric("f1_macro")
    expected.update(predictions, y)
    assert abs(result.history[-1].validation_metrics["f1_macro"] - expected.compute()) < 1e-10
    forecast = tl.DLinear(enc_in=1, dec_in=1, c_out=1, seq_len=4, label_len=2, pred_len=2, moving_avg=3)
    xs, ys = torch.randn(8, 4, 1), torch.randn(8, 2, 1)
    ds = TensorDataset(xs, ys)
    dl = DataLoader(ds, batch_size=2, sampler=create_distributed_sampler(ds, ctx))
    train_result = Trainer(
        forecast,
        TrainerConfig(epochs=2, device="cpu"),
        distributed_context=ctx,
        step_fn=lambda m, b: torch.nn.functional.mse_loss(m.predict(b.model_batch), b.targets),
    ).fit(dl)
    equal_parameters(forecast)
    return {
        "FTTransformer_runtime_loss": True,
        "DLinear_step_fn": True,
        "f1_exact": True,
        "forecast_epochs": train_result.epochs_ran,
    }


def failure(ctx, root, kind):
    seed(5)
    model = Tiny()
    train, val = loaders(ctx)
    directory = None
    current = trainer(model, ctx, epochs=1)
    if kind == "nan":
        current.loss_fn = lambda m, b: m.loss_step(b) * (float("nan") if ctx.rank == 1 else 1)
    elif kind == "bad_adapter":

        class Bad:
            def adapt(self, model, batch):
                if ctx.rank == 1:
                    raise ValueError("injected adaptation fault")
                return IdentityBatchAdapter().adapt(model, batch)

        current.batch_adapter = Bad()
    elif kind == "bad_validation":

        def prediction(m, batch):
            if ctx.rank:
                raise ValueError("bad prediction")
            return m(batch[0])

        current.prediction_fn = prediction
    elif kind == "config_mismatch":
        current = trainer(model, ctx, epochs=1 + ctx.rank)
    elif kind == "wrong_train_sampler":
        train = DataLoader(train.dataset, batch_size=3)
    elif kind == "wrong_val_sampler":
        val = DataLoader(val.dataset, batch_size=2, sampler=create_distributed_sampler(val.dataset, ctx))
    elif kind == "buffered_metric":
        current = Trainer(
            model,
            TrainerConfig(epochs=1, device="cpu"),
            distributed_context=ctx,
            batch_adapter=IdentityBatchAdapter(),
            metrics={"bad": lambda y, p: float(np.mean(p - y))},
        )
    elif kind == "write_error":
        directory = root / "occupied"
        if ctx.is_primary:
            directory.write_text("not a directory")
        dist.barrier()
        current = trainer(model, ctx, directory=directory)
    elif kind == "validation_empty":
        train, val = loaders(ctx, validation_size=0)
    else:
        raise AssertionError(kind)
    caught = None
    try:
        current.fit(train, val)
    except (ValueError, RuntimeError) as exc:
        caught = f"{type(exc).__name__}: {exc}"
    assert caught is not None, kind
    reports = [None, None]
    dist.all_gather_object(reports, caught)
    assert all(reports), reports
    return {"failure": kind, "coordinated": True, "message": caught}


def early_stop(ctx, root):
    seed(3)
    model = Tiny()
    train, val = loaders(ctx)
    current = trainer(
        model, ctx, epochs=8, directory=root / "early", early_stopping_patience=1, min_delta=1e6, restore_best=True
    )
    result = current.fit(train, val)
    assert result.epochs_ran == 2
    assert result.stopped_early
    assert result.best_epoch == 1
    expected = Tiny()
    load_training_checkpoint(result.best_checkpoint, model=expected, distributed_context=ctx, restore_rng=False)
    for a, b in zip(model.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    equal_parameters(model)
    return {"epochs_ran": 2, "best_epoch": 1, "restored_best": True}


def lifecycle(ctx, root):
    with DistributedSession(ctx, device="cpu", backend="gloo") as borrowed:
        assert not borrowed.owns_process_group
    assert dist.is_initialized()
    try:
        with DistributedSession(DistributedContext(rank=1 - ctx.rank, world_size=2), device="cpu"):
            raise AssertionError("unexpected")
    except ValueError:
        pass
    assert dist.is_initialized()
    return {"borrowed_group_preserved": True, "mismatch_rejected": True}


def resume_guards(ctx, root):
    seed(66)
    model = Tiny()
    train, val = loaders(ctx)
    first = trainer(model, ctx, epochs=1, directory=root / "guard").fit(train, val)
    checkpoint = Path(first.last_checkpoint)
    train2, val2 = loaders(ctx)
    train2.sampler.seed += 1
    caught = False
    try:
        trainer(Tiny(), ctx, epochs=2).fit(train2, val2, resume_from=checkpoint)
    except RuntimeError as exc:
        assert "signature" in str(exc)
        caught = True
    assert caught
    with_missing_rank = torch.load(checkpoint, weights_only=True)
    with_missing_rank["rank_states"].pop()
    damaged = root / "missing_rank.pt"
    if ctx.is_primary:
        torch.save(with_missing_rank, damaged)
    dist.barrier()
    try:
        load_training_checkpoint(damaged, model=Tiny(), distributed_context=ctx)
    except ValueError as exc:
        assert "rank-local states" in str(exc)
    else:
        raise AssertionError("missing rank state was accepted")
    # A topology mismatch must not be reported as successful restoration.
    try:
        load_training_checkpoint(checkpoint, model=Tiny(), distributed_context=DistributedContext())
    except ValueError as exc:
        assert "world_size" in str(exc)
    else:
        raise AssertionError("world_size mismatch accepted")
    return {"changed_sampler_rejected": True, "missing_rank_rejected": True, "world_size_rejected": True}


def cosine_resume(ctx, root):
    import shutil

    seed(82 + ctx.rank)
    model = Tiny(dropout=0.25)
    train, val = loaders(ctx)
    current = trainer(model, ctx, epochs=4, directory=root / "cosine", lr_scheduler="cosine")
    original_save = current._save_checkpoint
    snapshot = root / "epoch2.pt"

    def save(path, **kwargs):
        original_save(path, **kwargs)
        if ctx.is_primary and path.name == "last_checkpoint.pt" and kwargs["epoch"] == 2:
            shutil.copyfile(path, snapshot)

    current._save_checkpoint = save
    result = current.fit(train, val)
    seed(985 + ctx.rank)
    clone = Tiny(dropout=0.25)
    train2, val2 = loaders(ctx)
    restored = trainer(clone, ctx, epochs=4, directory=root / "cosine_resumed", lr_scheduler="cosine")
    resumed = restored.fit(train2, val2, resume_from=snapshot)
    assert result.history == resumed.history
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, clone.state_dict()[name], rtol=0, atol=0)
    assert current.scheduler.state_dict() == restored.scheduler.state_dict()
    return {"cosine_state_restored": True, "bitwise_parameters": True, "history_identical": True}


def unused_parameters(ctx, root):
    seed(41)
    model = Tiny()
    model.extra = torch.nn.Linear(2, 1)
    train, _ = loaders(ctx)
    current = Trainer(
        model,
        TrainerConfig(epochs=2, device="cpu"),
        distributed_context=ctx,
        ddp_config=DDPConfig(find_unused_parameters=True),
        batch_adapter=IdentityBatchAdapter(),
    )
    current.fit(train)
    equal_parameters(model)
    assert model.extra.weight.grad is None
    return {"unused_parameters_supported_when_explicit": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ctx = DistributedContext.from_environment()
    args.root.mkdir(parents=True, exist_ok=True)
    with DistributedSession(ctx, device="cpu", timeout_seconds=30) as session:
        assert session.owns_process_group
        functions = {
            "basic": basic,
            "equivalence": single_equivalence,
            "resume": resume,
            "empty_validation": empty_validation,
            "domains": domain_losses,
            "early_stop": early_stop,
            "lifecycle": lifecycle,
            "resume_guards": resume_guards,
            "cosine_resume": cosine_resume,
            "unused_parameters": unused_parameters,
        }
        if args.scenario.startswith("fail_"):
            result = failure(ctx, args.root, args.scenario.removeprefix("fail_"))
        else:
            result = functions[args.scenario](ctx, args.root)
    assert not dist.is_initialized()
    report = {**result, "clean_exit": True, "version": tl.__version__, "import_path": tl.__file__}
    (args.root / f"rank{ctx.rank}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
