"""Runner for comparing RUL models on XJTU-SY style datasets."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch_like as tl
from runner.common import average_total, build_run_payload, resolve_results_dir, save_json
from runner.compile_utils import CompileConfig, maybe_compile_model
from runner.performance import (
    ExecutionConfig,
    add_execution_args,
    autocast_context,
    backward_and_step,
    create_grad_scaler,
    execution_config_from_args,
    move_to_device,
)
from runner.rul_data import RULDataConfig, create_rul_loaders
from torch_like.evaluation import MetricEvaluator, RegressionMetric


@dataclass
class RULRunConfig:
    data: RULDataConfig
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: CompileConfig = field(default_factory=CompileConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    results_dir: str | Path | None = None
    model_shared: dict = field(
        default_factory=lambda: {
            "enc_in": 2,
            "d_model": 128,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 512,
            "dropout": 0.1,
        }
    )


def train_epoch(model, loader, optimizer, criterion, device, execution: ExecutionConfig | None = None, scaler=None):
    execution = execution or ExecutionConfig()
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = move_to_device(x, device, execution)
        y = move_to_device(y, device, execution)
        with autocast_context(device, execution):
            pred = model(x)
            loss = criterion(pred, y)
        backward_and_step(loss, optimizer, scaler, execution)
        total_loss += loss.item() * x.size(0)
    return average_total(total_loss, len(loader.dataset))


@torch.inference_mode()
def evaluate(model, loader, criterion, device, execution: ExecutionConfig | None = None):
    execution = execution or ExecutionConfig()
    model.eval()
    total_loss = 0.0
    total_count = 0
    evaluator = MetricEvaluator(
        {
            "rmse": RegressionMetric("rmse"),
            "mae": RegressionMetric("mae"),
        }
    )
    for x, y in loader:
        x = move_to_device(x, device, execution)
        y = move_to_device(y, device, execution)
        with autocast_context(device, execution):
            pred = model(x)
            loss = criterion(pred, y)
        batch_size = int(x.size(0))
        total_loss += loss.item() * batch_size
        total_count += batch_size
        evaluator.update(pred, y)
    metrics = evaluator.compute()
    return average_total(total_loss, total_count), metrics["rmse"], metrics["mae"]


def _development_loaders(bundle, config: RULRunConfig):
    """Hold out complete training bearings for validation; never use sealed test data for selection."""
    dataset = bundle.train_set
    names = getattr(dataset, "bearing_names", None)
    index_map = getattr(dataset, "index_map", None)
    if names is None or index_map is None or len(names) < 2:
        raise ValueError(
            "RUL model comparison requires at least two identifiable development bearings; "
            "use the group-CV AutoML task for other dataset formats."
        )
    validation_name = random.Random(config.data.seed).choice(sorted(names))
    validation_bearing = list(names).index(validation_name)
    training_indices = [i for i, index in enumerate(index_map) if index[0] != validation_bearing]
    validation_indices = [i for i, index in enumerate(index_map) if index[0] == validation_bearing]
    if not training_indices or not validation_indices:
        raise ValueError("RUL bearing-level validation produced an empty training or validation dataset")
    kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": str(config.device).startswith("cuda"),
    }
    train_loader = DataLoader(Subset(dataset, training_indices), shuffle=True, **kwargs)
    validation_loader = DataLoader(Subset(dataset, validation_indices), shuffle=False, **kwargs)
    return train_loader, validation_loader, validation_name


def run_experiment(model_cls, model_name, bundle, train_loader, validation_loader, config: RULRunConfig):
    """Compare one model using development validation only; never inspect sealed test metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    model_kwargs = dict(config.model_shared)
    model_kwargs["seq_len"] = config.data.seq_len
    model = model_cls(**model_kwargs).to(config.device)
    model = maybe_compile_model(model, config.compile)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Validation samples: {len(validation_loader.dataset):,}")
    print(f"  Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = create_grad_scaler(config.device, config.execution)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = {"epoch": [], "train_loss": [], "validation_loss": [], "validation_rmse": []}
    best_validation_rmse = float("inf")
    best_validation_mae = float("inf")
    best_epoch = 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, config.execution, scaler)
        validation_loss, rmse, mae = evaluate(model, validation_loader, criterion, config.device, config.execution)
        scheduler.step()
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_rmse"].append(rmse)
        if rmse < best_validation_rmse:
            best_validation_rmse = rmse
            best_validation_mae = mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"  Epoch {epoch:3d}/{config.epochs} | "
            f"train_loss={train_loss:.5f} | val_loss={validation_loss:.5f} | val_rmse={rmse:.4f}"
        )

    if best_state is None:
        raise RuntimeError("RUL training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s | Best validation RMSE: {best_validation_rmse:.4f}")
    return {
        "model": model_name,
        "params": n_params,
        "best_validation_rmse": best_validation_rmse,
        "best_validation_mae": best_validation_mae,
        "best_epoch": best_epoch,
        "train_time_s": round(elapsed, 1),
        "history": history,
        "_trained_model": model,
    }


def main(config: RULRunConfig):
    print(f"Device: {config.device}")
    print(f"Data:   {config.data.root_dir}/{config.data.condition}")
    print(
        f"Config: seq_len={config.data.seq_len}, stride={config.data.stride}, "
        f"batch={config.data.batch_size}, epochs={config.epochs}"
    )

    bundle, _, test_loader = create_rul_loaders(config.data, device=config.device)
    train_loader, validation_loader, validation_bearing = _development_loaders(bundle, config)
    print(f"Validation bearing (development only): {validation_bearing}")
    model_specs = [
        (tl.InformerRUL, "InformerRUL"),
        (tl.AutoformerRUL, "AutoformerRUL"),
        (tl.LogTransRUL, "LogTransRUL"),
        (tl.THGNN, "THGNN"),
    ]

    results = [run_experiment(cls, name, bundle, train_loader, validation_loader, config) for cls, name in model_specs]
    ranked = sorted(results, key=lambda value: value["best_validation_rmse"])
    selected = ranked[0]
    test_loss, test_rmse, test_mae = evaluate(
        selected["_trained_model"], test_loader, nn.MSELoss(), config.device, config.execution
    )
    selected["sealed_test_rmse"] = test_rmse
    selected["sealed_test_mae"] = test_mae
    selected["sealed_test_loss"] = test_loss

    print(f"\n{'=' * 60}")
    print(f"  RUL Regression - XJTU-SY ({config.data.condition}) Validation Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Model':<18s} {'Params':>10s} {'Val RMSE':>10s} {'Val MAE':>10s} {'Time':>8s}")
    print(f"  {'-' * 65}")
    for item in ranked:
        print(
            f"  {item['model']:<18s} {item['params']:>10,} "
            f"{item['best_validation_rmse']:>10.4f} {item['best_validation_mae']:>10.4f} "
            f"{item['train_time_s']:>7.1f}s"
        )
    print(f"  Selected: {selected['model']} | Sealed test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    out_dir = resolve_results_dir(config.results_dir, Path(__file__).parent)
    out_path = out_dir / f"rul_comparison_{config.data.condition}.json"
    clean = [
        {key: value for key, value in item.items() if key not in {"history", "_trained_model"}} for item in results
    ]
    save_json(
        out_path,
        build_run_payload(
            config=config,
            history=[],
            results=clean,
            extra={
                "validation_bearing": validation_bearing,
                "ranking_metric": "best_validation_rmse",
                "selected_model": selected["model"],
                "sealed_test_evaluation_count": 1,
            },
        ),
    )
    print(f"\n  Results saved to: {out_path}")
    return clean


def parse_args():
    parser = argparse.ArgumentParser(description="Run RUL comparison experiments.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--condition", default="35Hz12kN")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend")
    parser.add_argument("--compile-mode")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    add_execution_args(parser)
    return parser.parse_args()


def cli():
    args = parse_args()
    config = RULRunConfig(
        data=RULDataConfig(
            root_dir=args.root_dir,
            condition=args.condition,
            seq_len=args.seq_len,
            stride=args.stride,
            batch_size=args.batch_size,
        ),
        epochs=args.epochs,
        device=args.device,
        results_dir=args.results_dir,
        compile=CompileConfig(
            enabled=args.compile,
            backend=args.compile_backend,
            mode=args.compile_mode,
            fullgraph=args.compile_fullgraph,
            dynamic=True if args.compile_dynamic else None,
        ),
        execution=execution_config_from_args(args),
    )
    main(config)


if __name__ == "__main__":
    cli()
