"""Runner for comparing RUL models on XJTU-SY style datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch_like as tl
from runner.common import average_total, build_run_payload, resolve_results_dir, save_json
from runner.compile_utils import CompileConfig, maybe_compile_model
from runner.rul_data import RULDataConfig, create_rul_loaders


@dataclass
class RULRunConfig:
    data: RULDataConfig
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: CompileConfig = field(default_factory=CompileConfig)
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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return average_total(total_loss, len(loader.dataset))


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    mae = float(np.mean(np.abs(preds - labels)))
    return average_total(total_loss, len(loader.dataset)), rmse, mae


def run_experiment(model_cls, model_name, bundle, train_loader, test_loader, config: RULRunConfig):
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    model_kwargs = dict(config.model_shared)
    model_kwargs["seq_len"] = config.data.seq_len
    model = model_cls(**model_kwargs).to(config.device)
    model = maybe_compile_model(model, config.compile)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Train samples: {len(bundle.train_set):,}")
    print(f"  Test samples:  {len(bundle.test_set):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = {"epoch": [], "train_loss": [], "test_loss": [], "test_rmse": []}
    best_rmse = float("inf")
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        test_loss, rmse, mae = evaluate(model, test_loader, criterion, config.device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_rmse"].append(rmse)
        best_rmse = min(best_rmse, rmse)

        print(
            f"  Epoch {epoch:3d}/{config.epochs} | "
            f"train_loss={train_loss:.5f} | test_loss={test_loss:.5f} | RMSE={rmse:.4f}"
        )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s | Best RMSE: {best_rmse:.4f}")

    return {
        "model": model_name,
        "params": n_params,
        "best_rmse": best_rmse,
        "final_rmse": rmse,
        "final_mae": mae,
        "train_time_s": round(elapsed, 1),
        "history": history,
    }


def main(config: RULRunConfig):
    print(f"Device: {config.device}")
    print(f"Data:   {config.data.root_dir}/{config.data.condition}")
    print(
        f"Config: seq_len={config.data.seq_len}, stride={config.data.stride}, "
        f"batch={config.data.batch_size}, epochs={config.epochs}"
    )

    bundle, train_loader, test_loader = create_rul_loaders(config.data, device=config.device)
    model_specs = [
        (tl.InformerRUL, "InformerRUL"),
        (tl.AutoformerRUL, "AutoformerRUL"),
        (tl.LogTransRUL, "LogTransRUL"),
        (tl.THGNN, "THGNN"),
    ]

    results = []
    for cls, name in model_specs:
        results.append(run_experiment(cls, name, bundle, train_loader, test_loader, config))

    print(f"\n{'=' * 60}")
    print(f"  RUL Regression - XJTU-SY ({config.data.condition}) Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Model':<18s} {'Params':>10s} {'RMSE':>10s} {'MAE':>10s} {'Time':>8s}")
    print(f"  {'-' * 56}")
    for r in sorted(results, key=lambda x: x['best_rmse']):
        print(
            f"  {r['model']:<18s} {r['params']:>10,} "
            f"{r['best_rmse']:>10.4f} {r['final_mae']:>10.4f} {r['train_time_s']:>7.1f}s"
        )

    out_dir = resolve_results_dir(config.results_dir, Path(__file__).parent)
    out_path = out_dir / f"rul_comparison_{config.data.condition}.json"
    clean = [{k: v for k, v in r.items() if k != "history"} for r in results]
    save_json(out_path, build_run_payload(config=config, history=[], results=clean))
    print(f"\n  Results saved to: {out_path}")
    return results


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
    )
    main(config)


if __name__ == "__main__":
    cli()
