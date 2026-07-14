"""Runner for comparing forecasting models on a shared forecasting pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch_like as tl
from runner.common import build_run_payload, resolve_results_dir, save_json
from runner.compile_utils import CompileConfig
from runner.forecast_data import ForecastDataConfig, create_forecast_loaders
from runner.forecast_runner import evaluate, train_epoch


@dataclass
class ForecastComparisonConfig:
    data: ForecastDataConfig
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: CompileConfig = field(default_factory=CompileConfig)
    model_specs: list[dict] | None = None
    results_dir: str | Path | None = None


def default_model_specs(config: ForecastComparisonConfig, feature_dim: int, out_dim: int, adjacency=None):
    shared = {
        "enc_in": feature_dim,
        "dec_in": feature_dim,
        "c_out": out_dim,
        "seq_len": config.data.seq_len,
        "label_len": config.data.label_len,
        "pred_len": config.data.pred_len,
    }
    return [
        {
            "name": "NBEATS",
            "kwargs": {
                **shared,
                "d_model": 64,
                "e_layers": 3,
                "d_ff": 128,
                "dropout": 0.1,
                "stack_layers": 3,
            },
        },
        {
            "name": "PatchTST",
            "kwargs": {
                **shared,
                "d_model": 64,
                "n_heads": 4 if feature_dim % 4 == 0 else 1,
                "e_layers": 2,
                "d_ff": 128,
                "dropout": 0.1,
                "patch_len": max(2, min(16, config.data.seq_len // 4)),
            },
        },
        {
            "name": "TimeMixer",
            "kwargs": {
                **shared,
                "d_model": 64,
                "e_layers": 2,
                "d_ff": 128,
                "dropout": 0.1,
                "down_sampling_layers": 2,
                "moving_avg": 3,
            },
        },
        {
            "name": "TimesNet",
            "kwargs": {
                **shared,
                "d_model": 64,
                "e_layers": 2,
                "d_ff": 128,
                "dropout": 0.1,
                "top_k": 3,
                "num_kernels": 4,
            },
        },
        {
            "name": "Crossformer",
            "kwargs": {
                **shared,
                "d_model": 64,
                "n_heads": 4 if feature_dim % 4 == 0 else 1,
                "e_layers": 2,
                "d_ff": 128,
                "dropout": 0.1,
                "seg_len": max(2, min(6, config.data.seq_len // 3)),
            },
        },
        {
            "name": "MICN",
            "kwargs": {
                **shared,
                "d_model": 64,
                "e_layers": 2,
                "d_ff": 128,
                "dropout": 0.1,
                "moving_avg": 3,
                "conv_kernel_sizes": [3, 5, 7],
            },
        },
        {
            "name": "ASTGCN",
            "kwargs": {
                **shared,
                "d_model": 32,
                "e_layers": 2,
                "dropout": 0.1,
                "graph_mode": "prior" if adjacency is not None else "learned",
                "adjacency": adjacency,
                "adaptive_embedding_dim": 16,
            },
            "requires_graph": True,
        },
        {
            "name": "DMSTGCN",
            "kwargs": {
                **shared,
                "d_model": 32,
                "e_layers": 2,
                "dropout": 0.1,
                "graph_mode": "prior" if adjacency is not None else "learned",
                "adjacency": adjacency,
                "adaptive_embedding_dim": 16,
                "skip_channels": 32,
                "end_channels": 64,
            },
            "requires_graph": True,
        },
        {
            "name": "MTGNN",
            "kwargs": {
                **shared,
                "d_model": 32,
                "e_layers": 1,
                "d_ff": 64,
                "dropout": 0.1,
                "graph_mode": "prior" if adjacency is not None else "learned",
                "adjacency": adjacency,
                "gcn_depth": 2,
                "node_dim": 16,
                "conv_channels": 16,
                "residual_channels": 16,
                "skip_channels": 32,
                "end_channels": 64,
            },
            "requires_graph": True,
        },
        {
            "name": "StemGNN",
            "kwargs": {
                **shared,
                "d_model": 32,
                "e_layers": 2,
                "dropout": 0.1,
                "graph_mode": "prior" if adjacency is not None else "learned",
                "adjacency": adjacency,
                "latent_dim": 16,
                "stack_cnt": 2,
            },
            "requires_graph": True,
        },
        {
            "name": "GraphWaveNet",
            "kwargs": {
                **shared,
                "e_layers": 2,
                "dropout": 0.1,
                "graph_mode": "prior" if adjacency is not None else "learned",
                "adjacency": adjacency,
                "residual_channels": 16,
                "skip_channels": 32,
                "end_channels": 64,
                "adaptive_embedding_dim": 16,
            },
            "requires_graph": True,
        },
        {
            "name": "DCRNN",
            "kwargs": {
                **shared,
                "d_model": 32,
                "e_layers": 2,
                "dropout": 0.1,
                "graph_mode": "prior" if adjacency is not None else "learned",
                "adjacency": adjacency,
                "diffusion_steps": 2,
                "adaptive_embedding_dim": 16,
            },
            "requires_graph": True,
        },
    ]


def _should_skip(spec: dict, adjacency) -> bool:
    return bool(spec.get("requires_graph")) and adjacency is None


def run_experiment(spec: dict, train_loader, val_loader, test_loader, config: ForecastComparisonConfig):
    model_name = spec["name"]
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    model = tl.create_model(model_name, domain="forecasting", **spec["kwargs"]).to(config.device)
    from runner.compile_utils import maybe_compile_model  # local import to avoid circular noise at module import time

    model = maybe_compile_model(model, config.compile)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Train windows: {len(train_loader.dataset):,}")
    print(f"  Test windows:  {len(test_loader.dataset):,}")

    history = []
    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, config.data.label_len)
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, config.device, config.data.label_len)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            }
        )
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_rmse={val_rmse:.6f}"
        )

    test_loss, test_rmse, test_mae = evaluate(model, test_loader, criterion, config.device, config.data.label_len)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s | test_rmse={test_rmse:.6f} | test_mae={test_mae:.6f}")
    return {
        "model": model_name,
        "params": n_params,
        "train_time_s": round(elapsed, 1),
        "test_loss": test_loss,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "history": history,
    }


def main(config: ForecastComparisonConfig):
    bundle, train_loader, val_loader, test_loader = create_forecast_loaders(config.data, device=config.device)
    sample = next(iter(train_loader))
    if isinstance(sample, dict):
        feature_dim = sample["x_enc"].shape[-1]
        out_dim = sample["x_dec_target"].shape[-1]
    else:
        feature_dim = sample[0].shape[-1]
        out_dim = sample[1].shape[-1]
    adjacency = bundle.extra.get("adjacency")
    specs = config.model_specs or default_model_specs(
        config, feature_dim=feature_dim, out_dim=out_dim, adjacency=adjacency
    )

    results = []
    skipped = []
    for spec in specs:
        if _should_skip(spec, adjacency):
            skipped.append({"model": spec["name"], "reason": "graph adjacency unavailable"})
            continue
        results.append(run_experiment(spec, train_loader, val_loader, test_loader, config))

    print(f"\n{'=' * 60}")
    print("  Forecast Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Model':<16s} {'Params':>10s} {'RMSE':>12s} {'MAE':>12s} {'Time':>8s}")
    print(f"  {'-' * 64}")
    for r in results:
        print(
            f"  {r['model']:<16s} {r['params']:>10,} {r['test_rmse']:>12.6f} "
            f"{r['test_mae']:>12.6f} {r['train_time_s']:>7.1f}s"
        )
    if skipped:
        print("\n  Skipped models:")
        for item in skipped:
            print(f"  - {item['model']}: {item['reason']}")

    out_dir = resolve_results_dir(config.results_dir, Path(__file__).parent)
    out_path = out_dir / "forecast_comparison.json"
    clean = [{k: v for k, v in r.items() if k != "history"} for r in results]
    save_json(
        out_path,
        build_run_payload(
            config=config, history=[], extra={**bundle.extra, "task": "forecasting", "skipped": skipped}, results=clean
        ),
    )
    print(f"\n  Results saved to: {out_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run forecasting comparison experiments.")
    parser.add_argument("--dataset-type", default="ett_hour", choices=["ett_hour", "numpy", "graph_npy", "graph_csv"])
    parser.add_argument("--root-path")
    parser.add_argument("--data-path")
    parser.add_argument("--adjacency-path")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
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
    if args.dataset_type == "numpy":
        raise ValueError("CLI numpy mode is not supported; use the Python API to pass numpy_data.")
    config = ForecastComparisonConfig(
        data=ForecastDataConfig(
            dataset_type=args.dataset_type,
            root_path=args.root_path,
            data_path=args.data_path,
            adjacency_path=args.adjacency_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
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
