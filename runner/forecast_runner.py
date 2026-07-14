"""Runner for forecasting models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch_like as tl
from runner.common import average_total, build_run_payload, print_epoch_metrics, resolve_results_dir, save_json
from runner.compile_utils import CompileConfig, maybe_compile_model
from runner.forecast_data import ForecastDataConfig, create_forecast_loaders
from runner.performance import (
    ExecutionConfig,
    add_execution_args,
    autocast_context,
    backward_and_step,
    create_grad_scaler,
    execution_config_from_args,
    move_to_device,
)
from torch_like.models.forecasting.common import ForecastingBatch


@dataclass
class ForecastRunConfig:
    data: ForecastDataConfig
    model_name: str = "Informer"
    domain: str = "forecasting"
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: CompileConfig = field(default_factory=CompileConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    results_dir: str | Path | None = None
    model_kwargs: dict = field(
        default_factory=lambda: {
            "d_model": 128,
            "n_heads": 4,
            "e_layers": 2,
            "d_layers": 1,
            "d_ff": 512,
            "dropout": 0.1,
        }
    )


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) == 4:
        x_enc, x_dec_target, x_mark_enc, x_mark_dec = batch
        return {
            "x_enc": x_enc,
            "x_dec_target": x_dec_target,
            "x_mark_enc": x_mark_enc,
            "x_mark_dec": x_mark_dec,
            "x_target_enc": None,
            "x_exog_enc": None,
            "x_target_dec": None,
            "x_exog_dec": None,
        }
    raise ValueError("Expected forecasting batch of 4 tensors")


def _make_decoder_input(x_dec_target, label_len: int):
    if not 0 <= label_len <= x_dec_target.shape[1]:
        raise ValueError(f"label_len must be in [0, {x_dec_target.shape[1]}]")
    known = x_dec_target[:, :label_len, :]
    future = torch.zeros_like(x_dec_target[:, label_len:, :])
    return torch.cat([known, future], dim=1)


def _move_optional(value, device, execution):
    return None if value is None else move_to_device(value, device, execution)


def _to_forecasting_batch(
    batch,
    device,
    label_len: int | None = None,
    execution: ExecutionConfig | None = None,
) -> ForecastingBatch:
    batch = _unpack_batch(batch)
    execution = execution or ExecutionConfig()
    x_dec_target = move_to_device(batch["x_dec_target"], device, execution)
    batch["x_dec_target"] = x_dec_target
    if label_len is None:
        raise ValueError("label_len is required when converting a forecasting batch")
    return ForecastingBatch(
        x_enc=move_to_device(batch["x_enc"], device, execution),
        x_mark_enc=move_to_device(batch["x_mark_enc"], device, execution),
        x_dec=_make_decoder_input(x_dec_target, label_len),
        x_mark_dec=move_to_device(batch["x_mark_dec"], device, execution),
        x_target_enc=_move_optional(batch["x_target_enc"], device, execution),
        x_exog_enc=_move_optional(batch["x_exog_enc"], device, execution),
        x_target_dec=_move_optional(batch["x_target_dec"], device, execution),
        x_exog_dec=_move_optional(batch["x_exog_dec"], device, execution),
    )


def _merge_model_kwargs(args) -> dict:
    model_kwargs = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
    }
    if args.model_name == "MTGNN":
        model_kwargs.update(
            {
                "graph_mode": args.graph_mode,
                "prior_weight": args.prior_weight,
                "gcn_depth": args.gcn_depth,
                "subgraph_size": args.subgraph_size,
                "node_dim": args.node_dim,
                "dilation_exponential": args.dilation_exponential,
                "conv_channels": args.conv_channels,
                "residual_channels": args.residual_channels,
                "skip_channels": args.skip_channels,
                "end_channels": args.end_channels,
                "prop_alpha": args.prop_alpha,
            }
        )
    if args.model_name == "GraphWaveNet":
        model_kwargs.update(
            {
                "graph_mode": args.graph_mode,
                "prior_weight": args.prior_weight,
                "residual_channels": args.residual_channels,
                "skip_channels": args.skip_channels,
                "end_channels": args.end_channels,
                "kernel_size": args.kernel_size,
                "adaptive_embedding_dim": args.adaptive_embedding_dim,
            }
        )
    if args.model_name == "DCRNN":
        model_kwargs.update(
            {
                "graph_mode": args.graph_mode,
                "prior_weight": args.prior_weight,
                "diffusion_steps": args.diffusion_steps,
                "adaptive_embedding_dim": args.adaptive_embedding_dim,
            }
        )
    if args.model_name == "StemGNN":
        model_kwargs.update(
            {
                "graph_mode": args.graph_mode,
                "prior_weight": args.prior_weight,
                "stack_cnt": args.stack_cnt,
                "latent_dim": args.latent_dim,
            }
        )
    if args.model_kwargs_json:
        model_kwargs.update(json.loads(args.model_kwargs_json))
    return model_kwargs


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    label_len: int,
    execution: ExecutionConfig | None = None,
    scaler=None,
):
    execution = execution or ExecutionConfig()
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch_data = _unpack_batch(batch)
        model_batch = _to_forecasting_batch(batch_data, device, label_len=label_len, execution=execution)
        x_enc = model_batch.x_enc
        x_dec_target = batch_data["x_dec_target"]
        with autocast_context(device, execution):
            pred = model.predict(model_batch)
            target = x_dec_target[:, -pred.shape[1] :, :]
            loss = criterion(pred, target)
        backward_and_step(loss, optimizer, scaler, execution)
        total_loss += loss.item() * x_enc.size(0)
        total_count += x_enc.size(0)
    return average_total(total_loss, total_count)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, label_len: int, execution: ExecutionConfig | None = None):
    execution = execution or ExecutionConfig()
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds, all_targets = [], []
    for batch in loader:
        batch_data = _unpack_batch(batch)
        model_batch = _to_forecasting_batch(batch_data, device, label_len=label_len, execution=execution)
        x_enc = model_batch.x_enc
        x_dec_target = batch_data["x_dec_target"]
        with autocast_context(device, execution):
            pred = model.predict(model_batch)
            target = x_dec_target[:, -pred.shape[1] :, :]
            loss = criterion(pred, target)
        total_loss += loss.item() * x_enc.size(0)
        total_count += x_enc.size(0)
        all_preds.append(pred.float().cpu().numpy())
        all_targets.append(target.float().cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    return average_total(total_loss, total_count), rmse, mae


def main(config: ForecastRunConfig):
    bundle, train_loader, val_loader, test_loader = create_forecast_loaders(config.data, device=config.device)
    sample_batch = _unpack_batch(next(iter(train_loader)))
    sample_x_enc = sample_batch["x_enc"]
    sample_x_dec_target = sample_batch["x_dec_target"]
    feature_dim = sample_x_enc.shape[-1]
    out_dim = sample_x_dec_target.shape[-1]
    dec_in = out_dim
    enc_in = feature_dim
    if config.model_name == "TFT" and sample_batch.get("x_target_enc") is not None:
        mark_enc_dim = sample_batch["x_mark_enc"].shape[-1]
        mark_dec_dim = sample_batch["x_mark_dec"].shape[-1]
        target_dim = sample_batch["x_target_enc"].shape[-1]
        exog_enc_dim = sample_batch["x_exog_enc"].shape[-1]
        exog_dec_dim = sample_batch["x_exog_dec"].shape[-1]
        enc_in = target_dim + exog_enc_dim + mark_enc_dim
        dec_in = exog_dec_dim + mark_dec_dim
    model_kwargs = dict(
        enc_in=enc_in,
        dec_in=dec_in,
        c_out=out_dim,
        seq_len=config.data.seq_len,
        label_len=config.data.label_len,
        pred_len=config.data.pred_len,
        **config.model_kwargs,
    )
    if bundle.extra.get("adjacency") is not None and "adjacency" not in model_kwargs:
        model_kwargs["adjacency"] = bundle.extra["adjacency"]
    model = tl.create_model(config.model_name, domain=config.domain, **model_kwargs).to(config.device)
    model = maybe_compile_model(model, config.compile)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = create_grad_scaler(config.device, config.execution)

    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.device,
            config.data.label_len,
            config.execution,
            scaler,
        )
        val_loss, val_rmse, val_mae = evaluate(
            model, val_loader, criterion, config.device, config.data.label_len, config.execution
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            }
        )
        print_epoch_metrics(epoch, config.epochs, train_loss=train_loss, val_loss=val_loss, val_rmse=val_rmse)

    test_loss, test_rmse, test_mae = evaluate(
        model, test_loader, criterion, config.device, config.data.label_len, config.execution
    )
    out_dir = resolve_results_dir(config.results_dir, Path(__file__).parent)
    save_json(
        out_dir / f"forecast_{config.model_name.lower()}_run.json",
        build_run_payload(
            config=config,
            history=history,
            extra=bundle.extra,
            metrics={"test": {"loss": test_loss, "rmse": test_rmse, "mae": test_mae}},
        ),
    )
    return model, history, {"loss": test_loss, "rmse": test_rmse, "mae": test_mae}


def parse_args():
    parser = argparse.ArgumentParser(description="Run forecasting experiments.")
    parser.add_argument("--dataset-type", default="ett_hour", choices=["ett_hour", "numpy", "graph_npy", "graph_csv"])
    parser.add_argument("--root-path")
    parser.add_argument("--data-path")
    parser.add_argument("--model-name", default="Informer")
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
    add_execution_args(parser)
    parser.add_argument("--adjacency-path")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model-kwargs-json")
    parser.add_argument("--graph-mode", default="learned", choices=["learned", "prior", "hybrid"])
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--gcn-depth", type=int, default=2)
    parser.add_argument("--subgraph-size", type=int)
    parser.add_argument("--node-dim", type=int, default=32)
    parser.add_argument("--dilation-exponential", type=int, default=2)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--residual-channels", type=int, default=32)
    parser.add_argument("--skip-channels", type=int, default=64)
    parser.add_argument("--end-channels", type=int, default=128)
    parser.add_argument("--prop-alpha", type=float, default=0.05)
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--adaptive-embedding-dim", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--stack-cnt", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=32)
    return parser.parse_args()


def cli():
    args = parse_args()
    if args.dataset_type == "numpy":
        raise ValueError("CLI numpy mode is not supported; use the Python API to pass numpy_data.")
    config = ForecastRunConfig(
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
        model_name=args.model_name,
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
        model_kwargs=_merge_model_kwargs(args),
    )
    main(config)


if __name__ == "__main__":
    cli()
