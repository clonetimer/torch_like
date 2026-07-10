"""Runner for comparing anomaly detection models on a shared detection pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch_like as tl
from runner.common import average_total, build_run_payload, print_epoch_metrics, resolve_results_dir, save_json
from runner.compile_utils import CompileConfig, maybe_compile_model
from runner.detection_data import DetectionDataConfig, create_detection_loaders


@dataclass
class DetectionRunConfig:
    data: DetectionDataConfig = field(default_factory=DetectionDataConfig)
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: CompileConfig = field(default_factory=CompileConfig)
    model_specs: list[dict] | None = None
    results_dir: str | Path | None = None


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        windows = batch[0] if isinstance(batch, (tuple, list)) else batch
        det_batch = model.make_detection_batch(windows.to(device))
        loss = model.fit_step(det_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * windows.size(0)
        count += windows.size(0)
    return average_total(total, count)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    labels = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            windows, batch_labels = batch
            labels.append(batch_labels.cpu())
        else:
            windows = batch
        det_batch = model.make_detection_batch(windows.to(device))
        scores = model.score_batch(det_batch, reduction="mean").cpu()
        losses.append(scores)
    result = {"scores": torch.cat(losses).numpy()}
    if labels:
        result["labels"] = torch.cat(labels).numpy()
    return result


def default_model_specs(config: DetectionRunConfig, input_dim: int):
    return [
        {
            "name": "RTDetector",
            "cls": tl.RTDetector,
            "kwargs": {
                "input_dim": input_dim,
                "seq_len": config.data.seq_len,
                "ff_dim": 16,
                "projector_hidden_dim": 16,
                "projector_hidden_layers": 1,
                "dropout": 0.1,
            },
        },
        {
            "name": "MEMTO",
            "cls": tl.MEMTO,
            "kwargs": {
                "input_dim": input_dim,
                "seq_len": config.data.seq_len,
                "d_model": 64,
                "n_heads": 4 if input_dim % 4 == 0 else 1,
                "e_layers": 1,
                "d_ff": 128,
                "dropout": 0.1,
                "mem_size": 16,
                "latent_dim": 32,
            },
        },
    ]


def run_experiment(model_cls, model_name: str, model_kwargs: dict, bundle, train_loader, test_loader, config: DetectionRunConfig):
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    model = model_cls(**model_kwargs).to(config.device)
    model = maybe_compile_model(model, config.compile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Train windows: {len(train_loader.dataset):,}")
    print(f"  Test windows:  {len(test_loader.dataset):,}")

    history = []
    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        history.append({"epoch": epoch, "train_loss": train_loss})
        print_epoch_metrics(epoch, config.epochs, train_loss=train_loss)

    eval_result = evaluate(model, test_loader, config.device)
    elapsed = time.time() - t0
    score_mean = float(np.mean(eval_result["scores"]))
    score_std = float(np.std(eval_result["scores"]))
    print(f"  Done in {elapsed:.1f}s | score_mean={score_mean:.6f} | score_std={score_std:.6f}")

    return {
        "model": model_name,
        "params": n_params,
        "train_time_s": round(elapsed, 1),
        "num_scores": int(len(eval_result["scores"])),
        "score_mean": score_mean,
        "score_std": score_std,
        "history": history,
        "has_labels": "labels" in eval_result,
    }


def main(config: DetectionRunConfig):
    bundle, train_loader, test_loader = create_detection_loaders(config.data, device=config.device)
    specs = config.model_specs or default_model_specs(config, input_dim=bundle.train_data.shape[1])

    results = []
    for spec in specs:
        results.append(
            run_experiment(
                spec["cls"],
                spec["name"],
                spec["kwargs"],
                bundle,
                train_loader,
                test_loader,
                config,
            )
        )

    print(f"\n{'=' * 60}")
    print("  Detection Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Model':<16s} {'Params':>10s} {'Scores':>8s} {'Mean':>12s} {'Std':>12s} {'Time':>8s}")
    print(f"  {'-' * 72}")
    for r in results:
        print(
            f"  {r['model']:<16s} {r['params']:>10,} {r['num_scores']:>8d} "
            f"{r['score_mean']:>12.6f} {r['score_std']:>12.6f} {r['train_time_s']:>7.1f}s"
        )

    out_dir = resolve_results_dir(config.results_dir, Path(__file__).parent)
    out_path = out_dir / "detection_comparison.json"
    clean = [{k: v for k, v in r.items() if k != "history"} for r in results]
    save_json(out_path, build_run_payload(config=config, history=[], extra=bundle.extra, results=clean))
    print(f"\n  Results saved to: {out_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly detection comparison experiments.")
    parser.add_argument("--data-root")
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
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
    config = DetectionRunConfig(
        data=DetectionDataConfig(
            data_root=args.data_root,
            train_path=args.train_path,
            test_path=args.test_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_train_rows=args.max_train_rows,
            max_test_rows=args.max_test_rows,
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
