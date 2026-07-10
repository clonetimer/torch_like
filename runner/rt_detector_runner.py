"""Backward-compatible single-model RTDetector runner wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from runner.compile_utils import CompileConfig
from runner.detection_data import DetectionDataConfig
from runner.detection_runner import DetectionRunConfig, main as detection_main


def parse_args():
    parser = argparse.ArgumentParser(description="Run RTDetector anomaly detection experiments.")
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
        model_specs=None,
    )
    return detection_main(config)


if __name__ == "__main__":
    cli()
