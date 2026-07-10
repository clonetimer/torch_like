"""Launch a forecasting comparison run on local XJTU-SY bearing CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from runner.compile_utils import CompileConfig
from runner.forecast_comparison_runner import ForecastComparisonConfig, main
from runner.forecast_data import ForecastDataConfig


def load_bearing_series(source_dir: Path, max_files: int = 4, downsample_step: int = 16) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(source_dir.glob("*.csv"), key=lambda p: int(p.stem))[:max_files]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {source_dir}")
    frames = [pd.read_csv(path).astype("float32") for path in files]
    data = pd.concat(frames, ignore_index=True).to_numpy(dtype=np.float32)
    data = data[::downsample_step]
    corr = np.corrcoef(data.T)
    adjacency = np.nan_to_num(np.abs(corr), nan=0.0).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    return data, adjacency, [str(path) for path in files]


def parse_args():
    parser = argparse.ArgumentParser(description="Run forecasting models on a directory of bearing CSV files.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--downsample-step", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args=None):
    args = parse_args() if args is None else args
    if args.max_files <= 0 or args.downsample_step <= 0 or args.epochs <= 0:
        raise ValueError("max_files, downsample_step, and epochs must be positive")
    source_dir = args.source_dir
    out_path = args.output or Path(__file__).parent / "results" / "forecast_comparison_bearing.json"
    data, adjacency, files_used = load_bearing_series(
        source_dir,
        max_files=args.max_files,
        downsample_step=args.downsample_step,
    )
    config = ForecastComparisonConfig(
        data=ForecastDataConfig(
            dataset_type="graph_npy",
            numpy_data=data,
            adjacency=adjacency,
            seq_len=128,
            label_len=64,
            pred_len=32,
            batch_size=32,
        ),
        epochs=args.epochs,
        device=args.device,
        results_dir=out_path.parent,
        compile=CompileConfig(enabled=False),
    )
    results = main(config)
    payload = {
        "source_dir": str(source_dir),
        "files_used": files_used,
        "rows_after_downsample": int(data.shape[0]),
        "feature_dim": int(data.shape[1]),
        "adjacency": adjacency.tolist(),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"Saved detailed bearing comparison to: {out_path}")


if __name__ == "__main__":
    run()
