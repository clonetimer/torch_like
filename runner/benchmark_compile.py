"""Compile compatibility and latency benchmark for selected forecasting models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch_like as tl
from runner.compile_utils import CompileConfig, maybe_compile_model
from runner.forecast_runner import _to_forecasting_batch


BENCHMARK_MODELS = {
    "PatchTST": {
        "kwargs": {
            "enc_in": 4,
            "dec_in": 4,
            "c_out": 4,
            "seq_len": 32,
            "label_len": 16,
            "pred_len": 8,
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 64,
            "dropout": 0.1,
            "patch_len": 4,
        },
        "batch": "standard",
    },
    "ITransformer": {
        "kwargs": {
            "enc_in": 4,
            "dec_in": 4,
            "c_out": 4,
            "seq_len": 32,
            "label_len": 16,
            "pred_len": 8,
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 64,
            "dropout": 0.1,
        },
        "batch": "standard",
    },
    "TimeMixer": {
        "kwargs": {
            "enc_in": 4,
            "dec_in": 4,
            "c_out": 4,
            "seq_len": 32,
            "label_len": 16,
            "pred_len": 8,
            "d_ff": 64,
            "dropout": 0.1,
            "moving_avg": 3,
            "down_sampling_layers": 2,
        },
        "batch": "standard",
    },
    "TimeXer": {
        "kwargs": {
            "enc_in": 4,
            "dec_in": 2,
            "c_out": 2,
            "seq_len": 32,
            "label_len": 16,
            "pred_len": 8,
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 64,
            "dropout": 0.1,
            "patch_len": 4,
        },
        "batch": "timexer",
    },
}


def build_batch(kind: str, batch_size: int, seq_len: int, label_len: int, pred_len: int, enc_in: int, c_out: int, device: str):
    if kind == "standard":
        x_enc = torch.randn(batch_size, seq_len, enc_in, device=device)
        x_dec_target = torch.randn(batch_size, label_len + pred_len, c_out, device=device)
        x_mark_enc = torch.zeros(batch_size, seq_len, 4, device=device)
        x_mark_dec = torch.zeros(batch_size, label_len + pred_len, 4, device=device)
        batch = {
            "x_enc": x_enc,
            "x_dec_target": x_dec_target,
            "x_mark_enc": x_mark_enc,
            "x_mark_dec": x_mark_dec,
            "x_target_enc": x_enc[:, :, :0],
            "x_exog_enc": x_enc[:, :, :0],
            "x_target_dec": x_dec_target[:, :, :0],
            "x_exog_dec": x_dec_target[:, :, :0],
        }
        return _to_forecasting_batch(batch, device, label_len=label_len)
    if kind == "timexer":
        x_enc = torch.randn(batch_size, seq_len, enc_in, device=device)
        x_dec_target = torch.randn(batch_size, label_len + pred_len, c_out, device=device)
        x_mark_enc = torch.zeros(batch_size, seq_len, 4, device=device)
        x_mark_dec = torch.zeros(batch_size, label_len + pred_len, 4, device=device)
        x_target_enc = x_enc[..., :c_out]
        x_exog_enc = x_enc[..., c_out:]
        x_target_dec = x_dec_target
        exog_dim = max(enc_in - c_out, 1)
        x_exog_dec = torch.randn(batch_size, label_len + pred_len, exog_dim, device=device)
        batch = {
            "x_enc": x_enc,
            "x_dec_target": x_dec_target,
            "x_mark_enc": x_mark_enc,
            "x_mark_dec": x_mark_dec,
            "x_target_enc": x_target_enc,
            "x_exog_enc": x_exog_enc,
            "x_target_dec": x_target_dec,
            "x_exog_dec": x_exog_dec,
        }
        return _to_forecasting_batch(batch, device, label_len=label_len)
    raise ValueError(f"Unsupported batch kind: {kind}")


@torch.inference_mode()
def run_latency(model, batch, warmup: int, iters: int, device: str):
    for _ in range(warmup):
        _ = model.predict(batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.predict(batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / max(iters, 1)


def benchmark_one(model_name: str, spec: dict, device: str, warmup: int, iters: int, compile_cfg: CompileConfig):
    model = tl.create_model(model_name, domain="forecasting", **spec["kwargs"]).to(device)
    batch = build_batch(
        spec["batch"],
        batch_size=8,
        seq_len=spec["kwargs"]["seq_len"],
        label_len=spec["kwargs"]["label_len"],
        pred_len=spec["kwargs"]["pred_len"],
        enc_in=spec["kwargs"]["enc_in"],
        c_out=spec["kwargs"]["c_out"],
        device=device,
    )
    eager_latency = run_latency(model, batch, warmup=warmup, iters=iters, device=device)
    result = {
        "model": model_name,
        "device": device,
        "eager_ms": round(eager_latency * 1000.0, 3),
        "compile_enabled": compile_cfg.enabled,
        "compile_ok": False,
        "compile_ms": None,
        "speedup": None,
        "error": None,
    }
    if not compile_cfg.enabled:
        return result
    try:
        compiled_model = maybe_compile_model(model, compile_cfg)
        compile_latency = run_latency(compiled_model, batch, warmup=warmup, iters=iters, device=device)
        result["compile_ok"] = True
        result["compile_ms"] = round(compile_latency * 1000.0, 3)
        result["speedup"] = round(eager_latency / compile_latency, 3) if compile_latency > 0 else None
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark torch.compile compatibility for forecasting models.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="*", default=list(BENCHMARK_MODELS.keys()))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend")
    parser.add_argument("--compile-mode")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).parent / "results" / "compile_benchmark.json"))
    return parser.parse_args()


def cli():
    args = parse_args()
    compile_cfg = CompileConfig(
        enabled=args.compile,
        backend=args.compile_backend,
        mode=args.compile_mode,
        fullgraph=args.compile_fullgraph,
        dynamic=True if args.compile_dynamic else None,
    )
    results = []
    for model_name in args.models:
        spec = BENCHMARK_MODELS.get(model_name)
        if spec is None:
            raise KeyError(f"Unsupported benchmark model: {model_name}. Available: {list(BENCHMARK_MODELS)}")
        result = benchmark_one(model_name, spec, args.device, args.warmup, args.iters, compile_cfg)
        results.append(result)
        compile_part = f", compile={result['compile_ms']} ms, speedup={result['speedup']}" if result["compile_ok"] else ""
        print(f"{model_name}: eager={result['eager_ms']} ms{compile_part}")
        if result["error"]:
            print(f"  error: {result['error']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {output}")


if __name__ == "__main__":
    cli()
