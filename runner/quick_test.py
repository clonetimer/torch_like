#!/usr/bin/env python
"""Quick train/inference test for RUL regression on XJTU-SY."""

import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import time
import traceback as tb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
#  Config
# ------------------------------------------------------------------

DATA_ROOT = os.environ.get("TORCH_LIKE_RUL_DATA_ROOT")
CONDITION = os.environ.get("TORCH_LIKE_RUL_CONDITION", "35Hz12kN")
SEQ_LEN = 2048
STRIDE = 1024
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONF = dict(enc_in=2, seq_len=SEQ_LEN, d_model=64, n_heads=2, e_layers=1, d_ff=128, dropout=0.1)


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_epoch(model, loader, opt, criterion, device):
    model.train()
    loss_sum = 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        p = model(x)
        loss = criterion(p, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        if i % 80 == 0:
            log(f"  batch {i}: loss={loss.item():.5f}")
    return loss_sum / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        loss_sum += criterion(p, y).item() * x.size(0)
        preds.append(p.cpu().numpy())
        labels.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    mae = float(np.mean(np.abs(preds - labels)))
    return loss_sum / len(loader.dataset), rmse, mae


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------


def main():
    if not DATA_ROOT:
        raise ValueError("Set TORCH_LIKE_RUL_DATA_ROOT before running quick_test.py")
    log(f"Device: {DEVICE}")
    log(f"Data: {DATA_ROOT}/{CONDITION}")

    # ---- dataset -------------------------------------------------------
    log("Loading datasets...")
    from torch_like.data.dataset_rul import create_train_test_split

    train_set, test_set = create_train_test_split(
        DATA_ROOT,
        condition=CONDITION,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        train_ratio=0.6,
        seed=42,
    )
    log(f"  Train: {len(train_set):,} samples")
    log(f"  Test:  {len(test_set):,} samples")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- model ---------------------------------------------------------
    log("Creating models...")
    from torch_like.models.forecasting.rul_adapters import (
        AutoformerRUL,
        InformerRUL,
        LogTransRUL,
    )

    specs = [
        (InformerRUL, "InformerRUL"),
        (AutoformerRUL, "AutoformerRUL"),
        (LogTransRUL, "LogTransRUL"),
    ]

    results = []

    for cls, name in specs:
        log(f"\n{'=' * 55}")
        log(f"  {name}")
        log(f"{'=' * 55}")

        model = cls(**MODEL_CONF).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"  Params: {n_params:,}")

        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        t0 = time.time()
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_epoch(model, train_loader, opt, criterion, DEVICE)
            te_loss, rmse, mae = evaluate(model, test_loader, criterion, DEVICE)
            log(f"  Epoch {epoch:2d} | tr={tr_loss:.5f} | te={te_loss:.5f} | RMSE={rmse:.4f} | MAE={mae:.4f}")

        elapsed = time.time() - t0
        log(f"  Done in {elapsed:.1f}s")
        results.append(
            {
                "model": name,
                "params": n_params,
                "rmse": rmse,
                "mae": mae,
                "time_s": round(elapsed, 1),
            }
        )

    # ---- summary -------------------------------------------------------
    log(f"\n{'=' * 55}")
    log("  XJTU-SY RUL — 3-epoch quick comparison")
    log(f"{'=' * 55}")
    log(f"  {'Model':<18s} {'Params':>10s} {'RMSE':>8s} {'MAE':>8s} {'Time':>7s}")
    log(f"  {'-' * 54}")
    for r in sorted(results, key=lambda x: x["rmse"]):
        log(f"  {r['model']:<18s} {r['params']:>10,} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['time_s']:>6.1f}s")

    out = os.path.join(os.path.dirname(__file__), "results", "quick_test.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Results saved: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("ERROR:")
        tb.print_exc()
        raise
