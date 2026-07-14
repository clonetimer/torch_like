#!/usr/bin/env python
"""RUL comparison — 5 epochs, GPU-aware."""

import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import time
import traceback as tb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("TORCH_LIKE_RUL_DATA_ROOT")
CONDITION = os.environ.get("TORCH_LIKE_RUL_CONDITION", "35Hz12kN")
SEQ_LEN, STRIDE, BS, EPOCHS, LR = 2048, 1024, 16, 5, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MC = dict(enc_in=2, seq_len=SEQ_LEN, d_model=64, n_heads=2, e_layers=1, d_ff=128, dropout=0.1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_epoch(m, ld, opt, crit):
    m.train()
    loss_sum = 0.0
    for x, y in ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(m(x), y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    return loss_sum / max(1, len(ld))


@torch.no_grad()
def evaluate(m, ld, crit):
    m.eval()
    loss_sum = 0.0
    predictions, labels = [], []
    for x, y in ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        prediction = m(x)
        loss_sum += crit(prediction, y).item() * x.size(0)
        predictions.append(prediction.cpu().numpy())
        labels.append(y.cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    rmse = float(np.sqrt(np.mean((predictions - labels) ** 2)))
    mae = float(np.mean(np.abs(predictions - labels)))
    return loss_sum / len(ld.dataset), rmse, mae


def main():
    if not DATA_ROOT:
        raise ValueError("Set TORCH_LIKE_RUL_DATA_ROOT before running rul_test.py")
    log(f"Device={DEVICE}  Data={CONDITION}  Epochs={EPOCHS}")
    from torch_like.data.dataset_rul import create_train_test_split

    log("Loading data...")
    tr, te = create_train_test_split(
        DATA_ROOT, condition=CONDITION, seq_len=SEQ_LEN, stride=STRIDE, train_ratio=0.6, seed=42
    )
    log(f"Train={len(tr):,}  Test={len(te):,}")
    tr_ld = DataLoader(tr, BS, True, num_workers=0, drop_last=True)
    te_ld = DataLoader(te, BS, False, num_workers=0)

    from torch_like.models.forecasting.rul_adapters import AutoformerRUL, InformerRUL, LogTransRUL

    specs = [(InformerRUL, "InformerRUL"), (AutoformerRUL, "AutoformerRUL"), (LogTransRUL, "LogTransRUL")]
    results = []

    for cls, name in specs:
        log(f"\n── {name}")
        m = cls(**MC).to(DEVICE)
        log(f"Params={sum(p.numel() for p in m.parameters()):,}")
        crit = nn.MSELoss()
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        t0 = time.time()
        for ep in range(1, EPOCHS + 1):
            tr_l = train_epoch(m, tr_ld, opt, crit)
            te_l, rmse, mae = evaluate(m, te_ld, crit)
            log(f"  ep{ep:2d} tr={tr_l:.5f} te={te_l:.5f} RMSE={rmse:.4f} MAE={mae:.4f}")
        elapsed = time.time() - t0
        results.append(
            dict(
                model=name, params=sum(p.numel() for p in m.parameters()), rmse=rmse, mae=mae, time_s=round(elapsed, 1)
            )
        )
        log(f"  Done {elapsed:.0f}s")

    log(f"\n{'=' * 55}")
    log(f"  {'Model':<18s} {'Params':>10s} {'RMSE':>8s} {'MAE':>8s} {'Time':>7s}")
    log(f"  {'-' * 54}")
    for r in sorted(results, key=lambda x: x["rmse"]):
        log(f"  {r['model']:<18s} {r['params']:>10,} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['time_s']:>6.1f}s")
    out = os.path.join(os.path.dirname(__file__), "results", "rul_quick.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("ERROR:")
        tb.print_exc()
        raise
