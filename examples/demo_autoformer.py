"""Demo: Autoformer — verify model builds and forward pass works.

Tests:
  1. Import and instantiate
  2. Dummy data forward pass
  3. Output shape validation
  4. Parameter count
"""

import sys

import torch

sys.path.insert(0, "..")

from torch_like.models.forecasting import Autoformer
from torch_like.utils.metrics import MAE, MSE

torch.manual_seed(42)

# --- Config ---
SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 24
BATCH = 4
ENC_IN = 7
DEC_IN = 7
C_OUT = 7

print("=" * 64)
print("  Autoformer Demo")
print("=" * 64)

# --- Build model ---
model = Autoformer(
    enc_in=ENC_IN,
    dec_in=DEC_IN,
    c_out=C_OUT,
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    d_model=256,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=512,
    moving_avg=25,
    factor=1,
    dropout=0.05,
)
print(f"\nModel parameters: {model.param_count():,}")
print(f"dtype: {next(model.parameters()).dtype}")

# --- Dummy input ---
x_enc = torch.randn(BATCH, SEQ_LEN, ENC_IN)
x_mark_enc = torch.zeros(BATCH, SEQ_LEN, 4)
x_dec = torch.randn(BATCH, LABEL_LEN + PRED_LEN, DEC_IN)
x_mark_dec = torch.zeros(BATCH, LABEL_LEN + PRED_LEN, 4)

# --- Forward ---
model.eval()
with torch.no_grad():
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

print(f"\nInput:          x_enc={tuple(x_enc.shape)}  x_dec={tuple(x_dec.shape)}")
print(f"Output:         {tuple(out.shape)}  (expected: [{BATCH}, {PRED_LEN}, {C_OUT}])")
print(f"Output range:   [{out.min().item():.4f}, {out.max().item():.4f}]")
print(f"Output mean:    {out.mean().item():.4f}")
print(f"Output std:     {out.std().item():.4f}")

# --- Loss sanity ---
target = torch.randn(BATCH, PRED_LEN, C_OUT)
mae = MAE(out.numpy(), target.numpy())
mse = MSE(out.numpy(), target.numpy())
print(f"\nMAE (vs random):  {mae:.4f}")
print(f"MSE (vs random):  {mse:.4f}")

# --- Shape tests ---
assert out.shape == (BATCH, PRED_LEN, C_OUT), f"Shape mismatch: {out.shape}"
assert not torch.isnan(out).any(), "NaN in output!"
assert not torch.isinf(out).any(), "Inf in output!"

print("\n[OK] All checks passed — Autoformer is working.\n")
print("=" * 64)
print("  Architecture summary:")
print("    - AutoCorrelation (FFT-based period attention)")
print("    - Progressive SeriesDecomp (moving_avg=25)")
print("    - Encoder: 2 layers (trend discarded)")
print("    - Decoder: 1 layer (trend accumulated)")
print("=" * 64)
