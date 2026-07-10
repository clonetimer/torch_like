"""Demo: LogTrans — verify model builds and forward pass works."""

import torch
import sys
sys.path.insert(0, "..")

from torch_like.models.forecasting import LogTrans
from torch_like.nn.attention import log_sparse_mask

torch.manual_seed(123)

SEQ_LEN, LABEL_LEN, PRED_LEN = 96, 48, 24
BATCH, ENC_IN, DEC_IN, C_OUT = 4, 7, 7, 7

print("=" * 64)
print("  LogTrans Demo")
print("=" * 64)

model = LogTrans(
    enc_in=ENC_IN, dec_in=DEC_IN, c_out=C_OUT,
    seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN,
    d_model=256, n_heads=8, e_layers=2, d_layers=1,
    d_ff=512, dropout=0.05,
)
print(f"\nModel parameters: {model.param_count():,}")

x_enc = torch.randn(BATCH, SEQ_LEN, ENC_IN)
x_mark_enc = torch.zeros(BATCH, SEQ_LEN, 4)
x_dec = torch.randn(BATCH, LABEL_LEN + PRED_LEN, DEC_IN)
x_mark_dec = torch.zeros(BATCH, LABEL_LEN + PRED_LEN, 4)

model.eval()
with torch.no_grad():
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

print(f"Output: {tuple(out.shape)}  (expected: [{BATCH}, {PRED_LEN}, {C_OUT}])")

assert out.shape == (BATCH, PRED_LEN, C_OUT)
assert not torch.isnan(out).any()
assert not torch.isinf(out).any()

print(f"Output mean: {out.mean().item():.4f}  std: {out.std().item():.4f}")
print("\n[OK] LogTrans is working.\n")

# Show log-sparse mask pattern
mask = log_sparse_mask(16)
print("Log-sparse mask (L=16 first 8 rows):")
for i in range(8):
    row = "".join("■" if mask[i, j] else "·" for j in range(16))
    print(f"  t={i:2d}  {row}")
print(f"\nEach query attends to O(log L) ≈ {mask.sum(dim=1).float().mean().item():.1f} keys")
print("=" * 64)
