"""Demo: end-to-end verification of Informer model.

Creates random data, runs a forward pass, and checks output shapes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import torch_like as tl
from torch_like.models import create_model, model_zoo
from torch_like.utils.masking import triangular_causal_mask


def main():
    print("=" * 60)
    print("torch_like — Informer Demo")
    print("=" * 60)

    # ---- Config ----
    seq_len = 96
    label_len = 48
    pred_len = 24
    enc_in = 7
    dec_in = 7
    c_out = 7
    batch_size = 32
    d_model = 512
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048

    print(f"\nConfig: seq_len={seq_len}, label_len={label_len}, pred_len={pred_len}")
    print(f"        enc_in={enc_in}, d_model={d_model}, n_heads={n_heads}")
    print(f"        e_layers={e_layers}, d_layers={d_layers}, d_ff={d_ff}")

    # ---- Model ----
    model = tl.Informer(
        enc_in=enc_in,
        dec_in=dec_in,
        c_out=c_out,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        factor=5,
        dropout=0.05,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # ---- Random data ----
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)
    x_dec = torch.randn(batch_size, label_len + pred_len, dec_in)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4)

    # Triangular mask for decoder self-attention (auto-regressive)
    dec_self_mask = triangular_causal_mask((batch_size, label_len + pred_len), device=x_enc.device)

    print(f"\nEncoder input:  {list(x_enc.shape)}")
    print(f"Decoder input:  {list(x_dec.shape)}")

    # ---- Forward ----
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, dec_self_mask=dec_self_mask)

    print(f"Output:         {list(output.shape)}")
    assert output.shape == (batch_size, pred_len, c_out), (
        f"Expected {(batch_size, pred_len, c_out)}, got {output.shape}"
    )
    print("\n[OK] Forward pass successful, output shape matches expectation.")

    # ---- Model registry ----
    print("\n--- Model Registry ---")
    print(f"Forecasting models: {list(model_zoo['forecasting'].keys())}")
    created = create_model(
        "Informer",
        domain="forecasting",
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=96,
        label_len=48,
        pred_len=24,
    )
    print(f"Create by name:     {type(created).__name__}")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
