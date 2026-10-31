"""Train one synthetic batch and save/reload FITS and STGCN; not a benchmark.

Run from the source checkout: python -m examples.demo_workbuddy_forecasting
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

import torch_like as tl


def main() -> None:
    torch.manual_seed(17)
    torch.set_num_threads(1)
    common = dict(enc_in=3, dec_in=3, c_out=3, seq_len=24, label_len=0, pred_len=6, dropout=0.0)
    configs = {
        "FITS": {**common, "cut_freq": 5, "individual": False},
        "STGCN": {
            **common,
            "d_model": 8,
            "e_layers": 1,
            "bottleneck_channels": 4,
            # Known symmetric sensor topology, supplied by the caller.
            "adjacency": [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        },
    }
    t = torch.arange(30, dtype=torch.float32)
    signal = torch.stack([torch.sin(t / 3), torch.cos(t / 5), torch.sin(t / 3 + 0.4)], dim=-1)
    windows = signal.unsqueeze(0).repeat(4, 1, 1)
    x, target = windows[:, :24], windows[:, 24:]
    for name, config in configs.items():
        model = tl.create_model(name, domain="forecasting", **config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = model.make_forecast_batch(x)
        model.train()
        optimizer.zero_grad()
        loss = F.mse_loss(model.predict(batch), target)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = tl.as_runtime(model).predict(batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            tl.save_model(model, path)
            restored = tl.load_model(path).eval()
            with torch.inference_mode():
                actual = restored.predict(restored.make_forecast_batch(x))
            torch.testing.assert_close(prediction, actual)
        print(f"{name}: output={tuple(prediction.shape)}, finite={bool(torch.isfinite(prediction).all())}, reload=PASS")
    print("Synthetic API/optimizer smoke only; no real-data accuracy or SOTA claim.")


if __name__ == "__main__":
    main()
