"""Train a native FTTransformer with one process or torchrun.

Install this project first. CPU example:
  torchrun --standalone --nproc-per-node=2 examples/train_ddp_tabular.py --device cpu
Resume through a new total epoch count:
  torchrun --standalone --nproc-per-node=2 examples/train_ddp_tabular.py \
      --device cpu --epochs 4 --resume runs/ddp_tabular/last_checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import torch_like as tl
from torch_like.evaluation import classification_metric_set
from torch_like.training import (
    DistributedEvalSampler,
    DistributedSamplerConfig,
    DistributedSession,
    Trainer,
    TrainerConfig,
    create_distributed_sampler,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/ddp_tabular"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with DistributedSession(device=args.device, timeout_seconds=120) as session:
        context = session.context
        # All ranks reconstruct identical ordered data; the sampler partitions it.
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(24, 3, generator=generator)
        y = (x[:, 0] + 0.3 * x[:, 1] > 0).long()
        vx = torch.randn(7, 3, generator=generator)
        vy = (vx[:, 0] + 0.3 * vx[:, 1] > 0).long()
        train_data, val_data = TensorDataset(x, y), TensorDataset(vx, vy)
        train_sampler = (
            create_distributed_sampler(train_data, context, DistributedSamplerConfig(seed=42))
            if context.is_distributed else None
        )
        val_sampler = DistributedEvalSampler(val_data, context) if context.is_distributed else None
        train_loader = DataLoader(
            train_data, batch_size=4, sampler=train_sampler, shuffle=train_sampler is None,
            num_workers=0, generator=torch.Generator().manual_seed(51 + context.rank),
        )
        val_loader = DataLoader(
            val_data, batch_size=3, sampler=val_sampler, num_workers=0,
            generator=torch.Generator().manual_seed(61 + context.rank),
        )
        torch.manual_seed(7 + context.rank)
        model = tl.FTTransformer(
            input_dim=3, output_dim=2, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.1,
        )
        current = Trainer(
            model,
            TrainerConfig(epochs=args.epochs, device=str(session.device), learning_rate=1e-3,
                          checkpoint_dir=args.out),
            distributed_context=context,
            metrics=classification_metric_set(),
            prediction_fn=lambda native, batch: native.predict(batch).argmax(dim=-1),
        )
        result = current.fit(train_loader, val_loader, resume_from=args.resume)
        if context.is_primary:
            tl.save_model(model, args.out / "model", overwrite=True)
            summary = {
                "version": tl.__version__, "world_size": context.world_size,
                "device": result.device, "validation_samples": len(val_data),
                "epochs_ran": result.epochs_ran, "start_epoch": result.start_epoch,
                "metrics": result.history[-1].validation_metrics,
                "last_checkpoint": result.last_checkpoint,
            }
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
