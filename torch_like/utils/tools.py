"""General-purpose training utilities."""

from pathlib import Path

import torch


def adjust_learning_rate(optimizer, epoch, args):
    """Cosine annealing learning rate schedule."""
    lr_adjust = {
        epoch: args.learning_rate
        if not hasattr(args, "lradj")
        else _adjust_fn(args.lradj, args.learning_rate, epoch, args)
    }
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def _adjust_fn(lradj, lr, epoch, args):
    """Learning rate adjustment based on strategy."""
    if lradj == "type1":
        return lr if epoch < 3 else lr * (0.9 ** ((epoch - 3) // 1))
    elif lradj == "type2":
        return lr * (0.5 ** ((epoch - 1) // 2))
    elif lradj == "cosine":
        import math

        eta_min = lr * (0.1**3)
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.train_epochs)) / 2
    return lr


class EarlyStopping:
    """Early stopping with patience and delta threshold."""

    def __init__(self, patience=7, verbose=False, delta=0.0):
        if patience <= 0:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class dotdict(dict):
    """Dictionary with attribute-style access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
