"""Single-process and DDP training runtime shared by public API and AutoML."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from torch_like.evaluation import Metric, MetricEvaluator
from torch_like.runtime import as_runtime
from torch_like.training.adapters import AdaptedBatch, BatchAdapter, resolve_batch_adapter
from torch_like.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from torch_like.training.config import TrainerConfig
from torch_like.training.data import move_to_device
from torch_like.training.distributed import (
    DDPConfig, DistributedContext, _distributed_device, _require_process_group,
)
from torch_like.training.events import training_events
from torch_like.training.state import EpochResult, TrainingResult

LossFunction = Callable[[Any, Any], torch.Tensor]
StepFunction = Callable[[Any, AdaptedBatch], torch.Tensor]
MetricFunction = Callable[[Any, Any], torch.Tensor | float]
PredictionFunction = Callable[[Any, Any], Any]


class Trainer:
    """Train a native ``torch.nn.Module`` without replacing its task API.

    By default the trainer delegates loss computation to
    ``as_runtime(model).loss(batch)``.  Domains whose public contract does not
    define ``loss`` (for example point forecasting) can supply ``loss_fn`` and
    ``batch_adapter`` while still reusing optimization/checkpoint lifecycle.

    A multi-process context requires an initialized process group. DDP 0.9
    supports one optimizer, mean scalar losses, fixed equal training steps,
    and exact unpadded validation shards; see docs/DDP_RUNTIME.md.
    """

    def __init__(
        self,
        model: Any,
        config: TrainerConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        loss_fn: LossFunction | None = None,
        step_fn: StepFunction | None = None,
        batch_adapter: BatchAdapter | Callable[[Any], Any] | None = None,
        metric_fns: Mapping[str, MetricFunction] | None = None,
        metrics: Mapping[str, Metric | Callable[[Any, Any], float]] | None = None,
        prediction_fn: PredictionFunction | None = None,
        event_context: Mapping[str, Any] | None = None,
        distributed_context: DistributedContext | None = None,
        ddp_config: DDPConfig | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Trainer requires a torch.nn.Module; sklearn estimators should use their native fit API")
        self.model = model
        self.config = config or TrainerConfig()
        self.distributed_context = distributed_context or DistributedContext.from_environment()
        if self.distributed_context.is_distributed:
            _require_process_group(self.distributed_context)
            self.device = _distributed_device(self.config.device, self.distributed_context)
        else:
            self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)
        self.runtime = as_runtime(model)
        if loss_fn is not None and step_fn is not None:
            raise ValueError("loss_fn and step_fn are mutually exclusive")
        if loss_fn is None and step_fn is None and not self.runtime.supports("loss"):
            raise ValueError(
                f"{self.runtime.spec.name} does not advertise a standard loss operation; "
                "provide loss_fn(model, batch) or step_fn(model, adapted_batch) for this task"
            )
        self.loss_fn = loss_fn
        self.step_fn = step_fn
        self.batch_adapter = resolve_batch_adapter(
            model,
            batch_adapter,
            auto=(loss_fn is None or step_fn is not None) and batch_adapter is None,
        )
        self.metric_fns = dict(metric_fns or {})
        self.metric_evaluator = MetricEvaluator(metrics) if metrics else None
        if self.metric_evaluator is not None and set(self.metric_fns) & set(self.metric_evaluator.metrics):
            overlap = sorted(set(self.metric_fns) & set(self.metric_evaluator.metrics))
            raise ValueError(f"metric_fns and metrics contain duplicate names: {overlap}")
        self.prediction_fn = prediction_fn
        self.event_context = dict(event_context or {})
        self.optimizer = optimizer or self._build_optimizer()
        self.scheduler = scheduler if scheduler is not None else self._build_scheduler()
        self._amp_enabled = bool(self.config.amp and self.device.type == "cuda")
        self.scaler = self._make_scaler()
        self._execution = None
        self._val_loader = None
        if self.distributed_context.is_distributed:
            from torch_like.training._ddp import _DDPExecution

            self._execution = _DDPExecution(self, ddp_config or DDPConfig())

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
        return device

    def _build_optimizer(self) -> torch.optim.Optimizer:
        parameters = self.model.parameters()
        kwargs = {"lr": self.config.learning_rate, "weight_decay": self.config.weight_decay}
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(parameters, **kwargs)
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(parameters, **kwargs)
        return torch.optim.Adam(parameters, **kwargs)

    def _build_scheduler(self) -> Any:
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        if self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2, factor=0.5)
        return None

    def _make_scaler(self) -> Any:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
            except TypeError:  # pragma: no cover - torch 2.0/2.1 signature
                return torch.amp.GradScaler(enabled=self._amp_enabled)
        return torch.cuda.amp.GradScaler(enabled=self._amp_enabled)  # pragma: no cover

    def _autocast(self):
        if not self._amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _prepare_batch(self, raw_batch: Any) -> AdaptedBatch:
        raw_batch = move_to_device(raw_batch, self.device, non_blocking=self.device.type == "cuda")
        adapted = self.batch_adapter.adapt(self.model, raw_batch)
        model_batch = move_to_device(adapted.model_batch, self.device, non_blocking=self.device.type == "cuda")
        targets = move_to_device(adapted.targets, self.device, non_blocking=self.device.type == "cuda")
        return AdaptedBatch(
            model_batch=model_batch,
            targets=targets,
            sample_count=adapted.sample_count,
            raw_batch=raw_batch,
        )

    def _guard(self, stage: str, function: Callable[[], Any]) -> Any:
        if self._execution is not None:
            return self._execution.guard(stage, function)
        return function()

    def _emit(self, name: str, **payload: Any) -> None:
        def emit():
            if self.distributed_context.is_primary:
                training_events.emit(name, **payload)

        self._guard(f"event {name}", emit)

    def _loss(self, batch: AdaptedBatch) -> torch.Tensor:
        if self._execution is not None and self.model.training and torch.is_grad_enabled():
            return self._execution.loss(batch)
        return self._local_loss(batch)

    def _local_loss(self, batch: AdaptedBatch) -> torch.Tensor:
        model_batch = batch.model_batch
        if self.step_fn is not None:
            loss = self.step_fn(self.model, batch)
        elif self.loss_fn is not None:
            loss = self.loss_fn(self.model, model_batch)
        else:
            loss = self.runtime.loss(model_batch)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError("step_fn/loss_fn/runtime.loss must return a scalar torch.Tensor")
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN or infinite loss")
        return loss

    def _run_train_epoch(self, train_loader: Iterable[Any]) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        max_gradient_norm = 0.0
        batches = (
            self._execution.batches(train_loader) if self._execution is not None
            else (self._prepare_batch(raw) for raw in train_loader)
        )
        for batch in batches:
            batch_size = batch.resolved_sample_count()
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self._loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm if self.config.grad_clip_norm is not None else float("inf"),
            )
            error = None if torch.isfinite(gradient_norm) else FloatingPointError("NaN or infinite gradient norm")
            if self._execution is not None:
                self._execution.check_error("gradient norm", error)
            elif error is not None:
                raise error
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))

            def optimizer_step():
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self._guard("optimizer step", optimizer_step)
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
        if self._execution is not None:
            total_loss, total_samples = self._execution.sum_values([total_loss, total_samples])
            max_gradient_norm = self._execution.maximum(max_gradient_norm)
        if total_samples == 0:
            raise ValueError("train_loader produced no samples")
        return total_loss / total_samples, max_gradient_norm

    def _run_validation(self, loader: Iterable[Any]) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metric_totals = {name: 0.0 for name in self.metric_fns}
        if self.metric_evaluator is not None:
            self.metric_evaluator.reset()
        if self._execution is not None:
            self._execution.sync_buffers()
        error = None
        try:
            with torch.inference_mode():
                for raw_batch in loader:
                    batch = self._prepare_batch(raw_batch)
                    batch_size = batch.resolved_sample_count()
                    with self._autocast():
                        loss = self._loss(batch)
                        metric_values = {
                            name: fn(self.model, batch.model_batch) for name, fn in self.metric_fns.items()
                        }
                        if self.metric_evaluator is not None:
                            if batch.targets is None:
                                raise ValueError(
                                    "dataset-level metrics require targets from the batch adapter; "
                                    "provide an explicit BatchAdapter when targets are ambiguous"
                                )
                            prediction = (
                                self.prediction_fn(self.model, batch.model_batch)
                                if self.prediction_fn is not None
                                else self.runtime.predict(batch.model_batch)
                            )
                            self.metric_evaluator.update(prediction, batch.targets)
                    total_loss += float(loss) * batch_size
                    total_samples += batch_size
                    for name, value in metric_values.items():
                        scalar = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                        metric_totals[name] += scalar * batch_size
        except Exception as exc:
            error = exc
        if self._execution is not None:
            self._execution.check_error("validation", error)
            total_loss, total_samples = self._execution.sum_values([total_loss, total_samples])
        elif error is not None:
            raise error
        if total_samples == 0:
            raise ValueError("validation loader produced no samples")
        values = {name: value / total_samples for name, value in metric_totals.items()}
        if self.metric_evaluator is not None:
            values.update(
                self._execution.merge_metrics() if self._execution is not None else self.metric_evaluator.compute()
            )
        return total_loss / total_samples, values

    def _checkpoint_paths(self, resume_from: str | Path | None) -> tuple[Path | None, Path | None]:
        directory = self.config.checkpoint_dir
        if directory is None and resume_from is not None:
            directory = Path(resume_from).expanduser().resolve().parent
        if directory is None:
            return None, None
        root = Path(directory).expanduser().resolve()
        self._guard(
            "checkpoint directory",
            lambda: root.mkdir(parents=True, exist_ok=True) if self.distributed_context.is_primary else None,
        )
        return root / "last_checkpoint.pt", root / "best_checkpoint.pt"

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        history: list[EpochResult],
        best_loss: float,
        best_epoch: int,
        epochs_without_improvement: int,
        train_loader: Any,
    ) -> None:
        save = self._execution.save_checkpoint if self._execution is not None else save_training_checkpoint
        extra = {} if self._execution is not None else {"val_loader": self._val_loader}
        save(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            history=[item.to_dict() for item in history],
            best_monitored_loss=best_loss,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            trainer_config=self.config.to_dict(),
            train_loader=train_loader,
            distributed_context=self.distributed_context,
            **extra,
        )

    def _load_checkpoint(self, path: Any, **kwargs: Any) -> dict[str, Any]:
        return self._guard("checkpoint load", lambda: load_training_checkpoint(path, **kwargs))

    def fit(
        self,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any] | None = None,
        *,
        resume_from: str | Path | None = None,
    ) -> TrainingResult:
        """Train through the configured total epoch count."""

        self._val_loader = val_loader
        if self._execution is not None:
            self._execution.setup(train_loader, val_loader, resume_from)
        history: list[EpochResult] = []
        start_epoch = 0
        best_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        resume_path = str(Path(resume_from).expanduser().resolve()) if resume_from is not None else None
        if resume_from is not None:
            payload = self._load_checkpoint(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                train_loader=train_loader,
                val_loader=val_loader,
                map_location="cpu",
                distributed_context=self.distributed_context,
            )
            start_epoch = int(payload["epoch"])
            history = [EpochResult(**item) for item in payload.get("history", [])]
            best_loss = float(payload.get("best_monitored_loss", float("inf")))
            best_epoch = int(payload.get("best_epoch", 0))
            epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
            if start_epoch > self.config.epochs:
                raise ValueError(
                    f"checkpoint already completed epoch {start_epoch}, beyond configured epochs={self.config.epochs}"
                )

        last_path, best_path = self._checkpoint_paths(resume_from)
        best_state: dict[str, torch.Tensor] | None = None
        stopped_early = False
        start_event = {
            "model": self.runtime.spec.name,
            "backend": self.runtime.spec.backend.value,
            "task_type": self.runtime.spec.task.kind.value,
            "device": str(self.device),
            "epochs": self.config.epochs,
            "start_epoch": start_epoch,
            "resumed_from": resume_path,
            "rank": self.distributed_context.rank,
            "world_size": self.distributed_context.world_size,
        }
        start_event.update(self.event_context)
        self._emit("training_start", **start_event)

        for epoch_index in range(start_epoch, self.config.epochs):
            if self._execution is not None:
                train_loader.sampler.set_epoch(epoch_index)
            train_loss, max_gradient_norm = self._run_train_epoch(train_loader)
            validation_loss = None
            validation_metrics: dict[str, float] = {}
            if val_loader is not None:
                validation_loss, validation_metrics = self._run_validation(val_loader)
            monitored_loss = train_loss if validation_loss is None else validation_loss
            improved = monitored_loss < best_loss - self.config.min_delta
            if improved:
                best_loss = monitored_loss
                best_epoch = epoch_index + 1
                epochs_without_improvement = 0
                if self.config.restore_best and (best_path is None or not self.config.save_best):
                    best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if self.scheduler is not None:
                def scheduler_step():
                    if self.config.lr_scheduler == "plateau":
                        self.scheduler.step(monitored_loss)
                    else:
                        self.scheduler.step()

                self._guard("scheduler step", scheduler_step)

            epoch_result = EpochResult(
                epoch=epoch_index + 1,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                max_gradient_norm=max_gradient_norm,
                improved=improved,
                validation_metrics=validation_metrics,
            )
            history.append(epoch_result)
            epoch_event = epoch_result.to_dict()
            epoch_event.update(self.event_context)
            epoch_event.update(rank=self.distributed_context.rank, world_size=self.distributed_context.world_size)
            self._emit("epoch_end", **epoch_event)

            if improved and best_path is not None and self.config.save_best:
                self._save_checkpoint(
                    best_path,
                    epoch=epoch_index + 1,
                    history=history,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                    train_loader=train_loader,
                )
            if last_path is not None and self.config.save_last:
                self._save_checkpoint(
                    last_path,
                    epoch=epoch_index + 1,
                    history=history,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                    train_loader=train_loader,
                )
            patience = self.config.early_stopping_patience
            if patience is not None and epochs_without_improvement >= patience:
                stopped_early = True
                break

        if self.config.restore_best:
            if self.config.save_best and best_path is not None and best_path.is_file():
                self._load_checkpoint(
                    best_path,
                    model=self.model,
                    map_location="cpu",
                    restore_rng=False,
                    distributed_context=self.distributed_context,
                )
            elif best_state is not None:
                self.model.load_state_dict(best_state)

        result = TrainingResult(
            epochs_requested=self.config.epochs,
            epochs_ran=len(history),
            start_epoch=start_epoch,
            best_epoch=best_epoch,
            best_monitored_loss=best_loss,
            stopped_early=stopped_early,
            monitor="train_loss" if val_loader is None else "validation_loss",
            device=str(self.device),
            history=tuple(history),
            last_checkpoint=str(last_path) if last_path is not None and last_path.is_file() else None,
            best_checkpoint=str(best_path) if best_path is not None and best_path.is_file() else None,
            resumed_from=resume_path,
        )
        end_event = {"summary": result.to_dict(), "model": self.runtime.spec.name}
        end_event.update(self.event_context)
        self._emit("training_end", **end_event)
        return result


def train_model(
    model: Any,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any] | None = None,
    *,
    config: TrainerConfig | None = None,
    resume_from: str | Path | None = None,
    **trainer_kwargs: Any,
) -> TrainingResult:
    """Convenience wrapper around :class:`Trainer`."""

    return Trainer(model, config=config, **trainer_kwargs).fit(train_loader, val_loader, resume_from=resume_from)


__all__ = ["LossFunction", "MetricFunction", "PredictionFunction", "StepFunction", "Trainer", "train_model"]
