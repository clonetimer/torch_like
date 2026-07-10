"""Backward-compatible forecasting runner data imports."""

from torch_like.data.task_data import ForecastDataBundle, ForecastDataConfig, create_forecast_loaders, load_forecast_data

__all__ = [
    "ForecastDataConfig",
    "ForecastDataBundle",
    "load_forecast_data",
    "create_forecast_loaders",
]
