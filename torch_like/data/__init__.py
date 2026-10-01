"""Reusable dataset, loader, and task-data utilities."""

from torch_like.data.adapters import DataAdapter, DataLoaderConfig, TensorDataAdapter

from torch_like.data.dataset import Dataset_Custom, Dataset_ETT_hour
from torch_like.data.dataset_detection import SlidingWindowAnomalyDataset, create_train_test_anomaly_datasets
from torch_like.data.dataset_graph_forecasting import GraphForecastDataset, GraphForecastMeta, load_graph_forecast_array
from torch_like.data.dataset_rul import XJTUSYDataset, XJTUSYFileWindowDataset, create_train_test_split
from torch_like.data.loader import data_provider
from torch_like.data.task_data import (
    DetectionDataBundle,
    DetectionDataConfig,
    ForecastDataBundle,
    ForecastDataConfig,
    RULDataBundle,
    RULDataConfig,
    create_detection_loaders,
    create_forecast_loaders,
    create_rul_loaders,
    load_detection_data,
    load_esa_csv,
    load_forecast_data,
    load_numpy_detection_root,
    load_rul_data,
)
from torch_like.data.timefeatures import TimeFeature, time_features

__all__ = [
    "DataAdapter",
    "DataLoaderConfig",
    "TensorDataAdapter",
    "time_features",
    "TimeFeature",
    "Dataset_ETT_hour",
    "Dataset_Custom",
    "data_provider",
    "SlidingWindowAnomalyDataset",
    "create_train_test_anomaly_datasets",
    "GraphForecastDataset",
    "GraphForecastMeta",
    "load_graph_forecast_array",
    "XJTUSYDataset",
    "XJTUSYFileWindowDataset",
    "create_train_test_split",
    "ForecastDataConfig",
    "ForecastDataBundle",
    "DetectionDataConfig",
    "DetectionDataBundle",
    "RULDataConfig",
    "RULDataBundle",
    "load_forecast_data",
    "create_forecast_loaders",
    "load_detection_data",
    "create_detection_loaders",
    "load_rul_data",
    "create_rul_loaders",
    "load_numpy_detection_root",
    "load_esa_csv",
]
