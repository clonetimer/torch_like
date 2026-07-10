"""Time series forecasting metrics."""

import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """Mean Absolute Percentage Error (handles zero values)."""
    mask = true != 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100


def MSPE(pred, true):
    """Mean Square Percentage Error."""
    mask = true != 0
    return np.mean(np.square((pred[mask] - true[mask]) / true[mask])) * 100


def metric_fn(pred, true):
    """Compute a standard set of metrics. Returns a dict."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "MSPE": mspe}
