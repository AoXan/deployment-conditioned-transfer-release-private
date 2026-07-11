"""Regression metrics for yield models."""

from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Sequence


class MetricsError(ValueError):
    """Raised when metrics cannot be computed."""


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Return RMSE, MAE and R2 for equal-length numeric sequences."""

    if len(y_true) != len(y_pred):
        raise MetricsError("y_true and y_pred must have the same length.")
    if not y_true:
        raise MetricsError("At least one observation is required.")
    errors = [actual - predicted for actual, predicted in zip(y_true, y_pred)]
    mse = mean([error**2 for error in errors])
    mae = mean([abs(error) for error in errors])
    y_mean = mean(y_true)
    total = sum((actual - y_mean) ** 2 for actual in y_true)
    residual = sum(error**2 for error in errors)
    r2 = 0.0 if total == 0 else 1 - residual / total
    return {"rmse": sqrt(mse), "mae": mae, "r2": r2}
