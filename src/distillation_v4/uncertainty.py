from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IntervalResult:
    lower: np.ndarray
    upper: np.ndarray
    quantile: float


def split_conformal(*, calibration_y: np.ndarray, calibration_prediction: np.ndarray, test_prediction: np.ndarray, coverage: float) -> IntervalResult:
    if not 0 < coverage < 1:
        raise ValueError("COVERAGE_OUT_OF_RANGE")
    scores = np.abs(np.asarray(calibration_y) - np.asarray(calibration_prediction))
    n = len(scores)
    if n == 0:
        raise ValueError("CALIBRATION_EMPTY")
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    quantile = float(np.quantile(scores, level, method="higher"))
    prediction = np.asarray(test_prediction, dtype=float)
    return IntervalResult(prediction - quantile, prediction + quantile, quantile)
