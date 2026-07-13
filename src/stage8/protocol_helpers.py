"""Pure helpers for frozen Stage 8 selection and calibration protocols."""
from __future__ import annotations

import numpy as np


def select_by_validation_mae(y_validation, candidate_predictions: dict[str, np.ndarray]) -> str:
    y = np.asarray(y_validation, dtype=float)
    if not candidate_predictions:
        raise ValueError("no_validation_candidates")
    scores = {name: float(np.mean(np.abs(y - np.asarray(pred, dtype=float)))) for name, pred in candidate_predictions.items()}
    return min(scores, key=lambda name: (scores[name], name))


def conditional_quantiles(residuals, patterns, *, alpha: float, min_count: int) -> dict[str, float]:
    residuals = np.asarray(residuals, dtype=float)
    patterns = np.asarray(patterns).astype(str)
    pooled = float(np.quantile(residuals, 1 - alpha, method="higher"))
    result = {"__pooled__": pooled}
    for pattern in sorted(set(patterns)):
        values = residuals[patterns == pattern]
        result[pattern] = float(np.quantile(values, 1 - alpha, method="higher")) if len(values) >= min_count else pooled
    return result


def aulc(fraction_to_error: dict[float, float]) -> float:
    pairs = sorted((float(k), float(v)) for k, v in fraction_to_error.items())
    return float(np.trapezoid([v for _, v in pairs], [k for k, _ in pairs]))
