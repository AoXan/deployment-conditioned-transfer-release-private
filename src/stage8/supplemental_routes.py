"""Protocol-complete numerical primitives for Supplemental T1--U2 routes."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import deployable_aurc


def finite_sample_quantile(scores: Iterable[float], alpha: float) -> float:
    values = np.sort(np.asarray(list(scores), dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite non-empty calibration scores required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    rank = min(len(values), int(math.ceil((len(values) + 1) * (1 - alpha))))
    return float(values[rank - 1])


def weighted_quantile(scores: Sequence[float], weights: Sequence[float], alpha: float) -> tuple[float, float]:
    score = np.asarray(scores, float)
    weight = np.asarray(weights, float)
    if len(score) != len(weight) or not len(score) or np.any(weight < 0) or weight.sum() <= 0:
        raise ValueError("valid aligned non-negative weights required")
    order = np.argsort(score)
    score, weight = score[order], weight[order]
    cumulative = np.cumsum(weight) / weight.sum()
    q = float(score[min(np.searchsorted(cumulative, 1 - alpha, side="left"), len(score) - 1)])
    ess = float(weight.sum() ** 2 / np.square(weight).sum())
    return q, ess


def learning_curve_aulc(budgets: Sequence[float], errors: Sequence[float]) -> dict[str, float]:
    x, y = np.asarray(budgets, float), np.asarray(errors, float)
    if len(x) != len(y) or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("strictly increasing aligned learning curve required")
    span = float(x[-1] - x[0])
    return {"aulc": float(np.trapezoid(y, x) / span), "budget_min": float(x[0]), "budget_max": float(x[-1])}


def soft_route(expert_predictions: Sequence[Sequence[float]], logits: Sequence[Sequence[float]], temperature: float = 1.0):
    pred = np.asarray(expert_predictions, float)
    z = np.asarray(logits, float) / float(temperature)
    if pred.shape != z.shape or pred.ndim != 2 or temperature <= 0:
        raise ValueError("aligned 2D predictions/logits and positive temperature required")
    z -= z.max(axis=1, keepdims=True)
    weights = np.exp(z)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(pred * weights, axis=1), weights


def mondrian_intervals(*, calibration_residuals, calibration_patterns, test_predictions, test_patterns, alpha: float, min_count: int):
    residuals = np.asarray(calibration_residuals, float)
    patterns = np.asarray(calibration_patterns, str)
    test_pred = np.asarray(test_predictions, float)
    test_patterns = np.asarray(test_patterns, str)
    pooled = finite_sample_quantile(residuals, alpha)
    ledger: dict[str, dict[str, float | int | str]] = {}
    q = []
    for pattern in test_patterns:
        mask = patterns == pattern
        if int(mask.sum()) >= min_count:
            value, source = finite_sample_quantile(residuals[mask], alpha), "mondrian"
        else:
            value, source = pooled, "pooled_fallback"
        ledger[str(pattern)] = {"n_calibration": int(mask.sum()), "quantile": value, "source": source}
        q.append(value)
    qv = np.asarray(q)
    return test_pred - qv, test_pred + qv, ledger


def uncertainty_endpoints(*, y_true, y_pred, lower, upper, uncertainty, random_seed: int, alpha: float = 0.1):
    y = np.asarray(y_true, float); p = np.asarray(y_pred, float)
    lo = np.asarray(lower, float); hi = np.asarray(upper, float); score = np.asarray(uncertainty, float)
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    winkler = width + (2 / alpha) * np.where(y < lo, lo - y, np.where(y > hi, y - hi, 0.0))
    rng = np.random.default_rng(random_seed)
    random_score = rng.permutation(score)
    return {
        "coverage": float(covered.mean()),
        "absolute_coverage_error": float(abs(covered.mean() - (1 - alpha))),
        "mean_width": float(width.mean()),
        "winkler": float(winkler.mean()),
        "deployable_aurc": float(deployable_aurc(pd.Series(y), pd.Series(p), pd.Series(score), score_name="deployable_uncertainty")["aurc"]),
        "random_rejection_aurc": float(deployable_aurc(pd.Series(y), pd.Series(p), pd.Series(random_score), score_name="random_rejection")["aurc"]),
    }
