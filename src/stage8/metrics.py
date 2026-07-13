from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _finite(*series: pd.Series) -> list[pd.Series]:
    converted = [pd.to_numeric(value, errors="coerce").reset_index(drop=True) for value in series]
    mask = pd.Series(True, index=converted[0].index)
    for value in converted:
        mask &= value.notna() & np.isfinite(value)
    return [value[mask].reset_index(drop=True) for value in converted]


def point_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    truth, pred = _finite(y_true, y_pred)
    if truth.empty:
        raise ValueError("no finite prediction rows")
    return {
        "n": int(len(truth)),
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(math.sqrt(mean_squared_error(truth, pred))),
        "r2": float(r2_score(truth, pred)) if len(truth) > 1 else float("nan"),
    }


def conformal_metrics(
    y_true: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    *,
    alpha: float,
) -> dict[str, float | int]:
    truth, lo, hi = _finite(y_true, lower, upper)
    if truth.empty:
        raise ValueError("no finite interval rows")
    if (hi < lo).any():
        raise ValueError("interval upper bound below lower bound")
    width = hi - lo
    covered = (truth >= lo) & (truth <= hi)
    winkler = width.copy()
    winkler = winkler + (2 / alpha) * (lo - truth).clip(lower=0) + (2 / alpha) * (truth - hi).clip(lower=0)
    return {
        "n": int(len(truth)),
        "coverage": float(covered.mean()),
        "absolute_coverage_error": float(abs(float(covered.mean()) - (1 - alpha))),
        "mean_width": float(width.mean()),
        "winkler": float(winkler.mean()),
    }


def deployable_aurc(
    y_true: pd.Series,
    y_pred: pd.Series,
    uncertainty_score: pd.Series,
    *,
    score_name: str = "uncertainty_score",
) -> dict[str, float | int]:
    lowered = score_name.lower().replace("-", "_")
    if "residual" in lowered or "absolute_error" in lowered or lowered in {"error", "test_error"}:
        raise ValueError("test residual/error scores are forbidden for deployable AURC")
    truth, pred, score = _finite(y_true, y_pred, uncertainty_score)
    if truth.empty:
        raise ValueError("no finite selective prediction rows")
    order = np.argsort(score.to_numpy(), kind="stable")
    losses = np.abs(truth.to_numpy()[order] - pred.to_numpy()[order])
    cumulative_risk = np.cumsum(losses) / np.arange(1, len(losses) + 1)
    coverage = np.arange(1, len(losses) + 1) / len(losses)
    aurc = float(np.trapezoid(cumulative_risk, coverage)) if len(losses) > 1 else float(cumulative_risk[0])
    return {"n": int(len(losses)), "aurc": aurc, "full_coverage_risk": float(cumulative_risk[-1])}

