from __future__ import annotations

import numpy as np
import pandas as pd


def validate_common_point_predictor(rows: pd.DataFrame) -> list[str]:
    if "point_fingerprint" not in rows or rows.point_fingerprint.nunique(dropna=False) != 1:
        return ["point_predictor_fingerprint_mismatch"]
    return []


def _aurc(y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray) -> float:
    order = np.argsort(uncertainty)
    errors = np.abs(y_true[order] - y_pred[order])
    risks = np.array([errors[:k].mean() for k in range(1, len(errors) + 1)])
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapezoid(risks, coverage))


def reliability_metrics(*, y_true, y_pred, lower, upper, uncertainty, alpha: float) -> dict[str, float]:
    y = np.asarray(y_true, float)
    pred = np.asarray(y_pred, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    score = np.asarray(uncertainty, float)
    if not all(len(value) == len(y) for value in (pred, lo, hi, score)) or not np.isfinite(np.concatenate([y, pred, lo, hi, score])).all():
        raise ValueError("finite aligned reliability arrays required")
    covered = (y >= lo) & (y <= hi)
    return {
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean(np.square(y - pred)))),
        "coverage": float(covered.mean()),
        "coverage_error": float(abs(covered.mean() - (1 - alpha))),
        "mean_width": float(np.mean(hi - lo)),
        "aurc": _aurc(y, pred, score),
    }

def attribution_stability(first: pd.DataFrame, second: pd.DataFrame, *, top_k: int) -> dict[str, float | str]:
    if list(first.columns) != list(second.columns):
        raise ValueError("attribution feature groups must match")
    a = first.abs().mean().sort_values(ascending=False)
    b = second.abs().mean().sort_values(ascending=False)
    top_a, top_b = set(a.head(top_k).index), set(b.head(top_k).index)
    union = top_a | top_b
    return {
        "rank_spearman": float(a.rank().corr(b.rank(), method="spearman")),
        "top_k_jaccard": float(len(top_a & top_b) / len(union)) if union else 1.0,
        "interpretation": "MODEL_ATTRIBUTION_STABILITY_NOT_CAUSAL",
    }
