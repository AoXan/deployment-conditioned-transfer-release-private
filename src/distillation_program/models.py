from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.stage8.supplemental_torch import build_modality_model


def build_model(model_id: str, frame: pd.DataFrame, *, seed: int) -> Pipeline:
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column]) and not frame[column].isna().all()]
    if not numeric:
        raise ValueError("model_requires_numeric_features")
    estimators = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=10, random_state=seed),
        "matched_mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=250, early_stopping=True, random_state=seed),
    }
    if model_id not in estimators:
        raise ValueError(f"unknown_model:{model_id}")
    return Pipeline([
        ("select", _NumericSelector(numeric)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimators[model_id]),
    ])


class _NumericSelector(TransformerMixin, BaseEstimator):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, frame, target=None):
        return self

    def transform(self, frame):
        return frame[self.columns]


def build_modality_network(modality_dims: dict[str, int], hidden: int = 32):
    return build_modality_model(modality_dims, hidden=hidden)
