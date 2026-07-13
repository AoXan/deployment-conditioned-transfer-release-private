from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .cybench import validate_cybench_feature_policy


def prediction_fold_identity_errors(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    id_column: str = "sample_id",
    split_column: str = "split",
) -> list[str]:
    errors: list[str] = []
    if id_column not in predictions or id_column not in folds:
        return [f"missing_id_column:{id_column}"]
    if split_column not in folds:
        return [f"missing_split_column:{split_column}"]
    if predictions[id_column].isna().any() or folds[id_column].isna().any():
        errors.append("null_sample_ids")
    if predictions[id_column].duplicated().any():
        errors.append("duplicate_prediction_ids")
    test_ids = set(folds.loc[folds[split_column].astype(str).str.lower().eq("test"), id_column].astype(str))
    prediction_ids = set(predictions[id_column].astype(str))
    if prediction_ids != test_ids:
        errors.append("prediction_ids_not_equal_fold_test_ids")
    return errors


def temporal_split_errors(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    year_column: str = "Year",
    id_column: str = "sample_id",
) -> list[str]:
    merged = folds[[id_column, "split"]].merge(frame[[id_column, year_column]], on=id_column, how="left")
    if merged[year_column].isna().any():
        return ["fold_ids_missing_from_view"]
    train = pd.to_numeric(merged.loc[merged["split"].eq("train"), year_column], errors="coerce")
    test = pd.to_numeric(merged.loc[merged["split"].eq("test"), year_column], errors="coerce")
    if train.empty or test.empty:
        return ["empty_train_or_test"]
    return [] if train.max() < test.min() else ["temporal_order_violation"]


def feature_policy_errors(
    columns: Iterable[str],
    *,
    dataset: str,
    post_cutoff: Iterable[str] = (),
) -> list[str]:
    if dataset.lower().startswith("cybench"):
        return validate_cybench_feature_policy(columns, post_cutoff=post_cutoff)
    return []


def sample_unit_compatibility_errors(sample_units: Iterable[str]) -> list[str]:
    units = {str(value) for value in sample_units if str(value)}
    return [] if len(units) <= 1 else [f"mixed_sample_units:{'|'.join(sorted(units))}"]
