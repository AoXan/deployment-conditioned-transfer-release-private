from __future__ import annotations

import pandas as pd


def nested_temporal_split(frame: pd.DataFrame, *, outer_test_year: int) -> pd.DataFrame:
    years = sorted(int(year) for year in frame.loc[frame.year < outer_test_year, "year"].unique())
    if len(years) < 2:
        raise ValueError("INSUFFICIENT_PRE_OUTER_YEARS")
    validation_year = years[-1]
    result = frame[["sample_id", "year"]].copy()
    result["split"] = "excluded"
    result.loc[result.year < validation_year, "split"] = "inner_train"
    result.loc[result.year.eq(validation_year), "split"] = "inner_validation"
    result.loc[result.year.eq(outer_test_year), "split"] = "outer_test"
    return result


def assert_fold_local_fit(*, fitted_ids: set[str], allowed_ids: set[str]) -> None:
    if not fitted_ids <= allowed_ids:
        raise ValueError("PREPROCESSOR_SCOPE_LEAKAGE")
