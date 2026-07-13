from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path


def _record(axis: str, fold: str, frame: pd.DataFrame, train: pd.Series, validation: pd.Series, test: pd.Series, *, role: str = "ADAPTED_PROTOCOL") -> dict[str, Any]:
    ids = frame["sample_id"].astype(str)
    sets = [set(ids[mask]) for mask in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("AUSTRALIAN_SPLIT_ID_OVERLAP")
    target = next((name for name in ("target_yield", "observed_yield_t_ha", "target_value") if name in frame), None)
    unique = {part: int(frame.loc[mask, target].nunique(dropna=True)) if target else 0 for part, mask in zip(("train", "validation", "test"), (train, validation, test))}
    return {"axis": axis, "fold": fold, "role": role, "train_ids": sorted(sets[0]), "validation_ids": sorted(sets[1]), "test_ids": sorted(sets[2]), "target_unique": unique}


def build_waite_splits(frame: pd.DataFrame, *, crop: str) -> list[dict[str, Any]]:
    data = frame.loc[frame["crop"].astype(str).str.lower().eq(crop.lower()) & frame["observed_yield_t_ha"].notna()].copy()
    splits: list[dict[str, Any]] = []
    for year in (1991, 1992, 1993):
        if year in set(data.year):
            splits.append(_record("forward_temporal", f"test_{year}", data, data.year < year - 1, data.year.eq(year - 1), data.year.eq(year)))
    for year in (1950, 1960, 1970, 1980, 1990, 1993):
        if year in set(data.year) and year - 1 in set(data.year):
            splits.append(_record("rolling_origin", f"origin_{year}", data, data.year < year - 1, data.year.eq(year - 1), data.year.eq(year)))
    long_period = _record("long_period_transfer", "era_1925_1993", data, data.year.le(1969), data.year.between(1970, 1979), data.year.between(1980, 1993), role="TRANSFER_PROTOCOL")
    long_period["source_era_ids"] = sorted(data.loc[data.year.le(1959), "sample_id"].astype(str))
    long_period["adaptation_era_ids"] = sorted(data.loc[data.year.between(1960, 1969), "sample_id"].astype(str))
    splits.append(long_period)
    splits.append(_record("plot_repeated_temporal", "late_era", data, data.year.lt(1980), data.year.between(1980, 1989), data.year.ge(1990)))
    plots = sorted(data["plot"].dropna().astype(str).unique())
    groups = {plot: int(hashlib.sha256(plot.encode()).hexdigest(), 16) % 5 for plot in plots}
    held = {plot for plot, group in groups.items() if group == 0}
    validation = {plot for plot, group in groups.items() if group == 1}
    values = data["plot"].astype(str)
    splits.append(_record("plot_disjoint_diagnostic", "hash_group_0", data, ~values.isin(held | validation), values.isin(validation), values.isin(held), role="DIAGNOSTIC_PROTOCOL"))
    return [split for split in splits if split["train_ids"] and split["validation_ids"] and split["test_ids"]]


def build_roseworthy_splits(frame: pd.DataFrame) -> list[dict[str, Any]]:
    data = frame.dropna(subset=["sample_id", "Eastings_UTM", "Northings_UTM", "target_value"]).copy()
    splits: list[dict[str, Any]] = []
    for (year, crop), group in data.groupby(["year", "crop_product"], sort=True):
        x = pd.to_numeric(group["Eastings_UTM"])
        threshold = float(x.median())
        spread = max(float(np.median(np.abs(np.diff(np.sort(x.unique()))))) if x.nunique() > 1 else 1.0, 1.0)
        test = x.gt(threshold + spread)
        validation = x.between(threshold - spread, threshold + spread, inclusive="both")
        train = x.lt(threshold - spread)
        if train.any() and validation.any() and test.any():
            splits.append(_record("spatial_buffer", f"{int(year)}_{str(crop)}", group, train, validation, test, role="DIAGNOSTIC_PROTOCOL"))
        block = (x.rank(method="first", pct=True) > 0.8)
        val = x.rank(method="first", pct=True).between(0.6, 0.8, inclusive="right")
        train_block = ~(block | val)
        splits.append(_record("spatial_block", f"{int(year)}_{str(crop)}", group, train_block, val, block, role="ADAPTED_PROTOCOL"))
    years = sorted(data["year"].unique())
    if len(years) >= 2:
        first, last = years[0], years[-1]
        source = data.year.eq(first)
        target = data.year.eq(last)
        target_order = data.loc[target, "Eastings_UTM"].rank(method="first", pct=True)
        validation_ids = set(data.loc[target].loc[target_order.le(0.2), "sample_id"])
        validation = data.sample_id.isin(validation_ids)
        test = target & ~validation
        splits.append(_record("cross_crop_transfer_diagnostic", f"{int(first)}_to_{int(last)}", data, source, validation, test, role="TRANSFER_PROTOCOL"))
    return splits


def build_regional_splits(frame: pd.DataFrame) -> list[dict[str, Any]]:
    data = frame.loc[frame["target_yield"].notna()].copy()
    years = sorted(pd.to_numeric(data.year, errors="coerce").dropna().astype(int).unique())
    splits: list[dict[str, Any]] = []
    for year in years[-3:]:
        earlier = [value for value in years if value < year]
        if len(earlier) < 2:
            continue
        validation_year = earlier[-1]
        splits.append(_record("frozen_temporal", f"test_{year}", data, data.year < validation_year, data.year.eq(validation_year), data.year.eq(year), role="ADAPTED_PROTOCOL"))
    if "region_id" in data and data.region_id.nunique() >= 3:
        regions = sorted(data.region_id.astype(str).unique())
        splits.append(_record("region_holdout", f"region_{regions[-1]}", data, ~data.region_id.astype(str).isin(regions[-2:]), data.region_id.astype(str).eq(regions[-2]), data.region_id.astype(str).eq(regions[-1]), role="ADAPTED_PROTOCOL"))
    return [split for split in splits if split["train_ids"] and split["validation_ids"] and split["test_ids"]]


def build_frozen_regional_splits(frame: pd.DataFrame, paths: list[Path]) -> list[dict[str, Any]]:
    by_id = frame.set_index(frame.sample_id.astype(str), drop=False)
    output = []
    for path in sorted(paths):
        fold = pd.read_csv(path); fold.sample_id = fold.sample_id.astype(str)
        train_ids = fold.loc[fold.split.eq("train"), "sample_id"]
        test_ids = fold.loc[fold.split.eq("test"), "sample_id"]
        if not set(train_ids).issubset(by_id.index) or not set(test_ids).issubset(by_id.index):
            raise ValueError("FROZEN_FOLD_IDS_NOT_IN_VIEW:" + path.name)
        train_years = pd.to_numeric(by_id.loc[train_ids, "year"], errors="coerce")
        if "temporal_forward" in path.name:
            validation_year = int(train_years.max()); validation_ids = set(train_ids[train_years.to_numpy() == validation_year]); axis = "frozen_temporal"
        else:
            validation_ids = {value for value in train_ids if int(hashlib.sha256(value.encode()).hexdigest(), 16) % 5 == 0}; axis = "frozen_region_holdout"
        actual_train = set(train_ids) - validation_ids
        train_mask = frame.sample_id.astype(str).isin(actual_train); validation_mask = frame.sample_id.astype(str).isin(validation_ids); test_mask = frame.sample_id.astype(str).isin(set(test_ids))
        role = "DIAGNOSTIC_PROTOCOL" if "diagnostic" in path.name else "EXACT_PROTOCOL"
        result = _record(axis, path.stem, frame, train_mask, validation_mask, test_mask, role=role)
        result["outer_fold_artifact"] = str(path); result["inner_validation_source"] = "OUTER_TRAIN_ONLY"
        output.append(result)
    return [split for split in output if split["train_ids"] and split["validation_ids"] and split["test_ids"]]
