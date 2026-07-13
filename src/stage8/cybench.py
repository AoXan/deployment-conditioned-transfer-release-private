from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


BLOCKED_CYBENCH_FEATURES = {
    "yield",
    "target_yield",
    "production",
    "harvest_area",
    "observed_yield",
    "target_value",
}


def safe_feature_columns(frame: pd.DataFrame, *, allowlist: Iterable[str], blocklist: Iterable[str] = ()) -> list[str]:
    blocked = BLOCKED_CYBENCH_FEATURES | {str(value) for value in blocklist}
    return [name for name in allowlist if name in frame.columns and name not in blocked]


def crop_window_mask(dates: pd.Series, *, planting_year: int, sos: int, eos: int) -> pd.Series:
    """Select a crop window, including southern-hemisphere wraparound seasons."""
    timestamps = pd.to_datetime(dates, errors="coerce")
    start = pd.Timestamp(planting_year, 1, 1) + pd.Timedelta(days=int(sos) - 1)
    harvest_year = planting_year if int(eos) >= int(sos) else planting_year + 1
    end = pd.Timestamp(harvest_year, 1, 1) + pd.Timedelta(days=int(eos) - 1)
    return timestamps.between(start, end, inclusive="both")


def validate_cybench_feature_policy(columns: Iterable[str], *, post_cutoff: Iterable[str] = ()) -> list[str]:
    present = {str(value) for value in columns}
    violations = sorted(present & (BLOCKED_CYBENCH_FEATURES | {str(value) for value in post_cutoff}))
    return [f"blocked_feature:{name}" for name in violations]


def build_cybench_safe_view(
    subset_dir: Path,
    *,
    crop: str,
    country: str,
    output_path: Path,
    weather_variables: Iterable[str],
    chunksize: int = 500_000,
) -> dict[str, object]:
    """Build a leakage-safe mid-season administrative-unit-year table.

    Cross-year seasons are assigned to their planting year. Target components
    (`production`, `harvest_area`) are never copied to the feature table.
    """
    yield_path = subset_dir / f"yield_{crop}_{country}.csv"
    calendar_path = subset_dir / f"crop_calendar_{crop}_{country}.csv"
    meteo_path = subset_dir / f"meteo_{crop}_{country}.csv"
    for path in (yield_path, calendar_path, meteo_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    target = pd.read_csv(yield_path)
    calendar = pd.read_csv(calendar_path)[["adm_id", "sos", "eos"]].copy()
    calendar["sos"] = pd.to_numeric(calendar["sos"], errors="coerce").round().astype("Int64")
    calendar["eos"] = pd.to_numeric(calendar["eos"], errors="coerce").round().astype("Int64")
    variables = [str(value) for value in weather_variables]
    partials: list[pd.DataFrame] = []
    usecols = ["adm_id", "date", *variables]
    for chunk in pd.read_csv(meteo_path, usecols=usecols, chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d", errors="coerce")
        chunk = chunk.merge(calendar, on="adm_id", how="inner").dropna(subset=["date", "sos", "eos"])
        doy = chunk["date"].dt.dayofyear
        year = chunk["date"].dt.year
        wrap = chunk["sos"] > chunk["eos"]
        chunk["season_year"] = np.where(wrap & (doy <= chunk["eos"]), year - 1, year)
        start = pd.to_datetime(chunk["season_year"].astype(str) + "-01-01") + pd.to_timedelta(chunk["sos"].astype(int) - 1, unit="D")
        end_year = chunk["season_year"] + wrap.astype(int)
        end = pd.to_datetime(end_year.astype(str) + "-01-01") + pd.to_timedelta(chunk["eos"].astype(int) - 1, unit="D")
        cutoff = start + (end - start) / 2
        chunk = chunk[(chunk["date"] >= start) & (chunk["date"] <= cutoff)]
        if chunk.empty:
            continue
        for variable in variables:
            chunk[variable] = pd.to_numeric(chunk[variable], errors="coerce")
        agg = chunk.groupby(["adm_id", "season_year"])[variables].agg(["sum", "count", "min", "max"])
        agg.columns = [f"{name}__{stat}" for name, stat in agg.columns]
        partials.append(agg.reset_index())
    if not partials:
        raise ValueError("no weather rows matched crop-calendar mid-season windows")
    combined = pd.concat(partials, ignore_index=True)
    aggregation: dict[str, str] = {}
    for variable in variables:
        aggregation[f"{variable}__sum"] = "sum"
        aggregation[f"{variable}__count"] = "sum"
        aggregation[f"{variable}__min"] = "min"
        aggregation[f"{variable}__max"] = "max"
    weather = combined.groupby(["adm_id", "season_year"], as_index=False).agg(aggregation)
    for variable in variables:
        weather[f"{variable}_sum_midseason"] = weather.pop(f"{variable}__sum")
        count = weather.pop(f"{variable}__count").replace(0, np.nan)
        weather[f"{variable}_mean_midseason"] = weather[f"{variable}_sum_midseason"] / count
        weather[f"{variable}_min_midseason"] = weather.pop(f"{variable}__min")
        weather[f"{variable}_max_midseason"] = weather.pop(f"{variable}__max")
    target_year = pd.to_numeric(target.get("planting_year"), errors="coerce") if "planting_year" in target else pd.Series(np.nan, index=target.index)
    if target_year.isna().all():
        harvest = pd.to_numeric(target["harvest_year"], errors="coerce")
        wrap_map = calendar.assign(wrap=calendar["sos"] > calendar["eos"]).set_index("adm_id")["wrap"]
        target_year = harvest - target["adm_id"].map(wrap_map).fillna(False).astype(int)
    target_safe = target[["adm_id", "yield"]].copy()
    target_safe["season_year"] = target_year.astype("Int64")
    view = target_safe.merge(weather, on=["adm_id", "season_year"], how="inner")
    view = view.rename(columns={"yield": "target_yield"})
    view["sample_id"] = view["adm_id"].astype(str) + "|" + view["season_year"].astype(str)
    violations = validate_cybench_feature_policy([c for c in view.columns if c not in {"target_yield"}])
    if violations:
        raise ValueError(";".join(violations))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    view.to_csv(tmp, index=False, compression="gzip" if output_path.suffix == ".gz" else None)
    tmp.replace(output_path)
    return {"rows": int(len(view)), "columns": int(len(view.columns)), "wraparound_supported": True}
