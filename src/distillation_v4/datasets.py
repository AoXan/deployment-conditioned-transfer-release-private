from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BLOCKED_FEATURES = {
    "yield", "target_yield", "production", "harvest_area", "planted_area",
    "observed_yield", "target_value", "dry_yield_mass",
}
WEATHER = ("tmin", "tmax", "prec", "rad", "tavg", "et0", "vpd", "cwb")


def validate_feature_contract(frame: pd.DataFrame, *, target: str) -> None:
    features = set(frame.columns) - {target, "sample_id", "adm_id", "season_year"}
    violations = sorted(features & BLOCKED_FEATURES)
    if violations:
        raise ValueError(f"BLOCKED_FEATURE:{','.join(violations)}")
    if frame.sample_id.astype(str).duplicated().any():
        raise ValueError("DUPLICATE_SAMPLE_ID")


def _aggregate_weather(source: Path, calendar: pd.DataFrame, cutoff_fraction: float, chunksize: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=["adm_id", "date", *WEATHER], chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk.date.astype(str), format="%Y%m%d", errors="coerce")
        chunk = chunk.merge(calendar, on="adm_id", how="inner").dropna(subset=["date", "sos", "eos"])
        doy = chunk.date.dt.dayofyear
        year = chunk.date.dt.year
        wrap = chunk.sos > chunk.eos
        chunk["season_year"] = np.where(wrap & (doy <= chunk.eos), year - 1, year)
        start = pd.to_datetime(chunk.season_year.astype(int).astype(str) + "-01-01") + pd.to_timedelta(chunk.sos.astype(int) - 1, unit="D")
        end_year = chunk.season_year.astype(int) + wrap.astype(int)
        end = pd.to_datetime(end_year.astype(str) + "-01-01") + pd.to_timedelta(chunk.eos.astype(int) - 1, unit="D")
        cutoff = start + (end - start) * cutoff_fraction
        chunk = chunk.loc[chunk.date.between(start, cutoff)]
        if chunk.empty:
            continue
        grouped = chunk.groupby(["adm_id", "season_year"])[list(WEATHER)].agg(["sum", "count", "min", "max"])
        grouped.columns = [f"{name}__{stat}" for name, stat in grouped.columns]
        pieces.append(grouped.reset_index())
    if not pieces:
        raise ValueError("NO_WEATHER_IN_CROP_WINDOW")
    data = pd.concat(pieces, ignore_index=True)
    aggregation: dict[str, str] = {}
    for name in WEATHER:
        aggregation.update({f"{name}__sum": "sum", f"{name}__count": "sum", f"{name}__min": "min", f"{name}__max": "max"})
    data = data.groupby(["adm_id", "season_year"], as_index=False).agg(aggregation)
    output = data[["adm_id", "season_year"]].copy()
    for name in WEATHER:
        count = data[f"{name}__count"].replace(0, np.nan)
        output[f"weather_{name}_mean"] = data[f"{name}__sum"] / count
        output[f"weather_{name}_sum"] = data[f"{name}__sum"]
        output[f"weather_{name}_min"] = data[f"{name}__min"]
        output[f"weather_{name}_max"] = data[f"{name}__max"]
    return output


def build_cybench_view(subset: Path, *, crop: str, country: str, cutoff_fraction: float = 0.5, chunksize: int = 500_000) -> pd.DataFrame:
    target = pd.read_csv(subset / f"yield_{crop}_{country}.csv")
    calendar = pd.read_csv(subset / f"crop_calendar_{crop}_{country}.csv")[["adm_id", "sos", "eos"]]
    soil = pd.read_csv(subset / f"soil_{crop}_{country}.csv")
    calendar[["sos", "eos"]] = calendar[["sos", "eos"]].apply(pd.to_numeric, errors="coerce")
    weather = _aggregate_weather(subset / f"meteo_{crop}_{country}.csv", calendar, cutoff_fraction, chunksize)
    wrap = calendar.assign(wrap=calendar.sos > calendar.eos).set_index("adm_id").wrap
    if "planting_year" in target:
        season_year = pd.to_numeric(target.planting_year, errors="coerce")
    else:
        wrap_indicator = (
        target.adm_id
        .map(wrap)
        .eq(True)
        .astype(int)
    )

    season_year = (
        pd.to_numeric(
            target.harvest_year,
            errors="coerce",
        )
        - wrap_indicator
    )
    safe = target[["adm_id", "yield"]].rename(columns={"yield": "target_yield"})
    safe["season_year"] = season_year.astype("Int64")
    soil = soil.rename(columns={name: f"soil_{name}" for name in soil.columns if name not in {"adm_id", "crop_name"}})
    view = safe.merge(weather, on=["adm_id", "season_year"], how="inner").merge(soil.drop(columns=["crop_name"], errors="ignore"), on="adm_id", how="left")
    view["sample_id"] = view.adm_id.astype(str) + "|" + view.season_year.astype(str)
    view = view.sort_values(["season_year", "adm_id"]).reset_index(drop=True)
    validate_feature_contract(view, target="target_yield")
    return view


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    weather = [name for name in frame if name.startswith("weather_")]
    soil = [name for name in frame if name.startswith("soil_")]
    return {"weather": weather, "soil": soil, "soil_mask": [f"{name}__missing" for name in soil]}
