from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METEO_VARIABLES = (
    "tmin",
    "tmax",
    "prec",
    "rad",
    "tavg",
    "et0",
    "vpd",
    "cwb",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    return pd.read_csv(
        path,
        low_memory=False,
        compression="infer",
    )


def _require_unique(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    label: str,
) -> None:
    duplicated = frame.duplicated(
        keys,
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            f"DUPLICATE_JOIN_KEY:{label}:"
            f"keys={keys}:"
            f"rows={int(duplicated.sum())}"
        )


def _calendar_bounds(
    harvest_year: int,
    sos: float,
    eos: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not np.isfinite(sos) or not np.isfinite(eos):
        raise ValueError(
            "NONFINITE_CROP_CALENDAR"
        )

    sos_day = max(
        1,
        min(366, int(round(sos))),
    )

    eos_day = max(
        1,
        min(366, int(round(eos))),
    )

    # If SOS is later in the calendar year than EOS,
    # the crop season crosses the calendar-year boundary.
    start_year = (
        harvest_year - 1
        if sos_day > eos_day
        else harvest_year
    )

    start = (
        pd.Timestamp(
            year=start_year,
            month=1,
            day=1,
        )
        + pd.Timedelta(
            days=sos_day - 1
        )
    )

    end = (
        pd.Timestamp(
            year=harvest_year,
            month=1,
            day=1,
        )
        + pd.Timedelta(
            days=eos_day - 1
        )
    )

    if end < start:
        raise ValueError(
            "INVALID_CROP_CALENDAR_INTERVAL"
        )

    return start, end


def _aggregate_weather(
    meteo: pd.DataFrame,
    yield_calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meteo = meteo.copy()

    meteo["date"] = pd.to_datetime(
        meteo["date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    meteo = meteo.loc[
        meteo["date"].notna()
    ].copy()

    for column in METEO_VARIABLES:
        meteo[column] = pd.to_numeric(
            meteo[column],
            errors="coerce",
        )

    groups = {
        str(adm_id): group.sort_values(
            "date"
        )
        for adm_id, group in meteo.groupby(
            "adm_id",
            observed=True,
        )
    }

    records: list[dict[str, Any]] = []
    no_weather = 0
    incomplete_seasons = 0

    for row in yield_calendar.itertuples(
        index=False
    ):
        start, end = _calendar_bounds(
            int(row.harvest_year),
            float(row.sos),
            float(row.eos),
        )

        source = groups.get(
            str(row.adm_id)
        )

        if source is None:
            no_weather += 1
            continue

        season = source.loc[
            source["date"].between(
                start,
                end,
                inclusive="both",
            )
        ]

        if season.empty:
            no_weather += 1
            continue

        expected_days = (
            end - start
        ).days + 1

        observed_days = int(
            season["date"].nunique()
        )

        if observed_days < max(
            30,
            int(expected_days * 0.75),
        ):
            incomplete_seasons += 1
            continue

        record: dict[str, Any] = {
            "adm_id": str(row.adm_id),
            "harvest_year": int(
                row.harvest_year
            ),
            "weather_season_start": (
                start.strftime("%Y-%m-%d")
            ),
            "weather_season_end": (
                end.strftime("%Y-%m-%d")
            ),
            "weather_observed_days": (
                observed_days
            ),
            "weather_expected_days": (
                expected_days
            ),
            "weather_coverage": (
                observed_days
                / max(expected_days, 1)
            ),
        }

        for column in METEO_VARIABLES:
            values = pd.to_numeric(
                season[column],
                errors="coerce",
            )

            prefix = f"weather_{column}"

            record[
                f"{prefix}_mean"
            ] = float(values.mean())

            record[
                f"{prefix}_sum"
            ] = float(values.sum())

            record[
                f"{prefix}_min"
            ] = float(values.min())

            record[
                f"{prefix}_max"
            ] = float(values.max())

            record[
                f"{prefix}_std"
            ] = float(
                values.std(ddof=0)
            )

        records.append(record)

    result = pd.DataFrame(records)

    if result.empty:
        raise ValueError(
            "CYBENCH_WEATHER_AGGREGATION_EMPTY"
        )

    _require_unique(
        result,
        ["adm_id", "harvest_year"],
        label="aggregated_weather",
    )

    ledger = {
        "requested_yield_rows": int(
            len(yield_calendar)
        ),
        "aggregated_rows": int(
            len(result)
        ),
        "no_weather_rows": no_weather,
        "incomplete_season_rows": (
            incomplete_seasons
        ),
        "aggregation_variables": list(
            METEO_VARIABLES
        ),
        "aggregation_statistics": [
            "mean",
            "sum",
            "min",
            "max",
            "std",
        ],
        "date_filter": (
            "ADM_SPECIFIC_SOS_TO_EOS_FOR_HARVEST_YEAR"
        ),
    }

    return result, ledger


def build_cybench_view(
    *,
    dataset_id: str,
    directory: Path,
    crop: str,
    country: str,
    output_path: Path,
) -> dict[str, Any]:
    yield_path = (
        directory
        / f"yield_{crop}_{country}.csv"
    )

    meteo_path = (
        directory
        / f"meteo_{crop}_{country}.csv"
    )

    calendar_path = (
        directory
        / f"crop_calendar_{crop}_{country}.csv"
    )

    soil_path = (
        directory
        / f"soil_{crop}_{country}.csv"
    )

    location_path = (
        directory
        / f"location_{crop}_{country}.csv"
    )

    mask_path = (
        directory
        / f"crop_mask_{crop}_{country}.csv"
    )

    yield_frame = _read(yield_path)
    meteo = _read(meteo_path)
    calendar = _read(calendar_path)
    soil = _read(soil_path)
    location = _read(location_path)
    crop_mask = _read(mask_path)

    required_yield = {
        "adm_id",
        "harvest_year",
        "yield",
    }

    if not required_yield <= set(
        yield_frame
    ):
        raise ValueError(
            "CYBENCH_YIELD_CONTRACT_MISSING"
        )

    for frame, keys, label in [
        (
            yield_frame,
            ["adm_id", "harvest_year"],
            "yield",
        ),
        (
            calendar,
            ["adm_id"],
            "calendar",
        ),
        (
            soil,
            ["adm_id"],
            "soil",
        ),
        (
            location,
            ["adm_id"],
            "location",
        ),
        (
            crop_mask,
            ["adm_id"],
            "crop_mask",
        ),
    ]:
        _require_unique(
            frame,
            keys,
            label=label,
        )

    yield_frame = yield_frame.copy()

    yield_frame["adm_id"] = (
        yield_frame["adm_id"]
        .astype(str)
    )

    yield_frame["harvest_year"] = (
        pd.to_numeric(
            yield_frame[
                "harvest_year"
            ],
            errors="coerce",
        )
    )

    yield_frame["yield"] = pd.to_numeric(
        yield_frame["yield"],
        errors="coerce",
    )

    yield_frame = yield_frame.loc[
        yield_frame["harvest_year"].notna()
        & yield_frame["yield"].notna()
    ].copy()

    yield_frame["harvest_year"] = (
        yield_frame["harvest_year"]
        .astype(int)
    )

    calendar = calendar[
        ["adm_id", "sos", "eos"]
    ].copy()

    calendar["adm_id"] = (
        calendar["adm_id"].astype(str)
    )

    yield_calendar = yield_frame.merge(
        calendar,
        on="adm_id",
        how="inner",
        validate="many_to_one",
    )

    weather, weather_ledger = (
        _aggregate_weather(
            meteo,
            yield_calendar[
                [
                    "adm_id",
                    "harvest_year",
                    "sos",
                    "eos",
                ]
            ],
        )
    )

    result = yield_frame.merge(
        weather,
        on=[
            "adm_id",
            "harvest_year",
        ],
        how="inner",
        validate="one_to_one",
    )

    soil = soil.copy()
    soil["adm_id"] = (
        soil["adm_id"].astype(str)
    )

    soil = soil.rename(
        columns={
            column: f"privileged_soil_{column}"
            for column in soil.columns
            if column not in {
                "crop_name",
                "adm_id",
            }
        }
    )

    location = location.copy()
    location["adm_id"] = (
        location["adm_id"].astype(str)
    )

    location = location.rename(
        columns={
            "latitude": (
                "context_latitude"
            ),
            "longitude": (
                "context_longitude"
            ),
            "region_area": (
                "context_region_area"
            ),
        }
    )

    crop_mask = crop_mask.copy()
    crop_mask["adm_id"] = (
        crop_mask["adm_id"].astype(str)
    )

    crop_mask = crop_mask.rename(
        columns={
            "crop_area": (
                "context_crop_area"
            ),
            "crop_area_percentage": (
                "context_crop_area_percentage"
            ),
        }
    )

    result = result.merge(
        soil.drop(
            columns=["crop_name"],
            errors="ignore",
        ),
        on="adm_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        location.drop(
            columns=["crop_name"],
            errors="ignore",
        ),
        on="adm_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        crop_mask.drop(
            columns=["crop_name"],
            errors="ignore",
        ),
        on="adm_id",
        how="left",
        validate="many_to_one",
    )

    result["sample_id"] = (
        dataset_id
        + "|"
        + result["adm_id"].astype(str)
        + "|"
        + result[
            "harvest_year"
        ].astype(str)
    )

    result["year"] = result[
        "harvest_year"
    ].astype(int)

    result["target_yield"] = result[
        "yield"
    ].astype(float)

    result["dataset_id"] = dataset_id

    result["sample_unit"] = (
        "ADMIN_REGION_HARVEST_YEAR"
    )

    result["target_unit"] = (
        "tonne_ha"
    )

    result["adapter_version"] = (
        "universal_weather_cybench_v2"
    )

    if result["sample_id"].duplicated().any():
        raise ValueError(
            "CYBENCH_SAMPLE_ID_DUPLICATE"
        )

    weather_columns = [
        column
        for column in result
        if column.startswith(
            "weather_"
        )
        and column not in {
            "weather_season_start",
            "weather_season_end",
        }
    ]

    if not weather_columns:
        raise ValueError(
            "CYBENCH_NO_WEATHER_FEATURES"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    ledger = {
        "schema_version": (
            "universal_weather_cybench_adapter_v2"
        ),
        "dataset_id": dataset_id,
        "source_directory": str(
            directory
        ),
        "source_hashes": {
            path.name: sha256_file(path)
            for path in [
                yield_path,
                meteo_path,
                calendar_path,
                soil_path,
                location_path,
                mask_path,
            ]
        },
        "yield_rows": int(
            len(yield_frame)
        ),
        "output_rows": int(
            len(result)
        ),
        "output_weather_features": len(
            weather_columns
        ),
        "output_privileged_features": len(
            [
                column
                for column in result
                if column.startswith(
                    "privileged_"
                )
            ]
        ),
        "weather": weather_ledger,
        "join_policy": {
            "yield_calendar": (
                "many_to_one_on_adm_id"
            ),
            "yield_weather": (
                "one_to_one_on_adm_id_harvest_year"
            ),
            "soil_location_mask": (
                "many_to_one_on_adm_id"
            ),
            "many_to_many_allowed": False,
        },
        "target_source": str(
            yield_path
        ),
        "target_column": "yield",
        "outer_test_used": False,
        "target_metrics_used_for_selection": False,
        "output_path": str(
            output_path
        ),
        "output_sha256": sha256_file(
            output_path
        ),
    }

    return ledger
