from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniversalDatasetV2:
    dataset_id: str
    frame: pd.DataFrame
    weather_features: tuple[str, ...]
    privileged_features: tuple[str, ...]
    target_column: str
    sample_id_column: str
    year_column: str
    target_unit: str
    sample_unit: str
    source_path: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _first_existing(
    columns: set[str],
    candidates: list[str],
) -> str | None:
    lowered = {
        str(column).lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate in columns:
            return candidate

        found = lowered.get(candidate.lower())

        if found is not None:
            return found

    return None


def _numeric_features(
    frame: pd.DataFrame,
    declared: list[str] | None,
    *,
    prefixes: tuple[str, ...],
) -> list[str]:
    if declared:
        values = [
            str(column)
            for column in declared
            if (
                str(column) in frame
                and pd.api.types.is_numeric_dtype(
                    frame[str(column)]
                )
            )
        ]

        if values:
            return values

    return [
        str(column)
        for column in frame.columns
        if (
            str(column).lower().startswith(prefixes)
            and pd.api.types.is_numeric_dtype(
                frame[column]
            )
        )
    ]


def load_dataset_v2(
    *,
    dataset_id: str,
    registry_entry: dict[str, Any],
    repository_root: Path,
) -> UniversalDatasetV2:
    path = Path(
        registry_entry["path"]
    )

    if not path.is_absolute():
        path = repository_root / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"DATASET_ASSET_NOT_FOUND:{dataset_id}:{path}"
        )

    suffix = path.suffix.lower()
    suffixes = [
        value.lower()
        for value in path.suffixes
    ]

    if (
        suffix == ".csv"
        or suffixes[-2:] == [".csv", ".gz"]
    ):
        frame = pd.read_csv(
            path,
            low_memory=False,
            compression="infer",
        )

    elif suffix in {
        ".parquet",
        ".pq",
    }:
        frame = pd.read_parquet(path)

    else:
        raise ValueError(
            f"UNSUPPORTED_DATASET_FORMAT:{dataset_id}:{suffix}"
        )

    if frame.empty:
        raise ValueError(
            f"EMPTY_DATASET:{dataset_id}"
        )

    target = registry_entry.get(
        "target_column"
    ) or _first_existing(
        set(frame),
        [
            "target_yield",
            "yield",
            "Yield",
            "yield_value",
            "grain_yield",
        ],
    )

    sample_id = registry_entry.get(
        "sample_id_column"
    ) or _first_existing(
        set(frame),
        [
            "sample_id",
            "row_id",
            "observation_id",
            "record_id",
        ],
    )

    year = registry_entry.get(
        "year_column"
    ) or _first_existing(
        set(frame),
        [
            "Year",
            "year",
            "season_year",
            "harvest_year",
        ],
    )

    if target is None:
        raise ValueError(
            f"TARGET_COLUMN_UNRESOLVED:{dataset_id}"
        )

    if sample_id is None:
        frame = frame.copy()
        sample_id = "sample_id"
        frame[sample_id] = [
            f"{dataset_id}__{index}"
            for index in range(len(frame))
        ]

    if year is None:
        raise ValueError(
            f"YEAR_COLUMN_UNRESOLVED:{dataset_id}"
        )

    frame = frame.copy()

    frame[target] = pd.to_numeric(
        frame[target],
        errors="coerce",
    )

    frame[year] = pd.to_numeric(
        frame[year],
        errors="coerce",
    )

    frame[sample_id] = (
        frame[sample_id]
        .astype(str)
    )

    frame = frame.loc[
        frame[target].notna()
        & frame[year].notna()
    ].copy()

    if frame.empty:
        raise ValueError(
            f"NO_VALID_TARGET_YEAR_ROWS:{dataset_id}"
        )

    if frame[sample_id].duplicated().any():
        raise ValueError(
            f"DUPLICATE_SAMPLE_IDS:{dataset_id}"
        )

    declared_weather = registry_entry.get(
        "weather_features"
    )

    weather = _numeric_features(
        frame,
        declared_weather,
        prefixes=(
            "weather_",
            "rain",
            "precip",
            "tmin",
            "tmax",
            "temp",
            "solar",
            "radiation",
            "vap",
            "vpd",
            "humidity",
            "wind",
        ),
    )

    privileged = _numeric_features(
        frame,
        registry_entry.get(
            "privileged_features"
        ),
        prefixes=(
            "soil_",
            "genotype_",
            "management_",
            "ec_",
        ),
    )

    excluded = {
        target,
        sample_id,
        year,
    }

    weather = [
        column
        for column in weather
        if column not in excluded
    ]

    privileged = [
        column
        for column in privileged
        if (
            column not in excluded
            and column not in set(weather)
        )
    ]

    weather = [
        column
        for column in weather
        if frame[column].notna().any()
    ]

    privileged = [
        column
        for column in privileged
        if frame[column].notna().any()
    ]

    if not weather:
        raise ValueError(
            f"WEATHER_FEATURES_UNRESOLVED:{dataset_id}"
        )

    return UniversalDatasetV2(
        dataset_id=dataset_id,
        frame=frame,
        weather_features=tuple(weather),
        privileged_features=tuple(privileged),
        target_column=str(target),
        sample_id_column=str(sample_id),
        year_column=str(year),
        target_unit=str(
            registry_entry.get(
                "target_unit",
                "UNKNOWN",
            )
        ),
        sample_unit=str(
            registry_entry.get(
                "sample_unit",
                "UNKNOWN",
            )
        ),
        source_path=path,
    )


def temporal_split_v2(
    dataset: UniversalDatasetV2,
) -> dict[str, list[str]]:
    frame = dataset.frame
    year_column = dataset.year_column
    sample_id = dataset.sample_id_column

    years = sorted(
        int(value)
        for value in frame[
            year_column
        ].dropna().unique()
    )

    if len(years) < 3:
        raise ValueError(
            f"INSUFFICIENT_TEMPORAL_YEARS:{dataset.dataset_id}:{years}"
        )

    test_year = years[-1]
    validation_year = years[-2]

    train = frame.loc[
        frame[year_column] < validation_year
    ]

    validation = frame.loc[
        frame[year_column] == validation_year
    ]

    test = frame.loc[
        frame[year_column] == test_year
    ]

    if min(
        len(train),
        len(validation),
        len(test),
    ) == 0:
        raise ValueError(
            f"EMPTY_TEMPORAL_SPLIT:{dataset.dataset_id}"
        )

    return {
        "train_ids": (
            train[sample_id]
            .astype(str)
            .tolist()
        ),
        "validation_ids": (
            validation[sample_id]
            .astype(str)
            .tolist()
        ),
        "test_ids": (
            test[sample_id]
            .astype(str)
            .tolist()
        ),
        "validation_year": validation_year,
        "test_year": test_year,
    }


def dataset_contract_v2(
    dataset: UniversalDatasetV2,
) -> dict[str, Any]:
    frame = dataset.frame

    return {
        "dataset_id": dataset.dataset_id,
        "path": str(dataset.source_path),
        "path_sha256": file_sha256(
            dataset.source_path
        ),
        "rows": int(len(frame)),
        "weather_feature_count": len(
            dataset.weather_features
        ),
        "privileged_feature_count": len(
            dataset.privileged_features
        ),
        "weather_features": list(
            dataset.weather_features
        ),
        "privileged_features": list(
            dataset.privileged_features
        ),
        "target_column": (
            dataset.target_column
        ),
        "sample_id_column": (
            dataset.sample_id_column
        ),
        "year_column": dataset.year_column,
        "target_unit": dataset.target_unit,
        "sample_unit": dataset.sample_unit,
        "target_unique": int(
            frame[
                dataset.target_column
            ].nunique(
                dropna=True
            )
        ),
        "year_min": int(
            frame[
                dataset.year_column
            ].min()
        ),
        "year_max": int(
            frame[
                dataset.year_column
            ].max()
        ),
        "year_count": int(
            frame[
                dataset.year_column
            ].nunique()
        ),
        "weather_missing_fraction": float(
            frame[
                list(
                    dataset.weather_features
                )
            ].isna().mean().mean()
        ),
    }
