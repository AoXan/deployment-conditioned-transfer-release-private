from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitArtifact:
    split_id: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    algorithm: str
    random_state: int | None
    grouping_field: str | None = None

    @property
    def id_hash(self) -> str:
        payload = "\n".join(
            [self.split_id, *sorted(self.train_ids), "--VAL--", *sorted(self.validation_ids), "--TEST--", *sorted(self.test_ids)]
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def _ids(frame: pd.DataFrame, positions: Iterable[int]) -> tuple[str, ...]:
    return tuple(frame.iloc[list(positions)]["sample_id"].astype(str))


def _assert_partition(split: SplitArtifact, all_ids: set[str]) -> SplitArtifact:
    train, val, test = map(set, (split.train_ids, split.validation_ids, split.test_ids))
    if train & val or train & test or val & test:
        raise ValueError("SPLIT_OVERLAP")
    if train | val | test != all_ids:
        raise ValueError("SPLIT_NOT_EXHAUSTIVE")
    if not train or not val or not test:
        raise ValueError("EMPTY_SPLIT_PARTITION")
    return split


def build_random_split(frame: pd.DataFrame, *, seed: int) -> SplitArtifact:
    indices = np.arange(len(frame))
    train_val, test = train_test_split(indices, test_size=0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.2, random_state=seed)
    split = SplitArtifact("RANDOM", _ids(frame, train), _ids(frame, val), _ids(frame, test), "train_test_split_64_16_20", seed)
    return _assert_partition(split, set(frame.sample_id.astype(str)))


def build_group_split(frame: pd.DataFrame, *, group_column: str, seed: int) -> SplitArtifact:
    if group_column not in frame or frame[group_column].nunique(dropna=False) < 3:
        raise ValueError("GROUP_FIELD_REQUIRED")
    groups = frame[group_column].astype("object").where(frame[group_column].notna(), "__MISSING__").astype(str)
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups))
    remaining = frame.iloc[train_val]
    remaining_groups = groups.iloc[train_val]
    train_local, val_local = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(remaining, groups=remaining_groups)
    )
    split = SplitArtifact(
        "GROUP",
        _ids(frame, np.asarray(train_val)[train_local]),
        _ids(frame, np.asarray(train_val)[val_local]),
        _ids(frame, test),
        "group_shuffle_64_16_20",
        seed,
        group_column,
    )
    return _assert_partition(split, set(frame.sample_id.astype(str)))


def build_spatial_split(frame: pd.DataFrame, *, latitude: str, longitude: str) -> SplitArtifact:
    if latitude not in frame or longitude not in frame:
        raise ValueError("SPATIAL_COORDINATES_REQUIRED")
    lat = pd.to_numeric(frame[latitude], errors="coerce")
    lon = pd.to_numeric(frame[longitude], errors="coerce")
    if lat.isna().any() or lon.isna().any() or lat.nunique() < 2 or lon.nunique() < 2:
        raise ValueError("SPATIAL_COORDINATES_REQUIRED")
    lat_bin = pd.cut(lat, bins=4, labels=False, include_lowest=True)
    lon_bin = pd.cut(lon, bins=4, labels=False, include_lowest=True)
    blocks = (lat_bin.astype(int) * 4 + lon_bin.astype(int)).astype(int)
    unique = sorted(blocks.unique())
    if len(unique) < 5:
        raise ValueError("INSUFFICIENT_SPATIAL_BLOCKS")
    test_blocks = set(unique[::5])
    remaining = [value for value in unique if value not in test_blocks]
    validation_blocks = set(remaining[::4])
    test = np.flatnonzero(blocks.isin(test_blocks).to_numpy())
    val = np.flatnonzero(blocks.isin(validation_blocks).to_numpy())
    train = np.flatnonzero(~blocks.isin(test_blocks | validation_blocks).to_numpy())
    split = SplitArtifact("SPATIAL", _ids(frame, train), _ids(frame, val), _ids(frame, test), "strict_4x4_block_holdout", None, "spatial_block")
    return _assert_partition(split, set(frame.sample_id.astype(str)))


def build_temporal_split(frame: pd.DataFrame, *, year_column: str, split_id: str = "EXISTING_STAGE8_SPLIT") -> SplitArtifact:
    years = sorted(pd.to_numeric(frame[year_column], errors="coerce").dropna().astype(int).unique())
    if len(years) < 3:
        raise ValueError("INSUFFICIENT_TEMPORAL_YEARS")
    validation_year, test_year = years[-2], years[-1]
    values = pd.to_numeric(frame[year_column], errors="coerce")
    split = SplitArtifact(
        split_id,
        tuple(frame.loc[values < validation_year, "sample_id"].astype(str)),
        tuple(frame.loc[values == validation_year, "sample_id"].astype(str)),
        tuple(frame.loc[values == test_year, "sample_id"].astype(str)),
        f"frozen_temporal_validation_{validation_year}_test_{test_year}",
        None,
        year_column,
    )
    return _assert_partition(split, set(frame.sample_id.astype(str)))


def build_rolling_split(frame: pd.DataFrame, *, year_column: str, test_year: int) -> SplitArtifact:
    values = pd.to_numeric(frame[year_column], errors="coerce")
    eligible_years = sorted(values.dropna().astype(int).unique())
    prior = [year for year in eligible_years if year < test_year]
    if test_year not in eligible_years or len(prior) < 2:
        raise ValueError(f"INSUFFICIENT_ROLLING_HISTORY:{test_year}")
    validation_year = prior[-1]
    train = tuple(frame.loc[values < validation_year, "sample_id"].astype(str))
    validation = tuple(frame.loc[values == validation_year, "sample_id"].astype(str))
    test = tuple(frame.loc[values == test_year, "sample_id"].astype(str))
    split = SplitArtifact(
        f"ROLLING_{test_year}", train, validation, test,
        f"rolling_origin_validation_{validation_year}_test_{test_year}", None, year_column,
    )
    selected = set(train) | set(validation) | set(test)
    return _assert_partition(split, selected)


def deterministic_fraction_ids(ids: Iterable[str], fractions: list[float], seed: int) -> dict[float, list[str]]:
    ordered = sorted(set(map(str, ids)), key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    return {fraction: ordered[: max(1, int(np.ceil(len(ordered) * fraction)))] for fraction in sorted(fractions)}


def build_contract_splits(frame: pd.DataFrame, contract: dict, *, seed: int = 101) -> list[SplitArtifact]:
    """Resolve every declared split without substituting a different algorithm."""
    splits: list[SplitArtifact] = []
    for split_id in contract["splits"]:
        if split_id == "RANDOM":
            split = build_random_split(frame, seed=seed)
        elif split_id == "GROUP":
            split = build_group_split(frame, group_column=contract["group_column"], seed=seed)
        elif split_id == "SPATIAL":
            split = build_spatial_split(
                frame,
                latitude=contract["latitude_column"],
                longitude=contract["longitude_column"],
            )
        elif split_id == "EXISTING_STAGE8_SPLIT":
            split = build_temporal_split(frame, year_column=contract["year_column"], split_id=split_id)
        elif split_id.startswith("ROLLING_") and split_id.removeprefix("ROLLING_").isdigit():
            split = build_rolling_split(frame, year_column=contract["year_column"], test_year=int(split_id.removeprefix("ROLLING_")))
        else:
            raise ValueError(f"UNSUPPORTED_SPLIT_NO_FALLBACK:{split_id}")
        splits.append(split)
    return splits
