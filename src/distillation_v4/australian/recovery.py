from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def keyed_weather_join(samples: pd.DataFrame, weather: pd.DataFrame, *, keys: tuple[str, ...]) -> pd.DataFrame:
    if not keys or any(key not in samples or key not in weather for key in keys):
        raise ValueError("WEATHER_JOIN_KEYS_NOT_AVAILABLE")
    if weather.duplicated(list(keys)).any():
        raise ValueError("WEATHER_JOIN_NOT_MANY_TO_ONE")
    result = samples.merge(weather, on=list(keys), how="left", validate="many_to_one", suffixes=("", "_weather"))
    if len(result) != len(samples):
        raise RuntimeError("WEATHER_JOIN_CHANGED_SAMPLE_UNIT")
    return result


def nearest_soil_join(samples: pd.DataFrame, soil: pd.DataFrame, *, max_distance_km: float) -> pd.DataFrame:
    required = {"latitude", "longitude"}
    if not required.issubset(samples) or not required.issubset(soil):
        raise ValueError("SOIL_COORDINATE_CONTRACT_MISSING")
    soil_values = soil[["latitude", "longitude"]].to_numpy(float)
    output = samples.copy(); indices, distances = [], []
    for lat, lon in samples[["latitude", "longitude"]].to_numpy(float):
        dlat = np.radians(soil_values[:, 0] - lat); dlon = np.radians(soil_values[:, 1] - lon)
        a = np.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * np.cos(np.radians(soil_values[:, 0])) * np.sin(dlon / 2) ** 2
        values = 6371.0088 * 2 * np.arcsin(np.sqrt(a)); index = int(np.argmin(values))
        indices.append(index); distances.append(float(values[index]))
    covariates = soil.iloc[indices].drop(columns=["latitude", "longitude"]).reset_index(drop=True).add_prefix("soil_")
    output = pd.concat([output.reset_index(drop=True), covariates], axis=1)
    output["soil_join_distance_km"] = distances
    output["soil_join_eligible"] = output.soil_join_distance_km.le(max_distance_km)
    return output


def recovery_contract(asset_registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"asset": name, "role": value["role"], "status": value["status"], "may_define_target": False, "join_requires_declared_keys_or_coordinates": True} for name, value in sorted(asset_registry.items())]
