from __future__ import annotations

from typing import Any

import pandas as pd


FORBIDDEN_TOKENS = ("yield", "production", "harvest_area", "planted_area", "target")


def harmonise_frame(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rename = {
        spec["id_column"]: "sample_id",
        spec["year_column"]: "year",
        spec["target_column"]: "target_yield",
    }
    if spec.get("crop_column") in frame:
        rename[spec["crop_column"]] = "crop"
    out = frame.rename(columns=rename).copy()
    # Explicit semantic harmonisation; these are aggregate-to-aggregate mappings,
    # never row alignment across sample units.
    rain = next((c for c in ("weather_silo_rain_sum_mm", "weather_rain_sum_mm", "silo_monthly_rain_sum_apr_oct") if c in out), None)
    evap = next((c for c in ("weather_silo_evap_pan_sum_mm", "weather_evap_pan_sum_mm", "silo_monthly_open_pan_evaporation_sum_apr_oct") if c in out), None)
    mean_temp = next((c for c in ("silo_mean_daily_air_temperature_mean_apr_oct",) if c in out), None)
    max_temp = next((c for c in ("weather_silo_max_temp_mean_c", "weather_max_temp_mean_c") if c in out), None)
    min_temp = next((c for c in ("weather_silo_min_temp_mean_c", "weather_min_temp_mean_c") if c in out), None)
    if rain: out["harmonised_weather__rain_sum"] = pd.to_numeric(out[rain], errors="coerce")
    if evap: out["harmonised_weather__evap_sum"] = pd.to_numeric(out[evap], errors="coerce")
    if mean_temp: out["harmonised_weather__temp_mean"] = pd.to_numeric(out[mean_temp], errors="coerce")
    elif max_temp and min_temp: out["harmonised_weather__temp_mean"] = (pd.to_numeric(out[max_temp], errors="coerce") + pd.to_numeric(out[min_temp], errors="coerce")) / 2
    if "crop" not in out:
        out["crop"] = spec.get("crop", "unknown")
    excluded = set(rename) | set(spec.get("blocked_columns", []))
    numeric = [c for c in frame.select_dtypes(include="number").columns if c not in excluded and not any(token in c.lower() for token in FORBIDDEN_TOKENS)]
    include_tokens = tuple(str(token).lower() for token in spec.get("feature_include_tokens", []))
    if include_tokens:
        numeric = [c for c in numeric if c.startswith("harmonised_weather__") or any(token in c.lower() for token in include_tokens)]
    weather = [c for c in numeric if any(t in c.lower() for t in ("rain", "precip", "temp", "weather", "silo", "vpd", "radiation"))]
    soil = [c for c in numeric if any(t in c.lower() for t in ("soil", "slga", "apsoil", "clay", "sand", "ph_", "soc"))]
    deployable = sorted(set(numeric) - set(soil))
    split_columns = [c for c in ("plot", "region_id", "Eastings_UTM", "Northings_UTM", "crop_product") if c in out]
    keep = list(dict.fromkeys(["sample_id", "year", "crop", "target_yield"] + split_columns + sorted(set(deployable + soil))))
    out = out.loc[:, [c for c in keep if c in out]].dropna(subset=["sample_id", "target_yield"])
    if out["sample_id"].astype(str).duplicated().any():
        raise ValueError("DUPLICATE_SAMPLE_IDS_WITHIN_NATIVE_SAMPLE_UNIT")
    return out, {"deployable": deployable, "weather": weather, "soil": soil}
