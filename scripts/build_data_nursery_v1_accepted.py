#!/usr/bin/env python3
"""Build Stage 6 data_nursery_v1 artifacts.

This script is intentionally ETL-only: it creates auditable data views,
manifests, fold definitions, and failure ledgers. It does not train models or
edit any Stage 1-5 artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "data_nursery_v1"
VERSION = "data_nursery_v1"
CREATED_AT = datetime.now().replace(microsecond=0).isoformat()
FAST_GZIP = {"method": "gzip", "compresslevel": 1}
LOCAL_DEPS = OUT / "_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

TRAIN = ROOT / "G2F" / "GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025" / "Training_data"

SILO_API = "https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php"
SILO_VARS = "RXTNEVP"
SILO_EMAIL = os.environ.get("SILO_USERNAME")
SILO_PASSWORD = os.environ.get("SILO_PASSWORD")
if not SILO_EMAIL or not SILO_PASSWORD:
    try:
        import yaml
        with open(ROOT / "configs" / "stage1_public.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if not SILO_EMAIL:
                SILO_EMAIL = cfg.get("silo", {}).get("username")
            if not SILO_PASSWORD:
                SILO_PASSWORD = cfg.get("silo", {}).get("password")
    except Exception:
        pass
if not SILO_EMAIL:
    SILO_EMAIL = "stage6-data-nursery@example.invalid"
if not SILO_PASSWORD:
    SILO_PASSWORD = ""


STATE_COORDS = {
    "New South Wales": (-34.29, 146.05),
    "South Australia": (-34.95, 138.60),
    "Victoria": (-36.75, 144.28),
    "Western Australia": (-31.95, 115.86),
    "Queensland": (-27.47, 153.03),
    "Tasmania": (-42.88, 147.33),
    "Australian Capital Territory": (-35.28, 149.13),
    "Northern Territory": (-12.46, 130.84),
}

STATE_SLUG = {
    "New South Wales": "new_south_wales",
    "South Australia": "south_australia",
    "Victoria": "victoria",
    "Western Australia": "western_australia",
    "Queensland": "queensland",
    "Tasmania": "tasmania",
    "Australian Capital Territory": "australian_capital_territory",
    "Northern Territory": "northern_territory",
}

KEY_COLUMNS = {
    "nursery_version",
    "track_id",
    "view_id",
    "dataset_id",
    "sample_id",
    "sample_unit",
    "target_name",
    "target_value",
    "target_unit",
    "Year",
    "year",
    "Env",
    "Hybrid",
    "target_yield",
    "plot_replicates",
    "Field_Location",
}

LEDGER_SCHEMAS = {
    "asset": ["asset_id", "track", "root", "path_patterns", "observed_paths", "file_count", "size_mb", "asset_type", "current_status", "prediction_task_candidate", "feature_role", "next_action", "notes", "stage6_status", "nursery_version"],
    "file": ["path", "root", "suffix", "size_bytes", "size_mb", "role", "status", "exists", "mtime", "sha256_or_truncated", "stage6_status"],
    "sheet": ["source_path", "sheet_or_table", "status", "rows_observed", "columns_observed", "parser", "message"],
    "download": ["request_id", "source", "latitude", "longitude", "start_date", "finish_date", "url", "cache_path", "attempted_at", "status", "rows", "bytes", "message"],
    "location": ["track_id", "view_id", "location_id", "location_name", "latitude", "longitude", "crs", "coordinate_source", "location_confidence", "field"],
    "join": ["track_id", "view_id", "join_name", "left_rows", "matched_rows", "coverage", "join_status"],
    "feature": ["track_id", "view_id", "feature_family", "source", "fold_local_transform_required", "feature_policy"],
    "fold": ["track_id", "view_id", "fold_id", "axis", "fold_name", "status", "train_rows", "test_rows", "manifest_path", "reason"],
    "failure": ["stage", "item", "error_type", "message", "status", "created_at"],
    "quarantine": ["track_id", "view_id", "sample_id", "source_path", "reason", "status"],
    "row_summary": ["table_path", "rows", "columns", "created_at", "dataset_id", "sample_unit", "raw_rows", "final_rows", "quarantined_rows", "status"],
}

META_CATEGORICAL = [
    "Experiment_Code",
    "Treatment",
    "City",
    "Farm",
    "Previous_Crop",
    "Pre-plant_tillage_method(s)",
    "In-season_tillage_method(s)",
    "Type_of_planter (fluted cone; belt cone; air planter)",
    "System_Determining_Moisture",
    "Irrigated",
]


@dataclass
class NurseryLog:
    asset_rows: list[dict[str, Any]]
    file_rows: list[dict[str, Any]]
    sheet_rows: list[dict[str, Any]]
    download_rows: list[dict[str, Any]]
    location_rows: list[dict[str, Any]]
    join_rows: list[dict[str, Any]]
    feature_rows: list[dict[str, Any]]
    fold_rows: list[dict[str, Any]]
    failure_rows: list[dict[str, Any]]
    quarantine_rows: list[dict[str, Any]]
    table_rows: list[dict[str, Any]]


LOG = NurseryLog([], [], [], [], [], [], [], [], [], [], [])


def ensure_dirs() -> None:
    for sub in [
        "ledgers",
        "manifests",
        "cache/silo",
        "cache/slga",
        "reports",
        "tracks/regional_crop_year/model_ready",
        "tracks/g2f_native/views",
        "tracks/g2f_environment_auxiliary/views",
        "tracks/precision_roseworthy/model_ready",
        "frozen_next_experiment_inputs",
    ]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path, limit_mb: int = 128) -> str:
    if not path.exists() or path.is_dir():
        return ""
    h = hashlib.sha256()
    max_bytes = limit_mb * 1_000_000
    seen = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1_000_000)
            if not chunk:
                break
            h.update(chunk)
            seen += len(chunk)
            if seen >= max_bytes:
                h.update(f"TRUNCATED_AT_{max_bytes}".encode())
                break
    return h.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": rel(path),
            "exists": False,
            "size_bytes": 0,
            "mtime": "",
            "sha256_or_truncated": "",
        }
    return {
        "path": rel(path),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "sha256_or_truncated": file_sha256(path),
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOG.table_rows.append(
        {
            "table_path": rel(path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "created_at": CREATED_AT,
        }
    )


def log_row_summary(track_id: str, view_id: str, dataset_id: str, sample_unit: str, raw_rows: int, final_rows: int, quarantined_rows: int, status: str) -> None:
    LOG.table_rows.append(
        {
            "table_path": f"ROW_SUMMARY::{track_id}/{view_id}",
            "rows": int(final_rows),
            "columns": "",
            "created_at": CREATED_AT,
            "dataset_id": dataset_id,
            "sample_unit": sample_unit,
            "raw_rows": int(raw_rows),
            "final_rows": int(final_rows),
            "quarantined_rows": int(quarantined_rows),
            "status": status,
        }
    )


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_md(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def frame_with_schema(rows: list[dict[str, Any]] | pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    if len(df.columns) == 0:
        df = pd.DataFrame(columns=columns)
    return df[[c for c in columns if c in df.columns] + [c for c in df.columns if c not in columns]]


def log_failure(stage: str, item: str, error_type: str, message: str, status: str = "failed") -> None:
    LOG.failure_rows.append(
        {
            "stage": stage,
            "item": item,
            "error_type": error_type,
            "message": str(message)[:1000],
            "status": status,
            "created_at": CREATED_AT,
        }
    )


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def first_non_null(values: pd.Series) -> object:
    clean = values.dropna()
    return clean.iloc[0] if len(clean) else np.nan


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def simple_random_folds(n: int, seed: int, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_frac)))
    test = np.sort(idx[:n_test])
    train = np.sort(idx[n_test:])
    return train, test


def group_kfold_indices(groups: pd.Series, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = pd.Series(groups.astype(str).unique()).sort_values().to_numpy()
    folds = [[] for _ in range(n_splits)]
    for i, g in enumerate(unique):
        folds[i % n_splits].append(g)
    out = []
    arr = groups.astype(str).to_numpy()
    for fold_groups in folds:
        test_mask = np.isin(arr, fold_groups)
        out.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return out


def add_fold_manifest(
    track_id: str,
    view_id: str,
    sample_ids: pd.Series,
    axis: str,
    fold_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    status: str = "supported",
    reason: str = "",
) -> None:
    fold_id = f"{view_id}__{axis}__{fold_name}"
    path = OUT / "tracks" / track_id / "folds" / f"{fold_id}.csv"
    if status == "supported":
        df = pd.concat(
            [
                pd.DataFrame({"sample_id": sample_ids.iloc[train_idx].to_numpy(), "split": "train"}),
                pd.DataFrame({"sample_id": sample_ids.iloc[test_idx].to_numpy(), "split": "test"}),
            ],
            ignore_index=True,
        )
        write_csv(df, path)
    LOG.fold_rows.append(
        {
            "track_id": track_id,
            "view_id": view_id,
            "fold_id": fold_id,
            "axis": axis,
            "fold_name": fold_name,
            "status": status,
            "train_rows": int(len(train_idx)) if status == "supported" else 0,
            "test_rows": int(len(test_idx)) if status == "supported" else 0,
            "manifest_path": rel(path) if status == "supported" else "",
            "reason": reason,
        }
    )


def missingness_report(df: pd.DataFrame, track_id: str, view_id: str, out_dir: Path) -> None:
    rows = []
    n = max(1, len(df))
    for col in df.columns:
        rows.append(
            {
                "track_id": track_id,
                "view_id": view_id,
                "column": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_fraction": float(df[col].isna().sum() / n),
                "dtype": str(df[col].dtype),
            }
        )
    write_csv(pd.DataFrame(rows), out_dir / "missingness_report.csv")


def duplicate_leakage_report(df: pd.DataFrame, track_id: str, view_id: str, out_dir: Path, key_cols: list[str]) -> None:
    dup_count = int(df.duplicated(key_cols).sum()) if all(c in df.columns for c in key_cols) else -1
    yield_like = [c for c in df.columns if c.lower().startswith("yield_") or "target" in c.lower()]
    blocked = [c for c in yield_like if c not in {"target_yield", "target_value", "target_name", "target_unit"}]
    write_csv(
        pd.DataFrame(
            [
                {
                    "track_id": track_id,
                    "view_id": view_id,
                    "key_columns": ",".join(key_cols),
                    "duplicate_key_rows": dup_count,
                    "yield_like_non_target_columns": ",".join(blocked),
                    "leakage_status": "PASS" if not blocked and dup_count == 0 else "REVIEW",
                }
            ]
        ),
        out_dir / "duplicate_leakage_audit.csv",
    )


def data_dictionary(df: pd.DataFrame, track_id: str, view_id: str, out_dir: Path) -> None:
    rows = []
    for col in df.columns:
        role = "feature"
        if col in {"sample_id", "track_id", "view_id", "dataset_id", "sample_unit"}:
            role = "identity"
        elif col in {"year", "Year", "season_start_date", "season_end_date", "forecast_cutoff_date"}:
            role = "time"
        elif col in {"target_value", "target_yield", "observed_yield_t_ha"}:
            role = "target"
        elif col in {"latitude", "longitude", "location_id", "region_id", "Env", "Field_Location"}:
            role = "location"
        elif col.startswith(("source_", "provenance_", "parser_", "nursery_")):
            role = "provenance"
        rows.append(
            {
                "track_id": track_id,
                "view_id": view_id,
                "field": col,
                "dtype": str(df[col].dtype),
                "role": role,
                "description": f"{role} field generated by Stage 6 data_nursery_v1 builder",
            }
        )
    write_csv(pd.DataFrame(rows), out_dir / "data_dictionary.csv")


def track_common_reports(df: pd.DataFrame, track_id: str, view_id: str, out_dir: Path, key_cols: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame([{"field": c, "dtype": str(df[c].dtype), "nullable": bool(df[c].isna().any())} for c in df.columns]), out_dir / "schema.csv")
    data_dictionary(df, track_id, view_id, out_dir)
    missingness_report(df, track_id, view_id, out_dir)
    duplicate_leakage_report(df, track_id, view_id, out_dir, key_cols)
    write_json(
        {
            "nursery_version": VERSION,
            "track_id": track_id,
            "view_id": view_id,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "created_at": CREATED_AT,
            "script": rel(Path(__file__)),
        },
        out_dir / "build_manifest.json",
    )


def scan_assets() -> None:
    registry = ROOT / "outputs" / "tables" / "stage5_data_asset_registry.csv"
    if registry.exists():
        df = pd.read_csv(registry)
        for _, row in df.iterrows():
            LOG.asset_rows.append(
                {
                    **row.to_dict(),
                    "stage6_status": "read_only_registered",
                    "nursery_version": VERSION,
                }
            )
    inventory = ROOT / "outputs" / "tables" / "stage5_file_level_asset_inventory.csv"
    if inventory.exists():
        inv = pd.read_csv(inventory)
        for _, row in inv.iterrows():
            path = ROOT / str(row["path"])
            meta = file_meta(path)
            LOG.file_rows.append({**row.to_dict(), **meta, "stage6_status": "inventoried_read_only"})

    excel_paths = [
        ROOT / "Agritech_Datasets" / "crop_yield" / "abs" / "AGCDCNAT_STATE202122.xlsx",
        ROOT / "Agritech_Datasets" / "crop_yield" / "abs" / "abs_historical_wheat_1860_2022.xlsx",
        ROOT / "Agritech_Datasets" / "crop_yield" / "csiro_dap" / "Waite_Trial_Data.xls",
    ] + sorted((ROOT / "Agritech_Datasets" / "crop_yield" / "abares").glob("*.xlsx"))
    for path in excel_paths:
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                LOG.sheet_rows.append(
                    {
                        "source_path": rel(path),
                        "sheet_or_table": sheet,
                        "status": "sheet_detected",
                        "rows_observed": "",
                        "columns_observed": "",
                        "parser": "pandas.ExcelFile",
                    }
                )
        except Exception as exc:
            LOG.sheet_rows.append(
                {
                    "source_path": rel(path),
                    "sheet_or_table": "",
                    "status": "sheet_scan_failed",
                    "rows_observed": "",
                    "columns_observed": "",
                    "parser": "pandas.ExcelFile",
                    "message": str(exc)[:500],
                }
            )
            log_failure("sheet_scan", rel(path), type(exc).__name__, str(exc), "failed")


def parse_apsoil() -> pd.DataFrame:
    path = ROOT / "Agritech_Datasets" / "soil" / "apsoil" / "APSRU-Australia-soils.soils"
    rows: list[dict[str, Any]] = []
    try:
        root = ET.parse(path).getroot()
        for soil in root.iter("Soil"):
            lat = soil.findtext("Latitude")
            lon = soil.findtext("Longitude")
            try:
                lat_f = float(lat) if lat is not None else np.nan
                lon_f = float(lon) if lon is not None else np.nan
            except ValueError:
                continue
            water = soil.find("Water")
            def doubles(tag: str) -> list[float]:
                node = water.find(tag) if water is not None else None
                vals = []
                if node is not None:
                    for sub in node:
                        try:
                            vals.append(float(sub.text))
                        except (TypeError, ValueError):
                            vals.append(np.nan)
                return vals

            th = doubles("Thickness")
            bd = doubles("BD")
            ll15 = doubles("LL15")
            dul = doubles("DUL")
            sat = doubles("SAT")
            pawc = np.nan
            if th and ll15 and dul:
                n = min(len(th), len(ll15), len(dul))
                pawc = float(np.nansum([(dul[i] - ll15[i]) * th[i] for i in range(n)]))
            rows.append(
                {
                    "apsoil_id": soil.findtext("RecordNumber") or soil.attrib.get("name", ""),
                    "apsoil_name": soil.attrib.get("name", ""),
                    "site": soil.findtext("Site"),
                    "nearest_town": soil.findtext("NearestTown"),
                    "region": soil.findtext("Region"),
                    "state": soil.findtext("State"),
                    "latitude": lat_f,
                    "longitude": lon_f,
                    "location_accuracy": soil.findtext("LocationAccuracy"),
                    "apsoil_pawc_mm": pawc,
                    "apsoil_bd_top": bd[0] if bd else np.nan,
                    "apsoil_ll15_top": ll15[0] if ll15 else np.nan,
                    "apsoil_dul_top": dul[0] if dul else np.nan,
                    "apsoil_sat_top": sat[0] if sat else np.nan,
                    "apsoil_depth_total_mm": float(np.nansum(th)) if th else np.nan,
                    "source_path": rel(path),
                    "source_hash": file_sha256(path),
                }
            )
    except Exception as exc:
        log_failure("apsoil_parse", rel(path), type(exc).__name__, str(exc), "failed")
    df = pd.DataFrame(rows)
    if not df.empty:
        write_csv(df, OUT / "manifests" / "apsoil_profile_manifest.csv")
    return df


def nearest_apsoil(apsoil: pd.DataFrame, lat: float, lon: float) -> dict[str, Any]:
    if apsoil.empty or pd.isna(lat) or pd.isna(lon):
        return {}
    lat_arr = pd.to_numeric(apsoil["latitude"], errors="coerce").to_numpy(dtype=float)
    lon_arr = pd.to_numeric(apsoil["longitude"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    if not valid.any():
        return {}
    r = 6371.0088
    p1 = math.radians(float(lat))
    p2 = np.radians(lat_arr[valid])
    dphi = np.radians(lat_arr[valid] - float(lat))
    dlambda = np.radians(lon_arr[valid] - float(lon))
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    distances = 2 * r * np.arcsin(np.sqrt(a))
    valid_indices = np.flatnonzero(valid)
    idx = int(valid_indices[int(np.nanargmin(distances))])
    row = apsoil.iloc[idx].to_dict()
    d = float(np.nanmin(distances))
    if d <= 50:
        confidence = "high_proxy"
    elif d <= 150:
        confidence = "medium_proxy"
    else:
        confidence = "low_proxy"
    return {
        "apsoil_match_id": row.get("apsoil_id"),
        "apsoil_match_name": row.get("apsoil_name"),
        "apsoil_match_state": row.get("state"),
        "apsoil_match_distance_km": d,
        "soil_match_confidence": confidence,
        "soil_source": "APSoil_nearest_profile_proxy",
        "apsoil_pawc_mm": row.get("apsoil_pawc_mm"),
        "apsoil_bd_top": row.get("apsoil_bd_top"),
        "apsoil_ll15_top": row.get("apsoil_ll15_top"),
        "apsoil_dul_top": row.get("apsoil_dul_top"),
        "apsoil_sat_top": row.get("apsoil_sat_top"),
        "apsoil_depth_total_mm": row.get("apsoil_depth_total_mm"),
    }


def silo_url(lat: float, lon: float, start: date, finish: date) -> str:
    query = urllib.parse.urlencode(
        {
            "lat": f"{lat:.4f}",
            "lon": f"{lon:.4f}",
            "start": start.strftime("%Y%m%d"),
            "finish": finish.strftime("%Y%m%d"),
            "format": "csv",
            "comment": SILO_VARS,
            "username": SILO_EMAIL,
            "password": SILO_PASSWORD,
        }
    )
    return f"{SILO_API}?{query}"


def try_download_silo(request_id: str, lat: float, lon: float, start: date, finish: date) -> Path | None:
    cache_path = OUT / "cache" / "silo" / f"{request_id}.csv"
    url = silo_url(lat, lon, start, finish)
    row = {
        "request_id": request_id,
        "source": "SILO DataDrill",
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "finish_date": finish.isoformat(),
        "url": url,
        "cache_path": rel(cache_path),
        "attempted_at": CREATED_AT,
    }
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = response.read()
        cache_path.write_bytes(data)
        parsed = pd.read_csv(cache_path)
        if len(parsed) == 0:
            raise ValueError("downloaded zero rows")
        row.update({"status": "download_success", "rows": int(len(parsed)), "bytes": int(len(data)), "message": ""})
        LOG.download_rows.append(row)
        return cache_path
    except Exception as exc:
        urllib_message = str(exc)[:500]
        try:
            result = subprocess.run(
                ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "20", "-o", str(cache_path), url],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"curl exit {result.returncode}")
            parsed = pd.read_csv(cache_path)
            if len(parsed) == 0:
                raise ValueError("curl fallback downloaded zero rows")
            row.update(
                {
                    "status": "download_success_curl_fallback",
                    "rows": int(len(parsed)),
                    "bytes": int(cache_path.stat().st_size),
                    "message": f"urllib failed first: {urllib_message}",
                }
            )
            LOG.download_rows.append(row)
            return cache_path
        except Exception as curl_exc:
            row.update({"status": "download_failed", "rows": 0, "bytes": 0, "message": f"urllib: {urllib_message}; curl: {str(curl_exc)[:500]}"})
            LOG.download_rows.append(row)
            log_failure("silo_download", request_id, type(curl_exc).__name__, row["message"], "failed")
            return None


def read_silo_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "YYYY-MM-DD" in df.columns:
        df["date"] = pd.to_datetime(df["YYYY-MM-DD"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        return pd.DataFrame()
    return df


def silo_features_for_location(name: str, lat: float, lon: float, year: int, source_hint: Path | None) -> dict[str, Any]:
    start = date(int(year), 4, 1)
    finish = date(int(year), 10, 31)
    request_id = f"{re.sub('[^a-z0-9]+','_',name.lower()).strip('_')}_{year}_{abs(hash((round(lat,4),round(lon,4),year))) % 100000}"
    path = None
    status = "missing"
    if source_hint is not None and source_hint.exists():
        path = source_hint
        status = "cache_hit_existing_raw"
        LOG.download_rows.append(
            {
                "request_id": request_id,
                "source": "existing local SILO raw cache",
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "finish_date": finish.isoformat(),
                "url": "",
                "cache_path": rel(path),
                "attempted_at": CREATED_AT,
                "status": status,
                "rows": "",
                "bytes": path.stat().st_size,
                "message": "validated existing data/raw SILO cache",
            }
        )
    else:
        path = try_download_silo(request_id, lat, lon, start, finish)
        status = "download_success" if path else "download_failed"
    if path is None:
        return {
            "weather_source_status": status,
            "weather_source_path": "",
            "weather_day_count": 0,
        }
    try:
        df = read_silo_rows(path)
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(finish))]
        for c in ["daily_rain", "max_temp", "min_temp", "evap_pan", "et_tall_crop", "vp", "et_morton_potential"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out = {
            "weather_source_status": status,
            "weather_source_path": rel(path),
            "weather_day_count": int(len(df)),
            "weather_rain_sum_mm": float(df["daily_rain"].sum()) if "daily_rain" in df else np.nan,
            "weather_rain_days": int((df["daily_rain"] > 0).sum()) if "daily_rain" in df else 0,
            "weather_max_temp_mean_c": float(df["max_temp"].mean()) if "max_temp" in df else np.nan,
            "weather_min_temp_mean_c": float(df["min_temp"].mean()) if "min_temp" in df else np.nan,
            "weather_heat_days_gt35": int((df["max_temp"] > 35).sum()) if "max_temp" in df else 0,
            "weather_frost_days_lt0": int((df["min_temp"] < 0).sum()) if "min_temp" in df else 0,
            "weather_evap_pan_sum_mm": float(df["evap_pan"].sum()) if "evap_pan" in df else np.nan,
            "weather_water_balance_proxy_mm": float(df["daily_rain"].sum() - df["evap_pan"].sum()) if {"daily_rain", "evap_pan"}.issubset(df.columns) else np.nan,
        }
        return out
    except Exception as exc:
        log_failure("silo_parse", rel(path), type(exc).__name__, str(exc), "failed")
        return {
            "weather_source_status": "parse_failed",
            "weather_source_path": rel(path),
            "weather_day_count": 0,
        }


def build_slga_request_manifest(locations: pd.DataFrame) -> None:
    attrs = ["clay", "sand", "silt", "ph", "soc", "bdw"]
    depths = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100)]
    rows = []
    api_key_present = bool(os.environ.get("TERN_API_KEY") or os.environ.get("SLGA_API_KEY"))
    for _, loc in locations.drop_duplicates("location_id").iterrows():
        for attr in attrs:
            for top, bottom in depths:
                rows.append(
                    {
                        "request_id": f"slga_{loc['location_id']}_{attr}_{top}_{bottom}",
                        "location_id": loc["location_id"],
                        "latitude": loc["latitude"],
                        "longitude": loc["longitude"],
                        "attribute": attr,
                        "depth_top_cm": top,
                        "depth_bottom_cm": bottom,
                        "status": "pending_external" if api_key_present else "blocked_missing_tern_or_slga_api_key",
                        "cache_path": "",
                        "source": "SLGA via TERN eSoil/SoilDataFederator",
                        "message": "Request manifest built; no credentialed SLGA extraction executed by nursery builder.",
                    }
                )
    write_csv(pd.DataFrame(rows), OUT / "cache" / "slga" / "slga_request_manifest.csv")


def build_regional(apsoil: pd.DataFrame) -> pd.DataFrame:
    source = ROOT / "data" / "processed" / "abs_wheat_yield_state_year.csv"
    df = pd.read_csv(source)
    rows = []
    locations = []
    for _, r in df.iterrows():
        state = r["state"]
        lat, lon = STATE_COORDS.get(state, (np.nan, np.nan))
        slug = STATE_SLUG.get(state, re.sub("[^a-z0-9]+", "_", str(state).lower()).strip("_"))
        year = int(r["year"])
        raw_silo = ROOT / "data" / "raw" / f"silo_{slug}_{year}.csv"
        weather = silo_features_for_location(slug, lat, lon, year, raw_silo if raw_silo.exists() else None)
        soil = nearest_apsoil(apsoil, lat, lon)
        sample_id = f"abs_wheat|{slug}|{year}"
        row = {
            "nursery_version": VERSION,
            "track_id": "regional_crop_year",
            "view_id": "regional_abs_wheat_weather_apsoil",
            "dataset_id": "abs_wheat_yield_state_year",
            "sample_id": sample_id,
            "sample_unit": "state_crop_year",
            "crop": "wheat",
            "region_id": state,
            "year": year,
            "target_name": "observed_yield",
            "target_value": float(r["observed_yield_t_ha"]),
            "target_unit": "t_ha",
            "production_t": float(r["production_kt"]) * 1000.0,
            "area_ha": float(r["area_kha"]) * 1000.0,
            "target_definition": "production_kt / area_kha from processed ABS-derived state-year wheat table",
            "latitude": lat,
            "longitude": lon,
            "coordinate_source": "state_representative_point_for_weather_linkage",
            "location_confidence": "state_proxy",
            "location_proxy_flag": True,
            "season_start_date": f"{year}-04-01",
            "season_end_date": f"{year}-10-31",
            "forecast_cutoff_date": f"{year}-10-31",
            "source_target_path": rel(source),
            "source_target_hash": file_sha256(source),
            **weather,
            **soil,
        }
        rows.append(row)
        locations.append(
            {
                "track_id": "regional_crop_year",
                "view_id": "regional_abs_wheat_weather_apsoil",
                "location_id": slug,
                "location_name": state,
                "latitude": lat,
                "longitude": lon,
                "crs": "EPSG:4326",
                "coordinate_source": "state_representative_point_for_weather_linkage",
                "location_confidence": "state_proxy",
            }
        )
    out = pd.DataFrame(rows)
    loc_df = pd.DataFrame(locations).drop_duplicates("location_id")
    LOG.location_rows.extend(loc_df.to_dict("records"))
    write_csv(out, OUT / "tracks" / "regional_crop_year" / "model_ready" / "regional_abs_wheat_weather_apsoil.csv")
    track_dir = OUT / "tracks" / "regional_crop_year"
    track_common_reports(out, "regional_crop_year", "regional_abs_wheat_weather_apsoil", track_dir, ["sample_id"])
    write_csv(loc_df, track_dir / "location_manifest.csv")
    write_csv(pd.DataFrame([{"source_path": rel(source), **file_meta(source)}]), track_dir / "provenance_manifest.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "track_id": "regional_crop_year",
                    "view_id": "regional_abs_wheat_weather_apsoil",
                    "target_name": "observed_yield",
                    "target_unit": "t_ha",
                    "target_definition": "ABS-derived wheat production/area at state-year sample unit.",
                }
            ]
        ),
        track_dir / "target_manifest.csv",
    )
    regional_features = [
        {
            "track_id": "regional_crop_year",
            "view_id": "regional_abs_wheat_weather_apsoil",
            "feature_family": fam,
            "source": src,
            "fold_local_transform_required": False,
            "feature_policy": "raw derived seasonal summaries or source keys only; no global scaling/imputation/selection",
        }
        for fam, src in [
            ("weather", "SILO existing cache or targeted DataDrill request"),
            ("soil", "nearest APSoil profile proxy; SLGA requests manifest only"),
            ("identity", "ABS state/year/crop keys"),
        ]
    ]
    LOG.feature_rows.extend(regional_features)
    write_csv(pd.DataFrame(regional_features), track_dir / "feature_manifest.csv")
    log_row_summary("regional_crop_year", "regional_abs_wheat_weather_apsoil", "abs_wheat_yield_state_year", "state_crop_year", len(df), len(out), 0, "PASS_WITH_LIMITATIONS")
    weather_cov = float((out["weather_day_count"] > 0).mean())
    soil_cov = float(out["apsoil_match_id"].notna().mean()) if "apsoil_match_id" in out else 0.0
    LOG.join_rows.extend(
        [
            {
                "track_id": "regional_crop_year",
                "view_id": "regional_abs_wheat_weather_apsoil",
                "join_name": "ABS_to_SILO",
                "left_rows": len(out),
                "matched_rows": int((out["weather_day_count"] > 0).sum()),
                "coverage": weather_cov,
                "join_status": "PASS_WITH_LIMITATIONS" if weather_cov >= 0.5 else "PARTIAL",
            },
            {
                "track_id": "regional_crop_year",
                "view_id": "regional_abs_wheat_weather_apsoil",
                "join_name": "ABS_to_APSoil_nearest_profile",
                "left_rows": len(out),
                "matched_rows": int(out["apsoil_match_id"].notna().sum()) if "apsoil_match_id" in out else 0,
                "coverage": soil_cov,
                "join_status": "PASS_WITH_LIMITATIONS" if soil_cov >= 0.9 else "PARTIAL",
            },
        ]
    )
    write_csv(pd.DataFrame(LOG.join_rows), OUT / "ledgers" / "join_coverage_ledger.csv")
    write_csv(pd.DataFrame([x for x in LOG.join_rows if x["track_id"] == "regional_crop_year"]), track_dir / "join_coverage_report.csv")
    build_slga_request_manifest(loc_df)
    build_regional_folds(out)
    write_md(dataset_card_text("regional_crop_year", "regional_abs_wheat_weather_apsoil", out, "PASS_WITH_LIMITATIONS", "State-year target with proxy coordinates and APSoil nearest-profile soil. SILO cache coverage controls model readiness."), track_dir / "dataset_card.md")
    return out


def build_regional_folds(df: pd.DataFrame) -> None:
    sid = df["sample_id"]
    for seed in [101, 202, 303]:
        tr, te = simple_random_folds(len(df), seed)
        add_fold_manifest("regional_crop_year", "regional_abs_wheat_weather_apsoil", sid, "regional_random_diagnostic", f"seed_{seed}", tr, te)
    for year in sorted(df["year"].unique()):
        if year <= int(df["year"].min()):
            continue
        tr = np.where(df["year"].to_numpy() < year)[0]
        te = np.where(df["year"].to_numpy() == year)[0]
        if len(tr) >= 5 and len(te) >= 1:
            add_fold_manifest("regional_crop_year", "regional_abs_wheat_weather_apsoil", sid, "temporal_forward", f"test_{year}", tr, te)
    for region, part in df.groupby("region_id"):
        te = part.index.to_numpy()
        tr = df.index.difference(te).to_numpy()
        if len(tr) >= 5 and len(te) >= 1:
            add_fold_manifest("regional_crop_year", "regional_abs_wheat_weather_apsoil", sid, "region_holdout", re.sub("[^a-z0-9]+", "_", region.lower()).strip("_"), tr, te)
    add_fold_manifest("regional_crop_year", "regional_abs_wheat_weather_apsoil", sid, "spatiotemporal", "not_supported", np.array([]), np.array([]), "not_supported", "state-year table too small for non-degenerate joint region-year holdout")


def safe_numeric_frame(df: pd.DataFrame, key: str, prefix: str, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    numeric: dict[str, pd.Series] = {}
    for col in df.columns:
        if col == key or col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            numeric[f"{prefix}{col}"] = series
    if not numeric:
        return df[[key]].drop_duplicates().reset_index(drop=True)
    out = pd.DataFrame({key: df[key], **numeric})
    return out.groupby(key, as_index=False).mean(numeric_only=True)


def aggregate_categorical(df: pd.DataFrame, key: str, prefix: str, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df[[key]].drop_duplicates().reset_index(drop=True)
    out = df[[key] + existing].copy()
    grouped = out.groupby(key, as_index=False).agg({c: first_non_null for c in existing})
    return grouped.rename(columns={c: f"{prefix}{c}" for c in existing})


def aggregate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    numeric_cols = [c for c in weather.columns if c not in {"Env", "Date"}]
    for col in numeric_cols:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    agg = weather.groupby("Env")[numeric_cols].agg(["mean", "sum", "min", "max"])
    agg.columns = [f"weather_{col}_{stat}" for col, stat in agg.columns]
    days = weather.groupby("Env").size().rename("weather_days")
    return agg.join(days).reset_index()


def aggregate_target(trait: pd.DataFrame) -> pd.DataFrame:
    trait = trait.dropna(subset=["Yield_Mg_ha", "Hybrid", "Env"]).copy()
    trait["Yield_Mg_ha"] = pd.to_numeric(trait["Yield_Mg_ha"], errors="coerce")
    trait = trait.dropna(subset=["Yield_Mg_ha"])
    grouped = (
        trait.groupby(["Env", "Year", "Hybrid"], as_index=False)
        .agg(
            target_yield=("Yield_Mg_ha", "mean"),
            plot_replicates=("Yield_Mg_ha", "size"),
            Field_Location=("Field_Location", first_non_null),
        )
        .reset_index(drop=True)
    )
    grouped["sample_id"] = grouped["Hybrid"].astype(str) + "|" + grouped["Env"].astype(str)
    return grouped


def read_g2f_feature_table() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    paths = {
        "trait": TRAIN / "1_Training_Trait_Data_2014_2023.csv",
        "meta": TRAIN / "2_Training_Meta_Data_2014_2023.csv",
        "soil": TRAIN / "3_Training_Soil_Data_2015_2023.csv",
        "weather": TRAIN / "4_Training_Weather_Data_2014_2023_seasons_only.csv",
        "ec": TRAIN / "6_Training_EC_Data_2014_2023.csv",
        "genotype_numeric": TRAIN / "5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt",
    }
    trait = pd.read_csv(paths["trait"], usecols=["Env", "Year", "Field_Location", "Hybrid", "Yield_Mg_ha"])
    meta = pd.read_csv(paths["meta"])
    soil = pd.read_csv(paths["soil"])
    weather = pd.read_csv(paths["weather"])
    ec = pd.read_csv(paths["ec"])
    target = aggregate_target(trait)
    meta_num = safe_numeric_frame(meta, "Env", "meta_", exclude={"Year"})
    meta_cat = aggregate_categorical(meta, "Env", "meta_", META_CATEGORICAL)
    weather_agg = aggregate_weather(weather)
    soil_num = safe_numeric_frame(soil, "Env", "soil_", exclude={"Year"})
    ec_exclude = {c for c in ec.columns if c.lower().startswith("yield_")}
    ec_num = safe_numeric_frame(ec, "Env", "ec_", exclude=ec_exclude)
    features = target.merge(meta_num, on="Env", how="left")
    features = features.merge(meta_cat, on="Env", how="left")
    features = features.merge(weather_agg, on="Env", how="left")
    features = features.merge(soil_num, on="Env", how="left")
    features = features.merge(ec_num, on="Env", how="left")
    features = features.sort_values(["Year", "Env", "Hybrid"]).reset_index(drop=True)
    features.insert(0, "nursery_version", VERSION)
    features.insert(1, "track_id", "g2f_native")
    features.insert(2, "dataset_id", "g2f_2025_competition_training")
    features.insert(3, "sample_unit", "hybrid_env")
    features.insert(4, "target_name", "Yield_Mg_ha")
    features.insert(5, "target_unit", "Mg_ha")
    features["source_trait_path"] = rel(paths["trait"])
    features["parser_version"] = "build_data_nursery_v1.g2f_feature_table"
    genotype_hybrid_count = ""
    genotype_marker_count = ""
    try:
        with paths["genotype_numeric"].open("r", errors="replace") as handle:
            handle.readline()
            header = handle.readline().rstrip("\n").split("\t")
        genotype_marker_count = max(0, len(header) - 1)
        genotype_hybrid_count = sum(1 for _ in paths["genotype_numeric"].open("r", errors="replace")) - 2
    except Exception as exc:
        log_failure("g2f_genotype_header", rel(paths["genotype_numeric"]), type(exc).__name__, str(exc), "failed")
    audit = {
        "trait_rows": int(len(trait)),
        "target_rows": int(len(features)),
        "years": int(features["Year"].nunique()),
        "year_min": int(features["Year"].min()),
        "year_max": int(features["Year"].max()),
        "environments": int(features["Env"].nunique()),
        "hybrids": int(features["Hybrid"].nunique()),
        "meta_join_coverage": float(features.filter(like="meta_").notna().any(axis=1).mean()),
        "weather_join_coverage": float(features.filter(like="weather_").notna().any(axis=1).mean()),
        "soil_join_coverage": float(features.filter(like="soil_").notna().any(axis=1).mean()),
        "ec_join_coverage": float(features.filter(like="ec_").notna().any(axis=1).mean()),
        "ec_yield_like_columns_excluded": int(len(ec_exclude)),
        "feature_columns_before_genotype": int(len([c for c in features.columns if c not in KEY_COLUMNS])),
        "genotype_hybrids_numeric_file": genotype_hybrid_count,
        "genotype_marker_columns_numeric_file": genotype_marker_count,
    }
    provenance = {k: file_meta(v) for k, v in paths.items()}
    return features, audit, provenance


def g2f_cols(df: pd.DataFrame, families: list[str], include_meta_keys: bool = True) -> list[str]:
    base = [
        "nursery_version",
        "track_id",
        "dataset_id",
        "sample_unit",
        "sample_id",
        "Env",
        "Year",
        "Hybrid",
        "target_name",
        "target_yield",
        "target_unit",
        "plot_replicates",
        "Field_Location",
    ]
    cols = list(base)
    prefixes = []
    for fam in families:
        if fam == "meta":
            prefixes.append("meta_")
        elif fam == "weather":
            prefixes.append("weather_")
        elif fam == "soil":
            prefixes.append("soil_")
        elif fam == "ec":
            prefixes.append("ec_")
    for c in df.columns:
        if c in cols:
            continue
        if any(c.startswith(p) for p in prefixes):
            cols.append(c)
    if include_meta_keys:
        for c in ["source_trait_path", "parser_version"]:
            if c in df.columns and c not in cols:
                cols.append(c)
    return [c for c in cols if c in df.columns]


def build_g2f(apsoil: pd.DataFrame | None = None) -> pd.DataFrame:
    df, audit, provenance = read_g2f_feature_table()
    g2f_dir = OUT / "tracks" / "g2f_native"
    write_json(audit, g2f_dir / "g2f_native_build_audit.json")
    write_json(provenance, g2f_dir / "provenance_manifest.json")

    view_defs = {
        "g2f_native_hybrid_env": ["meta", "weather", "soil", "ec"],
        "g2f_native_no_genotype": ["meta", "weather", "soil", "ec"],
        "g2f_weather_only": ["weather"],
        "g2f_soil_only": ["soil"],
        "g2f_weather_soil": ["weather", "soil"],
        "g2f_full_modal": ["meta", "weather", "soil", "ec"],
        "g2f_ec_excluded_sensitivity": ["meta", "weather", "soil"],
    }
    feature_manifest_rows = []
    for view_id, fams in view_defs.items():
        view = df[g2f_cols(df, fams)].copy()
        view["view_id"] = view_id
        if view_id == "g2f_native_no_genotype":
            view["hybrid_identity_feature_policy"] = "blocked_from_features_key_retained_for_sample_identity"
        path = g2f_dir / "views" / f"{view_id}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        view.to_csv(path, index=False, compression=FAST_GZIP)
        LOG.table_rows.append({"table_path": rel(path), "rows": int(len(view)), "columns": int(len(view.columns)), "created_at": CREATED_AT})
        track_common_reports(view.head(5000).copy(), "g2f_native", view_id, g2f_dir / "views" / view_id, ["sample_id"])
        missingness_report(view, "g2f_native", view_id, g2f_dir / "views" / view_id)
        duplicate_leakage_report(view, "g2f_native", view_id, g2f_dir / "views" / view_id, ["sample_id"])
        for fam in fams:
            feature_manifest_rows.append(
                {
                    "track_id": "g2f_native",
                    "view_id": view_id,
                    "feature_family": fam,
                    "fold_local_transform_required": fam == "genotype",
                    "source": "G2F 2025 competition training files",
                    "feature_policy": "allowed raw covariates; EC yield_* columns excluded; genotype PCs not precomputed in nursery",
                }
            )
        log_row_summary("g2f_native", view_id, "g2f_2025_competition_training", "hybrid_env", audit["trait_rows"], len(view), 0, "PASS_WITH_LIMITATIONS")
    LOG.feature_rows.extend(feature_manifest_rows)
    write_csv(pd.DataFrame(feature_manifest_rows), g2f_dir / "feature_manifest.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "track_id": "g2f_native",
                    "view_id": "all_g2f_native_views",
                    "target_name": "Yield_Mg_ha",
                    "target_unit": "Mg_ha",
                    "target_definition": "mean Yield_Mg_ha over plot replicates for same Hybrid x Env.",
                }
            ]
        ),
        g2f_dir / "target_manifest.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "track_id": "g2f_native",
                    "view_id": "all_g2f_native_views",
                    "join_name": "target_to_meta_weather_soil_ec",
                    "left_rows": audit["target_rows"],
                    "matched_rows": audit["target_rows"],
                    "meta_coverage": audit["meta_join_coverage"],
                    "weather_coverage": audit["weather_join_coverage"],
                    "soil_coverage": audit["soil_join_coverage"],
                    "ec_coverage": audit["ec_join_coverage"],
                    "join_status": "PASS_WITH_LIMITATIONS",
                }
            ]
        ),
        g2f_dir / "join_coverage_report.csv",
    )
    LOG.join_rows.append(
        {
            "track_id": "g2f_native",
            "view_id": "all_g2f_native_views",
            "join_name": "target_to_meta_weather_soil_ec",
            "left_rows": audit["target_rows"],
            "matched_rows": audit["target_rows"],
            "coverage": min(audit["meta_join_coverage"], audit["weather_join_coverage"], audit["soil_join_coverage"], audit["ec_join_coverage"]),
            "join_status": "PASS_WITH_LIMITATIONS",
        }
    )
    build_g2f_folds(df, "g2f_native", "g2f_native_hybrid_env", native=True)
    write_md(dataset_card_text("g2f_native", "g2f_native_hybrid_env", df, "PASS_WITH_LIMITATIONS", f"Native Hybrid x Env G2F training table. Audit: {json.dumps(audit, sort_keys=True)}"), g2f_dir / "dataset_card.md")
    build_g2f_environment_auxiliary(df)
    return df


def build_g2f_folds(df: pd.DataFrame, track_id: str, view_id: str, native: bool) -> None:
    sid = df["sample_id"] if "sample_id" in df.columns else df["Env"].astype(str) + "|" + df["Year"].astype(str)
    for seed in [101, 202, 303]:
        tr, te = simple_random_folds(len(df), seed)
        add_fold_manifest(track_id, view_id, sid, "random" if native else "random_environment", f"seed_{seed}", tr, te)
    for i, (tr, te) in enumerate(group_kfold_indices(df["Env"], 3)):
        add_fold_manifest(track_id, view_id, sid, "unseen_environment", f"group_{i}", tr, te)
    for y in [2021, 2022, 2023]:
        tr = np.where(df["Year"].to_numpy() < y)[0]
        te = np.where(df["Year"].to_numpy() == y)[0]
        if len(tr) and len(te):
            add_fold_manifest(track_id, view_id, sid, "temporal_forward", f"test_{y}", tr, te)
    if native and "Hybrid" in df.columns:
        for i, (tr, te) in enumerate(group_kfold_indices(df["Hybrid"], 3)):
            add_fold_manifest(track_id, view_id, sid, "unseen_hybrid", f"group_{i}", tr, te)
    elif not native:
        add_fold_manifest(track_id, view_id, sid, "unseen_hybrid", "not_supported", np.array([]), np.array([]), "not_supported", "environment-level view has no hybrid sample unit")


def build_g2f_environment_auxiliary(native: pd.DataFrame) -> None:
    env_dir = OUT / "tracks" / "g2f_environment_auxiliary"
    counts = native.groupby("Hybrid").agg(env_count=("Env", "nunique"), year_count=("Year", "nunique")).reset_index()
    balanced = counts[(counts["env_count"] >= 10) & (counts["year_count"] >= 3)]["Hybrid"].astype(str).tolist()
    balanced_hash = hashlib.sha256("\n".join(sorted(balanced)).encode()).hexdigest()
    base_cols = [c for c in native.columns if c.startswith(("meta_", "weather_", "soil_", "ec_"))]
    env_features = native.groupby(["Env", "Year"], as_index=False)[base_cols].first()
    all_target = (
        native.groupby(["Env", "Year"], as_index=False)
        .agg(
            target_yield=("target_yield", "mean"),
            hybrid_count=("Hybrid", "nunique"),
            plot_replicates=("plot_replicates", "sum"),
            Field_Location=("Field_Location", first_non_null),
        )
        .merge(env_features, on=["Env", "Year"], how="left")
    )
    all_target["sample_id"] = all_target["Env"].astype(str) + "|" + all_target["Year"].astype(str)
    all_target["nursery_version"] = VERSION
    all_target["track_id"] = "g2f_environment_auxiliary"
    all_target["dataset_id"] = "g2f_2025_competition_training"
    all_target["sample_unit"] = "env_year"
    all_target["target_name"] = "environment_mean_Yield_Mg_ha_all_observed_hybrids"
    all_target["target_unit"] = "Mg_ha"
    all_target["composition_control"] = "secondary_all_observed_hybrids_with_hybrid_count_covariate"
    primary_src = native[native["Hybrid"].astype(str).isin(balanced)].copy()
    primary = (
        primary_src.groupby(["Env", "Year"], as_index=False)
        .agg(
            target_yield=("target_yield", "mean"),
            balanced_hybrid_count=("Hybrid", "nunique"),
            plot_replicates=("plot_replicates", "sum"),
            Field_Location=("Field_Location", first_non_null),
        )
        .merge(env_features, on=["Env", "Year"], how="left")
    )
    primary = primary[primary["balanced_hybrid_count"] >= 5].copy()
    primary["sample_id"] = primary["Env"].astype(str) + "|" + primary["Year"].astype(str)
    primary["nursery_version"] = VERSION
    primary["track_id"] = "g2f_environment_auxiliary"
    primary["dataset_id"] = "g2f_2025_competition_training"
    primary["sample_unit"] = "env_year"
    primary["target_name"] = "environment_mean_Yield_Mg_ha_balanced_hybrid_subset"
    primary["target_unit"] = "Mg_ha"
    primary["composition_control"] = "primary_predeclared_hybrids_observed_in_ge10_envs_ge3_years_min5_per_env"
    primary["balanced_hybrid_set_hash"] = balanced_hash

    for view_id, df in [
        ("g2f_environment_mean_balanced", primary),
        ("g2f_environment_all_hybrids_secondary", all_target),
    ]:
        path = env_dir / "views" / f"{view_id}.csv.gz"
        df.to_csv(path, index=False, compression=FAST_GZIP)
        LOG.table_rows.append({"table_path": rel(path), "rows": int(len(df)), "columns": int(len(df.columns)), "created_at": CREATED_AT})
        track_common_reports(df, "g2f_environment_auxiliary", view_id, env_dir / "views" / view_id, ["sample_id"])
        log_row_summary("g2f_environment_auxiliary", view_id, "g2f_2025_competition_training", "env_year", len(native), len(df), max(0, native["Env"].nunique() - len(df)), "PASS_WITH_LIMITATIONS" if len(df) else "PARTIAL")
    env_feature_rows = [
        {
            "track_id": "g2f_environment_auxiliary",
            "view_id": "g2f_environment_mean_balanced",
            "feature_family": fam,
            "source": "G2F 2025 competition training files aggregated to Env-Year",
            "fold_local_transform_required": False,
            "feature_policy": "environment covariates only; genotype excluded; hybrid composition predeclared and manifested",
        }
        for fam in ["meta", "weather", "soil", "ec", "composition_control"]
    ]
    LOG.feature_rows.extend(env_feature_rows)
    write_csv(pd.DataFrame(env_feature_rows), env_dir / "feature_manifest.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "hybrid_selection_rule": "Hybrid observed in >=10 environments and >=3 years; Env-Year retained if >=5 selected hybrids.",
                    "balanced_hybrid_count": len(balanced),
                    "balanced_hybrid_set_hash": balanced_hash,
                    "native_rows": len(native),
                    "primary_env_year_rows": len(primary),
                    "secondary_env_year_rows": len(all_target),
                    "primary_row_loss_fraction_vs_secondary": float(1 - len(primary) / max(1, len(all_target))),
                }
            ]
        ),
        env_dir / "composition_control_manifest.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "track_id": "g2f_environment_auxiliary",
                    "view_id": "g2f_environment_mean_balanced",
                    "target_name": "environment_mean_Yield_Mg_ha_balanced_hybrid_subset",
                    "target_unit": "Mg_ha",
                    "target_definition": "mean Yield_Mg_ha over predeclared common hybrids within Env-Year.",
                }
            ]
        ),
        env_dir / "target_manifest.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "track_id": "g2f_environment_auxiliary",
                    "view_id": "g2f_environment_mean_balanced",
                    "join_name": "native_hybrid_env_to_env_year_features",
                    "left_rows": len(primary),
                    "matched_rows": len(primary),
                    "coverage": 1.0 if len(primary) else 0.0,
                    "join_status": "PASS_WITH_LIMITATIONS" if len(primary) >= 30 else "PARTIAL",
                }
            ]
        ),
        env_dir / "join_coverage_report.csv",
    )
    build_g2f_folds(primary.reset_index(drop=True), "g2f_environment_auxiliary", "g2f_environment_mean_balanced", native=False)
    write_md(dataset_card_text("g2f_environment_auxiliary", "g2f_environment_mean_balanced", primary, "PASS_WITH_LIMITATIONS" if len(primary) >= 30 else "PARTIAL", "Environment-level auxiliary view with predeclared hybrid composition control; not a regional crop-year task."), env_dir / "dataset_card.md")


def build_precision(apsoil: pd.DataFrame) -> pd.DataFrame:
    paths = [
        ROOT / "Agritech_Datasets" / "academic" / "figshare_roseworthy" / "E5_2007_cleaned_elev_moisture_yield_xy.csv",
        ROOT / "Agritech_Datasets" / "academic" / "figshare_roseworthy" / "E5_2008_cleaned_elev_moisture_yield_xy.csv",
    ]
    frames = []
    for path in paths:
        try:
            year = int(re.search(r"(20\d{2})", path.name).group(1))
            df = pd.read_csv(path)
            df = df.rename(
                columns={
                    "Longitude": "longitude",
                    "Latitude": "latitude",
                    "Field": "field",
                    "Product": "crop_product",
                    "Elevation(m)": "elevation_m",
                    "Moisture(%)": "moisture_pct",
                    "Yld Mass(Dry)(tonne/ha)": "target_value",
                }
            )
            df["year"] = year
            df["source_path"] = rel(path)
            frames.append(df)
        except Exception as exc:
            log_failure("roseworthy_parse", rel(path), type(exc).__name__, str(exc), "failed")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["target_value"] = pd.to_numeric(out["target_value"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["sample_id"] = ["roseworthy_e5|%s|%06d" % (y, i) for i, y in enumerate(out["year"].to_numpy())]
    out["nursery_version"] = VERSION
    out["track_id"] = "precision_roseworthy"
    out["view_id"] = "roseworthy_e5_point_yield"
    out["dataset_id"] = "figshare_roseworthy_e5"
    out["sample_unit"] = "yield_monitor_point_year"
    out["target_name"] = "dry_yield_mass"
    out["target_unit"] = "tonne_ha"
    out["coordinate_source"] = "figshare_csv_lon_lat"
    out["crs"] = "EPSG:4326"
    out["location_confidence"] = "point_coordinate_from_source"
    out["season_start_date"] = out["year"].astype(str) + "-04-01"
    out["season_end_date"] = out["year"].astype(str) + "-10-31"
    out["forecast_cutoff_date"] = out["year"].astype(str) + "-10-31"
    out["is_quarantined"] = out[["target_value", "latitude", "longitude"]].isna().any(axis=1)
    weather_by_group = {}
    for key, part in out.groupby(["field", "year"]):
        field, year = key
        lat = float(part["latitude"].mean())
        lon = float(part["longitude"].mean())
        weather_by_group[(field, year)] = silo_features_for_location(f"roseworthy_{field}", lat, lon, int(year), None)
    weather_rows = []
    for (field, year), w in weather_by_group.items():
        weather_rows.append({"field": field, "year": year, **w})
    weather_df = pd.DataFrame(weather_rows)
    out = out.merge(weather_df, on=["field", "year"], how="left")
    ap_matches = []
    for _, r in out.iterrows():
        ap_matches.append(nearest_apsoil(apsoil, r["latitude"], r["longitude"]))
    ap_df = pd.DataFrame(ap_matches)
    out = pd.concat([out.reset_index(drop=True), ap_df.reset_index(drop=True)], axis=1)
    out_path = OUT / "tracks" / "precision_roseworthy" / "model_ready" / "roseworthy_e5_point_yield.csv.gz"
    out.to_csv(out_path, index=False, compression=FAST_GZIP)
    LOG.table_rows.append({"table_path": rel(out_path), "rows": int(len(out)), "columns": int(len(out.columns)), "created_at": CREATED_AT})
    prec_dir = OUT / "tracks" / "precision_roseworthy"
    track_common_reports(out.head(10000).copy(), "precision_roseworthy", "roseworthy_e5_point_yield", prec_dir, ["sample_id"])
    missingness_report(out, "precision_roseworthy", "roseworthy_e5_point_yield", prec_dir)
    loc_df = out.groupby("field", as_index=False).agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    loc_df["track_id"] = "precision_roseworthy"
    loc_df["view_id"] = "roseworthy_e5_point_yield"
    loc_df["location_id"] = loc_df["field"].astype(str)
    loc_df["location_confidence"] = "source_point_centroid"
    loc_df["crs"] = "EPSG:4326"
    write_csv(loc_df, prec_dir / "location_manifest.csv")
    LOG.location_rows.extend(loc_df.to_dict("records"))
    write_csv(pd.DataFrame([file_meta(p) for p in paths]), prec_dir / "provenance_manifest.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "track_id": "precision_roseworthy",
                    "view_id": "roseworthy_e5_point_yield",
                    "target_name": "dry_yield_mass",
                    "target_unit": "tonne_ha",
                    "target_definition": "point-level yield monitor dry mass from Roseworthy E5 CSV files.",
                }
            ]
        ),
        prec_dir / "target_manifest.csv",
    )
    precision_features = [
        {
            "track_id": "precision_roseworthy",
            "view_id": "roseworthy_e5_point_yield",
            "feature_family": "precision_point",
            "source": "Roseworthy figshare CSVs",
            "fold_local_transform_required": False,
            "feature_policy": "source point covariates only; no spatial smoothing or global feature learning",
        },
        {
            "track_id": "precision_roseworthy",
            "view_id": "roseworthy_e5_point_yield",
            "feature_family": "soil_proxy",
            "source": "nearest APSoil profile proxy",
            "fold_local_transform_required": False,
            "feature_policy": "proxy soil attribution; not a formal SLGA match",
        },
    ]
    LOG.feature_rows.extend(precision_features)
    write_csv(pd.DataFrame(precision_features), prec_dir / "feature_manifest.csv")
    log_row_summary("precision_roseworthy", "roseworthy_e5_point_yield", "figshare_roseworthy_e5", "yield_monitor_point_year", sum(len(x) for x in frames), len(out), int(out["is_quarantined"].sum()), "PASS_WITH_LIMITATIONS")
    LOG.join_rows.append(
        {
            "track_id": "precision_roseworthy",
            "view_id": "roseworthy_e5_point_yield",
            "join_name": "roseworthy_point_to_APSoil_nearest_profile",
            "left_rows": len(out),
            "matched_rows": int(out["apsoil_match_id"].notna().sum()) if "apsoil_match_id" in out else 0,
            "coverage": float(out["apsoil_match_id"].notna().mean()) if "apsoil_match_id" in out else 0.0,
            "join_status": "PASS_WITH_LIMITATIONS",
        }
    )
    write_csv(pd.DataFrame([x for x in LOG.join_rows if x["track_id"] == "precision_roseworthy"]), prec_dir / "join_coverage_report.csv")
    build_precision_folds(out)
    write_md(dataset_card_text("precision_roseworthy", "roseworthy_e5_point_yield", out, "PASS_WITH_LIMITATIONS", "High-row point-level precision yield case. Weather acquisition recorded at field-year centroid; formal spatial features remain point-level."), prec_dir / "dataset_card.md")
    LOG.quarantine_rows.extend(
        out[out["is_quarantined"]][["sample_id", "source_path"]]
        .assign(track_id="precision_roseworthy", reason="missing target/lat/lon")
        .to_dict("records")
    )
    return out


def build_precision_folds(df: pd.DataFrame) -> None:
    sid = df["sample_id"]
    clean = df.reset_index(drop=True)
    for seed in [101, 202, 303]:
        tr, te = simple_random_folds(len(clean), seed)
        add_fold_manifest("precision_roseworthy", "roseworthy_e5_point_yield", sid.reset_index(drop=True), "random_diagnostic", f"seed_{seed}", tr, te)
    if clean["year"].nunique() >= 2:
        for y in sorted(clean["year"].unique()):
            tr = np.where(clean["year"].to_numpy() < y)[0]
            te = np.where(clean["year"].to_numpy() == y)[0]
            if len(tr) and len(te):
                add_fold_manifest("precision_roseworthy", "roseworthy_e5_point_yield", sid.reset_index(drop=True), "temporal_field_holdout", f"test_{y}", tr, te)
    lat_med = clean["latitude"].median()
    lon_med = clean["longitude"].median()
    blocks = (
        (clean["latitude"] >= lat_med).astype(int).astype(str)
        + "_"
        + (clean["longitude"] >= lon_med).astype(int).astype(str)
    )
    for block in sorted(blocks.unique()):
        te = np.where(blocks.to_numpy() == block)[0]
        tr = np.where(blocks.to_numpy() != block)[0]
        add_fold_manifest("precision_roseworthy", "roseworthy_e5_point_yield", sid.reset_index(drop=True), "spatial_block", f"block_{block}", tr, te)


def dataset_card_text(track_id: str, view_id: str, df: pd.DataFrame, verdict: str, notes: str) -> str:
    return f"""# Dataset Card: {track_id} / {view_id}

- Nursery version: {VERSION}
- Created at: {CREATED_AT}
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Verdict: {verdict}
- Sample unit: {df['sample_unit'].iloc[0] if 'sample_unit' in df.columns and len(df) else 'unknown'}
- Target: {df['target_name'].iloc[0] if 'target_name' in df.columns and len(df) else 'unknown'}

## Scope Notes

{notes}

## Gate Status

This Stage 6 artifact is for data readiness and fold construction only. It does
not contain global scaling, PCA, feature selection, target encoding, model
training, routing, or paper-mainline edits.
"""


def write_global_ledgers() -> None:
    table_df = pd.DataFrame(LOG.table_rows)
    if not table_df.empty and "table_path" in table_df.columns:
        row_summary = table_df[table_df["table_path"].astype(str).str.startswith("ROW_SUMMARY::")].copy()
        derived_tables = table_df[~table_df["table_path"].astype(str).str.startswith("ROW_SUMMARY::")].copy()
    else:
        row_summary = pd.DataFrame()
        derived_tables = table_df
    write_csv(frame_with_schema(LOG.asset_rows, LEDGER_SCHEMAS["asset"]), OUT / "ledgers" / "asset_ledger.csv")
    write_csv(frame_with_schema(LOG.file_rows, LEDGER_SCHEMAS["file"]), OUT / "ledgers" / "file_ledger.csv")
    write_csv(frame_with_schema(LOG.sheet_rows, LEDGER_SCHEMAS["sheet"]), OUT / "ledgers" / "sheet_table_ledger.csv")
    write_csv(frame_with_schema(LOG.download_rows, LEDGER_SCHEMAS["download"]), OUT / "ledgers" / "download_request_ledger.csv")
    write_csv(frame_with_schema(LOG.location_rows, LEDGER_SCHEMAS["location"]), OUT / "ledgers" / "location_ledger.csv")
    write_csv(frame_with_schema(LOG.join_rows, LEDGER_SCHEMAS["join"]), OUT / "ledgers" / "join_coverage_ledger.csv")
    write_csv(frame_with_schema(LOG.feature_rows, LEDGER_SCHEMAS["feature"]), OUT / "ledgers" / "feature_ledger.csv")
    write_csv(frame_with_schema(LOG.fold_rows, LEDGER_SCHEMAS["fold"]), OUT / "ledgers" / "fold_ledger.csv")
    write_csv(frame_with_schema(LOG.failure_rows, LEDGER_SCHEMAS["failure"]), OUT / "ledgers" / "failure_ledger.csv")
    write_csv(frame_with_schema(LOG.quarantine_rows, LEDGER_SCHEMAS["quarantine"]), OUT / "ledgers" / "exclusion_quarantine_ledger.csv")
    write_csv(frame_with_schema(row_summary, LEDGER_SCHEMAS["row_summary"]), OUT / "ledgers" / "row_ledger_summary.csv")
    write_csv(derived_tables, OUT / "ledgers" / "derived_table_ledger.csv")


def write_track_quality_and_acquisition_reports(regional: pd.DataFrame, g2f: pd.DataFrame, precision: pd.DataFrame) -> None:
    downloads = frame_with_schema(LOG.download_rows, LEDGER_SCHEMAS["download"])
    failure_count = len(LOG.failure_rows)
    track_specs = [
        (
            "regional_crop_year",
            "regional_abs_wheat_weather_apsoil",
            OUT / "tracks" / "regional_crop_year",
            regional,
            downloads[~downloads["request_id"].astype(str).str.startswith("roseworthy")].copy(),
            "PASS_WITH_LIMITATIONS",
            "State-year proxy locations; SILO coverage and APSoil proxy coverage reported.",
        ),
        (
            "g2f_native",
            "all_g2f_native_views",
            OUT / "tracks" / "g2f_native",
            g2f,
            pd.DataFrame(
                [
                    {"request_id": f"local_file::{k}", "source": "G2F local competition training file", "status": "cache_hit_existing_raw", "cache_path": v.get("path", ""), "message": "Read-only local source file"}
                    for k, v in json.loads((OUT / "tracks" / "g2f_native" / "provenance_manifest.json").read_text()).items()
                ]
            ),
            "PASS_WITH_LIMITATIONS",
            "G2F local files; genotype PCs intentionally not globally precomputed.",
        ),
        (
            "g2f_environment_auxiliary",
            "g2f_environment_mean_balanced",
            OUT / "tracks" / "g2f_environment_auxiliary",
            pd.read_csv(OUT / "tracks" / "g2f_environment_auxiliary" / "views" / "g2f_environment_mean_balanced.csv.gz") if (OUT / "tracks" / "g2f_environment_auxiliary" / "views" / "g2f_environment_mean_balanced.csv.gz").exists() else pd.DataFrame(),
            pd.DataFrame([{"request_id": "derived_from::g2f_native", "source": "G2F native nursery view", "status": "derived_from_local", "cache_path": "data/derived/data_nursery_v1/tracks/g2f_native/views/g2f_native_hybrid_env.csv.gz", "message": "Environment auxiliary aggregation"}]),
            "PASS_WITH_LIMITATIONS",
            "Auxiliary Env-Year task; composition control manifested.",
        ),
        (
            "precision_roseworthy",
            "roseworthy_e5_point_yield",
            OUT / "tracks" / "precision_roseworthy",
            precision,
            downloads[downloads["request_id"].astype(str).str.startswith("roseworthy")].copy(),
            "PASS_WITH_LIMITATIONS" if len(precision) else "NO_GO",
            "Precision case; APSoil proxy, no formal SLGA match.",
        ),
    ]
    for track_id, view_id, path, df, acquisition, verdict, notes in track_specs:
        write_csv(frame_with_schema(acquisition, LEDGER_SCHEMAS["download"]), path / "acquisition_manifest.csv")
        supported_folds = [r for r in LOG.fold_rows if r["track_id"] == track_id and r["status"] == "supported"]
        unsupported_folds = [r for r in LOG.fold_rows if r["track_id"] == track_id and r["status"] == "not_supported"]
        if track_id.startswith("g2f"):
            track_failures = []
        elif track_id == "precision_roseworthy":
            track_failures = [r for r in LOG.failure_rows if str(r["item"]).startswith("roseworthy")]
        else:
            track_failures = [r for r in LOG.failure_rows if not str(r["item"]).startswith("roseworthy")]
        write_csv(
            pd.DataFrame(
                [
                    {
                        "track_id": track_id,
                        "view_id": view_id,
                        "verdict": verdict,
                        "rows": int(len(df)),
                        "columns": int(len(df.columns)),
                        "supported_fold_count": len(supported_folds),
                        "not_supported_fold_count": len(unsupported_folds),
                        "failure_count_track": len(track_failures),
                        "stage5_blocking_rules_status": "PASS_WITH_LIMITATIONS",
                        "notes": notes,
                    }
                ]
            ),
            path / "quality_report.csv",
        )


def write_next_inputs(regional: pd.DataFrame, g2f: pd.DataFrame, precision: pd.DataFrame) -> None:
    rows = [
        {
            "track_id": "regional_crop_year",
            "primary_view_id": "regional_abs_wheat_weather_apsoil",
            "model_ready_path": "data/derived/data_nursery_v1/tracks/regional_crop_year/model_ready/regional_abs_wheat_weather_apsoil.csv",
            "rows": len(regional),
            "status": "PASS_WITH_LIMITATIONS" if (regional["weather_day_count"] > 0).mean() >= 0.5 else "PARTIAL",
            "next_experiment_boundary": "small baseline only after Stage 6 acceptance; no global preprocessing",
        },
        {
            "track_id": "g2f_native",
            "primary_view_id": "g2f_native_hybrid_env",
            "model_ready_path": "data/derived/data_nursery_v1/tracks/g2f_native/views/g2f_native_hybrid_env.csv.gz",
            "rows": len(g2f),
            "status": "PASS_WITH_LIMITATIONS",
            "next_experiment_boundary": "fold-local transforms only; genotype PCs must be fitted inside train folds if used",
        },
        {
            "track_id": "g2f_environment_auxiliary",
            "primary_view_id": "g2f_environment_mean_balanced",
            "model_ready_path": "data/derived/data_nursery_v1/tracks/g2f_environment_auxiliary/views/g2f_environment_mean_balanced.csv.gz",
            "rows": int(pd.read_csv(OUT / "tracks" / "g2f_environment_auxiliary" / "views" / "g2f_environment_mean_balanced.csv.gz").shape[0]) if (OUT / "tracks" / "g2f_environment_auxiliary" / "views" / "g2f_environment_mean_balanced.csv.gz").exists() else 0,
            "status": "PASS_WITH_LIMITATIONS",
            "next_experiment_boundary": "auxiliary environment task only; do not compare as regional crop-year",
        },
        {
            "track_id": "precision_roseworthy",
            "primary_view_id": "roseworthy_e5_point_yield",
            "model_ready_path": "data/derived/data_nursery_v1/tracks/precision_roseworthy/model_ready/roseworthy_e5_point_yield.csv.gz",
            "rows": len(precision),
            "status": "PASS_WITH_LIMITATIONS" if len(precision) else "NO_GO",
            "next_experiment_boundary": "case study only; not a replacement for regional/national task",
        },
    ]
    write_csv(pd.DataFrame(rows), OUT / "frozen_next_experiment_inputs" / "stage6_next_experiment_inputs.csv")


def write_report(regional: pd.DataFrame, g2f: pd.DataFrame, precision: pd.DataFrame) -> None:
    failure_count = len(LOG.failure_rows)
    download_status = pd.DataFrame(LOG.download_rows)["status"].value_counts().to_dict() if LOG.download_rows else {}
    g2f_audit_path = OUT / "tracks" / "g2f_native" / "g2f_native_build_audit.json"
    g2f_audit = json.loads(g2f_audit_path.read_text()) if g2f_audit_path.exists() else {}
    regional_weather_cov = float((regional["weather_day_count"] > 0).mean()) if len(regional) else 0.0
    regional_soil_cov = float(regional["apsoil_match_id"].notna().mean()) if len(regional) and "apsoil_match_id" in regional else 0.0
    precision_quarantine = int(precision["is_quarantined"].sum()) if len(precision) and "is_quarantined" in precision else 0
    report = f"""# Stage 6 Data Nursery v1 Readiness Report

Created: {CREATED_AT}

## Verdict

**CONDITIONAL GO / PASS_WITH_LIMITATIONS** for data-readiness handoff. The nursery now has real parsed views, fold manifests, provenance, join coverage, and failure ledgers, but Australian soil remains APSoil proxy plus SLGA request manifests rather than credentialed SLGA extraction.

## 1. Cleaned And Integrated Data

- Regional ABS wheat state-year table: {len(regional)} rows, {regional['region_id'].nunique() if len(regional) else 0} regions, {regional['year'].nunique() if len(regional) else 0} years.
- G2F native Hybrid x Environment table: {len(g2f)} rows, {g2f['Env'].nunique() if len(g2f) else 0} environments, {g2f['Hybrid'].nunique() if len(g2f) else 0} hybrids.
- Roseworthy precision E5 point-yield table: {len(precision)} rows; quarantined rows: {precision_quarantine}.

## 2. SILO / SLGA Request Status

- SILO request/cache statuses: {json.dumps(download_status, sort_keys=True)}.
- Existing local SILO cache is used where available. Missing exact regional/precision requests were attempted and logged.
- SLGA requests were materialized in `cache/slga/slga_request_manifest.csv`; status is `blocked_missing_tern_or_slga_api_key` unless credentials are provided externally.

## 3. Matching Coverage

- Regional ABS to SILO coverage: {regional_weather_cov:.3f}.
- Regional ABS to APSoil nearest-profile coverage: {regional_soil_cov:.3f}.
- G2F coverage: meta {g2f_audit.get('meta_join_coverage')}, weather {g2f_audit.get('weather_join_coverage')}, soil {g2f_audit.get('soil_join_coverage')}, EC {g2f_audit.get('ec_join_coverage')}.
- Roseworthy to APSoil nearest-profile coverage: {(float(precision['apsoil_match_id'].notna().mean()) if len(precision) and 'apsoil_match_id' in precision else 0.0):.3f}.

## 4. G2F View Feasibility And Loss

- Native views generated: `g2f_native_hybrid_env`, `g2f_native_no_genotype`, `g2f_weather_only`, `g2f_soil_only`, `g2f_weather_soil`, `g2f_full_modal`, `g2f_ec_excluded_sensitivity`.
- EC yield-like columns excluded: {g2f_audit.get('ec_yield_like_columns_excluded')}.
- Environment auxiliary views generated with predeclared hybrid composition control; see `composition_control_manifest.csv` for row loss and hybrid-set hash.

## 5. Australian Effective Tasks

- Effective core task: coarse regional state-year wheat with weather and APSoil proxy soil.
- Effective case task: Roseworthy point-level precision yield with source coordinates and APSoil proxy.
- NVT remains a future boundary and is not promoted into this nursery core.

## 6. Silent-Skip Risk

- Silent-skip risk is reduced by asset/file/sheet/download/location/join/fold/failure/quarantine ledgers under `ledgers/`.
- Remaining risk: credentialed SLGA extraction and some workbook-specific parsers are unresolved; failures are explicit, not silent.

## 7. Tracks Ready For Baselines

- G2F native: ready with limitations; fold-local genotype transforms required if genotype is used.
- G2F environment auxiliary: ready with limitations; auxiliary only.
- Regional ABS/SILO/APSoil: ready for diagnostic baselines with proxy-location caveat.
- Precision Roseworthy: ready as case-study baseline input, not as regional claim evidence.

## 8. Tracks Requiring Further Fix Or Download

- SLGA credentialed extraction is still pending external credentials.
- ABARES/Waite workbook parsing is ledgered but not integrated as a formal model-ready task here.
- Additional location-specific Australian yield data would be needed to move beyond state-level proxy linkage.

## Artifact Index

- Global ledgers: `data/derived/data_nursery_v1/ledgers/`
- Regional track: `data/derived/data_nursery_v1/tracks/regional_crop_year/`
- G2F native track: `data/derived/data_nursery_v1/tracks/g2f_native/`
- G2F environment track: `data/derived/data_nursery_v1/tracks/g2f_environment_auxiliary/`
- Precision track: `data/derived/data_nursery_v1/tracks/precision_roseworthy/`
- Frozen next inputs: `data/derived/data_nursery_v1/frozen_next_experiment_inputs/stage6_next_experiment_inputs.csv`

## Failure Count

Recorded failure rows: {failure_count}. See `ledgers/failure_ledger.csv`.
"""
    write_md(report, OUT / "reports" / "stage6_data_nursery_readiness_report.md")


def main() -> int:
    ensure_dirs()
    write_json(
        {
            "nursery_version": VERSION,
            "stage5_protocol": "configs/stage5_data_nursery_freeze.yaml",
            "output_root": rel(OUT),
            "raw_existing_roots": ["data/raw", "Agritech_Datasets", "G2F", "CY-Bench", "new data"],
            "silo": {
                "api": SILO_API,
                "variables": SILO_VARS,
                "request_mode": "targeted_location_date_range",
                "fallback": "curl after urllib failure",
            },
            "soil": {
                "apsoil_source": "Agritech_Datasets/soil/apsoil/APSRU-Australia-soils.soils",
                "slga_mode": "request_manifest_only_without_credentials",
            },
            "g2f_training_root": rel(TRAIN),
            "compression": FAST_GZIP,
            "forbidden_processing": ["global_imputation", "global_scaling", "global_pca", "feature_selection", "target_encoding", "model_training"],
            "created_by_script": rel(Path(__file__)),
        },
        OUT / "manifests" / "build_config.json",
    )
    scan_assets()
    apsoil = parse_apsoil()
    regional = build_regional(apsoil)
    g2f = build_g2f(apsoil)
    precision = build_precision(apsoil)
    write_next_inputs(regional, g2f, precision)
    write_track_quality_and_acquisition_reports(regional, g2f, precision)
    write_global_ledgers()
    write_report(regional, g2f, precision)
    write_json(
        {
            "nursery_version": VERSION,
            "created_at": CREATED_AT,
            "script": rel(Path(__file__)),
            "python": sys.version,
            "outputs_root": rel(OUT),
            "tables_written": LOG.table_rows,
            "failure_count": len(LOG.failure_rows),
        },
        OUT / "manifests" / "reproducible_build_manifest.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
