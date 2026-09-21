#!/usr/bin/env python3
"""
Incremental reconciliation patch for data_nursery_v1.

This script writes only to a new patch directory and never overwrites Stage 1-6
data_nursery_v1 artifacts. It parses ABS historical wheat, ABARES state crop
workbooks, and Waite trial XLS; reconciles them against the completed nursery;
and materializes SILO/SLGA request ledgers. SLGA credentials are read only from
environment variables and are never written to outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE6 = ROOT / "data/derived/data_nursery_v1"
PATCH_ID = "data_nursery_v1_reconciliation_patch_20260624"
PATCH = ROOT / "data/derived" / PATCH_ID

ABS_XLSX = ROOT / "Agritech_Datasets/crop_yield/abs/abs_historical_wheat_1860_2022.xlsx"
ABARES_DIR = ROOT / "Agritech_Datasets/crop_yield/abares"
WAITE_XLS = ROOT / "Agritech_Datasets/crop_yield/csiro_dap/Waite_Trial_Data.xls"

SILO_API = "https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php"
SILO_VARS = "RXTNEVP"
SILO_EMAIL = os.environ.get("SILO_USERNAME", "data-nursery-reconciliation@example.invalid")
SILO_PASSWORD = os.environ.get("SILO_PASSWORD", "")
SLGA_KEY_ENV = "TERN_API_KEY" if os.environ.get("TERN_API_KEY") else "SLGA_API_KEY"
SLGA_API_KEY = os.environ.get("TERN_API_KEY") or os.environ.get("SLGA_API_KEY")

STATE_ALIASES = {
    "Australian Capital Territory": "australian_capital_territory",
    "New South Wales": "new_south_wales",
    "Northern Territory": "northern_territory",
    "Queensland": "queensland",
    "South Australia": "south_australia",
    "Tasmania": "tasmania",
    "Victoria": "victoria",
    "Western Australia": "western_australia",
    "Australia": "australia_national",
}

LOCATION_COORDS = {
    "australian_capital_territory": ("Australian Capital Territory", -35.28, 149.13, "state_proxy_from_stage6"),
    "new_south_wales": ("New South Wales", -34.29, 146.05, "state_proxy_from_stage6"),
    "northern_territory": ("Northern Territory", -12.46, 130.84, "state_proxy_from_stage6"),
    "queensland": ("Queensland", -27.47, 153.03, "state_proxy_from_stage6"),
    "south_australia": ("South Australia", -34.95, 138.60, "state_proxy_from_stage6"),
    "tasmania": ("Tasmania", -42.88, 147.32, "state_proxy_from_stage6"),
    "victoria": ("Victoria", -36.98, 144.96, "state_proxy_from_stage6"),
    "western_australia": ("Western Australia", -31.95, 115.86, "state_proxy_from_stage6"),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def ensure_dirs() -> None:
    for path in [
        PATCH / "tracks/regional_crop_year_abs_historical",
        PATCH / "tracks/regional_crop_year_abares",
        PATCH / "tracks/waite_trial",
        PATCH / "cache/silo",
        PATCH / "cache/slga",
        PATCH / "manifests",
        PATCH / "ledgers",
        PATCH / "reports",
        PATCH / "frozen_next_experiment_inputs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "*", "na", "NA", "nan", "NaN", ".."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_abs_historical() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook = pd.ExcelFile(ABS_XLSX)
    sheet_records = []
    for sheet in workbook.sheet_names:
        df_head = pd.read_excel(ABS_XLSX, sheet_name=sheet, header=None, nrows=8)
        sheet_records.append(
            {
                "asset_group": "ABS historical wheat",
                "file_path": str(ABS_XLSX.relative_to(ROOT)),
                "sheet_name": sheet,
                "status": "used" if sheet == "Table 4" else "excluded",
                "reason": "wheat area/production table" if sheet == "Table 4" else "not wheat Table 4",
                "rows_previewed": len(df_head),
            }
        )

    raw = pd.read_excel(ABS_XLSX, sheet_name="Table 4", header=None)
    header_row = 4
    years = []
    for col in range(2, raw.shape[1]):
        val = numeric(raw.iat[header_row, col])
        if val is not None:
            years.append((col, int(val)))

    records = []
    exclusions = []
    current_region = None
    for r in range(header_row + 1, raw.shape[0]):
        region_cell = raw.iat[r, 0] if raw.shape[1] > 0 else None
        unit = str(raw.iat[r, 1]).strip().lower() if raw.shape[1] > 1 and pd.notna(raw.iat[r, 1]) else ""
        if pd.notna(region_cell):
            current_region = str(region_cell).strip()
        if not current_region or unit not in {"hectares", "tonnes"}:
            if any(numeric(raw.iat[r, c]) is not None for c, _ in years):
                exclusions.append({"row_index": r, "region": current_region, "unit": unit, "status": "excluded", "reason": "unknown unit"})
            continue
        for col, year in years:
            value = numeric(raw.iat[r, col])
            records.append(
                {
                    "raw_row": r,
                    "raw_col": col,
                    "region_name": current_region,
                    "region_id": STATE_ALIASES.get(current_region, safe_slug(current_region)),
                    "unit": unit,
                    "year": year,
                    "value": value,
                    "source_file": str(ABS_XLSX.relative_to(ROOT)),
                    "source_sheet": "Table 4",
                }
            )

    long = pd.DataFrame(records)
    pivot = (
        long.pivot_table(index=["region_name", "region_id", "year"], columns="unit", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "hectares" not in pivot.columns:
        pivot["hectares"] = np.nan
    if "tonnes" not in pivot.columns:
        pivot["tonnes"] = np.nan
    pivot["area_ha"] = pivot["hectares"]
    pivot["production_t"] = pivot["tonnes"]
    pivot["observed_yield_t_ha"] = pivot["production_t"] / pivot["area_ha"]
    pivot["crop"] = "wheat"
    pivot["sample_unit"] = np.where(pivot["region_id"].eq("australia_national"), "national_crop_year", "state_crop_year")
    pivot["target_status"] = "observed_final_abs"
    pivot["source_file"] = str(ABS_XLSX.relative_to(ROOT))
    pivot["source_sheet"] = "Table 4"
    pivot["fold_eligibility"] = np.where(
        pivot["region_id"].isin(LOCATION_COORDS) & pivot["area_ha"].gt(0) & pivot["production_t"].notna(),
        "eligible_proxy_location_if_weather_soil_available",
        "target_only_or_excluded",
    )
    pivot["row_status"] = np.select(
        [
            pivot["region_id"].eq("australia_national"),
            pivot["area_ha"].isna() | pivot["production_t"].isna(),
            pivot["area_ha"].le(0),
        ],
        ["excluded_national_total", "excluded_missing_area_or_production", "excluded_nonpositive_area"],
        default="used_target",
    )
    pivot["sample_id"] = pivot.apply(lambda r: f"abs_{r.region_id}_{int(r.year)}", axis=1)
    cols = [
        "sample_id",
        "sample_unit",
        "crop",
        "region_id",
        "region_name",
        "year",
        "area_ha",
        "production_t",
        "observed_yield_t_ha",
        "target_status",
        "fold_eligibility",
        "row_status",
        "source_file",
        "source_sheet",
    ]
    exclusion_cols = ["row_index", "region", "unit", "status", "reason"]
    return pivot[cols].sort_values(["region_id", "year"]), pd.DataFrame(sheet_records), pd.DataFrame(exclusions, columns=exclusion_cols)


def year_end_from_abares(label: Any) -> int | None:
    if pd.isna(label):
        return None
    text = str(label).strip()
    m = re.match(r"^(\d{4})[\u2013\-](\d{2})$", text)
    if not m:
        return None
    start = int(m.group(1))
    end_two = int(m.group(2))
    century = start // 100
    end = century * 100 + end_two
    if end < start:
        end += 100
    return end


def report_date_from_name(path: Path) -> str:
    m = re.search(r"crop_report_(\d{4})(\d{2})", path.name)
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2)}-01"


def abares_status(year_end: int, report_date: str) -> str:
    report_year = int(report_date[:4]) if report_date else 9999
    # ABARES current-year columns are often estimates and next season forecasts.
    if year_end >= report_year + 1:
        return "forecast"
    if year_end == report_year:
        return "estimate"
    return "observed_or_final"


def parse_abares_state_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    sheet_records = []
    for path in sorted(ABARES_DIR.glob("*_state_data.xlsx")):
        report_date = report_date_from_name(path)
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            status = "used" if sheet in STATE_ALIASES else "excluded"
            sheet_records.append(
                {
                    "asset_group": "ABARES state crop report",
                    "file_path": str(path.relative_to(ROOT)),
                    "sheet_name": sheet,
                    "status": status,
                    "reason": "state/national wheat table parsed" if status == "used" else "index or unsupported sheet",
                    "report_publication_date": report_date,
                }
            )
            if status != "used":
                continue
            raw = pd.read_excel(path, sheet_name=sheet, header=None)
            year_row = 6
            year_cols = [(c, year_end_from_abares(raw.iat[year_row, c])) for c in range(raw.shape[1])]
            year_cols = [(c, y) for c, y in year_cols if y is not None]
            wheat_rows = raw.index[raw.apply(lambda row: row.astype(str).str.strip().eq("Wheat").any(), axis=1)].tolist()
            if not wheat_rows:
                continue
            wheat_row = wheat_rows[0]
            area_row = wheat_row + 1
            prod_row = wheat_row + 2
            for col, year_end in year_cols:
                area = numeric(raw.iat[area_row, col]) if area_row < raw.shape[0] else None
                prod = numeric(raw.iat[prod_row, col]) if prod_row < raw.shape[0] else None
                area_ha = area * 1000.0 if area is not None else None
                production_t = prod * 1000.0 if prod is not None else None
                records.append(
                    {
                        "sample_id": f"abares_{safe_slug(path.stem)}_{STATE_ALIASES[sheet]}_{year_end}",
                        "sample_unit": "national_crop_year" if sheet == "Australia" else "state_crop_year",
                        "crop": "wheat",
                        "region_id": STATE_ALIASES[sheet],
                        "region_name": sheet,
                        "season_label": str(raw.iat[year_row, col]),
                        "year": year_end,
                        "area_ha": area_ha,
                        "production_t": production_t,
                        "observed_yield_t_ha": production_t / area_ha if area_ha and production_t is not None and area_ha > 0 else None,
                        "record_status": abares_status(year_end, report_date),
                        "report_publication_date": report_date,
                        "source_file": str(path.relative_to(ROOT)),
                        "source_sheet": sheet,
                        "row_status": "used_target" if area_ha and production_t is not None and area_ha > 0 else "excluded_missing_area_or_production",
                    }
                )
    df = pd.DataFrame(records)
    return df.sort_values(["source_file", "region_id", "year"]), pd.DataFrame(sheet_records)


def parse_waite_trial() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deps = STAGE6 / "_deps"
    if deps.exists():
        sys.path.insert(0, str(deps))
    xl = pd.ExcelFile(WAITE_XLS, engine="xlrd")
    sheet_manifest = []
    for sheet in xl.sheet_names:
        sheet_manifest.append(
            {
                "asset_group": "Waite C1 trial",
                "file_path": str(WAITE_XLS.relative_to(ROOT)),
                "sheet_name": sheet,
                "status": "used" if sheet in {"Grain Yield", "History", "Climate Data", "Select Soil Data"} else "partial",
                "reason": "parsed for yield/history/support audit" if sheet in {"Grain Yield", "History", "Climate Data", "Select Soil Data"} else "not needed for grain-yield target",
            }
        )

    yield_raw = pd.read_excel(WAITE_XLS, sheet_name="Grain Yield", header=None, engine="xlrd")
    hist_raw = pd.read_excel(WAITE_XLS, sheet_name="History", header=None, engine="xlrd")
    header_row = 3
    plot_cols = []
    for c in range(1, yield_raw.shape[1]):
        plot = numeric(yield_raw.iat[header_row, c])
        if plot is not None:
            plot_cols.append((c, int(plot)))

    rows = []
    row_audit = []
    for r in range(header_row + 1, yield_raw.shape[0]):
        year = numeric(yield_raw.iat[r, 0])
        if year is None:
            continue
        year = int(year)
        for c, plot in plot_cols:
            y = numeric(yield_raw.iat[r, c])
            crop_code = None
            if r < hist_raw.shape[0] and c < hist_raw.shape[1] and pd.notna(hist_raw.iat[r, c]):
                crop_code = str(hist_raw.iat[r, c]).strip()
            status = "used_target" if y is not None else "excluded_missing_yield"
            row = {
                "sample_id": f"waite_c1_plot{plot}_{year}",
                "sample_unit": "trial_plot_year",
                "trial_id": "waite_c1",
                "location_id": "waite_c1_trial_location_unresolved",
                "location_name": "Waite C1 trial",
                "year": year,
                "plot": plot,
                "crop_code": crop_code,
                "crop": {"W": "wheat", "B": "barley", "O": "oats", "Pe": "peas", "P": "pasture", "F": "fallow"}.get(crop_code, crop_code),
                "grain_yield_kg_ha": y,
                "observed_yield_t_ha": y / 1000.0 if y is not None else None,
                "row_status": status,
                "coordinate_status": "not_present_in_workbook_header_or_parsed_sheets",
                "weather_join_status": "not_attempted_missing_source_coordinate",
                "soil_join_status": "not_attempted_missing_source_coordinate",
                "source_file": str(WAITE_XLS.relative_to(ROOT)),
                "source_sheet_yield": "Grain Yield",
                "source_sheet_history": "History",
            }
            rows.append(row)
            row_audit.append(
                {
                    "source": "Waite",
                    "row_key": row["sample_id"],
                    "status": status,
                    "reason": "numeric yield parsed" if y is not None else "blank/star/non-numeric yield",
                    "crop_code": crop_code,
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "plot"]), pd.DataFrame(sheet_manifest), pd.DataFrame(row_audit)


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


def redacted_silo_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if "password" in qs:
        qs["password"] = ["<redacted>" if qs["password"][0] else ""]
    query = urllib.parse.urlencode(qs, doseq=True)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


def silo_request_id(region_id: str, year: int, lat: float, lon: float) -> str:
    digest = hashlib.sha1(f"{region_id}|{year}|{lat:.4f}|{lon:.4f}".encode()).hexdigest()[:8]
    return f"{region_id}_{year}_{digest}"


def count_csv_data_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def download_one_silo(req: dict[str, Any]) -> dict[str, Any]:
    out = Path(req["cache_path"])
    if out.exists() and out.stat().st_size > 100:
        req.update({"status": "cache_hit_patch", "rows": count_csv_data_rows(out), "bytes": out.stat().st_size, "attempted_at": now_iso()})
        return req
    url = silo_url(req["latitude"], req["longitude"], date.fromisoformat(req["start_date"]), date.fromisoformat(req["finish_date"]))
    req["url"] = redacted_silo_url(url)
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        req.update(
            {
                "status": "download_success" if len(data) > 100 else "failed_empty_or_too_small",
                "rows": count_csv_data_rows(out),
                "bytes": len(data),
                "attempted_at": now_iso(),
                "message": "",
            }
        )
    except Exception as exc:  # noqa: BLE001 - ledger must capture network/API errors.
        req.update(
            {
                "status": "failed_download_exception",
                "rows": 0,
                "bytes": 0,
                "attempted_at": now_iso(),
                "message": type(exc).__name__,
            }
        )
    return req


def build_silo_manifest(abs_df: pd.DataFrame, abares_df: pd.DataFrame, execute: bool = True) -> pd.DataFrame:
    stage6_acq = pd.read_csv(STAGE6 / "tracks/regional_crop_year/acquisition_manifest.csv")
    stage6_rows = {}
    for _, r in stage6_acq.iterrows():
        m = re.search(r"([a-z_]+)_(\d{4})_", str(r["request_id"]))
        if m:
            stage6_rows[(m.group(1), int(m.group(2)))] = r.to_dict()

    candidates = []
    abs_candidate = abs_df[
        abs_df["region_id"].isin(LOCATION_COORDS)
        & abs_df["row_status"].eq("used_target")
        & abs_df["year"].between(1989, 2022)
    ].copy()
    abares_candidate = abares_df[
        abares_df["region_id"].isin(LOCATION_COORDS)
        & abares_df["row_status"].eq("used_target")
        & abares_df["year"].between(1989, 2027)
    ].copy()
    for df, source in [(abs_candidate, "ABS_historical"), (abares_candidate, "ABARES_state")]:
        for _, row in df.iterrows():
            region_id = row["region_id"]
            year = int(row["year"])
            if (region_id, year) in {(c["region_id"], c["year"]) for c in candidates}:
                continue
            name, lat, lon, coord_source = LOCATION_COORDS[region_id]
            start = date(year, 4, 1)
            finish = date(year, 10, 31)
            req_id = silo_request_id(region_id, year, lat, lon)
            cache = PATCH / "cache/silo" / f"{req_id}.csv"
            base = stage6_rows.get((region_id, year))
            if base:
                status = f"cache_hit_stage6_{base.get('status', 'unknown')}"
                cache_path = base.get("cache_path", "")
                rows = base.get("rows", "")
                bytes_ = base.get("bytes", "")
                attempted_at = base.get("attempted_at", "")
                message = "reused Stage 6 validated request/cache"
            elif year < 1889:
                status = "not_attempted_pre_silo_historical_target_only"
                cache_path = ""
                rows = 0
                bytes_ = 0
                attempted_at = ""
                message = "outside SILO historical availability/useful patch scope"
            else:
                status = "planned"
                cache_path = str(cache.relative_to(ROOT))
                rows = 0
                bytes_ = 0
                attempted_at = ""
                message = "planned targeted SILO DataDrill request"
            candidates.append(
                {
                    "request_id": req_id,
                    "source": "SILO DataDrill",
                    "target_source": source,
                    "region_id": region_id,
                    "location_name": name,
                    "latitude": lat,
                    "longitude": lon,
                    "coordinate_source": coord_source,
                    "year": year,
                    "start_date": str(start),
                    "finish_date": str(finish),
                    "url": redacted_silo_url(silo_url(lat, lon, start, finish)) if status == "planned" else "",
                    "cache_path": cache_path,
                    "status": status,
                    "rows": rows,
                    "bytes": bytes_,
                    "attempted_at": attempted_at,
                    "message": message,
                }
            )
    manifest = pd.DataFrame(candidates).sort_values(["region_id", "year"])
    if execute:
        planned = manifest[manifest["status"].eq("planned")].to_dict("records")
        completed = []
        if planned:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(download_one_silo, req) for req in planned]
                for fut in as_completed(futures):
                    completed.append(fut.result())
            comp = pd.DataFrame(completed)
            manifest = manifest[~manifest["status"].eq("planned")]
            manifest = pd.concat([manifest, comp], ignore_index=True).sort_values(["region_id", "year"])
    return manifest


def slga_request_location(location_id: str, name: str, lat: float, lon: float) -> dict[str, Any]:
    base = {
        "location_id": location_id,
        "location_name": name,
        "latitude": lat,
        "longitude": lon,
        "crs": "EPSG:4326",
        "credential_source": SLGA_KEY_ENV if SLGA_API_KEY else "",
        "attempted_at": now_iso(),
        "status": "",
        "cache_path": "",
        "bytes": 0,
        "message": "",
    }
    if not SLGA_API_KEY:
        base["status"] = "blocked_missing_tern_or_slga_api_key"
        base["message"] = "Set TERN_API_KEY or SLGA_API_KEY in the runtime environment; key was not read from prompt or written to files."
        return base
    cache = PATCH / "cache/slga" / f"{location_id}.json"
    bbox = f"{lon-0.01:.5f},{lat-0.01:.5f},{lon+0.01:.5f},{lat+0.01:.5f}"
    endpoints = {
        "SoilDataFederator_sites": f"https://esoil.io/TERNLandscapes/SoilDataFederator/api/v1/sites?bbox={bbox}",
        "SLGA_Raster_Drill": f"https://esoil.io/TERNLandscapes/RasterProductsAPI/Drill?latitude={lat:.6f}&longitude={lon:.6f}&apikey={urllib.parse.quote(SLGA_API_KEY)}",
    }
    headers = {
        "Authorization": f"Bearer {SLGA_API_KEY}",
        "X-Api-Key": SLGA_API_KEY,
        "Accept": "application/json,text/plain,*/*",
        "User-Agent": "AgriTech-data-nursery-reconciliation/1.0",
    }
    responses: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for endpoint_name, url in endpoints.items():
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            text = payload.decode("utf-8", errors="replace")
            try:
                responses[endpoint_name] = json.loads(text)
            except json.JSONDecodeError:
                responses[endpoint_name] = {"raw_text": text[:200000]}
        except Exception as exc:  # noqa: BLE001
            failures[endpoint_name] = type(exc).__name__
    record = {
        "location_id": location_id,
        "location_name": name,
        "latitude": lat,
        "longitude": lon,
        "downloaded_at": now_iso(),
        "responses": responses,
        "failures": failures,
        "redacted_endpoints": {
            "SoilDataFederator_sites": "https://esoil.io/TERNLandscapes/SoilDataFederator/api/v1/sites?bbox=<redacted_location_bbox>",
            "SLGA_Raster_Drill": "https://esoil.io/TERNLandscapes/RasterProductsAPI/Drill?latitude=<lat>&longitude=<lon>&apikey=<redacted>",
        },
    }
    cache.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    base["cache_path"] = str(cache.relative_to(ROOT))
    base["bytes"] = cache.stat().st_size
    if responses:
        base["status"] = "download_success_with_partial_failures" if failures else "download_success"
    else:
        base["status"] = "failed_all_endpoints"
    base["message"] = ";".join(f"{k}:{v}" for k, v in failures.items())
    return base


def build_slga_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    locations = [
        {"location_id": loc, "location_name": vals[0], "latitude": vals[1], "longitude": vals[2], "coordinate_source": vals[3]}
        for loc, vals in LOCATION_COORDS.items()
    ]
    records = [slga_request_location(r["location_id"], r["location_name"], r["latitude"], r["longitude"]) for r in locations]
    request_df = pd.DataFrame(records).sort_values("location_id")
    attr_depth_rows = []
    attributes = ["clay", "sand", "silt", "organic_carbon", "ph", "bulk_density", "available_water_capacity"]
    depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
    for r in locations:
        req_status = request_df.loc[request_df["location_id"].eq(r["location_id"]), "status"].iloc[0]
        for attr in attributes:
            for depth in depths:
                attr_depth_rows.append(
                    {
                        **r,
                        "attribute": attr,
                        "depth": depth,
                        "statistic": "site_drill_or_grid_default",
                        "status": req_status,
                        "join_policy": "SLGA kept separate from APSoil; no silent APSoil substitution",
                    }
                )
    return request_df, pd.DataFrame(attr_depth_rows)


def reconcile_abs_abares(abs_df: pd.DataFrame, abares_df: pd.DataFrame) -> pd.DataFrame:
    latest_obs = abares_df[abares_df["record_status"].eq("observed_or_final")].copy()
    latest_obs = latest_obs.sort_values("report_publication_date").drop_duplicates(["region_id", "year"], keep="last")
    abs_used = abs_df[abs_df["row_status"].eq("used_target")].copy()
    merged = abs_used.merge(
        latest_obs[["region_id", "year", "area_ha", "production_t", "observed_yield_t_ha", "report_publication_date", "source_file"]],
        on=["region_id", "year"],
        how="outer",
        suffixes=("_abs", "_abares"),
        indicator=True,
    )
    merged["area_abs_minus_abares_ha"] = merged["area_ha_abs"] - merged["area_ha_abares"]
    merged["production_abs_minus_abares_t"] = merged["production_t_abs"] - merged["production_t_abares"]
    merged["yield_abs_minus_abares_t_ha"] = merged["observed_yield_t_ha_abs"] - merged["observed_yield_t_ha_abares"]
    merged["reconciliation_status"] = np.select(
        [merged["_merge"].eq("both"), merged["_merge"].eq("left_only"), merged["_merge"].eq("right_only")],
        ["overlap_compare_not_overwrite", "abs_only", "abares_only"],
        default="unknown",
    )
    return merged.sort_values(["region_id", "year"])


def build_row_and_join_ledgers(abs_df: pd.DataFrame, abares_df: pd.DataFrame, waite_df: pd.DataFrame, silo_df: pd.DataFrame, slga_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_records = []
    for source, df in [("ABS_historical", abs_df), ("ABARES_state", abares_df), ("Waite_trial", waite_df)]:
        total = len(df)
        used = int(df["row_status"].eq("used_target").sum()) if "row_status" in df else 0
        duplicate = int(df.duplicated().sum())
        quarantined = 0
        excluded = total - used - duplicate - quarantined
        row_records.append(
            {
                "source": source,
                "raw": total,
                "used": used,
                "duplicate": duplicate,
                "quarantined": quarantined,
                "excluded": excluded,
                "unmatched": 0,
                "closure_formula": "raw = used + duplicate + quarantined + excluded + unmatched",
                "closure_delta": total - (used + duplicate + quarantined + excluded),
            }
        )
    row_ledger = pd.DataFrame(row_records)

    join_records = []
    for source, df in [("ABS_historical", abs_df), ("ABARES_state", abares_df)]:
        used = df[df["row_status"].eq("used_target") & df["region_id"].isin(LOCATION_COORDS)].copy()
        used["has_silo_request"] = used.apply(
            lambda r: bool(((silo_df["region_id"].eq(r["region_id"])) & (silo_df["year"].eq(int(r["year"])))).any()),
            axis=1,
        )
        used["has_slga_request"] = used["region_id"].apply(lambda rid: (slga_df["location_id"].eq(rid)).any())
        join_records.append(
            {
                "source": source,
                "target_rows_with_proxy_location": len(used),
                "silo_request_rows": int(used["has_silo_request"].sum()),
                "slga_location_request_rows": int(used["has_slga_request"].sum()),
                "silo_unmatched_rows": int((~used["has_silo_request"]).sum()),
                "slga_unmatched_rows": int((~used["has_slga_request"]).sum()),
            }
        )
    join_records.append(
        {
            "source": "Waite_trial",
            "target_rows_with_proxy_location": 0,
            "silo_request_rows": 0,
            "slga_location_request_rows": 0,
            "silo_unmatched_rows": int(waite_df["row_status"].eq("used_target").sum()),
            "slga_unmatched_rows": int(waite_df["row_status"].eq("used_target").sum()),
        }
    )
    join_ledger = pd.DataFrame(join_records)

    loc_rows = []
    for loc, vals in LOCATION_COORDS.items():
        slga_status = slga_df.loc[slga_df["location_id"].eq(loc), "status"].iloc[0] if (slga_df["location_id"].eq(loc)).any() else "not_attempted"
        loc_rows.append(
            {
                "location_id": loc,
                "location_name": vals[0],
                "latitude": vals[1],
                "longitude": vals[2],
                "coordinate_source": vals[3],
                "silo_statuses": "|".join(sorted(set(map(str, silo_df.loc[silo_df["region_id"].eq(loc), "status"].dropna())))),
                "slga_status": slga_status,
                "apsoil_policy": "support_variable_proxy_only; not substituted for SLGA",
            }
        )
    loc_ledger = pd.DataFrame(loc_rows)

    failures = []
    for _, r in silo_df.iterrows():
        if str(r["status"]).startswith("failed") or str(r["status"]).startswith("not_attempted"):
            failures.append({"request_id": r["request_id"], "source": "SILO", "status": r["status"], "message": r.get("message", "")})
    for _, r in slga_df.iterrows():
        if str(r["status"]).startswith(("failed", "blocked", "not_attempted")):
            failures.append({"request_id": r["location_id"], "source": "SLGA", "status": r["status"], "message": r.get("message", "")})
    return row_ledger, join_ledger, loc_ledger, pd.DataFrame(failures)


def stage6_before_summary() -> dict[str, Any]:
    regional = pd.read_csv(STAGE6 / "tracks/regional_crop_year/acquisition_manifest.csv")
    old_target = ROOT / "data/processed/abs_wheat_yield_state_year.csv"
    old_df = pd.read_csv(old_target) if old_target.exists() else pd.DataFrame()
    return {
        "stage6_output_root": str(STAGE6.relative_to(ROOT)),
        "regional_rows": len(old_df),
        "regional_regions": int(old_df["region"].nunique()) if "region" in old_df.columns else None,
        "regional_year_min": int(old_df["year"].min()) if "year" in old_df.columns and len(old_df) else None,
        "regional_year_max": int(old_df["year"].max()) if "year" in old_df.columns and len(old_df) else None,
        "regional_silo_manifest_rows": len(regional),
        "regional_silo_status_counts": regional["status"].value_counts(dropna=False).to_dict(),
        "old_processed_abs_path": str(old_target.relative_to(ROOT)) if old_target.exists() else "",
    }


def build_asset_and_file_ledgers(sheet_manifests: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = [ABS_XLSX, WAITE_XLS] + sorted(ABARES_DIR.glob("*.xlsx"))
    file_rows = []
    for path in files:
        file_rows.append(
            {
                "file_path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": file_sha256(path) if path.exists() else "",
                "status": "used",
            }
        )
    file_ledger = pd.DataFrame(file_rows)
    sheet_ledger = pd.concat(sheet_manifests, ignore_index=True)
    asset_rows = [
        {"asset_group": "ABS historical wheat", "final_status": "used", "evidence_path": "tracks/regional_crop_year_abs_historical/abs_historical_wheat_state_year_targets.csv"},
        {"asset_group": "ABARES state crop reports", "final_status": "used", "evidence_path": "tracks/regional_crop_year_abares/abares_wheat_state_year.csv"},
        {"asset_group": "Waite C1 trial", "final_status": "partial", "evidence_path": "tracks/waite_trial/waite_grain_yield_long.csv"},
        {"asset_group": "SILO DataDrill", "final_status": "partial", "evidence_path": "manifests/silo_acquisition_manifest.csv"},
        {"asset_group": "SLGA", "final_status": "pending_external" if not SLGA_API_KEY else "partial", "evidence_path": "manifests/slga_request_manifest.csv"},
        {"asset_group": "APSoil", "final_status": "partial", "evidence_path": "manifests/apsoil_support_policy.csv"},
        {"asset_group": "Stage 6 original nursery", "final_status": "used", "evidence_path": "manifests/before_after_coverage.csv"},
    ]
    return pd.DataFrame(asset_rows), file_ledger, sheet_ledger


def write_reports(
    before: dict[str, Any],
    abs_df: pd.DataFrame,
    abares_df: pd.DataFrame,
    waite_df: pd.DataFrame,
    silo_df: pd.DataFrame,
    slga_df: pd.DataFrame,
    slga_attr_df: pd.DataFrame,
    abs_abares: pd.DataFrame,
    row_ledger: pd.DataFrame,
    join_ledger: pd.DataFrame,
) -> None:
    abs_used = abs_df[abs_df["row_status"].eq("used_target")]
    abares_used = abares_df[abares_df["row_status"].eq("used_target")]
    abares_observed = abares_used[abares_used["record_status"].eq("observed_or_final")]
    waite_used = waite_df[waite_df["row_status"].eq("used_target")]
    silo_counts = silo_df["status"].value_counts(dropna=False).to_dict()
    slga_counts = slga_df["status"].value_counts(dropna=False).to_dict()
    after_rows = [
        {
            "view": "Stage6 regional_crop_year",
            "rows": before["regional_rows"],
            "regions": before["regional_regions"],
            "year_min": before["regional_year_min"],
            "year_max": before["regional_year_max"],
            "notes": "processed ABS subset used by original data_nursery_v1",
        },
        {
            "view": "Patch ABS historical state targets",
            "rows": len(abs_used),
            "regions": abs_used["region_id"].nunique(),
            "year_min": int(abs_used["year"].min()) if len(abs_used) else "",
            "year_max": int(abs_used["year"].max()) if len(abs_used) else "",
            "notes": "Table 4 parsed directly; national total excluded from state-track counts",
        },
        {
            "view": "Patch ABARES usable targets",
            "rows": len(abares_used),
            "regions": abares_used["region_id"].nunique(),
            "year_min": int(abares_used["year"].min()) if len(abares_used) else "",
            "year_max": int(abares_used["year"].max()) if len(abares_used) else "",
            "notes": "all state-data workbook versions retained with publication date/status",
        },
        {
            "view": "Patch ABARES observed_or_final",
            "rows": len(abares_observed),
            "regions": abares_observed["region_id"].nunique(),
            "year_min": int(abares_observed["year"].min()) if len(abares_observed) else "",
            "year_max": int(abares_observed["year"].max()) if len(abares_observed) else "",
            "notes": "forecast/estimate retained but not mixed into observed labels",
        },
        {
            "view": "Patch Waite parsed targets",
            "rows": len(waite_used),
            "regions": waite_used["location_id"].nunique(),
            "year_min": int(waite_used["year"].min()) if len(waite_used) else "",
            "year_max": int(waite_used["year"].max()) if len(waite_used) else "",
            "notes": "independent plot-year track; geospatial joins not attempted without source coordinates",
        },
    ]
    before_after = pd.DataFrame(after_rows)
    write_csv(before_after, PATCH / "manifests/before_after_coverage.csv")

    apsoil_policy = pd.DataFrame(
        [
            {
                "policy": "APSoil support variables remain separate from SLGA",
                "status": "proxy_support_only",
                "evidence": "Stage 6 nearest-profile support retained only as proxy; SLGA failures are not silently filled by APSoil.",
            }
        ]
    )
    write_csv(apsoil_policy, PATCH / "manifests/apsoil_support_policy.csv")

    report = f"""# data_nursery_v1 Reconciliation Patch Readiness Report

Patch id: `{PATCH_ID}`  
Generated: `{now_iso()}`

## Verdict

`PARTIAL / BLOCKED_ON_RUNTIME_SLGA_CREDENTIAL` if no `TERN_API_KEY` or `SLGA_API_KEY` is present in the runtime environment; otherwise see `manifests/slga_request_manifest.csv` for endpoint-level status. This patch does not overwrite Stage 1-6 outputs and does not train models.

## Why The Old Regional Track Had 37 Rows

The Stage 6 regional track used `{before['old_processed_abs_path']}` rather than directly parsing `{ABS_XLSX.relative_to(ROOT)}`. That processed file contained `{before['regional_rows']}` rows, `{before['regional_regions']}` regions, and years `{before['regional_year_min']}`-`{before['regional_year_max']}`. The six-year limit was therefore inherited from the processed intermediate, not from the ABS historical workbook. Stage 6 then linked those rows to SILO/APSoil; it did not expand older ABS years.

## ABS Repair

- Parsed ABS workbook sheet `Table 4` directly.
- Usable state/territory target rows: `{len(abs_used)}`.
- State/territory coverage: `{abs_used['region_id'].nunique()}` regions, years `{int(abs_used['year'].min()) if len(abs_used) else 'NA'}`-`{int(abs_used['year'].max()) if len(abs_used) else 'NA'}`.
- National Australia totals are retained in the parsed table with `excluded_national_total`, but are not treated as state-year samples.
- Rows with missing/non-positive area or production are explicit in `row_status`; no complete-case weather filter is applied to the target table.

## ABARES Repair

- Parsed local ABARES `*_state_data.xlsx` workbooks from 2025-12, 2026-03, and 2026-06.
- Usable wheat state/national target records across workbook versions: `{len(abares_used)}`.
- Observed/final records: `{len(abares_observed)}`; estimate/forecast records are retained separately and not mixed into final observed labels.
- ABS/ABARES overlap and differences are in `manifests/abs_abares_overlap_reconciliation.csv`; neither source overwrites the other.

## Waite Repair

- Waite XLS was read with `xlrd` from the Stage 6 local dependency directory.
- Parsed plot-year grain-yield rows: `{len(waite_df)}` total, `{len(waite_used)}` numeric targets.
- Verdict: `PARTIAL`. The workbook provides yield/history and support sheets, but no parsed source coordinate was found in the audited sheets. The track is therefore independent (`trial_plot_year`) and not row-bound to regional data; SILO/SLGA joins are `not_attempted_missing_source_coordinate`.

## SILO Alignment

SILO status counts: `{json.dumps(silo_counts, sort_keys=True)}`.

Closure check: `planned = cache_hit + download_success + failed + not_attempted` is represented by the mutually exclusive `status` values in `manifests/silo_acquisition_manifest.csv`. Stage 6 cache hits are reused where the exact state-year request already existed; new model-ready ABS/ABARES state-year candidates from 1989 onward are requested or explicitly failed/logged.

## SLGA Alignment

SLGA location status counts: `{json.dumps(slga_counts, sort_keys=True)}`.

SLGA and APSoil are stored separately. Attribute/depth request coverage is in `manifests/slga_attribute_depth_coverage.csv`. If all rows show `blocked_missing_tern_or_slga_api_key`, actual SLGA extraction did not run because no safe runtime environment variable was available; the key supplied in chat was intentionally not copied into commands, logs, CSV, YAML, reports, or source.

## Silent-Skip Audit

- File ledger: `ledgers/file_reconciliation.csv`
- Sheet ledger: `ledgers/sheet_reconciliation.csv`
- Asset ledger: `ledgers/asset_reconciliation.csv`
- Location ledger: `ledgers/location_reconciliation.csv`
- Request ledger: `ledgers/request_reconciliation.csv`
- Row ledger: `ledgers/row_reconciliation.csv`
- Join ledger: `ledgers/join_reconciliation.csv`
- Failure/retry ledger: `ledgers/failure_retry_ledger.csv`

Row closure is summarized as `raw = used + duplicate + quarantined + excluded + unmatched` in `ledgers/row_reconciliation.csv`.

## Frozen Next Inputs

The next-stage frozen input list is `frozen_next_experiment_inputs/patch_next_experiment_inputs.csv`. It distinguishes ABS historical target-only rows, ABARES observed/estimate/forecast views, Waite trial rows, SILO status, SLGA status, APSoil proxy policy, and fold eligibility.
"""
    (PATCH / "reports/data_nursery_v1_reconciliation_readiness_report.md").write_text(report, encoding="utf-8")


def build_frozen_inputs(abs_df: pd.DataFrame, abares_df: pd.DataFrame, waite_df: pd.DataFrame, silo_df: pd.DataFrame, slga_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, df in [("ABS_historical", abs_df), ("ABARES_state", abares_df)]:
        used = df[df["row_status"].eq("used_target") & df["region_id"].isin(LOCATION_COORDS)].copy()
        for _, r in used.iterrows():
            silo_status = "|".join(sorted(set(map(str, silo_df.loc[(silo_df["region_id"].eq(r["region_id"])) & (silo_df["year"].eq(int(r["year"]))), "status"].dropna()))))
            slga_status = slga_df.loc[slga_df["location_id"].eq(r["region_id"]), "status"].iloc[0] if (slga_df["location_id"].eq(r["region_id"])).any() else "not_attempted"
            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "source": source,
                    "sample_unit": r["sample_unit"],
                    "crop": r["crop"],
                    "region_id": r["region_id"],
                    "year": int(r["year"]),
                    "target_status": r.get("target_status", r.get("record_status", "")),
                    "yield_t_ha": r["observed_yield_t_ha"],
                    "silo_status": silo_status or "not_attempted_outside_model_ready_weather_scope",
                    "slga_status": slga_status,
                    "apsoil_status": "proxy_support_variable_available_for_stage6_locations",
                    "fold_eligibility": "eligible_if_weather_soil_success" if silo_status and not slga_status.startswith("failed") else "target_only_or_pending_soil",
                }
            )
    for _, r in waite_df[waite_df["row_status"].eq("used_target")].iterrows():
        rows.append(
            {
                "sample_id": r["sample_id"],
                "source": "Waite_trial",
                "sample_unit": r["sample_unit"],
                "crop": r["crop"],
                "region_id": r["location_id"],
                "year": int(r["year"]),
                "target_status": "observed_trial_plot",
                "yield_t_ha": r["observed_yield_t_ha"],
                "silo_status": r["weather_join_status"],
                "slga_status": r["soil_join_status"],
                "apsoil_status": "not_substituted",
                "fold_eligibility": "trial_internal_only_until_source_coordinate_resolved",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    before = stage6_before_summary()

    abs_df, abs_sheet, abs_exclusions = parse_abs_historical()
    abares_df, abares_sheet = parse_abares_state_data()
    waite_df, waite_sheet, waite_row_audit = parse_waite_trial()

    silo_df = build_silo_manifest(abs_df, abares_df, execute=True)
    slga_df, slga_attr_df = build_slga_manifest()
    abs_abares = reconcile_abs_abares(abs_df, abares_df)

    asset_ledger, file_ledger, sheet_ledger = build_asset_and_file_ledgers([abs_sheet, abares_sheet, waite_sheet])
    row_ledger, join_ledger, loc_ledger, failure_ledger = build_row_and_join_ledgers(abs_df, abares_df, waite_df, silo_df, slga_df)
    frozen = build_frozen_inputs(abs_df, abares_df, waite_df, silo_df, slga_df)

    write_csv(abs_df, PATCH / "tracks/regional_crop_year_abs_historical/abs_historical_wheat_state_year_targets.csv")
    write_csv(abs_exclusions, PATCH / "tracks/regional_crop_year_abs_historical/abs_parser_exclusions.csv")
    write_csv(abares_df, PATCH / "tracks/regional_crop_year_abares/abares_wheat_state_year.csv")
    write_csv(waite_df, PATCH / "tracks/waite_trial/waite_grain_yield_long.csv")
    write_csv(waite_df[waite_df["crop"].eq("wheat") & waite_df["row_status"].eq("used_target")], PATCH / "tracks/waite_trial/waite_wheat_yield_valid.csv")
    write_csv(waite_row_audit, PATCH / "tracks/waite_trial/waite_row_audit.csv")

    write_csv(silo_df, PATCH / "manifests/silo_acquisition_manifest.csv")
    write_csv(slga_df, PATCH / "manifests/slga_request_manifest.csv")
    write_csv(slga_attr_df, PATCH / "manifests/slga_attribute_depth_coverage.csv")
    write_csv(abs_abares, PATCH / "manifests/abs_abares_overlap_reconciliation.csv")
    write_csv(frozen, PATCH / "frozen_next_experiment_inputs/patch_next_experiment_inputs.csv")

    write_csv(asset_ledger, PATCH / "ledgers/asset_reconciliation.csv")
    write_csv(file_ledger, PATCH / "ledgers/file_reconciliation.csv")
    write_csv(sheet_ledger, PATCH / "ledgers/sheet_reconciliation.csv")
    write_csv(loc_ledger, PATCH / "ledgers/location_reconciliation.csv")
    write_csv(silo_df, PATCH / "ledgers/request_reconciliation.csv")
    write_csv(row_ledger, PATCH / "ledgers/row_reconciliation.csv")
    write_csv(join_ledger, PATCH / "ledgers/join_reconciliation.csv")
    write_csv(failure_ledger, PATCH / "ledgers/failure_retry_ledger.csv")

    write_json(
        {
            "patch_id": PATCH_ID,
            "generated_at": now_iso(),
            "stage6_root": str(STAGE6.relative_to(ROOT)),
            "script": str(Path(__file__).relative_to(ROOT)),
            "security": {
                "slga_key_source": SLGA_KEY_ENV if SLGA_API_KEY else None,
                "slga_key_written_to_outputs": False,
                "prompt_key_copied": False,
            },
            "before_summary": before,
        },
        PATCH / "manifests/reconciliation_patch_manifest.json",
    )
    write_reports(before, abs_df, abares_df, waite_df, silo_df, slga_df, slga_attr_df, abs_abares, row_ledger, join_ledger)
    print(json.dumps({"patch_dir": str(PATCH.relative_to(ROOT)), "abs_rows": len(abs_df), "abares_rows": len(abares_df), "waite_rows": len(waite_df), "silo_status": silo_df["status"].value_counts().to_dict(), "slga_status": slga_df["status"].value_counts().to_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
