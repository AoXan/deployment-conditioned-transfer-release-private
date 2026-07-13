#!/usr/bin/env python3
"""Build locked APSIM configurations and frozen model/sample manifests.

This script never uses observed yield to set APSIM parameters. Yield is copied
only into the immutable evaluation manifest after configuration fields are set.
"""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "apsim_comparison"
OUT = ROOT / "outputs" / "apsim_comparison"
CONFIG = EXP / "configs"
GENERATED = CONFIG / "generated"
WEATHER_DIR = GENERATED / "weather"
RUNTIME = EXP / "runtime" / "ApsimX"
TEMPLATE = CONFIG / "test-wheat.apsimx"
LINEAGE = Path(
    os.environ.get(
        "AGRITECH_LINEAGE_MANIFEST",
        ROOT / "data_manifest/checkpoints/checkpoint_lineage.csv",
    )
)
CYB = Path(os.environ.get("AGRITECH_CYBENCH_ROOT", ROOT / "data/external/cybench/wheat/AU"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def child(node: dict, name: str) -> dict:
    return next(c for c in node.get("Children", []) if c.get("Name") == name)


def set_parameter(manager: dict, key: str, value: object) -> None:
    for item in manager["Parameters"]:
        if item["Key"] == key:
            item["Value"] = str(value)
            return
    raise KeyError(f"Manager parameter not found: {key}")


def best_validation_loss(candidate_id: str, official_predictions: str) -> float:
    prediction_path = resolve_prediction_path(candidate_id, official_predictions)
    namespace = prediction_path.parents[2]
    manifest = namespace / "target_checkpoints" / f"{candidate_id}.pt.manifest.json"
    if not manifest.exists():
        return math.nan
    return float(json.loads(manifest.read_text())["metadata"]["best_validation_loss"])


def resolve_prediction_path(candidate_id: str, recorded_path: str) -> Path:
    recorded = Path(recorded_path)
    if recorded.is_file():
        return recorded
    root_value = os.environ.get("AGRITECH_PREDICTION_ROOT")
    if not root_value:
        raise FileNotFoundError(
            "frozen prediction bundle is not at its recorded location; set AGRITECH_PREDICTION_ROOT"
        )
    root = Path(root_value)
    candidates = [
        root / candidate_id / "predictions.csv",
        root / "jobs" / "targets" / candidate_id / "predictions.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"prediction missing for {candidate_id}; checked: {candidates}")


def load_frozen_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    lineage = pd.read_csv(LINEAGE)
    lineage = lineage[lineage["source_dataset"].eq("PRIMARY_CYBENCH_MAIZE_US")].copy()
    lineage["validation_mse"] = [
        best_validation_loss(row.candidate_id, row.official_predictions_path)
        for row in lineage.itertuples()
    ]

    # The manuscript's local reference uses the imputation scratch family.
    scratch = lineage[
        lineage["transfer_strategy"].eq("target_scratch")
        & lineage["missing_modality_method"].eq("imputation")
    ].copy()
    ordinary = lineage[lineage["transfer_strategy"].eq("ordinary_transfer")].copy()
    selected_lineage = pd.concat([scratch, ordinary], ignore_index=True)

    transfer_only = ordinary[ordinary["source_route"].ne("supervised")]
    winners = (
        transfer_only.sort_values(["split_id", "seed", "validation_mse", "candidate_id"])
        .groupby(["split_id", "seed"], as_index=False)
        .first()[["split_id", "seed", "candidate_id"]]
        .rename(columns={"candidate_id": "validation_selected_candidate"})
    )
    selected_lineage = selected_lineage.merge(winners, on=["split_id", "seed"], how="left")
    selected_lineage["is_validation_selected_transfer"] = (
        selected_lineage["candidate_id"] == selected_lineage["validation_selected_candidate"]
    )
    selected_lineage["model_role"] = np.where(
        selected_lineage["transfer_strategy"].eq("target_scratch"),
        "local_scratch",
        np.where(selected_lineage["source_route"].eq("supervised"), "supervised_transfer", selected_lineage["source_route"]),
    )

    prediction_rows: list[pd.DataFrame] = []
    for row in selected_lineage.itertuples():
        prediction_path = resolve_prediction_path(row.candidate_id, row.official_predictions_path)
        p = pd.read_csv(prediction_path)
        p.insert(0, "candidate_id", row.candidate_id)
        p.insert(1, "contract", row.split_id)
        p.insert(2, "seed", int(row.seed))
        p.insert(3, "model_role", row.model_role)
        p.insert(4, "validation_mse", row.validation_mse)
        p.insert(5, "is_validation_selected_transfer", bool(row.is_validation_selected_transfer))
        prediction_rows.append(p)
        selected_lineage.loc[selected_lineage["candidate_id"].eq(row.candidate_id), "official_predictions_path"] = (
            f"${{AGRITECH_PREDICTION_ROOT}}/{row.candidate_id}/predictions.csv"
        )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return selected_lineage, predictions


def parse_sample_id(sample_id: str) -> tuple[str, int]:
    _, adm_id, year = sample_id.split("|")
    return adm_id, int(year)


def build_sample_manifest(predictions: pd.DataFrame) -> pd.DataFrame:
    target_check = predictions.groupby("sample_id")["y_true"].agg(["min", "max"])
    if not np.allclose(target_check["min"], target_check["max"], atol=1e-6):
        raise ValueError("Target mismatch across frozen prediction files")

    samples = predictions.groupby("sample_id", as_index=False).agg(
        observed_yield_t_ha=("y_true", "first"),
        prediction_rows=("y_pred", "size"),
        contracts=("contract", lambda x: ";".join(sorted(set(x)))),
        seeds=("seed", lambda x: ";".join(map(str, sorted(set(x))))),
    )
    parsed = samples["sample_id"].map(parse_sample_id)
    samples["adm_id"] = parsed.map(lambda x: x[0])
    samples["year"] = parsed.map(lambda x: x[1])

    locations = pd.read_csv(CYB / "location_wheat_AU.csv")
    calendar = pd.read_csv(CYB / "crop_calendar_wheat_AU.csv")
    soil = pd.read_csv(CYB / "soil_wheat_AU.csv")
    samples = samples.merge(locations[["adm_id", "latitude", "longitude"]], on="adm_id", validate="many_to_one")
    samples = samples.merge(calendar[["adm_id", "sos", "eos"]], on="adm_id", validate="many_to_one")
    samples = samples.merge(soil[["adm_id", "awc", "bulk_density", "drainage_class"]], on="adm_id", validate="many_to_one")

    weather = pd.read_csv(CYB / "meteo_wheat_AU.csv", usecols=["adm_id", "date", "tmin", "tmax", "prec", "rad"])
    weather["year"] = weather["date"].astype(str).str[:4].astype(int)
    coverage = weather.groupby(["adm_id", "year"]).agg(weather_days=("date", "size"), missing_weather=("tmin", lambda x: int(x.isna().sum())))
    samples = samples.merge(coverage.reset_index(), on=["adm_id", "year"], how="left", validate="many_to_one")
    samples["eligible"] = samples[["latitude", "longitude", "sos", "awc", "bulk_density", "weather_days"]].notna().all(axis=1)
    samples["exclusion_reason"] = np.where(samples["eligible"], "", "missing required location/calendar/soil/weather field")
    return samples.sort_values(["adm_id", "year"]).reset_index(drop=True)


def write_met_files(samples: pd.DataFrame) -> None:
    weather = pd.read_csv(CYB / "meteo_wheat_AU.csv")
    weather["date_dt"] = pd.to_datetime(weather["date"].astype(str), format="%Y%m%d")
    weather["year"] = weather["date_dt"].dt.year
    for row in samples[samples["eligible"]].itertuples():
        w = weather[(weather["adm_id"] == row.adm_id) & (weather["year"] == row.year)].copy()
        w = w.sort_values("date_dt")
        if w.empty or w[["tmin", "tmax", "prec", "rad"]].isna().any().any():
            continue
        monthly = ((w["tmin"] + w["tmax"]) / 2).groupby(w["date_dt"].dt.month).mean()
        tav = float(((w["tmin"] + w["tmax"]) / 2).mean())
        amp = float(monthly.max() - monthly.min())
        path = WEATHER_DIR / f"{row.adm_id}_{row.year}.met"
        with path.open("w", newline="") as stream:
            stream.write(f"[weather.met.weather]\nlatitude = {row.latitude:.6f}\n")
            stream.write(f"tav = {tav:.4f}\namp = {amp:.4f}\n")
            stream.write("year day radn maxt mint rain\n")
            stream.write("() () (MJ/m2/day) (oC) (oC) (mm)\n")
            for d in w.itertuples():
                stream.write(
                    f"{d.date_dt.year} {d.date_dt.dayofyear} {d.rad / 1_000_000.0:.5f} "
                    f"{d.tmax:.4f} {d.tmin:.4f} {max(0.0, d.prec):.4f}\n"
                )


def nearest_initial_fraction(adm_id: str, year: int, sow_doy: int, awc: float) -> tuple[float, str]:
    moisture = getattr(nearest_initial_fraction, "moisture", None)
    if moisture is None:
        path = CYB / "soil_moisture_wheat_AU.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"CY-Bench soil-moisture input missing: {path}; set AGRITECH_CYBENCH_ROOT"
            )
        moisture = pd.read_csv(path)
        nearest_initial_fraction.moisture = moisture
    subset = moisture[moisture["adm_id"].eq(adm_id)].copy()
    subset["date_dt"] = pd.to_datetime(subset["date"].astype(str), format="%Y%m%d")
    sow = datetime(year, 1, 1) + timedelta(days=sow_doy - 1)
    subset = subset[subset["date_dt"].le(sow)]
    if subset.empty:
        return 0.5, "fixed_0.5_no_pre_sowing_rsm"
    nearest = subset.iloc[(subset["date_dt"] - sow).abs().argmin()]
    if (sow - nearest["date_dt"]).days > 90 or pd.isna(nearest["rsm"]):
        return 0.5, "fixed_0.5_no_recent_pre_sowing_rsm"
    pawc_mm = max(1.0, awc / 100.0 * 1800.0)
    return float(np.clip(nearest["rsm"] / pawc_mm, 0.1, 1.0)), "cybench_pre_sowing_rsm_proxy"

def configure_soil(field: dict, condition: str, row: object, water_fraction: float, no3_total: float, nh4_total: float = 10.0) -> str:
    soil = child(field, "Soil")
    physical = child(soil, "Physical")
    soil_crop = child(physical, "WheatSoil")
    water = child(soil, "Water")
    no3 = child(soil, "NO3")
    nh4 = child(soil, "NH4")

    if condition == "available":
        ll15 = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
        delta = float(row.awc) / 100.0
        bd = float(row.bulk_density)
        sat_base = float(np.clip(1.0 - bd / 2.65 - 0.03, 0.34, 0.55))
        dul = [min(v + delta, sat_base - 0.03) for v in ll15]
        sat = [max(v + 0.03, sat_base) for v in dul]
        physical["BD"] = [bd] * len(ll15)
        physical["AirDry"] = [0.5 * v if i < 2 else 0.8 * v for i, v in enumerate(ll15)]
        physical["LL15"] = ll15
        physical["DUL"] = dul
        physical["SAT"] = sat
        soil_crop["LL"] = ll15
        sw = child(soil, "SoilWater")
        swcon = float(np.clip(0.12 + 0.06 * float(row.drainage_class), 0.18, 0.60))
        sw["SWCON"] = [swcon] * len(ll15)

    ll15 = np.asarray(physical["LL15"], dtype=float)
    dul = np.asarray(physical["DUL"], dtype=float)
    fraction = float(np.clip(water_fraction, 0.0, 1.0))
    water["InitialValues"] = (ll15 + fraction * (dul - ll15)).tolist()
    water["FilledFromTop"] = False

    thickness = np.asarray(physical["Thickness"], dtype=float)
    weights = thickness / thickness.sum()
    no3["InitialValuesUnits"] = 1
    no3["InitialValues"] = (weights * no3_total).tolist()
    nh4["InitialValuesUnits"] = 1
    nh4["InitialValues"] = (weights * nh4_total).tolist()
    return "apsim_template_soil" if condition == "constrained" else "cybench_awc_bd_drainage_proxy"


def scenario_definitions() -> list[dict]:
    base = {"sowing_offset": 0, "water_fraction": 0.5, "fertiliser": 50.0, "no3": 50.0, "cultivar": "Hartog"}
    scenarios = [
        {"name": "constrained_central", "condition": "constrained", **base},
        {"name": "available_central", "condition": "available", **base, "water_fraction": None},
    ]
    for value in (-14, 14):
        scenarios.append({"name": f"sens_sowing_{value:+d}", "condition": "constrained", **base, "sowing_offset": value})
    for value in (0.25, 0.75):
        scenarios.append({"name": f"sens_water_{value:.2f}", "condition": "constrained", **base, "water_fraction": value})
    for value in (0.0, 100.0):
        scenarios.append({"name": f"sens_fertiliser_{int(value)}", "condition": "constrained", **base, "fertiliser": value})
    for value in (25.0, 100.0):
        scenarios.append({"name": f"sens_no3_{int(value)}", "condition": "constrained", **base, "no3": value})
    for value in ("Janz", "Mace"):
        scenarios.append({"name": f"sens_cultivar_{value.lower()}", "condition": "constrained", **base, "cultivar": value})
    return scenarios


def build_apsim_files(samples: pd.DataFrame) -> pd.DataFrame:
    template = json.loads(TEMPLATE.read_text())
    template_sim = child(template, "Simulation")
    datastore = child(template, "DataStore")
    provenance: list[dict] = []

    for scenario in scenario_definitions():
        root = copy.deepcopy(template)
        root["Children"] = []
        for row in samples[samples["eligible"]].itertuples():
            sim = copy.deepcopy(template_sim)
            sim["Name"] = f"{scenario['name']}__{row.adm_id.replace('-', '_')}__{row.year}"
            clock = child(sim, "Clock")
            clock["Start"] = f"{row.year}-01-01T00:00:00"
            eos_doy = int(round(float(row.eos)))
            eos_date = datetime(row.year, 1, 1) + timedelta(days=eos_doy - 1)
            clock["End"] = f"{eos_date.date().isoformat()}T00:00:00"
            weather = child(sim, "Weather")
            weather["FileName"] = f"weather/{row.adm_id}_{row.year}.met"
            field = child(sim, "Field")
            sowing = child(field, "Sowing")
            sow_doy = int(round(float(row.sos))) + int(scenario["sowing_offset"])
            sow_date = datetime(row.year, 1, 1) + timedelta(days=sow_doy - 1)
            sow_text = f"{sow_date.day}-{sow_date.strftime('%b').lower()}"
            set_parameter(sowing, "StartDate", sow_text)
            set_parameter(sowing, "EndDate", sow_text)
            set_parameter(sowing, "MinESW", -1)
            set_parameter(sowing, "MinRain", -1)
            set_parameter(sowing, "RainDays", 1)
            set_parameter(sowing, "CultivarName", scenario["cultivar"])
            fertiliser = child(field, "Fertilise at sowing")
            set_parameter(fertiliser, "Amount", scenario["fertiliser"])
            if scenario["condition"] == "available" and scenario["water_fraction"] is None:
                fraction, water_source = nearest_initial_fraction(row.adm_id, row.year, sow_doy, row.awc)
            else:
                fraction, water_source = float(scenario["water_fraction"]), "fixed_fraction"
            soil_source = configure_soil(field, scenario["condition"], row, fraction, scenario["no3"])

            report = child(field, "Report")
            report["VariableNames"] = [
                "[Clock].Today as Date",
                "[Wheat].Grain.Total.Wt*10 as YieldKgHa",
                "[Wheat].Grain.Protein as GrainProtein",
            ]
            report["EventNames"] = ["[Wheat].Harvesting", "[Clock].EndOfSimulation"]
            sim["Children"] = [c for c in sim["Children"] if c.get("Name") != "Graph"]
            root["Children"].append(sim)
            provenance.append({
                "scenario": scenario["name"], "condition": scenario["condition"], "sample_id": row.sample_id,
                "sowing_date": sow_date.date().isoformat(), "cultivar": scenario["cultivar"],
                "initial_water_fraction": fraction, "initial_water_source": water_source,
                "initial_no3_kg_ha": scenario["no3"], "initial_nh4_kg_ha": 10.0,
                "fertiliser_n_kg_ha": scenario["fertiliser"], "soil_source": soil_source,
                "weather_source": "CY-Bench daily meteorology", "yield_used_for_configuration": False,
            })
        root["Children"].append(copy.deepcopy(datastore))
        path = GENERATED / f"{scenario['name']}.apsimx"
        path.write_text(json.dumps(root, indent=2) + "\n")
    return pd.DataFrame(provenance)


def write_runtime_manifest() -> None:
    commit = __import__("subprocess").check_output(["git", "-C", str(RUNTIME), "rev-parse", "HEAD"], text=True).strip()
    models = RUNTIME / "bin" / "Release" / "net8.0" / "Models.dll"
    manifest = {
        "apsim_family": "APSIM Next Generation",
        "source_repository": "https://github.com/APSIMInitiative/ApsimX.git",
        "source_commit": commit,
        "models_dll": str(models.relative_to(ROOT)),
        "models_dll_sha256": sha256(models),
        "template": str(TEMPLATE.relative_to(ROOT)),
        "template_sha256": sha256(TEMPLATE),
        "crop_module": "Wheat",
        "central_cultivar": "Hartog",
        "dotnet_info": str((OUT / "dotnet_info.txt").relative_to(ROOT)),
    }
    (OUT / "apsim_runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"executed": False, "scenarios": [item["name"] for item in scenario_definitions()]}, indent=2))
        return
    if args.validate_only:
        required = [LINEAGE, TEMPLATE, ROOT / "experiments/apsim_comparison/protocol_lock.yaml"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing APSIM build input: {missing}")
        print(json.dumps({"status": "PASS", "required_inputs": len(required)}, indent=2))
        return
    for path in (OUT, CONFIG, GENERATED, WEATHER_DIR):
        path.mkdir(parents=True, exist_ok=True)
    lineage, predictions = load_frozen_predictions()
    samples = build_sample_manifest(predictions)
    write_met_files(samples)
    provenance = build_apsim_files(samples)
    lineage.to_csv(OUT / "checkpoint_manifest.csv", index=False)
    predictions.to_csv(OUT / "data_driven_predictions.csv", index=False)
    samples.to_csv(OUT / "sample_manifest.csv", index=False)
    provenance.to_csv(OUT / "parameter_provenance.csv", index=False)
    write_runtime_manifest()

    hash_paths = [LINEAGE, TEMPLATE, ROOT / "experiments" / "apsim_comparison" / "protocol_lock.yaml"]
    hash_paths += sorted(GENERATED.glob("*.apsimx")) + sorted(WEATHER_DIR.glob("*.met"))
    with (OUT / "pre_simulation_checksums.sha256").open("w") as stream:
        for path in hash_paths:
            stream.write(f"{sha256(path)}  {path}\n")
    summary = {
        "sample_count": int(len(samples)),
        "eligible_count": int(samples["eligible"].sum()),
        "prediction_rows": int(len(predictions)),
        "scenarios": [s["name"] for s in scenario_definitions()],
        "apsim_outputs_viewed": False,
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
