#!/usr/bin/env python3
"""Evaluate APSIM and frozen Stage 8 predictions on locked common samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "apsim_comparison"
OUT = ROOT / "outputs" / "apsim_comparison"
GENERATED = EXP / "configs" / "generated"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20_260_713


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    denom = np.sum((y - np.mean(y)) ** 2)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / denom) if denom > 0 else np.nan,
        "bias": float(np.mean(residual)),
    }


def read_apsim_database(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = path.stem
    with sqlite3.connect(path) as con:
        report = pd.read_sql_query(
            'SELECT s.Name AS simulation_name, r.Date AS report_date, '
            'r.YieldKgHa AS yield_kg_ha, r.GrainProtein AS grain_protein '
            'FROM Report r JOIN _Simulations s ON r.SimulationID=s.ID', con
        )
        messages = pd.read_sql_query(
            "SELECT s.Name AS simulation_name, m.ComponentName, m.Date, m.Message, m.MessageType "
            "FROM _Messages m JOIN _Simulations s ON m.SimulationID=s.ID", con
        )
    expected_prefix = scenario + "__"
    if not report["simulation_name"].str.startswith(expected_prefix).all():
        raise ValueError(f"Unexpected simulation names in {path}")
    parts = report["simulation_name"].str[len(expected_prefix):].str.rsplit("__", n=1, expand=True)
    report["adm_id"] = parts[0].str.replace("_", "-", regex=False)
    report["year"] = parts[1].astype(int)
    report["scenario"] = scenario
    report["apsim_prediction_t_ha"] = report["yield_kg_ha"] / 1000.0
    # Harvest and end-of-season can both emit rows. Maximum seasonal grain mass
    # is fixed in the pre-evaluation correction log.
    seasonal = report.groupby(["scenario", "adm_id", "year"], as_index=False).agg(
        apsim_prediction_t_ha=("apsim_prediction_t_ha", "max"),
        report_rows=("apsim_prediction_t_ha", "size"),
        final_report_date=("report_date", "max"),
    )
    return seasonal, messages


def load_all_apsim() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, messages = [], []
    for path in sorted(GENERATED.glob("*.db")):
        p, m = read_apsim_database(path)
        predictions.append(p)
        m.insert(0, "scenario", path.stem)
        messages.append(m)
    return pd.concat(predictions, ignore_index=True), pd.concat(messages, ignore_index=True)


def model_comparison_rows(data_predictions: pd.DataFrame) -> pd.DataFrame:
    selected = data_predictions[
        data_predictions["model_role"].isin(["local_scratch", "supervised_transfer"])
        | data_predictions["is_validation_selected_transfer"].astype(bool)
    ].copy()
    selected["comparison_model"] = np.where(
        selected["is_validation_selected_transfer"].astype(bool),
        "validation_selected_weather_transfer",
        selected["model_role"],
    )
    return selected


def bootstrap_difference(frame: pd.DataFrame, reps: int = BOOTSTRAP_REPS) -> tuple[float, float, float]:
    d = frame["paired_abs_error_difference"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(d), size=(reps, len(d)))
    boot = d[indices].mean(axis=1)
    return float(d.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    required = [OUT / "sample_manifest.csv", OUT / "data_driven_predictions.csv"]
    databases = sorted(GENERATED.glob("*.db"))
    if args.dry_run:
        print(json.dumps({"executed": False, "required": [str(path) for path in required], "database_count": len(databases)}, indent=2))
        return
    if args.validate_only:
        missing = [str(path) for path in required if not path.is_file()]
        if missing or not databases:
            raise FileNotFoundError(f"missing APSIM analysis input: files={missing}, databases={len(databases)}")
        print(json.dumps({"status": "PASS", "database_count": len(databases)}, indent=2))
        return
    samples = pd.read_csv(OUT / "sample_manifest.csv")
    data_predictions = pd.read_csv(OUT / "data_driven_predictions.csv")
    apsim, messages = load_all_apsim()
    apsim = apsim.merge(samples[["sample_id", "adm_id", "year", "observed_yield_t_ha"]], on=["adm_id", "year"], validate="many_to_one")
    apsim.to_csv(OUT / "apsim_native_predictions.csv", index=False)

    expected = pd.MultiIndex.from_product(
        [sorted(apsim["scenario"].unique()), samples.loc[samples["eligible"], "sample_id"]],
        names=["scenario", "sample_id"],
    ).to_frame(index=False)
    observed = apsim[["scenario", "sample_id"]].drop_duplicates()
    failures = expected.merge(observed.assign(present=True), on=["scenario", "sample_id"], how="left")
    failures = failures[failures["present"].isna()].drop(columns="present")
    failures["failure_reason"] = "no APSIM report row"
    failures.to_csv(OUT / "failure_ledger.csv", index=False)

    message_summary = messages.groupby(["scenario", "MessageType"]).size().rename("count").reset_index()
    message_summary.to_csv(OUT / "apsim_message_summary.csv", index=False)

    common_models = model_comparison_rows(data_predictions)
    common = common_models.merge(
        apsim[["scenario", "sample_id", "apsim_prediction_t_ha"]], on="sample_id", how="inner", validate="many_to_many"
    )
    common["data_abs_error"] = (common["y_pred"] - common["y_true"]).abs()
    common["apsim_abs_error"] = (common["apsim_prediction_t_ha"] - common["y_true"]).abs()
    common["paired_abs_error_difference"] = common["data_abs_error"] - common["apsim_abs_error"]
    common.to_csv(OUT / "common_sample_predictions.csv", index=False)
    common.to_csv(OUT / "aggregated_predictions.csv", index=False)

    metric_rows: list[dict] = []
    for (contract, seed, model), group in common_models.groupby(["contract", "seed", "comparison_model"]):
        row = {"contract": contract, "seed": int(seed), "condition": "data_driven", "model": model, "coverage": 1.0, "failed_simulations": 0}
        row.update(metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
        metric_rows.append(row)
    for (contract, seed), base in common_models.groupby(["contract", "seed"]):
        ids = base["sample_id"].drop_duplicates()
        target_n = len(ids)
        for scenario, ap in apsim[apsim["sample_id"].isin(ids)].groupby("scenario"):
            ap = ap.drop_duplicates("sample_id")
            row = {
                "contract": contract, "seed": int(seed), "condition": scenario,
                "model": "APSIM", "coverage": len(ap) / target_n,
                "failed_simulations": target_n - len(ap),
                "zero_grain_outputs": int((ap["apsim_prediction_t_ha"] <= 1e-12).sum()),
            }
            row.update(metrics(ap["observed_yield_t_ha"].to_numpy(), ap["apsim_prediction_t_ha"].to_numpy()))
            metric_rows.append(row)
    metric_table = pd.DataFrame(metric_rows).sort_values(["contract", "seed", "condition", "model"])
    metric_table.to_csv(OUT / "metrics.csv", index=False)

    boot_rows: list[dict] = []
    central = common[common["scenario"].isin(["constrained_central", "available_central"])]
    for keys, group in central.groupby(["contract", "seed", "comparison_model", "scenario"]):
        contract, seed, model, scenario = keys
        point, lo, hi = bootstrap_difference(group.drop_duplicates(["sample_id", "comparison_model", "scenario"]))
        boot_rows.append({
            "contract": contract, "seed": int(seed), "data_model": model, "apsim_condition": scenario,
            "n": group["sample_id"].nunique(), "mean_abs_error_difference_data_minus_apsim": point,
            "ci_low": lo, "ci_high": hi, "bootstrap_replicates": BOOTSTRAP_REPS,
            "interpretation": "negative favours data-driven; positive favours APSIM",
        })
    pd.DataFrame(boot_rows).to_csv(OUT / "paired_bootstrap.csv", index=False)

    sensitivity = metric_table[metric_table["model"].eq("APSIM")].copy()
    sensitivity.to_csv(OUT / "sensitivity_metrics.csv", index=False)
    range_rows = sensitivity.groupby(["contract", "seed"]).agg(
        scenario_count=("condition", "nunique"),
        mae_min=("mae", "min"), mae_max=("mae", "max"),
        rmse_min=("rmse", "min"), rmse_max=("rmse", "max"),
        r2_min=("r2", "min"), r2_max=("r2", "max"),
        bias_min=("bias", "min"), bias_max=("bias", "max"),
    ).reset_index()
    range_rows.to_csv(OUT / "sensitivity_range.csv", index=False)

    run_summary = metric_table.groupby(["contract", "condition", "model"]).agg(
        runs=("seed", "nunique"), n_mean=("n", "mean"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        bias_mean=("bias", "mean"), bias_sd=("bias", "std"),
        coverage_min=("coverage", "min"), failures_total=("failed_simulations", "sum"),
        zero_grain_outputs_mean=("zero_grain_outputs", "mean"),
    ).reset_index()
    run_summary.to_csv(OUT / "run_summary.csv", index=False)

    integrity = {
        "expected_samples_per_scenario": int(samples["eligible"].sum()),
        "apsim_scenarios": int(apsim["scenario"].nunique()),
        "apsim_prediction_rows": int(len(apsim)),
        "missing_apsim_rows": int(len(failures)),
        "common_prediction_rows": int(len(common)),
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "test_yield_used_for_parameter_selection": False,
        "equivalence_test_performed": False,
    }
    (OUT / "analysis_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")

    with (OUT / "result_checksums.sha256").open("w") as stream:
        for path in sorted(OUT.glob("*.csv")) + sorted(GENERATED.glob("*.db")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stream.write(f"{digest}  {path}\n")
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
