#!/usr/bin/env python3
"""Export path-neutral Roseworthy and Waite publication result sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROUTES = ("scratch", "supervised", "prediction_kd", "representation_kd", "combined_kd", "missing_aware")
CONDITIONS = {"complete": "complete", "synthetic_no_soil": "no_soil"}
SEEDS = (101, 202, 303)


def _validate_roseworthy(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "route", "seed", "n", "mae", "rmse", "r2", "source_dataset"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Roseworthy source missing columns: {sorted(missing)}")
    if len(frame) != 36:
        raise ValueError(f"expected 36 Roseworthy run rows, found {len(frame)}")
    expected = {(condition, route, seed) for condition in CONDITIONS.values() for route in ROUTES for seed in SEEDS}
    observed = set(frame[["condition", "route", "seed"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("Roseworthy source does not contain the complete condition/route/seed grid")
    if set(frame["source_dataset"]) != {"PRIMARY_CYBENCH_MAIZE_US"}:
        raise ValueError("Roseworthy source must use the publication CY-Bench US-maize checkpoints")
    return frame.sort_values(["condition", "route", "seed"]).reset_index(drop=True)


def _validate_waite(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"protocol", "seed", "n_test", "mae", "rmse", "r2", "model"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Waite source missing columns: {sorted(missing)}")
    if len(frame) != 6:
        raise ValueError(f"expected six Waite run rows, found {len(frame)}")
    expected = {(protocol, seed) for protocol in ("temporal_forward", "plot_group") for seed in SEEDS}
    observed = set(frame[["protocol", "seed"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("Waite source does not contain both protocols for all three runs")
    if set(frame["model"]) != {"random_forest_light"}:
        raise ValueError("Waite publication rows must use random_forest_light")
    order = pd.Categorical(frame["protocol"], ["temporal_forward", "plot_group"], ordered=True)
    return frame.assign(_order=order).sort_values(["_order", "seed"]).drop(columns="_order").reset_index(drop=True)


def extract_roseworthy(formal_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for manifest_path in sorted(formal_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        job = manifest.get("job", {})
        if (
            job.get("dataset_id") != "ROSEWORTHY_E5_POINT"
            or job.get("source_dataset") != "PRIMARY_CYBENCH_MAIZE_US"
            or job.get("deployment_condition") not in CONDITIONS
            or job.get("seed") not in SEEDS
        ):
            continue
        strategy = job.get("transfer_strategy")
        route = job.get("source_route")
        if strategy == "target_scratch":
            if job.get("missing_modality_method") != "imputation" or route != "supervised":
                continue
            route = "scratch"
        elif strategy == "ordinary_transfer":
            if route not in ROUTES[1:]:
                continue
        else:
            continue
        metrics_path = manifest_path.with_name("metrics.json")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics beside {manifest_path}")
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "condition": CONDITIONS[job["deployment_condition"]],
                "route": route,
                "seed": int(job["seed"]),
                "n": int(metrics["n"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
                "source_dataset": job["source_dataset"],
                "split_id": job["split_id"],
                "source_job_id": manifest_path.parent.name,
            }
        )
    return _validate_roseworthy(pd.DataFrame(rows))


def extract_waite(metric_index: Path) -> pd.DataFrame:
    source = pd.read_csv(metric_index)
    selected = source[
        source["view_id"].eq("waite_full_environment_proxy")
        & source["model"].eq("random_forest_light")
        & source["validation_axis"].isin(["temporal_forward", "plot_group_diagnostic"])
    ].copy()
    selected["protocol"] = selected["validation_axis"].replace(
        {"plot_group_diagnostic": "plot_group"}
    )
    frame = selected.rename(
        columns={"MAE": "mae", "RMSE": "rmse", "R2": "r2"}
    )[
        ["protocol", "seed", "n_test", "mae", "rmse", "r2", "model", "job_id"]
    ].rename(columns={"job_id": "source_job_id"})
    return _validate_waite(frame)


def validate_included(output: Path) -> dict[str, object]:
    roseworthy = _validate_roseworthy(pd.read_csv(output / "roseworthy_run_metrics.csv"))
    waite = _validate_waite(pd.read_csv(output / "waite_run_metrics.csv"))
    return {
        "status": "PASS",
        "mode": "validate-only",
        "roseworthy_rows": len(roseworthy),
        "waite_rows": len(waite),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, help="Formal campaign jobs/targets directory.")
    parser.add_argument("--waite-index", type=Path, help="Formal Stage 7 metric index.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_manifest/publication"),
        help="Path-neutral publication source-table directory.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate included exported tables.")
    args = parser.parse_args()

    if args.validate_only:
        print(json.dumps(validate_included(args.output), indent=2))
        return 0
    if args.formal_root is None or args.waite_index is None:
        parser.error("--formal-root and --waite-index are required when exporting")

    roseworthy = extract_roseworthy(args.formal_root)
    waite = extract_waite(args.waite_index)
    args.output.mkdir(parents=True, exist_ok=True)
    roseworthy.to_csv(args.output / "roseworthy_run_metrics.csv", index=False)
    waite.to_csv(args.output / "waite_run_metrics.csv", index=False)
    manifest = {
        "status": "PASS",
        "roseworthy_rows": len(roseworthy),
        "waite_rows": len(waite),
        "contains_absolute_paths": False,
    }
    (args.output / "case_metric_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
