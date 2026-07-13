#!/usr/bin/env python3
"""Render APSIM supplementary tables from verified CSV outputs."""

import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "apsim_comparison"
TABLES = ROOT / "paper" / "generated_tables"


def fmt(mean, sd):
    return f"{mean:.3f} ({0.0 if pd.isna(sd) else sd:.3f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tables_dir = args.tables_dir.resolve()
    required = [OUT / "run_summary.csv", OUT / "paired_bootstrap.csv", OUT / "sensitivity_range.csv"]
    if args.dry_run:
        print(json.dumps({"executed": False, "tables_dir": str(tables_dir), "required": [str(path) for path in required]}, indent=2))
        return
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing APSIM table input: {missing}")
    if args.validate_only:
        print(json.dumps({"status": "PASS", "inputs": len(required)}, indent=2))
        return
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "apsim_input_conditions.tex").write_text(r"""\begin{table}[tb]
\centering
\caption{Input conditions and provenance for APSIM and data-driven models. Both data-driven routes deploy with the ten weather summaries. APSIM uses the underlying daily weather and the regional crop calendar because these are required to run the process model.}
\label{tab:apsim-inputs}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{.19\textwidth}p{.24\textwidth}p{.24\textwidth}p{.23\textwidth}}
\toprule
Input & Weather-only models & APSIM constrained-input & APSIM available-input\\
\midrule
Weather & Ten seasonal summaries & Daily CY-Bench minimum/maximum temperature, precipitation, and radiation & Same daily weather\\
Crop timing & Implicit in the seasonal window & Regional start/end of season; sowing at rounded start day & Same regional calendar\\
Soil & Not used at deployment & Fixed APSIM example soil & CY-Bench AWC, bulk density, and drainage mapped to a fixed 1.8-m profile\\
Initial water & Not used & 50\% of plant-available water & Pre-sowing root-zone moisture proxy; 50\% fallback when unavailable\\
Initial N & Not used & 50 kg NO$_3$-N/ha and 10 kg NH$_4$-N/ha & Same fixed initial N\\
Fertiliser & Not used & 50 kg N/ha at sowing & Same fixed fertiliser\\
Cultivar & Not used & Hartog & Hartog\\
Selection & Validation MSE & Fixed before simulation & Fixed mapping without yield\\
\bottomrule
\end{tabular}
\end{table}
""")

    summary = pd.read_csv(OUT / "run_summary.csv")
    wanted = summary[
        summary["condition"].isin(["data_driven", "constrained_central", "available_central"])
        & summary["model"].isin(["local_scratch", "supervised_transfer", "validation_selected_weather_transfer", "APSIM"])
    ].copy()
    order = {
        ("data_driven", "local_scratch"): 0,
        ("data_driven", "supervised_transfer"): 1,
        ("data_driven", "validation_selected_weather_transfer"): 2,
        ("constrained_central", "APSIM"): 3,
        ("available_central", "APSIM"): 4,
    }
    labels = {
        ("data_driven", "local_scratch"): "Local scratch",
        ("data_driven", "supervised_transfer"): "Supervised transfer",
        ("data_driven", "validation_selected_weather_transfer"): "Validation-selected transfer",
        ("constrained_central", "APSIM"): "APSIM constrained",
        ("available_central", "APSIM"): "APSIM available",
    }
    wanted["order"] = [order[(r.condition, r.model)] for r in wanted.itertuples()]
    lines = []
    for contract in ["GROUP", "SPATIAL"]:
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textbf{{{contract}}}}}\\")
        for r in wanted[wanted.contract.eq(contract)].sort_values("order").itertuples():
            label = labels[(r.condition, r.model)]
            zero = "--" if pd.isna(r.zero_grain_outputs_mean) else f"{r.zero_grain_outputs_mean:.1f}"
            n_display = "93--115" if contract == "GROUP" else "31"
            lines.append(
                f"{label} & {n_display} & {fmt(r.mae_mean,r.mae_sd)} & {fmt(r.rmse_mean,r.rmse_sd)} & "
                f"{fmt(r.r2_mean,r.r2_sd)} & {fmt(r.bias_mean,r.bias_sd)} & {r.coverage_min:.3f} & {zero}\\\\"
            )
        if contract == "GROUP":
            lines.append(r"\addlinespace")
    body = "\n".join(lines)
    (tables_dir / "apsim_common_comparison.tex").write_text(r"""\begin{landscape}
\begin{table}[tb]
\centering
\caption{Common-sample predictive comparison between APSIM and data-driven models. Values are mean (SD) across S1--S3; MAE, RMSE, and bias are in t/ha. The $n$ column gives the per-run range. GROUP runs use different held-out regions, whereas SPATIAL runs share the same 31 rows. APSIM is deterministic for a given sample and configuration. ``Zero'' is the mean per-run count of simulations with a valid report but zero end-of-season grain.}
\label{tab:apsim-comparison}
\small
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{lrrrrrrr}
\toprule
Model & $n$ & MAE & RMSE & $R^2$ & Bias & Coverage & Zero\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
\end{landscape}
""")

    boot = pd.read_csv(OUT / "paired_bootstrap.csv")
    boot = boot[boot["data_model"].eq("validation_selected_weather_transfer")]
    lines = []
    for r in boot.sort_values(["contract", "apsim_condition", "seed"]).itertuples():
        condition = "Available" if r.apsim_condition == "available_central" else "Constrained"
        lines.append(f"{r.contract} & S{ {101:1,202:2,303:3}[r.seed] } & {condition} & {r.n} & {r.mean_abs_error_difference_data_minus_apsim:.3f} & [{r.ci_low:.3f}, {r.ci_high:.3f}]\\\\")
    (tables_dir / "apsim_paired_differences.tex").write_text(r"""\begin{table}[tb]
\centering
\caption{Paired absolute-error differences for the validation-selected weather-only transfer route relative to APSIM. The difference is data-driven minus APSIM, so negative values favour the transferred predictor. Intervals are 95\% paired bootstrap intervals over region--years (10,000 resamples).}
\label{tab:apsim-paired}
\small
\begin{tabular}{lllr rr}
\toprule
Contract & Run & APSIM input & $n$ & Mean difference & 95\% interval\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    sens = pd.read_csv(OUT / "sensitivity_range.csv")
    lines = []
    for r in sens.sort_values(["contract", "seed"]).itertuples():
        run = {101: "S1", 202: "S2", 303: "S3"}[r.seed]
        lines.append(f"{r.contract} & {run} & {r.scenario_count} & {r.mae_min:.3f}--{r.mae_max:.3f} & {r.rmse_min:.3f}--{r.rmse_max:.3f} & {r.r2_min:.3f}--{r.r2_max:.3f} & {r.bias_min:.3f}--{r.bias_max:.3f}\\\\")
    (tables_dir / "apsim_sensitivity.tex").write_text(r"""\begin{table}[tb]
\centering
\refstepcounter{table}
\parbox{.96\textwidth}{\centering\textbf{Table \thetable.} APSIM one-factor sensitivity across sowing date, initial water, initial nitrate, fertiliser, cultivar, and the available-input soil mapping. Ranges include the two central configurations and all ten prespecified variants; they are descriptive and are not used to select the reported central result.}\par\medskip
\label{tab:apsim-sensitivity}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrr}
\toprule
Contract & Run & Scenarios & MAE range & RMSE range & $R^2$ range & Bias range\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    main()
