from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (EmptyDataError, Exception):
        return pd.DataFrame()


def value(number, scientific: bool = False) -> str:
    if number is None:
        return "—"

    try:
        if scientific:
            return f"{float(number):.3e}"
        return f"{float(number):.5g}"
    except Exception:
        return html.escape(str(number))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()

    progress = {}

    if (output / "progress.json").is_file():
        try:
            progress = json.loads(
                (output / "progress.json").read_text()
            )
        except Exception:
            progress = {}

    replay = read_csv(
        output / "checkpoint_forward_replay.csv"
    )
    quality = read_csv(
        output / "attribution_quality.csv"
    )
    summary = read_csv(
        output / "kd_vs_supervised_summary.csv"
    )
    failures = read_csv(
        output / "failure_ledger.csv"
    )

    max_replay = (
        replay["max_replay_error"].max()
        if (
            not replay.empty
            and "max_replay_error"
            in replay.columns
        )
        else None
    )

    max_array = (
        replay[
            [
                "npz_train_scaled_array_error",
                "npz_test_scaled_array_error",
            ]
        ].max().max()
        if (
            not replay.empty
            and {
                "npz_train_scaled_array_error",
                "npz_test_scaled_array_error",
            }.issubset(replay.columns)
        )
        else None
    )

    shap_residual = (
        quality[
            "shap_efficiency_residual"
        ].abs().mean()
        if (
            not quality.empty
            and "shap_efficiency_residual"
            in quality.columns
        )
        else None
    )

    first_residual = (
        quality[
            "taylor_first_residual"
        ].abs().mean()
        if (
            not quality.empty
            and "taylor_first_residual"
            in quality.columns
        )
        else None
    )

    second_residual = (
        quality[
            "taylor_second_residual"
        ].abs().mean()
        if (
            not quality.empty
            and "taylor_second_residual"
            in quality.columns
        )
        else None
    )

    replay_rows = ""

    if not replay.empty:
        for row in replay.itertuples(index=False):
            replay_rows += (
                "<tr>"
                f"<td>{html.escape(str(row.contract))}</td>"
                f"<td>{html.escape(str(row.method))}</td>"
                f"<td>{int(row.seed)}</td>"
                f"<td>{float(row.max_replay_error):.3e}</td>"
                f"<td>{html.escape(str(row.status))}</td>"
                "</tr>"
            )

    top_rows = ""

    if (
        not summary.empty
        and "delta_mean_abs_shap_mean"
        in summary.columns
    ):
        top = (
            summary.assign(
                magnitude=lambda frame: frame[
                    "delta_mean_abs_shap_mean"
                ].abs()
            )
            .sort_values(
                "magnitude",
                ascending=False,
            )
            .head(20)
        )

        for row in top.itertuples(index=False):
            top_rows += (
                "<tr>"
                f"<td>{html.escape(str(row.contract))}</td>"
                f"<td>{html.escape(str(row.method))}</td>"
                f"<td>{html.escape(str(row.feature_name))}</td>"
                f"<td>{float(row.delta_mean_abs_shap_mean):.5g}</td>"
                f"<td>{float(row.delta_mean_abs_taylor_first_mean):.5g}</td>"
                f"<td>{float(row.delta_mean_abs_derivative_per_raw_unit_mean):.5g}</td>"
                "</tr>"
            )

    failure_rows = ""

    if not failures.empty:
        for row in failures.tail(10).itertuples(
            index=False
        ):
            failure_rows += (
                "<tr>"
                f"<td>{html.escape(str(getattr(row, 'phase', '')))}</td>"
                f"<td>{html.escape(str(getattr(row, 'contract', '')))}</td>"
                f"<td>{html.escape(str(getattr(row, 'method', '')))}</td>"
                f"<td>{html.escape(str(getattr(row, 'message', '')))}</td>"
                "</tr>"
            )

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Raw-weather SHAP + Taylor</title>
<style>
body {{
  margin: 0;
  padding: 24px;
  background: #0d1117;
  color: #e6edf3;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
  gap: 14px;
  margin-bottom: 22px;
}}
.card {{
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 18px;
}}
.label {{
  color: #8b949e;
  font-size: 13px;
}}
.metric {{
  font-size: 26px;
  font-weight: 700;
  margin-top: 7px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
th, td {{
  padding: 8px;
  border-bottom: 1px solid #30363d;
  text-align: left;
}}
th {{
  color: #8b949e;
}}
.small {{
  color: #8b949e;
  font-size: 13px;
}}
</style>
</head>
<body>
<h1>Raw-weather SHAP + Taylor</h1>
<p class="small">
Formal preprocessing reconstruction → 30-cell replay gate →
raw-value permutation SHAP → raw-unit Taylor derivatives.
Refreshes every 5 seconds.
</p>

<div class="grid">
<div class="card">
<div class="label">Status</div>
<div class="metric">{html.escape(str(progress.get("status", "WAITING")))}</div>
</div>
<div class="card">
<div class="label">Phase</div>
<div class="metric">{html.escape(str(progress.get("phase", "WAITING")))}</div>
</div>
<div class="card">
<div class="label">Replay cells</div>
<div class="metric">{progress.get("replayed_cells", 0)}/30</div>
</div>
<div class="card">
<div class="label">Attributed cells</div>
<div class="metric">{progress.get("attributed_cells", 0)}/30</div>
</div>
<div class="card">
<div class="label">Samples</div>
<div class="metric">{progress.get("completed_samples", 0)}/{progress.get("total_samples", 0)}</div>
</div>
<div class="card">
<div class="label">Failures</div>
<div class="metric">{len(failures)}</div>
</div>
<div class="card">
<div class="label">Max replay error</div>
<div class="metric">{value(max_replay, True)}</div>
</div>
<div class="card">
<div class="label">Max scaled-array error</div>
<div class="metric">{value(max_array, True)}</div>
</div>
<div class="card">
<div class="label">Mean |SHAP residual|</div>
<div class="metric">{value(shap_residual, True)}</div>
</div>
<div class="card">
<div class="label">Mean |Taylor-1 residual|</div>
<div class="metric">{value(first_residual, True)}</div>
</div>
<div class="card">
<div class="label">Mean |Taylor-2 residual|</div>
<div class="metric">{value(second_residual, True)}</div>
</div>
</div>

<div class="card">
<h2>Current activity</h2>
<p><strong>Cell:</strong> {html.escape(str(progress.get("current_cell", "—")))}</p>
<p><strong>Sample:</strong> {html.escape(str(progress.get("current_sample", "—")))}</p>
<p><strong>Message:</strong> {html.escape(str(progress.get("last_message", "—")))}</p>
<p><strong>ETA:</strong> {value(progress.get("eta_seconds"))} seconds</p>
</div>

<div class="card">
<h2>Formal replay matrix</h2>
<table>
<thead>
<tr>
<th>Contract</th>
<th>Method</th>
<th>Seed</th>
<th>Max error</th>
<th>Status</th>
</tr>
</thead>
<tbody>{replay_rows}</tbody>
</table>
</div>

<div class="card">
<h2>Largest KD–supervised shifts</h2>
<table>
<thead>
<tr>
<th>Contract</th>
<th>Method</th>
<th>Feature</th>
<th>Δ mean |SHAP|</th>
<th>Δ mean |Taylor-1|</th>
<th>Δ mean |raw derivative|</th>
</tr>
</thead>
<tbody>{top_rows}</tbody>
</table>
</div>

<div class="card">
<h2>Failures</h2>
<table>
<thead>
<tr>
<th>Phase</th>
<th>Contract</th>
<th>Method</th>
<th>Message</th>
</tr>
</thead>
<tbody>{failure_rows}</tbody>
</table>
</div>

<p class="small">
Attributions describe model behaviour in raw weather-value space.
They are not causal effects.
</p>
</body>
</html>
"""

    (output / "dashboard.html").write_text(page)


if __name__ == "__main__":
    main()
