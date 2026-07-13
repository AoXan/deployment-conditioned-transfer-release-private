from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "apsim_comparison"))
import build_experiment as build
import analyse_results as analyse


def test_sample_id_parser():
    assert build.parse_sample_id("CY-Bench_wheat_AU|AU-222|2001") == ("AU-222", 2001)


def test_scenarios_have_one_central_per_condition():
    scenarios = build.scenario_definitions()
    assert [s["name"] for s in scenarios].count("constrained_central") == 1
    assert [s["name"] for s in scenarios].count("available_central") == 1


def test_no_yield_drives_parameter_provenance():
    path = ROOT / "outputs" / "apsim_comparison" / "parameter_provenance.csv"
    if path.exists():
        data = pd.read_csv(path)
        assert not data["yield_used_for_configuration"].astype(bool).any()


def test_generated_configs_precede_results():
    summary = ROOT / "outputs" / "apsim_comparison" / "build_summary.json"
    if summary.exists():
        assert json.loads(summary.read_text())["apsim_outputs_viewed"] is False


def test_metric_definition():
    result = analyse.metrics(pd.Series([1.0, 2.0]).to_numpy(), pd.Series([1.5, 1.5]).to_numpy())
    assert result["mae"] == 0.5
    assert result["bias"] == 0.0
