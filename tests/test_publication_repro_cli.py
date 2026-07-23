from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    "validate",
    "data-audit",
    "preprocess",
    "reproduce-primary",
    "evaluate-checkpoints",
    "reproduce-diagnostics",
    "reproduce-cases",
    "reproduce-apsim",
    "render-paper-assets",
    "render-paper-tables",
    "numeric-integrity",
    "verify-publication",
)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agritech_repro", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
    )


def test_every_public_command_has_help() -> None:
    for command in COMMANDS:
        result = run_cli(command, "--help")
        assert result.returncode == 0, (command, result.stderr)
        assert "--dry-run" in result.stdout
        assert "--validate-only" in result.stdout


def test_fixture_validation_is_clean_and_machine_readable() -> None:
    result = run_cli("validate", "--fixture")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["fixture"] is True


def test_repository_root_prefers_a_publication_checkout(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "configs/publication_repro.yaml").write_text("schema_version: test\n")
    monkeypatch.chdir(checkout)
    from agritech_repro.cli import repository_root

    assert repository_root() == checkout


def test_primary_dry_run_does_not_execute_training() -> None:
    result = run_cli("reproduce-primary", "--fixture", "--dry-run")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_RUN"
    assert payload["executed"] is False


def test_exact_primary_runner_requires_a_plan_fingerprint_value() -> None:
    text = (ROOT / "scripts/run_universal_weather_v2_remediation.py").read_text()
    assert 'metavar="PLAN_FINGERPRINT"' in text
    assert "args.confirm_full_campaign != fingerprint" in text


def test_public_config_contains_no_personal_absolute_path() -> None:
    text = "\n".join(
        path.read_text()
        for path in (
            ROOT / "configs/publication_repro.yaml",
            ROOT / "configs/publication/universal_weather_full_campaign_v2_remediation_v1.yaml",
            ROOT / "scripts/run_stage8_formal_adapter.py",
        )
    )
    assert "/Users/" not in text
    assert "OneDrive" not in text
    assert "PRIMARY_CYBENCH_MAIZE_US" in text
    assert "CY-Bench_wheat_AU" in text


def test_missing_authorized_data_is_an_explicit_failure(tmp_path: Path) -> None:
    result = run_cli(
        "reproduce-primary",
        "--data-root",
        str(tmp_path / "missing"),
        "--validate-only",
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "BLOCKED"
    assert "missing" in payload["reason"].lower()


def test_preprocess_validate_only_checks_all_cybench_source_tables(tmp_path: Path) -> None:
    raw = tmp_path / "cybench/raw/cybench-data"
    raw.mkdir(parents=True)
    prefixes = ("yield", "meteo", "crop_calendar", "soil", "location", "crop_mask")
    for crop, country in (("maize", "US"), ("wheat", "AU")):
        for prefix in prefixes:
            (raw / f"{prefix}_{crop}_{country}.csv").write_text("fixture\n")
    result = run_cli("preprocess", "--data-root", str(tmp_path), "--validate-only")
    assert result.returncode == 0, result.stderr + result.stdout
    assert '"inputs": 12' in result.stdout
    assert '"status": "PASS"' in result.stdout


def test_numeric_integrity_replays_frozen_publication_summaries() -> None:
    result = run_cli("numeric-integrity", "--validate-only")
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["checks"] >= 10


def test_release_tracks_the_desktop_submission_snapshot() -> None:
    config = yaml.safe_load((ROOT / "configs/publication_repro.yaml").read_text())
    required = set(config["publication_assets"]["required_files"])
    assert "paper/final_ajcai_manuscript_v4_18_6_submission_anonymous.tex" in required
    assert "paper/final_ajcai_supplement_v4_18_6_submission_anonymous.tex" in required
    assert not any("v4_18_4" in path for path in required)


def test_latest_tex_sources_have_all_local_dependencies() -> None:
    sources = (
        ROOT / "paper/final_ajcai_manuscript_v4_18_6_submission_anonymous.tex",
        ROOT / "paper/final_ajcai_supplement_v4_18_6_submission_anonymous.tex",
    )
    for source in sources:
        text = source.read_text()
        for command, value in re.findall(r"\\(includegraphics|input)\{([^}]+)\}", text):
            candidate = source.parent / value
            choices = (candidate,) if candidate.suffix else tuple(
                candidate.with_suffix(suffix) for suffix in (".tex", ".pdf", ".png")
            )
            assert any(choice.is_file() for choice in choices), (
                source.name,
                command,
                value,
            )
        bibliography = re.search(r"\\bibliography\{([^}]+)\}", text)
        assert bibliography
        assert (source.parent / f"{bibliography.group(1)}.bib").is_file()


def test_case_validation_replays_publication_sources_without_restricted_data() -> None:
    result = run_cli("reproduce-cases", "--validate-only")
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["mode"] == "frozen-output-validation"
    assert payload["case_sources"]["roseworthy_rows"] == 36
    assert payload["case_sources"]["waite_rows"] == 6
    assert payload["case_sources"]["rf_pi_rows"] == 15


def test_table_builder_reconstructs_every_main_table2_block(tmp_path: Path) -> None:
    output = tmp_path / "tables"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_publication_tables.py",
            "--root",
            str(ROOT),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    roseworthy_runs = pd.read_csv(output / "table2_roseworthy_run_metrics.csv")
    roseworthy = pd.read_csv(output / "table2_roseworthy_summary.csv")
    waite = pd.read_csv(output / "table2_waite_run_metrics.csv")
    behaviour = pd.read_csv(output / "table2_behavioural_diagnostics.csv")

    assert len(roseworthy_runs) == 36
    assert len(roseworthy) == 12
    assert len(waite) == 6
    assert len(behaviour) == 6
    assert set(roseworthy["route"]) == {
        "scratch",
        "supervised",
        "prediction_kd",
        "representation_kd",
        "combined_kd",
        "missing_aware",
    }
    supervised_no_soil = roseworthy[
        roseworthy["condition"].eq("no_soil") & roseworthy["route"].eq("supervised")
    ].iloc[0]
    assert round(float(supervised_no_soil["mae"]), 3) == 0.465
    scratch_no_soil = roseworthy[
        roseworthy["condition"].eq("no_soil") & roseworthy["route"].eq("scratch")
    ].iloc[0]
    assert round(float(scratch_no_soil["mae"]), 3) == 0.499
    assert waite["mae"].round(3).tolist() == [0.691, 0.698, 0.685, 0.502, 0.477, 0.499]
    assert dict(zip(behaviour["measure"], behaviour["display_value"])) == {
        "rank_agreement_mean": "-0.040",
        "rank_agreement_group": "0.067",
        "rank_agreement_spatial": "-0.147",
        "top_group_match": "30.0%",
        "ig_completeness": "98.9%",
        "taylor_local_fidelity": "28/180 (15.6%)",
    }


def test_figure_builder_writes_every_referenced_publication_figure(tmp_path: Path) -> None:
    output = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            "figures/final_publication_v4_18/build_final_figures.py",
            "--root",
            str(ROOT),
            "--output-dir",
            str(output),
            "--write",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    expected = {
        *(f"figure_{number:02d}.pdf" for number in range(2, 7)),
        "supplement_figure_supervised_landscapes.pdf",
        "supplement_figure_s4.pdf",
        "supplement_figure_representative_stress.pdf",
    }
    assert expected.issubset({path.name for path in output.glob("*.pdf")})
    signed = output / "final_submission_v4_18_5/supplement_figure_s2.pdf"
    assert signed.is_file()
    assert signed.stat().st_size > 10_000


def test_traceability_uses_the_actual_primary_split_implementation() -> None:
    traceability = (ROOT / "docs/reproducibility/paper_code_traceability.md").read_text()
    assert "src/stage8/splits.py" not in traceability
    assert "src/distillation_v4/universal_weather_v2_remediation/splits.py" in traceability


def test_public_release_scope_has_no_personal_paths() -> None:
    manifest = ROOT / "configs/publication_release_manifest.yaml"
    text = manifest.read_text()
    assert "/Users/" not in text
    assert "OneDrive" not in text


def test_release_allowlist_passes_identity_and_secret_scan() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_publication_release.py", "--validate-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["status"] == "PASS"


def test_direct_publication_scripts_have_nonexecuting_help() -> None:
    scripts = (
        "experiments/apsim_comparison/build_experiment.py",
        "experiments/apsim_comparison/analyse_results.py",
        "experiments/apsim_comparison/render_supplement_tables.py",
        "scripts/render_publication_tables.py",
        "scripts/verify_publication_numbers.py",
        "scripts/acquire_publication_data.py",
        "scripts/prepare_publication_views.py",
        "scripts/run_stage8_formal_adapter.py",
        "scripts/export_publication_case_metrics.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"], cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, (script, result.stderr, result.stdout)
        assert "usage:" in result.stdout.lower()
