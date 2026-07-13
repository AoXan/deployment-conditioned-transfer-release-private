from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"], cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, (script, result.stderr, result.stdout)
        assert "usage:" in result.stdout.lower()
