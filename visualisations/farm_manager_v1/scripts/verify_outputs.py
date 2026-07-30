#!/usr/bin/env python3
"""Validate generated farm-manager visual assets and frozen headline values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "frozen_sources"
EXPORT = ROOT / "exports"
QA = ROOT / "qa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_close(actual: float, expected: float, tolerance: float = 0.0006) -> None:
    if not np.isclose(actual, expected, atol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def check_numbers() -> list[dict]:
    checks: list[dict] = []
    perf = pd.read_csv(DATA / "source_performance_landscape_seed.csv")
    for contract, method, expected in [
        ("GROUP_complete", "local_scratch", 0.729),
        ("GROUP_complete", "prediction_kd", 1.157),
        ("SPATIAL_complete", "local_scratch", 1.204),
        ("SPATIAL_complete", "prediction_kd", 0.954),
    ]:
        actual = perf.query("contract == @contract and method == @method")["mae"].mean()
        assert_close(actual, expected)
        checks.append({"cell": f"{contract}/{method}", "actual": actual, "expected": expected})

    rose = pd.read_csv(DATA / "roseworthy_run_metrics.csv")
    for condition, route, expected in [
        ("complete", "scratch", 0.465),
        ("no_soil", "scratch", 0.499),
        ("no_soil", "supervised", 0.465),
    ]:
        actual = rose.query("condition == @condition and route == @route")["mae"].mean()
        assert_close(actual, expected)
        checks.append({"cell": f"Roseworthy/{condition}/{route}", "actual": actual, "expected": expected})

    apsim = pd.read_csv(DATA / "apsim_run_summary.csv")
    for contract, condition, model, expected in [
        ("GROUP", "data_driven", "validation_selected_weather_transfer", 0.837),
        ("GROUP", "constrained_central", "APSIM", 0.948),
        ("SPATIAL", "data_driven", "validation_selected_weather_transfer", 0.990),
        ("SPATIAL", "constrained_central", "APSIM", 1.057),
    ]:
        actual = apsim.query("contract == @contract and condition == @condition and model == @model")["mae_mean"].iloc[0]
        assert_close(actual, expected)
        checks.append({"cell": f"{contract}/{condition}/{model}", "actual": actual, "expected": expected})

    rf = pd.read_csv(DATA / "claim_gate_seed_results.csv")
    rf = rf.query("gate == 'late_fusion' and split_id == 'SPATIAL'")
    for condition, field, expected in [
        ("complete", "primary_mae", 0.944),
        ("complete", "official_mae", 1.053),
        ("complete", "secondary_mae", 1.024),
        ("synthetic_random_0_15", "primary_mae", 0.920),
        ("synthetic_random_0_15", "official_mae", 1.015),
        ("synthetic_random_0_15", "secondary_mae", 1.004),
        ("synthetic_no_weather", "primary_mae", 1.024),
        ("synthetic_no_weather", "official_mae", 1.024),
        ("synthetic_no_weather", "secondary_mae", 1.024),
    ]:
        actual = rf.query("condition == @condition")[field].mean()
        assert_close(actual, expected)
        checks.append({"cell": f"RF/{condition}/{field}", "actual": actual, "expected": expected})
    return checks


def check_assets() -> list[dict]:
    expected_slugs = [
        "v01_data_to_decision",
        "v02_long_term_season_context",
        "v03_deployment_contract_reversal",
        "v04_data_availability_value",
        "v05_apsim_complementarity",
        "v06_model_trust_and_scope",
        "v07_paddock_product_preview",
        "v08_data_partnership_ladder",
        "v10_3d_paddock_preview",
    ]
    results = []
    for slug in expected_slugs:
        for ext in ("png", "svg", "pdf"):
            path = EXPORT / ext / f"{slug}.{ext}"
            if not path.exists() or path.stat().st_size < 1500:
                raise AssertionError(f"Missing or undersized asset: {path}")
        image = Image.open(EXPORT / "png" / f"{slug}.png").convert("RGB")
        if image.width < 1500 or image.height < 800:
            raise AssertionError(f"Insufficient resolution: {slug} {image.size}")
        variance = sum(ImageStat.Stat(image).var) / 3
        if variance < 80:
            raise AssertionError(f"Image appears blank: {slug}")
        results.append({"slug": slug, "size": list(image.size), "variance": variance})

    gif = EXPORT / "animation" / "v09_long_term_season_story.gif"
    with Image.open(gif) as image:
        if getattr(image, "n_frames", 1) < 30:
            raise AssertionError("Animation has too few frames")
        image.seek(0)
        first = np.asarray(image.convert("RGB"), dtype=np.int16)
        image.seek(image.n_frames - 1)
        last = np.asarray(image.convert("RGB"), dtype=np.int16)
        if float(np.abs(first - last).mean()) < 1.0:
            raise AssertionError("Animation frames do not change materially")
        results.append({"slug": "v09_long_term_season_story", "frames": image.n_frames, "bytes": gif.stat().st_size})

    html_path = EXPORT / "interactive" / "v10_interactive_3d_paddock_preview.html"
    text = html_path.read_text()
    for token in ("Illustrative 3D paddock", "not a Seabrook prediction", "plotly"):
        if token.lower() not in text.lower():
            raise AssertionError(f"Interactive preview missing token: {token}")
    if "/Users/" in text:
        raise AssertionError("Interactive output contains a personal absolute path")
    if "<script src=" in text.lower() or html_path.stat().st_size < 2_000_000:
        raise AssertionError("Interactive preview is not self-contained")

    for sheet_name in ("contact_sheet.png", "contact_sheet_grayscale.png"):
        sheet = Image.open(QA / sheet_name).convert("RGB")
        if sheet.size != (1920, 1080):
            raise AssertionError(f"Unexpected contact sheet size: {sheet_name} {sheet.size}")
        if sum(ImageStat.Stat(sheet).var) / 3 < 60:
            raise AssertionError(f"Contact sheet appears blank: {sheet_name}")
    return results


def check_manifests() -> None:
    source_manifest = json.loads((QA / "source_manifest.json").read_text())
    for item in source_manifest["sources"]:
        path = ROOT / item["file"]
        if sha256(path) != item["sha256"]:
            raise AssertionError(f"Source manifest mismatch: {path}")
    output_manifest = json.loads((QA / "output_manifest.json").read_text())
    for item in output_manifest["outputs"]:
        path = ROOT / item["file"]
        if sha256(path) != item["sha256"]:
            raise AssertionError(f"Output manifest mismatch: {path}")


def main() -> None:
    report = {
        "status": "PASS",
        "numeric_checks": check_numbers(),
        "asset_checks": check_assets(),
    }
    check_manifests()
    (QA / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "numeric_checks": len(report["numeric_checks"]), "assets": len(report["asset_checks"])}, indent=2))


if __name__ == "__main__":
    main()
