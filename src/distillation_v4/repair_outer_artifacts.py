from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
from typing import Any

import pandas as pd


ARTIFACT_FIELDS = (
    ("predictions", "predictions_sha256"),
    ("metrics", "metrics_sha256"),
    (
        "fold_assignments",
        "fold_assignments_sha256",
    ),
    ("checkpoint", "checkpoint_sha256"),
    ("preprocessor", "preprocessor_sha256"),
    ("manifest", "manifest_sha256"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_repair_outer_record(
    *,
    output: Path,
    record_path: Path,
    expected_job: dict[str, Any],
    expected_execution_fingerprint: str,
) -> dict[str, Any]:
    if not record_path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_RECORD_MISSING"
        )

    record: dict[str, Any] = json.loads(
        record_path.read_text()
    )

    if record.get("status") != (
        "ENGINEERING_ACCEPTED"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_RECORD_NOT_ACCEPTED"
        )

    for field in (
        "job_id",
        "dataset",
        "route",
        "fold",
        "seed",
    ):
        if record.get(field) != expected_job.get(
            field
        ):
            raise RuntimeError(
                "REPAIR_OUTER_RECORD_LINEAGE_MISMATCH:"
                + field
            )

    if record.get(
        "execution_fingerprint"
    ) != expected_execution_fingerprint:
        raise RuntimeError(
            "REPAIR_OUTER_EXECUTION_FINGERPRINT_MISMATCH"
        )

    if record.get(
        "route_fingerprint_before_outer_test"
    ) != expected_job.get(
        "route_fingerprint_before_outer_test"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_FINGERPRINT_MISMATCH"
        )

    for path_field, hash_field in ARTIFACT_FIELDS:
        relative = record.get(path_field)
        expected_hash = record.get(hash_field)

        if not relative or not expected_hash:
            raise RuntimeError(
                "REPAIR_OUTER_ARTIFACT_LINEAGE_INCOMPLETE:"
                + path_field
            )

        artifact = (
            output / str(relative)
        ).resolve()

        try:
            artifact.relative_to(output.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "REPAIR_OUTER_ARTIFACT_OUTSIDE_OUTPUT:"
                + path_field
            ) from exc

        if not artifact.is_file():
            raise RuntimeError(
                "REPAIR_OUTER_ARTIFACT_MISSING:"
                + path_field
            )

        if sha256_file(artifact) != str(
            expected_hash
        ):
            raise RuntimeError(
                "REPAIR_OUTER_ARTIFACT_HASH_MISMATCH:"
                + path_field
            )

    prediction_path = (
        output / str(record["predictions"])
    )
    metric_path = (
        output / str(record["metrics"])
    )

    predictions = pd.read_csv(prediction_path)
    metrics = json.loads(metric_path.read_text())

    required_columns = {
        "sample_id",
        "y_true",
        "y_pred",
    }

    if not required_columns.issubset(
        predictions.columns
    ):
        raise RuntimeError(
            "REPAIR_OUTER_PREDICTION_COLUMNS_INVALID"
        )

    if predictions.empty:
        raise RuntimeError(
            "REPAIR_OUTER_PREDICTIONS_EMPTY"
        )

    if predictions["sample_id"].astype(
        str
    ).duplicated().any():
        raise RuntimeError(
            "REPAIR_OUTER_SAMPLE_ID_DUPLICATE"
        )

    mae = float(
        (
            predictions["y_true"]
            - predictions["y_pred"]
        ).abs().mean()
    )

    if not math.isfinite(mae):
        raise RuntimeError(
            "REPAIR_OUTER_RECOMPUTED_MAE_NONFINITE"
        )

    if abs(
        mae - float(metrics["mae"])
    ) > 5e-7:
        raise RuntimeError(
            "REPAIR_OUTER_METRIC_RECOMPUTE_MISMATCH"
        )

    fold_path = (
        output / str(record["fold_assignments"])
    )
    folds = pd.read_csv(fold_path)

    required_fold_columns = {
        "sample_id",
        "split",
        "year",
    }
    if not required_fold_columns.issubset(
        folds.columns
    ):
        raise RuntimeError(
            "REPAIR_OUTER_FOLD_COLUMNS_INVALID"
        )

    test_ids = set(
        folds.loc[
            folds["split"].eq("test"),
            "sample_id",
        ].astype(str)
    )
    prediction_ids = set(
        predictions["sample_id"].astype(str)
    )

    if prediction_ids != test_ids:
        raise RuntimeError(
            "REPAIR_OUTER_TEST_ID_SET_MISMATCH"
        )

    split_sets = {
        split: set(
            folds.loc[
                folds["split"].eq(split),
                "sample_id",
            ].astype(str)
        )
        for split in (
            "train",
            "calibration",
            "test",
        )
    }

    if (
        split_sets["train"]
        & split_sets["calibration"]
        or split_sets["train"]
        & split_sets["test"]
        or split_sets["calibration"]
        & split_sets["test"]
    ):
        raise RuntimeError(
            "REPAIR_OUTER_FOLD_ASSIGNMENT_OVERLAP"
        )

    manifest_path = (
        output / str(record["manifest"])
    )
    manifest = json.loads(
        manifest_path.read_text()
    )

    if manifest.get(
        "teacher_refit_performed"
    ) is not False:
        raise RuntimeError(
            "REPAIR_OUTER_TEACHER_REFIT_DETECTED"
        )

    if manifest.get(
        "outer_test_may_enable_downstream_jobs"
    ) is not False:
        raise RuntimeError(
            "REPAIR_OUTER_DOWNSTREAM_MUTATION_ALLOWED"
        )

    if float(
        manifest.get(
            "checkpoint_replay_max_abs_error",
            float("inf"),
        )
    ) > 1e-6:
        raise RuntimeError(
            "REPAIR_OUTER_CHECKPOINT_REPLAY_FAILED"
        )

    marker = (
        record_path.parent
        / "completion_marker.json"
    )

    if not marker.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_COMPLETION_MARKER_MISSING"
        )

    marker_payload = json.loads(
        marker.read_text()
    )

    if marker_payload.get("job_id") != record.get(
        "job_id"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_COMPLETION_MARKER_ID_MISMATCH"
        )

    if marker_payload.get("status") != (
        "ENGINEERING_ACCEPTED"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_COMPLETION_MARKER_INVALID"
        )

    return record


# STAGE8_V4_SOIL_SCENARIO_VALIDATION
_validate_repair_outer_record_without_soil_scenarios = (
    validate_repair_outer_record
)


def validate_repair_outer_record(
    *,
    output: Path,
    record_path: Path,
    expected_job: dict[str, Any],
    expected_execution_fingerprint: str,
) -> dict[str, Any]:
    record = (
        _validate_repair_outer_record_without_soil_scenarios(
            output=output,
            record_path=record_path,
            expected_job=expected_job,
            expected_execution_fingerprint=(
                expected_execution_fingerprint
            ),
        )
    )

    if record.get("route") not in {
        "soil_direct",
        "missing_aware",
    }:
        return record

    for path_field, hash_field in (
        (
            "scenario_predictions",
            "scenario_predictions_sha256",
        ),
        (
            "scenario_metrics",
            "scenario_metrics_sha256",
        ),
    ):
        relative = record.get(path_field)
        expected_hash = record.get(
            hash_field
        )

        if not relative or not expected_hash:
            raise RuntimeError(
                "REPAIR_OUTER_SOIL_SCENARIO_"
                "ARTIFACT_LINEAGE_INCOMPLETE:"
                + path_field
            )

        artifact = (
            output / str(relative)
        ).resolve()

        try:
            artifact.relative_to(
                output.resolve()
            )
        except ValueError as exc:
            raise RuntimeError(
                "REPAIR_OUTER_SOIL_SCENARIO_"
                "ARTIFACT_OUTSIDE_OUTPUT:"
                + path_field
            ) from exc

        if not artifact.is_file():
            raise RuntimeError(
                "REPAIR_OUTER_SOIL_SCENARIO_"
                "ARTIFACT_MISSING:"
                + path_field
            )

        if sha256_file(artifact) != str(
            expected_hash
        ):
            raise RuntimeError(
                "REPAIR_OUTER_SOIL_SCENARIO_"
                "ARTIFACT_HASH_MISMATCH:"
                + path_field
            )

    scenario_metrics = json.loads(
        (
            output
            / str(
                record[
                    "scenario_metrics"
                ]
            )
        ).read_text()
    )

    if scenario_metrics.get(
        "primary_scenario"
    ) != "native":
        raise RuntimeError(
            "REPAIR_OUTER_SOIL_PRIMARY_"
            "SCENARIO_INVALID"
        )

    if set(
        scenario_metrics
    ).issuperset(
        {
            "native",
            "weather_only",
            "complete_case",
        }
    ) is False:
        raise RuntimeError(
            "REPAIR_OUTER_SOIL_SCENARIOS_INCOMPLETE"
        )

    return record
