from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any


ARTIFACT_FIELDS = (
    ("checkpoint", "checkpoint_sha256"),
    ("preprocessor", "preprocessor_sha256"),
    (
        "prediction_artifact",
        "prediction_artifact_sha256",
    ),
    (
        "representation_artifact",
        "representation_artifact_sha256",
    ),
    ("loss_ledger", "loss_ledger_sha256"),
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


def validate_phase4_job_artifacts(
    *,
    record: dict[str, Any],
    output_root: Path,
    require_completion_marker: bool = True,
) -> None:
    if record.get("status") != "ENGINEERING_ACCEPTED":
        raise RuntimeError(
            "PHASE4_RECORD_NOT_ENGINEERING_ACCEPTED"
        )

    if record.get("outer_test_used") is not False:
        raise RuntimeError(
            "PHASE4_RECORD_OUTER_TEST_FORBIDDEN"
        )

    route = str(record.get("route"))

    if route == "reference":
        checkpoint = record.get("checkpoint")
        checkpoint_hash = record.get(
            "checkpoint_sha256"
        )

        if not checkpoint or not checkpoint_hash:
            raise RuntimeError(
                "PHASE4_REFERENCE_CHECKPOINT_MISSING"
            )

        path = (
            output_root / str(checkpoint)
        ).resolve()

        try:
            path.relative_to(output_root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "PHASE4_REFERENCE_OUTSIDE_OUTPUT_ROOT"
            ) from exc

        if not path.is_file():
            raise RuntimeError(
                "PHASE4_REFERENCE_CHECKPOINT_FILE_MISSING"
            )

        if sha256_file(path) != str(checkpoint_hash):
            raise RuntimeError(
                "PHASE4_REFERENCE_CHECKPOINT_HASH_MISMATCH"
            )

        prediction_value = record.get(
            "prediction_artifact"
        )
        prediction_hash = record.get(
            "prediction_artifact_sha256"
        )

        if not prediction_value or not prediction_hash:
            raise RuntimeError(
                "PHASE4_REFERENCE_PREDICTION_LINEAGE_MISSING"
            )

        prediction_path = (
            output_root / str(prediction_value)
        ).resolve()

        try:
            prediction_path.relative_to(
                output_root.resolve()
            )
        except ValueError as exc:
            raise RuntimeError(
                "PHASE4_REFERENCE_PREDICTION_OUTSIDE_ROOT"
            ) from exc

        if not prediction_path.is_file():
            raise RuntimeError(
                "PHASE4_REFERENCE_PREDICTION_FILE_MISSING"
            )

        if sha256_file(
            prediction_path
        ) != str(prediction_hash):
            raise RuntimeError(
                "PHASE4_REFERENCE_PREDICTION_HASH_MISMATCH"
            )

        validation_mae = record.get(
            "validation_mae"
        )
        if validation_mae is None:
            raise RuntimeError(
                "PHASE4_REFERENCE_VALIDATION_MAE_MISSING"
            )

    else:
        for path_field, hash_field in ARTIFACT_FIELDS:
            value = record.get(path_field)
            digest = record.get(hash_field)

            if not value or not digest:
                raise RuntimeError(
                    "PHASE4_ARTIFACT_LINEAGE_INCOMPLETE:"
                    + path_field
                )

            path = (
                output_root / str(value)
            ).resolve()

            try:
                path.relative_to(
                    output_root.resolve()
                )
            except ValueError as exc:
                raise RuntimeError(
                    "PHASE4_ARTIFACT_OUTSIDE_OUTPUT_ROOT:"
                    + path_field
                ) from exc

            if not path.is_file():
                raise RuntimeError(
                    "PHASE4_ARTIFACT_FILE_MISSING:"
                    + path_field
                )

            if sha256_file(path) != str(digest):
                raise RuntimeError(
                    "PHASE4_ARTIFACT_HASH_MISMATCH:"
                    + path_field
                )

    if require_completion_marker:
        record_path = record.get("_record_path")
        if not record_path:
            raise RuntimeError(
                "PHASE4_RECORD_PATH_NOT_BOUND"
            )

        job_root = Path(record_path).parent
        marker_path = (
            job_root / "completion_marker.json"
        )

        if not marker_path.is_file():
            raise RuntimeError(
                "PHASE4_COMPLETION_MARKER_MISSING"
            )

        marker = json.loads(
            marker_path.read_text()
        )

        expected_id = record.get(
            "route_id",
            record.get("job_id"),
        )

        if marker.get("route_id") != expected_id:
            raise RuntimeError(
                "PHASE4_COMPLETION_MARKER_ID_MISMATCH"
            )

        if marker.get("status") != (
            "ENGINEERING_ACCEPTED"
        ):
            raise RuntimeError(
                "PHASE4_COMPLETION_MARKER_STATUS_INVALID"
            )

        if marker.get("outer_test_used") is not False:
            raise RuntimeError(
                "PHASE4_COMPLETION_MARKER_OUTER_TEST_INVALID"
            )


def load_and_validate_phase4_record(
    *,
    record_path: Path,
    output_root: Path,
    expected_execution_fingerprint: str,
) -> dict[str, Any]:
    if not record_path.is_file():
        raise RuntimeError(
            "PHASE4_JOB_RECORD_MISSING"
        )

    record: dict[str, Any] = json.loads(
        record_path.read_text()
    )
    record["_record_path"] = str(record_path)

    if record.get("execution_fingerprint") != (
        expected_execution_fingerprint
    ):
        raise RuntimeError(
            "PHASE4_EXECUTION_FINGERPRINT_MISMATCH"
        )

    validate_phase4_job_artifacts(
        record=record,
        output_root=output_root,
    )

    record.pop("_record_path", None)
    return record
