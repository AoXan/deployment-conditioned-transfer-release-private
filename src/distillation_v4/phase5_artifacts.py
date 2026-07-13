from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any


FIELDS = (
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


def validate_phase5_record(
    *,
    record: dict[str, Any],
    output: Path,
    expected_execution_fingerprint: str,
    record_path: Path,
) -> None:
    if record.get("status") != (
        "ENGINEERING_ACCEPTED"
    ):
        raise RuntimeError(
            "PHASE5_RECORD_NOT_ACCEPTED"
        )

    if record.get("outer_test_used") is not False:
        raise RuntimeError(
            "PHASE5_RECORD_OUTER_TEST_FORBIDDEN"
        )

    if record.get(
        "execution_fingerprint"
    ) != expected_execution_fingerprint:
        raise RuntimeError(
            "PHASE5_EXECUTION_FINGERPRINT_MISMATCH"
        )

    for path_field, hash_field in FIELDS:
        value = record.get(path_field)
        digest = record.get(hash_field)

        if not value or not digest:
            raise RuntimeError(
                "PHASE5_ARTIFACT_LINEAGE_INCOMPLETE:"
                + path_field
            )

        path = (output / str(value)).resolve()

        try:
            path.relative_to(output.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "PHASE5_ARTIFACT_OUTSIDE_OUTPUT:"
                + path_field
            ) from exc

        if not path.is_file():
            raise RuntimeError(
                "PHASE5_ARTIFACT_MISSING:"
                + path_field
            )

        if sha256_file(path) != str(digest):
            raise RuntimeError(
                "PHASE5_ARTIFACT_HASH_MISMATCH:"
                + path_field
            )

    marker_path = (
        record_path.parent
        / "completion_marker.json"
    )

    if not marker_path.is_file():
        raise RuntimeError(
            "PHASE5_COMPLETION_MARKER_MISSING"
        )

    marker = json.loads(marker_path.read_text())

    if marker.get("job_id") != record.get("job_id"):
        raise RuntimeError(
            "PHASE5_COMPLETION_MARKER_ID_MISMATCH"
        )

    if marker.get("status") != (
        "ENGINEERING_ACCEPTED"
    ):
        raise RuntimeError(
            "PHASE5_COMPLETION_MARKER_STATUS_INVALID"
        )

    if marker.get("outer_test_used") is not False:
        raise RuntimeError(
            "PHASE5_COMPLETION_MARKER_OUTER_TEST_INVALID"
        )
