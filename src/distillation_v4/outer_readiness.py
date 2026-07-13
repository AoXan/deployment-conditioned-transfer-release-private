from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any


REQUIRED_FROZEN_FILES = (
    "control/frozen_phase4_route_manifest.json",
    "control/frozen_phase5_control_requirements.json",
    "control/frozen_phase5_job_matrix.json",
    "control/frozen_teacher_registry.json",
    "development/phase4_aggregation.json",
    "development/phase5_aggregation.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "OUTER_READINESS_FILE_MISSING:"
            + str(path)
        )

    return json.loads(path.read_text())


def build_outer_readiness_contract(
    *,
    output: Path,
) -> dict[str, Any]:
    phase3 = _json(
        output / "status/phase3_repair.json"
    )
    phase4 = _json(
        output / "status/phase4_repair.json"
    )
    phase5 = _json(
        output / "status/phase5_repair.json"
    )

    for name, payload in (
        ("phase3", phase3),
        ("phase4", phase4),
        ("phase5", phase5),
    ):
        if payload.get("status") != (
            "SCIENTIFIC_PHASE_COMPLETE"
        ):
            raise RuntimeError(
                "OUTER_READINESS_PHASE_INCOMPLETE:"
                + name
            )

        if payload.get("outer_test_used") is not False:
            raise RuntimeError(
                "OUTER_READINESS_DEVELOPMENT_OUTER_USE:"
                + name
            )

    frozen_files: list[dict[str, str]] = []

    for relative in REQUIRED_FROZEN_FILES:
        path = output / relative

        if not path.is_file():
            raise RuntimeError(
                "OUTER_READINESS_FROZEN_FILE_MISSING:"
                + relative
            )

        frozen_files.append(
            {
                "path": relative,
                "sha256": _sha256(path),
            }
        )

    route_manifest = _json(
        output
        / "control/frozen_phase4_route_manifest.json"
    )
    phase5_matrix = _json(
        output
        / "control/frozen_phase5_job_matrix.json"
    )
    phase4_aggregation = _json(
        output
        / "development/phase4_aggregation.json"
    )
    phase5_aggregation = _json(
        output
        / "development/phase5_aggregation.json"
    )

    route_fingerprint = route_manifest.get(
        "phase4_route_manifest_fingerprint"
    )

    if phase4_aggregation.get(
        "route_manifest_fingerprint"
    ) != route_fingerprint:
        raise RuntimeError(
            "OUTER_READINESS_PHASE4_FINGERPRINT_MISMATCH"
        )

    phase5_fingerprint = phase5_matrix.get(
        "phase5_job_matrix_fingerprint"
    )

    if phase5_aggregation.get(
        "phase5_job_matrix_fingerprint"
    ) != phase5_fingerprint:
        raise RuntimeError(
            "OUTER_READINESS_PHASE5_FINGERPRINT_MISMATCH"
        )

    payload = {
        "status": "OUTER_READINESS_CONTRACT_FROZEN",
        "development_phases_complete": [
            "phase3",
            "phase4",
            "phase5",
        ],
        "frozen_files": frozen_files,
        "phase4_route_manifest_fingerprint": (
            route_fingerprint
        ),
        "phase5_job_matrix_fingerprint": (
            phase5_fingerprint
        ),
        "development_outer_test_used": False,
        "route_matrix_may_change_after_release": False,
        "control_matrix_may_change_after_release": False,
        "outer_test_role": "FINAL_ESTIMATION_ONLY",
        "outer_test_may_enable_downstream_jobs": False,
        "outer_test_execution_allowed": False,
        "formal_approval_required": True,
        "claim_boundary": (
            "READINESS_CONTRACT_DOES_NOT_AUTHORIZE_EXECUTION"
        ),
    }

    unsigned = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    payload["outer_readiness_fingerprint"] = (
        hashlib.sha256(unsigned).hexdigest()
    )

    return payload


def freeze_outer_readiness_contract(
    *,
    output: Path,
) -> dict[str, Any]:
    payload = build_outer_readiness_contract(
        output=output,
    )

    path = (
        output
        / "control/frozen_outer_readiness_contract.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)

    return payload


def validate_frozen_outer_readiness_contract(
    *,
    path: Path,
    output: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "FROZEN_OUTER_READINESS_CONTRACT_MISSING"
        )

    payload: dict[str, Any] = json.loads(
        path.read_text()
    )

    if payload.get("status") != (
        "OUTER_READINESS_CONTRACT_FROZEN"
    ):
        raise RuntimeError(
            "OUTER_READINESS_CONTRACT_INVALID"
        )

    if payload.get(
        "outer_test_execution_allowed"
    ) is not False:
        raise RuntimeError(
            "OUTER_READINESS_PREMATURE_EXECUTION_AUTHORIZATION"
        )

    if payload.get(
        "formal_approval_required"
    ) is not True:
        raise RuntimeError(
            "OUTER_READINESS_FORMAL_APPROVAL_NOT_REQUIRED"
        )

    if payload.get(
        "development_outer_test_used"
    ) is not False:
        raise RuntimeError(
            "OUTER_READINESS_DEVELOPMENT_OUTER_USE"
        )

    stored = payload.get(
        "outer_readiness_fingerprint"
    )

    if not isinstance(stored, str) or not stored:
        raise RuntimeError(
            "OUTER_READINESS_FINGERPRINT_MISSING"
        )

    unsigned = dict(payload)
    unsigned.pop(
        "outer_readiness_fingerprint",
        None,
    )

    actual = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    if actual != stored:
        raise RuntimeError(
            "OUTER_READINESS_FINGERPRINT_MISMATCH"
        )

    frozen_files = payload.get("frozen_files")

    if not isinstance(frozen_files, list):
        raise RuntimeError(
            "OUTER_READINESS_FROZEN_FILES_INVALID"
        )

    for record in frozen_files:
        if not isinstance(record, dict):
            raise RuntimeError(
                "OUTER_READINESS_FROZEN_FILE_RECORD_INVALID"
            )

        relative = record.get("path")
        expected_hash = record.get("sha256")

        if not relative or not expected_hash:
            raise RuntimeError(
                "OUTER_READINESS_FROZEN_FILE_LINEAGE_INCOMPLETE"
            )

        frozen_path = (
            output / str(relative)
        ).resolve()

        try:
            frozen_path.relative_to(output.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "OUTER_READINESS_FROZEN_FILE_OUTSIDE_OUTPUT"
            ) from exc

        if not frozen_path.is_file():
            raise RuntimeError(
                "OUTER_READINESS_FROZEN_FILE_MISSING:"
                + str(relative)
            )

        if _sha256(frozen_path) != str(
            expected_hash
        ):
            raise RuntimeError(
                "OUTER_READINESS_FROZEN_FILE_HASH_MISMATCH:"
                + str(relative)
            )

    return payload
