from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from .fingerprints import sha256_file
from .outer_readiness import (
    validate_frozen_outer_readiness_contract,
)


REPAIR_RELEASE_STATUS = (
    "REPAIR_OUTER_TEST_FORMALLY_RELEASED"
)


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def validate_formal_outer_approval(
    *,
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "FORMAL_OUTER_APPROVAL_MISSING"
        )

    payload: dict[str, Any] = json.loads(
        path.read_text()
    )

    if payload.get("status") != "FORMALLY_APPROVED":
        raise RuntimeError(
            "FORMAL_OUTER_APPROVAL_INVALID"
        )

    if payload.get("scope") != (
        "STAGE8_V4_REPAIR_OUTER_TEST_ONLY"
    ):
        raise RuntimeError(
            "FORMAL_OUTER_APPROVAL_SCOPE_INVALID"
        )

    if payload.get("approved") is not True:
        raise RuntimeError(
            "FORMAL_OUTER_APPROVAL_NOT_GRANTED"
        )

    if payload.get(
        "route_matrix_mutation_allowed"
    ) is not False:
        raise RuntimeError(
            "FORMAL_APPROVAL_ROUTE_MUTATION_FORBIDDEN"
        )

    if payload.get(
        "control_matrix_mutation_allowed"
    ) is not False:
        raise RuntimeError(
            "FORMAL_APPROVAL_CONTROL_MUTATION_FORBIDDEN"
        )

    approval_id = payload.get("approval_id")

    if not isinstance(approval_id, str) or not approval_id:
        raise RuntimeError(
            "FORMAL_OUTER_APPROVAL_ID_MISSING"
        )

    return payload


def create_repair_outer_release_token(
    *,
    output: Path,
    approval_path: Path,
    outer_release_enabled: bool,
) -> dict[str, Any]:
    if outer_release_enabled is not True:
        raise RuntimeError(
            "FORMAL_RUN_NOT_APPROVED"
        )

    readiness_path = (
        output
        / "control/frozen_outer_readiness_contract.json"
    )
    readiness = (
        validate_frozen_outer_readiness_contract(
            path=readiness_path,
            output=output,
        )
    )

    approval = validate_formal_outer_approval(
        path=approval_path,
    )

    registry_path = (
        output
        / "control/frozen_route_registry_repair.json"
    )

    if not registry_path.is_file():
        raise RuntimeError(
            "REPAIR_ROUTE_REGISTRY_MISSING"
        )

    registry: dict[str, Any] = json.loads(
        registry_path.read_text()
    )

    if registry.get("status") != (
        "REPAIR_ROUTE_REGISTRY_FROZEN"
    ):
        raise RuntimeError(
            "REPAIR_ROUTE_REGISTRY_INVALID"
        )

    if registry.get(
        "outer_test_execution_allowed"
    ) is not False:
        raise RuntimeError(
            "REPAIR_REGISTRY_PREMATURE_OUTER_AUTHORIZATION"
        )

    if registry.get(
        "outer_readiness_fingerprint"
    ) != readiness.get(
        "outer_readiness_fingerprint"
    ):
        raise RuntimeError(
            "REPAIR_REGISTRY_READINESS_FINGERPRINT_MISMATCH"
        )

    route_fingerprint = registry.get(
        "route_fingerprint_before_outer_test"
    )

    if not isinstance(route_fingerprint, str):
        raise RuntimeError(
            "REPAIR_ROUTE_FINGERPRINT_MISSING"
        )

    payload = {
        "status": REPAIR_RELEASE_STATUS,
        "release_type": "REPAIR_ONLY",
        "approval_id": approval["approval_id"],
        "approval_sha256": sha256_file(
            approval_path
        ),
        "approval_scope": approval["scope"],
        "outer_readiness_contract": (
            "control/"
            "frozen_outer_readiness_contract.json"
        ),
        "outer_readiness_sha256": sha256_file(
            readiness_path
        ),
        "outer_readiness_fingerprint": readiness[
            "outer_readiness_fingerprint"
        ],
        "repair_route_registry": (
            "control/frozen_route_registry_repair.json"
        ),
        "repair_route_registry_sha256": (
            sha256_file(registry_path)
        ),
        "route_fingerprint_before_outer_test": (
            route_fingerprint
        ),
        "outer_test_role": "FINAL_ESTIMATION_ONLY",
        "route_matrix_may_change": False,
        "control_matrix_may_change": False,
        "outer_test_may_enable_downstream_jobs": False,
        "legacy_registry_allowed": False,
        "legacy_release_allowed": False,
        "execution_authorized": True,
    }

    payload["repair_outer_release_fingerprint"] = (
        _fingerprint(payload)
    )

    token_path = (
        output
        / "control/repair_outer_release_token.json"
    )
    _atomic_json(token_path, payload)

    return payload


def validate_repair_outer_release_token(
    *,
    output: Path,
    token_path: Path,
) -> dict[str, Any]:
    if not token_path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_RELEASE_TOKEN_MISSING"
        )

    token: dict[str, Any] = json.loads(
        token_path.read_text()
    )

    if token.get("status") != REPAIR_RELEASE_STATUS:
        raise RuntimeError(
            "REPAIR_OUTER_RELEASE_TOKEN_STATUS_INVALID"
        )

    if token.get("release_type") != "REPAIR_ONLY":
        raise RuntimeError(
            "LEGACY_OUTER_RELEASE_FORBIDDEN_FOR_REPAIR"
        )

    if token.get("execution_authorized") is not True:
        raise RuntimeError(
            "REPAIR_OUTER_EXECUTION_NOT_AUTHORIZED"
        )

    stored = token.get(
        "repair_outer_release_fingerprint"
    )
    unsigned = dict(token)
    unsigned.pop(
        "repair_outer_release_fingerprint",
        None,
    )

    if stored != _fingerprint(unsigned):
        raise RuntimeError(
            "REPAIR_OUTER_RELEASE_TOKEN_FINGERPRINT_MISMATCH"
        )

    readiness_path = (
        output
        / str(token["outer_readiness_contract"])
    )
    readiness = (
        validate_frozen_outer_readiness_contract(
            path=readiness_path,
            output=output,
        )
    )

    if sha256_file(readiness_path) != token.get(
        "outer_readiness_sha256"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_READINESS_HASH_MISMATCH"
        )

    if readiness.get(
        "outer_readiness_fingerprint"
    ) != token.get(
        "outer_readiness_fingerprint"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_READINESS_FINGERPRINT_MISMATCH"
        )

    registry_path = (
        output
        / str(token["repair_route_registry"])
    )

    if not registry_path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_REGISTRY_MISSING"
        )

    if sha256_file(registry_path) != token.get(
        "repair_route_registry_sha256"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_REGISTRY_HASH_MISMATCH"
        )

    registry = json.loads(registry_path.read_text())

    if registry.get("status") != (
        "REPAIR_ROUTE_REGISTRY_FROZEN"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_REGISTRY_INVALID"
        )

    if registry.get(
        "route_fingerprint_before_outer_test"
    ) != token.get(
        "route_fingerprint_before_outer_test"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_FINGERPRINT_MISMATCH"
        )

    if token.get("legacy_registry_allowed") is not False:
        raise RuntimeError(
            "REPAIR_OUTER_LEGACY_REGISTRY_NOT_BLOCKED"
        )

    if token.get("legacy_release_allowed") is not False:
        raise RuntimeError(
            "REPAIR_OUTER_LEGACY_RELEASE_NOT_BLOCKED"
        )

    return token
