from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Callable

import pandas as pd

from .contracts import CampaignConfig
from .phase4_contract import (
    validate_phase4_prerequisites,
)
from .phase4_execution import (
    execute_phase4_route,
)
from .phase4_artifacts import (
    load_and_validate_phase4_record,
)
from .phase4_fingerprint import (
    dataframe_fingerprint,
    phase4_execution_fingerprint,
)


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


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def run_phase4_repair_matrix(
    config: CampaignConfig,
    output: Path,
    *,
    frame_loader: Callable[
        [CampaignConfig, Path, str],
        pd.DataFrame,
    ],
) -> dict[str, Any]:
    manifest_path = (
        output
        / "control/frozen_phase4_route_manifest.json"
    )

    if not manifest_path.is_file():
        raise RuntimeError(
            "PHASE4_FROZEN_ROUTE_MANIFEST_MISSING"
        )

    manifest = json.loads(
        manifest_path.read_text()
    )

    if manifest.get("status") != (
        "ROUTE_MANIFEST_FROZEN"
    ):
        raise RuntimeError(
            "PHASE4_ROUTE_MANIFEST_NOT_FROZEN"
        )

    if manifest.get("outer_test_used") is not False:
        raise RuntimeError(
            "OUTER_TEST_PHASE4_EXECUTION_FORBIDDEN"
        )

    stored_fingerprint = manifest.get(
        "phase4_route_manifest_fingerprint"
    )
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop(
        "phase4_route_manifest_fingerprint",
        None,
    )

    actual_fingerprint = _fingerprint(
        fingerprint_payload
    )

    if stored_fingerprint != actual_fingerprint:
        raise RuntimeError(
            "PHASE4_ROUTE_MANIFEST_FINGERPRINT_MISMATCH"
        )

    prerequisites = (
        validate_phase4_prerequisites(output)
    )

    routes = manifest.get("routes")
    if not isinstance(routes, list):
        raise RuntimeError(
            "PHASE4_ROUTE_MANIFEST_ROUTES_INVALID"
        )

    route_ids = [
        str(route["route_id"])
        for route in routes
    ]
    if len(set(route_ids)) != len(route_ids):
        raise RuntimeError(
            "PHASE4_ROUTE_ID_DUPLICATE"
        )

    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    execution_fingerprints: set[str] = set()

    frames: dict[str, pd.DataFrame] = {}
    frame_fingerprints: dict[str, str] = {}

    training_contract_mapping = dict(
        config.raw.get("training", {})
    )
    code_contract = (
        "stage8_v4_phase4_execution_v2"
    )

    for route_record in routes:
        dataset = str(route_record["dataset"])

        if dataset not in frames:
            frames[dataset] = frame_loader(
                config,
                output,
                dataset,
            )
            frame_fingerprints[dataset] = (
                dataframe_fingerprint(
                    frames[dataset]
                )
            )

        execution_fingerprint = (
            phase4_execution_fingerprint(
                route_record=route_record,
                training_contract=(
                    training_contract_mapping
                ),
                frame_fingerprint=(
                    frame_fingerprints[dataset]
                ),
                route_manifest_fingerprint=(
                    stored_fingerprint
                ),
                code_contract=code_contract,
            )
        )

        if execution_fingerprint in (
            execution_fingerprints
        ):
            raise RuntimeError(
                "PHASE4_EXECUTION_FINGERPRINT_DUPLICATE"
            )

        execution_fingerprints.add(
            execution_fingerprint
        )

        job_output = (
            output
            / "development/phase4_repair_matrix"
            / dataset
            / str(route_record["route_id"])
        )

        existing_record_path = (
            job_output / "job_record.json"
        )

        if existing_record_path.is_file():
            try:
                record = (
                    load_and_validate_phase4_record(
                        record_path=(
                            existing_record_path
                        ),
                        output_root=output,
                        expected_execution_fingerprint=(
                            execution_fingerprint
                        ),
                    )
                )
                record["resume_status"] = (
                    "REUSED_VALIDATED_ARTIFACT"
                )
                accepted.append(record)
                continue
            except RuntimeError as exc:
                failure = {
                    "route_id": route_record.get(
                        "route_id"
                    ),
                    "dataset": dataset,
                    "fold": route_record.get("fold"),
                    "seed": route_record.get("seed"),
                    "route": route_record.get("route"),
                    "status": (
                        "BLOCKED_EXISTING_ARTIFACT_INVALID"
                    ),
                    "failure_type": type(exc).__name__,
                    "reason": str(exc),
                    "outer_test_used": False,
                }
                failures.append(failure)
                continue

        try:
            record = execute_phase4_route(
                route_record=route_record,
                frame=frames[dataset],
                output_root=output,
                output=job_output,
                teacher_lookup=(
                    prerequisites.teachers
                ),
                contract=(
                    config.neural_training
                ),
                ckd_weights=(
                    manifest
                    .get(
                        "ckd_balancing_by_dataset",
                        {},
                    )
                    .get(dataset)
                ),
            )
            record["execution_fingerprint"] = (
                execution_fingerprint
            )

            _atomic_json(
                job_output / "job_record.json",
                record,
            )
            accepted.append(record)

        except Exception as exc:
            failure = {
                "route_id": route_record.get(
                    "route_id"
                ),
                "dataset": dataset,
                "fold": route_record.get("fold"),
                "seed": route_record.get("seed"),
                "route": route_record.get("route"),
                "status": "FAILED_RETRYABLE",
                "failure_type": type(exc).__name__,
                "reason": str(exc),
                "outer_test_used": False,
            }
            failures.append(failure)

            ledger = (
                output
                / "failures/phase4_repair_matrix.jsonl"
            )
            ledger.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with ledger.open("a") as handle:
                handle.write(
                    json.dumps(
                        failure,
                        sort_keys=True,
                    )
                    + "\n"
                )

    planned_jobs = len(routes)
    accepted_jobs = len(accepted)
    failed_jobs = len(failures)

    engineering_complete = (
        accepted_jobs == planned_jobs
        and failed_jobs == 0
    )

    execution_status = {
        "phase": "phase4",
        "status": (
            "ENGINEERING_MATRIX_COMPLETE"
            if engineering_complete
            else "ENGINEERING_MATRIX_INCOMPLETE"
        ),
        "planned_jobs": planned_jobs,
        "accepted_jobs": accepted_jobs,
        "failed_jobs": failed_jobs,
        "configured_campaign_ceiling": (
            manifest["campaign_job_ceiling"]
        ),
        "budget_semantics": (
            manifest["budget_semantics"]
        ),
        "unique_execution_fingerprints": len(
            execution_fingerprints
        ),
        "route_manifest_fingerprint": (
            stored_fingerprint
        ),
        "outer_test_used": False,
        "scientific_phase_complete": False,
        "phase5_release_allowed": False,
        "outer_release_allowed": False,
    }

    _atomic_json(
        output
        / "status/phase4_repair_matrix_execution.json",
        execution_status,
    )

    _atomic_json(
        output / "status/phase4_repair.json",
        {
            **execution_status,
            "status": "SCIENTIFIC_PHASE_INCOMPLETE",
            "reason": (
                "ENGINEERING_EXECUTION_REQUIRES_"
                "MATCHED_ROUTE_AGGREGATION_AND_CONTROLS"
                if engineering_complete
                else "ENGINEERING_MATRIX_INCOMPLETE"
            ),
        },
    )

    return {
        "execution_status": execution_status,
        "accepted_records": accepted,
        "failures": failures,
    }
