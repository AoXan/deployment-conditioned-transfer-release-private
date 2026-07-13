from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from .contracts import CampaignConfig
from .fingerprints import sha256_file
from .repair_outer_release import (
    validate_repair_outer_release_token,
)


OUTER_FOLDS = (
    "test_2021",
    "test_2022",
    "test_2023",
)


def _fingerprint(payload: Any) -> str:
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


def build_repair_outer_job_matrix(
    *,
    config: CampaignConfig,
    output: Path,
    token_path: Path,
) -> dict[str, Any]:
    token = validate_repair_outer_release_token(
        output=output,
        token_path=token_path,
    )

    registry_path = (
        output
        / "control/frozen_route_registry_repair.json"
    )
    registry = json.loads(
        registry_path.read_text()
    )

    seeds = tuple(
        int(seed)
        for seed in config.stochastic_seeds
    )

    if not seeds:
        raise RuntimeError(
            "REPAIR_OUTER_STOCHASTIC_SEEDS_EMPTY"
        )

    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            "REPAIR_OUTER_STOCHASTIC_SEEDS_DUPLICATE"
        )

    jobs: list[dict[str, Any]] = []

    for route in registry["jobs"]:
        dataset = str(route["dataset"])
        route_name = str(route["route"])

        for fold in OUTER_FOLDS:
            for seed in seeds:
                job_id = "__".join(
                    (
                        dataset,
                        "repair_outer",
                        route_name,
                        fold,
                        f"seed_{seed}",
                    )
                )

                jobs.append(
                    {
                        "job_id": job_id,
                        "dataset": dataset,
                        "route": route_name,
                        "fold": fold,
                        "outer_test_year": int(
                            fold.removeprefix(
                                "test_"
                            )
                        ),
                        "seed": seed,
                        "route_contract": route,
                        "route_fingerprint_before_outer_test": (
                            registry[
                                "route_fingerprint_before_outer_test"
                            ]
                        ),
                        "repair_outer_release_fingerprint": (
                            token[
                                "repair_outer_release_fingerprint"
                            ]
                        ),
                        "selection_scope": (
                            "FROZEN_BEFORE_OUTER_TEST"
                        ),
                        "outer_test_role": (
                            "FINAL_ESTIMATION_ONLY"
                        ),
                        "may_enable_downstream_jobs": False,
                    }
                )

    identifiers = [
        job["job_id"]
        for job in jobs
    ]

    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(
            "REPAIR_OUTER_JOB_ID_DUPLICATE"
        )

    payload = {
        "status": (
            "REPAIR_OUTER_JOB_MATRIX_FROZEN"
        ),
        "jobs": jobs,
        "job_count": len(jobs),
        "folds": list(OUTER_FOLDS),
        "seeds": list(seeds),
        "source_registry": (
            "control/"
            "frozen_route_registry_repair.json"
        ),
        "source_registry_sha256": (
            sha256_file(registry_path)
        ),
        "route_fingerprint_before_outer_test": (
            registry[
                "route_fingerprint_before_outer_test"
            ]
        ),
        "release_token": (
            "control/"
            "repair_outer_release_token.json"
        ),
        "release_token_sha256": (
            sha256_file(token_path)
        ),
        "repair_outer_release_fingerprint": (
            token[
                "repair_outer_release_fingerprint"
            ]
        ),
        "matrix_generated_from_outer_results": False,
        "outer_test_used_during_generation": False,
        "execution_started": False,
    }

    payload[
        "repair_outer_job_matrix_fingerprint"
    ] = _fingerprint(payload)

    return payload


def freeze_repair_outer_job_matrix(
    *,
    config: CampaignConfig,
    output: Path,
    token_path: Path,
) -> dict[str, Any]:
    payload = build_repair_outer_job_matrix(
        config=config,
        output=output,
        token_path=token_path,
    )

    path = (
        output
        / "control/"
        "frozen_repair_outer_job_matrix.json"
    )
    _atomic_json(path, payload)

    return payload


def validate_frozen_repair_outer_job_matrix(
    *,
    output: Path,
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_JOB_MATRIX_MISSING"
        )

    payload: dict[str, Any] = json.loads(
        path.read_text()
    )

    if payload.get("status") != (
        "REPAIR_OUTER_JOB_MATRIX_FROZEN"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_JOB_MATRIX_INVALID"
        )

    stored = payload.get(
        "repair_outer_job_matrix_fingerprint"
    )
    unsigned = dict(payload)
    unsigned.pop(
        "repair_outer_job_matrix_fingerprint",
        None,
    )

    if stored != _fingerprint(unsigned):
        raise RuntimeError(
            "REPAIR_OUTER_JOB_MATRIX_"
            "FINGERPRINT_MISMATCH"
        )

    registry_path = (
        output
        / str(payload["source_registry"])
    )

    if not registry_path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_MATRIX_REGISTRY_MISSING"
        )

    if sha256_file(registry_path) != payload.get(
        "source_registry_sha256"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_MATRIX_REGISTRY_HASH_MISMATCH"
        )

    token_path = (
        output
        / str(payload["release_token"])
    )

    token = validate_repair_outer_release_token(
        output=output,
        token_path=token_path,
    )

    if sha256_file(token_path) != payload.get(
        "release_token_sha256"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_MATRIX_TOKEN_HASH_MISMATCH"
        )

    if token.get(
        "repair_outer_release_fingerprint"
    ) != payload.get(
        "repair_outer_release_fingerprint"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_MATRIX_RELEASE_"
            "FINGERPRINT_MISMATCH"
        )

    jobs = payload.get("jobs")

    if not isinstance(jobs, list):
        raise RuntimeError(
            "REPAIR_OUTER_MATRIX_JOBS_INVALID"
        )

    identifiers = [
        str(job["job_id"])
        for job in jobs
    ]

    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(
            "REPAIR_OUTER_MATRIX_JOB_ID_DUPLICATE"
        )

    if payload.get(
        "outer_test_used_during_generation"
    ) is not False:
        raise RuntimeError(
            "REPAIR_OUTER_MATRIX_USED_OUTER_RESULTS"
        )

    return payload
