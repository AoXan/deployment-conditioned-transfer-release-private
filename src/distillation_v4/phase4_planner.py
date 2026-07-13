from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
from typing import Any, Iterable

from .contracts import CampaignConfig
from .phase4_contract import (
    validate_phase4_prerequisites,
)


@dataclass(frozen=True)
class Phase4Route:
    route_id: str
    dataset: str
    fold: str
    seed: int
    route: str
    prediction_teacher_key: (
        tuple[str, str, str] | None
    )
    representation_teacher_key: (
        tuple[str, str, str] | None
    )
    generation_scope: str
    outer_test_used: bool

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)

        for field in (
            "prediction_teacher_key",
            "representation_teacher_key",
        ):
            if value[field] is not None:
                value[field] = list(value[field])

        return value


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"PHASE4_GATE_FILE_MISSING:{path.name}"
        )
    return json.loads(path.read_text())


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ckd_balancing_contract(
    output: Path,
    dataset: str,
) -> dict[str, float] | None:
    path = (
        output
        / "control/gates"
        / f"{dataset}__ckd_balancing.json"
    )

    if not path.is_file():
        return None

    gate = json.loads(path.read_text())

    if gate.get("decision") not in {
        "DEVELOPMENT_OPEN",
        "DEVELOPMENT_CONDITIONAL_OPEN",
    }:
        return None

    if gate.get("selection_scope") != (
        "OUTER_TRAIN_INNER_VALIDATION_ONLY"
    ):
        raise RuntimeError(
            "CKD_BALANCING_SELECTION_SCOPE_INVALID"
        )

    if gate.get("outer_test_used") is not False:
        raise RuntimeError(
            "OUTER_TEST_CKD_BALANCING_FORBIDDEN"
        )

    weights = gate.get("loss_weights")
    if not isinstance(weights, dict):
        raise RuntimeError(
            "CKD_BALANCING_WEIGHTS_MISSING"
        )

    required = {
        "supervised",
        "prediction",
        "representation",
    }
    if set(weights) != required:
        raise RuntimeError(
            "CKD_BALANCING_WEIGHT_KEYS_INVALID"
        )

    normalized: dict[str, float] = {}

    for name in sorted(required):
        value = float(weights[name])

        if not (value > 0.0):
            raise RuntimeError(
                "CKD_BALANCING_WEIGHT_NONPOSITIVE:"
                + name
            )

        if not (
            value < float("inf")
        ):
            raise RuntimeError(
                "CKD_BALANCING_WEIGHT_NONFINITE:"
                + name
            )

        normalized[name] = value

    total = sum(normalized.values())
    if total <= 0.0:
        raise RuntimeError(
            "CKD_BALANCING_WEIGHT_TOTAL_INVALID"
        )

    return {
        name: value / total
        for name, value in normalized.items()
    }


def _folds(
    config: CampaignConfig,
    dataset: str,
) -> tuple[str, ...]:
    return tuple(
        config.raw
        .get("datasets", {})
        .get(dataset, {})
        .get(
            "outer_folds",
            (
                "test_2021",
                "test_2022",
                "test_2023",
            ),
        )
    )


def _student_open(
    output: Path,
    dataset: str,
) -> bool:
    gate = _json(
        output
        / "control/gates"
        / f"{dataset}__student_credibility.json"
    )
    return gate.get("decision") == (
        "STUDENT_CREDIBILITY_OPEN"
    )


def _soil_survives(
    output: Path,
    dataset: str,
) -> bool:
    gate = _json(
        output
        / "control/gates"
        / f"{dataset}__soil_repair.json"
    )
    return gate.get("decision") in {
        "DEVELOPMENT_OPEN",
        "DEVELOPMENT_CONDITIONAL_OPEN",
    }


def _missing_aware_open(
    output: Path,
    dataset: str,
) -> bool:
    gate = _json(
        output
        / "control/gates"
        / f"{dataset}__missing_aware.json"
    )
    if gate.get("outer_test_used") is not False:
        raise RuntimeError(
            "OUTER_TEST_MISSING_AWARE_GATE_FORBIDDEN"
        )
    return gate.get("decision") in {
        "DEVELOPMENT_OPEN",
        "DEVELOPMENT_CONDITIONAL_OPEN",
    }


def build_phase4_routes(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    prerequisites = (
        validate_phase4_prerequisites(output)
    )

    routes: list[Phase4Route] = []
    blocked: list[dict[str, Any]] = []

    seeds = tuple(
        int(seed)
        for seed in config.stochastic_seeds
    )

    for dataset in config.primary_datasets:
        student_open = _student_open(
            output,
            dataset,
        )
        soil_survives = _soil_survives(
            output,
            dataset,
        )
        missing_aware_open = _missing_aware_open(
            output,
            dataset,
        )
        ckd_balancing = _ckd_balancing_contract(
            output,
            dataset,
        )

        for fold in _folds(config, dataset):
            prediction_key = (
                dataset,
                fold,
                "prediction",
            )
            representation_key = (
                dataset,
                fold,
                "representation",
            )

            prediction_available = (
                prediction_key
                in prerequisites.teachers
            )
            representation_available = (
                representation_key
                in prerequisites.teachers
            )

            # REF is a frozen teacher reference.
            # It is generated once per dataset × fold and is not
            # duplicated for each downstream student seed.
            reference_key = None
            if prediction_available:
                reference_key = prediction_key
            elif representation_available:
                reference_key = representation_key

            if reference_key is not None:
                reference_route_id = "__".join(
                    (
                        dataset,
                        "phase4",
                        "reference",
                        fold,
                        "frozen_teacher",
                    )
                )

                routes.append(
                    Phase4Route(
                        route_id=reference_route_id,
                        dataset=dataset,
                        fold=fold,
                        seed=int(
                            prerequisites
                            .teachers[reference_key]["seed"]
                        ),
                        route="reference",
                        prediction_teacher_key=(
                            prediction_key
                            if prediction_available
                            else None
                        ),
                        representation_teacher_key=(
                            representation_key
                            if (
                                not prediction_available
                                and representation_available
                            )
                            else None
                        ),
                        generation_scope=(
                            "FROZEN_TEACHER_REFERENCE_ONLY"
                        ),
                        outer_test_used=False,
                    )
                )
            else:
                blocked.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": None,
                        "route": "reference",
                        "generated": False,
                        "reason": (
                            "NO_FROZEN_TEACHER_AVAILABLE"
                        ),
                    }
                )

            for seed in seeds:
                candidate_routes = {
                    "supervised": {
                        "generated": student_open,
                        "reason": (
                            "STUDENT_CREDIBILITY_GATE_NOT_OPEN"
                        ),
                    },
                    "soil_direct": {
                        "generated": (
                            student_open
                            and soil_survives
                        ),
                        "reason": (
                            "SOIL_GATE_NOT_OPEN"
                        ),
                    },
                    "missing_aware": {
                        "generated": (
                            student_open
                            and missing_aware_open
                        ),
                        "reason": (
                            "MISSING_AWARE_GATE_NOT_OPEN"
                        ),
                    },
                    "fine_tune": {
                        "generated": (
                            student_open
                            and soil_survives
                            and representation_available
                        ),
                        "reason": (
                            "LEGAL_MULTIMODAL_PRETRAINING_"
                            "REPRESENTATION_NOT_AVAILABLE"
                        ),
                    },
                    "prediction_kd": {
                        "generated": (
                            student_open
                            and prediction_available
                        ),
                        "reason": (
                            "PREDICTION_TEACHER_NOT_AVAILABLE"
                        ),
                    },
                    "representation_kd": {
                        "generated": (
                            student_open
                            and representation_available
                        ),
                        "reason": (
                            "REPRESENTATION_TEACHER_NOT_AVAILABLE"
                        ),
                    },
                    "combined_kd": {
                        "generated": (
                            student_open
                            and prediction_available
                            and representation_available
                            and ckd_balancing is not None
                        ),
                        "reason": (
                            "CKD_PREREQUISITE_OR_BALANCING_"
                            "CONTRACT_NOT_OPEN"
                        ),
                    },
                }

                for route, decision in (
                    candidate_routes.items()
                ):
                    if not decision["generated"]:
                        blocked.append(
                            {
                                "dataset": dataset,
                                "fold": fold,
                                "seed": seed,
                                "route": route,
                                "generated": False,
                                "reason": decision["reason"],
                            }
                        )
                        continue

                    route_id = "__".join(
                        (
                            dataset,
                            "phase4",
                            route,
                            fold,
                            f"seed_{seed}",
                        )
                    )

                    routes.append(
                        Phase4Route(
                            route_id=route_id,
                            dataset=dataset,
                            fold=fold,
                            seed=seed,
                            route=route,
                            prediction_teacher_key=(
                                prediction_key
                                if route in {
                                    "prediction_kd",
                                    "combined_kd",
                                }
                                else None
                            ),
                            representation_teacher_key=(
                                representation_key
                                if route in {
                                    "fine_tune",
                                    "representation_kd",
                                    "combined_kd",
                                }
                                else None
                            ),
                            generation_scope=(
                                "OUTER_TRAIN_"
                                "INNER_VALIDATION_ONLY"
                            ),
                            outer_test_used=False,
                        )
                    )

    route_mappings = [
        route.to_mapping()
        for route in routes
    ]

    soil_contract = config.raw.get(
        "soil_missing_aware",
        {
            "hidden": 32,
            "representation_dim": 32,
            "primary_soil_representation": (
                "raw"
            ),
            "compact_components": 8,
            "primary_dropout_probability": (
                0.5
            ),
            "explicit_availability_mask": (
                True
            ),
        },
    )

    for mapping in route_mappings:
        route_name = str(
            mapping["route"]
        )

        if route_name not in {
            "soil_direct",
            "missing_aware",
        }:
            continue

        mapping.update(
            {
                "soil_representation": str(
                    soil_contract[
                        "primary_soil_representation"
                    ]
                ),
                "soil_compact_components": int(
                    soil_contract[
                        "compact_components"
                    ]
                ),
                "soil_hidden": int(
                    soil_contract[
                        "hidden"
                    ]
                ),
                "soil_representation_dim": int(
                    soil_contract[
                        "representation_dim"
                    ]
                ),
                "soil_dropout_probability": (
                    0.0
                    if route_name
                    == "soil_direct"
                    else float(
                        soil_contract[
                            "primary_dropout_probability"
                        ]
                    )
                ),
                "explicit_availability_mask": bool(
                    soil_contract[
                        "explicit_availability_mask"
                    ]
                ),
                "mask_granularity": (
                    "WHOLE_SOIL_MODALITY"
                ),
                "weather_modality_dropped": False,
                "parameter_matching_group": (
                    "SOIL_DIRECT_MISSING_AWARE_V1"
                ),
            }
        )

    ckd_balancing_by_dataset = {
        dataset: _ckd_balancing_contract(
            output,
            dataset,
        )
        for dataset in config.primary_datasets
    }

    generated_by_dataset = {
        dataset: sum(
            1
            for route in routes
            if route.dataset == dataset
        )
        for dataset in config.primary_datasets
    }

    per_dataset_ceiling = int(
        config.budgets["phase4"]
    )
    campaign_ceiling = (
        per_dataset_ceiling
        * len(config.primary_datasets)
    )

    exceeded = {
        dataset: count
        for dataset, count
        in generated_by_dataset.items()
        if count > per_dataset_ceiling
    }
    if exceeded:
        raise RuntimeError(
            "PHASE4_PER_DATASET_CEILING_EXCEEDED:"
            + json.dumps(
                exceeded,
                sort_keys=True,
            )
        )

    reserve_by_dataset = {
        dataset: (
            per_dataset_ceiling
            - generated_by_dataset[dataset]
        )
        for dataset in config.primary_datasets
    }

    payload = {
        "phase": "phase4",
        "status": "ROUTE_MANIFEST_FROZEN",
        "routes": route_mappings,
        "generated_route_count": len(routes),
        "budget_semantics": (
            "PER_DATASET_MAXIMUM_NOT_REQUIRED_COUNT"
        ),
        "per_dataset_job_ceiling": (
            per_dataset_ceiling
        ),
        "campaign_job_ceiling": campaign_ceiling,
        "generated_jobs_by_dataset": (
            generated_by_dataset
        ),
        "unused_ceiling_by_dataset": (
            reserve_by_dataset
        ),
        "core_route_families": [
            "supervised",
            "soil_direct",
            "missing_aware",
            "fine_tune",
            "prediction_kd",
            "representation_kd",
            "combined_kd",
            "reference",
        ],
        "student_route_seed_policy": (
            "THREE_STOCHASTIC_SEEDS_PER_FOLD"
        ),
        "reference_seed_policy": (
            "ONE_FROZEN_SELECTED_TEACHER_PER_FOLD"
        ),
        "ckd_balancing_by_dataset": (
            ckd_balancing_by_dataset
        ),
        "conditional_reserve_policy": (
            "UNMATERIALIZED_UNLESS_A_PREDECLARED_"
            "DEVELOPMENT_CONTROL_GATE_OPENS"
        ),
        "blocked_candidate_count": len(blocked),
        "blocked_candidates": blocked,
        "outer_test_used": False,
        "teacher_registry_release_status": (
            prerequisites
            .teacher_registry["release_status"]
        ),
        "teacher_registry_outer_refit_allowed": (
            prerequisites
            .teacher_registry[
                "outer_refit_allowed"
            ]
        ),
    }

    required_manifest_fields = {
        "budget_semantics",
        "per_dataset_job_ceiling",
        "campaign_job_ceiling",
        "generated_jobs_by_dataset",
        "unused_ceiling_by_dataset",
        "core_route_families",
        "conditional_reserve_policy",
    }
    missing_manifest_fields = sorted(
        required_manifest_fields.difference(payload)
    )
    if missing_manifest_fields:
        raise RuntimeError(
            "PHASE4_MANIFEST_FIELDS_MISSING:"
            + ",".join(missing_manifest_fields)
        )

    required_manifest_fields = {
        "budget_semantics",
        "per_dataset_job_ceiling",
        "campaign_job_ceiling",
        "generated_jobs_by_dataset",
        "unused_ceiling_by_dataset",
        "core_route_families",
        "conditional_reserve_policy",
    }
    missing_manifest_fields = sorted(
        required_manifest_fields.difference(payload)
    )
    if missing_manifest_fields:
        raise RuntimeError(
            "PHASE4_MANIFEST_FIELDS_MISSING:"
            + ",".join(missing_manifest_fields)
        )

    payload[
        "phase4_route_manifest_fingerprint"
    ] = _fingerprint(payload)

    return payload
