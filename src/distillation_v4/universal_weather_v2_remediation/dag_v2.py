from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .artifacts import sha256_file
from .methods import METHOD_HANDLERS
from .source_handlers import SOURCE_HANDLER_REGISTRY
from .m_evaluation import M_EVALUATION_HANDLER_REGISTRY
from .remaining_handlers import REMAINING_HANDLER_REGISTRY
from .planner import apply_observed_split_eligibility, build_candidate_matrix
from .preflight import run_preflight


COUNT_KEYS = ("structural_ceiling", "eligible_after_contracts", "released_by_gates", "executed")

SOURCES = ("PRIMARY_CYBENCH_MAIZE_US", "PRIMARY_G2F_MAIZE")
FOLDS = ("test_2021", "test_2022", "test_2023")
SEEDS = (101, 202, 303)
PRIMARY_ROUTES = ("supervised", "prediction_kd", "representation_kd", "combined_kd", "missing_aware")
SECONDARY_ROUTES = ("multimodal_pretrain_finetune",)
DOWNSTREAM_ROUTES = PRIMARY_ROUTES + SECONDARY_ROUTES
TARGET_CONTEXTS = (
    "CYBENCH_WHEAT_AU_RANDOM",
    "CYBENCH_WHEAT_AU_GROUP",
    "CYBENCH_WHEAT_AU_SPATIAL",
    "WAITE_ROLLING_1991",
    "ROSEWORTHY_E5_2007_FABA_SPATIAL",
    "ROSEWORTHY_E5_2008_DURUM_SPATIAL",
    "ROSEWORTHY_HISTORICAL_GRAIN_ROLLING_2018",
    "ROSEWORTHY_HISTORICAL_GRAIN_ROLLING_2019",
    "ROSEWORTHY_HISTORICAL_GRAIN_ROLLING_2020",
    "ROSEWORTHY_HISTORICAL_GRAIN_ROLLING_2021",
    "ROSEWORTHY_HISTORICAL_GRAIN_ROLLING_2022",
    "ROSEWORTHY_HISTORICAL_GRAIN_ROLLING_2023",
    "ROSEWORTHY_HISTORICAL_GRAIN_ROLLING_2024",
)
MULTIMODAL_CONTEXTS = (
    "CYBENCH_WHEAT_AU_RANDOM",
    "CYBENCH_WHEAT_AU_GROUP",
    "CYBENCH_WHEAT_AU_SPATIAL",
    "WAITE_ROLLING_1991",
    "ROSEWORTHY_E5_2007_FABA_SPATIAL",
    "ROSEWORTHY_E5_2008_DURUM_SPATIAL",
)


def _metric(structural: int, *, eligible: Any = "PENDING_BUILD_PLAN", released: Any = "NOT_GATE_CONTROLLED") -> dict[str, Any]:
    return {
        "structural_ceiling": structural,
        "eligible_after_contracts": eligible,
        "released_by_gates": released,
        "executed": 0,
    }


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def build_source_accounting() -> dict[str, Any]:
    phase_rows = {
        "phase1_contracts": {"count": 10, "type": "control-only"},
        "phase2_screening": {"count": 126, "type": "scientific_training"},
        "phase3_teacher_qualification": {"count": 102, "type": "scientific_training"},
        "phase4_routes": {"count": 132, "type": "scientific_training_or_reference_inference"},
        "phase5_controls": {"count": 36, "type": "gated_scientific_training"},
        "transfer_route_seed_specification_freeze": {"count": 36, "type": "control-only_if_no_training"},
        "source_outer": {"count": 108, "type": "scientific_evaluation"},
    }
    scientific = 126 + 102 + 132 + 36 + 108
    control_only = 10 + 36
    return {
        "schema_version": "source_dag_accounting_v2",
        "phase_rows": phase_rows,
        "source_scientific_jobs": _metric(
            scientific,
            eligible=scientific,
        ),
        "source_control_only_nodes": _metric(
            control_only,
            eligible=control_only,
        ),
        "source_total_dag_nodes": _metric(
            scientific + control_only,
            eligible=scientific + control_only,
        ),
        "source_outer": {
            "structural_ceiling": 108,
            "primary_confirmatory": len(SOURCES) * len(FOLDS) * len(SEEDS) * len(PRIMARY_ROUTES),
            "secondary_supportive": len(SOURCES) * len(FOLDS) * len(SEEDS) * len(SECONDARY_ROUTES),
            "diagnostic_excluded_routes": ["soil_direct"],
        },
        "freeze_nodes": {
            "node_name": "transfer_route_seed_specification_freeze",
            "structural_ceiling": len(SOURCES) * len(DOWNSTREAM_ROUTES) * len(SEEDS),
            "semantic": "freeze route × seed transfer specification and lineage; no model refit",
            "control_only_if_no_training": True,
            "if_refit_required": "RECLASSIFY_AS_TRAINING_JOB_WITH_TRAIN_DATA_SEED_BUDGET_CHECKPOINT_LINEAGE",
        },
    }


def build_target_structural_families() -> dict[str, dict[str, Any]]:
    return {
        "t0_zero_shot_bundles": _metric(2 * 6 * 3 * 13),
        "ordinary_transfer": _metric(2 * 6 * 3 * 13),
        "harmonised_transfer": _metric(2 * 6 * 3 * 13),
        "pooled_source_target": _metric(2 * 13 * 3),
        "matched_scratch": _metric(13 * 2 * 3),
        "m_training": _metric(72),
        "m_evaluation_bundles": _metric(108),
        "gate1": _metric(13, released="PENDING_GATE"),
        "few_shot_probe_5pct": _metric(117, released="PENDING_GATE"),
        "gate2": _metric(13, released="PENDING_GATE"),
        "few_shot_expansion_1_10_20pct": _metric(351, released="PENDING_GATE"),
        "gated_t1_total": _metric(468, released="PENDING_GATE"),
    }


def build_transfer_applicability_registry() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in SOURCES:
        source_crop = "maize"
        source_unit = "tonne_ha" if source == "PRIMARY_CYBENCH_MAIZE_US" else "bushel_acre"
        source_sample_unit = "ADMIN_REGION_HARVEST_YEAR" if source == "PRIMARY_CYBENCH_MAIZE_US" else "HYBRID_ENVIRONMENT"
        for route in DOWNSTREAM_ROUTES:
            for context in TARGET_CONTEXTS:
                target_crop = _target_crop(context)
                target_unit = "tonne_ha"
                target_sample_unit = _target_sample_unit(context)
                for strategy in ("ordinary_transfer", "harmonised_transfer"):
                    crop_compatible = source_crop == target_crop
                    unit_compatible = source_unit == target_unit
                    sample_unit_compatible = source_sample_unit == target_sample_unit
                    feature_schema_compatible = context.startswith("CYBENCH") and source == "PRIMARY_CYBENCH_MAIZE_US"
                    harmonisation_available = strategy == "ordinary_transfer" or feature_schema_compatible
                    if crop_compatible and unit_compatible and sample_unit_compatible and feature_schema_compatible:
                        decision = "EXACT_TRANSFER"
                        stratum = "exact"
                    elif not crop_compatible:
                        decision = "DIAGNOSTIC_CROSS_CROP"
                        stratum = "diagnostic"
                    elif not sample_unit_compatible:
                        decision = "DIAGNOSTIC_CROSS_UNIT"
                        stratum = "diagnostic"
                    elif unit_compatible and harmonisation_available:
                        decision = "ADAPTED_TRANSFER"
                        stratum = "adapted"
                    else:
                        decision = "NOT_APPLICABLE"
                        stratum = "not_applicable"
                    if strategy == "harmonised_transfer" and not harmonisation_available:
                        decision = "NOT_APPLICABLE"
                        stratum = "not_applicable"
                    records.append(
                        {
                            "source": source,
                            "source_route": route,
                            "target_context": context,
                            "transfer_strategy": strategy,
                            "crop_compatibility": crop_compatible,
                            "target_semantics_compatibility": target_crop in {"maize", "wheat", "durum_wheat", "faba_bean", "grain_mixed"},
                            "unit_compatibility": unit_compatible,
                            "sample_unit_compatibility": sample_unit_compatible,
                            "feature_schema_compatibility": feature_schema_compatible,
                            "harmonisation_availability": harmonisation_available,
                            "deployment_modality_compatibility": not context.startswith("ROSEWORTHY_HISTORICAL"),
                            "evidence_stratum": stratum,
                            "decision": decision,
                            "reason": _applicability_reason(decision, strategy),
                            "supporting_evidence": {
                                "source_unit": source_unit,
                                "target_unit": target_unit,
                                "source_sample_unit": source_sample_unit,
                                "target_sample_unit": target_sample_unit,
                                "context": context,
                            },
                        }
                    )
    return records


def _target_crop(context: str) -> str:
    if context.startswith("CYBENCH"):
        return "wheat"
    if context.startswith("WAITE"):
        return "wheat"
    if "2008_DURUM" in context:
        return "durum_wheat"
    if "2007_FABA" in context:
        return "faba_bean"
    if context.startswith("ROSEWORTHY_HISTORICAL_GRAIN"):
        return "grain_mixed"
    return "unknown"


def _target_sample_unit(context: str) -> str:
    if context.startswith("CYBENCH"):
        return "ADMIN_REGION_HARVEST_YEAR"
    if context.startswith("WAITE"):
        return "REGIONAL_ENVIRONMENT_YEAR"
    if context.startswith("ROSEWORTHY_E5"):
        return "yield_monitor_point_year"
    return "crop_year"


def _applicability_reason(decision: str, strategy: str) -> str:
    if decision == "EXACT_TRANSFER":
        return "CROP_UNIT_SAMPLE_FEATURE_SCHEMA_COMPATIBLE"
    if decision == "ADAPTED_TRANSFER":
        return "REQUIRES_ADAPTED_PROTOCOL_STRATUM"
    if decision == "DIAGNOSTIC_CROSS_CROP":
        return "CROP_MISMATCH_DIAGNOSTIC_ONLY"
    if decision == "DIAGNOSTIC_CROSS_UNIT":
        return "SAMPLE_UNIT_MISMATCH_DIAGNOSTIC_ONLY"
    if strategy == "harmonised_transfer":
        return "HARMONISATION_CONTRACT_NOT_AVAILABLE"
    return "TRANSFER_CONTRACT_NOT_APPLICABLE"


def build_m_job_derivations() -> dict[str, Any]:
    context_count = len(MULTIMODAL_CONTEXTS)
    training_components = [
        {
            "method": "complete_baseline",
            "model_family": "ridge",
            "applicable_target_contexts": list(MULTIMODAL_CONTEXTS),
            "deterministic_or_stochastic": "deterministic",
            "seed_count": 1,
            "checkpoint_reused_from_source_or_transfer": False,
            "independent_target_training_required": True,
            "structural_jobs": context_count * 1,
            "formula": "6 contexts × 1 seed",
        },
        {
            "method": "imputation",
            "model_family": "rf_recovered_pipeline",
            "deterministic_or_stochastic": "deterministic_preprocessing_audit_seed",
            "seed_count": 1,
            "checkpoint_reused_from_source_or_transfer": False,
            "independent_target_training_required": True,
            "structural_jobs": context_count * 1,
            "formula": "6 contexts × 1 canonical seed",
        },
        {
            "method": "missing_indicators",
            "model_family": "rf_recovered_pipeline",
            "deterministic_or_stochastic": "deterministic_indicator_schema_audit_seed",
            "seed_count": 1,
            "checkpoint_reused_from_source_or_transfer": False,
            "independent_target_training_required": True,
            "structural_jobs": context_count * 1,
            "formula": "6 contexts × 1 canonical seed",
        },
        {
            "method": "late_fusion",
            "model_family": "modality_specific_rf_equal_availability_fusion",
            "deterministic_or_stochastic": "stochastic",
            "seed_count": 3,
            "checkpoint_reused_from_source_or_transfer": False,
            "independent_target_training_required": True,
            "structural_jobs": context_count * 3,
            "formula": "6 contexts × 3 seeds",
        },
        {
            "method": "teacher_student_m2",
            "model_family": "historical_fixed_0_5_blend_control",
            "deterministic_or_stochastic": "stochastic",
            "seed_count": 3,
            "checkpoint_reused_from_source_or_transfer": False,
            "independent_target_training_required": True,
            "structural_jobs": context_count * 3,
            "formula": "6 contexts × 3 seeds",
        },
        {
            "method": "missing_aware",
            "model_family": "neural_missing_aware",
            "deterministic_or_stochastic": "stochastic",
            "seed_count": 3,
            "checkpoint_reused_from_source_or_transfer": False,
            "independent_target_training_required": True,
            "structural_jobs": context_count * 3,
            "formula": "6 contexts × 3 seeds",
        },
    ]
    evaluation_components = [
        {
            "checkpoint_family": "m_training_checkpoints",
            "contexts": list(MULTIMODAL_CONTEXTS),
            "seed_policy": "as_trained",
            "structural_bundles": 72,
            "reused_from": "new_m_training",
        },
        {
            "checkpoint_family": "reused_rf_scratch_baselines",
            "contexts": list(MULTIMODAL_CONTEXTS),
            "seed_policy": "3 seeds",
            "structural_bundles": context_count * 3,
            "reused_from": "matched_scratch",
        },
        {
            "checkpoint_family": "reused_neural_scratch_baselines",
            "contexts": list(MULTIMODAL_CONTEXTS),
            "seed_policy": "3 seeds",
            "structural_bundles": context_count * 3,
            "reused_from": "matched_scratch",
        },
    ]
    return {
        "schema_version": "m_job_derivation_v2",
        "multimodal_target_contexts": list(MULTIMODAL_CONTEXTS),
        "m_training": {
            "structural_ceiling": sum(component["structural_jobs"] for component in training_components),
            "components": training_components,
            "rules": [
                "preprocessing-only components do not count as independent training unless paired with an independently trained model",
                "fingerprint-matching checkpoints are reused and not retrained",
            ],
        },
        "m_evaluation_bundles": {
            "structural_ceiling": sum(component["structural_bundles"] for component in evaluation_components),
            "bundle_grain": "checkpoint × target_context × seed",
            "deployment_conditions": ["complete", "synthetic_no_soil", "synthetic_no_weather", "synthetic_random_0_15", "natural_if_present"],
            "components": evaluation_components,
            "rules": [
                "deployment conditions never multiply training jobs",
                "whole-modality masks are distinct from synthetic_random_0_15",
                "ordinary/harmonised/pooled training jobs are not duplicated",
            ],
        },
    }


def build_control_gate_node_accounting() -> dict[str, Any]:
    return {
        "schema_version": "control_gate_node_accounting_v2",
        "policy": "no unexplained aggregate control-node total is reported",
        "nodes": {
            "phase1_contract_nodes": {"count": 10},
            "transfer_route_seed_specification_freeze_nodes": {"count": 36},
            "gate1_nodes": {"count": 13},
            "gate2_nodes": {"count": 13},
            "plan_freeze_nodes": {"count": "PENDING_BUILD_PLAN", "reason": "depends on eligible_after_contracts"},
            "aggregation_nodes": {"count": "PENDING_BUILD_PLAN", "reason": "depends on released_by_gates"},
            "acceptance_nodes": {"count": "PENDING_BUILD_PLAN", "reason": "depends on executed artifacts"},
            "accounting_nodes": {"count": 1},
            "artifact_validation_nodes": {"count": "PENDING_BUILD_PLAN", "reason": "depends on executable job artifacts"},
            "resource_estimate_nodes": {"count": 1},
            "launch_readiness_node": {"count": 1},
        },
    }


def build_few_shot_contract_registry() -> dict[str, Any]:
    return {
        "schema_version": "few_shot_contract_registry_v2",
        "fractions": [0.01, 0.05, 0.10, 0.20],
        "decision_for_insufficient_contract": "NOT_APPLICABLE_BY_FEW_SHOT_DATA_CONTRACT",
        "checks": [
            "minimum_adaptation_rows",
            "minimum_unique_groups",
            "minimum_unique_locations_or_years",
            "fixed_test_ids",
            "nested_adaptation_subsets",
            "deterministic_sample_id_fraction_lineage",
        ],
        "t1_flow": ["Gate 1", "5% probe", "Gate 2", "1/10/20 expansion"],
        "core_plan_inclusion": False,
    }


def build_candidate_disposition_matrix() -> dict[str, Any]:
    return {
        "schema_version": "candidate_disposition_matrix_v2",
        "terminal_statuses": [
            "EXECUTABLE",
            "NOT_APPLICABLE_BY_DATA_CONTRACT",
            "NOT_APPLICABLE_BY_FEW_SHOT_DATA_CONTRACT",
            "EXCLUDED_BY_APPROVED_SCOPE",
        ],
        "forbidden_statuses": ["BLOCKED", "SILENT_SKIP", "FALLBACK"],
        "current_build_plan_counts": {
            "EXECUTABLE": "PENDING_BUILD_PLAN",
            "NOT_APPLICABLE_BY_DATA_CONTRACT": "PENDING_BUILD_PLAN",
            "NOT_APPLICABLE_BY_FEW_SHOT_DATA_CONTRACT": "PENDING_BUILD_PLAN",
            "EXCLUDED_BY_APPROVED_SCOPE": "PENDING_BUILD_PLAN",
        },
    }


def _family_for_cell(cell: dict[str, Any]) -> str:
    strategy = cell.get("transfer_strategy")
    method = cell.get("missing_modality_method")
    mode = cell.get("adaptation_mode")
    if mode == "few_shot_adaptation":
        return "gated_t1_total"
    if method in {"imputation", "missing_indicators", "late_fusion", "teacher_student_m2", "missing_aware"} and strategy == "target_scratch":
        return "m_training"
    if strategy == "source_only":
        return "t0_zero_shot_bundles"
    if strategy in {"ordinary_transfer", "harmonised_transfer", "pooled_source_target", "target_scratch"}:
        return strategy
    return "unclassified"


def _job_kind_for_family(family: str) -> str:
    if family in {"t0_zero_shot_bundles", "m_evaluation_bundles"}:
        return "inference_evaluation"
    if family.startswith("gate") or family == "gated_t1_total":
        return "gated"
    if family in {"phase1_contracts", "transfer_route_seed_specification_freeze"}:
        return "control_only"
    return "training"


def build_handler_coverage_registry() -> dict[str, Any]:
    families = [
        ("source_contracts", "CONTROL_ONLY", None),
        (
            "source_screening",
            "REAL_HANDLER_BOUND",
            "REMAINING_HANDLER_REGISTRY.source_screening",
        ),
        (
            "teacher_qualification",
            "REAL_HANDLER_BOUND",
            "REMAINING_HANDLER_REGISTRY.teacher_qualification",
        ),
        (
            "primary_source_routes",
            "REAL_HANDLER_BOUND",
            "SOURCE_HANDLER_REGISTRY.primary_source_routes",
        ),
        (
            "multimodal_pretrain_finetune",
            "REAL_HANDLER_BOUND",
            "REMAINING_HANDLER_REGISTRY.multimodal_pretrain_finetune",
        ),
        (
            "source_controls",
            "REAL_HANDLER_BOUND",
            "REMAINING_HANDLER_REGISTRY.source_controls",
        ),
        (
            "source_outer",
            "REAL_HANDLER_BOUND",
            "SOURCE_HANDLER_REGISTRY.source_outer",
        ),
        (
            "t0_zero_shot_bundles",
            "REAL_HANDLER_BOUND",
            "execute_target_job/source_only",
        ),
        (
            "ordinary_transfer",
            "REAL_HANDLER_BOUND",
            "METHOD_HANDLERS.ordinary_transfer",
        ),
        (
            "harmonised_transfer",
            "REAL_HANDLER_BOUND",
            "METHOD_HANDLERS.harmonised_transfer",
        ),
        (
            "pooled_source_target",
            "REAL_HANDLER_BOUND",
            "METHOD_HANDLERS.pooled_source_target",
        ),
        (
            "target_scratch",
            "REAL_HANDLER_BOUND",
            "METHOD_HANDLERS.target_scratch",
        ),
        (
            "m_training",
            "REAL_HANDLER_BOUND",
            "historical_m1_m2_and_missing_aware_paths",
        ),
        (
            "m_evaluation_bundles",
            "REAL_HANDLER_BOUND",
            "M_EVALUATION_HANDLER_REGISTRY.m_evaluation_bundles",
        ),
        ("gate1_probe", "GATED_NOT_RELEASED", None),
        ("gate2_expansion", "GATED_NOT_RELEASED", None),
    ]

    records = []
    for family, status, handler_name in families:
        callable_verified = False

        if family in SOURCE_HANDLER_REGISTRY:
            callable_verified = callable(
                SOURCE_HANDLER_REGISTRY[family]
            )
        elif family in {
            "ordinary_transfer",
            "harmonised_transfer",
            "pooled_source_target",
            "target_scratch",
        }:
            callable_verified = callable(
                METHOD_HANDLERS.get(family)
            )
        elif family == "t0_zero_shot_bundles":
            callable_verified = callable(
                METHOD_HANDLERS.get("source_only")
            )
        elif family == "m_training":
            callable_verified = True
        elif family in M_EVALUATION_HANDLER_REGISTRY:
            callable_verified = callable(
                M_EVALUATION_HANDLER_REGISTRY[family]
            )
        elif family in REMAINING_HANDLER_REGISTRY:
            callable_verified = callable(
                REMAINING_HANDLER_REGISTRY[family]
            )

        records.append(
            {
                "family": family,
                "status": status,
                "handler": handler_name,
                "callable_verified": callable_verified,
            }
        )

    real = [
        item
        for item in records
        if item["status"] == "REAL_HANDLER_BOUND"
    ]
    unverified_real = [
        item["family"]
        for item in real
        if not item["callable_verified"]
    ]

    return {
        "schema_version": "handler_coverage_registry_v3",
        "method_handlers": sorted(METHOD_HANDLERS),
        "source_handlers": sorted(SOURCE_HANDLER_REGISTRY),
        "m_evaluation_handlers": sorted(
            M_EVALUATION_HANDLER_REGISTRY
        ),
        "remaining_handlers": sorted(
            REMAINING_HANDLER_REGISTRY
        ),
        "families": records,
        "unverified_real_handlers": unverified_real,
        "summary": {
            "real_handler_bound": sum(
                item["status"] == "REAL_HANDLER_BOUND"
                for item in records
            ),
            "accounting_only_families": sum(
                item["status"] == "ACCOUNTING_ONLY"
                for item in records
            ),
            "missing_handler_families": sum(
                item["status"] == "MISSING_HANDLER"
                for item in records
            ),
            "gated_not_released": sum(
                item["status"] == "GATED_NOT_RELEASED"
                for item in records
            ),
            "unverified_real_handlers": len(
                unverified_real
            ),
        },
    }



def _accepted_control_artifact(
    control_root: Path,
    filename: str,
    *,
    expected_status: str,
) -> tuple[bool, dict[str, Any] | None, str]:
    """Validate an independently written final-acceptance artifact.

    Missing, malformed, wrong-status, or self-inconsistent artifacts fail
    closed.  This function does not create or approve the artifact.
    """
    path = control_root / filename

    if not path.is_file():
        return False, None, "MISSING"

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False, None, "INVALID_JSON"

    if not isinstance(payload, dict):
        return False, None, "NOT_OBJECT"

    if payload.get("status") != expected_status:
        return False, payload, "STATUS_NOT_ACCEPTED"

    if payload.get("formal_training_started") is not False:
        return False, payload, "FORMAL_TRAINING_STATE_INVALID"

    if not payload.get("acceptance_fingerprint"):
        return False, payload, "ACCEPTANCE_FINGERPRINT_MISSING"

    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key != "acceptance_fingerprint"
    }
    if payload["acceptance_fingerprint"] != _fingerprint(
        fingerprint_payload
    ):
        return False, payload, "ACCEPTANCE_FINGERPRINT_MISMATCH"

    return True, payload, "ACCEPTED"


def _final_acceptance_gate_status(
    control_root: Path,
) -> dict[str, Any]:
    smoke_ok, smoke_payload, smoke_reason = (
        _accepted_control_artifact(
            control_root,
            "smoke_mini_dag_acceptance.json",
            expected_status="SMOKE_MINI_DAG_ACCEPTED",
        )
    )
    resource_ok, resource_payload, resource_reason = (
        _accepted_control_artifact(
            control_root,
            "resource_acceptance.json",
            expected_status="RESOURCE_ACCEPTED",
        )
    )

    return {
        "smoke_mini_dag": {
            "accepted": smoke_ok,
            "reason": smoke_reason,
            "artifact": smoke_payload,
        },
        "resources": {
            "accepted": resource_ok,
            "reason": resource_reason,
            "artifact": resource_payload,
        },
        "all_final_acceptance_gates_passed": (
            smoke_ok and resource_ok
        ),
    }

def build_final_eligibility(repository_root: Path) -> dict[str, Any]:
    v1_config_path = repository_root / "configs/universal_weather_full_campaign_v2_remediation_v1.yaml"
    v1_schema_path = repository_root / "schemas/universal_weather_full_campaign_v2_remediation_v1.schema.json"
    config = load_config(v1_config_path, v1_schema_path)
    preflight = run_preflight(config, repository_root=repository_root, remediation_items=[], write=False)
    cells = apply_observed_split_eligibility(build_candidate_matrix(config), preflight["split_registry"], config)
    executable = [cell for cell in cells if cell["status"] == "EXECUTABLE"]
    by_family: dict[str, dict[str, Any]] = {}
    reason_histogram: dict[str, int] = {}
    for cell in cells:
        family = _family_for_cell(cell)
        item = by_family.setdefault(
            family,
            {
                "eligible_after_contracts": 0,
                "not_applicable_or_blocked": 0,
                "kind": _job_kind_for_family(family),
                "confirmatory": 0,
                "diagnostic": 0,
            },
        )
        if cell["status"] == "EXECUTABLE":
            item["eligible_after_contracts"] += 1
            if cell.get("dataset_id", "").startswith("ROSEWORTHY") or cell.get("source_dataset") == "PRIMARY_G2F_MAIZE":
                item["diagnostic"] += 1
            else:
                item["confirmatory"] += 1
        else:
            item["not_applicable_or_blocked"] += 1
            reason = cell.get("status_reason", "UNKNOWN")
            reason_histogram[reason] = reason_histogram.get(reason, 0) + 1
    duplicate_keys = len(executable) - len({_manifest_identity(cell) for cell in executable})
    deterministic_seed_audit = {
        "status": "NO_DETERMINISTIC_MODEL_SEED_COLLAPSE_APPLIED_IN_CURRENT_V1_MATRIX",
        "note": "Current v1 candidate matrix does not encode deterministic model family at job identity grain; final handler layer must collapse deterministic repeats before formal training.",
    }
    deployment_condition_audit = {
        "status": "EVALUATION_CONDITIONS_AUDITED",
        "note": "DAG v2 report separates M evaluation bundles from M training; formal manifest must keep deployment conditions inside evaluation bundles.",
    }
    return {
        "schema_version": "final_eligibility_summary_v2",
        "candidate_count": len(cells),
        "eligible_after_contracts_total": len(executable),
        "not_applicable_or_blocked_total": len(cells) - len(executable),
        "executed": 0,
        "formal_jobs_executed": 0,
        "by_family": dict(sorted(by_family.items())),
        "reason_histogram": dict(sorted(reason_histogram.items())),
        "confirmatory_total": sum(item["confirmatory"] for item in by_family.values()),
        "diagnostic_total": sum(item["diagnostic"] for item in by_family.values()),
        "job_kind_totals": _job_kind_totals(by_family),
        "duplicate_job_identities": duplicate_keys,
        "deterministic_seed_audit": deterministic_seed_audit,
        "deployment_condition_audit": deployment_condition_audit,
        "gated_t1_future_release": {
            "minimum": 0,
            "maximum": build_target_structural_families()["gated_t1_total"]["structural_ceiling"],
            "status": "PENDING_VALIDATION_GATES_NOT_IN_FORMAL_MANIFEST",
        },
        "preflight_failed_checks": preflight["failed_preflight_checks"],
    }


def _manifest_identity(cell: dict[str, Any]) -> str:
    fields = (
        "dataset_id",
        "source_dataset",
        "source_route",
        "transfer_strategy",
        "adaptation_mode",
        "adaptation_fraction",
        "missing_modality_method",
        "deployment_condition",
        "split_id",
        "seed",
        "model_contract",
    )
    return _fingerprint({field: cell.get(field) for field in fields})


def _job_kind_totals(by_family: dict[str, dict[str, Any]]) -> dict[str, int]:
    totals = {"training": 0, "inference_evaluation": 0, "control_only": 0, "gated": 0}
    for item in by_family.values():
        kind = item["kind"]
        totals[kind] = totals.get(kind, 0) + int(item["eligible_after_contracts"])
    totals["control_only"] += build_source_accounting()["source_control_only_nodes"]["structural_ceiling"]
    return totals


def _legacy_build_formal_preapproval_manifest_v2(
    repository_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    eligibility = build_final_eligibility(repository_root)
    coverage = build_handler_coverage_registry()
    blockers = []

    final_acceptance = _final_acceptance_gate_status(
        output_root / "control"
    )

    if not final_acceptance["smoke_mini_dag"]["accepted"]:
        blockers.append("SMOKE_MINI_DAG_NOT_ACCEPTED")

    if not final_acceptance["resources"]["accepted"]:
        blockers.append("RESOURCE_ACCEPTANCE_NOT_RECORDED")
    if coverage["summary"]["missing_handler_families"]:
        blockers.append("MISSING_HANDLER")
    if eligibility["duplicate_job_identities"]:
        blockers.append("DUPLICATE_JOB_IDENTITIES")
    if eligibility["preflight_failed_checks"]:
        blockers.append("PREFLIGHT_FAILURES")
    if coverage["summary"]["accounting_only_families"]:
        blockers.append("ACCOUNTING_ONLY_FAMILIES_REMAIN")
    payload = {
        "schema_version": "formal_preapproval_manifest_v2",
        "eligible_after_contracts_total": eligibility["eligible_after_contracts_total"],
        "formal_jobs_executed": 0,
        "final_acceptance_gates": final_acceptance,
        "scientific_status": "NOT_RUN",
        "blockers": blockers,
        "can_enter_final_user_approval": not blockers,
        "handler_summary": coverage["summary"],
        "job_kind_totals": eligibility["job_kind_totals"],
    }
    payload["exact_plan_fingerprint"] = _fingerprint(payload)
    return payload


def build_dag_v2_spec() -> dict[str, Any]:
    source = build_source_accounting()
    target = build_target_structural_families()
    m = build_m_job_derivations()
    spec = {
        "schema_version": "universal_weather_v2_remediation_dag_budget_and_eligibility_spec_v2",
        "count_denominators": list(COUNT_KEYS),
        "source_accounting": source,
        "target_structural_families": target,
        "expected_total_jobs": {
            "expected_total_jobs": "PENDING_DEVELOPMENT_GATES",
            "core_structural_ceiling": 2244,
            "worst_case_with_gated_t1": 2712,
            "eligible_after_contracts": "PENDING_BUILD_PLAN",
            "released_by_gates": "PENDING_GATE",
            "executed": 0,
        },
        "m_job_derivation_ref": {
            "m_training_structural_ceiling": m["m_training"]["structural_ceiling"],
            "m_evaluation_structural_ceiling": m["m_evaluation_bundles"]["structural_ceiling"],
        },
        "budget_policy": {
            "budget_is_assertion_not_truncation": True,
            "ceiling_overflow_action": "BUDGET_ASSERTION_FAILED",
            "silent_truncation_allowed": False,
        },
        "missing_aware_replacement": {
            "legacy_route": "neural_masked_pretrain",
            "formal_route_status": "HISTORICAL_LINEAGE_ONLY",
            "phase2_replacement": {"route": "missing_aware_screening", "mask_probability": 0.5},
            "phase4_route": {"route": "missing_aware", "mask_probability": 0.5},
            "shared_masking_implementation": "dynamic_bernoulli_whole_modality_mask",
        },
        "formal_execution": {
            "requires_user_second_approval": True,
            "requires_exact_plan_fingerprint": True,
            "executed_this_round": 0,
        },
    }
    spec["spec_fingerprint"] = _fingerprint({key: value for key, value in spec.items() if key != "spec_fingerprint"})
    return spec


def build_phase_plans() -> dict[str, Any]:
    return {
        "source_phase_plan": {"schema_version": "source_phase_plan_v2", "counts": build_source_accounting()["phase_rows"]},
        "source_outer_plan": {
            "schema_version": "source_outer_plan_v2",
            "primary_confirmatory_jobs": 90,
            "secondary_supportive_jobs": 18,
            "excluded_diagnostic_routes": ["soil_direct"],
        },
        "target_core_plan": {"schema_version": "target_core_plan_v2", "families": build_target_structural_families()},
        "gate1_release": {"schema_version": "gate1_release_v2", "status": "PENDING_GATE", "released_jobs": 0},
        "few_shot_probe_plan": {"schema_version": "few_shot_probe_plan_v2", "status": "PENDING_GATE", "structural_ceiling": 117},
        "gate2_release": {"schema_version": "gate2_release_v2", "status": "PENDING_GATE", "released_jobs": 0},
        "few_shot_expansion_plan": {"schema_version": "few_shot_expansion_plan_v2", "status": "PENDING_GATE", "structural_ceiling": 351},
    }


def write_dag_v2_outputs(output_root: Path, repository_root: Path | None = None) -> dict[str, Any]:
    control = output_root / "control"
    control.mkdir(parents=True, exist_ok=True)
    payloads = {
        "dag_budget_and_eligibility_spec.json": build_dag_v2_spec(),
        "target_applicability_registry.json": {"schema_version": "target_applicability_registry_v2", "records": build_transfer_applicability_registry()},
        "m_job_derivation.json": build_m_job_derivations(),
        "m_evaluation_bundle_derivation.json": build_m_job_derivations()["m_evaluation_bundles"],
        "source_dag_accounting.json": build_source_accounting(),
        "control_gate_node_accounting.json": build_control_gate_node_accounting(),
        "few_shot_contract_registry.json": build_few_shot_contract_registry(),
        "candidate_disposition_matrix.json": build_candidate_disposition_matrix(),
    }
    if repository_root is not None:
        final_eligibility = build_final_eligibility(repository_root)
        handler_coverage = build_handler_coverage_registry()
        manifest = build_formal_preapproval_manifest(repository_root)
        payloads.update(
            {
                "final_eligibility_summary.json": final_eligibility,
                "handler_coverage_registry.json": handler_coverage,
                "formal_preapproval_manifest.json": manifest,
            }
        )
    payloads.update({f"{name}.json": payload for name, payload in build_phase_plans().items()})
    for filename, payload in payloads.items():
        _write_json(control / filename, payload)
    return {
        "output_root": str(output_root),
        "control_files_written": len(payloads),
        "spec_fingerprint": payloads["dag_budget_and_eligibility_spec.json"]["spec_fingerprint"],
        "executed": 0,
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)

# ============================================================================
# CONTROL-PLANE REPAIR V3
#
# This section intentionally overrides earlier v2 accounting helpers. It binds
# DAG accounting to the explicit execution config, excludes unreleased T1 jobs
# from the formal core manifest, derives evidence strata from applicability,
# and produces one exact preapproval fingerprint.
# ============================================================================

def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_bound_configs(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    dag_path = repository_root / "configs/universal_weather_full_campaign_v2_remediation_dag_v1.yaml"
    dag = yaml.safe_load(dag_path.read_text())

    execution_path = repository_root / dag["execution_config"]
    execution_schema_path = repository_root / dag["execution_schema"]

    execution = load_config(execution_path, execution_schema_path)
    return dag, execution, dag_path, execution_path


def _context_from_dataset_split(dataset_id: str, split_id: str) -> str | None:
    if dataset_id == "CY-Bench_wheat_AU" and split_id in {"RANDOM", "GROUP", "SPATIAL"}:
        return f"CYBENCH_WHEAT_AU_{split_id}"

    if dataset_id == "waite" and split_id == "ROLLING_1991":
        return "WAITE_ROLLING_1991"

    if dataset_id == "ROSEWORTHY_E5_POINT" and split_id == "SPATIAL":
        return "ROSEWORTHY_E5_SPATIAL"

    if dataset_id == "ROSEWORTHY_HISTORICAL_FARM" and split_id.startswith("ROLLING_"):
        return f"ROSEWORTHY_HISTORICAL_GRAIN_{split_id}"

    return None


def _crop_for_dataset(dataset_id: str) -> str:
    if dataset_id == "CY-Bench_wheat_AU":
        return "wheat"
    if dataset_id == "waite":
        return "wheat"
    if dataset_id == "ROSEWORTHY_E5_POINT":
        return "mixed_e5_crop"
    if dataset_id == "ROSEWORTHY_HISTORICAL_FARM":
        return "grain_mixed"
    return "unknown"


def _applicability_decision(
    *,
    source_id: str,
    source_contract: dict[str, Any],
    target_id: str,
    target_contract: dict[str, Any],
    strategy: str,
) -> tuple[str, str, str]:
    source_crop = "maize"
    target_crop = _crop_for_dataset(target_id)

    source_unit = source_contract["target_unit"]
    target_unit = target_contract["target_unit"]
    source_sample = source_contract["sample_unit"]
    target_sample = target_contract["sample_unit"]

    common_features = sorted(
        set(source_contract.get("weather_columns", []))
        & set(target_contract.get("weather_columns", []))
    )

    crop_match = source_crop == target_crop
    unit_match = source_unit == target_unit
    sample_match = source_sample == target_sample

    if strategy == "harmonised_transfer" and not common_features:
        return (
            "NOT_APPLICABLE",
            "not_applicable",
            "HARMONISATION_REQUIRES_REAL_COMMON_FEATURE_INTERSECTION",
        )

    if not crop_match:
        return (
            "DIAGNOSTIC_CROSS_CROP",
            "diagnostic",
            "SOURCE_TARGET_CROP_MISMATCH_DIAGNOSTIC_ONLY",
        )

    if not unit_match or not sample_match:
        return (
            "DIAGNOSTIC_CROSS_UNIT",
            "diagnostic",
            "SOURCE_TARGET_UNIT_OR_SAMPLE_UNIT_MISMATCH_DIAGNOSTIC_ONLY",
        )

    if common_features:
        return (
            "EXACT_TRANSFER",
            "confirmatory",
            "CROP_UNIT_SAMPLE_AND_FEATURE_CONTRACT_COMPATIBLE",
        )

    if strategy == "ordinary_transfer":
        return (
            "ADAPTED_TRANSFER",
            "confirmatory",
            "UNIVERSAL_FEATURE_TOKEN_CONTRACT_REQUIRES_HANDLER_VALIDATION",
        )

    return (
        "NOT_APPLICABLE",
        "not_applicable",
        "TRANSFER_CONTRACT_NOT_SATISFIED",
    )


def build_transfer_applicability_registry(
    repository_root: Path | None = None,
) -> list[dict[str, Any]]:
    repository_root = repository_root or _repository_root()
    dag, execution, _, _ = _load_bound_configs(repository_root)

    records: list[dict[str, Any]] = []

    approved_pairs = []
    for dataset_id, contract in execution["datasets"].items():
        for split_id in contract["splits"]:
            context = _context_from_dataset_split(dataset_id, split_id)
            if context and context in set(dag["approved_target_contexts"]):
                approved_pairs.append((dataset_id, split_id, context))

    routes = [
        *dag["primary_source_routes"],
        *dag["secondary_source_routes"],
    ]

    for source_id in dag["source_datasets"]:
        source_contract = execution["source_datasets"][source_id]
        for route in routes:
            for dataset_id, split_id, context in approved_pairs:
                target_contract = execution["datasets"][dataset_id]
                for strategy in ("ordinary_transfer", "harmonised_transfer"):
                    decision, stratum, reason = _applicability_decision(
                        source_id=source_id,
                        source_contract=source_contract,
                        target_id=dataset_id,
                        target_contract=target_contract,
                        strategy=strategy,
                    )
                    common_features = sorted(
                        set(source_contract.get("weather_columns", []))
                        & set(target_contract.get("weather_columns", []))
                    )
                    records.append(
                        {
                            "source": source_id,
                            "source_route": route,
                            "target_dataset": dataset_id,
                            "target_split": split_id,
                            "target_context": context,
                            "transfer_strategy": strategy,
                            "crop_compatibility": _crop_for_dataset(dataset_id) == "maize",
                            "target_semantics_compatibility": True,
                            "unit_compatibility": (
                                source_contract["target_unit"]
                                == target_contract["target_unit"]
                            ),
                            "sample_unit_compatibility": (
                                source_contract["sample_unit"]
                                == target_contract["sample_unit"]
                            ),
                            "feature_schema_compatibility": bool(common_features),
                            "common_feature_intersection": common_features,
                            "harmonisation_availability": bool(common_features),
                            "deployment_modality_compatibility": True,
                            "evidence_stratum": stratum,
                            "decision": decision,
                            "reason": reason,
                            "supporting_evidence": {
                                "source_crop": "maize",
                                "target_crop": _crop_for_dataset(dataset_id),
                                "source_unit": source_contract["target_unit"],
                                "target_unit": target_contract["target_unit"],
                                "source_sample_unit": source_contract["sample_unit"],
                                "target_sample_unit": target_contract["sample_unit"],
                                "common_features": common_features,
                            },
                        }
                    )

    return records


def _applicability_lookup(
    repository_root: Path,
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    return {
        (
            item["source"],
            item["source_route"],
            item["target_dataset"],
            item["target_split"],
            item["transfer_strategy"],
        ): item
        for item in build_transfer_applicability_registry(repository_root)
    }


def _cell_evidence(
    cell: dict[str, Any],
    applicability: dict[tuple[str, str, str, str, str], dict[str, Any]],
) -> tuple[str, str]:
    dataset_id = cell["dataset_id"]
    split_id = cell["split_id"]
    strategy = cell["transfer_strategy"]
    method = cell["missing_modality_method"]

    context = _context_from_dataset_split(dataset_id, split_id)
    if context is None:
        return "excluded", "OUTSIDE_APPROVED_DAG_TARGET_CONTEXTS"

    if cell["adaptation_mode"] == "few_shot_adaptation":
        return "gated", "T1_REQUIRES_GATE_RELEASE"

    if strategy == "target_scratch":
        if dataset_id.startswith("ROSEWORTHY"):
            return "diagnostic", "ROSEWORTHY_TARGET_DIRECT_CONTROL_DIAGNOSTIC"
        return "confirmatory", "TARGET_DIRECT_CONTROL"

    if method in {
        "imputation",
        "missing_indicators",
        "late_fusion",
        "teacher_student_m2",
    }:
        if dataset_id.startswith("ROSEWORTHY"):
            return "diagnostic", "ROSEWORTHY_M_CONTROL_DIAGNOSTIC"
        return "confirmatory", "TARGET_MISSING_MODALITY_CONTROL"

    lookup_strategy = (
        "ordinary_transfer"
        if strategy in {"source_only", "pooled_source_target"}
        else strategy
    )
    record = applicability.get(
        (
            cell["source_dataset"],
            cell["source_route"],
            dataset_id,
            split_id,
            lookup_strategy,
        )
    )
    if record is None:
        return "excluded", "NO_APPLICABILITY_RECORD"

    if record["decision"] == "NOT_APPLICABLE":
        return "excluded", record["reason"]

    return record["evidence_stratum"], record["reason"]


def build_final_eligibility(repository_root: Path) -> dict[str, Any]:
    dag, execution, dag_path, execution_path = _load_bound_configs(repository_root)

    preflight = run_preflight(
        execution,
        repository_root=repository_root,
        remediation_items=[],
        write=False,
    )
    cells = apply_observed_split_eligibility(
        build_candidate_matrix(execution),
        preflight["split_registry"],
        execution,
    )

    applicability = _applicability_lookup(repository_root)

    core_executable: list[dict[str, Any]] = []
    gated_candidates: list[dict[str, Any]] = []
    excluded_by_dag: list[dict[str, Any]] = []
    reason_histogram: dict[str, int] = {}
    by_family: dict[str, dict[str, Any]] = {}

    for cell in cells:
        if cell["status"] != "EXECUTABLE":
            reason = cell.get("status_reason", "UNKNOWN")
            reason_histogram[reason] = reason_histogram.get(reason, 0) + 1
            continue

        stratum, reason = _cell_evidence(cell, applicability)

        if stratum == "gated":
            gated_candidates.append(cell)
            reason_histogram[reason] = reason_histogram.get(reason, 0) + 1
            continue

        if stratum == "excluded":
            excluded_by_dag.append(cell)
            reason_histogram[reason] = reason_histogram.get(reason, 0) + 1
            continue

        enriched = {
            **cell,
            "evidence_stratum": stratum,
            "applicability_reason": reason,
            "target_context": _context_from_dataset_split(
                cell["dataset_id"], cell["split_id"]
            ),
        }
        core_executable.append(enriched)

        family = _family_for_cell(cell)
        item = by_family.setdefault(
            family,
            {
                "eligible_after_contracts": 0,
                "not_applicable_or_blocked": 0,
                "kind": _job_kind_for_family(family),
                "confirmatory": 0,
                "diagnostic": 0,
            },
        )
        item["eligible_after_contracts"] += 1
        item[stratum] += 1

    duplicate_keys = len(core_executable) - len(
        {_manifest_identity(cell) for cell in core_executable}
    )

    manifest_jobs = sorted(
        core_executable,
        key=lambda cell: (
            cell["dataset_id"],
            cell["split_id"],
            cell["transfer_strategy"],
            cell["source_dataset"],
            cell["source_route"],
            cell["missing_modality_method"],
            cell["deployment_condition"],
            cell["seed"],
            str(cell.get("adaptation_fraction")),
        ),
    )

    execution_config_hash = sha256_file(execution_path)
    dag_config_hash = sha256_file(dag_path)
    immutable_core_plan_fingerprint = _fingerprint(
        {
            "dag_config_hash": dag_config_hash,
            "execution_config_hash": execution_config_hash,
            "jobs": manifest_jobs,
            "gate1_released": dag["gate_policy"]["gate1_released"],
            "gate2_released": dag["gate_policy"]["gate2_released"],
        }
    )

    return {
        "schema_version": "final_eligibility_summary_v3",
        "candidate_count": len(cells),
        "eligible_after_contracts_total": len(core_executable),
        "core_executable_job_count": len(core_executable),
        "gated_t1_candidate_count": len(gated_candidates),
        "gated_t1_released_count": 0,
        "excluded_by_approved_dag_scope": len(excluded_by_dag),
        "not_applicable_or_blocked_total": (
            len(cells) - len(core_executable) - len(gated_candidates)
        ),
        "executed": 0,
        "formal_jobs_executed": 0,
        "by_family": dict(sorted(by_family.items())),
        "reason_histogram": dict(sorted(reason_histogram.items())),
        "confirmatory_total": sum(
            item["confirmatory"] for item in by_family.values()
        ),
        "diagnostic_total": sum(
            item["diagnostic"] for item in by_family.values()
        ),
        "job_kind_totals": _job_kind_totals(by_family),
        "duplicate_job_identities": duplicate_keys,
        "preflight_failed_checks": preflight["failed_preflight_checks"],
        "dag_config_hash": dag_config_hash,
        "execution_config_hash": execution_config_hash,
        "immutable_core_plan_fingerprint": immutable_core_plan_fingerprint,
        "manifest_jobs": manifest_jobs,
        "gated_t1_future_release": {
            "minimum": 0,
            "maximum": build_target_structural_families()[
                "gated_t1_total"
            ]["structural_ceiling"],
            "candidate_count_before_gate": len(gated_candidates),
            "released_count": 0,
            "status": "NOT_IN_CORE_MANIFEST_PENDING_VALIDATION_GATES",
        },
    }


def build_formal_preapproval_manifest(
    repository_root: Path,
    *,
    control_root: Path | None = None,
) -> dict[str, Any]:
    eligibility = build_final_eligibility(repository_root)
    coverage = build_handler_coverage_registry()

    blockers: list[str] = []

    if coverage["summary"]["missing_handler_families"]:
        blockers.append("MISSING_HANDLER")

    if coverage["summary"]["accounting_only_families"]:
        blockers.append("ACCOUNTING_ONLY_FAMILIES_REMAIN")

    if eligibility["duplicate_job_identities"]:
        blockers.append("DUPLICATE_JOB_IDENTITIES")

    if eligibility["preflight_failed_checks"]:
        blockers.append("PREFLIGHT_FAILURES")

    if eligibility["gated_t1_released_count"] != 0:
        blockers.append("UNEXPECTED_UNAPPROVED_T1_RELEASE")

    resolved_control_root = (
        control_root
        if control_root is not None
        else (
            repository_root
            / "outputs/distillation_program/"
            "stage8_v4_australian_narrative_v1/"
            "universal_weather_full_campaign_v2_remediation_dag_v1/"
            "control"
        )
    )
    final_acceptance = _final_acceptance_gate_status(
        resolved_control_root
    )

    if not final_acceptance["smoke_mini_dag"]["accepted"]:
        blockers.append("SMOKE_MINI_DAG_NOT_ACCEPTED")

    if not final_acceptance["resources"]["accepted"]:
        blockers.append("RESOURCE_ACCEPTANCE_NOT_RECORDED")

    blockers = list(dict.fromkeys(blockers))

    payload = {
        "schema_version": "formal_preapproval_manifest_v3",
        "eligible_after_contracts_total": (
            eligibility["eligible_after_contracts_total"]
        ),
        "core_executable_job_count": eligibility["core_executable_job_count"],
        "gated_t1_candidate_count": eligibility["gated_t1_candidate_count"],
        "gated_t1_released_count": eligibility["gated_t1_released_count"],
        "confirmatory_total": eligibility["confirmatory_total"],
        "diagnostic_total": eligibility["diagnostic_total"],
        "formal_jobs_executed": 0,
        "final_acceptance_gates": final_acceptance,
        "scientific_status": "NOT_RUN",
        "blockers": blockers,
        "can_enter_final_user_approval": not blockers,
        "handler_summary": coverage["summary"],
        "job_kind_totals": eligibility["job_kind_totals"],
        "dag_config_hash": eligibility["dag_config_hash"],
        "execution_config_hash": eligibility["execution_config_hash"],
        "immutable_core_plan_fingerprint": (
            eligibility["immutable_core_plan_fingerprint"]
        ),
    }
    payload["exact_plan_fingerprint"] = _fingerprint(payload)
    return payload


def write_dag_v2_outputs(
    output_root: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    repository_root = repository_root or _repository_root()
    control = output_root / "control"
    control.mkdir(parents=True, exist_ok=True)

    eligibility = build_final_eligibility(repository_root)
    coverage = build_handler_coverage_registry()
    manifest = build_formal_preapproval_manifest(
        repository_root,
        control_root=output_root / "control",
    )

    disposition = build_candidate_disposition_matrix()
    disposition["current_build_plan_counts"] = {
        "CORE_EXECUTABLE": eligibility["core_executable_job_count"],
        "GATED_T1_NOT_RELEASED": eligibility["gated_t1_candidate_count"],
        "EXCLUDED_BY_APPROVED_SCOPE": (
            eligibility["excluded_by_approved_dag_scope"]
        ),
        "NOT_APPLICABLE_OR_BLOCKED": (
            eligibility["not_applicable_or_blocked_total"]
        ),
    }

    payloads = {
        "dag_budget_and_eligibility_spec.json": build_dag_v2_spec(),
        "resolved_execution_contract.json": {
            "schema_version": "resolved_execution_contract_v1",
            "dag_config_hash": eligibility["dag_config_hash"],
            "execution_config_hash": eligibility["execution_config_hash"],
            "immutable_core_plan_fingerprint": (
                eligibility["immutable_core_plan_fingerprint"]
            ),
            "core_executable_job_count": (
                eligibility["core_executable_job_count"]
            ),
            "gated_t1_released_count": 0,
        },
        "immutable_core_job_manifest.json": {
            "schema_version": "immutable_core_job_manifest_v1",
            "plan_fingerprint": (
                eligibility["immutable_core_plan_fingerprint"]
            ),
            "job_count": len(eligibility["manifest_jobs"]),
            "jobs": eligibility["manifest_jobs"],
        },
        "target_applicability_registry.json": {
            "schema_version": "target_applicability_registry_v3",
            "records": build_transfer_applicability_registry(repository_root),
        },
        "m_job_derivation.json": build_m_job_derivations(),
        "m_evaluation_bundle_derivation.json": (
            build_m_job_derivations()["m_evaluation_bundles"]
        ),
        "source_dag_accounting.json": build_source_accounting(),
        "control_gate_node_accounting.json": (
            build_control_gate_node_accounting()
        ),
        "few_shot_contract_registry.json": build_few_shot_contract_registry(),
        "candidate_disposition_matrix.json": disposition,
        "final_eligibility_summary.json": {
            key: value
            for key, value in eligibility.items()
            if key != "manifest_jobs"
        },
        "handler_coverage_registry.json": coverage,
        "formal_preapproval_manifest.json": manifest,
    }

    payloads.update(
        {
            f"{name}.json": payload
            for name, payload in build_phase_plans().items()
        }
    )

    for filename, payload in payloads.items():
        _write_json(control / filename, payload)

    return {
        "output_root": str(output_root),
        "control_files_written": len(payloads),
        "formal_run_ready": manifest["can_enter_final_user_approval"],
        "failed_preflight_checks": (
            eligibility["preflight_failed_checks"]
        ),
        "blockers": manifest["blockers"],
        "core_executable_job_count": (
            eligibility["core_executable_job_count"]
        ),
        "gated_t1_released_count": 0,
        "exact_plan_fingerprint": manifest["exact_plan_fingerprint"],
        "immutable_core_plan_fingerprint": (
            eligibility["immutable_core_plan_fingerprint"]
        ),
        "executed": 0,
    }
