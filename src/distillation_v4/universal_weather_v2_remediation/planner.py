from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


TERMINAL_STATES = {"EXECUTABLE", "NOT_APPLICABLE_BY_DATA_CONTRACT", "EXCLUDED_BY_APPROVED_SCOPE"}


def _id(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _adjudicate(cell: dict, contract: dict, source_contract: dict, canonical_source: str) -> tuple[str, str]:
    modalities = set(contract.get("modalities", []))
    strategy, mode = cell["transfer_strategy"], cell["adaptation_mode"]
    method, condition = cell["missing_modality_method"], cell["deployment_condition"]
    if cell["dataset_id"].startswith("ROSEWORTHY_HISTORICAL"):
        if method != "imputation":
            return "NOT_APPLICABLE_BY_DATA_CONTRACT", "HISTORICAL_ROSEWORTHY_INVALID_M_METHOD"
        if condition != "complete":
            return "NOT_APPLICABLE_BY_DATA_CONTRACT", "HISTORICAL_ROSEWORTHY_INVALID_DEPLOYMENT_CONDITION"
    if strategy == "source_only" and mode != "no_adaptation":
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "SOURCE_ONLY_HAS_NO_TARGET_ADAPTATION"
    if strategy != "source_only" and mode == "no_adaptation":
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "TARGET_STRATEGY_REQUIRES_ADAPTATION"
    if strategy == "target_scratch" and cell["source_route"] != "supervised":
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "SCRATCH_HAS_NO_SOURCE_ROUTE"
    if strategy == "target_scratch" and method == "missing_aware":
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "MISSING_AWARE_REQUIRES_PRETRAINED_ROUTE"
    if strategy == "target_scratch" and cell["source_dataset"] != canonical_source:
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "SCRATCH_IS_SOURCE_INDEPENDENT_DEDUPLICATED"
    common_weather = set(contract.get("weather_columns", [])) & set(source_contract.get("weather_columns", []))
    if strategy in {"harmonised_transfer", "pooled_source_target"} and not common_weather:
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "NO_RECOVERED_COMMON_FEATURE_INTERSECTION"
    if strategy == "pooled_source_target" and contract.get("target_unit") != source_contract.get("target_unit"):
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "POOLED_TARGET_UNIT_MISMATCH"
    historical_missing = {"imputation", "missing_indicators", "late_fusion", "teacher_student_m2"}
    if method in historical_missing and (
        cell["source_route"] != "supervised"
        or strategy != "target_scratch"
        or mode == "no_adaptation"
    ):
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "HISTORICAL_M_METHOD_IS_TARGET_DIRECT_CONTROL"
    if condition == "natural" and not contract.get("natural_missingness", False):
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "NO_AUDITED_NATURAL_MISSINGNESS"
    soil_needed = method in {"late_fusion", "teacher_student_m2"} or condition == "synthetic_no_soil"
    if soil_needed and "soil" not in modalities:
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "SOIL_MODALITY_ABSENT"
    if mode == "few_shot_adaptation" and int(contract.get("years", 0)) < 3:
        return "NOT_APPLICABLE_BY_DATA_CONTRACT", "INSUFFICIENT_ADAPTATION_YEARS"
    return "EXECUTABLE", "CONTRACT_SATISFIED"


def build_candidate_matrix(config: dict) -> list[dict]:
    cells = []
    axes = [
        config["sources"],
        config["source_routes"],
        config["transfer_strategies"],
        config["adaptation_modes"],
        config["missing_modality_methods"],
        config["deployment_conditions"],
        config["seeds"],
    ]
    canonical_source = config["sources"][0]
    for dataset_id, contract in config["datasets"].items():
        for split_id in contract["splits"]:
            for source, route, strategy, mode, method, condition, seed in itertools.product(*axes):
                fractions = config["adaptation_fractions"] if mode == "few_shot_adaptation" else [None]
                for fraction in fractions:
                    cell = {
                        "dataset_id": dataset_id,
                        "source_dataset": source,
                        "source_route": route,
                        "transfer_strategy": strategy,
                        "adaptation_mode": mode,
                        "adaptation_fraction": fraction,
                        "missing_modality_method": method,
                        "deployment_condition": condition,
                        "split_id": split_id,
                        "seed": seed,
                        "model_contract": config.get("model_contract", "UNIVERSAL_WEATHER_V2"),
                    }
                    source_contract = config.get("source_datasets", {}).get(
                        source, {"weather_columns": contract.get("weather_columns", []), "target_unit": contract.get("target_unit")}
                    )
                    status, reason = _adjudicate(cell, contract, source_contract, canonical_source)
                    cell.update({"candidate_id": _id(cell), "status": status, "status_reason": reason})
                    cells.append(cell)
    if any(cell["status"] not in TERMINAL_STATES for cell in cells):
        raise RuntimeError("INVALID_CANDIDATE_TERMINAL_STATE")
    return cells


def build_executable_plan(cells: list[dict], *, remediation_items: list[dict]) -> dict:
    unresolved = [item for item in remediation_items if item.get("status") != "CLOSED_VERIFIED"]
    if unresolved:
        raise RuntimeError(f"UNRESOLVED_REQUIRED_ITEMS:{len(unresolved)}")
    invalid = [cell for cell in cells if cell.get("status") not in TERMINAL_STATES]
    if invalid:
        raise RuntimeError(f"INVALID_CANDIDATE_STATES:{len(invalid)}")
    jobs = [cell for cell in cells if cell["status"] == "EXECUTABLE"]
    return {
        "schema_version": "universal_weather_v2_remediation_plan_v1",
        "formal_run_ready": True,
        "unresolved_required_items": 0,
        "silent_skips": 0,
        "fallbacks": 0,
        "jobs": jobs,
        "disposition_counts": {state: sum(cell["status"] == state for cell in cells) for state in sorted(TERMINAL_STATES)},
    }


def apply_observed_split_eligibility(cells: list[dict], split_registry: list[dict], config: dict) -> list[dict]:
    """Close sample-size legality using split facts without changing a split."""
    rules = config.get("eligibility", {})
    facts = {(item["dataset_id"], item["split_id"]): item for item in split_registry}
    for cell in cells:
        if cell["status"] != "EXECUTABLE":
            continue
        fact = facts.get((cell["dataset_id"], cell["split_id"]))
        if fact is None:
            cell.update(status="NOT_APPLICABLE_BY_DATA_CONTRACT", status_reason="SPLIT_FACTS_UNAVAILABLE")
            continue
        limits = (
            ("train_rows", rules.get("minimum_train_rows", 1), "MINIMUM_TRAIN_ROWS_NOT_MET"),
            ("validation_rows", rules.get("minimum_validation_rows", 1), "MINIMUM_VALIDATION_ROWS_NOT_MET"),
            ("test_rows", rules.get("minimum_test_rows", 1), "MINIMUM_TEST_ROWS_NOT_MET"),
            ("train_target_unique", rules.get("minimum_unique_train_targets", 2), "TARGET_DISTRIBUTION_COVERAGE_NOT_MET"),
        )
        failed = next((reason for field, minimum, reason in limits if fact.get(field, 0) < minimum), None)
        if failed:
            cell.update(status="NOT_APPLICABLE_BY_DATA_CONTRACT", status_reason=failed)
            continue
        if cell["adaptation_mode"] == "few_shot_adaptation":
            available = int(np.ceil(fact["train_rows"] * float(cell["adaptation_fraction"])))
            if available < rules.get("minimum_adaptation_rows", 1):
                cell.update(status="NOT_APPLICABLE_BY_DATA_CONTRACT", status_reason="MINIMUM_ADAPTATION_ROWS_NOT_MET")
    return cells
