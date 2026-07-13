from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import numpy as np

from .artifacts import atomic_json, fingerprint
from .harmonisation import harmonise_frame
from .inventory import inventory_datasets
from .methods import recover_method_inventory
from .planner import build_candidate_matrix
from .splits import build_frozen_regional_splits, build_regional_splits, build_roseworthy_splits, build_waite_splits
from .recovery import recovery_contract


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if value.get("extends"):
        parent = yaml.safe_load((path.parents[1] / value["extends"]).read_text())
        parent.update({k: v for k, v in value.items() if k != "extends"})
        value = parent
    return value


def _splits(frame: pd.DataFrame, spec: dict[str, Any]) -> list[dict[str, Any]]:
    builder = spec["split_builder"]
    if builder == "waite": return build_waite_splits(frame.rename(columns={"target_yield": "observed_yield_t_ha"}), crop=spec.get("crop", "wheat"))
    if builder == "roseworthy": return build_roseworthy_splits(frame.rename(columns={"target_yield": "target_value", "crop": "crop_product"}))
    if builder == "regional" and spec.get("fold_glob"):
        return build_frozen_regional_splits(frame, list(Path(spec["fold_glob"]).parent.glob(Path(spec["fold_glob"]).name)))
    if builder == "regional": return build_regional_splits(frame)
    raise ValueError(f"UNKNOWN_SPLIT_BUILDER:{builder}")


def plan_campaign(config: dict[str, Any]) -> dict[str, Any]:
    formal_root = Path(config["formal_root"])
    methods = recover_method_inventory(formal_root)
    profiles = inventory_datasets(config)
    datasets = []
    split_registry: dict[str, list[dict[str, Any]]] = {}
    feature_registry: dict[str, dict[str, list[str]]] = {}
    for profile in profiles:
        spec = config["datasets"][profile["dataset_id"]]
        dataset = {**spec, **profile, "exact_compatible_components": []}
        datasets.append(dataset)
        if profile["contract_status"] != "CONTRACT_VALID":
            split_registry[profile["dataset_id"]] = []
            continue
        raw = pd.read_csv(spec["view"])
        frame, features = harmonise_frame(raw, spec)
        split_registry[profile["dataset_id"]] = _splits(frame, spec)
        feature_registry[profile["dataset_id"]] = features
    candidates = build_candidate_matrix(datasets=datasets, methods=methods)
    jobs = []
    transfer_components = {"fine_tune", "prediction_kd", "representation_kd", "combined_kd", "random_representation_transfer_control"}
    sources = {"waite": ["regional_stage6", "regional_stage7"], "roseworthy": ["regional_stage6", "regional_stage7", "waite"], "regional_stage6": ["regional_stage7"], "regional_stage7": ["regional_stage6"]}
    for candidate in candidates["candidates"]:
        dataset_id = candidate["dataset_id"]
        spec = config["datasets"][dataset_id]
        method = next(item for item in methods if item["component_id"] == candidate["component_id"])
        stochastic = method.get("model_family") in {"mlp", "neural"} or any(token in candidate["component_id"] for token in ("kd", "neural", "fusion", "dropout", "representation", "missing_aware", "fine_tune"))
        seeds = config["runtime"]["stochastic_seeds"] if stochastic else config["runtime"]["deterministic_seeds"]
        for split in split_registry[dataset_id]:
            source_options = sources.get(dataset_id, []) if candidate["component_id"] in transfer_components else [None]
            fractions = [0.01, 0.05, 0.1, 0.2, 1.0] if source_options != [None] else [1.0]
            for source in source_options:
                for fraction in fractions:
                    selected = max(1, int(len(split["train_ids"]) * fraction))
                    distribution_ok = all(split.get("target_unique", {}).get(part, 0) >= 2 for part in ("train", "validation", "test"))
                    budget_ok = selected >= int(config["execution"]["minimum_train"]) and len(split["validation_ids"]) >= int(config["execution"]["minimum_validation"]) and len(split["test_ids"]) >= int(config["execution"]["minimum_test"]) and distribution_ok
                    for seed in seeds:
                        payload = {"candidate_id": candidate["candidate_id"], "dataset_id": dataset_id, "source_dataset_id": source, "component_id": candidate["component_id"], "fold": split["fold"], "axis": split["axis"], "seed": seed, "label_budget": fraction}
                        split_protocol = {"DIAGNOSTIC_PROTOCOL": "DIAGNOSTIC", "TRANSFER_PROTOCOL": "TRANSFER", "ADAPTED_PROTOCOL": "ADAPTED", "EXACT_PROTOCOL": "EXACT"}.get(split.get("role"), candidate["planned_protocol"])
                        licence_ok = spec.get("licence_status") not in {"REQUIRES_FORMAL_CONFIRMATION", "LICENCE_BLOCKED"}
                        status = "CONFIGURED_NOT_RUN" if budget_ok and licence_ok else ("BLOCKED_LICENCE_CONFIRMATION" if not licence_ok else "BLOCKED_INSUFFICIENT_LABEL_BUDGET")
                        reason = None if status == "CONFIGURED_NOT_RUN" else ("ROSEWORTHY_PUBLICATION_AND_COMPUTATION_PERMISSION_NOT_PROVEN" if not licence_ok else f"TRAIN_{selected}_VALIDATION_{len(split['validation_ids'])}_TEST_{len(split['test_ids'])}_TARGET_COVERAGE_{distribution_ok}_BELOW_PREDECLARED_MINIMUM")
                        execution_phase = "teachers_baselines" if candidate.get("source_phase") == "phase3" else ("controls_ablations" if candidate.get("source_phase") in {"phase5", "australian_adaptation"} else "routes")
                        jobs.append({**payload, "job_id": fingerprint(payload)[:20], "execution_phase": execution_phase, "planned_protocol": "TRANSFER" if source else split_protocol, "sample_unit": spec["sample_unit"], "target_unit": spec["target_unit"], "status": status, "block_reason": reason})
    runtime_records = []
    artifact_bytes = []
    for path in formal_root.glob("**/job_record.json"):
        try:
            value = json.loads(path.read_text())
            if value.get("fit_time_seconds") is not None:
                runtime_records.append(float(value["fit_time_seconds"]))
            artifact_bytes.append(sum(item.stat().st_size for item in path.parent.rglob("*") if item.is_file()))
        except (OSError, ValueError, TypeError):
            continue
    runnable = sum(job["status"] == "CONFIGURED_NOT_RUN" for job in jobs)
    median = float(np.median(runtime_records)) if runtime_records else None
    p90 = float(np.quantile(runtime_records, .9)) if runtime_records else None
    byte_median = float(np.median(artifact_bytes)) if artifact_bytes else None
    resource_estimate = {"source": str(formal_root), "source_job_count": len(runtime_records), "fit_seconds_median": median, "fit_seconds_p90": p90, "planned_job_count": len(jobs), "runnable_job_count": runnable, "serial_fit_hours_at_median": median * runnable / 3600 if median else None, "serial_fit_hours_at_p90": p90 * runnable / 3600 if p90 else None, "artifact_bytes_per_source_job_median": byte_median, "projected_storage_bytes_at_source_median": byte_median * runnable if byte_median else None, "runtime_evidence_status": "EMPIRICAL_PHASE3_TO_5_LEDGER_EXTRAPOLATION_NOT_A_WALL_CLOCK_GUARANTEE" if runtime_records else "RUNTIME_EVIDENCE_INCOMPLETE", "cpu_profile": "1_PROCESS_1_THREAD_DEFAULT", "gpu_profile": "PYTORCH_JOBS_OPTIONAL_CPU_OR_MPS"}
    risks = [
        {"risk_id": "AU-R1", "dataset": "roseworthy", "status": "OPEN_EXTERNAL", "evidence": "config.datasets.roseworthy.licence_status=REQUIRES_FORMAL_CONFIRMATION", "impact": "Formal use requires licence confirmation; existing local view remains diagnostic."},
        {"risk_id": "AU-R2", "dataset": "roseworthy", "status": "OPEN_ENGINEERING", "evidence": "Existing view contains failed/partial weather provenance; weather recovery contract is explicit.", "impact": "Weather-dependent routes may block or remain spatial diagnostics."},
        {"risk_id": "AU-R3", "dataset": "all_transfer", "status": "DATA_DEPENDENT", "evidence": "Shared-feature intersection is evaluated per source-target pair before training.", "impact": "No shared deployable feature produces typed BLOCKED_DATA_CONTRACT, never row alignment."},
        {"risk_id": "AU-R4", "dataset": "regional_stage6_stage7", "status": "CLOSED_BY_SEPARATION", "evidence": "Separate dataset contracts and namespaces; no concatenation.", "impact": "Lineage duplicates cannot inflate a pooled evaluation."},
    ]
    dag = [
        {"phase": "inventory", "depends_on": []},
        {"phase": "provenance_license", "depends_on": ["inventory"]},
        {"phase": "raw_recovery", "depends_on": ["provenance_license"]},
        {"phase": "harmonisation", "depends_on": ["raw_recovery"]},
        {"phase": "split_construction", "depends_on": ["harmonisation"]},
        {"phase": "method_recovery", "depends_on": ["split_construction"]},
        {"phase": "eligibility_recovery_ladder", "depends_on": ["method_recovery"]},
        {"phase": "teachers_baselines", "depends_on": ["eligibility_recovery_ladder"]},
        {"phase": "routes", "depends_on": ["teachers_baselines"]},
        {"phase": "controls_ablations", "depends_on": ["routes"]},
        {"phase": "acceptance", "depends_on": ["controls_ablations"]},
        {"phase": "aggregation_report", "depends_on": ["acceptance"]},
    ]
    return {"schema_version": "stage8_v4_australian_plan_v1", "methods": methods, "datasets": datasets, "candidates": candidates, "jobs": jobs, "splits": split_registry, "features": feature_registry, "dag": dag, "resource_estimate": resource_estimate, "risk_registry": risks, "plan_fingerprint": fingerprint({"candidates": candidates, "jobs": jobs, "dag": dag})}


def write_plan(plan: dict[str, Any], root: Path) -> None:
    control = root / "control"
    atomic_json(control / "method_inventory.json", plan["methods"])
    atomic_json(control / "exhaustive_candidate_matrix.json", plan["candidates"])
    atomic_json(control / "executable_job_matrix.json", {"jobs": plan["jobs"], "plan_fingerprint": plan["plan_fingerprint"]})
    atomic_json(control / "split_registry.json", plan["splits"])
    atomic_json(control / "dataset_inventory.json", plan["datasets"])
    atomic_json(control / "resource_estimate.json", plan["resource_estimate"])
    atomic_json(control / "risk_registry.json", plan["risk_registry"])
    atomic_json(control / "dag.json", plan["dag"])
    recovery = []
    for candidate in plan["candidates"]["candidates"]:
        for order, attempt in enumerate(candidate["recovery_attempts"], start=1):
            recovery.append({"candidate_id": candidate["candidate_id"], "dataset_id": candidate["dataset_id"], "component_id": candidate["component_id"], "attempt_order": order, "attempt": attempt, "status": "PLANNED_NOT_RUN", "terminal_proven_untestable_allowed": order == len(candidate["recovery_attempts"])})
    atomic_json(control / "recovery_attempt_registry.json", recovery)
    with (control / "recovery_attempt_registry.jsonl").open("w") as handle:
        for row in recovery: handle.write(json.dumps(row, sort_keys=True) + "\n")
    for name, rows in (("exhaustive_candidate_matrix.csv", plan["candidates"]["candidates"]), ("executable_job_matrix.csv", plan["jobs"])):
        path = control / name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}), extrasaction="ignore", lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
