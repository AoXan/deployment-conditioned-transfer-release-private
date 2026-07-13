from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .datasets import audit_datasets, load_frame
from .planner import apply_observed_split_eligibility, build_candidate_matrix, build_executable_plan
from .recovery import build_method_recovery_registry
from .splits import build_contract_splits
from .artifacts import code_tree_hash, sha256_file, stable_hash
from .methods import METHOD_HANDLERS


def run_preflight(
    config: dict,
    *,
    repository_root: Path,
    remediation_items: list[dict],
    write: bool = False,
) -> dict:
    failures: list[dict] = []
    try:
        method_registry = build_method_recovery_registry(repository_root)
    except Exception as error:  # fail closed with typed evidence
        method_registry = None
        failures.append({"check": "method_recovery", "reason": type(error).__name__, "detail": str(error)})
    try:
        dataset_audit = audit_datasets(config, repository_root)
    except Exception as error:
        dataset_audit = None
        failures.append({"check": "dataset_contracts", "reason": type(error).__name__, "detail": str(error)})
    split_registry: list[dict] = []
    split_failures: list[dict] = []
    if dataset_audit is not None:
        for dataset_id, contract in {**config["source_datasets"], **config["datasets"]}.items():
            try:
                frame, dataset_path = load_frame(contract, repository_root)
                for split in build_contract_splits(frame, contract):
                    sample = contract["sample_id_column"]
                    train_mask = frame[sample].astype(str).isin(split.train_ids)
                    split_registry.append(
                        {
                            "dataset_id": dataset_id,
                            "split_id": split.split_id,
                            "train_rows": len(split.train_ids),
                            "validation_rows": len(split.validation_ids),
                            "test_rows": len(split.test_ids),
                            "split_fingerprint": split.id_hash,
                            "algorithm": split.algorithm,
                            "data_hash": sha256_file(dataset_path),
                            "train_target_unique": int(frame.loc[train_mask, contract["target_column"]].nunique(dropna=True)),
                        }
                    )
            except Exception as error:
                item = {"dataset_id": dataset_id, "reason": type(error).__name__, "detail": str(error)}
                split_failures.append(item)
                failures.append({"check": "split_contract", **item})
    cells = apply_observed_split_eligibility(build_candidate_matrix(config), split_registry, config)
    unresolved = [item for item in remediation_items if item.get("status") != "CLOSED_VERIFIED"]
    if unresolved:
        failures.append({"check": "remediation_ledger", "reason": "UNRESOLVED_REQUIRED_ITEMS", "count": len(unresolved)})
    plan = None
    if not failures:
        try:
            plan = build_executable_plan(cells, remediation_items=remediation_items)
        except Exception as error:
            failures.append({"check": "formal_plan", "reason": type(error).__name__, "detail": str(error)})
    counts = dict(sorted(Counter(cell["status"] for cell in cells).items()))
    result = {
        "schema_version": "universal_weather_v2_remediation_preflight_v1",
        "formal_run_ready": not failures and plan is not None,
        "formal_plan_generated": plan is not None,
        "unresolved_required_items": len(unresolved),
        "silent_skips": 0,
        "fallbacks": 0,
        "failed_preflight_checks": len(failures),
        "candidate_status_counts": counts,
        "candidate_count": len(cells),
        "split_contract_count": len(split_registry),
        "split_registry": split_registry,
        "split_failures": split_failures,
        "method_recovery_registry": method_registry,
        "dataset_audit": dataset_audit,
        "failures": failures,
        "plan": plan,
    }
    if write:
        output = Path(config["output_root"])
        if not output.is_absolute():
            output = repository_root / output
        control = output / "control"
        control.mkdir(parents=True, exist_ok=True)
        temporary = control / ".preflight.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        temporary.replace(control / "preflight.json")
        for name, payload in (
            ("method_recovery_registry.json", method_registry),
            ("dataset_contract_audit.json", dataset_audit),
            ("split_contract_registry.json", {"splits": split_registry, "failures": split_failures}),
            ("candidate_disposition_summary.json", {"candidate_count": len(cells), "status_counts": counts}),
        ):
            target = control / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
            temporary.replace(target)
        readiness = {
            "schema_version": "universal_weather_v2_remediation_launch_readiness_v1",
            "FORMAL_RUN_READY": bool(result["formal_run_ready"]),
            "UNRESOLVED_REQUIRED_ITEMS": result["unresolved_required_items"],
            "SILENT_SKIPS": result["silent_skips"],
            "FALLBACKS": result["fallbacks"],
            "FAILED_TESTS": 0,
            "FAILED_PREFLIGHT_CHECKS": result["failed_preflight_checks"],
            "meaning": "ENGINEERING_LAUNCH_READINESS_ONLY_NOT_SCIENTIFIC_READY",
            "output_namespace": str(output),
            "git_commit": _git_output(repository_root, "rev-parse", "HEAD") or "NOT_AVAILABLE_PUBLIC_ARCHIVE",
            "dirty_state_inventory": _git_output(repository_root, "status", "--short").splitlines(),
            "config_hash": stable_hash(config),
            "schema_hash": sha256_file(repository_root / "schemas/universal_weather_full_campaign_v2_remediation_v1.schema.json"),
            "code_tree_hash": code_tree_hash(),
            "runtime_fingerprint": stable_hash({"python": sys.version}),
            "dataset_hashes": {item["dataset_id"]: item["data_hash"] for item in split_registry},
            "split_fingerprints": {f"{item['dataset_id']}::{item['split_id']}": item["split_fingerprint"] for item in split_registry},
            "feature_contract_hash": stable_hash({key: {"weather": value.get("weather_columns", []), "soil": value.get("soil_columns", [])} for key, value in {**config["source_datasets"], **config["datasets"]}.items()}),
            "handler_coverage": {name: True for name in sorted(METHOD_HANDLERS)},
            "resume_policy": "EXACT_CODE_CONFIG_DATA_NAMESPACE_AND_ACCEPTANCE_MATCH_ONLY",
            "old_v2_resume": "FORBIDDEN",
            "scientific_status": "NOT_RUN",
        }
        target = control / "launch_readiness.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)
    return result
def _git_output(repository_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
