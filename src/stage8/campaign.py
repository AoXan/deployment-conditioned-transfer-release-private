from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import canonical_sha256, file_sha256
from .schema import validate_json_schema
from .forbidden_paths import assert_path_allowed


class CampaignError(ValueError):
    pass


PLANNING_STATES = {"READY", "BLOCKED", "CONDITIONAL", "SKIPPED"}
KINDS = {
    "audit",
    "data_rebuild",
    "metric_recompute",
    "baseline_train",
    "gate",
    "route",
    "prospective_score",
    "acceptance",
    "physical_replay",
    "summary",
    "decision",
}


def load_campaign(path: Path) -> dict[str, Any]:
    assert_path_allowed(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"campaign YAML must be a mapping: {path}")
    schema_value = payload.get("campaign", {}).get("schema")
    schema_errors: list[str] = []
    if schema_value:
        schema_path = Path(schema_value)
        if not schema_path.is_absolute():
            schema_path = path.resolve().parents[1] / schema_path
        if not schema_path.is_file():
            schema_errors.append(f"missing_schema:{schema_path}")
        else:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema_errors.extend(validate_json_schema(payload, schema))
    else:
        schema_errors.append("missing_campaign_schema_reference")
    supplemental_errors: list[str] = []
    if payload.get("schema_version") == "stage8-supplemental-v1":
        from .supplemental import validate_supplemental

        supplemental_errors = validate_supplemental(payload)
    errors = [*schema_errors, *validate_campaign(payload), *supplemental_errors]
    if errors:
        raise CampaignError("; ".join(errors))
    payload["_config_path"] = str(path.resolve())
    return payload


def validate_campaign(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("schema_version", "campaign", "phases", "resources", "jobs", "outputs"):
        if key not in payload:
            errors.append(f"missing_required:{key}")
    if errors:
        return errors
    if not isinstance(payload["phases"], list) or not payload["phases"]:
        errors.append("phases_must_be_nonempty_list")
    if not isinstance(payload["jobs"], list) or not payload["jobs"]:
        errors.append("jobs_must_be_nonempty_list")
        return errors
    phase_ids = [str(row.get("id", "")) for row in payload["phases"]]
    if len(phase_ids) != len(set(phase_ids)) or "" in phase_ids:
        errors.append("phase_ids_must_be_unique")
    job_ids = [str(row.get("id", "")) for row in payload["jobs"]]
    if len(job_ids) != len(set(job_ids)) or "" in job_ids:
        errors.append("job_ids_must_be_unique")
    known_jobs = set(job_ids)
    resource_profiles = set(payload.get("resources", {}))
    for job in payload["jobs"]:
        job_id = str(job.get("id", ""))
        if job.get("phase") not in phase_ids:
            errors.append(f"unknown_phase:{job_id}:{job.get('phase')}")
        if job.get("kind") not in KINDS:
            errors.append(f"unknown_kind:{job_id}:{job.get('kind')}")
        if job.get("initial_state", "READY") not in PLANNING_STATES:
            errors.append(f"invalid_state:{job_id}")
        if not isinstance(job.get("command"), list) or not job.get("command"):
            errors.append(f"missing_command:{job_id}")
        if job.get("resource_profile") not in resource_profiles:
            errors.append(f"unknown_resource_profile:{job_id}:{job.get('resource_profile')}")
        if job.get("initial_state") == "CONDITIONAL" and not job.get("gate"):
            errors.append(f"conditional_without_gate:{job_id}")
        if job.get("route") and job.get("route") not in payload.get("route_protocols", {}):
            errors.append(f"missing_route_protocol:{job_id}:{job.get('route')}")
        for dep in job.get("depends_on", []):
            if dep not in known_jobs:
                errors.append(f"unknown_dependency:{job_id}:{dep}")
        if job.get("kind") == "metric_recompute" and job.get("requires_training"):
            errors.append(f"metric_only_requires_training:{job_id}")
        if job.get("kind") == "metric_recompute" and "train" in set(job.get("capabilities", [])):
            errors.append(f"metric_only_training_capability:{job_id}")
        if job.get("deployability") not in {None, "deployable_baseline", "deployable_ensemble", "test_oracle"}:
            errors.append(f"invalid_deployability:{job_id}")
    return errors


def _topological_order(jobs: list[dict[str, Any]]) -> list[str]:
    by_id = {str(job["id"]): job for job in jobs}
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(job_id: str) -> None:
        if job_id in visiting:
            raise CampaignError(f"dependency cycle detected at {job_id}")
        if job_id in visited:
            return
        visiting.add(job_id)
        for dep in by_id[job_id].get("depends_on", []):
            if dep not in by_id:
                raise CampaignError(f"unknown dependency {dep} for {job_id}")
            visit(str(dep))
        visiting.remove(job_id)
        visited.add(job_id)
        ordered.append(job_id)

    for job_id in by_id:
        visit(job_id)
    return ordered


def build_plan(
    campaign: dict[str, Any],
    root: Path,
    *,
    from_phase: str | None = None,
    only_job: str | None = None,
    gate_results: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    errors = validate_campaign(campaign)
    if errors:
        raise CampaignError("; ".join(errors))
    gate_results = gate_results or {}
    jobs = campaign["jobs"]
    by_id = {str(row["id"]): row for row in jobs}
    order = _topological_order(jobs)
    phase_order = [str(row["id"]) for row in campaign["phases"]]
    if from_phase and from_phase not in phase_order:
        raise CampaignError(f"unknown phase: {from_phase}")
    phase_floor = phase_order.index(from_phase) if from_phase else 0
    if only_job and only_job not in by_id:
        raise CampaignError(f"unknown job: {only_job}")

    selected: set[str] = set(order)
    if only_job:
        selected = set()

        def add_with_dependencies(job_id: str) -> None:
            if job_id in selected:
                return
            selected.add(job_id)
            for dep in by_id[job_id].get("depends_on", []):
                add_with_dependencies(str(dep))

        add_with_dependencies(only_job)

    rows: list[dict[str, Any]] = []
    state_by_id: dict[str, str] = {}
    config_for_hash = {key: value for key, value in campaign.items() if not key.startswith("_")}
    config_sha = canonical_sha256(config_for_hash)
    runtime_cache: dict[str, Any] = {}
    for job_id in order:
        job = copy.deepcopy(by_id[job_id])
        enabled = bool(job.get("enabled", True))
        state = str(job.get("initial_state", "READY"))
        if not enabled or job_id not in selected or phase_order.index(str(job["phase"])) < phase_floor:
            state = "SKIPPED"
        failed_dependencies = [dep for dep in job.get("depends_on", []) if state_by_id.get(str(dep)) in {"BLOCKED", "SKIPPED"}]
        if failed_dependencies and state == "READY":
            state = "BLOCKED"
            job["blocked_reason"] = f"dependency_not_runnable:{'|'.join(failed_dependencies)}"
        gate = job.get("gate")
        if gate:
            if gate in gate_results:
                if gate_results[gate]:
                    if state == "CONDITIONAL":
                        state = "READY"
                else:
                    state = "SKIPPED"
                    job["blocked_reason"] = f"gate_failed:{gate}"
            elif state != "SKIPPED":
                state = "CONDITIONAL"
        command_script = next((Path(value) for value in job.get("command", []) if str(value).endswith(".py")), None)
        if command_script and not command_script.is_absolute():
            command_script = root / command_script
        declared_hashes = job.get("hashes", {})
        source_hashes: dict[str, str] = {}
        for hash_key, values in job.get("hash_sources", {}).items():
            paths = values if isinstance(values, list) else [values]
            digests = []
            for value in paths:
                assert_path_allowed(value)
                source = Path(value)
                if not source.is_absolute(): source = root / source
                if not source.is_file():
                    digests = []
                    break
                digests.append(file_sha256(source))
            if digests: source_hashes[str(hash_key)] = canonical_sha256(digests)
        effective_hashes = {
            "code": file_sha256(command_script) if command_script and command_script.is_file() else "",
            "config": config_sha,
            "data": source_hashes.get("data", str(declared_hashes.get("data", ""))),
            "view": source_hashes.get("view", str(declared_hashes.get("view", ""))),
            "split": source_hashes.get("split", str(declared_hashes.get("split", ""))),
            "feature": source_hashes.get("feature", str(declared_hashes.get("feature", ""))),
        }
        runtime_profile = str(job.get("runtime_profile", ""))
        if payload_is_supplemental := (campaign.get("schema_version") == "stage8-supplemental-v1"):
            from .runtime import RuntimeBlocked, resolve_job_runtime
            try:
                if runtime_profile not in runtime_cache:
                    runtime_cache[runtime_profile] = resolve_job_runtime(campaign, job, root=root)
                resolved_runtime = runtime_cache[runtime_profile]
            except RuntimeBlocked as exc:
                raise CampaignError(str(exc)) from exc
            runtime_payload = {"profile": resolved_runtime.profile, "interpreter": str(resolved_runtime.interpreter), "fingerprint": resolved_runtime.fingerprint}
        else:
            runtime_payload = None
        fingerprint_payload = {
            "config_sha256": config_sha,
            "job": {key: value for key, value in job.items() if key not in {"state", "will_execute"}},
            "effective_hashes": effective_hashes,
            "runtime": runtime_payload,
        }
        fingerprint = canonical_sha256(fingerprint_payload)
        output_root = Path(campaign["outputs"]["root"])
        row = {
            **job,
            "job_id": job_id,
            "state": state,
            "will_execute": state == "READY",
            "fingerprint": fingerprint,
            "hashes": effective_hashes,
            "hash_reuse_ready": all(effective_hashes.get(key) for key in campaign.get("hash_policy", {}).get("required_for_artifact_reuse", [])),
            "runtime": runtime_payload,
            "completion_marker": str(output_root / "completion_markers" / f"{job_id}.json"),
            "log_path": str(output_root / "logs" / f"{job_id}.log"),
            "status_path": str(output_root / "status" / f"{job_id}.json"),
        }
        rows.append(row)
        state_by_id[job_id] = state
    return rows


def deterministic_plan_payload(plan: list[dict[str, Any]]) -> dict[str, Any]:
    return {"plan_sha256": canonical_sha256(plan), "job_count": len(plan), "jobs": plan}
