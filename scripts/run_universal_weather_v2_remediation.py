#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation_v4.universal_weather_v2_remediation.config import load_config
from src.distillation_v4.universal_weather_v2_remediation.datasets import audit_datasets
from src.distillation_v4.universal_weather_v2_remediation.execution import execute_target_job
from src.distillation_v4.universal_weather_v2_remediation.planner import apply_observed_split_eligibility, build_candidate_matrix, build_executable_plan
from src.distillation_v4.universal_weather_v2_remediation.preflight import run_preflight
from src.distillation_v4.universal_weather_v2_remediation.recovery import build_method_recovery_registry
from src.distillation_v4.universal_weather_v2_remediation.artifacts import code_tree_hash, sha256_file, stable_hash


SCHEMA = ROOT / "schemas/universal_weather_full_campaign_v2_remediation_v1.schema.json"


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _remediation_items(output: Path) -> list[dict]:
    path = output / "control" / "remediation_ledger.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reconcile_smoke_failures(output: Path) -> None:
    ledger = output / "control" / "remediation_ledger.jsonl"
    existing = _remediation_items(output)
    known = {item.get("failure_artifact") for item in existing}
    additions = []
    for failure_path in sorted((output / "failures").glob("*.json")):
        failure = json.loads(failure_path.read_text())
        job_id = failure.get("job_id")
        marker = output / "smoke" / "targets" / str(job_id) / "COMPLETED.json"
        if marker.is_file() and str(failure_path) not in known:
            additions.append(
                {
                    "id": f"SMOKE_REMEDIATION_{job_id}",
                    "status": "CLOSED_VERIFIED",
                    "failure_artifact": str(failure_path),
                    "completion_artifact": str(marker),
                    "root_cause": failure.get("detail"),
                }
            )
    if additions:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(json.dumps(item, sort_keys=True) + "\n" for item in [*existing, *additions])
        temporary = ledger.with_suffix(".tmp")
        temporary.write_text(content)
        temporary.replace(ledger)


def _status(output: Path) -> dict:
    completed = list(output.glob("jobs/targets/*/COMPLETED.json")) if output.exists() else []
    failures = list(output.glob("failures/*.json")) if output.exists() else []
    return {
        "schema_version": "universal_weather_v2_remediation_status_v1",
        "output_root": str(output),
        "completed_jobs": len(completed),
        "failed_jobs": len(failures),
        "executed_jobs": len(completed) + len(failures),
    }


def _resume_is_safe(job_directory: Path, config: dict, output: Path) -> bool:
    manifest_path = job_directory / "manifest.json"
    marker_path = job_directory / "COMPLETED.json"
    acceptance_path = job_directory / "acceptance.json"
    if not all(path.is_file() for path in (manifest_path, marker_path, acceptance_path)):
        return False
    manifest = json.loads(manifest_path.read_text())
    dataset_path = Path(manifest.get("dataset_path", ""))
    return bool(
        manifest.get("code_tree_hash") == code_tree_hash()
        and manifest.get("resolved_config_hash") == stable_hash(config)
        and manifest.get("namespace") == str(output)
        and dataset_path.is_file()
        and manifest.get("data_hash") == sha256_file(dataset_path)
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--output-root", type=Path)
    actions = value.add_mutually_exclusive_group(required=True)
    for flag in ("method-audit", "data-contract-audit", "build-candidates", "preflight", "build-plan", "dry-run", "run-formal", "status", "progress", "failures", "resources"):
        actions.add_argument(f"--{flag}", action="store_true")
    value.add_argument("--job")
    value.add_argument("--resume", action="store_true")
    value.add_argument(
        "--confirm-full-campaign",
        metavar="PLAN_FINGERPRINT",
        help="Exact fingerprint emitted by --dry-run; required for an unfiltered formal campaign.",
    )
    value.add_argument("--smoke", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config, SCHEMA)
    output = args.output_root or ROOT / config["output_root"]
    if not output.is_absolute():
        output = ROOT / output
    config["output_root"] = str(output)
    if args.run_formal and not args.job and not args.confirm_full_campaign:
        print("FORMAL_EXECUTION_CONFIRMATION_REQUIRED", file=sys.stderr)
        return 2
    if args.method_audit:
        print(json.dumps(build_method_recovery_registry(ROOT), indent=2, sort_keys=True)); return 0
    if args.data_contract_audit:
        print(json.dumps(audit_datasets(config, ROOT), indent=2, sort_keys=True)); return 0
    cells = build_candidate_matrix(config)
    counts = dict(sorted(Counter(cell["status"] for cell in cells).items()))
    if args.status or args.progress or args.failures:
        print(json.dumps(_status(output), indent=2, sort_keys=True)); return 0
    _reconcile_smoke_failures(output)
    preflight = run_preflight(config, repository_root=ROOT, remediation_items=_remediation_items(output), write=args.preflight or args.build_plan)
    if args.build_candidates:
        cells = apply_observed_split_eligibility(cells, preflight["split_registry"], config)
        counts = dict(sorted(Counter(cell["status"] for cell in cells).items()))
        payload = {"candidate_count": len(cells), "status_counts": counts, "candidates": cells}
        _write(output / "control" / "candidate_matrix.json", payload)
        print(json.dumps({key: payload[key] for key in ("candidate_count", "status_counts")}, sort_keys=True)); return 0
    if args.preflight:
        print(json.dumps({key: preflight[key] for key in ("formal_run_ready", "failed_preflight_checks", "candidate_count", "candidate_status_counts")}, indent=2, sort_keys=True))
        return 0 if preflight["formal_run_ready"] else 2
    if not preflight["formal_run_ready"]:
        print("FORMAL_RUN_NOT_READY", file=sys.stderr); return 2
    plan = preflight["plan"] or build_executable_plan(cells, remediation_items=[])
    fingerprint = _hash(plan["jobs"])
    if args.run_formal and not args.job and args.confirm_full_campaign != fingerprint:
        print("FORMAL_EXECUTION_FINGERPRINT_MISMATCH", file=sys.stderr)
        return 2
    if args.resources:
        executable = len(plan["jobs"])
        legacy = ROOT / "outputs/distillation_program/stage8_v4_australian_narrative_v1/universal_weather_full_campaign_v2"
        progress_path = legacy / "status/full_chain_progress.json"
        if progress_path.is_file():
            measured = json.loads(progress_path.read_text())
            throughput = float(measured.get("throughput_jobs_per_hour", 0.0))
            projected_hours = executable / throughput if throughput > 0 else None
            files = [path for path in legacy.rglob("*") if path.is_file()]
            old_bytes = sum(path.stat().st_size for path in files)
            old_jobs = int(measured.get("completed_jobs", 0))
            projected_bytes = int(old_bytes / old_jobs * executable) if old_jobs else None
            evidence = "MEASURED_V2_CAMPAIGN_EXTRAPOLATION_WITH_METHOD_MIX_UNCERTAINTY"
        else:
            throughput = projected_hours = projected_bytes = None
            evidence = "RUNTIME_EVIDENCE_INCOMPLETE"
        print(json.dumps({
            "executable_jobs": executable, "plan_fingerprint": fingerprint,
            "runtime_evidence_status": evidence,
            "measured_v2_throughput_jobs_per_hour": throughput,
            "projected_serial_wall_hours_at_measured_mix": projected_hours,
            "projected_storage_bytes_at_measured_mix": projected_bytes,
            "cpu": "required", "gpu": "optional_not_execution_condition",
            "caveat": "T/M method mix differs; refresh after first accepted jobs"
        }, indent=2)); return 0
    if args.build_plan:
        _write(output / "control" / "formal_job_plan.json", {**plan, "plan_fingerprint": fingerprint})
        print(json.dumps({"formal_run_ready": True, "job_count": len(plan["jobs"]), "plan_fingerprint": fingerprint}, sort_keys=True)); return 0
    if args.dry_run:
        print(json.dumps({"formal_run_ready": True, "candidate_count": len(cells), "executable_jobs": len(plan["jobs"]), "executed_jobs": 0, "plan_fingerprint": fingerprint}, sort_keys=True)); return 0
    if args.run_formal:
        selected = [job for job in plan["jobs"] if not args.job or job["candidate_id"] == args.job]
        if args.job and not selected:
            print(f"UNKNOWN_JOB:{args.job}", file=sys.stderr); return 2
        completed = 0
        for job in selected:
            job_directory = output / ("smoke" if args.smoke else "jobs") / "targets" / job["candidate_id"]
            marker = job_directory / "COMPLETED.json"
            if marker.is_file():
                if args.resume:
                    if _resume_is_safe(job_directory, config, output):
                        completed += 1
                        continue
                    print(f"RESUME_FINGERPRINT_MISMATCH:{job['candidate_id']}", file=sys.stderr)
                    return 2
                print(f"DUPLICATE_PROCESS_OR_COMPLETED_JOB:{job['candidate_id']}", file=sys.stderr)
                return 2
            running = output / "status" / "running" / f"{job['candidate_id']}.json"
            _write(running, {"job_id": job["candidate_id"], "started_unix": time.time(), "job": job})
            try:
                execute_target_job(config, job, repository_root=ROOT, output_root=output, smoke=args.smoke)
                completed += 1
            except Exception as error:
                failure = {
                    "job_id": job["candidate_id"], "type": type(error).__name__, "detail": str(error),
                    "retryable": True, "job": job, "timestamp_unix": time.time(),
                }
                _write(output / "failures" / f"{job['candidate_id']}.{int(time.time())}.json", failure)
                print(json.dumps(failure, sort_keys=True), file=sys.stderr)
                return 2
            finally:
                running.unlink(missing_ok=True)
        print(json.dumps({"executed_jobs": completed, "formal": not args.smoke, "smoke_only_not_scientific_evidence": args.smoke}, sort_keys=True)); return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
