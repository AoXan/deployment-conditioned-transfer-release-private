#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Compatibility alias retained because the existing CLI uses ROOT for
# schema, repository and output-path resolution.
ROOT = REPOSITORY_ROOT
APPROVED_OUTPUT_NAMESPACES = {
    "universal_weather_full_campaign_v2_remediation_dag_v1",
    "universal_weather_full_campaign_v2_parallel_v1",
}

import argparse
import json
import signal

from src.distillation_v4.universal_weather_v2_remediation.config import (
    load_config,
)
from src.distillation_v4.universal_weather_v2_remediation.formal_dispatcher import (
    IMMUTABLE_CORE_JOB_MANIFEST,
    run_formal_campaign,
)
from src.distillation_v4.universal_weather_v2_remediation.runtime_control import (
    append_failure,
    exact_resume_decision,
    request_stop,
    stop_requested,
    write_heartbeat,
)

from typing import Sequence

import yaml
from jsonschema import Draft202012Validator

from src.distillation_v4.universal_weather_v2_remediation.dag_v2 import (
    write_dag_v2_outputs,
)

SCHEMA = (
    ROOT
    / "schemas/universal_weather_full_campaign_v2_remediation_dag_v1.schema.json"
)


FORMAL_RUNTIME_SAFETY_CONTRACTS = {
    "exact_resume_decision": exact_resume_decision,
    "write_heartbeat": write_heartbeat,
    "stop_requested": stop_requested,
    "append_failure": append_failure,
    "immutable_core_job_manifest.json": IMMUTABLE_CORE_JOB_MANIFEST,
}


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate Universal Weather V2 remediation DAG."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--build-plan", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--resources", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--run-formal", action="store_true")

    parser.add_argument("--confirm-full-campaign")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--source-workers", type=int, default=1)
    parser.add_argument("--target-workers", type=int, default=1)
    parser.add_argument("--source-worker-threads", type=int, default=1)
    parser.add_argument("--target-worker-threads", type=int, default=1)
    parsed = parser.parse_args(argv)
    for field in (
        "source_workers",
        "target_workers",
        "source_worker_threads",
        "target_worker_threads",
    ):
        if getattr(parsed, field) < 1:
            parser.error(f"{field.replace('_', '-')} must be > 0")
    return parsed


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    schema = json.loads(SCHEMA.read_text())
    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda item: list(item.path),
    )
    if errors:
        detail = "; ".join(
            f"{list(error.path)}: {error.message}"
            for error in errors
        )
        raise ValueError(f"CONFIG_SCHEMA_INVALID:{detail}")

    if (
        Path(config["output_root"]).name
        != "universal_weather_full_campaign_v2_remediation_dag_v1"
    ):
        raise ValueError("REMEDIATION_DAG_NAMESPACE_REQUIRED")

    for field in ("execution_config", "execution_schema"):
        path_value = ROOT / config[field]
        if not path_value.is_file():
            raise FileNotFoundError(
                f"BOUND_EXECUTION_CONTRACT_MISSING:{path_value}"
            )

    return config


def _resolve_output(config: dict, override: Path | None) -> Path:
    output = override or ROOT / config["output_root"]
    if not output.is_absolute():
        output = ROOT / output

    if output.name not in APPROVED_OUTPUT_NAMESPACES and "pytest-" not in str(output):
        raise ValueError("REMEDIATION_DAG_NAMESPACE_REQUIRED")

    return output


def _build(config: dict, output: Path) -> dict:
    return write_dag_v2_outputs(output, repository_root=ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    config = _load_config(args.config)
    output = _resolve_output(config, args.output_root)

    if args.status:
        manifest_path = output / "control/formal_preapproval_manifest.json"
        if not manifest_path.is_file():
            payload = {
                "status": "PLAN_NOT_BUILT",
                "formal_run_ready": False,
                "executed": 0,
            }
        else:
            manifest = json.loads(manifest_path.read_text())
            payload = {
                "status": (
                    "READY_FOR_FINAL_USER_APPROVAL"
                    if manifest["can_enter_final_user_approval"]
                    else "IMPLEMENTATION_REPAIR_REQUIRED"
                ),
                "formal_run_ready": (
                    manifest["can_enter_final_user_approval"]
                ),
                "blockers": manifest["blockers"],
                "exact_plan_fingerprint": (
                    manifest["exact_plan_fingerprint"]
                ),
                "executed": 0,
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    result = _build(config, output)

    if args.preflight:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["formal_run_ready"] else 2

    if args.build_plan:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.dry_run:
        print(
            json.dumps(
                {
                    **result,
                    "formal_jobs_executed": 0,
                    "scientific_status": "NOT_RUN",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.resources:
        summary = json.loads(
            (
                output
                / "control/final_eligibility_summary.json"
            ).read_text()
        )
        print(
            json.dumps(
                {
                    "runtime_evidence_status": (
                        "PENDING_HANDLER_COMPLETION"
                    ),
                    "core_executable_job_count": (
                        summary["core_executable_job_count"]
                    ),
                    "gated_t1_candidate_count": (
                        summary["gated_t1_candidate_count"]
                    ),
                    "gated_t1_released_count": 0,
                    "executed": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.run_formal:
        manifest = json.loads(
            (
                output
                / "control/formal_preapproval_manifest.json"
            ).read_text()
        )

        expected = manifest["exact_plan_fingerprint"]

        if args.confirm_full_campaign != expected:
            print(
                json.dumps(
                    {
                        "status": (
                            "FORMAL_EXECUTION_CONFIRMATION_REQUIRED"
                        ),
                        "required_plan_fingerprint": expected,
                        "executed": 0,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

        if not manifest["can_enter_final_user_approval"]:
            print(
                json.dumps(
                    {
                        "status": "FORMAL_EXECUTION_BLOCKED",
                        "blockers": manifest["blockers"],
                        "executed": 0,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

        dag_payload = yaml.safe_load(
            Path(args.config).read_text()
        )
        execution_config_path = Path(
            dag_payload["execution_config"]
        )
        execution_schema_path = Path(
            dag_payload["execution_schema"]
        )

        execution_config = load_config(
            execution_config_path,
            execution_schema_path,
        )
        execution_config["output_root"] = str(output)

        def _request_operator_stop(
            signum: int,
            frame: object,
        ) -> None:
            del frame
            request_stop(
                output,
                reason=f"signal:{signum}",
            )

        signal.signal(
            signal.SIGINT,
            _request_operator_stop,
        )
        signal.signal(
            signal.SIGTERM,
            _request_operator_stop,
        )

        result = run_formal_campaign(
            config=execution_config,
            repository_root=ROOT,
            output_root=output,
            plan_fingerprint=expected,
            resume=True,
            parallel=args.parallel,
            source_workers=args.source_workers,
            target_workers=args.target_workers,
            source_worker_threads=args.source_worker_threads,
            target_worker_threads=args.target_worker_threads,
        )

        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )

        return (
            0
            if result["status"] == "COMPLETED"
            else 2
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
