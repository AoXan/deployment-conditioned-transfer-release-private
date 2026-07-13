from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .artifact_hooks import persist_replay_inputs
from .formal_bridge import FormalCodeBridge, ensure_source_dependency_alias, formal_source_dependency_path, write_json
from .guards import validate_output_root
from .mechanisms import run_mechanism_suite, write_final_synthesis
from .reproduction import compare_predictions, write_reproduction_record
from .runner_core import ensure_manifest_hash, expected_jobs_match, selected_formal_ids
from .status import stop_requested, write_heartbeat, write_status


def read_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _cell_for_formal_id(manifest: dict, formal_candidate_id: str) -> dict:
    for cell in [*manifest.get("cells", []), *manifest.get("baseline_cells", [])]:
        if formal_candidate_id in cell.get("all_seed_candidate_ids", []):
            return cell
    raise ValueError(f"FORMAL_ID_NOT_IN_MANIFEST:{formal_candidate_id}")


def _job_for_formal_id(cell: dict, formal_candidate_id: str) -> dict:
    for job in cell.get("formal_jobs", []):
        if str(job.get("candidate_id")) == formal_candidate_id:
            return job
    raise ValueError(f"FORMAL_JOB_NOT_IN_CELL:{formal_candidate_id}")


def _formal_prediction_path(formal_campaign: Path, candidate_id: str) -> Path:
    return Path(formal_campaign) / "jobs" / "targets" / candidate_id / "predictions.csv"


def _replay_prediction_path(output_root: Path, candidate_id: str) -> Path:
    return Path(output_root) / "jobs" / "targets" / candidate_id / "predictions.csv"


def validate_manifest_for_run(*, manifest: dict, formal_campaign: Path, output_root: Path, writable_root: Path) -> dict:
    output = validate_output_root(output_root=output_root, formal_campaign=formal_campaign, writable_root=writable_root)
    bridge = FormalCodeBridge()
    config_status = {"status": "NOT_CHECKED"}
    try:
        config = bridge.load_config()
        execution = bridge.execution_module()
        config_status = {
            "status": "PASS",
            "schema_version": config.get("schema_version"),
            "dataset_count": len(config.get("datasets", {})),
            "source_dataset_count": len(config.get("source_datasets", {})),
            "execution_module": str(Path(execution.__file__).resolve()),
            "formal_config": str(bridge.formal_config),
            "formal_schema": str(bridge.formal_schema),
        }
    except Exception as error:
        config_status = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
    missing_formal = [
        candidate_id
        for candidate_id in manifest.get("all_formal_candidate_ids", [])
        if not _formal_prediction_path(formal_campaign, candidate_id).is_file()
    ]
    missing_source_dependencies = []
    for cell in manifest.get("cells", []):
        for job in cell.get("formal_jobs", []):
            if str(job.get("transfer_strategy")) == "target_scratch":
                continue
            path = formal_source_dependency_path(formal_campaign, job)
            if not path.is_dir():
                missing_source_dependencies.append(
                    {
                        "candidate_id": str(job.get("candidate_id")),
                        "source_dependency": str(path),
                    }
                )
    return {
        "status": "PASS" if not missing_formal and not missing_source_dependencies and config_status["status"] == "PASS" else "FAIL",
        "output_root": str(output),
        "formal_campaign": str(Path(formal_campaign).resolve()),
        "expected_target_jobs": manifest.get("expected_target_jobs"),
        "formal_code_config": config_status,
        "missing_formal_predictions": missing_formal,
        "missing_source_dependencies": missing_source_dependencies,
    }


def run_pipeline(
    *,
    manifest: dict,
    formal_campaign: Path,
    output_root: Path,
    writable_root: Path,
    confirm_manifest_hash: str | None,
    expected_target_jobs: int | None,
    subset: str | None,
    single_job: str | None,
    no_explanations: bool,
    explanations_only: bool,
    resume: bool,
) -> dict:
    output = validate_output_root(output_root=output_root, formal_campaign=formal_campaign, writable_root=writable_root)
    ensure_manifest_hash(manifest, confirm_manifest_hash)
    expected_jobs_match(manifest, expected_target_jobs)
    ids = selected_formal_ids(manifest, subset=subset, single_job=single_job)

    preflight = validate_manifest_for_run(
        manifest=manifest,
        formal_campaign=formal_campaign,
        output_root=output,
        writable_root=writable_root,
    )
    write_json(output / "status" / "preflight_status.json", preflight)
    if preflight["status"] != "PASS":
        raise ValueError(
            "REPLAY_PREFLIGHT_FAILED:"
            f"formal_code_config={preflight.get('formal_code_config')};"
            f"missing_formal_predictions={len(preflight.get('missing_formal_predictions', []))};"
            f"missing_source_dependencies={len(preflight.get('missing_source_dependencies', []))}"
        )

    bridge = FormalCodeBridge()
    config = bridge.load_config()

    summary = {
        "started_unix": time.time(),
        "requested_jobs": len(ids),
        "replay_completed": 0,
        "replay_failed": 0,
        "reproduction_pass": 0,
        "reproduction_failed": 0,
        "reproduction_blocked": 0,
        "explanation_completed": 0,
        "explanation_failed": 0,
        "explanation_blocked": 0,
    }

    replay_status = {"pending": len(ids), "running": 0, "completed": 0, "failed": 0, "blocked": 0}
    reproduction_status = {"pending": len(ids), "pass": 0, "failed": 0, "blocked": 0}
    explanation_status = {"pending": 0 if no_explanations else len(ids), "completed": 0, "failed": 0, "blocked": 0, "not_applicable": 0}
    write_status(output, "replay", replay_status)
    write_status(output, "reproduction", reproduction_status)
    write_status(output, "explanation", explanation_status)

    for candidate_id in ids:
        if stop_requested(output):
            summary["stopped"] = True
            break
        write_heartbeat(output, current_job=candidate_id, stage="start")
        cell = _cell_for_formal_id(manifest, candidate_id)
        job = _job_for_formal_id(cell, candidate_id)
        replay_prediction = _replay_prediction_path(output, candidate_id)
        try:
            if explanations_only:
                replay_status["pending"] = max(0, replay_status["pending"] - 1)
            elif replay_prediction.is_file() and resume:
                replay_status["completed"] += 1
                replay_status["pending"] = max(0, replay_status["pending"] - 1)
            else:
                replay_status["running"] = 1
                write_status(output, "replay", replay_status)
                ensure_source_dependency_alias(formal_campaign=formal_campaign, output_root=output, job=job)
                capture = persist_replay_inputs(bridge=bridge, config=config, job=job, output_root=output)
                write_json(output / "jobs" / "replay_artifacts" / candidate_id / "artifact_capture_status.json", capture)
                bridge.execute_target_job(config=config, job=job, output_root=output, smoke=False)
                replay_status["running"] = 0
                replay_status["completed"] += 1
                replay_status["pending"] = max(0, replay_status["pending"] - 1)
                summary["replay_completed"] += 1
                write_status(output, "replay", replay_status)
        except Exception as error:
            replay_status["running"] = 0
            replay_status["failed"] += 1
            replay_status["pending"] = max(0, replay_status["pending"] - 1)
            summary["replay_failed"] += 1
            write_status(output, "replay", replay_status)
            write_json(
                output / "failures" / f"{candidate_id}.replay.json",
                {"stage": "replay", "candidate_id": candidate_id, "error_type": type(error).__name__, "error": str(error)},
            )
            continue

        if not replay_prediction.is_file():
            reproduction_status["blocked"] += 1
            reproduction_status["pending"] = max(0, reproduction_status["pending"] - 1)
            summary["reproduction_blocked"] += 1
            write_status(output, "reproduction", reproduction_status)
            continue

        formal_prediction = _formal_prediction_path(formal_campaign, candidate_id)
        reproduction_record = compare_predictions(
            formal_predictions=formal_prediction,
            replay_predictions=replay_prediction,
        )
        write_reproduction_record(output, candidate_id, reproduction_record)
        if reproduction_record["reproduction_status"] == "PASS":
            reproduction_status["pass"] += 1
            summary["reproduction_pass"] += 1
            # Store a small alignment copy for downstream provenance.
            alignment_dir = output / "reproduction" / "alignment"
            alignment_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(replay_prediction, alignment_dir / f"{candidate_id}.replay_predictions.csv")
        else:
            reproduction_status["failed"] += 1
            summary["reproduction_failed"] += 1
        reproduction_status["pending"] = max(0, reproduction_status["pending"] - 1)
        write_status(output, "reproduction", reproduction_status)

        if no_explanations:
            explanation_status["pending"] = max(0, explanation_status["pending"] - 1)
            explanation_status["blocked"] += 1
            write_status(output, "explanation", explanation_status)
            continue
        if reproduction_record["reproduction_status"] != "PASS":
            explanation_status["pending"] = max(0, explanation_status["pending"] - 1)
            explanation_status["blocked"] += 1
            summary["explanation_blocked"] += 1
            write_status(output, "explanation", explanation_status)
            continue
        mechanism_result = run_mechanism_suite(manifest, output, candidate_id)
        failed = any(value.get("status") == "failed" for value in mechanism_result.values())
        if failed:
            explanation_status["failed"] += 1
            summary["explanation_failed"] += 1
        else:
            explanation_status["completed"] += 1
            summary["explanation_completed"] += 1
        explanation_status["pending"] = max(0, explanation_status["pending"] - 1)
        write_status(output, "explanation", explanation_status)

    synthesis_path = write_final_synthesis(manifest, output)
    summary["synthesis_path"] = str(synthesis_path)
    summary["finished_unix"] = time.time()
    write_json(output / "synthesis" / "phase3b_runner_summary.json", summary)
    write_heartbeat(output, current_job=None, stage="finished")
    return summary
