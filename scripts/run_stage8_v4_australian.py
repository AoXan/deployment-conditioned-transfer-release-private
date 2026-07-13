#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

import jsonschema
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distillation_v4.australian.artifacts import artifact_complete, atomic_json, fingerprint, sha256_file
from src.distillation_v4.australian.campaign import load_config, plan_campaign, write_plan
from src.distillation_v4.australian.execution import execute_job
from src.distillation_v4.australian.harmonisation import harmonise_frame
from src.distillation_v4.australian.acceptance import accept_jobs
from src.distillation_v4.australian.recovery import recovery_contract


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--output-root", type=Path)
    modes = value.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "data-audit", "dry-run", "run-formal", "smoke", "status", "progress", "failures", "resources", "accept", "report"):
        modes.add_argument("--" + name, action="store_true")
    value.add_argument("--dataset")
    value.add_argument("--model")
    value.add_argument("--route")
    value.add_argument("--job")
    value.add_argument("--resume", action="store_true")
    return value


def _filter(jobs, args):
    for job in jobs:
        if args.dataset and job["dataset_id"] != args.dataset: continue
        if (args.model or args.route) and job["component_id"] != (args.model or args.route): continue
        if args.job and job["job_id"] != args.job: continue
        yield job


def _summary(root: Path) -> dict:
    manifests = list((root / "jobs").glob("*/manifest.json")) if (root / "jobs").exists() else []
    failures = [json.loads(line) for line in (root / "failures/failure_ledger.jsonl").read_text().splitlines()] if (root / "failures/failure_ledger.jsonl").is_file() else []
    plan = json.loads((root / "control/executable_job_matrix.json").read_text()) if (root / "control/executable_job_matrix.json").is_file() else {"jobs": []}
    blocked = sum(job.get("status", "").startswith("BLOCKED") for job in plan["jobs"])
    completed_ids = {json.loads(path.read_text())["job_id"] for path in manifests if (path.parent / "completion.json").is_file()}
    failed_ids = {row["job_id"] for row in failures} - completed_ids
    return {"planned": len(plan["jobs"]), "runnable": len(plan["jobs"]) - blocked, "blocked": blocked, "completed": len(completed_ids), "failed": len(failed_ids), "failure_events": len(failures), "remaining_runnable": max(0, len(plan["jobs"]) - blocked - len(completed_ids) - len(failed_ids))}


def _write_report(root: Path, config: dict, config_path: Path) -> dict:
    summary = _summary(root); accepted = accept_jobs(root, config=config, config_path=config_path); groups = {}
    for row in accepted["accepted"]:
        key = "|".join(str(row[name]) for name in ("protocol_tier", "dataset_id", "sample_unit", "target_unit", "component"))
        groups.setdefault(key, []).append(float(row["metrics"]["mae"]))
    aggregation = {key: {"job_count": len(values), "mean_mae": sum(values) / len(values)} for key, values in groups.items()}
    report = {"engineering": summary, "acceptance": {k: accepted[k] for k in ("accepted_count", "rejected_count", "strata")}, "stratified_metrics": aggregation, "claim_boundary": "EXACT_ADAPTED_TRANSFER_DIAGNOSTIC_NOT_POOLED"}
    atomic_json(root / "reports/summary.json", report)
    return report


@contextmanager
def _process_guard(root: Path, enabled: bool):
    if not enabled:
        yield; return
    lock = root / "campaign.lock"; lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"BLOCKED_DUPLICATE_PROCESS:{lock}") from exc
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try: yield
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()): lock.unlink()


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config)
    schema = json.loads(Path("schemas/stage8_v4_australian.schema.json").read_text())
    jsonschema.validate(config, schema)
    expected = Path(config["runtime"]["interpreter"]).resolve()
    if Path(sys.executable).resolve() != expected:
        raise SystemExit(f"BLOCKED_RUNTIME_MISMATCH expected={expected} actual={Path(sys.executable).resolve()}")
    root = args.output_root or Path(config["output_root"])
    if args.status or args.progress or args.failures or args.resources:
        payload = _summary(root)
        if args.failures:
            path = root / "failures/failure_ledger.jsonl"; print(path.read_text() if path.is_file() else "NO_FAILURES")
        elif args.resources:
            path = root / "control/resource_estimate.json"; print(path.read_text() if path.is_file() else json.dumps({"status": "RESOURCE_PLAN_NOT_GENERATED"}))
        else: print(json.dumps(payload, indent=2))
        return 0
    for dataset in config["datasets"].values():
        lowered = str(dataset.get("view", "")).lower()
        if "nvt" in lowered or ("g2f" in lowered and "2024" in lowered): raise SystemExit("FORBIDDEN_DATASET_PATH_DECLARED")
    plan = plan_campaign(config)
    code_files = [Path("src/distillation_v4/australian/execution.py"), Path("src/distillation_v4/australian/campaign.py"), Path("scripts/run_stage8_v4_australian.py")]
    code_fingerprint = fingerprint({str(path): sha256_file(path) for path in code_files})
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "NOT_INSTALLED"
    runtime_fingerprint = fingerprint({"interpreter": str(expected), "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch_version})
    config_fingerprint = fingerprint({"entry_file_sha256": sha256_file(args.config), "resolved_config": config})
    if args.preflight:
        if plan["candidates"]["unresolved_components"]: raise SystemExit("PREFLIGHT_UNRESOLVED_COMPONENTS")
        seen = set()
        for phase in plan["dag"]:
            if not set(phase["depends_on"]).issubset(seen): raise SystemExit("PREFLIGHT_DAG_DEPENDENCY_INVALID")
            seen.add(phase["phase"])
        print(json.dumps({"status": "PREFLIGHT_PASSED", "datasets": len(plan["datasets"]), "methods": len(plan["methods"]), "candidate_cells": plan["candidates"]["candidate_count"], "jobs": len(plan["jobs"]), "executed_jobs": 0}, indent=2)); return 0
    write_plan(plan, root)
    if args.data_audit:
        asset_contract = recovery_contract(config.get("assets", {})); atomic_json(root / "control/asset_recovery_contract.json", asset_contract)
        print(json.dumps({"datasets": plan["datasets"], "assets": asset_contract}, indent=2)); return 0
    if args.dry_run:
        atomic_json(root / "control/dry_run.json", {"plan_fingerprint": plan["plan_fingerprint"], "candidate_cells": plan["candidates"]["candidate_count"], "jobs": len(plan["jobs"]), "executed_jobs": 0})
        print(json.dumps(json.loads((root / "control/dry_run.json").read_text()), indent=2)); return 0
    if args.accept:
        print(json.dumps(accept_jobs(root, config=config, config_path=args.config), indent=2)); return 0
    if args.report:
        print(json.dumps(_write_report(root, config, args.config), indent=2)); return 0
    selected = list(_filter(plan["jobs"], args))
    selected = [job for job in selected if job["status"] == "CONFIGURED_NOT_RUN"]
    phase_order = {"teachers_baselines": 0, "routes": 1, "controls_ablations": 2}
    selected.sort(key=lambda job: (phase_order[job["execution_phase"]], job["dataset_id"], job["component_id"], job["fold"], job["seed"]))
    if args.smoke:
        preferred = [job for job in selected if job["component_id"] == "supervised" and job["dataset_id"] == "regional_stage7"]
        selected = (preferred or selected)[:1]
    if not selected: raise SystemExit("NO_MATCHING_JOBS")
    frames = {}; failures = root / "failures/failure_ledger.jsonl"; failures.parent.mkdir(parents=True, exist_ok=True)
    with _process_guard(root, args.run_formal):
      for job in selected:
        spec = config["datasets"][job["dataset_id"]]
        if job["dataset_id"] not in frames:
            frames[job["dataset_id"]] = harmonise_frame(pd.read_csv(spec["view"]), spec)
        frame, features = frames[job["dataset_id"]]
        source_train = None
        split = next(s for s in plan["splits"][job["dataset_id"]] if s["fold"] == job["fold"] and s["axis"] == job["axis"])
        if job.get("source_dataset_id"):
            source_id = job["source_dataset_id"]
            source_spec = config["datasets"][source_id]
            if source_id not in frames:
                frames[source_id] = harmonise_frame(pd.read_csv(source_spec["view"]), source_spec)
            source_frame, source_features = frames[source_id]
            common_deployable = sorted(set(features["deployable"]) & set(source_features["deployable"]))
            common_soil = sorted(set(features["soil"]) & set(source_features["soil"]))
            if not common_deployable:
                with failures.open("a") as handle: handle.write(json.dumps({"job_id": job["job_id"], "type": "BLOCKED_DATA_CONTRACT", "reason": f"NO_SHARED_DEPLOYABLE_FEATURES:{source_id}->{job['dataset_id']}", "at": time.time()}) + "\n")
                continue
            target_cutoff_ids = set(split["train_ids"]) | set(split["validation_ids"])
            target_cutoff = pd.to_numeric(frame.loc[frame.sample_id.astype(str).isin(target_cutoff_ids), "year"], errors="coerce").max()
            source_frame = source_frame.loc[pd.to_numeric(source_frame.year, errors="coerce").le(target_cutoff)].copy()
            if source_frame.empty:
                with failures.open("a") as handle: handle.write(json.dumps({"job_id": job["job_id"], "type": "BLOCKED_DATA_CONTRACT", "reason": f"NO_SOURCE_ROWS_BEFORE_TARGET_CUTOFF:{source_id}->{job['dataset_id']}:{target_cutoff}", "at": time.time()}) + "\n")
                continue
            keep_target = sorted(split["train_ids"], key=lambda value: fingerprint({"id": value, "seed": job["seed"]}))[:max(1, int(len(split["train_ids"]) * float(job["label_budget"])))]
            split = {**split, "train_ids": keep_target}
            features = {"deployable": common_deployable, "weather": sorted(set(features["weather"]) & set(source_features["weather"])), "soil": common_soil}
            source_train = source_frame
        if args.smoke:
            split = {**split, **{f"{part}_ids": sorted(split[f"{part}_ids"], key=lambda value: fingerprint({"smoke_id": value, "part": part}))[:32] for part in ("train", "validation", "test")}}
            job = {**job, "smoke_only": True}
        target_train_ids = set(split["train_ids"])
        target_train = frame.loc[frame.sample_id.astype(str).isin(target_train_ids)]
        features = {name: [column for column in columns if column in target_train and target_train[column].notna().any()] for name, columns in features.items()}
        dataset_manifest = next(item for item in plan["datasets"] if item["dataset_id"] == job["dataset_id"])
        fingerprints = {"config": config_fingerprint, "data": dataset_manifest.get("view_sha256"), "view": dataset_manifest.get("view_sha256"), "split": fingerprint(split), "feature": fingerprint(features), "model": fingerprint({"component": job["component_id"], "seed": job["seed"]}), "runtime": runtime_fingerprint, "code": code_fingerprint}
        base_fingerprint = fingerprint({"job": job, "fingerprints": fingerprints, "plan": plan["plan_fingerprint"]})
        namespace = root / ("smoke" if args.smoke else "jobs")
        existing = sorted(namespace.glob(job["job_id"] + "__attempt_*")) if namespace.exists() else []
        if args.resume:
            reusable = any((prior / "manifest.json").is_file() and (prior / "completion.json").is_file() and json.loads((prior / "manifest.json").read_text()).get("base_fingerprint") == base_fingerprint for prior in existing)
            if reusable: continue
        attempt = len(existing) + 1
        contract_fp = fingerprint({"base_fingerprint": base_fingerprint, "attempt": attempt})
        job = {**job, "base_fingerprint": base_fingerprint, "attempt": attempt, "execution_fingerprint": contract_fp, "fingerprints": fingerprints}
        destination = namespace / f"{job['job_id']}__attempt_{attempt:02d}_{contract_fp[:8]}"
        try:
            result = execute_job(job, frame, features, split, destination, source_train=source_train)
            with (root / "status/progress.jsonl").open("a") if (root / "status").exists() else _opened(root / "status/progress.jsonl") as handle:
                handle.write(json.dumps({"job_id": job["job_id"], "status": "SMOKE_COMPLETE" if args.smoke else "EXECUTION_COMPLETE", "at": time.time(), "metrics": result["metrics"]}) + "\n")
        except Exception as exc:
            with failures.open("a") as handle: handle.write(json.dumps({"job_id": job["job_id"], "type": type(exc).__name__, "reason": str(exc), "at": time.time()}) + "\n")
            if not args.run_formal: raise
    if args.run_formal:
        _write_report(root, config, args.config)
    return 0


def _opened(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a")


if __name__ == "__main__": raise SystemExit(main())
