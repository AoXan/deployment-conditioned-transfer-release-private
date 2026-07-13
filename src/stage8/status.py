from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_status(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "manifests" / "job_plan.json"
    plan = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"jobs": []}
    counts = Counter()
    current: list[str] = []
    completed: list[str] = []
    remaining: list[str] = []
    blocked: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    for job in plan.get("jobs", []):
        status_path = output_root / "status" / f"{job['job_id']}.json"
        status = job.get("state", "UNKNOWN")
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            status = payload.get("status", status)
            if status == "RUNNING":
                current.append(job["job_id"])
        if status == "VERIFIED":
            completed.append(job["job_id"])
        elif status != "SKIPPED":
            remaining.append(job["job_id"])
        if status in {"BLOCKED", "CONDITIONAL"}:
            blocked.append({"job_id": job["job_id"], "state": str(status), "reason": str(job.get("blocked_reason", job.get("gate", "")))})
        counts[str(status)] += 1
    failure_path = output_root / "failures" / "failure_ledger.jsonl"
    if failure_path.exists():
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                failed.append(json.loads(line))
    return {"counts": dict(sorted(counts.items())), "completed": completed, "current": current, "remaining": remaining, "blocked": blocked, "failures": failed, "total": sum(counts.values())}
