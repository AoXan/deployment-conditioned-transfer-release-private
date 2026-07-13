"""Deterministic targeted-rerun planning."""
from __future__ import annotations
import json
from pathlib import Path


def plan_reruns(manifest: dict, *, job_id: str | None = None) -> list[dict]:
    rows = sorted(manifest.get("reruns", []), key=lambda row: str(row["job_id"]))
    if job_id is not None:
        rows = [row for row in rows if row["job_id"] == job_id]
        if not rows:
            raise KeyError(f"unknown rerun job: {job_id}")
    return rows


def route_is_fully_accepted(route_dir: Path, accepted_index: Path) -> bool:
    predictions = sorted(path.resolve() for path in route_dir.glob("**/predictions.csv"))
    if not predictions or not accepted_index.is_file():
        return False
    accepted = {Path(value).resolve() for value in json.loads(accepted_index.read_text()).get("predictions", [])}
    return all(path in accepted for path in predictions)
