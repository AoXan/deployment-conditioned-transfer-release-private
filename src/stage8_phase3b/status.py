from __future__ import annotations

import json
import time
from pathlib import Path


DEFAULT_STATUS = {
    "replay": {"pending": 0, "running": 0, "completed": 0, "failed": 0, "blocked": 0},
    "reproduction": {"pending": 0, "pass": 0, "failed": 0, "blocked": 0},
    "explanation": {"pending": 0, "completed": 0, "failed": 0, "blocked": 0, "not_applicable": 0},
}


def _read_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return dict(default)
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def summarize_status(output_root: Path) -> dict:
    status_dir = Path(output_root) / "status"
    return {
        "replay": _read_json(status_dir / "replay_status.json", DEFAULT_STATUS["replay"]),
        "reproduction": _read_json(status_dir / "reproduction_status.json", DEFAULT_STATUS["reproduction"]),
        "explanation": _read_json(status_dir / "explanation_status.json", DEFAULT_STATUS["explanation"]),
    }


def write_status(output_root: Path, category: str, payload: dict) -> None:
    if category not in {"replay", "reproduction", "explanation"}:
        raise ValueError(f"UNKNOWN_STATUS_CATEGORY:{category}")
    _write_json(Path(output_root) / "status" / f"{category}_status.json", payload)


def write_heartbeat(output_root: Path, *, current_job: str | None, stage: str) -> None:
    _write_json(
        Path(output_root) / "status" / "heartbeat.json",
        {
            "timestamp_unix": time.time(),
            "current_job": current_job,
            "stage": stage,
        },
    )


def request_stop(output_root: Path, *, reason: str) -> Path:
    path = Path(output_root) / "status" / "STOP_REQUESTED.json"
    _write_json(path, {"timestamp_unix": time.time(), "reason": reason})
    return path


def stop_requested(output_root: Path) -> bool:
    return (Path(output_root) / "status" / "STOP_REQUESTED.json").is_file()

