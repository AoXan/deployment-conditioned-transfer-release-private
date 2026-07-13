from __future__ import annotations

import os
import subprocess
from typing import Any


def classify_process(
    pid: int,
    *,
    expected_command_fragment: str | None = None,
) -> dict[str, Any]:
    if pid <= 0:
        return {"pid": pid, "state": "STALE_PID", "reason": "PID_INVALID"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"pid": pid, "state": "STALE_PID", "reason": "PROCESS_NOT_FOUND"}
    except PermissionError:
        return {"pid": pid, "state": "ALIVE_UNINSPECTABLE"}

    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    line = completed.stdout.strip()
    if completed.returncode != 0 or not line:
        return {"pid": pid, "state": "STALE_PID", "reason": "PS_NO_RECORD"}
    stat, _, command = line.partition(" ")
    if "Z" in stat:
        return {"pid": pid, "state": "ZOMBIE", "stat": stat, "command": command.strip()}
    if expected_command_fragment and expected_command_fragment not in command:
        return {"pid": pid, "state": "PID_REUSED", "stat": stat, "command": command.strip()}
    return {"pid": pid, "state": "ALIVE", "stat": stat, "command": command.strip()}
