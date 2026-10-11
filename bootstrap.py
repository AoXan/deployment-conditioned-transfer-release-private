"""Create an isolated environment and execute a public-data reproduction check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parent


def run(command, name, logs, environment=None):
    started = time.time()
    with (logs / (name + ".log")).open("w") as stream:
        result = subprocess.run(
            list(map(str, command)), cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
    entry = {
        "command": list(map(str, command)),
        "returncode": result.returncode,
        "seconds": time.time() - started,
        "log": str(logs / (name + ".log")),
    }
    with (logs / "steps.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {entry['log']}")
    print(f"Completed {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/public_smoke")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12, the version verified with the pinned lock.")
    args.venv = args.venv.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    logs = args.venv.parent / (args.venv.name + "_setup_logs")
    logs.mkdir(parents=True, exist_ok=False)
    if args.venv.exists():
        parser.error("Choose a new environment directory; existing environments are preserved.")
    if args.smoke and args.output.exists():
        parser.error("Choose a new output directory; existing results are preserved.")
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(args.venv)
    python = args.venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    run([python, "-m", "pip", "install", "-r", ROOT / "requirements-verified.lock"], "dependencies", logs, environment)
    run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", ROOT], "project", logs, environment)
    run([python, "-m", "pip", "check"], "dependency_check", logs, environment)
    run([python, "-m", "pip", "freeze", "--all"], "installed_versions", logs, environment)
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "system_site_packages": False,
        "requirements_sha256": hashlib.sha256((ROOT / "requirements-verified.lock").read_bytes()).hexdigest(),
        "smoke_requested": args.smoke,
        "output": str(args.output),
    }
    (logs / "setup_receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.smoke:
        raw = args.data_root / "cybench/raw"
        run(
            [python, ROOT / "scripts/acquire_cybench_subset.py", "--output-root", raw, "--pair", "wheat_AU"],
            "raw_data",
            logs,
            environment,
        )
        run(
            [python, ROOT / "scripts/run_real_data_smoke.py", "--raw-root", raw, "--output", args.output],
            "training_and_diagnostics",
            logs,
            environment,
        )
        run(
            [
                python,
                ROOT / "scripts/render_new_experiment_assets.py",
                "--source",
                args.output,
                "--output",
                args.output / "tables",
            ],
            "new_result_assets",
            logs,
            environment,
        )
    print(f"Environment: {args.venv}\nLogs: {logs}", flush=True)


if __name__ == "__main__":
    main()
