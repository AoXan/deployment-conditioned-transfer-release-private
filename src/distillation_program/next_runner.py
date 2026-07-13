from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import jsonschema
import torch
import yaml

from src.stage8.forbidden_paths import forbidden_reason


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/distillation_next_mechanism_v1.yaml"
DEFAULT_SCHEMA = ROOT / "schemas/distillation_next_mechanism_v1.schema.json"


class ProcessGuard:
    def __init__(self, output_root: Path):
        self.lock = output_root / "status/program.lock"
        self.pid = output_root / "status/program.pid"
        self.acquired = False

    def __enter__(self):
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"DUPLICATE_PROCESS:{self.lock}") from exc
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        self.pid.write_text(str(os.getpid()))
        self.acquired = True
        return self

    def __exit__(self, *_):
        if self.acquired:
            for path in (self.pid, self.lock):
                if path.is_file() and path.read_text().strip() == str(os.getpid()):
                    path.unlink()


def load_next_config(path: Path, schema_path: Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    jsonschema.validate(config, json.loads(schema_path.read_text()))
    return config


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_strings(nested)
    elif isinstance(value, str):
        yield value


def build_matrix_plan(config: dict[str, Any], matrix: str) -> dict[str, Any]:
    jobs = []
    folds = config["dataset"]["frozen_folds"]
    seeds = config["runtime"]["seeds"]
    for declared in config["jobs"]:
        if declared["matrix"] != matrix:
            continue
        if declared["scientific_status"].startswith("BLOCKED_"):
            jobs.append(dict(declared, subjob_id=declared["id"]))
            continue
        balancing_values = declared.get("balancing", [None])
        for fold in folds if declared["handler"] not in {"round1_audit", "diagnostics"} else [None]:
            for seed in seeds if fold is not None else [None]:
                for balancing in balancing_values:
                    suffix = "" if fold is None else f"__{fold}__seed_{seed}"
                    if balancing is not None:
                        suffix += f"__{balancing}"
                    row = dict(declared, subjob_id=declared["id"] + suffix)
                    if fold is not None:
                        row.update(fold=fold, seed=seed)
                    if balancing is not None:
                        row["balancing_method"] = balancing
                    jobs.append(row)
    limit = config["matrix_limits"][matrix]
    if len(jobs) > limit:
        raise ValueError(f"matrix_expansion_exceeds_limit:{matrix}:{len(jobs)}>{limit}")
    normalized = [{key: job[key] for key in sorted(job)} for job in sorted(jobs, key=lambda row: row["subjob_id"])]
    return {
        "program": config["program"],
        "matrix": matrix,
        "jobs": normalized,
        "plan_fingerprint": hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest(),
        "executed_jobs": 0,
    }


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    forbidden = sorted({reason for value in _walk_strings(config) if (reason := forbidden_reason(value))})
    runtime = ROOT / config["runtime"]["interpreter"]
    matrices = {name: len(build_matrix_plan(config, name)["jobs"]) for name in config["matrix_limits"]}
    fixed_threshold_fields = [key for key in _walk_strings(config) if "fixed_effect_threshold" in key or "conflict_ratio_threshold" in key]
    return {
        "status": "PREFLIGHT_PASSED" if not forbidden and runtime.is_file() and not fixed_threshold_fields else "PREFLIGHT_BLOCKED",
        "config": str(config_path),
        "runtime": {"path": str(runtime), "available": runtime.is_file()},
        "forbidden_path_errors": forbidden,
        "fixed_threshold_fields": fixed_threshold_fields,
        "matrices": matrices,
        "formal_jobs_started": 0,
    }


def _read(path: Path) -> str:
    return path.read_text() if path.is_file() else "NOT_AVAILABLE\n"


def build_parser(default_matrix: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 8 next-mechanism distillation runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--safe-smoke", action="store_true")
    parser.add_argument("--run-formal", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--matrix", default=default_matrix)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--resources", action="store_true")
    return parser


def cli_main(default_matrix: str, formal_handler=None) -> int:
    parser = build_parser(default_matrix)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output_root.resolve()
    config = load_next_config(config_path)
    if args.preflight:
        print(json.dumps(preflight(config, config_path), indent=2, sort_keys=True)); return 0
    if args.dry_run:
        print(json.dumps(build_matrix_plan(config, args.matrix), indent=2, sort_keys=True)); return 0
    if args.status:
        print(_read(output / "status/status.json"), end=""); return 0
    if args.progress:
        print(_read(output / "progress/progress.jsonl"), end=""); return 0
    if args.failures:
        print(_read(output / "failures/failure_ledger.jsonl"), end=""); return 0
    if args.resources:
        print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "cpu_count": os.cpu_count(), "mps_available": bool(torch.backends.mps.is_available())}, indent=2, sort_keys=True)); return 0
    if args.safe_smoke:
        if formal_handler is None:
            parser.error("safe smoke is not implemented for this entrypoint")
        print(json.dumps(formal_handler(config=config, output_root=output, job_id=args.job, matrix=args.matrix, smoke=True, resume=False), indent=2, sort_keys=True, default=str)); return 0
    if args.run_formal:
        if formal_handler is None:
            parser.error("formal handler is not implemented for this entrypoint")
        with ProcessGuard(output):
            result = formal_handler(config=config, output_root=output, job_id=args.job, matrix=args.matrix, smoke=False, resume=args.resume)
        print(json.dumps(result, indent=2, sort_keys=True, default=str)); return 0
    parser.error("one action flag is required")
    return 2
