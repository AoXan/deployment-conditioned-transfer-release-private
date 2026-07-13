from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_FORMAL_WORKTREE = Path(os.environ.get("AGRITECH_FORMAL_ROOT", "artifacts/formal_campaign"))
DEFAULT_FORMAL_CONFIG = DEFAULT_FORMAL_WORKTREE / "configs/universal_weather_full_campaign_v2_remediation_v1.yaml"
DEFAULT_FORMAL_SCHEMA = DEFAULT_FORMAL_WORKTREE / "schemas/universal_weather_full_campaign_v2_remediation_v1.schema.json"


class FormalCodeBridge:
    """Import and call the original Stage 8 v4 implementation.

    The bridge extends the already-loaded local ``src`` package search path so
    that the formal worktree's ``universal_weather_v2*`` packages can be
    imported without copying their training implementation into this worktree.
    """

    def __init__(
        self,
        *,
        formal_worktree: Path = DEFAULT_FORMAL_WORKTREE,
        formal_config: Path = DEFAULT_FORMAL_CONFIG,
        formal_schema: Path = DEFAULT_FORMAL_SCHEMA,
    ) -> None:
        self.formal_worktree = Path(formal_worktree).resolve()
        self.formal_config = Path(formal_config).resolve()
        self.formal_schema = Path(formal_schema).resolve()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        formal_src = self.formal_worktree / "src"
        formal_distillation = formal_src / "distillation_v4"
        if not formal_src.is_dir():
            raise FileNotFoundError(f"FORMAL_SRC_NOT_FOUND:{formal_src}")
        if str(self.formal_worktree) not in sys.path:
            sys.path.insert(0, str(self.formal_worktree))
        import src
        import src.distillation_v4

        if str(formal_src) not in src.__path__:
            src.__path__.insert(0, str(formal_src))
        if str(formal_distillation) not in src.distillation_v4.__path__:
            src.distillation_v4.__path__.insert(0, str(formal_distillation))
        self._loaded = True

    def load_config(self) -> dict:
        self.load()
        config_module = importlib.import_module("src.distillation_v4.universal_weather_v2_remediation.config")
        return config_module.load_config(self.formal_config, self.formal_schema)

    def execution_module(self) -> Any:
        self.load()
        return importlib.import_module("src.distillation_v4.universal_weather_v2_remediation.execution")

    def training_module(self) -> Any:
        self.load()
        return importlib.import_module("src.distillation_v4.universal_weather_v2.training")

    def artifacts_module(self) -> Any:
        self.load()
        return importlib.import_module("src.distillation_v4.universal_weather_v2_remediation.artifacts")

    def execute_target_job(self, *, config: dict, job: dict, output_root: Path, smoke: bool = False) -> dict:
        execution = self.execution_module()
        return execution.execute_target_job(
            config,
            job,
            repository_root=self.formal_worktree,
            output_root=Path(output_root),
            smoke=smoke,
        )


def source_dependency_path(output_root: Path, job: dict) -> Path:
    return (
        Path(output_root)
        / "jobs"
        / "source"
        / str(job["source_dataset"])
        / str(job["source_route"])
        / f"seed_{int(job['seed'])}"
    )


def formal_source_dependency_path(formal_campaign: Path, job: dict) -> Path:
    return (
        Path(formal_campaign)
        / "jobs"
        / "source"
        / str(job["source_dataset"])
        / str(job["source_route"])
        / f"seed_{int(job['seed'])}"
    )


def ensure_source_dependency_alias(*, formal_campaign: Path, output_root: Path, job: dict) -> dict:
    if str(job.get("transfer_strategy")) == "target_scratch":
        return {"status": "not_required"}
    source = formal_source_dependency_path(formal_campaign, job)
    target = source_dependency_path(output_root, job)
    if not source.is_dir():
        raise FileNotFoundError(f"FORMAL_SOURCE_DEPENDENCY_NOT_FOUND:{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {"status": "exists", "path": str(target), "source": str(source)}
    os.symlink(source, target, target_is_directory=True)
    return {"status": "symlinked", "path": str(target), "source": str(source)}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
