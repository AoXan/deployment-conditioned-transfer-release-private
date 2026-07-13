from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import CampaignConfig
from .gates import DevelopmentDecision


@dataclass(frozen=True)
class FrozenRoutes:
    route_fingerprint: str
    jobs: tuple[dict[str, Any], ...]


class DynamicPlanner:
    def __init__(self, config: CampaignConfig):
        self.config = config
        self.gates: dict[tuple[str, str], DevelopmentDecision] = {}
        self._frozen = False
        self._phase2 = [
            {"job_id": f"{dataset}__soil_screen", "dataset": dataset, "phase": "phase2"}
            for dataset in config.primary_datasets
        ]

    def record_gate(self, dataset: str, gate: str, decision: DevelopmentDecision) -> None:
        if self._frozen:
            raise RuntimeError("ROUTE_REGISTRY_FROZEN")
        self.gates[(dataset, gate)] = decision

    def jobs_for(self, phase: str) -> list[dict[str, Any]]:
        if phase == "phase2":
            return list(self._phase2)
        jobs: list[dict[str, Any]] = []
        for dataset in self.config.primary_datasets:
            soil = self.gates.get((dataset, "soil"))
            if phase == "phase3" and soil in {DevelopmentDecision.OPEN, DevelopmentDecision.CONDITIONAL}:
                jobs.append({"job_id": f"{dataset}__teacher_qualification", "dataset": dataset, "phase": phase})
            if phase == "phase4":
                for gate, route in (("prediction_teacher", "pkd"), ("representation_teacher", "rkd")):
                    if self.gates.get((dataset, gate)) in {DevelopmentDecision.OPEN, DevelopmentDecision.CONDITIONAL}:
                        jobs.append({"job_id": f"{dataset}__{route}", "dataset": dataset, "phase": phase})
        return jobs

    def freeze_routes(self, path: Path) -> FrozenRoutes:
        jobs = tuple(self.jobs_for("phase4"))
        normalized = json.dumps(jobs, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"route_fingerprint_before_outer_test": digest, "jobs": jobs, "outer_test_may_enable_downstream_jobs": False}, indent=2, sort_keys=True) + "\n")
        self._frozen = True
        return FrozenRoutes(digest, jobs)
