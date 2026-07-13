from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from .training_contract import NeuralTrainingContract


@dataclass(frozen=True)
class CampaignConfig:
    schema_version: str
    program: str
    output_root: Path
    interpreter: Path
    deterministic_seed: int
    stochastic_seeds: tuple[int, ...]
    primary_datasets: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    budgets: dict[str, int]
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CampaignConfig":
        runtime = value["runtime"]
        primary = tuple(value["primary_datasets"])
        if primary != ("PRIMARY_G2F_MAIZE", "PRIMARY_CYBENCH_MAIZE_US"):
            raise ValueError("PRIMARY_DATASET_SELECTION_NOT_FROZEN")
        budgets = {
            key: int(number)
            for key, number
            in value["budgets"].items()
        }

        schema_version = str(
            value["schema_version"]
        )

        expected_budgets = {
            "stage8_v4_1": {
                "phase2": 63,
                "phase3": 51,
                "phase4": 63,
                "phase5": 12,
            },
            "stage8_v4_repair_v1": {
                "phase2": 63,
                "phase3": 51,
                "phase4": 66,
                "phase5": 18,
            },
        }

        if schema_version not in expected_budgets:
            raise ValueError(
                "UNKNOWN_V4_SCHEMA_VERSION"
            )

        if budgets != expected_budgets[
            schema_version
        ]:
            raise ValueError(
                "V4_BUDGET_CONTRACT_MISMATCH"
            )
        return cls(
            schema_version=str(value["schema_version"]),
            program=str(value["program"]),
            output_root=Path(value["output_root"]),
            interpreter=Path(runtime["interpreter"]),
            deterministic_seed=int(runtime["deterministic_seed"]),
            stochastic_seeds=tuple(int(seed) for seed in runtime["stochastic_seeds"]),
            primary_datasets=primary,
            forbidden_paths=tuple(str(path) for path in value["forbidden_paths"]),
            budgets=budgets,
            raw=value,
        )

    @property
    def core_job_ceiling(self) -> int:
        return len(self.primary_datasets) * sum(self.budgets.values())

    @property
    def neural_training(self) -> NeuralTrainingContract:
        training = self.raw.get("training")
        if not isinstance(training, dict):
            raise ValueError("TRAINING_CONTRACT_REQUIRED")
        return NeuralTrainingContract.from_mapping(training)


def load_campaign(path: Path, schema_path: Path | None = None) -> CampaignConfig:
    text = path.read_text()
    value = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if schema_path is not None:
        jsonschema.validate(value, json.loads(schema_path.read_text()))
    return CampaignConfig.from_mapping(value)
