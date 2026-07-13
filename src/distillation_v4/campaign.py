from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .contracts import CampaignConfig
from .datasets import build_cybench_view, feature_groups, validate_feature_contract
from .fingerprints import code_tree_fingerprint, mapping_fingerprint, sha256_file
from .gates import DevelopmentDecision, evaluate_development_gate
from .student_gate import evaluate_student_credibility
from .phase_contract import read_phase_completion
from .teacher_registry import validate_frozen_teacher_registry
from .outer_readiness import (
    validate_frozen_outer_readiness_contract,
)
from .matrix import phase2_jobs
from .outer import Student, fit_outer_route
from .training import fit_phase2_job, fit_phase2_job_repair
from .uncertainty import split_conformal
from .explanations import grouped_interventional_shap


ROOT = Path(__file__).resolve().parents[2]
V3_FRAGMENT = "outputs/distillation_program/kd_full_repair_v3"
HISTORICAL_V4_OUTPUT = (
    ROOT / "outputs/distillation_program/stage8_v4"
).resolve()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(path)


def _append(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def assert_v4_output(output: Path, *, write: bool = False) -> None:
    resolved_path = output.resolve()
    resolved = str(resolved_path)

    if V3_FRAGMENT in resolved or "next_mechanism_v1" in resolved:
        raise ValueError("V3_OR_LEGACY_OUTPUT_FORBIDDEN")

    historical_target = (
        resolved_path == HISTORICAL_V4_OUTPUT
        or HISTORICAL_V4_OUTPUT in resolved_path.parents
    )
    if write and historical_target:
        raise ValueError("HISTORICAL_V4_OUTPUT_READ_ONLY")


def runtime_record(config: CampaignConfig) -> dict[str, Any]:
    return {
        "interpreter": str(config.interpreter),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def code_record(
    config: CampaignConfig | None = None,
) -> dict[str, Any]:
    schema_version = (
        config.raw.get("schema_version")
        if config is not None
        else "stage8_v4_1"
    )

    if schema_version == "stage8_v4_repair_v1":
        config_path = (
            ROOT
            / "configs/stage8_v4_repair_v1.yaml"
        )
        schema_path = (
            ROOT
            / "schemas/stage8_v4_repair_v1.schema.json"
        )
    elif schema_version == "stage8_v4_1":
        config_path = (
            ROOT
            / "configs/stage8_v4_campaign.yaml"
        )
        schema_path = (
            ROOT
            / "schemas/stage8_v4_campaign.schema.json"
        )
    else:
        raise RuntimeError(
            "CODE_RECORD_SCHEMA_VERSION_UNKNOWN:"
            + str(schema_version)
        )

    paths = (
        list(
            (
                ROOT / "src/distillation_v4"
            ).glob("*.py")
        )
        + [
            ROOT
            / "src/distillation_program/next_models.py",
            config_path,
            schema_path,
        ]
    )

    digest, files = code_tree_fingerprint(
        ROOT,
        paths,
    )

    return {
        "code_tree_fingerprint": digest,
        "schema_version": schema_version,
        "config_path": str(
            config_path.relative_to(ROOT)
        ),
        "schema_path": str(
            schema_path.relative_to(ROOT)
        ),
        "files": files,
    }


def _g2f_frame(
    config: CampaignConfig,
) -> pd.DataFrame:
    cfg = config.raw[
        "datasets"
    ]["PRIMARY_G2F_MAIZE"]

    frame = pd.read_csv(
        cfg["view"],
        low_memory=False,
    )

    frame = frame.rename(
        columns={
            "Year": "year",
            "Env": "group",
        }
    )

    frame["year"] = pd.to_numeric(
        frame.year,
        errors="raise",
    ).astype(int)

    return frame


def _cybench_path(output: Path) -> Path:
    return output / "control/views/PRIMARY_CYBENCH_MAIZE_US.csv.gz"


def _cybench_frame(config: CampaignConfig, output: Path) -> pd.DataFrame:
    path = _cybench_path(output)
    if not path.is_file():
        raise FileNotFoundError("CYBENCH_V4_VIEW_NOT_BUILT")
    frame = pd.read_csv(path).rename(columns={"season_year": "year", "adm_id": "group"})
    frame["year"] = frame.year.astype(int)
    return frame


def load_primary_frame(config: CampaignConfig, output: Path, dataset: str) -> pd.DataFrame:
    if dataset == "PRIMARY_G2F_MAIZE":
        return _g2f_frame(config)
    if dataset == "PRIMARY_CYBENCH_MAIZE_US":
        return _cybench_frame(config, output)
    raise ValueError(f"UNKNOWN_PRIMARY_DATASET:{dataset}")


def phase1_contracts(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    assert_v4_output(
        output,
        write=True,
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    cy = config.raw[
        "datasets"
    ]["PRIMARY_CYBENCH_MAIZE_US"]

    raw = Path(cy["raw_root"])

    required = [
        raw / f"{name}_maize_US.csv"
        for name in (
            "yield",
            "soil",
            "meteo",
            "crop_calendar",
            "location",
        )
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        payload = {
            "data_contract_status": (
                "BLOCKED_DATA_CONTRACT"
            ),
            "missing": missing,
        }

        _atomic_json(
            output
            / "control/data_contracts/"
            "PRIMARY_CYBENCH_MAIZE_US.json",
            payload,
        )

        return payload

    view_path = _cybench_path(output)

    if not view_path.is_file():
        frame = build_cybench_view(
            raw,
            crop="maize",
            country="US",
            cutoff_fraction=float(
                cy[
                    "forecast_cutoff_fraction"
                ]
            ),
        )

        view_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = (
            view_path.with_suffix(
                ".tmp.gz"
            )
        )

        frame.to_csv(
            temporary,
            index=False,
            compression="gzip",
        )

        temporary.replace(view_path)

    g2f = _g2f_frame(config)
    cyframe = _cybench_frame(
        config,
        output,
    )

    contracts: dict[
        str,
        dict[str, Any],
    ] = {}

    for dataset, frame in (
        (
            "PRIMARY_G2F_MAIZE",
            g2f,
        ),
        (
            "PRIMARY_CYBENCH_MAIZE_US",
            cyframe,
        ),
    ):
        validate_feature_contract(
            frame,
            target="target_yield",
        )

        groups = feature_groups(frame)
        soil_columns = list(
            groups["soil"]
        )

        if soil_columns:
            soil_missing = (
                frame[
                    soil_columns
                ].isna()
            )

            any_missing = (
                soil_missing.any(axis=1)
            )
            all_missing = (
                soil_missing.all(axis=1)
            )
            partial_missing = (
                any_missing
                & ~all_missing
            )
            complete = ~any_missing

            soil_missing_cell_fraction = float(
                soil_missing
                .to_numpy()
                .mean()
            )

            rows_with_any_missing_soil_fraction = float(
                any_missing.mean()
            )
            rows_with_all_soil_missing_fraction = float(
                all_missing.mean()
            )
            rows_with_partial_soil_missing_fraction = float(
                partial_missing.mean()
            )
            rows_with_complete_soil_fraction = float(
                complete.mean()
            )

            soil_availability_by_year = {}

            for year, year_frame in frame.groupby(
                "year",
                sort=True,
            ):
                year_missing = (
                    year_frame[
                        soil_columns
                    ].isna()
                )

                year_any = (
                    year_missing.any(
                        axis=1
                    )
                )
                year_all = (
                    year_missing.all(
                        axis=1
                    )
                )
                year_partial = (
                    year_any
                    & ~year_all
                )

                soil_availability_by_year[
                    str(int(year))
                ] = {
                    "rows": int(
                        len(year_frame)
                    ),
                    "complete_soil_rows": int(
                        (~year_any).sum()
                    ),
                    "partial_soil_missing_rows": int(
                        year_partial.sum()
                    ),
                    "whole_soil_missing_rows": int(
                        year_all.sum()
                    ),
                }
        else:
            soil_missing_cell_fraction = 0.0
            rows_with_any_missing_soil_fraction = 0.0
            rows_with_all_soil_missing_fraction = 0.0
            rows_with_partial_soil_missing_fraction = 0.0
            rows_with_complete_soil_fraction = 1.0
            soil_availability_by_year = {}

        partition_total = (
            rows_with_all_soil_missing_fraction
            + rows_with_partial_soil_missing_fraction
            + rows_with_complete_soil_fraction
        )

        if abs(
            partition_total - 1.0
        ) > 1e-9:
            raise RuntimeError(
                "SOIL_MISSINGNESS_PARTITION_INVALID:"
                f"{dataset}:{partition_total}"
            )

        contract = {
            "dataset": dataset,
            "rows": int(len(frame)),
            "years": [
                int(frame.year.min()),
                int(frame.year.max()),
            ],
            "sample_ids_unique": bool(
                frame.sample_id.is_unique
            ),
            "weather_features": len(
                groups["weather"]
            ),
            "soil_features": len(
                soil_columns
            ),
            "soil_missing_cell_fraction": (
                soil_missing_cell_fraction
            ),
            "rows_with_missing_soil_fraction": (
                rows_with_any_missing_soil_fraction
            ),
            "rows_with_any_missing_soil_fraction": (
                rows_with_any_missing_soil_fraction
            ),
            "rows_with_all_soil_missing_fraction": (
                rows_with_all_soil_missing_fraction
            ),
            "rows_with_partial_soil_missing_fraction": (
                rows_with_partial_soil_missing_fraction
            ),
            "rows_with_complete_soil_fraction": (
                rows_with_complete_soil_fraction
            ),
            "soil_availability_by_year": (
                soil_availability_by_year
            ),
            "missingness_semantics": {
                "natural_missingness": (
                    "OBSERVED_IN_SOURCE_VIEW"
                ),
                "synthetic_modality_dropout": (
                    "NOT_APPLIED_DURING_PHASE1_AUDIT"
                ),
                "whole_modality_missing_rule": (
                    "ALL_SOIL_FEATURES_NULL"
                ),
                "partial_missing_rule": (
                    "AT_LEAST_ONE_BUT_NOT_ALL_"
                    "SOIL_FEATURES_NULL"
                ),
                "complete_rule": (
                    "NO_SOIL_FEATURES_NULL"
                ),
            },
            "target": "target_yield",
            "data_contract_status": (
                "DATA_CONTRACT_ACCEPTED"
            ),
        }

        contracts[dataset] = contract

        _atomic_json(
            output
            / "control/data_contracts"
            / f"{dataset}.json",
            contract,
        )

    source_hashes = {
        str(path): sha256_file(path)
        for path in required
        if path.name
        != "meteo_maize_US.csv"
    }

    meteo = (
        raw / "meteo_maize_US.csv"
    )
    source_hashes[str(meteo)] = (
        sha256_file(meteo)
    )

    selection = {
        "selected_before_model_results": True,
        "primary_datasets": list(
            config.primary_datasets
        ),
        "selected_cybench_track": (
            "maize_US"
        ),
        "source_hashes": source_hashes,
        "view_sha256": sha256_file(
            view_path
        ),
        "selection_policy": (
            config.raw[
                "primary_selection"
            ]
        ),
    }

    selection[
        "primary_dataset_selection_fingerprint"
    ] = mapping_fingerprint(selection)

    _atomic_json(
        output
        / "control/"
        "primary_dataset_selection.json",
        selection,
    )

    runtime = runtime_record(config)

    _atomic_json(
        output
        / "runtime/environment.json",
        {
            **runtime,
            "runtime_fingerprint": (
                mapping_fingerprint(
                    runtime
                )
            ),
        },
    )

    _atomic_json(
        output
        / "control/code_tree.json",
        code_record(config),
    )

    return {
        "data_contract_status": (
            "DATA_CONTRACT_ACCEPTED"
        ),
        "contracts": contracts,
        **selection,
    }


def run_phase2(config: CampaignConfig, output: Path, *, dataset: str | None = None, job_id: str | None = None, resume: bool = False) -> dict[str, Any]:
    if not (output / "control/primary_dataset_selection.json").is_file():
        raise RuntimeError("PHASE1_REQUIRED")
    datasets = [dataset] if dataset else list(config.primary_datasets)
    rows = []
    for dataset_id in datasets:
        frame = load_primary_frame(config, output, dataset_id)
        jobs = [job for job in phase2_jobs(dataset_id) if job_id is None or job.job_id == job_id]
        registry = output / f"control/job_registries/{dataset_id}__phase2.json"
        _atomic_json(registry, {"phase": "phase2", "dataset": dataset_id, "jobs": [asdict(job) for job in jobs], "count": len(jobs)})
        for job in jobs:
            root = output / "development/phase2" / dataset_id / job.job_id
            if resume and (root / "completion_marker.json").is_file():
                row = {"job_id": job.job_id, "dataset": dataset_id, "status": "ENGINEERING_ACCEPTED_REUSED"}
            else:
                start = time.time()
                try:
                    result = fit_phase2_job(job, frame, root, max_epochs=int(config.neural_training.max_epochs))
                    row = {"job_id": job.job_id, "dataset": dataset_id, "status": result.engineering_status, "metrics": result.metrics, "runtime": result.runtime, "timestamp_unix": time.time()}
                except Exception as exc:
                    row = {"job_id": job.job_id, "dataset": dataset_id, "status": "FAILED_RETRYABLE", "failure_type": type(exc).__name__, "reason": str(exc), "timestamp_unix": time.time()}
                    _append(output / "failures/failure_ledger.jsonl", row)
            _append(output / "status/progress.jsonl", row)
            rows.append(row)
    _atomic_json(output / "status/phase2.json", {"jobs": rows, "count": len(rows)})
    return {"jobs": rows, "count": len(rows)}




def run_phase2_repair(
    config: CampaignConfig,
    output: Path,
    *,
    dataset: str | None = None,
    job_id: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    assert_v4_output(output, write=True)

    if resume:
        raise RuntimeError(
            "REPAIR_RESUME_NOT_IMPLEMENTED_UNTIL_FINGERPRINT_REVALIDATION"
        )

    if not (
        output / "control/primary_dataset_selection.json"
    ).is_file():
        raise RuntimeError("PHASE1_REQUIRED")

    contract = config.neural_training
    datasets = (
        [dataset]
        if dataset
        else list(config.primary_datasets)
    )

    rows: list[dict[str, Any]] = []

    for dataset_id in datasets:
        frame = load_primary_frame(
            config,
            output,
            dataset_id,
        )
        jobs = [
            job
            for job in phase2_jobs(dataset_id)
            if job_id is None or job.job_id == job_id
        ]

        registry = (
            output
            / "control/job_registries"
            / f"{dataset_id}__phase2_repair.json"
        )
        _atomic_json(
            registry,
            {
                "phase": "phase2_repair",
                "dataset": dataset_id,
                "jobs": [
                    asdict(job)
                    for job in jobs
                ],
                "count": len(jobs),
                "trainer": (
                    "stage8_v4_repair_minibatch_v1"
                ),
                "outer_test_may_enable_downstream_jobs": False,
            },
        )

        for job in jobs:
            root = (
                output
                / "development/phase2_repair"
                / dataset_id
                / job.job_id
            )

            started = time.time()

            try:
                result = fit_phase2_job_repair(
                    job,
                    frame,
                    root,
                    contract=contract,
                )
                row = {
                    "job_id": job.job_id,
                    "dataset": dataset_id,
                    "phase": "phase2_repair",
                    "status": result.engineering_status,
                    "metrics": result.metrics,
                    "runtime": result.runtime,
                    "timestamp_unix": time.time(),
                    "elapsed_wall_seconds": (
                        time.time() - started
                    ),
                }
            except Exception as exc:
                row = {
                    "job_id": job.job_id,
                    "dataset": dataset_id,
                    "phase": "phase2_repair",
                    "status": "FAILED_RETRYABLE",
                    "failure_type": type(exc).__name__,
                    "reason": str(exc),
                    "timestamp_unix": time.time(),
                    "elapsed_wall_seconds": (
                        time.time() - started
                    ),
                }
                _append(
                    output
                    / "failures"
                    / "failure_ledger.jsonl",
                    row,
                )

            _append(
                output
                / "status"
                / "progress.jsonl",
                row,
            )
            rows.append(row)

    payload = {
        "phase": "phase2_repair",
        "jobs": rows,
        "count": len(rows),
        "formal_run_approved": bool(
            config.raw
            .get("repair_contract", {})
            .get("formal_run_approved", False)
        ),
        "outer_test_may_enable_downstream_jobs": False,
    }

    _atomic_json(
        output / "status/phase2_repair.json",
        payload,
    )
    return payload



def _aggregate_student_credibility_fold(
    fold: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("EMPTY_STUDENT_CREDIBILITY_FOLD")

    return {
        "fold": fold,
        "seed_count": len(records),
        "seeds": sorted(
            int(record["seed"])
            for record in records
        ),
        "student_mae": float(np.mean([
            float(record["student_mae"])
            for record in records
        ])),
        "naive_mae": float(np.mean([
            float(record["naive_mae"])
            for record in records
        ])),
        "classical_mae": float(np.mean([
            float(record["classical_mae"])
            for record in records
        ])),
        "optimizer_steps": int(min(
            int(record["optimizer_steps"])
            for record in records
        )),
        "parameter_delta_l2": float(min(
            float(record["parameter_delta_l2"])
            for record in records
        )),
        "checkpoint_replay_max_abs_error": float(max(
            float(
                record[
                    "checkpoint_replay_max_abs_error"
                ]
            )
            for record in records
        )),
        "prediction_std": float(min(
            float(record["prediction_std"])
            for record in records
        )),
        "converged": bool(all(
            bool(record["converged"])
            for record in records
        )),
    }


def evaluate_student_credibility_gates(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    assert_v4_output(output, write=True)

    gate_config = config.raw.get(
        "student_credibility_gate"
    )
    if not isinstance(gate_config, dict):
        raise ValueError(
            "STUDENT_CREDIBILITY_GATE_CONFIG_REQUIRED"
        )

    variant = str(gate_config["variant"])
    minimum_folds = int(
        gate_config["minimum_folds"]
    )
    required_seed_count = int(
        gate_config["required_seeds_per_fold"]
    )
    minimum_improved_folds = int(
        gate_config["minimum_improved_folds"]
    )
    maximum_mean_classical_gap = float(
        gate_config["maximum_mean_classical_gap"]
    )
    replay_tolerance = float(
        gate_config["checkpoint_replay_tolerance"]
    )

    ledgers: dict[str, Any] = {}

    for dataset in config.primary_datasets:
        dataset_root = (
            output
            / "development/phase2_repair"
            / dataset
        )

        raw_records: list[dict[str, Any]] = []

        if dataset_root.is_dir():
            for path in sorted(
                dataset_root.glob(
                    "*/student_credibility_record.json"
                )
            ):
                record = json.loads(path.read_text())
                if record.get("dataset") != dataset:
                    raise RuntimeError(
                        "CREDIBILITY_DATASET_MISMATCH"
                    )
                if record.get("variant") != variant:
                    continue
                if record.get("data_scope") != (
                    "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                ):
                    raise RuntimeError(
                        "INVALID_CREDIBILITY_DATA_SCOPE"
                    )
                if (
                    float(
                        record[
                            "checkpoint_replay_max_abs_error"
                        ]
                    )
                    > replay_tolerance
                ):
                    record["converged"] = False
                raw_records.append(record)

        expected_folds = tuple(
            config.raw
            .get("datasets", {})
            .get(dataset, {})
            .get(
                "outer_folds",
                ("test_2021", "test_2022", "test_2023"),
            )
        )
        expected_seeds = tuple(
            int(seed)
            for seed in config.stochastic_seeds
        )

        grouped: dict[str, list[dict[str, Any]]] = {
            fold: []
            for fold in expected_folds
        }

        for record in raw_records:
            fold = str(record["fold"])
            if fold in grouped:
                grouped[fold].append(record)

        missing: list[dict[str, Any]] = []

        for fold in expected_folds:
            observed = {
                int(record["seed"])
                for record in grouped[fold]
            }
            missing_seeds = sorted(
                set(expected_seeds).difference(observed)
            )
            if (
                len(observed) != required_seed_count
                or missing_seeds
            ):
                missing.append(
                    {
                        "fold": fold,
                        "observed_seeds": sorted(observed),
                        "missing_seeds": missing_seeds,
                    }
                )

        if missing:
            ledger = {
                "dataset": dataset,
                "gate": "student_credibility",
                "decision": (
                    "STUDENT_CREDIBILITY_"
                    "INSUFFICIENT_EVIDENCE"
                ),
                "reason": (
                    "MISSING_EXPECTED_FOLD_SEED_RECORDS"
                ),
                "variant": variant,
                "expected_folds": list(expected_folds),
                "expected_seeds": list(expected_seeds),
                "missing": missing,
                "raw_record_count": len(raw_records),
                "gate_data_scope": (
                    "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                ),
                "outer_test_may_enable_downstream_jobs": False,
            }
        else:
            fold_records = [
                _aggregate_student_credibility_fold(
                    fold,
                    grouped[fold],
                )
                for fold in expected_folds
            ]

            gate = evaluate_student_credibility(
                fold_records,
                data_scope=(
                    "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                ),
                minimum_folds=minimum_folds,
                minimum_improved_folds=(
                    minimum_improved_folds
                ),
                maximum_mean_classical_gap=(
                    maximum_mean_classical_gap
                ),
            )

            ledger = {
                "dataset": dataset,
                "gate": "student_credibility",
                "decision": gate.decision.value,
                "reason": gate.evidence.get("reason"),
                "variant": variant,
                "fold_records": fold_records,
                "evidence": gate.evidence,
                "raw_record_count": len(raw_records),
                "gate_data_scope": (
                    "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                ),
                "outer_test_may_enable_downstream_jobs": False,
            }

        _atomic_json(
            output
            / "control/gates"
            / f"{dataset}__student_credibility.json",
            ledger,
        )
        ledgers[dataset] = ledger

    _atomic_json(
        output / "status/student_credibility_gates.json",
        {
            "gates": ledgers,
            "dataset_count": len(ledgers),
            "outer_test_may_enable_downstream_jobs": False,
        },
    )

    return ledgers

def _metric(output: Path, dataset: str, variant: str, fold: str, seed: int, phase_root: str = "phase2") -> dict[str, Any]:
    job_id = f"{dataset}__phase2__{variant}__{fold}__seed_{seed}"
    return json.loads((output / f"development/{phase_root}" / dataset / job_id / "metrics.json").read_text())


def evaluate_phase2_gates(config: CampaignConfig, output: Path) -> dict[str, Any]:
    ledgers = {}
    for dataset in config.primary_datasets:
        records=[]
        for fold in ("test_2021", "test_2022", "test_2023"):
            deploy=_metric(output,dataset,"hgb_deployable",fold,101)
            teacher=_metric(output,dataset,"hgb_weather_soil",fold,101)
            control=_metric(output,dataset,"hgb_shuffled_soil",fold,101)
            records.append({"fold":fold,"effect":teacher["environment_balanced_mae"]-deploy["environment_balanced_mae"],"negative_control_effect":control["environment_balanced_mae"]-deploy["environment_balanced_mae"],"worst_group_change":teacher["worst_group_mae"]-deploy["worst_group_mae"]})
        result=evaluate_development_gate(records,data_scope="OUTER_TRAIN_INNER_VALIDATION_ONLY")
        ledger={"dataset":dataset,"gate":"soil_representation","gate_data_scope":"OUTER_TRAIN_INNER_VALIDATION_ONLY","decision":result.decision.value,"evidence":result.evidence,"records":records,"outer_test_may_enable_downstream_jobs":False}
        _atomic_json(output/f"control/gates/{dataset}__soil.json",ledger); ledgers[dataset]=ledger
    return ledgers




def evaluate_phase2_repair_soil_gates(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    assert_v4_output(output, write=True)

    gate_config = config.raw.get("soil_gate")
    if not isinstance(gate_config, dict):
        raise ValueError("SOIL_GATE_CONFIG_REQUIRED")

    if bool(
        gate_config.get("historical_gate_may_be_reused", True)
    ):
        raise ValueError("HISTORICAL_SOIL_GATE_REUSE_FORBIDDEN")

    maximum_harm = float(
        gate_config["maximum_worst_group_harm"]
    )

    ledgers: dict[str, Any] = {}

    for dataset in config.primary_datasets:
        records: list[dict[str, Any]] = []

        for fold in ("test_2021", "test_2022", "test_2023"):
            deploy = _metric(
                output,
                dataset,
                "hgb_deployable",
                fold,
                101,

                phase_root="phase2_repair",)
            teacher = _metric(
                output,
                dataset,
                "hgb_weather_soil",
                fold,
                101,

                phase_root="phase2_repair",)
            control = _metric(
                output,
                dataset,
                "hgb_shuffled_soil",
                fold,
                101,

                phase_root="phase2_repair",)

            records.append(
                {
                    "fold": fold,
                    "effect": (
                        teacher["environment_balanced_mae"]
                        - deploy["environment_balanced_mae"]
                    ),
                    "negative_control_effect": (
                        control["environment_balanced_mae"]
                        - deploy["environment_balanced_mae"]
                    ),
                    "worst_group_change": (
                        teacher["worst_group_mae"]
                        - deploy["worst_group_mae"]
                    ),
                }
            )

        result = evaluate_development_gate(
            records,
            data_scope=(
                "OUTER_TRAIN_INNER_VALIDATION_ONLY"
            ),
            maximum_worst_group_harm=maximum_harm,
        )

        ledger = {
            "dataset": dataset,
            "gate": "soil_representation_repair",
            "decision": result.decision.value,
            "evidence": result.evidence,
            "records": records,
            "historical_gate_reused": False,
            "requires_recomputation": True,
            "gate_data_scope": (
                "OUTER_TRAIN_INNER_VALIDATION_ONLY"
            ),
            "outer_test_may_enable_downstream_jobs": False,
        }

        _atomic_json(
            output
            / "control/gates"
            / f"{dataset}__soil_repair.json",
            ledger,
        )
        ledgers[dataset] = ledger

    _atomic_json(
        output / "status/phase2_repair_soil_gates.json",
        {
            "gates": ledgers,
            "dataset_count": len(ledgers),
            "historical_gate_reused": False,
            "outer_test_may_enable_downstream_jobs": False,
        },
    )

    return ledgers


def evaluate_missing_aware_eligibility_gates(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    """Freeze eligibility for the distinct variable-modality experiment.

    This is not a claim that soil has incremental predictive value.  It only
    permits the predeclared missing-aware development experiment when a
    credible deployable student, legal soil block, explicit mask, and either
    natural or predeclared synthetic whole-modality masking are available.
    """
    contract = config.raw.get("soil_missing_aware")
    if not isinstance(contract, dict):
        raise ValueError("SOIL_MISSING_AWARE_CONFIG_REQUIRED")

    ledgers: dict[str, Any] = {}
    for dataset in config.primary_datasets:
        soil = json.loads(
            (output / "control/gates" / f"{dataset}__soil_repair.json").read_text()
        )
        student = json.loads(
            (output / "control/gates" / f"{dataset}__student_credibility.json").read_text()
        )
        data = json.loads(
            (output / "control/data_contracts" / f"{dataset}.json").read_text()
        )
        natural_fraction = float(data.get("rows_with_missing_soil_fraction", 0.0))
        synthetic_probability = float(contract.get("primary_dropout_probability", 0.0))
        mask_valid = (
            contract.get("explicit_availability_mask") is True
            and contract.get("mask_granularity") == "WHOLE_SOIL_MODALITY"
        )
        eligible = (
            student.get("decision") == "STUDENT_CREDIBILITY_OPEN"
            and data.get("data_contract_status") == "DATA_CONTRACT_ACCEPTED"
            and int(data.get("soil_features", 0)) > 0
            and mask_valid
            and (natural_fraction > 0.0 or synthetic_probability > 0.0)
        )
        ledger = {
            "dataset": dataset,
            "gate": "missing_aware_development_eligibility",
            "decision": (
                "DEVELOPMENT_CONDITIONAL_OPEN" if eligible else "DEVELOPMENT_BLOCKED"
            ),
            "reason": (
                "DISTINCT_VARIABLE_MODALITY_EXPERIMENT_ELIGIBLE"
                if eligible
                else "MISSING_AWARE_ELIGIBILITY_CONTRACT_FAILED"
            ),
            "soil_increment_gate": soil.get("decision"),
            "student_credibility_gate": student.get("decision"),
            "natural_missingness_fraction": natural_fraction,
            "synthetic_dropout_probability": synthetic_probability,
            "explicit_availability_mask": mask_valid,
            "selection_scope": "OUTER_TRAIN_INNER_VALIDATION_ONLY",
            "outer_test_used": False,
            "outer_test_may_enable_downstream_jobs": False,
            "claim_boundary": (
                "ELIGIBILITY_ONLY_NOT_SOIL_INCREMENT_OR_SCIENTIFIC_GO"
            ),
        }
        _atomic_json(
            output / "control/gates" / f"{dataset}__missing_aware.json",
            ledger,
        )
        ledgers[dataset] = ledger

    _atomic_json(
        output / "status/missing_aware_eligibility_gates.json",
        {"gates": ledgers, "outer_test_used": False},
    )
    return ledgers

def qualify_teachers(config: CampaignConfig, output: Path) -> dict[str, Any]:
    decisions={}
    for dataset in config.primary_datasets:
        soil=json.loads((output/f"control/gates/{dataset}__soil.json").read_text())
        if soil["decision"] not in {DevelopmentDecision.OPEN.value,DevelopmentDecision.CONDITIONAL.value}:
            prediction=representation="DEVELOPMENT_BLOCKED"
            reason="SOIL_GATE_NOT_OPEN"
        else:
            # HGB is the predeclared prediction candidate; its evidence is inherited from Phase 2.
            prediction=soil["decision"]
            # Phase 2 lacks an independently trained random-representation candidate with a valid probe.
            representation="DEVELOPMENT_INSUFFICIENT_EVIDENCE"
            reason="REPRESENTATION_NEGATIVE_CONTROL_NOT_QUALIFIED"
        payload={"dataset":dataset,"prediction_teacher":prediction,"representation_teacher":representation,"reason":reason,"gate_data_scope":"OUTER_TRAIN_INNER_VALIDATION_ONLY","outer_test_may_enable_downstream_jobs":False}
        _atomic_json(output/f"control/gates/{dataset}__teachers.json",payload); decisions[dataset]=payload
        contract=json.loads((output/f"control/data_contracts/{dataset}.json").read_text())
        missing_fraction=float(contract.get("rows_with_missing_soil_fraction", 0.0))
        missing_payload={
            "dataset": dataset,
            "gate": "missing_aware_deployment",
            "decision": (
                soil["decision"]
                if missing_fraction > 0.0
                else "DEVELOPMENT_BLOCKED"
            ),
            "reason": (
                "NATURAL_MISSINGNESS_PRESENT"
                if missing_fraction > 0.0
                else "NO_NATURAL_MISSINGNESS_OR_APPROVED_SYNTHETIC_REGIME"
            ),
            "rows_with_missing_soil_fraction": missing_fraction,
            "gate_data_scope": "OUTER_TRAIN_INNER_VALIDATION_ONLY",
            "outer_test_may_enable_downstream_jobs": False,
        }
        _atomic_json(output/f"control/gates/{dataset}__missing_aware.json", missing_payload)
    return decisions


def freeze_routes(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    if config.raw.get("schema_version") == (
        "stage8_v4_repair_v1"
    ):
        raise RuntimeError(
            "LEGACY_ROUTE_FREEZE_FORBIDDEN_FOR_REPAIR"
        )

    routes=[]
    for dataset in config.primary_datasets:
        soil=json.loads((output/f"control/gates/{dataset}__soil.json").read_text())
        teacher=json.loads((output/f"control/gates/{dataset}__teachers.json").read_text())
        missing=json.loads((output/f"control/gates/{dataset}__missing_aware.json").read_text())
        if soil["decision"] in {DevelopmentDecision.OPEN.value,DevelopmentDecision.CONDITIONAL.value}:
            routes.extend([{"dataset":dataset,"route":"supervised"},{"dataset":dataset,"route":"fine_tune"}])
        if missing["decision"] in {DevelopmentDecision.OPEN.value,DevelopmentDecision.CONDITIONAL.value}:
            routes.append({"dataset":dataset,"route":"missing_aware"})
        if teacher["prediction_teacher"] in {DevelopmentDecision.OPEN.value,DevelopmentDecision.CONDITIONAL.value}:
            routes.append({"dataset":dataset,"route":"prediction_kd"})
        if teacher["representation_teacher"] in {DevelopmentDecision.OPEN.value,DevelopmentDecision.CONDITIONAL.value}:
            routes.append({"dataset":dataset,"route":"representation_kd"})
        if all(teacher[key] in {DevelopmentDecision.OPEN.value,DevelopmentDecision.CONDITIONAL.value} for key in ("prediction_teacher","representation_teacher")):
            routes.append({"dataset":dataset,"route":"combined_kd"})
    frozen={"jobs":routes,"outer_test_may_enable_downstream_jobs":False,"primary_dataset_selection_fingerprint":json.loads((output/"control/primary_dataset_selection.json").read_text())["primary_dataset_selection_fingerprint"],"code_tree_fingerprint":code_record()["code_tree_fingerprint"]}
    frozen["route_fingerprint_before_outer_test"]=mapping_fingerprint(frozen)
    _atomic_json(output/"control/frozen_route_registry.json",frozen)
    release={"status":"OUTER_TEST_RELEASED","route_fingerprint_before_outer_test":frozen["route_fingerprint_before_outer_test"],"registry_sha256":sha256_file(output/"control/frozen_route_registry.json"),"outer_test_role":"FINAL_ESTIMATION_ONLY","outer_test_may_enable_downstream_jobs":False}
    _atomic_json(output/"control/outer_test_release.json",release)
    return frozen




def freeze_routes_repair(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    assert_v4_output(output, write=True)

    repair = config.raw.get("repair_contract", {})
    if bool(repair.get("outer_release_enabled", False)):
        raise RuntimeError(
            "OUTER_RELEASE_MUST_REMAIN_DISABLED_DURING_REPAIR"
        )

    phase_completions = {
        phase: read_phase_completion(
            output / "status" / f"{phase}_repair.json",
            expected_phase=phase,
        )
        for phase in ("phase3", "phase4", "phase5")
    }

    readiness_path = (
        output
        / "control"
        / "frozen_outer_readiness_contract.json"
    )
    readiness = (
        validate_frozen_outer_readiness_contract(
            path=readiness_path,
            output=output,
        )
    )

    teacher_registry_path = (
        output
        / "control"
        / "frozen_teacher_registry.json"
    )
    teacher_registry = validate_frozen_teacher_registry(
        teacher_registry_path,
        output_root=output,
        require_formal_release=True,
    )

    teacher_lookup: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for record in teacher_registry["teachers"]:
        key = (
            str(record["dataset"]),
            str(record["role"]),
        )
        teacher_lookup.setdefault(key, []).append(record)

    routes: list[dict[str, Any]] = []

    for dataset in config.primary_datasets:
        soil_path = (
            output
            / "control/gates"
            / f"{dataset}__soil_repair.json"
        )
        student_path = (
            output
            / "control/gates"
            / f"{dataset}__student_credibility.json"
        )
        teacher_path = (
            output
            / "control/gates"
            / f"{dataset}__teachers_repair.json"
        )
        missing_path = (
            output
            / "control/gates"
            / f"{dataset}__missing_aware.json"
        )

        for required in (
            soil_path,
            student_path,
            teacher_path,
            missing_path,
        ):
            if not required.is_file():
                raise RuntimeError(
                    f"REPAIR_GATE_MISSING:{required.name}"
                )

        soil = json.loads(soil_path.read_text())
        student = json.loads(student_path.read_text())
        teacher = json.loads(teacher_path.read_text())
        missing = json.loads(missing_path.read_text())

        student_open = student["decision"] == (
            "STUDENT_CREDIBILITY_OPEN"
        )

        if not student_open:
            continue

        routes.append(
            {
                "dataset": dataset,
                "route": "supervised",
                "student_credibility_required": True,
            }
        )

        if missing.get("decision") in {
            DevelopmentDecision.OPEN.value,
            DevelopmentDecision.CONDITIONAL.value,
        }:
            routes.append(
                {
                    "dataset": dataset,
                    "route": "missing_aware",
                    "student_credibility_required": True,
                    "missing_aware_gate_required": True,
                    "soil_increment_gate_required": False,
                    "evidence_stratum": "VARIABLE_MODALITY_DEPLOYMENT",
                }
            )

        if soil["decision"] in {
            DevelopmentDecision.OPEN.value,
            DevelopmentDecision.CONDITIONAL.value,
        }:
            routes.append(
                {
                    "dataset": dataset,
                    "route": "fine_tune",
                    "student_credibility_required": True,
                    "soil_gate_required": True,
                }
            )

        if teacher.get("prediction_teacher") in {
            DevelopmentDecision.OPEN.value,
            DevelopmentDecision.CONDITIONAL.value,
        }:
            frozen_prediction = teacher_lookup.get(
                (dataset, "prediction"),
                [],
            )
            if not frozen_prediction:
                raise RuntimeError(
                    f"FROZEN_PREDICTION_TEACHER_MISSING:{dataset}"
                )

            routes.append(
                {
                    "dataset": dataset,
                    "route": "prediction_kd",
                    "student_credibility_required": True,
                    "frozen_teacher_required": True,
                    "teacher_role": "prediction",
                    "teacher_registry": (
                        "control/frozen_teacher_registry.json"
                    ),
                    "teacher_registry_sha256": sha256_file(
                        teacher_registry_path
                    ),
                    "available_teacher_folds": sorted(
                        record["fold"]
                        for record in frozen_prediction
                    ),
                }
            )

        if teacher.get("representation_teacher") in {
            DevelopmentDecision.OPEN.value,
            DevelopmentDecision.CONDITIONAL.value,
        }:
            frozen_representation = (
                teacher_lookup.get(
                    (dataset, "representation"),
                    [],
                )
            )
            if not frozen_representation:
                raise RuntimeError(
                    f"FROZEN_REPRESENTATION_TEACHER_MISSING:"
                    f"{dataset}"
                )

            routes.append(
                {
                    "dataset": dataset,
                    "route": "representation_kd",
                    "student_credibility_required": True,
                    "frozen_teacher_required": True,
                    "teacher_registry": (
                        "control/"
                        "frozen_teacher_registry.json"
                    ),
                    "teacher_registry_sha256": (
                        sha256_file(
                            teacher_registry_path
                        )
                    ),
                    "available_teacher_folds": sorted(
                        record["fold"]
                        for record
                        in frozen_representation
                    ),
                }
            )

    frozen = {
        "status": "REPAIR_ROUTE_REGISTRY_FROZEN",
        "jobs": routes,
        "outer_test_release_created": False,
        "outer_test_may_enable_downstream_jobs": False,
        "required_phases_complete": True,
        "required_phase_completion_status": {
            phase: completion.status
            for phase, completion
            in phase_completions.items()
        },
        "outer_readiness_contract": (
            "control/"
            "frozen_outer_readiness_contract.json"
        ),
        "outer_readiness_fingerprint": readiness[
            "outer_readiness_fingerprint"
        ],
        "formal_approval_required": True,
        "outer_test_execution_allowed": False,
        "repair_outer_release_token_created": False,
        "repair_outer_release_token_required": True,
        "legacy_outer_release_allowed": False,
        "legacy_outer_registry_allowed": False,
        "historical_routes_reused": False,
        "primary_dataset_selection_fingerprint": json.loads(
            (
                output
                / "control/primary_dataset_selection.json"
            ).read_text()
        )["primary_dataset_selection_fingerprint"],
        "code_tree_fingerprint": code_record()[
            "code_tree_fingerprint"
        ],
    }

    frozen["route_fingerprint_before_outer_test"] = (
        mapping_fingerprint(frozen)
    )

    _atomic_json(
        output
        / "control/frozen_route_registry_repair.json",
        frozen,
    )

    blocked_release = {
        "status": "OUTER_TEST_RELEASE_BLOCKED",
        "reason": "REPAIR_CONTRACT_OUTER_RELEASE_DISABLED",
        "route_fingerprint_before_outer_test": (
            frozen["route_fingerprint_before_outer_test"]
        ),
        "outer_test_may_enable_downstream_jobs": False,
        "outer_test_execution_allowed": False,
        "formal_approval_required": True,
        "outer_readiness_fingerprint": readiness[
            "outer_readiness_fingerprint"
        ],
    }

    _atomic_json(
        output
        / "control/outer_test_release_blocked.json",
        blocked_release,
    )

    return frozen

def australian_inventory(config: CampaignConfig, output: Path) -> dict[str, Any]:
    records=[]
    for name,spec in config.raw.get("australian_assets",{}).items():
        path=Path(spec["view"]) if spec.get("view") else None
        record={"asset":name,"declared_role":spec["role"],"minimal_baseline_role":"ENGINEERING_DATA_CONTRACT_ONLY","may_select_source_method":False}
        if path is None:
            record.update(status=spec["role"],reason="NO_PAIRED_TARGET_VIEW_DECLARED")
        elif not path.is_file():
            record.update(status="ENGINEERING_BLOCKED",reason="DECLARED_VIEW_MISSING")
        else:
            frame=pd.read_csv(path); record.update(status=spec["role"],columns=list(frame),rows=len(frame),view_sha256=sha256_file(path))
            target=next((column for column in ("observed_yield_t_ha","yield_t_ha","target_value") if column in frame),None)
            year=next((column for column in ("year","Year") if column in frame),None)
            if target and year and spec["role"] not in {"LICENCE_BLOCKED","SCIENTIFICALLY_INCOMPATIBLE"}:
                numeric=[column for column in frame.select_dtypes(include=[np.number]).columns if column not in {target,year} and not any(token in column.lower() for token in ("yield","production","harvest_area"))]
                train=frame.loc[frame[year]<frame[year].max()].dropna(subset=[target]); test=frame.loc[frame[year].eq(frame[year].max())].dropna(subset=[target])
                if numeric and not train.empty and not test.empty:
                    from sklearn.impute import SimpleImputer
                    from sklearn.linear_model import Ridge
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler
                    model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=1.0))]); model.fit(train[numeric],train[target]); prediction=model.predict(test[numeric]); baseline={"asset":name,"role":"ENGINEERING_DATA_CONTRACT_ONLY","train_rows":len(train),"test_rows":len(test),"target":target,"unit":"dataset_declared_unit_requires_attestation","mae":float(np.mean(abs(test[target].to_numpy()-prediction))),"source_method_mutated":False}
                    _atomic_json(output/f"australian/minimal_baselines/{name}.json",baseline); record["minimal_baseline"]="ENGINEERING_ACCEPTED"
        records.append(record)
    payload={"assets":records,"australian_validation_status":"CONTRACT_AUDIT_COMPLETE","source_method_mutated":False}
    _atomic_json(output/"australian/inventory.json",payload); return payload


def run_outer_test(
    config: CampaignConfig,
    output: Path,
    release_path: Path,
) -> dict[str, Any]:
    if config.raw.get("schema_version") == (
        "stage8_v4_repair_v1"
    ):
        raise RuntimeError(
            "LEGACY_OUTER_RUNNER_FORBIDDEN_FOR_REPAIR"
        )

    release = json.loads(release_path.read_text())

    if release.get("release_type") == "REPAIR_ONLY":
        raise RuntimeError(
            "REPAIR_TOKEN_REQUIRES_REPAIR_OUTER_RUNNER"
        )

    registry_path = (
        output / "control/frozen_route_registry.json"
    )
    registry = json.loads(registry_path.read_text())
    if release.get("registry_sha256")!=sha256_file(registry_path): raise RuntimeError("OUTER_RELEASE_REGISTRY_HASH_MISMATCH")
    if release.get("route_fingerprint_before_outer_test")!=registry.get("route_fingerprint_before_outer_test"): raise RuntimeError("OUTER_RELEASE_ROUTE_FINGERPRINT_MISMATCH")
    before=sha256_file(registry_path); rows=[]
    for route in registry["jobs"]:
        frame=load_primary_frame(config,output,route["dataset"])
        for year in (2021,2022,2023):
            for seed in config.stochastic_seeds:
                root=output/f"outer_test/{route['dataset']}/{route['route']}__test_{year}__seed_{seed}"
                try:
                    result=fit_outer_route(route=route["route"],frame=frame,outer_test_year=year,seed=seed,output=root,route_fingerprint=registry["route_fingerprint_before_outer_test"],max_epochs=int(config.neural_training.max_epochs))
                    calibration=pd.read_csv(root/"calibration_predictions.csv"); predictions=pd.read_csv(root/"predictions.csv"); interval=split_conformal(calibration_y=calibration.y_true.to_numpy(),calibration_prediction=calibration.y_pred.to_numpy(),test_prediction=predictions.y_pred.to_numpy(),coverage=.9); predictions["lower_90"]=interval.lower; predictions["upper_90"]=interval.upper; predictions.to_csv(root/"predictions_with_intervals.csv",index=False); coverage=float(((predictions.y_true>=predictions.lower_90)&(predictions.y_true<=predictions.upper_90)).mean()); uncertainty={"u1":{"coverage":coverage,"target_coverage":.9,"interval_width":float(np.mean(interval.upper-interval.lower)),"quantile":interval.quantile},"u2":{"status":"INSUFFICIENT_EVIDENCE","reason":"PATTERN_CALIBRATION_REQUIRES_MINIMUM_SUPPORT"}}
                    _atomic_json(root/"uncertainty.json",uncertainty); row={"dataset":route["dataset"],"route":route["route"],"fold":f"test_{year}","seed":seed,"status":result["engineering_status"],"metrics":result["metrics"]}
                except Exception as exc:
                    row={"dataset":route["dataset"],"route":route["route"],"fold":f"test_{year}","seed":seed,"status":"FAILED_RETRYABLE","reason":str(exc),"failure_type":type(exc).__name__}; _append(output/"failures/failure_ledger.jsonl",row)
                _append(output/"status/progress.jsonl",row); rows.append(row)
    if sha256_file(registry_path)!=before: raise RuntimeError("OUTER_TEST_MUTATED_ROUTE_REGISTRY")
    payload={"outer_test_status":"OUTER_TEST_COMPLETE" if all(row["status"]=="ENGINEERING_ACCEPTED" for row in rows) else "OUTER_TEST_INCOMPLETE","jobs":rows,"route_registry_unchanged":True,"route_fingerprint_before_outer_test":registry["route_fingerprint_before_outer_test"]}; _atomic_json(output/"status/outer_test.json",payload); return payload


def run_explanations(config: CampaignConfig, output: Path) -> dict[str, Any]:
    """Compute fixed, grouped interventional SHAP after route freeze.

    Explanations are diagnostics only and cannot mutate gates or route registries.
    To bound cost without outcome-driven sampling, rows are sorted by sample ID and
    the first 256 are used for both background and explained sets.
    """
    registry_path=output/"control/frozen_route_registry.json"
    before=sha256_file(registry_path)
    registry=json.loads(registry_path.read_text())
    records=[]
    for route in registry["jobs"]:
        dataset=route["dataset"]
        frame=load_primary_frame(config,output,dataset).sort_values("sample_id")
        for year in (2021,2022,2023):
            root=output/f"outer_test/{dataset}/{route['route']}__test_{year}__seed_101"
            if not (root/"completion_marker.json").is_file():
                records.append({"dataset":dataset,"route":route["route"],"fold":f"test_{year}","status":"BLOCKED_OUTER_ARTIFACT_MISSING"}); continue
            checkpoint=torch.load(root/"model.pt",map_location="cpu",weights_only=True)
            features=list(checkpoint["features"]); preprocess=joblib.load(root/"preprocessor.joblib")
            train=frame.loc[frame.year<year-1].head(256); test=frame.loc[frame.year.eq(year)].head(256)
            x_background=np.asarray(preprocess.transform(train[features]),dtype=np.float32)
            x_test=np.asarray(preprocess.transform(test[features]),dtype=np.float32)
            hidden=int(checkpoint["state_dict"]["encoder.0.weight"].shape[0]); student=Student(x_test.shape[1],hidden=hidden); student.load_state_dict(checkpoint["state_dict"]); student.eval()
            def predict_student(values: np.ndarray) -> np.ndarray:
                with torch.no_grad(): prediction,_=student(torch.tensor(values,dtype=torch.float32))
                return prediction.numpy()
            student_shap=grouped_interventional_shap(predict_student,x_test,x_background,{"deployable_weather":list(range(x_test.shape[1]))})
            explanation=pd.DataFrame({"sample_id":test.sample_id.astype(str).to_numpy(),"base_value":student_shap["base_value"],"prediction":student_shap["prediction"],"shap_deployable_weather":student_shap["deployable_weather"]})
            explanation_root=output/f"reports/explanations/{dataset}/{route['route']}__test_{year}__seed_101"; explanation_root.mkdir(parents=True,exist_ok=True); explanation.to_csv(explanation_root/"student_grouped_shap.csv",index=False)
            summary={"dataset":dataset,"route":route["route"],"fold":f"test_{year}","seed":101,"method":"EXACT_GROUPED_INTERVENTIONAL_SHAP","selection_role":"POST_FREEZE_MODEL_RELIANCE_DIAGNOSTIC_ONLY","rows":len(explanation),"student_mean_abs_shap":{"deployable_weather":float(np.mean(np.abs(student_shap["deployable_weather"])))}}
            if route["route"]=="prediction_kd" and (root/"prediction_teacher.joblib").is_file():
                teacher=joblib.load(root/"prediction_teacher.joblib"); teacher_features=[column for column in frame if column.startswith("weather_") or (column.startswith("soil_") and not column.endswith("__missing"))]; raw_background=train[teacher_features].to_numpy(dtype=float); raw_test=test[teacher_features].to_numpy(dtype=float); weather_indices=[index for index,name in enumerate(teacher_features) if name.startswith("weather_")]; soil_indices=[index for index,name in enumerate(teacher_features) if name.startswith("soil_")]
                def predict_teacher(values: np.ndarray) -> np.ndarray:
                    return teacher.predict(pd.DataFrame(values,columns=teacher_features))
                teacher_shap=grouped_interventional_shap(predict_teacher,raw_test,raw_background,{"weather":weather_indices,"static_soil":soil_indices})
                pd.DataFrame({"sample_id":test.sample_id.astype(str).to_numpy(),"base_value":teacher_shap["base_value"],"prediction":teacher_shap["prediction"],"shap_weather":teacher_shap["weather"],"shap_static_soil":teacher_shap["static_soil"]}).to_csv(explanation_root/"teacher_grouped_shap.csv",index=False)
                summary["teacher_mean_abs_shap"]={"weather":float(np.mean(np.abs(teacher_shap["weather"]))),"static_soil":float(np.mean(np.abs(teacher_shap["static_soil"])))}
            _atomic_json(explanation_root/"summary.json",summary); records.append({**summary,"status":"ENGINEERING_ACCEPTED"})
    if sha256_file(registry_path)!=before: raise RuntimeError("EXPLANATION_MUTATED_ROUTE_REGISTRY")
    payload={"status":"EXPLANATION_DIAGNOSTICS_COMPLETE" if records and all(row["status"]=="ENGINEERING_ACCEPTED" for row in records) else "EXPLANATION_DIAGNOSTICS_INCOMPLETE","records":records,"causal_claim_allowed":False}; _atomic_json(output/"reports/explanation_registry.json",payload); return payload


def accept_campaign(config: CampaignConfig, output: Path) -> dict[str, Any]:
    accepted=[]; rejected=[]
    for marker in output.glob("**/completion_marker.json"):
        root=marker.parent
        if not all((root/name).is_file() for name in ("predictions.csv","metrics.json","fold_assignments.csv","manifest.json","acceptance.json")): rejected.append({"path":str(root),"reason":"ARTIFACT_CONTRACT_INCOMPLETE"}); continue
        predictions=pd.read_csv(root/"predictions.csv"); folds=pd.read_csv(root/"fold_assignments.csv"); metrics=json.loads((root/"metrics.json").read_text())
        ids=set(predictions.sample_id.astype(str)); test_ids=set(folds.loc[folds.split.eq("test"),"sample_id"].astype(str)); recomputed=float(np.mean(abs(predictions.y_true-predictions.y_pred)))
        if ids!=test_ids or abs(recomputed-float(metrics["mae"]))>5e-7: rejected.append({"path":str(root),"reason":"INDEPENDENT_ACCEPTANCE_FAILED"})
        else: accepted.append({"path":str(root),"predictions_sha256":sha256_file(root/"predictions.csv"),"metrics_sha256":sha256_file(root/"metrics.json")})
    payload={"engineering_status":"ENGINEERING_ACCEPTED" if not rejected else "ENGINEERING_INCOMPLETE","accepted":accepted,"rejected":rejected,"accepted_count":len(accepted),"rejected_count":len(rejected)}; _atomic_json(output/"acceptance/accepted_artifact_index.json",payload); return payload


def final_report(config: CampaignConfig, output: Path) -> dict[str, Any]:
    gates={}
    for path in (output/"control/gates").glob("*.json"): gates[path.stem]=json.loads(path.read_text())
    registry=json.loads((output/"control/frozen_route_registry.json").read_text()) if (output/"control/frozen_route_registry.json").is_file() else {"jobs":[]}
    outer=json.loads((output/"status/outer_test.json").read_text()) if (output/"status/outer_test.json").is_file() else {"outer_test_status":"NOT_RUN"}
    acceptance=json.loads((output/"acceptance/accepted_artifact_index.json").read_text()) if (output/"acceptance/accepted_artifact_index.json").is_file() else {"engineering_status":"NOT_ACCEPTED"}
    scientific="INSUFFICIENT_EVIDENCE" if not registry["jobs"] else ("INSUFFICIENT_EVIDENCE" if outer["outer_test_status"]!="OUTER_TEST_COMPLETE" else "CONDITIONAL_GO")
    payload={"ENGINEERING_STATUS":acceptance["engineering_status"],"DATA_CONTRACT_STATUS":"DATA_CONTRACT_ACCEPTED","DEVELOPMENT_GATE_STATUS":gates,"OUTER_TEST_STATUS":outer["outer_test_status"],"SCIENTIFIC_STATUS":scientific,"AUSTRALIAN_VALIDATION_STATUS":"CONTRACT_AUDIT_COMPLETE" if (output/"australian/inventory.json").is_file() else "NOT_RUN","RUNTIME_EVIDENCE_STATUS":"RUNTIME_EVIDENCE_INCOMPLETE","CLAIM_BOUNDARY":"Results are dataset-, fold-, route-, and observed-post-hoc-scope specific; no causal or universal soil/KD claim.","generated_routes":registry["jobs"],"v3_mutated_or_resumed":False}
    _atomic_json(output/"reports/final_evidence_registry.json",payload); _atomic_json(output/"status/final_status.json",payload)
    report="# Stage 8 V4 Final Scientific Report\n\n"+"\n".join(f"- **{key}**: `{value}`" for key,value in payload.items() if key.isupper())+"\n\nNo V3 job was modified, resumed, or rerun.\n"; path=output/"reports/final_scientific_report.md"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(report); return payload
