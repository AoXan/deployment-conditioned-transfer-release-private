from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .next_artifacts import ArtifactWriter, fingerprint_bundle, validate_resume_artifact
from .next_audit import audit_round1_run
from .next_losses import DistillationObjective
from .next_models import ModalitySpecificRegressor, RoutedModalityRegressor, count_capacity
from .next_training import derive_decision_rule_from_outer_train, load_formal_data, run_architecture_subjob, run_privileged_subjob, run_teacher_subjob, run_true_kd_subjob


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _expected_resume_fingerprints(config: dict[str, Any], job: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    from . import next_training

    actual = manifest.get("fingerprint_inputs", {})
    if set(actual) != {"code", "config", "data", "view", "split", "feature", "model", "runtime"}:
        return {}
    materials: dict[str, Any] = {
        "code": hashlib.sha256(Path(next_training.__file__).read_bytes()).hexdigest(),
        "config": config,
        "feature": "preserve_actual",
        "model": "preserve_actual",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "data": "unavailable",
        "view": "unavailable",
        "split": "unavailable",
    }
    if job.get("fold"):
        frame, folds = load_formal_data(config, job["fold"], Path(__file__).resolve().parents[2])
        materials["data"] = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
        materials["view"] = {"columns": list(frame), "sample_unit": config["dataset"]["sample_unit"]}
        materials["split"] = hashlib.sha256(pd.util.hash_pandas_object(folds, index=True).values.tobytes()).hexdigest()
    expected = fingerprint_bundle(materials)
    expected["feature"] = actual["feature"]
    expected["model"] = actual["model"]
    if not job.get("fold"):
        expected["data"] = actual["data"]; expected["view"] = actual["view"]; expected["split"] = actual["split"]
    return expected


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
        "n": int(len(target)),
    }


def run_synthetic_smoke(handler: str, output: Path, *, seed: int) -> dict[str, Any]:
    """Exercise the real torch teacher/student path without producing scientific evidence."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n = 96
    weather = rng.normal(size=(n, 3)).astype("float32")
    soil = rng.normal(size=(n, 2)).astype("float32")
    target = (1.2 * weather[:, 0] - 0.4 * weather[:, 1] + 0.8 * soil[:, 0] + rng.normal(scale=0.1, size=n)).astype("float32")
    train_index = np.arange(0, 64)
    validation_index = np.arange(64, 80)
    test_index = np.arange(80, 96)
    rich_values = {"weather": torch.from_numpy(weather), "soil": torch.from_numpy(soil)}
    deployable_values = {"weather": rich_values["weather"]}
    rich_mask = torch.ones(n, 2)
    deployable_mask = torch.ones(n, 1)
    y = torch.from_numpy(target)

    teacher = ModalitySpecificRegressor({"weather": 3, "soil": 2}, hidden=8)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-2)
    teacher_steps = 0
    for _ in range(20):
        teacher_optimizer.zero_grad()
        prediction, _ = teacher(rich_values, rich_mask)
        loss = torch.nn.functional.huber_loss(prediction[train_index], y[train_index])
        if not torch.isfinite(loss):
            raise FloatingPointError("non_finite_teacher_smoke_loss")
        loss.backward()
        teacher_optimizer.step()
        teacher_steps += 1
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "teacher_checkpoint.pt"
    torch.save({"state_dict": teacher.state_dict(), "seed": seed, "capacity": count_capacity(teacher)}, checkpoint)
    replay_teacher = ModalitySpecificRegressor({"weather": 3, "soil": 2}, hidden=8)
    replay_teacher.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    replay_teacher.eval()

    student = ModalitySpecificRegressor({"weather": 3}, hidden=8)
    projector = torch.nn.Linear(8, 16)
    objective = DistillationObjective(["supervised", "prediction", "representation"], balancing="loss_scale_normalisation")
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(projector.parameters()) + list(objective.parameters()), lr=1e-2)
    trajectory = []
    student_steps = 0
    for epoch in range(20):
        optimizer.zero_grad()
        student_prediction, student_aux = student(deployable_values, deployable_mask)
        with torch.no_grad():
            teacher_prediction, teacher_aux = replay_teacher(rich_values, rich_mask)
        losses = {
            "supervised": torch.nn.functional.huber_loss(student_prediction[train_index], y[train_index]),
            "prediction": torch.nn.functional.huber_loss(student_prediction[train_index], teacher_prediction[train_index]),
            "representation": torch.nn.functional.mse_loss(projector(student_aux["representation"][train_index]), teacher_aux["representation"][train_index]),
        }
        total, ledger = objective(losses, shared_parameters=list(student.parameters()))
        total.backward()
        optimizer.step()
        trajectory.append({"epoch": epoch, **ledger})
        student_steps += 1
    student.eval()
    with torch.no_grad():
        prediction, _ = student(deployable_values, deployable_mask)
        test_prediction = prediction[test_index].numpy()
    if not np.isfinite(test_prediction).all() or np.std(test_prediction) == 0:
        raise RuntimeError("invalid_smoke_predictions")

    ids = np.array([f"synthetic_{index:03d}" for index in range(n)])
    predictions = pd.DataFrame({"sample_id": ids[test_index], "y_true": target[test_index], "y_pred": test_prediction})
    split = np.full(n, "train", dtype=object)
    split[validation_index] = "validation"
    split[test_index] = "test"
    folds = pd.DataFrame({"sample_id": ids, "split": split})
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    fingerprints = fingerprint_bundle({
        "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {"handler": handler, "seed": seed, "smoke": True},
        "data": hashlib.sha256(weather.tobytes() + soil.tobytes() + target.tobytes()).hexdigest(),
        "view": {"modalities": {"weather": 3, "soil": 2}, "namespace": "synthetic_smoke"},
        "split": hashlib.sha256(pd.util.hash_pandas_object(folds, index=False).values.tobytes()).hexdigest(),
        "feature": {"teacher": ["weather", "soil"], "student": ["weather"]},
        "model": {"teacher": count_capacity(teacher), "student": count_capacity(student)},
        "runtime": runtime,
    })
    writer = ArtifactWriter(output)
    writer.write_predictions(predictions)
    writer.write_fold_ids(folds)
    writer.write_metrics(_regression_metrics(target[test_index], test_prediction))
    writer.write_manifest({
        "status": "SMOKE_VERIFIED",
        "scientific_status": "NOT_APPLICABLE",
        "evidence_status": "SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "handler": handler,
        "teacher_frozen": all(not parameter.requires_grad for parameter in teacher.parameters()),
        "teacher_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "fingerprint_inputs": fingerprints,
    })
    _atomic_json(output / "loss_trajectory.json", trajectory)
    _atomic_json(output / "capacity_compute_ledger.json", {
        "teacher": count_capacity(teacher),
        "student": count_capacity(student),
        "teacher_optimizer_steps": teacher_steps,
        "student_optimizer_steps": student_steps,
        "compute_proxy": "parameter_count_times_optimizer_steps",
    })
    writer.accept()
    writer.write_completion_marker()
    return {
        "status": "SMOKE_VERIFIED",
        "scientific_status": "NOT_APPLICABLE",
        "teacher_frozen": True,
        "optimizer_steps": student_steps,
        "output": str(output),
    }


def run_outer_train_safe_smoke(config: dict[str, Any], output: Path, *, seed: int) -> dict[str, Any]:
    """Use a deterministic tiny subset of a frozen fold's outer-train only."""
    repository_root = Path(__file__).resolve().parents[2]
    frame, frozen_folds = load_formal_data(config, "test_2023", repository_root)
    allowed_ids = set(frozen_folds.loc[frozen_folds.split.eq("train"), "sample_id"].astype(str))
    safe = frame.loc[frame.sample_id.astype(str).isin(allowed_ids)].copy()
    years = sorted(safe.Year.dropna().astype(int).unique())[-4:]
    if len(years) < 3:
        raise RuntimeError("safe_outer_train_smoke_requires_three_years")
    safe = safe.loc[safe.Year.astype(int).isin(years)].copy()
    safe["_sample_hash"] = safe.sample_id.astype(str).map(lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    safe = safe.sort_values(["Year", "_sample_hash"]).groupby("Year", observed=True).head(48).drop(columns="_sample_hash")
    smoke_test_year = years[-1]
    folds = pd.DataFrame({"sample_id": safe.sample_id.astype(str), "split": np.where(safe.Year.astype(int).eq(smoke_test_year), "test", "train")})
    job = {
        "id": "safe_outer_train_prediction_kd",
        "subjob_id": "safe_outer_train_prediction_kd",
        "variant": "prediction_distillation",
        "balancing_method": "loss_scale_normalisation",
        "seed": seed,
        "fold": "SAFE_OUTER_TRAIN_ONLY",
        "scientific_status": "CONFIGURED_NOT_RUN",
    }
    result = run_true_kd_subjob(config, job, output, frame=safe, folds=folds, smoke_budget=True, smoke_only=True)
    return {**result, "status": "SMOKE_VERIFIED", "scientific_status": "NOT_APPLICABLE", "source_scope": "FROZEN_OUTER_TRAIN_ONLY", "outer_test_rows_used": 0}


def _not_directly_executable(*, job: dict[str, Any], **_: Any) -> dict[str, Any]:
    return {"job_id": job["id"], "status": job["scientific_status"], "reason": job.get("reason", "scientific_or_fidelity_gate_not_satisfied")}


def _round1_handler(*, job: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    root = Path(context["config"]["historical_input"]["root"])
    if not root.is_absolute():
        root = Path(context["repository_root"]) / root
    files = sorted(root.glob("**/predictions.csv"))
    audits = []
    for path in files:
        audits.append(audit_round1_run(path.parent))
    result = {"job_id": job["id"], "status": "AUDIT_COMPLETE_NOT_SCIENTIFIC_VERDICT", "audited_prediction_files": len(audits), "audits": audits}
    _atomic_json(Path(context["output"]) / "round1_audit.json", result)
    return result


def _teacher_handler(**kwargs: Any) -> dict[str, Any]:
    return run_teacher_subjob(kwargs["context"]["config"], kwargs["job"], kwargs["context"]["output"])


def _privileged_handler(**kwargs: Any) -> dict[str, Any]:
    return run_privileged_subjob(kwargs["context"]["config"], kwargs["job"], kwargs["context"]["output"])


def _kd_handler(**kwargs: Any) -> dict[str, Any]:
    return run_true_kd_subjob(kwargs["context"]["config"], kwargs["job"], kwargs["context"]["output"])


def _diagnostic_handler(**kwargs: Any) -> dict[str, Any]:
    job, context = kwargs["job"], kwargs["context"]
    if job["id"] == "diagnostic_decision_rule":
        rule = derive_decision_rule_from_outer_train(context["config"])
        _atomic_json(Path(context["output"]) / "decision_rule.json", rule)
        return {"job_id": job["id"], "status": "DECISION_RULE_FROZEN_FROM_OUTER_TRAIN", "decision_rule_fingerprint": rule["decision_rule_fingerprint"], "scientific_status": "NOT_APPLICABLE"}
    accepted = sorted(Path(context["output_root"]).glob("scientific/**/acceptance.json"))
    if not accepted:
        return {"job_id": job["id"], "status": "BLOCKED_PREREQUISITE", "reason": "accepted_scientific_inputs_required"}
    return {"job_id": job["id"], "status": "DIAGNOSTIC_INPUTS_DISCOVERED", "accepted_inputs": [str(path) for path in accepted], "scientific_status": "INSUFFICIENT_EVIDENCE"}


def _mechanism_handler(**kwargs: Any) -> dict[str, Any]:
    return _not_directly_executable(**kwargs)


def _architecture_handler(**kwargs: Any) -> dict[str, Any]:
    return run_architecture_subjob(kwargs["context"]["config"], kwargs["job"], kwargs["context"]["output"])


HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "round1_audit": _round1_handler,
    "teacher_benchmark": _teacher_handler,
    "privileged_ablation": _privileged_handler,
    "true_kd": _kd_handler,
    "diagnostics": _diagnostic_handler,
    "mechanism_adapter": _mechanism_handler,
    "architecture_mechanism": _architecture_handler,
}


def execute_program(
    *, config: dict[str, Any], output_root: Path, job_id: str | None, matrix: str,
    smoke: bool, resume: bool,
) -> dict[str, Any]:
    from .next_runner import ROOT, build_matrix_plan

    if smoke:
        root = output_root / "test_runs" / matrix
        synthetic = run_synthetic_smoke(matrix, root / "synthetic_smoke", seed=config["runtime"]["seeds"][0])
        real = run_outer_train_safe_smoke(config, root / "safe_outer_train_smoke", seed=config["runtime"]["seeds"][0])
        return {"status": "SMOKE_VERIFIED", "scientific_status": "NOT_APPLICABLE", "synthetic": synthetic, "safe_outer_train": real}

    plan = build_matrix_plan(config, matrix)
    jobs = plan["jobs"]
    if job_id:
        jobs = [job for job in jobs if job["subjob_id"] == job_id or job["id"] == job_id]
        if not jobs:
            raise ValueError(f"unknown_job_or_subjob:{job_id}")
    results = []
    for job in jobs:
        if job["scientific_status"].startswith("BLOCKED_"):
            event = {"job_id": job["subjob_id"], "status": job["scientific_status"], "reason": job.get("reason", "declared_gate_not_satisfied"), "timestamp_unix": time.time()}
            results.append(event)
            _append_jsonl(output_root / "progress/progress.jsonl", event)
            _append_jsonl(output_root / "failures/failure_ledger.jsonl", event)
            continue
        destination = output_root / "scientific" / matrix / job["subjob_id"]
        marker = destination / "completion_marker.json"
        if marker.is_file():
            if not resume:
                results.append({"job_id": job["subjob_id"], "status": "BLOCKED_EXISTING_ARTIFACT", "reason": "use_resume_or_new_output_namespace"})
                continue
            manifest_path = destination / "manifest.json"
            if not manifest_path.is_file():
                results.append({"job_id": job["subjob_id"], "status": "BLOCKED_RESUME_FINGERPRINT", "reason": "manifest_missing"})
                continue
            manifest = json.loads(manifest_path.read_text())
            validation = validate_resume_artifact(destination, _expected_resume_fingerprints(config, job, manifest))
            if validation != "REUSABLE":
                results.append({"job_id": job["subjob_id"], "status": "BLOCKED_RESUME_FINGERPRINT", "reason": validation})
                continue
            results.append({"job_id": job["subjob_id"], "status": "ARTIFACT_ACCEPTED_REUSED"})
            continue
        context = {"config": config, "repository_root": str(ROOT), "output_root": str(output_root), "output": destination}
        try:
            result = HANDLERS[job["handler"]](job=job, context=context)
        except Exception as exc:
            result = {"job_id": job["subjob_id"], "status": "FAILED_RETRYABLE", "failure_type": type(exc).__name__, "message": str(exc)}
            _atomic_json(destination / "failure.json", result)
        event = {**result, "job_id": job["subjob_id"], "timestamp_unix": time.time()}
        results.append(event)
        _append_jsonl(output_root / "progress/progress.jsonl", event)
        if event["status"].startswith(("FAILED_", "BLOCKED_")):
            _append_jsonl(output_root / "failures/failure_ledger.jsonl", event)
    status = {"matrix": matrix, "jobs": results, "scientific_promotion_performed": False}
    _atomic_json(output_root / "status/status.json", status)
    return status
