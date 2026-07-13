from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .next_artifacts import ArtifactWriter, fingerprint_bundle
from .next_audit import grouped_regression_metrics
from .next_losses import DistillationObjective
from .next_models import ModalitySpecificRegressor, RoutedModalityRegressor, SharedDenseRegressor, count_capacity
from .next_contracts import derive_outer_train_decision_rule


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def load_formal_data(config: dict[str, Any], fold: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    view = Path(config["dataset"]["view"])
    fold_path = Path(config["dataset"]["folds"].format(fold=fold))
    if not view.is_absolute():
        view = root / view
    if not fold_path.is_absolute():
        fold_path = root / fold_path
    frame = pd.read_csv(view, low_memory=False)
    folds = pd.read_csv(fold_path)
    return frame, folds


def _split_frame(frame: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"sample_id", "target_yield", "Year"}
    if not required <= set(frame) or not {"sample_id", "split"} <= set(folds):
        raise ValueError("formal_frame_or_fold_contract_missing")
    merged = frame.merge(folds, on="sample_id", validate="one_to_one")
    outer_train = merged.loc[merged.split.eq("train")].copy()
    test = merged.loc[merged.split.eq("test")].copy()
    if outer_train.empty or test.empty:
        raise ValueError("formal_outer_train_or_test_empty")
    validation_year = int(outer_train.Year.max())
    train = outer_train.loc[outer_train.Year.lt(validation_year)].copy()
    validation = outer_train.loc[outer_train.Year.eq(validation_year)].copy()
    if train.empty or validation.empty:
        raise ValueError("formal_inner_validation_unavailable")
    return train, validation, test


def _groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "weather": [column for column in frame if column.startswith("weather_") and pd.api.types.is_numeric_dtype(frame[column])],
        "soil": [column for column in frame if column.startswith("soil_") and pd.api.types.is_numeric_dtype(frame[column])],
        "ec": [column for column in frame if column.startswith("ec_") and pd.api.types.is_numeric_dtype(frame[column])],
        "genotype": [column for column in frame if column.startswith("genotype_") and pd.api.types.is_numeric_dtype(frame[column])],
        "management": [column for column in frame if column.startswith("management_") and pd.api.types.is_numeric_dtype(frame[column])],
    }
    return {name: columns for name, columns in groups.items() if columns}


def _features_for_ablation(groups: dict[str, list[str]], ablation: str) -> list[str]:
    requested = {
        "deployable_only": ["weather"],
        "plus_soil": ["weather", "soil"],
        "plus_genotype": ["weather", "genotype"],
        "plus_legal_ec": ["weather", "ec"],
        "plus_management": ["weather", "management"],
        "plus_soil_genotype": ["weather", "soil", "genotype"],
        "all_legal_privileged": ["weather", "soil", "genotype", "management", "ec"],
        "retrospective_ec_diagnostic": ["weather", "ec"],
    }[ablation]
    features = [column for name in requested for column in groups.get(name, [])]
    if not features or not groups.get("weather"):
        raise ValueError(f"ablation_feature_contract_unavailable:{ablation}")
    return features


def _build_sklearn(model_id: str, features: list[str], params: dict[str, Any], seed: int) -> Pipeline:
    if model_id == "ridge":
        estimator = Ridge(**params)
    elif model_id == "random_forest":
        estimator = RandomForestRegressor(**params, random_state=seed, n_jobs=1)
    elif model_id == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(**params, max_iter=300, random_state=seed, early_stopping=False)
    elif model_id == "matched_mlp":
        converted = dict(params)
        converted["hidden_layer_sizes"] = tuple(converted["hidden_layer_sizes"])
        estimator = MLPRegressor(**converted, max_iter=300, early_stopping=False, random_state=seed)
    else:
        raise ValueError(f"sklearn_model_not_supported:{model_id}")
    return Pipeline([
        ("preprocess", ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)])),
        ("model", estimator),
    ])


def _model_capacity(model: Pipeline) -> dict[str, Any]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "estimators_"):
        nodes = sum(tree.tree_.node_count for tree in estimator.estimators_)
        return {"total_parameters": None, "active_parameters": None, "tree_nodes": int(nodes), "compute_proxy": "tree_nodes"}
    if hasattr(estimator, "coef_"):
        coefficients = estimator.coef_ if isinstance(estimator.coef_, list) else [estimator.coef_]
        intercepts = estimator.intercepts_ if hasattr(estimator, "intercepts_") else [np.asarray(estimator.intercept_)]
        parameters = sum(np.asarray(value).size for value in coefficients) + sum(np.asarray(value).size for value in intercepts)
        return {"total_parameters": int(parameters), "active_parameters": int(parameters), "tree_nodes": None, "compute_proxy": "parameter_count"}
    return {"total_parameters": None, "active_parameters": None, "tree_nodes": None, "compute_proxy": "fit_time_and_training_steps"}


def _metrics(test: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mae": float(mean_absolute_error(test.target_yield, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(test.target_yield, prediction))),
        "r2": float(r2_score(test.target_yield, prediction)),
        "n_rows": int(len(test)),
    }
    group = next((column for column in ("Environment", "environment", "Env") if column in test), None)
    if group:
        group_frame = pd.DataFrame({group: test[group].astype(str), "y_true": test.target_yield.to_numpy(), "y_pred": prediction})
        result["environment"] = grouped_regression_metrics(group_frame, group)
    return result


def _fingerprints(config: dict[str, Any], frame: pd.DataFrame, folds: pd.DataFrame, features: Any, model: Any) -> dict[str, str]:
    return fingerprint_bundle({
        "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config,
        "data": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        "view": {"columns": list(frame), "sample_unit": config["dataset"]["sample_unit"]},
        "split": hashlib.sha256(pd.util.hash_pandas_object(folds, index=True).values.tobytes()).hexdigest(),
        "feature": features,
        "model": model,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    })


def _write_run(
    *, output: Path, config: dict[str, Any], job: dict[str, Any], frame: pd.DataFrame, folds: pd.DataFrame,
    test: pd.DataFrame, prediction: np.ndarray, features: Any, model_ledger: dict[str, Any], selection: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None, smoke_only: bool = False,
) -> dict[str, Any]:
    predictions = pd.DataFrame({"sample_id": test.sample_id.astype(str), "y_true": test.target_yield, "y_pred": prediction})
    metrics = _metrics(test, prediction)
    writer = ArtifactWriter(output)
    writer.write_predictions(predictions)
    writer.write_fold_ids(folds)
    writer.write_metrics(metrics)
    manifest = {
        "job_id": job["id"], "subjob_id": job["subjob_id"], "fold": job.get("fold"), "seed": job.get("seed"),
        "status": "EXECUTION_COMPLETE_UNVERIFIED", "scientific_status": "NOT_APPLICABLE" if smoke_only else "INSUFFICIENT_EVIDENCE",
        "evidence_status": "SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE" if smoke_only else "SCIENTIFIC_EXECUTION_REQUIRES_INDEPENDENT_REVIEW",
        "outer_test_used_for_selection": False, "selection_ledger": selection, "capacity_compute_ledger": model_ledger,
        "fingerprint_inputs": _fingerprints(config, frame, folds, features, model_ledger),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    writer.write_manifest(manifest)
    _atomic_json(output / "selection_ledger.json", selection)
    _atomic_json(output / "capacity_compute_ledger.json", model_ledger)
    writer.accept()
    writer.write_completion_marker()
    return {"status": "ARTIFACT_ACCEPTED_NOT_SCIENTIFIC_VERDICT", "scientific_status": "INSUFFICIENT_EVIDENCE", "output": str(output)}


def run_teacher_subjob(
    config: dict[str, Any], job: dict[str, Any], output: Path, *, frame: pd.DataFrame | None = None, folds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if frame is None or folds is None:
        frame, folds = load_formal_data(config, job["fold"], Path(__file__).resolve().parents[2])
    train, validation, test = _split_frame(frame, folds)
    groups = _groups(frame)
    input_profile = job.get("input_profile", "legal_privileged")
    features = groups.get("weather", []) if input_profile == "deployable" else groups.get("weather", []) + groups.get("soil", [])
    if not features:
        raise ValueError("legal_teacher_features_unavailable")
    model_id = job["model"]
    if model_id in {"early_fusion_neural", "modality_specific_late_fusion"}:
        return _run_neural_teacher(config, job, output, frame, folds, train, validation, test, groups)
    candidates = config["model_search"][model_id]
    validation_rows = []
    start = time.perf_counter()
    for index, params in enumerate(candidates):
        candidate = _build_sklearn(model_id, features, params, job["seed"])
        candidate_start = time.perf_counter()
        candidate.fit(train[features], train.target_yield)
        score = float(mean_absolute_error(validation.target_yield, candidate.predict(validation[features])))
        validation_rows.append({"candidate_index": index, "params": params, "validation_mae": score, "fit_time_seconds": time.perf_counter() - candidate_start, "capacity": _model_capacity(candidate)})
    selected = min(validation_rows, key=lambda row: (row["validation_mae"], row["candidate_index"]))
    final = _build_sklearn(model_id, features, selected["params"], job["seed"])
    fit_frame = pd.concat([train, validation], ignore_index=True)
    final.fit(fit_frame[features], fit_frame.target_yield)
    prediction_start = time.perf_counter()
    prediction = final.predict(test[features])
    inference_seconds = time.perf_counter() - prediction_start
    elapsed = time.perf_counter() - start
    ledger = {**_model_capacity(final), "candidate_fits": len(candidates), "fit_time_seconds": elapsed, "inference_seconds": inference_seconds, "fitted_examples": int(len(fit_frame)), "training_steps": None}
    return _write_run(output=output, config=config, job=job, frame=frame, folds=folds, test=test, prediction=prediction, features=features, model_ledger=ledger, selection={"scope": "OUTER_TRAIN_INNER_VALIDATION_ONLY", "candidates": validation_rows, "selected": selected}, extra_manifest={"input_profile": input_profile, "feature_count": len(features)})


def _prepare_modalities(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, groups: dict[str, list[str]], names: list[str]):
    transformers = {}
    outputs = {"train": {}, "validation": {}, "test": {}}
    for name in names:
        declared_columns = groups.get(name, [])
        columns = [column for column in declared_columns if not train[column].isna().all()]
        dropped = sorted(set(declared_columns) - set(columns))
        if not columns:
            raise ValueError(f"required_modality_missing:{name}")
        transformer = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        outputs["train"][name] = torch.tensor(transformer.fit_transform(train[columns]), dtype=torch.float32)
        outputs["validation"][name] = torch.tensor(transformer.transform(validation[columns]), dtype=torch.float32)
        outputs["test"][name] = torch.tensor(transformer.transform(test[columns]), dtype=torch.float32)
        transformers[name] = {"columns": columns, "dropped_all_missing_in_train": dropped, "statistics_fit_scope": "outer_train_before_inner_validation"}
    return outputs, transformers


def _fit_torch_model(model, values, mask, target, validation_values, validation_mask, validation_target, config, seed, max_epochs=None):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    maximum = max_epochs or config["training"]["max_epochs"]
    patience = config["training"]["early_stopping_patience"]
    best = None
    no_improvement = 0
    trajectory = []
    for epoch in range(maximum):
        model.train(); optimizer.zero_grad()
        prediction, aux = model(values, mask)
        loss = torch.nn.functional.huber_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("non_finite_neural_teacher_loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction, _ = model(validation_values, validation_mask)
            validation_mae = float(torch.mean(torch.abs(validation_prediction - validation_target)))
        trajectory.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_mae": validation_mae})
        if best is None or validation_mae < best["validation_mae"]:
            best = {"validation_mae": validation_mae, "epoch": epoch, "state": deepcopy(model.state_dict())}; no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break
    model.load_state_dict(best["state"])
    return model, trajectory, {"selected_epoch": best["epoch"], "validation_mae": best["validation_mae"], "scope": "OUTER_TRAIN_INNER_VALIDATION_ONLY"}


def _run_neural_teacher(config, job, output, frame, folds, train, validation, test, groups):
    input_profile = job.get("input_profile", "legal_privileged")
    names = ["weather"] if input_profile == "deployable" else [name for name in ("weather", "soil") if name in groups]
    values, transformer = _prepare_modalities(train, validation, test, groups, names)
    widths = {name: values["train"][name].shape[1] for name in names}
    candidates = config["model_search"][job["model"]]
    validations = []
    trained = []
    for index, params in enumerate(candidates):
        torch.manual_seed(job["seed"] + index)
        model = SharedDenseRegressor(widths, hidden=params["hidden"]) if job["model"] == "early_fusion_neural" else ModalitySpecificRegressor(widths, hidden=params["hidden"])
        train_mask = torch.ones(len(train), len(names)); validation_mask = torch.ones(len(validation), len(names))
        model, trajectory, selection = _fit_torch_model(model, values["train"], train_mask, torch.tensor(train.target_yield.to_numpy(), dtype=torch.float32), values["validation"], validation_mask, torch.tensor(validation.target_yield.to_numpy(), dtype=torch.float32), config, job["seed"])
        validations.append({"candidate_index": index, "params": params, "validation_mae": selection["validation_mae"], "capacity": count_capacity(model), "training_steps": len(trajectory)})
        trained.append((model, trajectory, selection))
    selected = min(validations, key=lambda row: (row["validation_mae"], row["candidate_index"]))
    model, trajectory, selection = trained[selected["candidate_index"]]
    model.eval()
    with torch.no_grad(): prediction, _ = model(values["test"], torch.ones(len(test), len(names)))
    ledger = {**count_capacity(model), "candidate_fits": len(candidates), "training_steps": sum(len(item[1]) for item in trained), "fit_time_seconds": None, "inference_seconds": None, "compute_proxy": "active_parameters_times_training_steps"}
    _atomic_json(output / "training_trajectory.json", trajectory)
    return _write_run(output=output, config=config, job=job, frame=frame, folds=folds, test=test, prediction=prediction.numpy(), features=transformer, model_ledger=ledger, selection={"scope": "OUTER_TRAIN_INNER_VALIDATION_ONLY", "candidates": validations, "selected": selected}, extra_manifest={"input_profile": input_profile, "modalities": names})


def run_privileged_subjob(config, job, output, *, frame=None, folds=None):
    if frame is None or folds is None:
        frame, folds = load_formal_data(config, job["fold"], Path(__file__).resolve().parents[2])
    train, validation, test = _split_frame(frame, folds)
    groups = _groups(frame)
    features = _features_for_ablation(groups, job["ablation"])
    benchmark_job = {**job, "model": "hist_gradient_boosting"}
    candidates = config["model_search"]["hist_gradient_boosting"]
    validation_rows = []
    for index, params in enumerate(candidates):
        model = _build_sklearn("hist_gradient_boosting", features, params, job["seed"]); model.fit(train[features], train.target_yield)
        validation_rows.append({"candidate_index": index, "params": params, "validation_mae": float(mean_absolute_error(validation.target_yield, model.predict(validation[features])))})
    selected = min(validation_rows, key=lambda row: (row["validation_mae"], row["candidate_index"]))
    final = _build_sklearn("hist_gradient_boosting", features, selected["params"], job["seed"]); final.fit(pd.concat([train, validation])[features], pd.concat([train, validation]).target_yield)
    prediction = final.predict(test[features])
    return _write_run(output=output, config=config, job=benchmark_job, frame=frame, folds=folds, test=test, prediction=prediction, features=features, model_ledger={**_model_capacity(final), "candidate_fits": len(candidates), "fit_time_seconds": None, "inference_seconds": None, "fitted_examples": int(len(train) + len(validation)), "training_steps": None}, selection={"scope": "OUTER_TRAIN_INNER_VALIDATION_ONLY", "candidates": validation_rows, "selected": selected}, extra_manifest={"ablation": job["ablation"], "namespace_role": job.get("namespace", "scientific")})


def run_true_kd_subjob(config, job, output, *, frame=None, folds=None, smoke_budget=False, smoke_only=False):
    torch.manual_seed(job["seed"])
    torch.use_deterministic_algorithms(True)
    if frame is None or folds is None:
        frame, folds = load_formal_data(config, job["fold"], Path(__file__).resolve().parents[2])
    train, validation, test = _split_frame(frame, folds)
    groups = _groups(frame)
    names = [name for name in ("weather", "soil") if name in groups]
    if names != ["weather", "soil"]:
        raise ValueError("kd_requires_weather_and_legal_soil")
    values, transformer = _prepare_modalities(train, validation, test, groups, names)
    widths = {name: values["train"][name].shape[1] for name in names}
    maximum = 5 if smoke_budget else config["training"]["max_epochs"]
    teacher = ModalitySpecificRegressor(widths, hidden=32)
    teacher, teacher_trajectory, teacher_selection = _fit_torch_model(teacher, values["train"], torch.ones(len(train), 2), torch.tensor(train.target_yield.to_numpy(), dtype=torch.float32), values["validation"], torch.ones(len(validation), 2), torch.tensor(validation.target_yield.to_numpy(), dtype=torch.float32), config, job["seed"], max_epochs=maximum)
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "teacher_checkpoint.pt"; torch.save({"state_dict": teacher.state_dict(), "selection": teacher_selection}, checkpoint)

    student_values = {split: {"weather": values[split]["weather"]} for split in values}
    student = ModalitySpecificRegressor({"weather": widths["weather"]}, hidden=32)
    projector = torch.nn.Linear(32, 64)
    variant = job["variant"]
    names_for_loss = ["supervised"]
    if variant in {"prediction_distillation", "prediction_representation_distillation", "shuffled_teacher_control"}: names_for_loss.append("prediction")
    if variant in {"representation_distillation", "prediction_representation_distillation", "random_representation_control"}: names_for_loss.append("representation")
    if variant == "complete_missing_prediction_consistency": names_for_loss.append("prediction_consistency")
    if variant == "complete_missing_representation_consistency": names_for_loss.append("representation_consistency")
    balancing = job.get("balancing_method", "equal")
    objective = DistillationObjective(names_for_loss, balancing=balancing)
    parameters = list(student.parameters()) + list(projector.parameters()) + list(objective.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    trajectory = []
    rng = np.random.default_rng(job["seed"])
    permutation = torch.tensor(rng.permutation(len(train)), dtype=torch.long)
    target = torch.tensor(train.target_yield.to_numpy(), dtype=torch.float32)
    validation_target = torch.tensor(validation.target_yield.to_numpy(), dtype=torch.float32)
    best = None
    for epoch in range(maximum):
        student.train(); optimizer.zero_grad()
        prediction, student_aux = student(student_values["train"], torch.ones(len(train), 1))
        with torch.no_grad(): teacher_prediction, teacher_aux = teacher(values["train"], torch.ones(len(train), 2))
        losses = {"supervised": torch.nn.functional.huber_loss(prediction, target)}
        if "prediction" in names_for_loss:
            teacher_target = teacher_prediction[permutation] if variant == "shuffled_teacher_control" else teacher_prediction
            losses["prediction"] = torch.nn.functional.huber_loss(prediction, teacher_target)
        if "representation" in names_for_loss:
            teacher_representation = teacher_aux["representation"]
            if variant == "random_representation_control": teacher_representation = teacher_representation[permutation]
            losses["representation"] = torch.nn.functional.mse_loss(projector(student_aux["representation"]), teacher_representation)
        if "prediction_consistency" in names_for_loss:
            missing_prediction, _ = student(student_values["train"], torch.zeros(len(train), 1)); losses["prediction_consistency"] = torch.nn.functional.huber_loss(missing_prediction, prediction.detach())
        if "representation_consistency" in names_for_loss:
            _, missing_aux = student(student_values["train"], torch.zeros(len(train), 1)); losses["representation_consistency"] = torch.nn.functional.mse_loss(missing_aux["representation"], student_aux["representation"].detach())
        if variant == "fixed_blend_historical_control":
            with torch.no_grad(): blended = 0.5 * target + 0.5 * teacher_prediction
            losses["supervised"] = torch.nn.functional.huber_loss(prediction, blended)
        total, ledger = objective(losses, shared_parameters=list(student.parameters()))
        total.backward(); torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip_norm"]); optimizer.step()
        student.eval()
        with torch.no_grad(): validation_prediction, _ = student(student_values["validation"], torch.ones(len(validation), 1)); validation_mae = float(torch.mean(torch.abs(validation_prediction - validation_target)))
        trajectory.append({"epoch": epoch, "validation_mae": validation_mae, **ledger})
        if best is None or validation_mae < best["validation_mae"]: best = {"validation_mae": validation_mae, "epoch": epoch, "state": deepcopy(student.state_dict()), "projector": deepcopy(projector.state_dict())}
    student.load_state_dict(best["state"]); projector.load_state_dict(best["projector"]); student.eval()
    with torch.no_grad(): prediction, _ = student(student_values["test"], torch.ones(len(test), 1))
    _atomic_json(output / "teacher_training_trajectory.json", teacher_trajectory); _atomic_json(output / "loss_trajectory.json", trajectory)
    ledger = {"teacher": count_capacity(teacher), "student": count_capacity(student), "projector": count_capacity(projector), "teacher_training_steps": len(teacher_trajectory), "student_training_steps": len(trajectory), "fit_time_seconds": None, "inference_seconds": None, "compute_proxy": "active_parameters_times_training_steps"}
    return {**_write_run(output=output, config=config, job=job, frame=frame, folds=folds, test=test, prediction=prediction.numpy(), features=transformer, model_ledger=ledger, selection={"scope": "OUTER_TRAIN_INNER_VALIDATION_ONLY", "teacher": teacher_selection, "student_selected_epoch": best["epoch"], "student_validation_mae": best["validation_mae"], "balancing": balancing}, extra_manifest={"variant": variant, "teacher_frozen": all(not parameter.requires_grad for parameter in teacher.parameters()), "teacher_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}, smoke_only=smoke_only), "teacher_frozen": True}


def run_architecture_subjob(config, job, output, *, frame=None, folds=None, smoke_budget=False):
    torch.manual_seed(job["seed"])
    torch.use_deterministic_algorithms(True)
    if frame is None or folds is None:
        frame, folds = load_formal_data(config, job["fold"], Path(__file__).resolve().parents[2])
    train, validation, test = _split_frame(frame, folds)
    groups = _groups(frame)
    architecture = job["architecture"]
    modality_names = ["weather"] if architecture == "deployable_dense_compute_control" else [name for name in ("weather", "soil") if name in groups]
    if not modality_names:
        raise ValueError("architecture_modalities_unavailable")
    values, transformer = _prepare_modalities(train, validation, test, groups, modality_names)
    widths = {name: values["train"][name].shape[1] for name in modality_names}
    if architecture in {"shared_dense", "deployable_dense_compute_control"}:
        model = SharedDenseRegressor(widths, hidden=32)
    elif architecture == "modality_specific_dense":
        model = ModalitySpecificRegressor(widths, hidden=32)
    else:
        model = RoutedModalityRegressor(widths, hidden=32, experts=4)
    maximum = 5 if smoke_budget else config["training"]["max_epochs"]
    model, trajectory, selection = _fit_torch_model(model, values["train"], torch.ones(len(train), len(modality_names)), torch.tensor(train.target_yield.to_numpy(), dtype=torch.float32), values["validation"], torch.ones(len(validation), len(modality_names)), torch.tensor(validation.target_yield.to_numpy(), dtype=torch.float32), config, job["seed"], max_epochs=maximum)
    model.eval()
    with torch.no_grad():
        prediction, auxiliary = model(values["test"], torch.ones(len(test), len(modality_names)))
    _atomic_json(output / "training_trajectory.json", trajectory)
    if "routing_weights" in auxiliary:
        pd.DataFrame(auxiliary["routing_weights"].numpy()).to_csv(output / "routing_weights.csv", index=False)
    ledger = {**count_capacity(model), "training_steps": len(trajectory), "fit_time_seconds": None, "inference_seconds": None, "compute_proxy": "active_parameters_times_training_steps"}
    return _write_run(output=output, config=config, job=job, frame=frame, folds=folds, test=test, prediction=prediction.numpy(), features=transformer, model_ledger=ledger, selection=selection, extra_manifest={"architecture": architecture, "load_balancing_present": "load_balancing_loss" in auxiliary})


def derive_decision_rule_from_outer_train(config: dict[str, Any], *, bootstrap_draws: int = 2000) -> dict[str, Any]:
    """Estimate decision evidence scales without using any frozen outer-test row."""
    root = Path(__file__).resolve().parents[2]
    evidence_rows = []
    for fold in config["dataset"]["frozen_folds"]:
        frame, folds = load_formal_data(config, fold, root)
        train, validation, _ = _split_frame(frame, folds)
        groups = _groups(frame)
        deployable = groups.get("weather", [])
        privileged = deployable + groups.get("soil", [])
        if not deployable or len(privileged) == len(deployable):
            raise ValueError("decision_rule_requires_deployable_and_privileged_features")
        cluster_column = next((column for column in ("Environment", "environment", "Env") if column in validation), None)
        if cluster_column is None:
            raise ValueError("decision_rule_cluster_column_missing")
        for seed in config["runtime"]["seeds"]:
            baseline = _build_sklearn("ridge", deployable, {"alpha": 1.0}, seed)
            candidate = _build_sklearn("ridge", privileged, {"alpha": 1.0}, seed)
            baseline.fit(train[deployable], train.target_yield); candidate.fit(train[privileged], train.target_yield)
            baseline_prediction = baseline.predict(validation[deployable]); candidate_prediction = candidate.predict(validation[privileged])
            for index, row in enumerate(validation.itertuples()):
                evidence_rows.append({
                    "fold": fold, "seed": seed, "cluster": f"{getattr(row, cluster_column)}|{int(row.Year)}",
                    "baseline_error": abs(float(row.target_yield) - float(baseline_prediction[index])),
                    "candidate_error": abs(float(row.target_yield) - float(candidate_prediction[index])),
                })
    return derive_outer_train_decision_rule(pd.DataFrame(evidence_rows), unit=config["dataset"]["unit"], seed=config["runtime"]["seeds"][0], bootstrap_draws=bootstrap_draws)
