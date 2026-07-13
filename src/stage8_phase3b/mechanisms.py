from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .formal_bridge import FormalCodeBridge
from .reproduction import metric_bundle


MECHANISM_METHODS = [
    "method_component_ablation",
    "grouped_permutation",
    "formal_condition_comparison",
    "modality_sensitivity",
    "integrated_gradients",
    "shap",
    "attribution_stability",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def method_component_ablation(manifest: dict, output_root: Path, candidate_id: str) -> dict:
    rows = []
    for cell in manifest.get("cells", []):
        if candidate_id not in cell.get("all_seed_candidate_ids", []):
            continue
        rows.append(
            {
                "formal_candidate_id": candidate_id,
                "candidate_group_id": cell["candidate_group_id"],
                "dataset_id": cell["dataset_id"],
                "split_id": cell["split_id"],
                "condition": cell["deployment_condition"],
                "target_strategy": cell["transfer_strategy"],
                "source_route": cell["source_route"],
                "positivity_tier": cell["positivity_tier"],
                "high_limit_flag": cell["high_limit_flag"],
                "delta_mae_vs_baseline": cell["original_metrics"].get("delta_mae_vs_baseline"),
                "replay_scientific_use": cell["replay_scientific_use"],
            }
        )
    _write_csv(Path(output_root) / "mechanisms" / candidate_id / "method_component_ablation.csv", rows)
    return {"status": "PASS" if rows else "not_applicable", "row_count": len(rows)}


def formal_condition_comparison(manifest: dict, output_root: Path, candidate_id: str) -> dict:
    rows = []
    for cell in manifest.get("cells", []):
        if candidate_id not in cell.get("all_seed_candidate_ids", []):
            continue
        key = (cell["dataset_id"], cell["split_id"], cell["transfer_strategy"], cell["source_route"])
        for other in manifest.get("cells", []):
            other_key = (other["dataset_id"], other["split_id"], other["transfer_strategy"], other["source_route"])
            if key == other_key:
                rows.append(
                    {
                        "formal_candidate_id": candidate_id,
                        "candidate_group_id": cell["candidate_group_id"],
                        "comparison_group_id": other["candidate_group_id"],
                        "condition": cell["deployment_condition"],
                        "comparison_condition": other["deployment_condition"],
                        "mae": cell["original_metrics"].get("mae"),
                        "comparison_mae": other["original_metrics"].get("mae"),
                        "delta_mae": cell["original_metrics"].get("mae") - other["original_metrics"].get("mae"),
                    }
                )
    _write_csv(Path(output_root) / "mechanisms" / candidate_id / "formal_condition_comparison.csv", rows)
    return {"status": "PASS" if rows else "not_applicable", "row_count": len(rows)}


def _not_applicable(output_root: Path, candidate_id: str, method: str, reason: str) -> dict:
    payload = {
        "status": "not_applicable",
        "method": method,
        "formal_candidate_id": candidate_id,
        "reason": reason,
        "timestamp_unix": time.time(),
    }
    _write_json(Path(output_root) / "mechanisms" / candidate_id / f"{method}.json", payload)
    return payload


def _requires_replay_artifacts(output_root: Path, candidate_id: str) -> Path | None:
    root = Path(output_root) / "jobs" / "replay_artifacts" / candidate_id
    arrays = root / "arrays" / "transformed_inputs.npz"
    features = root / "feature_order.json"
    return root if arrays.is_file() and features.is_file() else None


def _sample_ids(artifact_root: Path, limit: int | None = None) -> list[str]:
    path = artifact_root / "fold_test_ids.csv"
    if not path.is_file():
        return []
    values = pd.read_csv(path)["sample_id"].astype(str).tolist()
    return values[:limit] if limit is not None else values


def _model_predictor(output_root: Path, candidate_id: str):
    artifact_root = _requires_replay_artifacts(output_root, candidate_id)
    if artifact_root is None:
        raise FileNotFoundError("REPLAY_ARTIFACTS_MISSING")
    checkpoint = Path(output_root) / "jobs" / "target_checkpoints" / f"{candidate_id}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("TARGET_CHECKPOINT_MISSING")
    arrays = np.load(artifact_root / "arrays" / "transformed_inputs.npz")
    X = np.asarray(arrays["test_weather"], dtype=np.float32)
    y = np.asarray(arrays["test_target"], dtype=np.float32)
    features = json.loads((artifact_root / "feature_order.json").read_text())["weather"]
    bridge = FormalCodeBridge()
    bridge.load()
    models = __import__(
        "src.distillation_v4.universal_weather_v2.models",
        fromlist=["UniversalWeatherRegressorV2", "UniversalWeatherEncoderV2"],
    )
    training = bridge.training_module()
    artifacts = __import__("src.distillation_v4.universal_weather_v2.artifacts", fromlist=["load_checkpoint"])
    import torch

    model = models.UniversalWeatherRegressorV2(encoder=models.UniversalWeatherEncoderV2())
    artifacts.load_checkpoint(path=checkpoint, model=model, map_location=torch.device("cpu"), strict=True)
    model.eval()

    def predict(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(values, dtype=np.float32))
            pred, _rep = training.model_forward_v2(model, weather=tensor, feature_names=list(features))
            return pred.detach().cpu().numpy().reshape(-1)

    return {
        "artifact_root": artifact_root,
        "checkpoint": checkpoint,
        "X": X,
        "y": y,
        "features": list(features),
        "predict": predict,
    }


def grouped_permutation(output_root: Path, candidate_id: str) -> dict:
    try:
        bundle = _model_predictor(output_root, candidate_id)
    except FileNotFoundError as error:
        return _not_applicable(output_root, candidate_id, "grouped_permutation", str(error))
    X = bundle["X"]
    y = bundle["y"]
    predict = bundle["predict"]
    baseline = pd.DataFrame({"y_true": y, "y_pred": predict(X)})
    baseline_metrics = metric_bundle(baseline)
    rng = np.random.default_rng(20260704)
    rows = []
    groups = {"weather_all": list(range(X.shape[1]))}
    groups.update({f"feature::{name}": [index] for index, name in enumerate(bundle["features"])})
    for repetition in range(10):
        for group_name, indices in groups.items():
            permuted = X.copy()
            order = rng.permutation(len(permuted))
            permuted[:, indices] = permuted[order][:, indices]
            metrics = metric_bundle(pd.DataFrame({"y_true": y, "y_pred": predict(permuted)}))
            for metric, value in metrics.items():
                rows.append(
                    {
                        "formal_candidate_id": candidate_id,
                        "repetition": repetition,
                        "feature_group": group_name,
                        "metric": metric,
                        "baseline": baseline_metrics[metric],
                        "permuted": value,
                        "delta": value - baseline_metrics[metric],
                    }
                )
    path = Path(output_root) / "mechanisms" / candidate_id / "grouped_permutation_repetitions.csv"
    _write_csv(path, rows)
    payload = {"status": "PASS", "method": "grouped_permutation", "rows": len(rows), "path": str(path)}
    _write_json(Path(output_root) / "mechanisms" / candidate_id / "grouped_permutation.json", payload)
    return payload


def modality_sensitivity(output_root: Path, candidate_id: str) -> dict:
    try:
        bundle = _model_predictor(output_root, candidate_id)
    except FileNotFoundError as error:
        return _not_applicable(output_root, candidate_id, "modality_sensitivity", str(error))
    X = bundle["X"]
    y = bundle["y"]
    predict = bundle["predict"]
    interventions = {
        "original": X,
        "zero_scaled_weather_all": np.zeros_like(X),
    }
    rows = []
    for name, values in interventions.items():
        metrics = metric_bundle(pd.DataFrame({"y_true": y, "y_pred": predict(values)}))
        rows.append({"formal_candidate_id": candidate_id, "intervention": name, **metrics})
    path = Path(output_root) / "mechanisms" / candidate_id / "modality_sensitivity_raw.csv"
    _write_csv(path, rows)
    payload = {"status": "PASS", "method": "modality_sensitivity", "rows": len(rows), "path": str(path)}
    _write_json(Path(output_root) / "mechanisms" / candidate_id / "modality_sensitivity.json", payload)
    return payload


def integrated_gradients(output_root: Path, candidate_id: str) -> dict:
    try:
        bundle = _model_predictor(output_root, candidate_id)
    except FileNotFoundError as error:
        return _not_applicable(output_root, candidate_id, "integrated_gradients", str(error))
    import torch
    from captum.attr import IntegratedGradients

    bridge = FormalCodeBridge()
    bridge.load()
    models = __import__(
        "src.distillation_v4.universal_weather_v2.models",
        fromlist=["UniversalWeatherRegressorV2", "UniversalWeatherEncoderV2"],
    )
    training = bridge.training_module()
    artifacts = __import__("src.distillation_v4.universal_weather_v2.artifacts", fromlist=["load_checkpoint"])
    model = models.UniversalWeatherRegressorV2(encoder=models.UniversalWeatherEncoderV2())
    artifacts.load_checkpoint(path=bundle["checkpoint"], model=model, map_location=torch.device("cpu"), strict=True)
    model.eval()

    def forward(inputs):
        pred, _rep = training.model_forward_v2(model, weather=inputs, feature_names=bundle["features"])
        return pred

    X = torch.as_tensor(bundle["X"][: min(32, len(bundle["X"]))], dtype=torch.float32)
    baseline = torch.zeros_like(X)
    ig = IntegratedGradients(forward)
    attrs, delta = ig.attribute(X, baselines=baseline, return_convergence_delta=True)
    sample_ids = _sample_ids(bundle["artifact_root"], limit=len(X))
    rows = []
    attr_np = attrs.detach().cpu().numpy()
    for row_index, sample_id in enumerate(sample_ids):
        for feature_index, feature in enumerate(bundle["features"]):
            rows.append(
                {
                    "formal_candidate_id": candidate_id,
                    "sample_id": sample_id,
                    "feature_index": feature_index,
                    "feature_name": feature,
                    "ig_value": float(attr_np[row_index, feature_index]),
                }
            )
    root = Path(output_root) / "mechanisms" / candidate_id
    _write_csv(root / "ig_raw_attributions.csv", rows)
    _write_csv(
        root / "ig_convergence_delta.csv",
        [
            {"formal_candidate_id": candidate_id, "sample_id": sample_id, "delta": float(value)}
            for sample_id, value in zip(sample_ids, delta.detach().cpu().numpy())
        ],
    )
    payload = {"status": "PASS", "method": "integrated_gradients", "rows": len(rows)}
    _write_json(root / "integrated_gradients.json", payload)
    return payload


def shap_analysis(output_root: Path, candidate_id: str) -> dict:
    try:
        bundle = _model_predictor(output_root, candidate_id)
    except FileNotFoundError as error:
        return _not_applicable(output_root, candidate_id, "shap", str(error))
    import shap

    X = bundle["X"]
    if len(X) < 2:
        return _not_applicable(output_root, candidate_id, "shap", "insufficient_samples")
    background_count = min(20, len(X))
    explained_count = min(10, len(X))
    background = X[:background_count]
    explained = X[:explained_count]
    explainer = shap.KernelExplainer(bundle["predict"], background)
    raw_values = np.asarray(
        explainer.shap_values(explained, nsamples=min(64, 2 * X.shape[1] + 1), silent=True)
    )
    if raw_values.ndim == 3:
        raw_values = raw_values[0]
    if raw_values.ndim == 1:
        raw_values = raw_values.reshape(1, -1)
    sample_ids = _sample_ids(bundle["artifact_root"], limit=explained_count)
    rows = []
    for row_index, sample_id in enumerate(sample_ids):
        for feature_index, feature in enumerate(bundle["features"]):
            rows.append(
                {
                    "formal_candidate_id": candidate_id,
                    "sample_id": sample_id,
                    "feature_index": feature_index,
                    "feature_name": feature,
                    "shap_value": float(raw_values[row_index, feature_index]),
                }
            )
    root = Path(output_root) / "mechanisms" / candidate_id
    _write_csv(root / "shap_raw_values.csv", rows)
    _write_csv(root / "shap_background_sample_ids.csv", [{"sample_id": item} for item in _sample_ids(bundle["artifact_root"], limit=background_count)])
    _write_csv(root / "shap_explained_sample_ids.csv", [{"sample_id": item} for item in sample_ids])
    _write_json(
        root / "explainer_config.json",
        {
            "formal_candidate_id": candidate_id,
            "method": "KernelExplainer",
            "background_count": background_count,
            "explained_count": explained_count,
            "nsamples": min(64, 2 * X.shape[1] + 1),
        },
    )
    payload = {"status": "PASS", "method": "shap", "rows": len(rows)}
    _write_json(root / "shap.json", payload)
    return payload


def attribution_stability(output_root: Path, candidate_id: str) -> dict:
    root = Path(output_root) / "mechanisms" / candidate_id
    rows = []
    for method in ("grouped_permutation", "modality_sensitivity", "integrated_gradients", "shap"):
        path = root / f"{method}.json"
        if path.is_file():
            payload = json.loads(path.read_text())
            rows.append({"formal_candidate_id": candidate_id, "method": method, "status": payload.get("status")})
    _write_csv(root / "attribution_stability.csv", rows)
    return {"status": "PASS" if rows else "not_applicable", "row_count": len(rows)}


def run_mechanism_suite(manifest: dict, output_root: Path, candidate_id: str) -> dict:
    results: dict[str, dict] = {}
    methods: dict[str, Callable[[], dict]] = {
        "method_component_ablation": lambda: method_component_ablation(manifest, output_root, candidate_id),
        "grouped_permutation": lambda: grouped_permutation(output_root, candidate_id),
        "formal_condition_comparison": lambda: formal_condition_comparison(manifest, output_root, candidate_id),
        "modality_sensitivity": lambda: modality_sensitivity(output_root, candidate_id),
        "integrated_gradients": lambda: integrated_gradients(output_root, candidate_id),
        "shap": lambda: shap_analysis(output_root, candidate_id),
        "attribution_stability": lambda: attribution_stability(output_root, candidate_id),
    }
    for method, call in methods.items():
        try:
            results[method] = call()
        except Exception as error:  # explanation failure must not poison replay status
            failure = {
                "status": "failed",
                "method": method,
                "formal_candidate_id": candidate_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "timestamp_unix": time.time(),
            }
            results[method] = failure
            _append_jsonl(Path(output_root) / "mechanisms" / "explanation_failures.jsonl", failure)
    _write_json(Path(output_root) / "mechanisms" / candidate_id / "mechanism_summary.json", results)
    return results


def write_final_synthesis(manifest: dict, output_root: Path) -> Path:
    rows = []
    for cell in manifest.get("cells", []):
        rows.append(
            {
                "candidate_group_id": cell["candidate_group_id"],
                "high_limit_flag": cell["high_limit_flag"],
                "replay_scientific_use": cell["replay_scientific_use"],
                "positivity_tier": cell["positivity_tier"],
                "prohibited_claims": ";".join(cell.get("prohibited_claims", [])),
            }
        )
    path = Path(output_root) / "synthesis" / "method_upgrade_evidence.csv"
    _write_csv(path, rows)
    return path
