"""Shared formal training and immutable artifact contracts."""
from __future__ import annotations
import hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .formal_routes import validate_formal_output

BLOCKED={"production","production_t","harvest_area","area_ha","yield","target_yield","target_value","observed_yield","observed_yield_t_ha","grain_yield_kg_ha","Yield_Mg_ha","sample_id","target_name","target_unit","source_target_path","source_file"}

def feature_columns(frame: pd.DataFrame, target: str)->list[str]:
    return [c for c in frame.columns if c != target and c not in BLOCKED and not c.lower().startswith(("pred_","target_","source_"))]

def make_pipeline(frame: pd.DataFrame, model: str, seed: int, *, model_params: dict | None = None)->Pipeline:
    numeric=[c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and not frame[c].isna().all()]
    categorical=[c for c in frame.columns if c not in numeric and frame[c].nunique(dropna=True)<=100]
    pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    estimators={"ridge":Ridge(alpha=1.0,solver="lsqr"),"huber":HuberRegressor(alpha=1e-4,epsilon=1.35,max_iter=500),"rf":RandomForestRegressor(n_estimators=150,min_samples_leaf=2,n_jobs=-1,random_state=seed),"mlp":MLPRegressor(hidden_layer_sizes=(128,64),early_stopping=True,validation_fraction=.15,max_iter=300,random_state=seed)}
    if model_params:
        estimators[model].set_params(**model_params)
    return Pipeline([("pre",pre),("model",estimators[model])])

def fingerprint(payload: dict)->str:
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()

REQUIRED_FINGERPRINT_INPUTS = {"code", "config", "data", "view", "split", "feature"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def write_artifacts(
    out: Path,
    predictions: pd.DataFrame,
    manifest: dict,
    *,
    expected_ids: set[str],
    fold_assignments: pd.DataFrame,
    fingerprint_inputs: dict[str, str],
) -> dict:
    missing = REQUIRED_FINGERPRINT_INPUTS - set(fingerprint_inputs)
    if missing:
        raise ValueError(f"missing_fingerprint_inputs:{sorted(missing)}")
    if any(not str(fingerprint_inputs[key]).strip() for key in REQUIRED_FINGERPRINT_INPUTS):
        raise ValueError("empty_fingerprint_input")
    errors=validate_formal_output(predictions.to_dict("records"),expected_ids)
    if errors: raise ValueError(";".join(errors))
    test_folds = fold_assignments
    if "split" not in test_folds:
        raise ValueError("fold_assignments_missing_split")
    test_folds = test_folds[test_folds["split"].astype(str).str.lower().eq("test")]
    fold_ids = set(test_folds["sample_id"].astype(str))
    if fold_ids != set(map(str, expected_ids)):
        raise ValueError("fold_assignment_expected_id_mismatch")
    out.mkdir(parents=True,exist_ok=True)
    fp=fingerprint({"manifest": manifest, "inputs": fingerprint_inputs})
    predictions=predictions.copy(); predictions["fingerprint"]=fp; predictions["evidence_status"]="FORMAL"
    _atomic_csv(out/"predictions.csv", predictions)
    _atomic_csv(out/"fold_assignments.csv", fold_assignments)
    y=predictions.y_true.to_numpy(); p=predictions.y_pred.to_numpy()
    metrics={"mae":float(mean_absolute_error(y,p)),"rmse":float(mean_squared_error(y,p)**.5),"n":len(y)}
    _atomic_json(out/"metrics.json", metrics)
    full={**manifest,"fingerprint":fp,"fingerprint_inputs":fingerprint_inputs,"environment":{"python":platform.python_version()},"prediction_rows":len(predictions),"artifact_sha256":{"predictions":_sha256(out/"predictions.csv"),"metrics":_sha256(out/"metrics.json"),"fold_assignments":_sha256(out/"fold_assignments.csv")}}
    _atomic_json(out/"manifest.json", full)
    _atomic_json(out/"completion_marker.json", {"status":"COMPLETE_UNVERIFIED","fingerprint":fp,"manifest_sha256":_sha256(out/"manifest.json")})
    return metrics
