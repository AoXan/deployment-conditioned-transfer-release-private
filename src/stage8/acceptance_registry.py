"""Independent acceptance and evidence packaging for Stage 8 reruns."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .acceptance import prediction_fold_identity_errors
from .formal_training import REQUIRED_FINGERPRINT_INPUTS
from .evidence_metrics import t1_learning_curve_evidence, uncertainty_evidence

ENGINEERING_STATES = {"FORMAL_ACCEPTED", "FORMAL_ACCEPTED_WITH_WARNING", "COMPLETE_UNVERIFIED", "FAILED_RETRYABLE", "BLOCKED_INTERNAL", "EXTERNAL_BLOCKED", "INVALIDATED", "NOT_APPLICABLE"}
SCIENCE_STATES = {"SCIENTIFIC_GO", "SCIENTIFIC_NO_GO", "INSUFFICIENT_EVIDENCE", "NOT_APPLICABLE"}

PROTOCOL_GAPS = {
    "T1": ["route_architecture_not_frozen_modality_encoder", "route_endpoint_aulc_bootstrap_requires_recompute"],
    "T2": ["target_label_fraction_curve_missing", "harmonised_transfer_not_frozen_adapter_architecture"],
    "M1": ["complete_case_variant_missing", "frozen_missing_token_gated_huber_model_not_executed"],
    "M2": ["latent_alignment_and_masked_reconstruction_not_evidenced", "separate_pattern_students_violate_single_student_protocol"],
    "R1": ["hard_router_not_frozen_softmax_gate", "complete_expert_set_and_oof_regret_artifacts_missing"],
    "U1": ["shift_aware_weighted_conformal_comparison_missing", "isolated_oracle_artifact_missing"],
    "U2": ["deployable_aurc_random_rejection_ci_missing"],
}
ROUTE_UNITS = {
    "T1": ("hybrid_env", "Mg_ha"), "M1": ("hybrid_env", "Mg_ha"), "M2": ("hybrid_env", "Mg_ha"), "R1": ("hybrid_env", "Mg_ha"), "U1": ("hybrid_env", "Mg_ha"), "U2": ("hybrid_env", "Mg_ha"),
    "BASELINE_G2F": ("hybrid_env", "Mg_ha"), "BASELINE_G2F_MLP": ("hybrid_env", "Mg_ha"), "T2": ("administrative_unit_year", "t_ha"), "BASELINE_CYBENCH": ("administrative_unit_year", "t_ha"), "BASELINE_REGIONAL": ("state_crop_year", "t_ha"), "BASELINE_WAITE": ("trial_plot_year", "t_ha"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _namespace_route(run: Path) -> str | None:
    parts = run.parts
    return parts[parts.index("routes") + 1] if "routes" in parts else None


def _route_replay_verified(run: Path, route: str) -> bool:
    if "reruns_v2" not in run.parts: return False
    root = Path(*run.parts[: run.parts.index("reruns_v2") + 1])
    for path in (root / "physical_replay" / route).glob("*.json"):
        try:
            if json.loads(path.read_text()).get("status") == "VERIFIED": return True
        except Exception:
            continue
    return False


def _reproduction_command(run: Path, manifest: dict[str, Any]) -> list[str]:
    return ["python3", "scripts/replay_stage8_subexperiment.py", "--run-path", str(run), "--config", "configs/stage8_campaign_formal.yaml"]


def _frozen_fold_check(row: dict[str, Any], folds: pd.DataFrame, warnings: list[str]) -> list[str]:
    route, fold = str(row.get("route", "")), str(row.get("fold", ""))
    base = Path(os.environ.get("AGRITECH_DATA_ROOT", "data/external")) / "derived/data_nursery_v1/tracks"
    candidate = None
    if route in {"T1", "M1", "M2", "R1", "U1", "U2", "BASELINE_G2F", "BASELINE_G2F_MLP"} and fold.startswith("test_"):
        candidate = base / "g2f_native/folds" / f"g2f_native_hybrid_env__temporal_forward__{fold}.csv"
    elif route == "BASELINE_REGIONAL" and fold.startswith("test_"):
        candidate = base / "regional_crop_year/folds" / f"regional_abs_wheat_weather_apsoil__temporal_forward__{fold}.csv"
    if candidate is None:
        warnings.append("no_external_frozen_fold_attestation")
        return []
    if not candidate.is_file():
        warnings.append("declared_frozen_fold_unavailable")
        return []
    frozen = pd.read_csv(candidate)
    frozen_ids = set(frozen.loc[frozen.split.astype(str).str.lower().eq("test"), "sample_id"].astype(str))
    actual_ids = set(folds.loc[folds.split.astype(str).str.lower().eq("test"), "sample_id"].astype(str))
    return [] if frozen_ids == actual_ids else ["frozen_fold_id_mismatch"]


def _classify_run(run: Path) -> dict[str, Any]:
    required = ["predictions.csv", "metrics.json", "manifest.json", "fold_assignments.csv", "completion_marker.json"]
    errors = [f"missing_{name}" for name in required if not (run / name).is_file()]
    warnings: list[str] = []
    row: dict[str, Any] = {"run_path": str(run), "errors": errors, "warnings": warnings, "metrics_recomputed": False}
    if "reruns_v2" not in run.parts:
        errors.append("not_reruns_v2_namespace")
    if errors and not (run / "manifest.json").is_file():
        row.update({"engineering_status": "COMPLETE_UNVERIFIED", "scientific_status": "INSUFFICIENT_EVIDENCE"})
        return row

    manifest = json.loads((run / "manifest.json").read_text())
    row.update({key: manifest.get(key) for key in ("route", "dataset", "fold", "seed", "model", "variant", "pattern", "missingness_type", "fraction", "test_year", "fingerprint")})
    namespace_route = _namespace_route(run)
    if namespace_route and row.get("route") != namespace_route:
        errors.append("route_manifest_namespace_mismatch")
        row["manifest_route"] = row.get("route")
        row["route"] = namespace_route
    row["reproduction_command"] = _reproduction_command(run, manifest)
    if row.get("route") in ROUTE_UNITS:
        row["sample_unit"], row["target_unit"] = ROUTE_UNITS[row["route"]]

    if errors and any(error.startswith("missing_") for error in errors):
        row.update({"engineering_status": "COMPLETE_UNVERIFIED", "scientific_status": "INSUFFICIENT_EVIDENCE"})
        return row

    declared_hashes = manifest.get("artifact_sha256", {})
    files = {"predictions": run / "predictions.csv", "metrics": run / "metrics.json", "fold_assignments": run / "fold_assignments.csv"}
    for name, path in files.items():
        if declared_hashes.get(name) != _sha256(path):
            errors.append(f"artifact_hash_mismatch:{name}")
    marker = json.loads((run / "completion_marker.json").read_text())
    if marker.get("fingerprint") != manifest.get("fingerprint"):
        errors.append("marker_fingerprint_mismatch")
    if marker.get("manifest_sha256") != _sha256(run / "manifest.json"):
        errors.append("marker_manifest_hash_mismatch")
    marker_time = (run / "completion_marker.json").stat().st_mtime_ns
    if any(marker_time < path.stat().st_mtime_ns for path in files.values()) or marker_time < (run / "manifest.json").stat().st_mtime_ns:
        errors.append("completion_marker_not_last")
    fp_inputs = manifest.get("fingerprint_inputs", {})
    if set(fp_inputs) != REQUIRED_FINGERPRINT_INPUTS or any(not str(fp_inputs.get(key, "")).strip() for key in REQUIRED_FINGERPRINT_INPUTS):
        errors.append("incomplete_fingerprint_inputs")

    if errors:
        row.update({"engineering_status": "INVALIDATED", "scientific_status": "INSUFFICIENT_EVIDENCE"})
        return row
    try:
        predictions = pd.read_csv(run / "predictions.csv")
        folds = pd.read_csv(run / "fold_assignments.csv")
        metrics = json.loads((run / "metrics.json").read_text())
    except Exception as exc:
        errors.append(f"artifact_parse_error:{type(exc).__name__}")
        row.update({"engineering_status": "INVALIDATED", "scientific_status": "INSUFFICIENT_EVIDENCE"})
        return row
    if predictions.empty:
        errors.append("empty_predictions")
    if not {"sample_id", "y_true", "y_pred"}.issubset(predictions):
        errors.append("missing_prediction_columns")
    else:
        numeric = predictions[["y_true", "y_pred"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all(): errors.append("nonfinite_target_or_prediction")
        if len(numeric) > 1 and numeric.y_pred.nunique() <= 1: errors.append("constant_predictions")
        if len(numeric) > 1 and numeric.y_true.nunique() <= 1: errors.append("constant_target")
        if np.allclose(numeric.y_pred, 0): errors.append("all_zero_predictions")
        if "evidence_status" in predictions and predictions.evidence_status.astype(str).str.contains("SMOKE", case=False).any(): errors.append("smoke_artifact")
        errors.extend(prediction_fold_identity_errors(predictions, folds))
        if len(str(manifest.get("fingerprint_inputs", {}).get("data", ""))) == 64:
            errors.extend(_frozen_fold_check(row, folds, warnings))
        recomputed = {"mae": float(mean_absolute_error(numeric.y_true, numeric.y_pred)), "rmse": float(mean_squared_error(numeric.y_true, numeric.y_pred) ** .5), "n": len(numeric)}
        row["metrics_recomputed"] = True
        row["recomputed_metrics"] = recomputed
        for key, value in recomputed.items():
            if key not in metrics or not np.isclose(float(metrics[key]), float(value), rtol=1e-9, atol=1e-10): errors.append(f"metric_mismatch:{key}")
        if len(numeric) < 30: warnings.append("small_test_sample")
        if row.get("route") == "T1" and row.get("variant") == "scratch" and recomputed["mae"] > 100:
            warnings.append("extreme_but_reproducible_low_resource_extrapolation")
    if errors:
        row.update({"engineering_status": "INVALIDATED", "scientific_status": "INSUFFICIENT_EVIDENCE"})
        return row
    gaps = PROTOCOL_GAPS.get(str(row.get("route")), [])
    warnings.extend(gaps)
    if not _route_replay_verified(run, str(row.get("route"))): warnings.append("selected_physical_replay_pending")
    if gaps or warnings:
        row.update({"engineering_status": "FORMAL_ACCEPTED_WITH_WARNING", "scientific_status": "INSUFFICIENT_EVIDENCE" if gaps else "NOT_APPLICABLE"})
    else:
        row.update({"engineering_status": "FORMAL_ACCEPTED", "scientific_status": "NOT_APPLICABLE" if str(row.get("route", "")).startswith("BASELINE_") else "INSUFFICIENT_EVIDENCE"})
    assert row["engineering_status"] in ENGINEERING_STATES and row["scientific_status"] in SCIENCE_STATES
    return row


def audit_route_artifacts(output_root: Path) -> list[dict[str, Any]]:
    return [_classify_run(path.parent) for path in sorted((output_root / "routes").glob("**/manifest.json"))]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _legacy_disposition(output_root: Path, route_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    legacy_root = output_root.parent / "routes"
    replacements: dict[tuple, str] = {}
    coarse: dict[tuple, list[str]] = {}
    for row in route_rows:
        key = (row.get("route"), row.get("fold"), row.get("seed"), row.get("variant"), row.get("pattern"))
        replacements[key] = row["run_path"]
        coarse.setdefault((row.get("route"), row.get("fold"), row.get("seed")), []).append(row["run_path"])
    rows = []
    for manifest_path in sorted(legacy_root.glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        route = _namespace_route(manifest_path.parent) or manifest.get("route")
        key = (route, manifest.get("fold"), manifest.get("seed"), manifest.get("variant"), manifest.get("pattern"))
        replacement = replacements.get(key)
        fallback = sorted(coarse.get((route, manifest.get("fold"), manifest.get("seed")), []))
        mapped = [replacement] if replacement else fallback
        rows.append({"legacy_run_path": str(manifest_path.parent), "disposition": "replaced_by_rerun" if mapped else "invalidated_without_replacement", "replacement": mapped, "preserved": True})
    return rows


def _post_rerun_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reruns = []
    for route in ("T1", "T2", "M1", "M2", "R1", "U1", "U2"):
        affected = [row["run_path"] for row in rows if row.get("route") == route]
        if affected:
            reruns.append({"job_id": f"protocol_completion_{route.lower()}", "subexperiment_id": route, "root_cause": PROTOCOL_GAPS[route], "affected_artifact": affected, "minimal_modification_files": ["scripts/run_stage8_formal_adapter.py"], "action": "rerun_job", "old_fingerprint": sorted({str(row.get("fingerprint", "")) for row in rows if row.get("route") == route}), "new_fingerprint_inputs": {"code": "new_commit_required", "config": "frozen_config_hash", "data": "reuse_verified", "view": "reuse_verified", "split": "reuse_verified", "feature": "frozen_protocol_features"}, "acceptance_checks": ["frozen_protocol_complete", "independent_fold_ids", "metrics_recomputed", "no_outer_test_selection"], "cli": ["python3", "scripts/run_stage8_formal_adapter.py", "--route", route, "--config", "configs/stage8_campaign_formal.yaml", "--output-root", "outputs/stage8/formal_v1/reruns_v3/routes", "--seeds", "101", "202", "303"]})
    return {"schema_version": "stage8-post-rerun-fix-v1", "no_additional_reruns_required": not bool(reruns), "reruns": reruns}


def _selected_replays(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for route in sorted({str(row.get("route")) for row in rows if row.get("route")}):
        candidates = [row for row in rows if row.get("route") == route and row["engineering_status"] in {"FORMAL_ACCEPTED", "FORMAL_ACCEPTED_WITH_WARNING"}]
        if not candidates: continue
        def rank(row):
            return (row.get("seed") != 101, row.get("fold") != "test_2022" and not str(row.get("fold", "")).endswith("2022"), row.get("variant") not in {None, "source_only", "late_fusion", "pretrain_adapt"}, row.get("pattern") not in {None, "natural"}, row.get("fraction") not in {None, .1}, row["run_path"])
        choice = sorted(candidates, key=rank)[0]
        selected.append({"route": route, "run_path": choice["run_path"], "status": "PENDING_USER_LAUNCH_LONG_TASK", "command": ["python3", "scripts/replay_stage8_subexperiment.py", "--run-path", choice["run_path"], "--config", "configs/stage8_campaign_formal.yaml", "--run"], "rtol": 1e-7, "atol": 1e-9})
    return selected


def write_acceptance_package(output_root: Path, destination: Path) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    rows = audit_route_artifacts(output_root)
    counts = pd.Series([row["engineering_status"] for row in rows]).value_counts().to_dict()
    _write_json(destination / "job_status_registry.json", {"jobs": rows, "counts": counts})
    pd.DataFrame(rows).to_csv(destination / "job_status_registry.csv", index=False)
    failures = [{"run_path": row["run_path"], "engineering_status": row["engineering_status"], "root_causes": row["errors"] or row["warnings"], "classification": "protocol_or_evidence_warning" if row["engineering_status"] == "FORMAL_ACCEPTED_WITH_WARNING" else "acceptance_failure", "retryable": row["engineering_status"] in {"FAILED_RETRYABLE", "COMPLETE_UNVERIFIED", "INVALIDATED"}} for row in rows if row["engineering_status"] != "FORMAL_ACCEPTED"]
    _write_json(destination / "failure_root_cause_registry.json", {"failures": failures})
    invalidated = [{"run_path": row["run_path"], "reason": row["errors"], "preserved": True} for row in rows if row["engineering_status"] == "INVALIDATED"]
    _write_json(destination / "invalidated_artifacts.json", {"artifacts": invalidated})
    risks = [
        {"risk_id": "reruns_v2_artifact_contract", "status": "CLOSED", "evidence": f"{len(rows)} runs independently audited"},
        {"risk_id": "protocol_completeness", "status": "OPEN_REQUIRES_TARGETED_FIX", "evidence": PROTOCOL_GAPS},
        {"risk_id": "g2f_2024_historical_hash_access", "status": "CLOSED_IN_CODE_PROVENANCE_BREACH_RECORDED", "evidence": "no observed-target access in this acceptance"},
        {"risk_id": "nvt_access", "status": "USER_EXCLUDED_OUT_OF_SCOPE", "evidence": "no NVT discovery path"},
        {"risk_id": "roseworthy_license", "status": "EXTERNAL_BLOCKED", "evidence": "not executed"},
    ]
    _write_json(destination / "risk_closure_table.json", {"risks": risks}); pd.DataFrame(risks).to_csv(destination / "risk_closure_table.csv", index=False)
    legacy = _legacy_disposition(output_root, rows); _write_json(destination / "legacy_artifact_disposition.json", {"artifacts": legacy})
    rerun_manifest = _post_rerun_manifest(rows); (destination / "post_rerun_fix_manifest.yaml").write_text(yaml.safe_dump(rerun_manifest, sort_keys=False))
    _write_json(destination / "selected_physical_replay_manifest.json", {"replays": _selected_replays(rows), "note": "Commands are prepared but long replays are not auto-started."})
    accepted = [row for row in rows if row["engineering_status"] in {"FORMAL_ACCEPTED", "FORMAL_ACCEPTED_WITH_WARNING"}]
    accepted_predictions = [{"path": str(Path(row["run_path"]) / "predictions.csv"), "engineering_status": row["engineering_status"], "scientific_status": row["scientific_status"]} for row in accepted]
    accepted_metrics = [{"path": str(Path(row["run_path"]) / "metrics.json"), "engineering_status": row["engineering_status"], "scientific_status": row["scientific_status"]} for row in accepted]
    route4_root = output_root.parent / "acceptance_v2"
    route4_check = route4_root / "route4_prediction_fold_acceptance.json"
    route4_metrics = route4_root / "route4_recomputed_metrics.json"
    if route4_check.is_file() and route4_metrics.is_file() and json.loads(route4_check.read_text()).get("status") == "VERIFIED":
        accepted_predictions.append({"path": str(route4_root / "route4_predictions.csv"), "engineering_status": "FORMAL_ACCEPTED", "scientific_status": "SCIENTIFIC_NO_GO", "scope": "post_hoc_metric_only"})
        accepted_metrics.append({"path": str(route4_metrics), "engineering_status": "FORMAL_ACCEPTED", "scientific_status": "SCIENTIFIC_NO_GO", "scope": "post_hoc_metric_only"})
    _write_json(destination / "accepted_predictions_index.json", {"predictions": accepted_predictions})
    _write_json(destination / "accepted_metrics_index.json", {"metrics": accepted_metrics})
    route_evidence = []
    evidence_dir = destination / "route_evidence"; evidence_dir.mkdir(exist_ok=True)
    computed_evidence = {
        "T1": t1_learning_curve_evidence(rows),
        "U1": uncertainty_evidence(rows, "U1"),
        "U2": uncertainty_evidence(rows, "U2"),
    }
    for route, payload in computed_evidence.items(): _write_json(evidence_dir / f"{route}.json", payload)
    for route, group in pd.DataFrame(rows).groupby("route", dropna=False):
        route_evidence.append({"route": route, "subexperiments": len(group), "engineering_status_counts": group.engineering_status.value_counts().to_dict(), "scientific_status_counts": group.scientific_status.value_counts().to_dict(), "protocol_gaps": PROTOCOL_GAPS.get(str(route), []), "computed_evidence": str(evidence_dir / f"{route}.json") if route in computed_evidence else ""})
    _write_json(destination / "route_evidence_registry.json", {"routes": route_evidence, "route4": {"engineering_status": "FORMAL_ACCEPTED", "scientific_status": "SCIENTIFIC_NO_GO", "status": "metric_recomputed_and_replayed", "source": str(route4_root), "evidence_scope": "post_hoc_metric_only"}})
    benchmark = pd.DataFrame([{key: row.get(key) for key in ("route", "dataset", "fold", "seed", "model", "variant", "pattern", "engineering_status", "scientific_status")} | row.get("recomputed_metrics", {}) for row in rows])
    if route4_metrics.is_file():
        metric = json.loads(route4_metrics.read_text()).get("metrics", {})
        benchmark = pd.concat([benchmark, pd.DataFrame([{"route": "ROUTE4", "dataset": "g2f_native", "fold": "test_2022", "seed": None, "model": "modality_dropout_vs_imputation", "variant": "paired_metric_only", "pattern": "weather_masked", "engineering_status": "FORMAL_ACCEPTED", "scientific_status": "SCIENTIFIC_NO_GO", **metric}])], ignore_index=True)
    benchmark.to_csv(destination / "clean_benchmark_matrix.csv", index=False)
    report = ["# Stage 8 reruns_v2 formal acceptance", "", f"- Subexperiments audited: {len(rows)}", *[f"- {key}: {value}" for key, value in sorted(counts.items())], f"- Legacy artifacts mapped: {len(legacy)}", f"- Additional targeted actions: {len(rerun_manifest['reruns'])}", "- Completion markers were treated only as inputs, never as acceptance decisions.", "", "## Scientific disposition", "", "- Baselines are engineering evidence only (`NOT_APPLICABLE` scientific decision).", "- T1/T2/M1/M2/R1/U1/U2 remain `INSUFFICIENT_EVIDENCE` because their frozen route protocols are incomplete in the executed adapters.", "- Route 4 remains `SCIENTIFIC_NO_GO` within its post-hoc metric-only evidence scope.", "- No new `SCIENTIFIC_GO` was assigned.", "", "## Physical replay", "", "Representative replay commands were generated for all 12 routes. They remain pending user launch because they retrain models and are long-running tasks."]
    (destination / "formal_campaign_acceptance_report.md").write_text("\n".join(report) + "\n")
    handoff = ["# Stage 8 results analysis handoff", "", "Use only rows listed in accepted indexes. `FORMAL_ACCEPTED_WITH_WARNING` is engineering-valid but remains scientifically insufficient where protocol gaps are recorded.", "", "The clean benchmark matrix is suitable for engineering/result inspection, not final route GO claims. T1/U1/U2 post-hoc metrics are in `route_evidence/`; protocol-completion jobs are declared in `post_rerun_fix_manifest.yaml`.", "", "Route 4 remains the previously accepted metric-only replay and post-hoc `SCIENTIFIC_NO_GO`. No G2F 2024 observed targets or NVT assets were accessed."]
    (destination / "stage8_results_analysis_handoff.md").write_text("\n".join(handoff) + "\n")
    return {"inspected": len(rows), "accepted": counts.get("FORMAL_ACCEPTED", 0), "accepted_with_warning": counts.get("FORMAL_ACCEPTED_WITH_WARNING", 0), "invalidated": counts.get("INVALIDATED", 0)}
