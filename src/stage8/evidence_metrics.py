"""Post-hoc metric recomputation from accepted Stage 8 row predictions."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .metrics import conformal_metrics, deployable_aurc


def _aulc(points: dict[float, float]) -> float:
    pairs = sorted((float(k), float(v)) for k, v in points.items())
    return float(np.trapezoid([v for _, v in pairs], [x for x, _ in pairs]))


def _environment(sample_ids: pd.Series) -> pd.Series:
    return sample_ids.astype(str).str.split("|", n=1).str[-1]


def t1_learning_curve_evidence(rows: list[dict], *, bootstrap_repetitions: int = 1000, seed: int = 20260628) -> dict:
    eligible = [row for row in rows if row.get("route") == "T1" and row.get("engineering_status") in {"FORMAL_ACCEPTED", "FORMAL_ACCEPTED_WITH_WARNING"}]
    curves = []
    comparisons = []
    rng = np.random.default_rng(seed)
    for (fold, run_seed), group in pd.DataFrame(eligible).groupby(["fold", "seed"]):
        loaded = {}
        for row in group.to_dict("records"):
            if row.get("fraction") is None or pd.isna(row.get("fraction")): continue
            pred = pd.read_csv(Path(row["run_path"]) / "predictions.csv", usecols=["sample_id", "y_true", "y_pred"])
            pred["environment"] = _environment(pred.sample_id)
            loaded[(str(row["variant"]), float(row["fraction"]))] = pred
        variant_curves = {}
        for variant in sorted({key[0] for key in loaded}):
            points = {fraction: float(np.mean(np.abs(frame.y_true - frame.y_pred))) for (name, fraction), frame in loaded.items() if name == variant}
            if len(points) >= 2:
                value = _aulc(points); variant_curves[variant] = value
                curves.append({"fold": fold, "seed": int(run_seed), "variant": variant, "aulc": value, "fractions": sorted(points), "mae_by_fraction": points})
        if {"scratch", "pretrain_adapt"}.issubset(variant_curves):
            deltas = []
            reference = next(frame for (name, _), frame in loaded.items() if name == "scratch")
            clusters = reference.environment.unique()
            aggregate = {}
            for key, frame in loaded.items():
                loss = np.abs(frame.y_true - frame.y_pred)
                grouped = pd.DataFrame({"environment": frame.environment, "loss": loss}).groupby("environment").loss.agg(["sum", "count"])
                aggregate[key] = grouped
            for _ in range(bootstrap_repetitions):
                sampled = rng.choice(clusters, size=len(clusters), replace=True)
                boot_curves = {}
                for variant in ("scratch", "pretrain_adapt"):
                    points = {}
                    for (name, fraction), frame in loaded.items():
                        if name != variant: continue
                        grouped = aggregate[(name, fraction)].reindex(sampled)
                        points[fraction] = float(grouped["sum"].sum() / grouped["count"].sum())
                    boot_curves[variant] = _aulc(points)
                deltas.append(boot_curves["scratch"] - boot_curves["pretrain_adapt"])
            comparisons.append({"fold": fold, "seed": int(run_seed), "delta_aulc_scratch_minus_transfer": variant_curves["scratch"] - variant_curves["pretrain_adapt"], "ci_low": float(np.quantile(deltas, .025)), "ci_high": float(np.quantile(deltas, .975)), "bootstrap_repetitions": bootstrap_repetitions})
    return {"bootstrap_cluster": "environment_from_sample_id", "learning_curves": curves, "comparisons": comparisons}


def uncertainty_evidence(rows: list[dict], route: str) -> dict:
    evidence = []
    for row in rows:
        if row.get("route") != route or row.get("engineering_status") not in {"FORMAL_ACCEPTED", "FORMAL_ACCEPTED_WITH_WARNING"}: continue
        frame = pd.read_csv(Path(row["run_path"]) / "predictions.csv")
        item = {key: row.get(key) for key in ("run_path", "fold", "seed", "pattern")}
        if {"lower", "upper"}.issubset(frame):
            item.update(conformal_metrics(frame.y_true, frame.lower, frame.upper, alpha=.1))
        if "uncertainty_score" in frame:
            item["deployable_aurc"] = deployable_aurc(frame.y_true, frame.y_pred, frame.uncertainty_score, score_name="uncertainty_score")["aurc"]
        if "random_rejection_score" in frame:
            item["random_rejection_aurc"] = deployable_aurc(frame.y_true, frame.y_pred, frame.random_rejection_score, score_name="random_rejection_score")["aurc"]
        if "missing_pattern" in frame:
            item["pattern_metrics"] = {pattern: conformal_metrics(group.y_true, group.lower, group.upper, alpha=.1) for pattern, group in frame.groupby("missing_pattern")}
        evidence.append(item)
    return {"route": route, "runs": evidence}
