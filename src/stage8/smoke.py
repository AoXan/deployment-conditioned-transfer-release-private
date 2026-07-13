"""Small real fit/predict smoke jobs; outputs are never scientific evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROUTES = ("T1", "T2", "M1", "M2", "R1", "U1", "U2")
SMOKE_STATUS = "SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE"


def _fingerprint(route: str, seed: int) -> str:
    return hashlib.sha256(f"stage8-smoke-v1|{route}|{seed}".encode()).hexdigest()


def run_smoke_suite(root: Path, *, seed: int, routes=ROUTES, resume: bool = False) -> dict:
    unknown = set(routes) - set(ROUTES)
    if unknown:
        raise ValueError(f"unknown routes: {sorted(unknown)}")
    results = {}
    for route in routes:
        out = root / route
        marker = out / "completion_marker.json"
        fingerprint = _fingerprint(route, seed)
        if resume and marker.is_file():
            old = json.loads(marker.read_text())
            if old.get("fingerprint") == fingerprint and old.get("verified") is True:
                results[route] = {"status": SMOKE_STATUS, "resume": "REUSED_VERIFIED"}
                continue
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed + ROUTES.index(route))
        x = rng.normal(size=(36, 6)); y = 2 * x[:, 0] - x[:, 1] + rng.normal(0, .1, 36)
        x[::7, 2] = np.nan
        train, test = np.arange(24), np.arange(24, 36)
        model = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        predictions = pd.DataFrame({"sample_id": [f"{route}|{i}" for i in test], "fold": "smoke_fold_0", "y_true": y[test], "y_pred": pred, "evidence_status": SMOKE_STATUS})
        predictions.to_csv(out / "predictions.csv", index=False)
        metrics = {"route": route, "mae": float(np.mean(np.abs(y[test] - pred))), "scientific_status": "NOT_EVALUATED", "evidence_status": SMOKE_STATUS}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        marker.write_text(json.dumps({"route": route, "fingerprint": fingerprint, "verified": True, "evidence_status": SMOKE_STATUS}, indent=2) + "\n")
        results[route] = {"status": SMOKE_STATUS, "resume": "EXECUTED"}
    return results
