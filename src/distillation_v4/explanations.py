from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Mapping, Sequence

import numpy as np


def grouped_interventional_shap(
    predict: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    background: np.ndarray,
    groups: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    """Exact grouped interventional SHAP for a small, predeclared block set.

    Missing coalitions are evaluated by replacing absent blocks with the fixed
    background mean.  The routine is intended for final, frozen models only;
    it does not perform feature or route selection.
    """
    x = np.asarray(values, dtype=float)
    reference = np.asarray(background, dtype=float).mean(axis=0)
    if x.ndim != 2 or reference.shape != (x.shape[1],):
        raise ValueError("SHAP_INPUT_SHAPE_MISMATCH")
    names = tuple(groups)
    covered = [index for name in names for index in groups[name]]
    if len(set(covered)) != len(covered) or any(index < 0 or index >= x.shape[1] for index in covered):
        raise ValueError("SHAP_GROUP_CONTRACT_INVALID")

    cache: dict[frozenset[str], np.ndarray] = {}

    def coalition(active: frozenset[str]) -> np.ndarray:
        if active not in cache:
            materialized = np.broadcast_to(reference, x.shape).copy()
            for name in active:
                indices = list(groups[name])
                materialized[:, indices] = x[:, indices]
            prediction = np.asarray(predict(materialized), dtype=float).reshape(-1)
            if prediction.shape[0] != x.shape[0] or not np.isfinite(prediction).all():
                raise ValueError("SHAP_PREDICTION_INVALID")
            cache[active] = prediction
        return cache[active]

    count = len(names)
    attributions: dict[str, np.ndarray] = {}
    for name in names:
        others = [candidate for candidate in names if candidate != name]
        contribution = np.zeros(x.shape[0], dtype=float)
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(count - size - 1) / factorial(count)
            for subset in combinations(others, size):
                active = frozenset(subset)
                contribution += weight * (coalition(active | {name}) - coalition(active))
        attributions[name] = contribution
    attributions["base_value"] = coalition(frozenset())
    attributions["prediction"] = coalition(frozenset(names))
    return attributions
