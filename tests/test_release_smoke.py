from pathlib import Path

import pytest

from src.reproduction.metrics import regression_metrics
from src.reproduction.protocol import ROUTE_COEFFICIENTS, stress_ratio
from src.reproduction.splits import holdout_split


def test_regression_metrics_are_deterministic():
    result = regression_metrics([1.0, 2.0, 3.0], [1.0, 1.0, 4.0])
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["rmse"] == pytest.approx((2 / 3) ** 0.5)


def test_split_is_reproducible():
    rows = list(range(10))
    assert holdout_split(rows, 0.2, 7) == holdout_split(rows, 0.2, 7)


def test_route_coefficients_and_stress():
    assert ROUTE_COEFFICIENTS["prediction_kd"] == (0.5, 0.0, 0.0)
    assert stress_ratio(2.0, 4.0) == pytest.approx(0.5)


def test_derived_figures_are_present():
    root = Path(__file__).resolve().parents[1]
    assert all((root / "derived_results" / "figures" / f"figure_0{i}.pdf").exists() for i in range(1, 6))
