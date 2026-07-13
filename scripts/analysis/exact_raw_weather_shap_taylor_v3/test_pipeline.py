from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


path = Path(__file__).parent / "run_analysis.py"
name = "raw_shap_taylor_test_module"

spec = importlib.util.spec_from_file_location(name, path)

if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import analysis module")

module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)


def test_raw_transform() -> None:
    raw = np.array(
        [
            [1.0, np.nan],
            [3.0, 8.0],
        ]
    )

    result = module.transform_raw(
        raw,
        imputer_statistics=np.array([2.0, 6.0]),
        scaler_mean=np.array([2.0, 7.0]),
        scaler_scale=np.array([1.0, 1.0]),
    )

    expected = np.array(
        [
            [-1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    assert np.allclose(result, expected)


def test_physical_steps() -> None:
    train = np.array(
        [
            [0.0, 5.0],
            [2.0, 5.0],
            [4.0, 5.0],
        ]
    )

    steps = module.physical_steps(
        train,
        relative_sd=0.01,
        relative_range_fallback=0.001,
        minimum_step=1e-8,
    )

    assert np.isfinite(steps).all()
    assert (steps > 0).all()


def test_selection() -> None:
    metadata = np.array(
        [
            (str(index), float(index))
            for index in range(31)
        ],
        dtype=[
            ("sample_id", "U20"),
            ("abs_error", "f8"),
        ],
    )

    selected = module.select_rows(metadata, 24)

    assert len(selected) == 24
    assert len(set(selected.tolist())) == 24


def test_bootstrap_constant() -> None:
    mean, low, high = module.bootstrap_mean_ci(
        np.ones(5),
        repetitions=100,
        seed=1,
    )

    assert mean == 1.0
    assert low == 1.0
    assert high == 1.0


def main() -> None:
    tests = [
        test_raw_transform,
        test_physical_steps,
        test_selection,
        test_bootstrap_constant,
    ]

    for test in tests:
        test()
        print("PASS", test.__name__)

    print("RAW_SHAP_TAYLOR_UNIT_TESTS_PASS")


if __name__ == "__main__":
    main()
