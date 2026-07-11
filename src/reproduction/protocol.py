"""Public protocol constants used by the frozen-result checks."""

ROUTE_COEFFICIENTS = {
    "supervised": (0.0, 0.0, 0.0),
    "prediction_kd": (0.5, 0.0, 0.0),
    "representation_kd": (0.0, 0.25, 0.0),
    "combined_kd": (0.5, 0.25, 0.0),
    "missing_aware": (0.25, 0.0, 0.25),
}

CONTRACTS = ("GROUP", "SPATIAL")
RUNS = ("S1", "S2", "S3")
FEATURE_GROUPS = ("observation_support", "temperature", "water_balance", "radiation")


def stress_ratio(mean_abs_prediction_change: float,
                 mean_perturbation_distance: float,
                 epsilon: float = 1e-12) -> float:
    """Return prediction change per standardised perturbation distance."""

    return mean_abs_prediction_change / (mean_perturbation_distance + epsilon)
