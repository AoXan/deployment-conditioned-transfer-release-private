"""Validate the supplied derived tables and publication figures.

This script reads no raw data and performs no model computation. It checks the
public release structure so that a fresh checkout can be inspected consistently.
"""

from pathlib import Path
import csv
import hashlib

from src.reproduction.protocol import CONTRACTS, FEATURE_GROUPS, ROUTE_COEFFICIENTS, RUNS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "derived_results"


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        "source_performance_landscape_seed.csv",
        "source_route_loss_coefficients.csv",
        "source_scratch_baseline_status.csv",
        "source_support_delta_summary.csv",
        "source_linkage_summary.csv",
        "source_ood_quality_9d.csv",
    ]
    missing = [name for name in required if not (RESULTS / name).exists()]
    if missing:
        raise SystemExit(f"Missing derived tables: {', '.join(missing)}")
    performance_header = read_header(RESULTS / required[0])
    if not {"contract", "method"}.issubset({item.lower() for item in performance_header}):
        raise SystemExit("Performance table is missing contract or method columns.")
    coeff_header = read_header(RESULTS / "source_route_loss_coefficients.csv")
    required_coeff = {"route", "lambda_prediction", "lambda_representation", "lambda_consistency"}
    if not required_coeff.issubset({item.lower() for item in coeff_header}):
        raise SystemExit("Route coefficient table has an unexpected schema.")
    figure_hashes = {}
    for number in range(1, 6):
        figure = RESULTS / "figures" / f"figure_0{number}.pdf"
        if not figure.exists():
            raise SystemExit(f"Missing publication figure: {figure}")
        figure_hashes[figure.name] = sha256(figure)
    print("Contracts:", ", ".join(CONTRACTS))
    print("Runs:", ", ".join(RUNS))
    print("Feature groups:", ", ".join(FEATURE_GROUPS))
    print("Routes:", ", ".join(ROUTE_COEFFICIENTS))
    print("Validated tables:", len(required))
    print("Figure hashes:", figure_hashes)


if __name__ == "__main__":
    main()
