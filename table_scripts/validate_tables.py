"""Validate that the released CSV files are readable and non-empty."""

from pathlib import Path
import csv


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "derived_results"
    tables = sorted(root.glob("*.csv"))
    if not tables:
        raise SystemExit("No derived tables found.")
    for table in tables:
        with table.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2 or not all(rows[0]):
            raise SystemExit(f"Unreadable or empty table: {table.name}")
    print(f"Validated {len(tables)} derived tables.")


if __name__ == "__main__":
    main()
