"""Check publication figures against the supplied derived-result directory."""

from pathlib import Path
import hashlib


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    figures = root / "derived_results" / "figures"
    paths = [figures / f"figure_0{i}.pdf" for i in range(1, 6)]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty figure: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name},{digest}")


if __name__ == "__main__":
    main()
