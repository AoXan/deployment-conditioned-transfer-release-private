#!/usr/bin/env python3
"""Create path-neutral checkpoint lineage and replay manifests for release."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data_manifest/checkpoints"))
    args = parser.parse_args()
    lineage = pd.read_csv(args.lineage)
    replay = pd.read_csv(args.replay)
    lineage["checkpoint_path"] = lineage.apply(
        lambda row: "" if pd.isna(row["checkpoint_sha256"]) else f"${{AGRITECH_CHECKPOINT_ROOT}}/{row['candidate_id']}.pt",
        axis=1,
    )
    lineage["official_predictions_path"] = lineage["candidate_id"].map(
        lambda value: f"${{AGRITECH_PREDICTION_ROOT}}/{value}/predictions.csv"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lineage.to_csv(args.output_dir / "checkpoint_lineage.csv", index=False)
    replay.to_csv(args.output_dir / "prediction_replay_validation.csv", index=False)
    print(f"Wrote {len(lineage)} lineage and {len(replay)} replay rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
