# Deployment-Conditioned Crop-Yield Transfer

This repository contains the code and publication assets for the AJCAI study of
US-maize-to-Australian-wheat transfer under GROUP and SPATIAL deployment
contracts. It covers the primary distillation routes, matched behavioural
diagnostics, agricultural case studies, and the supplementary APSIM comparison.

The publication interface is deliberately separate from historical research
scripts. The camera-ready verification entry point needs no restricted data
and regenerates the paper figures, machine-readable tables, and reported
summary statistics from the frozen publication outputs:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[publication,dev]'
make reproduce-results
```

`make reproduce-smoke` is the shorter interface and code-path check.
`make reproduce-full` deliberately stops before expensive execution and points
to the authorised-data instructions. A fully unattended public-data rerun is
not claimed because the exact cleaned Roseworthy point files used by the
accepted study are absent from the cited Figshare v1/v2 file inventories. The
exact Waite workbook is publicly obtainable from CSIRO under CC BY 4.0.

Use `python -m agritech_repro <command> --help` for every workflow. The main
entry points are:

```bash
python -m agritech_repro validate --fixture
python -m agritech_repro data-audit
python -m agritech_repro preprocess --data-root /path/to/authorised/data --validate-only
python -m agritech_repro reproduce-primary --validate-only
python -m agritech_repro evaluate-checkpoints --validate-only
python -m agritech_repro reproduce-diagnostics --validate-only
python -m agritech_repro reproduce-cases --validate-only
python -m agritech_repro reproduce-apsim --validate-only
python -m agritech_repro render-paper-assets --validate-only
python -m agritech_repro render-paper-tables --validate-only
python -m agritech_repro numeric-integrity --validate-only
python -m agritech_repro verify-publication --validate-only
```

Full reruns require the authorised datasets and, for exact replay, the frozen
checkpoint/prediction bundle. Paths are supplied through `--data-root`,
`--checkpoint-root`, `AGRITECH_DATA_ROOT`, `AGRITECH_CHECKPOINT_ROOT`,
`AGRITECH_PREDICTION_ROOT`, and `AGRITECH_CYBENCH_ROOT`; no personal path is
required. Provenance-unresolved raw data are never downloaded automatically;
the openly licensed Waite workbook has a checksum-locked acquisition command.

See:

- `docs/reproducibility/REPRODUCIBILITY.md` for end-to-end commands.
- `docs/reproducibility/paper_code_traceability.md` for the paper-to-code matrix.
- `docs/reproducibility/publication_reproducibility_audit_2026-07-23.md` for the
  submission-package audit and repaired gaps.
- `docs/reproducibility/data_access.md` for licences, schemas, and expected paths.
- `docs/reproducibility/release_scope.md` for the public allowlist and exclusions.

The code is released under the MIT licence. Dataset and APSIM terms remain those
of their respective providers.
