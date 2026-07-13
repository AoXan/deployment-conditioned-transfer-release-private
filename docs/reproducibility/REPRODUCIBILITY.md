# Publication Reproducibility Guide

## 1. Install and smoke-test

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[publication,dev]'
make smoke
```

The smoke path uses only the synthetic fixture and frozen publication source
tables. It does not train, infer, attribute, perturb, or run APSIM.

## 2. Audit external inputs

```bash
python -m agritech_repro data-audit --data-root /path/to/authorised/data
python -m agritech_repro preprocess --data-root /path/to/authorised/data --validate-only
```

Missing data return exit code 2 and name the blocked workflow. No restricted
source is downloaded automatically.
The preprocessing command validates the 12 CY-Bench maize-US and wheat-AU
source tables, then uses the formal adapter to create
`PRIMARY_CYBENCH_MAIZE_US.csv.gz` and `CY-Bench_wheat_AU.csv.gz`. Omit
`--validate-only` only when these harmonised views should be written.

## 3. Primary Stage 8 study

Validate the resolved config and formal matrix without fitting:

```bash
python -m agritech_repro reproduce-primary \
  --data-root /path/to/authorised/data --validate-only
```

This invokes the exact Universal Weather V2 remediation campaign used to create
the formal US-maize-to-Australian-wheat route family. The returned dry-run JSON
contains the immutable plan fingerprint.

Run the frozen protocol only when a full rerun is intended:

```bash
python -m agritech_repro reproduce-primary \
  --data-root /path/to/authorised/data \
  --checkpoint-root /path/to/checkpoints \
  --confirm-full-campaign PLAN_FINGERPRINT
```

`scripts/run_universal_weather_v2_remediation.py` writes manifests,
fingerprints, fold assignments,
checkpoints, predictions, metrics, failures, and progress under
`outputs/publication_repro/primary/`.

## 4. Saved-checkpoint and behavioural replay

```bash
python -m agritech_repro evaluate-checkpoints --validate-only
FORMAL_CODE_ROOT="$PWD" \
AGRITECH_FORMAL_REPLAY_ROOT=/path/to/prediction_bundle/jobs/targets \
AGRITECH_PRIMARY_VIEW=/path/to/CY-Bench_wheat_AU.csv.gz \
python3 scripts/analysis/common_sample_attribution_v1/run_analysis.py \
  --worktree "$PWD" --formal-code "$PWD" \
  --output outputs/publication_repro/diagnostics \
  --shap-permutations 64 --background-rows 32 --ig-steps 64 \
  --perm-reps 4999 --random-seed 20260709
```

The common-sample pipeline constructs the sample/background registries,
Permutation SHAP and IG values, matched error changes, redistribution,
finite-stress responses, bootstrap/permutation summaries, and quality checks.
The validation scripts under
`scripts/analysis/common_sample_attribution_validation_v1/` implement clustered
bootstrap, trimming, leave-one-run-out, and Holm-family reporting.

## 5. Agricultural cases

Roseworthy and Waite use the same public Stage 8 runner with the corresponding
dataset filter after materialising `configs/publication/stage8_v4_australian.yaml`.
The G2F temporal diagnostic has a path-clean entry:

```bash
python3 src/phase3/run_g2f_route1_pilot_fixed.py \
  --data /path/to/g2f_native_hybrid_env.csv.gz \
  --split /path/to/g2f_native_hybrid_env__temporal_forward__test_2022.csv \
  --output outputs/publication_repro/cases/g2f --validate-only
```

Privileged-information training is exposed by:

```bash
python3 scripts/run_privileged_ablation.py --help
```

The local RF, PI, Roseworthy, and Waite publication rows are also checked when
the final evidence builder is run in validation mode.

## 6. APSIM comparison

```bash
python3 experiments/apsim_comparison/bootstrap_apsim.py
export AGRITECH_CYBENCH_ROOT=/path/to/cybench/wheat/AU
export AGRITECH_PREDICTION_ROOT=/path/to/prediction_bundle
python3 experiments/apsim_comparison/build_experiment.py
experiments/apsim_comparison/run_apsim.sh
python3 experiments/apsim_comparison/analyse_results.py
python3 experiments/apsim_comparison/render_supplement_tables.py
python3 -m pytest -q experiments/apsim_comparison/tests
```

`protocol_lock.yaml` fixes checkpoint selection, central APSIM conditions,
sensitivity settings, common unit, bootstrap seed, and leakage controls. The
runtime source is pinned to commit
`2c3639e456746e16ab9ae95a4c492dbfbe6e567f`.

## 7. Paper assets and integrity

```bash
python -m agritech_repro render-paper-assets --validate-only
python -m agritech_repro render-paper-tables --validate-only
python -m agritech_repro numeric-integrity --validate-only
python -m agritech_repro verify-publication --validate-only
```

Figure generation validates panel source parity. Table generation produces
machine-readable primary and APSIM summaries. Numeric integrity checks the
displayed primary reversal, fixed SPATIAL scratch, retained negative R2 values,
and the four APSIM Discussion values.
