# Experiment execution guide

All commands require the isolated environment. Use `DATA` and `RAW` as external authorised input roots and `OUT` as a new experiment root. Public raw acquisition is separate from code delivery. The accepted paper/figures remain frozen comparison material.

## Public real-data chain actually tested

```sh
python scripts/acquire_cybench_subset.py --output-root DATA/cybench/raw
python scripts/run_real_data_smoke.py --raw-root DATA/cybench/raw --output OUT/regional_smoke
python scripts/render_publication_tables.py --fresh-results OUT/regional_smoke --output OUT/fresh_assets
python figures/final_publication_v4_18/build_final_figures.py --fresh-results OUT/regional_smoke --output-dir OUT/fresh_figures --write
python scripts/prepare_source_smoke_view.py --raw-root DATA/cybench/raw --output OUT/us_source_view --regions 4
```

The regional smoke calls original preprocessing, split construction and five real components (supervised, prediction KD, representation KD, combined KD, HistGradientBoosting reference). It saves new `.pt`, preprocessor, predictions, metrics, SHAP/IG arrays and replay/hash receipts. It is an engineering regional protocol, not the accepted GROUP/SPATIAL comparison. The new renderer recomputes MAE/RMSE/R2 from fresh predictions; it never reads frozen table numbers.

For tested authentic US->AU GROUP transfer, put the source view at `DATA/cybench/PRIMARY_CYBENCH_MAIZE_US.csv.gz` and the full newly processed AU view at `DATA/cybench/CY-Bench_wheat_AU.csv.gz`. Materialise `configs/publication/verified_source_target_smoke.yaml` with `AGRITECH_DATA_ROOT=DATA` using `scripts/materialize_publication_config.py --input ... --output OUT/source_target_config.yaml`, then `python scripts/run_primary_source_target_smoke.py --config OUT/source_target_config.yaml --output OUT/source_target`. It invokes original source training/screening/qualification and true GROUP target training. Reduced source regions/epochs and seed 101 verify execution; they do not establish full-scale numerical equality.

## Full primary and supplementary campaign

`PYTHONPATH=src python -m agritech_repro preprocess --data-root DATA --output-root OUT` materialises authentic views/config. `PYTHONPATH=src python -m agritech_repro reproduce-primary --validate-only --data-root DATA --output-root OUT` produces the plan fingerprint. Use that fingerprint with `make reproduce-full DATA_ROOT=DATA OUTPUT_ROOT=OUT PLAN_FINGERPRINT=VALUE`. The formal dispatcher schedules original source training/screening/teacher qualification before target jobs. Keep complete/group/spatial, original seeds, routes, weather/soil missingness and transfer definitions. Exact full campaign cannot run without its declared G2F/Waite/Roseworthy views and permitted compute budget. Use the preflight fingerprint to bind execution to the inspected protocol.

## G2F, Roseworthy, Waite, Stage 7/8

```sh
python scripts/preprocess_authorised_cases.py --dataset g2f --raw-root RAW --output OUT/g2f_nursery
python scripts/preprocess_authorised_cases.py --dataset roseworthy --raw-root RAW --output OUT/roseworthy_nursery
AGRITECH_RAW_ROOT=RAW AGRITECH_WAITE_OUTPUT=OUT/waite_nursery python scripts/build_stage7_waite_slga_unified_nursery.py
python scripts/run_stage7_unified_nursery_experiments.py --config RESOLVED_STAGE7_CONFIG --stage all --run-formal
make reproduce-cases DATASET=roseworthy DATA_ROOT=DATA OUTPUT_ROOT=OUT
make reproduce-cases DATASET=waite DATA_ROOT=DATA OUTPUT_ROOT=OUT
make reproduce-cases DATASET=g2f DATA_ROOT=DATA OUTPUT_ROOT=OUT
```

These invoke actual ETL/training, not validation-only case exports. G2F expects the original model-ready native view at `DATA/g2f/g2f_native_hybrid_env.csv.gz`; materialise/adapt config paths to fresh nursery outputs. Roseworthy/Waite select original Stage8 Australian jobs. Stage7 code includes references to original prerequisite expert prediction inputs for some variants; regenerate them through their original upstream experiment or provide an authorised raw/prediction bundle as that experiment defines. They were not newly verified here. Do not treat absent prerequisite predictions as successful execution. RF/PI use original Stage8 formal routes and prediction/common-sample diagnostics; exact full inherited baselines require the original input/fold lineage.

## SHAP, IG, perturbation and statistical tests

For fresh smoke diagnostics use `run_real_data_smoke.py` (actual fresh-checkpoint SHAP/IG). For accepted formal architecture: `scripts/analysis/representation_support_v1/extract_embeddings.py --repository-root . --output OUT/embeddings --lineage AUTHORISED_LINEAGE`; `scripts/analysis/exact_raw_weather_shap_taylor_v3/run_analysis.py --worktree BUNDLE_ROOT --formal-code . --extract-script scripts/analysis/representation_support_v1/extract_embeddings.py --manifest OUT/embeddings/embedding_manifest.csv --lineage AUTHORISED_LINEAGE --output OUT/exact_attribution` rebuilds formal preprocessing/SHAP/Taylor inputs. Lineage columns `checkpoint_path` and `official_predictions_path` must point to the external authorised files; they are not copied into the repository. Consult `--help` for mandatory full-replay controls.

Set the three `AGRITECH_*` lineage/manifest/preprocessor variables, then `python scripts/analysis/common_sample_attribution_v1/run_analysis.py --worktree BUNDLE_ROOT --formal-code . --output OUT/attribution --shap-permutations 64 --background-rows 32 --ig-steps 64 --perm-reps 4999 --random-seed 20260709`. Follow original validation phase scripts (`run_phases_1_3`, `run_phases_4_5`, `run_phase_6`, `run_phase_7`) to common-sample gates/statistical checks and `plot_publication_figures.py`/`generate_latex_tables.py` to diagnostic assets. Full 30-cell lineage compatibility and exact accepted attribution values remain conditional on authorised bundles; the new regional smoke receipt is not a substitute for that lineage.

## APSIM

```sh
python scripts/acquire_cybench_subset.py --output-root DATA/cybench/raw --pair wheat_AU --apsim
python scripts/run_apsim_real_data_smoke.py --raw-root DATA/cybench/raw --prediction OUT/regional_smoke/models/supervised/predictions.csv --runtime /licensed/Models.dll --output OUT/apsim_smoke
```

This calls the original scenario builder and actual runtime, writes new weather/configs/databases, and exports SQLite reports. Tested one real region-year, all 11 scenarios; it does not assert complete accepted APSIM comparison parity. The full original scripts in `experiments/apsim_comparison/` require their exact common sample cohort, raw data and runtime; never use locked SQLite/CSV as new simulator output.

## Commands that inspect frozen evidence

`make reproduce-results`, validation-only CLI commands and default `render_publication_tables.py` / `build_final_figures.py` use accepted frozen evidence. They check/re-render historical evidence and do not execute new experiments. Fresh commands above are explicitly separate. `make smoke` includes fixtures and historical integrity checks; only `make smoke-real` actually trains. The distinction is intentional and must be kept in any external report.
