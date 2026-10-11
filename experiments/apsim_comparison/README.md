# APSIM comparison reproduction

Install the pinned environment using the repository README. Obtain a lawful
APSIM runtime separately; runtime binaries are not distributed. Fresh execution
against current model predictions can be performed from the repository root:

```sh
.venv/bin/python scripts/acquire_cybench_subset.py --output-root data/external/apsim_raw --pair wheat_AU --apsim
.venv/bin/python scripts/run_real_data_smoke.py --raw-root data/external/apsim_raw --output outputs/new_training
.venv/bin/python scripts/run_apsim_real_data_smoke.py --raw-root data/external/apsim_raw \
  --prediction outputs/new_training/models/supervised/predictions.csv \
  --runtime /absolute/path/to/authorised/Models.dll --output outputs/new_apsim
```

This runs the simulator and generates new reports; it does not validate an old
SQLite output as a substitute for simulation. The representative route has been
executed for eleven scenarios. Full cohort and accepted numerical equality are
not claimed. Runtime and dataset access requirements are in
`DATA_ACQUISITION_GUIDE.md` and `EXPERIMENT_EXECUTION_GUIDE.md` at the repository root.

The historical accepted protocol is locked in `protocol_lock.yaml`. Its separate
checkpoint/prediction replay route is:

```bash
python3 experiments/apsim_comparison/bootstrap_apsim.py
python3 experiments/apsim_comparison/build_experiment.py
experiments/apsim_comparison/run_apsim.sh
python3 experiments/apsim_comparison/analyse_results.py
python3 experiments/apsim_comparison/render_supplement_tables.py
python3 -m pytest experiments/apsim_comparison/tests -q
```

The pipeline uses the pinned APSIM Next Generation source revision recorded in
`outputs/apsim_comparison/apsim_runtime_manifest.json`. It does not train a data-driven
model. Frozen predictions are selected using existing validation MSE, joined to the
immutable sample manifest, and compared on region-year units.

`build_experiment.py` requires an authorised CY-Bench checkout and the frozen
prediction bundle described in `docs/reproducibility/data_access.md`. Set
`AGRITECH_CYBENCH_ROOT` and `AGRITECH_PREDICTION_ROOT` rather than editing paths.
The APSIM source checkout under `experiments/apsim_comparison/runtime/` is ignored
by the publication release and is recreated by the pinned bootstrap command.
