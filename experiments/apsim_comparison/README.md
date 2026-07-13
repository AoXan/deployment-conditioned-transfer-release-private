# APSIM comparison reproduction

The protocol is locked in `protocol_lock.yaml`. From the repository root:

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
