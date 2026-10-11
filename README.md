# When Transfer Accuracy Is Not Enough

Code for *When Transfer Accuracy Is Not Enough: Deployment-Conditioned
Behavioural Diagnostics for Knowledge Transfer*. Accepted at AJCAI; the public
paper link will be added when available. Manuscript files are not distributed here.

## Real public-data execution from a clean environment

Install Python 3.12 and Git. With an internet connection:

```sh
git clone https://github.com/AoXan/deployment-conditioned-transfer-release-private.git
cd deployment-conditioned-transfer-release-private
python3.12 bootstrap.py --smoke
```

The bootstrap creates `.venv`, installs `requirements-verified.lock`, checks
dependencies and downloads six authentic CY-Bench wheat-AU source tables.
Original preprocessing and split construction feed five actual model components:
supervised learning, prediction KD, representation KD, combined KD and a
HistGradientBoosting reference. New checkpoints, preprocessors, predictions,
metrics, SHAP/IG arrays and tables/figures are written to `outputs/public_smoke/`.
This engineering regional test verifies the calculation chain; it does not
replace the accepted GROUP/SPATIAL experiment protocol.

## Formal primary, case and diagnostic experiments

`python3.12 bootstrap.py` installs the environment only. Follow
[DATA_ACQUISITION_GUIDE.md](DATA_ACQUISITION_GUIDE.md) for exact filenames, access
terms and external input placement; then use
[EXPERIMENT_EXECUTION_GUIDE.md](EXPERIMENT_EXECUTION_GUIDE.md) for source/target
training, Stage 7/8, Roseworthy, Waite, G2F, common-sample SHAP/IG, statistical
tests and APSIM commands.

```sh
PYTHONPATH=src .venv/bin/python -m agritech_repro preprocess --data-root DATA --output-root OUT
PYTHONPATH=src .venv/bin/python -m agritech_repro reproduce-primary --validate-only --data-root DATA --output-root OUT
make reproduce-full PYTHON=.venv/bin/python DATA_ROOT=DATA OUTPUT_ROOT=OUT PLAN_FINGERPRINT=VALUE
```

Use the fingerprint returned by preflight. The full dispatcher trains/screens
source models and qualifies teachers before target training. Outputs use a new
external experiment directory; checkpoints and restricted inputs are not bundled.
Fresh-table commands use `--fresh-results OUT`. Commands explicitly labelled
validation-only or `make reproduce-results` inspect frozen comparison evidence.

## Verification scope and access conditions

The public preprocessing/training/attribution/export chain is **PIPELINE_VERIFIED**.
Reduced native US-to-AU GROUP transfer and eleven real APSIM scenarios have also
executed. Full accepted numerical identity is not claimed. The overall project
is **CONDITIONALLY_REPRODUCIBLE**: exact Roseworthy cleaned files, compatible
historical attribution lineage and some case inputs require authorised access;
APSIM requires a lawful external runtime. See
[PAPER_TO_EXPERIMENT_MATRIX.md](PAPER_TO_EXPERIMENT_MATRIX.md) for every main and
supplementary path. Dataset and software licences remain those of their providers.
