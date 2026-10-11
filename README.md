# Deployment-Conditioned Crop-Yield Transfer

Reproduction code for *When Transfer Accuracy Is Not Enough: Deployment-Conditioned
Behavioural Diagnostics for Knowledge Transfer*. The accepted paper and evidence
remain unchanged. Full reproduction is conditional on authorised farm inputs,
historical checkpoint access and a lawful APSIM runtime.

## Actual public-data training from a clean environment

Use Python 3.12, Git and an internet connection. The bootstrap creates an
isolated virtual environment with no system site packages, installs
`requirements-verified.lock`, installs this checkout without changing the
pinned dependencies, and runs `pip check`. Commands, installed versions and
return codes are saved in `.venv_setup_logs/`. No old experiment results or
trained checkpoints are reused. Dependency download caches are optional.

The commands below use a POSIX shell. On Windows replace `.venv/bin/python`
with `.venv\\Scripts\\python.exe`. macOS ARM64 is verified; other systems need
their own validation. Graphviz is required for native graph visualisation;
XGBoost on macOS also requires the OpenMP runtime. Install system components
using their official installers or package manager.

An authorised account is required for this private repository. A legally supplied
source ZIP can be extracted instead.

```sh
git clone https://github.com/AoXan/deployment-conditioned-transfer-release-private.git
cd deployment-conditioned-transfer-release-private
python3.12 bootstrap.py --smoke
```

This acquires six authentic CY-Bench Australian-wheat tables, checks ZIP member
CRCs and records hashes, preprocesses real records and trains all five original
components on a reduced split. New checkpoints, preprocessors, predictions,
SHAP/Integrated Gradients, metrics and figures are written to
`outputs/public_smoke/`. This verifies computation; it does not reproduce the
accepted full US-to-Australia performance.

In a fresh checkout, `python3.12 bootstrap.py` installs only the environment.
The separate public commands are:

```sh
.venv/bin/python scripts/acquire_cybench_subset.py --output-root data/external/cybench/raw --pair wheat_AU
.venv/bin/python scripts/run_real_data_smoke.py --raw-root data/external/cybench/raw --output outputs/new_training
.venv/bin/python scripts/render_new_experiment_assets.py --source outputs/new_training --output outputs/new_training/tables
```

Use new acquisition and output directories; existing inputs are preserved.

## Conditional full experiments

[DATA_ACQUISITION_GUIDE.md](DATA_ACQUISITION_GUIDE.md) lists exact files,
official sources, terms and placement. [EXPERIMENT_EXECUTION_GUIDE.md](EXPERIMENT_EXECUTION_GUIDE.md)
covers Stage 7/8, Roseworthy, Waite, G2F, GROUP/SPATIAL training, statistics,
attribution, fresh assets and APSIM.

`run_primary_source_target_smoke.py` trains a new US source model, screens and
qualifies it, then trains an AU target. `run_apsim_real_data_smoke.py` executes
the external runtime and compares new simulator reports with current predictions.
`make reproduce-full` performs authorised preprocessing and formal primary
training with the original plan fingerprint. `make reproduce-cases` executes
the conditional case pipeline; `make smoke-real` runs the public real-data chain.

## Historical publication checks and limits

`make reproduce-results` renders accepted frozen outputs. `make smoke` and
`make reproduce-smoke` are fixture/interface checks. These commands do not train
models. For current predictions use
`render_publication_tables.py --fresh-results NEW_RUN --output NEW_ASSETS`.

Real preprocessing, five training components, SHAP/IG, source-to-target training
and eleven APSIM scenarios have run. Exact Roseworthy cleaned 2007/2008 inputs
and some historical attribution bundles remain unavailable. Waite/G2F full cases
and accepted-scale numerical equality remain unverified. See
[VERIFICATION_REPORT.md](VERIFICATION_REPORT.md),
[PAPER_TO_EXPERIMENT_MATRIX.md](PAPER_TO_EXPERIMENT_MATRIX.md) and
[REPRODUCIBILITY_REPAIR.md](REPRODUCIBILITY_REPAIR.md).
Code is MIT-licensed; data, checkpoint and APSIM terms remain separate.

The current documentation, installation and verification review is in
[RELEASE_REVIEW.md](RELEASE_REVIEW.md).
