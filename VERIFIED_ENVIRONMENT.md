# Verified environment

Use Python 3.12 in a new virtual environment. `requirements-verified.lock` records the actual isolated environment used for the evidence runs, excluding editable installation paths. Install with `python -m pip install -r requirements-verified.lock`. Scientific source entrypoints set their own import paths; tests in the process projects need `PYTHONPATH=src`. External native sources and raw inputs are deliberately outside the code archive. Use new output directories; never point an execution output at frozen `results/`, accepted manuscript assets, or a raw input directory. CPU and solver timings are environment dependent.

The lock was obtained on macOS arm64. Linux portability of the complete environment was not executed in this audit. Public source or recorded experiment descriptions are the provenance authority when independent metadata was not available, as instructed by the owner on 2026-10-11.
