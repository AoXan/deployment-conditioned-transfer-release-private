# Reproduction commands

1. Install the minimal environment with `python3 -m pip install -r requirements.txt`.
2. Run `make smoke`.
3. Run `make reproduce-frozen`.
4. Inspect the CSV files under `derived_results/` and the five PDFs under
   `derived_results/figures/`.

The commands validate supplied derived results only. They intentionally do not
train, infer, regenerate attribution, or apply perturbations.
