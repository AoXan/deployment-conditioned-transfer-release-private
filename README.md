# Deployment-Conditioned Transfer Reproduction Materials

This private release accompanies the anonymous AJCAI 2026 submission. It contains
analysis interfaces, configuration templates, derived result tables, and
publication figures for inspecting the reported study without redistributing
restricted raw data, checkpoints, or prediction files.

## Reproduction scope

The supplied commands validate the frozen derived tables and publication figures.
They do not train models, run forward inference, or recreate attribution and
perturbation calculations. Those operations require the original datasets and
are intentionally outside this anonymous release.

## Quick start

```bash
python3 -m pip install -r requirements.txt
make smoke
make reproduce-frozen
```

The Makefile sets `PYTHONPATH=.` for the table checker.

All paths are relative to this directory. Restricted data are described in
`DATA_SOURCES.md` and `DATA_LICENSES.md`.

## Study map

The primary comparison is US maize to Australian wheat under GROUP and SPATIAL
deployment contracts. Derived tables cover performance, common-sample attribution,
error linkage, finite stress response, OOD movement, and explanation quality.
