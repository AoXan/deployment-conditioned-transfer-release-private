# Results manifest

The `derived_results/` directory contains the numeric tables used by the paper's
figures and Supplement. Files are grouped by analysis object: performance,
attribution shares and changes, error linkage, stress sensitivity, OOD movement,
explanation quality, route coefficients, and the fixed SPATIAL scratch reference.

`make reproduce-frozen` validates required columns, route names, contract names,
and figure presence. It does not alter any result table.
