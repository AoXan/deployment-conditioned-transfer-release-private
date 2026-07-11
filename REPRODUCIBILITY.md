# Reproducibility

The release is organised around deterministic, row-level derived tables. The
primary study uses three prespecified runs, contract-specific target splits, and
common target samples for route comparisons. The statistical construction is
described in the manuscript and Supplement.

The release validates table schemas, recomputes basic regression summaries from
supplied numeric columns, and verifies the five publication figure files.
Recreating model fitting, attribution, stress, or OOD calculations requires
restricted source data and is not performed by the supplied smoke commands.

No raw observations, checkpoints, prediction files, credentials, or private
repository identifiers are distributed here.
