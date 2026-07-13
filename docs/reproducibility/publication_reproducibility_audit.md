# Publication Reproducibility Audit

## Scope and verdict

This audit covers the final anonymous manuscript, its Supplement, the Stage 8
primary and agricultural-case analyses, all publication figures/tables, and the
APSIM comparison. The repository now has a single public CLI, clean configuration
templates, an allowlist release manifest, data schemas and acquisition rules,
numeric checks, fixture tests, and paper-to-code traceability.

**Verdict: executable with tiered access.** The public fixture, frozen source-table
verification, figure/table rebuilding, numeric checks, and APSIM result analysis
run without restricted raw data. Full scientific reruns require authorised
CY-Bench/G2F/Roseworthy/Waite inputs and, for exact checkpoint replay, the frozen
prediction/checkpoint bundle. These are explicit data-access conditions rather
than missing code paths.

## Repairs completed

1. Added `agritech_repro` commands for repository/data validation, preprocessing,
   primary preflight/rerun, checkpoint replay, diagnostics, cases, APSIM, figures,
   tables, numeric integrity, and full verification.
2. Imported the formal `universal_weather_v2` model, loss, token, adapter, training,
   and artifact modules that previously lived only in another worktree.
3. Replaced personal-path defaults in the publication analysis paths with the
   repository root or documented environment variables.
4. Added clean Stage 8 configuration materialisation and made missing datasets,
   checkpoints, APSIM runtime, and generated configs fail with non-zero status.
5. Added a pinned APSIM bootstrap command and removed the Homebrew-specific runtime
   assumption from the simulation wrapper.
6. Added an exact numeric checker for the primary reversal, fixed SPATIAL scratch,
   negative R2 retention, and the four APSIM Discussion values.
7. Added machine-readable Table 1/APSIM generation, figure source validation, a
   release allowlist, dataset manifest, checksum interface, and synthetic fixture.
8. Refactored the G2F diagnostic entry to accept CLI paths and validation mode.
9. Replaced the legacy top-level README with publication reproduction commands.
10. Removed a hidden Figure-builder dependency on the v4.16 source directory;
    v4.18 now validates and rebuilds from its own source tables.
11. Added a path-neutral 45-row checkpoint lineage and 45-row replay manifest,
    preserving candidate IDs and hashes without publishing personal paths.
12. Added an allowlist release builder and verified a 309-file clean release in
    a newly created virtual environment.
13. Restored the exact Universal Weather V2 remediation runner, schema, planner,
    execution modules, and US-maize/Australian-wheat campaign configuration; the
    full campaign requires its preflight fingerprint before execution.
14. Added an executable CY-Bench preprocessing entry that validates the 12 raw
    maize-US and wheat-AU source tables and builds the two harmonised views used
    by the exact campaign without relying on notebooks or personal paths.
15. Restored the formal Stage 8 route adapter required by the remediation
    method registry and replaced its machine-specific data locations with
    documented environment variables and release-relative defaults.
16. Connected the CY-Bench downloader to preprocessing through a checksum-
    verified, safe extraction layout and support for the archive's own nested
    source directory.
17. Extended release validation to reject common private-key and cloud-token
    patterns as well as sensitive filenames, personal paths, and symlinks.
18. Added the formal RF/privileged-information claim-gate summary to the release
    and rebuilt 36 run-level Table 2 source rows through the public table command.
19. Made the CLI locate a publication checkout from the current working directory,
    so verification works after both editable and ordinary wheel installation.

## Open-source and data boundaries

The MIT licence applies to repository code only. Dataset and APSIM rights remain
with their providers. The release must not contain the Roseworthy cleaned file,
Waite source workbook, unreviewed G2F files, raw/derived research data,
checkpoints, private prediction bundles, or the cloned APSIM runtime. Public
code remains available for every such experiment; authorised users supply data
through documented directories or environment variables.

## Residual risks

- Exact from-scratch numerical identity is conditional on the published software
  environment, authorised data version, and checkpoint/prediction checksums.
- Some historical Stage 7 scripts remain outside the release allowlist; only the
  entry points needed by the paper are in scope.
- Table 2's compact reader-facing layout is authored in LaTeX, while its scientific
  cells are checked through formal result/evidence sources rather than generated
  by one monolithic table script.
- The historical method registry reports plotting provenance separately from the
  publication figure pipeline. Figure provenance is therefore verified through
  the v4.18 source and panel-integrity manifests rather than that legacy field.
- The Roseworthy cleaned-file/public-record correspondence remains unresolved;
  code is releasable, but the cleaned input is not.

No paper number, split, route, checkpoint choice, figure datum, statistical
conclusion, or claim was changed during this audit.
