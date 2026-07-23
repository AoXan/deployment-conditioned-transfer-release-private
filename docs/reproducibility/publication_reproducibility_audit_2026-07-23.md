# Publication Reproducibility Audit

## Scope

This audit compared the current anonymous AJCAI submission package with the
release repository. The reviewed paper snapshot is:

- `final_ajcai_manuscript_v4_18_6_submission_anonymous.tex`
- `final_ajcai_supplement_v4_18_6_submission_anonymous.tex`
- `final_references_v4_18_5_submission.bib`

The submitted main PDF has SHA256
`b9ee7b59f2ea1601d94bbd2ed3b26a976f7ea1c2760632ab215d1da487bc6966`.
The submitted Supplement PDF has SHA256
`225b7b075226675f50fb4882ee7c7fd6e1e25815341108b9d6cb7a801a0fd8dc`.
The main TeX, Supplement TeX, and bibliography in the desktop Overleaf archive
are byte-identical to the three source files in this repository. All ten PDF
figure assets referenced by those sources are also byte-identical to the
Overleaf archive.

## Audit method

Each result in the main paper and Supplement was assigned to one of three
reproduction levels:

1. full rerun from authorised external input;
2. saved-checkpoint or saved-prediction replay;
3. exact frozen-output replay from redistributable derived source tables.

The detailed mapping is in `paper_code_traceability.md`. A result was not
marked reproducible merely because a similarly named file existed. The audit
checked the runner, configuration, expected input, evaluation code, asset
builder, command, and output.

## Coverage

| Result family | Full runner/config | Frozen source replay | Publication builder | Status |
|---|---|---|---|---|
| CY-Bench US maize to Australian wheat | present | 36 run rows | Table 1 and Figures 1--2 | PASS |
| GROUP/SPATIAL splits and fixed scratch | present | split and replay checks | numeric verification | PASS |
| Scratch, supervised, prediction, representation, combined, missing-aware | present | route/run metrics | table and figure builders | PASS |
| Common-sample SHAP/IG/error/stress analyses | present | sample and summary tables | Figures 3--6 and Supplement assets | PASS |
| Bootstrap, permutation, Holm, trimming, leave-one-run-out | present | inference summaries | Figure 4 and Supplement tables | PASS |
| Roseworthy complete/no-soil | present | 36 path-neutral run rows | Table 2(a) source | PASS |
| Local RF and privileged information | present | 15 gate rows expanded to 36 cells | Table 2(b--c) source | PASS |
| Waite temporal-forward/plot-group | present | six path-neutral run rows | Table 2(d) source | PASS |
| G2F temporal diagnostic | present | authorised input required | documented diagnostic output | PASS |
| APSIM constrained/available/sensitivity comparison | present | locked outputs and integrity manifest | four Supplement tables | PASS |
| Main and Supplement figures | present | bundled source CSVs | PDF/SVG/PNG builder | PASS |
| Main Table 1 and every Table 2 block | present | bundled formal sources | machine-readable table builder | PASS |

## Gaps found and repaired

1. The repository paper snapshot still referred to v4.18.4. It now tracks the
   v4.18.6 anonymous main and Supplement used by the submission package.
2. Table 2 previously had no unified generator for Roseworthy, Waite, or the
   behavioural checks. A path-neutral exporter and complete Table 2 builder
   now cover all five blocks.
3. The case workflow previously returned success without validating the case
   sources. It now checks 36 Roseworthy rows, six Waite rows, 15 RF/PI rows,
   and all four executable case entry points.
4. The traceability matrix pointed to a split module that was not present. It
   now identifies the exact remediation split implementation.
5. Roseworthy and Waite were previously mapped to the wrong runner family.
   They now map to the exact Universal Weather V2 and Stage 7 protocols,
   respectively.
6. The representation extractor contained local absolute paths. It now uses
   repository-relative defaults and explicit command-line paths.
7. The figure write workflow omitted Figure 6 and three Supplement figures.
   It now regenerates every data-driven figure referenced by the current TeX;
   Figure 1 remains the fixed conceptual asset.
8. The public source snapshot lacked the LNCS class, bibliography style,
   Stage 8 schema, and one Supplement figure needed for independent TeX
   compilation. These dependencies are now allowlisted.

## Numeric replay

The numeric verifier checks 40 publication cells or invariants, including the
primary reversal, fixed SPATIAL scratch, negative primary R-squared values,
APSIM Discussion values, RF/PI means, Roseworthy metrics, all six Waite MAEs,
rank agreement, top-group match, IG completeness, and Taylor local fidelity.

## External-input boundary

The release does not redistribute restricted raw agricultural data,
checkpoints, or the APSIM executable. Full fitting and checkpoint-level replay
therefore require the authorised inputs described in `data_access.md`.
This boundary does not remove code: preprocessing, split construction,
training, evaluation, diagnostics, APSIM configuration, and asset generation
remain present. The bundled derived sources support exact replay of the
reported publication tables and figures without rerunning training.

## Clean-environment verification

The allowlist produced a 325-file release tree. A new Python 3.12 environment
was created from that tree and installed with the publication and development
extras. Verification then completed with:

- 23 focused publication and APSIM tests passed;
- 40 numeric cells or invariants passed;
- all public experiment and analysis entry points returned usable `--help`;
- every referenced data-driven figure was written to a new output directory;
- every Table 1/Table 2 machine-readable product was written to a new output
  directory;
- the identity, absolute-path, secret, symlink, and allowlist scan passed.

## Conclusion

The current release provides an executable path from legal external inputs to
the reported experiments, and an input-independent verification path from
bundled derived sources to every displayed table and data-driven figure.
Publication verification must fail explicitly when an authorised input is
missing; it must not silently skip an experiment.
