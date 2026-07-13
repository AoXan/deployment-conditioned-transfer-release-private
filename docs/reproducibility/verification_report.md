# Publication Reproducibility Verification Report

Date: 2026-07-13

## Commands and outcomes

| Check | Command | Outcome |
|---|---|---|
| Public command interface | `pytest -q tests/test_publication_repro_cli.py experiments/apsim_comparison/tests` | PASS, 17 tests |
| Python syntax | `python3 -m py_compile` on the new CLI, table/numeric/data/release scripts and publication analysis entry points | PASS |
| Focused lint | `ruff check` on new/modified publication entry points | PASS |
| Fixture validation | `python -m agritech_repro validate --fixture` | PASS |
| Frozen diagnostic outputs | `python -m agritech_repro reproduce-diagnostics --validate-only` | PASS; 1,240 support rows, 80 feature summaries, 10 SHAP–IG summaries, 180 stress rows, 8 linkage cells |
| Checkpoint/replay manifests | `python -m agritech_repro evaluate-checkpoints --validate-only` | PASS; 45 lineage and 45 replay rows |
| Numeric integrity | `python -m agritech_repro numeric-integrity --validate-only` | PASS, 17 checks, including RF and privileged-information means |
| Figure source parity | `python -m agritech_repro render-paper-assets --validate-only` | PASS; 36 performance rows and all behavioural panel sources |
| Table source validation | `python -m agritech_repro render-paper-tables --validate-only` | PASS; 12 primary rows, 36 RF/PI run-level rows, and 10 central APSIM/data-driven rows |
| APSIM locked outputs | `python -m agritech_repro reproduce-apsim --validate-only` | PASS |
| Full publication gate | `python -m agritech_repro verify-publication --validate-only` | PASS |
| CY-Bench preprocessing interface | synthetic 12-file nested source tree, `python -m agritech_repro preprocess --data-root ... --validate-only` | PASS; all source files and exact resolved config checked without writing views |
| Exact method registry | materialise exact config, then `run_universal_weather_v2_remediation.py --method-audit` in clean release | PASS; six entries, `complete=true`, no missing sources, empty stderr |
| Release scan/build | `python3 scripts/build_publication_release.py --output /tmp/agritech-publication-release` | PASS; 309 allowlisted source files, no symlinks, identity/path and secret scan clean |
| Fresh installation | new venv, `pip install -e '/tmp/agritech-publication-release[publication,dev]'` | PASS |
| Fresh release smoke | clean venv on `PATH`, `make -C /tmp/agritech-publication-release smoke` | PASS, including 17 tests |

The data audit intentionally returns a non-zero status when authorised datasets
are absent. This is the expected behaviour; it lists CY-Bench, G2F, Roseworthy,
and Waite separately and does not silently skip any requested workflow.

## Frozen publication hashes

| Asset | SHA256 |
|---|---|
| Anonymous main TeX | `c1a4516dbb5299961e6911e78153a801a25c6cacf76f953b953c3a607f167dc5` |
| Anonymous main PDF | `159e538583c748acae6d6cffa6a248a0e8af4567c336516e130fdc3d134b1110` |
| Anonymous Supplement TeX | `0c24f49da99dd96772de6f4da43f29ef8b7fde2f7776ee159af240c354b62e51` |
| Anonymous Supplement PDF | `b55f01e0eeee0a2ce3e6d5d0928f5f9a97b88d73768c12c6ab56048aeeb1a1dc` |
| Figure 1 PDF | `6ed108365277f23c7a0ec52c042ff93574f1158f1076011837e0c031e5dec58e` |
| Figure 2 PDF | `f575a3977f768b9db6c5731f7eeeafb9fd0f7c301052de31228de8bfb28c932c` |
| Figure 3 PDF | `c16b057e00e29f8c5b6d12b9055f227b03349c6d6f811496a97b36fe84cadf6f` |
| Figure 4 PDF | `69c7c16ba863dd1e73cc8ea7a8dbaa940274545cdb01fb923ce3bd361faec860` |
| Figure 5 PDF | `4f6000d4a990f52021e37a77b23b6156a60fe83f96630ade7adf3094fc757237` |

No manuscript, Supplement, formal result number, split, route, checkpoint choice,
or statistical conclusion was edited by this reproducibility work. The existing
tracked change to `paper/references.bib` predates and is outside this audit.

## Verification status

**PASS WITH AUTHORISED-INPUT CONDITIONS.** Public code, clean configuration,
fixture checks, frozen-output checks, figure/table generation, APSIM analysis,
and release construction are executable in a clean environment. Full scientific
reruns remain conditional on legally obtained data and the checksum-matched
checkpoint/prediction bundle described in `data_access.md`.
