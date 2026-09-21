PYTHON ?= python3
REPRO = PYTHONPATH=src $(PYTHON) -m agritech_repro

.PHONY: help setup validate data-audit preprocess primary-preflight checkpoints diagnostics cases apsim-validate figures tables numeric-integrity verify smoke test reproduce-results reproduce-smoke reproduce-full

help:
	@printf '%s\n' 'Publication reproducibility targets:' \
	  '  setup              install the project and publication extras' \
	  '  validate           validate repository structure with the public fixture' \
	  '  data-audit         report required external inputs (never downloads restricted data)' \
	  '  preprocess         validate CY-Bench inputs, build views, and materialise config' \
	  '  primary-preflight  validate the formal primary experiment plan without training' \
	  '  checkpoints        validate checkpoint lineage and prediction replay records' \
	  '  diagnostics        validate frozen behavioural-analysis inputs' \
	  '  cases              validate G2F, Roseworthy, Waite, RF, and PI inputs' \
	  '  apsim-validate      validate locked APSIM outputs and protocol files' \
	  '  figures            validate all publication figure sources' \
	  '  tables             validate publication table sources' \
	  '  numeric-integrity  verify frozen headline and APSIM values' \
	  '  verify             run publication-level validation' \
	  '  smoke              run fixture, numeric, figure, table, and focused tests' \
	  '  reproduce-results  regenerate paper figures, tables, and reported statistics' \
	  '  reproduce-smoke    fast alias for the clean-clone smoke validation' \
	  '  reproduce-full     explain and enforce the authorised-data full-rerun boundary'

setup:
	$(PYTHON) -m pip install -e '.[publication,dev]'

validate:
	$(REPRO) validate --fixture

data-audit:
	$(REPRO) data-audit

preprocess:
	$(REPRO) preprocess --validate-only

primary-preflight:
	$(REPRO) reproduce-primary --validate-only

checkpoints:
	$(REPRO) evaluate-checkpoints --validate-only

diagnostics:
	$(REPRO) reproduce-diagnostics --validate-only

cases:
	$(REPRO) reproduce-cases --validate-only

apsim-validate:
	$(REPRO) reproduce-apsim --validate-only

figures:
	$(REPRO) render-paper-assets --validate-only

tables:
	$(REPRO) render-paper-tables --validate-only

numeric-integrity:
	$(REPRO) numeric-integrity --validate-only

verify:
	$(REPRO) verify-publication --validate-only

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q tests/test_publication_repro_cli.py experiments/apsim_comparison/tests

smoke: validate numeric-integrity figures tables apsim-validate test

reproduce-results:
	$(REPRO) reproduce-cases --validate-only
	$(REPRO) render-paper-assets --output-root outputs/reproduced
	$(REPRO) render-paper-tables --output-root outputs/reproduced
	$(REPRO) numeric-integrity --output-root outputs/reproduced
	$(REPRO) reproduce-apsim --validate-only
	$(REPRO) verify-publication --validate-only

reproduce-smoke: smoke

reproduce-full:
	@printf '%s\n' \
	  'A single unattended full rerun is not available because the exact Roseworthy cleaned point files are not in the cited public record.' \
	  'The accepted preprocessing, training, evaluation, diagnostic, Waite, and APSIM entry points are preserved.' \
	  'Follow docs/reproducibility/REPRODUCIBILITY.md with authorised inputs; this target exits before starting expensive work.'
	@exit 2
