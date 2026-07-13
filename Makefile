PYTHON ?= python3
REPRO = PYTHONPATH=src $(PYTHON) -m agritech_repro

.PHONY: help setup validate data-audit preprocess primary-preflight checkpoints diagnostics cases apsim-validate figures tables numeric-integrity verify smoke test

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
	  '  smoke              run fixture, numeric, figure, table, and focused tests'

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
