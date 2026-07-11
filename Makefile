.PHONY: smoke reproduce-frozen

smoke:
	python3 -m pytest -q

reproduce-frozen:
	PYTHONPATH=. python3 scripts/reproduce_frozen_results.py
