PYTHON ?= python3

.PHONY: all verify clean

all:
	$(PYTHON) scripts/build_all.py

verify:
	$(PYTHON) scripts/verify_outputs.py

clean:
	$(PYTHON) scripts/build_all.py --clean-only
