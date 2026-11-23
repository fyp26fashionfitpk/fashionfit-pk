.PHONY: test lint profile verify health

PYTHON ?= python

lint:
	$(PYTHON) -m py_compile lib/*.py tests/*.py bin/profile_inference.py

test:
	$(PYTHON) -m unittest discover -s tests -v

profile:
	$(PYTHON) bin/profile_inference.py

health:
	bash bin/check_health.sh

verify: lint test profile
	@echo "[OK] All FYP I verification checks passed."
