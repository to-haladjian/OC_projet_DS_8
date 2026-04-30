SHELL:=/bin/bash -eu

.PHONY: help test test-cov
default: help

### Avaialble commands:
###
### - make help: Display this help message
help:
	@cat $(MAKEFILE_LIST) | grep ^\#\#\#\

### - make test: Run all tests with pytest
test:
	if [ ! -d "venv" ]; then echo "Creating venv..." && python3 -m venv venv; fi
	@. venv/bin/activate && pip install -r requirements-tests.txt && pip install -r requirements.txt && pytest tests/ -v

### - make test-cov: Run all tests with pytest and generate coverage report
test-cov:
	if [ ! -d "venv" ]; then echo "Creating venv..." && python3 -m venv venv; fi
	@. venv/bin/activate && pip install -r requirements-tests.txt && pip install -r requirements.txt && pytest tests/ --cov=app --cov-report=html:docs/coverage --cov-report=term-missing | tee docs/coverage_report.txt
