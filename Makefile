PY ?= python
export PYTHONPATH := src

.PHONY: all ingest run test
all: ingest run
ingest:
	$(PY) -m rrs.ingest
run:
	$(PY) -m rrs.pipeline
test:
	$(PY) -m pytest -q tests
