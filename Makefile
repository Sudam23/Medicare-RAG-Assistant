PY := ./.venv/bin/python
PIP := uv pip install --python $(PY)
export PYTHONPATH := src

.DEFAULT_GOAL := help

help: ## show this help
	@grep -E '^[a-z0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## create the virtualenv and install everything
	uv venv --python 3.12 .venv
	$(PIP) -e ".[dev,e2e]"

data: ## download the CMS publications, parse and chunk them (~2 min)
	$(PY) -m medicare_rag.data.download
	$(PY) -m medicare_rag.data.parse
	$(PY) -m medicare_rag.data.chunk

index: ## embed every chunk and build both indexes (~1 min)
	$(PY) -m medicare_rag.indexing.build

eval: ## score the held-out test split and write docs/evaluation_report.json
	$(PY) -m medicare_rag.evaluation.run --split test

eval-dev: ## score the dev split (the only split used for tuning)
	$(PY) -m medicare_rag.evaluation.run --split dev --output /dev/null

tune: ## grid-search the fusion weights and the refusal threshold, on dev
	$(PY) scripts/tune_fusion.py
	$(PY) scripts/tune_threshold.py

serve: ## run the API and web page on http://127.0.0.1:8000
	$(PY) -m uvicorn medicare_rag.api.main:app --host 127.0.0.1 --port 8000 --reload

test: ## run the unit and integration tests (stop `make serve` first)
	@# Embedded Qdrant locks its storage directory, so the tests cannot open the
	@# index while a dev server is holding it.
	$(PY) -m pytest

e2e: ## drive a real browser against a running instance (needs Chrome)
	$(PY) scripts/e2e_browser_test.py $(ARGS)

lint: ## check formatting and lint rules
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt: ## apply formatting
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

docs: ## regenerate the generated sections of README.md and docs/
	$(PY) scripts/render_graph.py
	$(PY) scripts/render_results.py

docs-check: ## fail if the published numbers disagree with the evaluation report
	$(PY) scripts/render_results.py --check

docker-build: ## build the container image
	docker build -t medicare-rag-assistant .

docker-run: ## run the container on http://127.0.0.1:8000
	docker run --rm -p 8000:8000 medicare-rag-assistant

.PHONY: help venv data index eval eval-dev tune serve test e2e lint fmt docs docs-check \
	docker-build docker-run
