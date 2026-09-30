# Everything runs in the api container: nothing to install on the host but Docker.
DC  = docker compose
RUN = $(DC) run --rm --no-deps -v "$(CURDIR)":/app api

.PHONY: up down build logs test lint sample fixture ingest

up:
	$(DC) up -d --build

down:
	$(DC) down

build:
	$(DC) build api

logs:
	$(DC) logs -f api

test:  ## no network, no LLM keys (FakeLLM)
	$(RUN) pytest -q

lint:
	$(RUN) sh -c "ruff check . && ruff format --check ."

sample:  ## >= 50k useful rows from HF without downloading the 7.86 GB file -> data/sample_50k.parquet
	$(RUN) python -m scripts.build_sample $(ARGS)

fixture:  ## tests/fixtures/sample_products.parquet from the local sample (every filter case covered)
	$(RUN) python -m scripts.build_fixture $(ARGS)

ingest:  ## data/sample_50k.parquet -> RQ queue -> workers -> Postgres; waits and reports (needs `make up`)
	$(RUN) python -m app.ingest.producer $(ARGS)
