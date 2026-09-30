# Everything runs in the api container: nothing to install on the host but Docker.
DC  = docker compose
RUN = $(DC) run --rm --no-deps -v "$(CURDIR)":/app api

.PHONY: up down build logs test lint

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
