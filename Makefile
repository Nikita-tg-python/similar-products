# Everything runs in the api container: nothing to install on the host but Docker.
DC  = docker compose
RUN = $(DC) run --rm --no-deps -v "$(CURDIR)":/app api

.PHONY: up down build logs test test-offline lint sample fixture ingest check-stores compare kind-up kind-down

up:  ## start everything and wait until /health is 200 (the first start downloads the embedding model)
	$(DC) up -d --build
	@echo "waiting for the API: the first start downloads the embedding model (~90 MB), a few minutes..."
	@url=http://$$($(DC) port api 8000 | tail -1 | sed 's/0.0.0.0/localhost/'); \
	for i in $$(seq 1 300); do \
		if curl -sf $$url/health >/dev/null; then echo "API is up: $$url  (docs: $$url/docs)"; exit 0; fi; \
		sleep 2; \
	done; \
	echo "API is not healthy after 10 min: see docker compose logs api"; exit 1

down:
	$(DC) down

build:
	$(DC) build api

logs:
	$(DC) logs -f api

test:  ## no network, no LLM keys (FakeLLM)
	$(RUN) pytest -q

test-offline:  ## the unit suite in a container with no network at all and no .env (no keys)
	$(DC) build -q api
	docker run --rm --network none $$($(DC) config --images | grep -- '-api$$') pytest -q -m "not integration"

lint:
	$(RUN) sh -c "ruff check . && ruff format --check ."

sample:  ## >= 50k useful rows from HF without downloading the 7.86 GB file -> data/sample_50k.parquet
	$(RUN) python -m scripts.build_sample $(ARGS)

fixture:  ## tests/fixtures/sample_products.parquet from the local sample (every filter case covered)
	$(RUN) python -m scripts.build_fixture $(ARGS)

ingest:  ## data/sample_50k.parquet -> RQ queue -> workers -> Postgres; waits and reports (needs `make up`)
	$(RUN) python -m app.ingest.producer $(ARGS)

check-stores:  ## Qdrant vs pgvector: same counts, same vectors, top-10 neighbour overlap
	$(RUN) python -m scripts.check_stores $(ARGS)

compare:  ## Qdrant (HTTP, gRPC) vs pgvector: p50/p95, recall@10 vs exact search -> eval/comparison_report.md + README
	$(RUN) python -m eval.compare_backends $(ARGS)

KIND_CLUSTER = similar-products
KUBECTL = kubectl --context kind-$(KIND_CLUSTER) -n similar-products

kind-up:  ## kind cluster + local image + k8s/*.yaml; waits until every pod is Ready
	kind get clusters | grep -qx $(KIND_CLUSTER) || kind create cluster --config kind-config.yaml
	docker build -t similar-products:dev .
	kind load docker-image similar-products:dev --name $(KIND_CLUSTER)
	kubectl --context kind-$(KIND_CLUSTER) apply -f k8s/
	$(KUBECTL) rollout restart deployment/api deployment/worker  # same tag, new image: pick it up
	$(KUBECTL) rollout status statefulset/postgres statefulset/qdrant deployment/redis deployment/api deployment/worker --timeout=10m

kind-down:
	kind delete cluster --name $(KIND_CLUSTER)
